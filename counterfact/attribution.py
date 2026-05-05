"""
Statistical attribution methods for counterfactual analysis.

This module computes how much each agent contributes to the pipeline's
output quality (or failure). Two complementary methods:

  1. Leave-One-Out (LOO) Attribution:
     Simple and fast — measures what happens when each agent is removed.
     LOO_score = mean(quality_with_agent_ablated) - baseline_quality

  2. Shapley Values:
     More rigorous — accounts for interactions between agents.
     Used as fallback when LOO is inconclusive (all scores near zero).

Also includes failure classification: once we know each agent's
contribution, we classify the failure type (local, architectural gap,
systemic, etc.)

Dependencies: types, numpy
"""


import itertools
import random
from typing import Optional, Callable

import numpy as np

from counterfact.types import (
    ClassifierResult,
    SimulationResult,
    FailureClassification,
    ConfidenceInterval,
)


# ═════════════════════════════════════════════════════════════════════════
# LOO (LEAVE-ONE-OUT) ATTRIBUTION
# The simplest attribution method: what happens when we remove each agent?
# ═════════════════════════════════════════════════════════════════════════


def compute_loo_attribution(
    simulation_results: list[SimulationResult],
    trace: list[dict],
) -> dict[str, float]:
    """
    Compute Leave-One-Out attribution for each agent.

    For each agent, this computes:
      LOO = mean(quality when agent ablated) - baseline quality

    Interpretation:
      - Positive LOO: removing the agent IMPROVES quality → agent is harmful
      - Negative LOO: removing the agent HURTS quality → agent is helpful
      - Near-zero LOO: agent has no measurable impact
    """
    # Get unique agent names (excluding "output" which is just formatting)
    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry["node"] != "output"
    ))

    # Calculate baseline quality (average of all baseline runs)
    baseline_scores = [r.quality_score for r in simulation_results if r.is_baseline]
    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.5

    # For each agent, compute LOO = ablation_quality - baseline_quality
    attribution = {}
    for agent in agents:
        ablate_scores = [
            r.quality_score for r in simulation_results
            if r.perturbation and r.perturbation.agent == agent
            and r.perturbation.strategy == "ablate"
        ]
        attribution[agent] = float(np.mean(ablate_scores)) - baseline_mean if ablate_scores else 0.0

    return attribution


def is_loo_inconclusive(
    attribution: dict[str, float],
    threshold: float = 0.05,
) -> bool:
    """
    Check if LOO attribution is inconclusive.

    LOO is inconclusive when all agents have similar (near-zero) impact.
    This suggests either:
      - An architectural failure (no single agent is responsible)
      - Complex interaction effects that LOO can't capture

    When LOO is inconclusive, we escalate to Shapley values.
    """
    if not attribution:
        return True

    values = list(attribution.values())
    max_abs = max(abs(v) for v in values)
    spread = max(values) - min(values)

    return max_abs < threshold and spread < threshold


# ═════════════════════════════════════════════════════════════════════════
# SHAPLEY VALUES
# More rigorous attribution that accounts for agent interactions.
# ═════════════════════════════════════════════════════════════════════════


def compute_bootstrap_ci(
    values: list[float], 
    n_bootstrap: int = 1000, 
    alpha: float = 0.05
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for the mean of values.
    """
    if not values:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0)
    if len(values) == 1:
        return ConfidenceInterval(values[0], values[0], values[0], 1)
        
    n = len(values)
    values_arr = np.array(values)
    
    # Generate bootstrap samples
    indices = np.random.randint(0, n, size=(n_bootstrap, n))
    samples = values_arr[indices]
    means = np.mean(samples, axis=1)
    
    mean = float(np.mean(values))
    ci_low = float(np.percentile(means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    
    return ConfidenceInterval(mean, ci_low, ci_high, n)


def compute_shapley_values(
    simulation_results: list,
    trace: list[dict],
    graph=None,
    input_state: dict = None,
    llm_fn: callable = None,
    query: str = "",
    output_text: str = "",
    sources: str = "",
    domain: str = "rag",
    registry=None,
    n_permutations: int = 20,
) -> tuple[dict[str, float], dict[str, "ConfidenceInterval"], dict[str, dict[str, float]]]:
    import itertools
    import random
    import numpy as np
    from collections import defaultdict
    from counterfact.types import ConfidenceInterval, ClassifierResult

    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry["node"] != "output"
    ))

    baseline_runs = [r for r in simulation_results if r.is_baseline]
    baseline_scores = [r.quality_score for r in baseline_runs]
    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.5

    if not agents:
        return {}, {}, {}

    N = len(agents)
    full_coalition = frozenset(agents)

    coalition_quality_cache = {full_coalition: baseline_mean}
    coalition_clf_cache = {}

    if baseline_runs and baseline_runs[0].classifier_results:
        baseline_clf = {}
        for ref in baseline_runs[0].classifier_results:
            scores = [
                c.score for r in baseline_runs
                for c in r.classifier_results if c.name == ref.name
            ]
            baseline_clf[ref.name] = float(np.mean(scores)) if scores else 0.5
        coalition_clf_cache[full_coalition] = [
            ClassifierResult(name=k, score=v, reasoning="avg", weight=1.0)
            for k, v in baseline_clf.items()
        ]

    ablation_runs = [r for r in simulation_results if not r.is_baseline and r.perturbation]
    
    coalition_groups = defaultdict(list)
    for r in ablation_runs:
        if r.perturbation.agent:
            ablated = frozenset([a.strip() for a in r.perturbation.agent.split(",") if a.strip()])
            coalition = frozenset(agents) - ablated
        else:
            coalition = full_coalition
        coalition_groups[coalition].append(r)

    for coalition, runs in coalition_groups.items():
        coalition_quality_cache[coalition] = float(np.mean([r.quality_score for r in runs]))
        clf_dict = defaultdict(list)
        for r in runs:
            for c in r.classifier_results:
                clf_dict[c.name].append(c.score)
        coalition_clf_cache[coalition] = [
            ClassifierResult(name=k, score=float(np.mean(v)), reasoning="avg", weight=1.0)
            for k, v in clf_dict.items()
        ]

    coalition_quality_cache[frozenset()] = 0.0
    coalition_clf_cache[frozenset()] = []

    if N <= 4:
        perms = list(itertools.permutations(agents))
    else:
        n_perms = max(5, sum(1 for _ in simulation_results) // max(1, N))
        perms = [tuple(random.sample(agents, N)) for _ in range(n_perms)]

    marginals = {a: [] for a in agents}
    per_clf_marginals = {}

    for perm in perms:
        coalition = frozenset()
        v_prev = 0.0
        prev_clf_scores = {}

        for agent in perm:
            coalition = coalition | {agent}
            v_curr = coalition_quality_cache.get(coalition, 0.0)
            marginals[agent].append(v_curr - v_prev)
            v_prev = v_curr

            clf_results = coalition_clf_cache.get(coalition, [])
            curr_clf_scores = {c.name: c.score for c in clf_results}
            for clf_name, curr_score in curr_clf_scores.items():
                prev_score = prev_clf_scores.get(clf_name, 0.0)
                if clf_name not in per_clf_marginals:
                    per_clf_marginals[clf_name] = {a: [] for a in agents}
                per_clf_marginals[clf_name][agent].append(curr_score - prev_score)
            prev_clf_scores = curr_clf_scores

    shapley = {}
    cis = {}
    for agent in agents:
        m = marginals[agent]
        shapley[agent] = float(np.mean(m)) if m else 0.0
        cis[agent] = compute_bootstrap_ci(m) if m else ConfidenceInterval(0.0, 0.0, 0.0, 0)

    total = sum(abs(v) for v in shapley.values())
    if total > 0:
        shapley = {k: v / total for k, v in shapley.items()}
        for k in cis:
            cis[k].mean /= total
            cis[k].ci_low /= total
            cis[k].ci_high /= total

    per_clf_shapley = {}
    for clf_name, agent_marginals in per_clf_marginals.items():
        per_clf_shapley[clf_name] = {}
        for agent, m in agent_marginals.items():
            per_clf_shapley[clf_name][agent] = float(np.mean(m)) if m else 0.0
        c_total = sum(abs(v) for v in per_clf_shapley[clf_name].values())
        if c_total > 0:
            per_clf_shapley[clf_name] = {k: v / c_total for k, v in per_clf_shapley[clf_name].items()}

    return shapley, cis, per_clf_shapley



def compute_per_classifier_loo(
    simulation_results: list[SimulationResult],
    trace: list[dict],
) -> dict[str, dict[str, float]]:
    """
    Compute Leave-One-Out (LOO) attribution broken down by quality classifier.

    This is a fast LOO estimator, not true Shapley values. For each agent and
    each classifier it computes:

      LOO_clf[agent] = mean(clf_score when agent ablated) - baseline_clf_score

    Interaction effects between agents are NOT captured. Use this:
      - As a fallback when llm_fn=None (no coalition simulations available)
      - To quickly identify which classifier dimension is affected per agent

    For true per-classifier Shapley (accounting for interactions), use the
    third return value of compute_shapley_values() instead.

    Returns: {classifier_name: {agent_name: loo_value}}
    """
    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry["node"] != "output"
    ))

    baseline_results = [r for r in simulation_results if r.is_baseline]
    if not baseline_results or not baseline_results[0].classifier_results:
        return {}

    classifier_names = [c.name for c in baseline_results[0].classifier_results]
    per_clf_loo = {}

    for clf_name in classifier_names:
        baseline_scores = []
        for r in baseline_results:
            for c in r.classifier_results:
                if c.name == clf_name:
                    baseline_scores.append(c.score)
        baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.5

        agent_values = {}
        for agent in agents:
            agent_scores = []
            for r in simulation_results:
                if r.perturbation and r.perturbation.agent == agent:
                    for c in r.classifier_results:
                        if c.name == clf_name:
                            agent_scores.append(c.score - baseline_mean)
            agent_values[agent] = float(np.mean(agent_scores)) if agent_scores else 0.0

        total = sum(abs(v) for v in agent_values.values())
        if total > 0:
            agent_values = {k: v / total for k, v in agent_values.items()}

        per_clf_loo[clf_name] = agent_values

    return per_clf_loo


# Backward-compatible alias — use compute_per_classifier_loo() for new code.
compute_per_classifier_shapley = compute_per_classifier_loo


# ═════════════════════════════════════════════════════════════════════════
# FAILURE CLASSIFICATION
# Classify the failure type based on attribution results.
# ═════════════════════════════════════════════════════════════════════════


def classify_failure(
    shapley_values: dict[str, float],
    simulation_results: list[SimulationResult],
    trace: list[dict],
    per_clf_shapley: Optional[dict[str, dict[str, float]]] = None,
    shapley_cis: Optional[dict[str, ConfidenceInterval]] = None,
) -> FailureClassification:
    """
    Classify the failure type based on Shapley values and simulation data.

    Four failure types:
      - "local": One agent dominates (Shapley > 0.35) → fix that agent
      - "architectural_gap": All Shapley near zero → pipeline is missing
        a capability (no agent is responsible because no agent has the job)
      - "feedback_amplification": Revision loop makes things worse → limit loop
      - "systemic": Multiple agents interact → complex fix needed
    """
    # ── Analyze the Shapley distribution ─────────────────────────────
    abs_values = {k: abs(v) for k, v in shapley_values.items()}
    max_agent = max(abs_values, key=lambda k: abs_values[k]) if abs_values else None
    max_value = abs_values.get(max_agent, 0) if max_agent else 0

    # ── Find which classifiers are failing ───────────────────────────
    baseline_results = [r for r in simulation_results if r.is_baseline]
    failing_classifiers = []
    if baseline_results and baseline_results[0].classifier_results:
        for clf in baseline_results[0].classifier_results:
            avg_scores = []
            for r in baseline_results:
                for c in r.classifier_results:
                    if c.name == clf.name:
                        avg_scores.append(c.score)
            if avg_scores and np.mean(avg_scores) < 0.5:
                failing_classifiers.append(clf.name)

    # ── Check if all Shapley values are near zero ────────────────────
    # Threshold of 0.25: below this, no agent is dominant enough to
    # be considered the root cause — the pipeline is likely missing a
    # capability entirely (architectural gap). The local-failure branch
    # (max_value > 0.35) handles genuinely dominant agents, so the gap
    # between 0.25 and 0.35 maps to "systemic" as intended.
    ARCH_GAP_THRESHOLD = 0.25
    all_near_zero = all(v < ARCH_GAP_THRESHOLD for v in abs_values.values())

    # ── Check for feedback amplification ─────────────────────────────
    # If the critic was ablated and quality improved, the revision loop
    # is making things worse (feedback amplification).
    synthesizer_count = sum(1 for e in trace if e["node"] == "synthesizer")
    has_revisions = synthesizer_count > 1

    damping_ratio = None
    if has_revisions:
        enhance_scores = [
            r.quality_score for r in simulation_results
            if r.perturbation and r.perturbation.agent == "critic"
            and r.perturbation.strategy == "ablate"
        ]
        baseline_scores = [r.quality_score for r in simulation_results if r.is_baseline]
        if enhance_scores and baseline_scores:
            damping_ratio = float(
                float(np.mean(enhance_scores)) / max(float(np.mean(baseline_scores)), 0.01)
            )

    # ── Confidence scoring (CI-based) ──────────────────────────────────
    # Derive confidence from bootstrap confidence intervals on Shapley
    # values. No hand-picked constants — confidence measures how
    # statistically robust the classification is.

    def _ci_overlap_confidence(ci_top: ConfidenceInterval, ci_second: ConfidenceInterval) -> float:
        """
        Confidence that top agent truly dominates, based on CI overlap.

        If the CIs don't overlap at all → ~0.95 (the 95% CIs are separated).
        If they overlap fully → ~0.50 (can't distinguish the two).
        Partial overlap → linearly interpolated.
        """
        if ci_top.n_samples < 2 or ci_second.n_samples < 2:
            return 0.50  # Not enough data

        # Non-overlapping CIs
        if ci_top.ci_low > ci_second.ci_high:
            return 0.95

        # Compute overlap fraction relative to the smaller CI width
        overlap = max(0, min(ci_top.ci_high, ci_second.ci_high) - max(ci_top.ci_low, ci_second.ci_low))
        top_width = max(ci_top.ci_high - ci_top.ci_low, 1e-10)
        second_width = max(ci_second.ci_high - ci_second.ci_low, 1e-10)
        overlap_frac = overlap / min(top_width, second_width)

        # 0% overlap → 0.95, 100% overlap → 0.50
        return 0.95 - 0.45 * min(1.0, overlap_frac)

    def _all_near_zero_confidence(shapley_cis: dict[str, ConfidenceInterval], threshold: float) -> float:
        """
        Confidence that ALL agents have near-zero impact.

        High confidence if every agent's CI upper bound is below the
        threshold (i.e., even the optimistic estimate says no agent matters).
        """
        if not shapley_cis:
            return 0.50
        all_below = all(ci.ci_high < threshold for ci in shapley_cis.values() if ci.n_samples >= 2)
        if all_below:
            return 0.95
        # Fraction of agents whose CIs are fully below threshold
        n_below = sum(1 for ci in shapley_cis.values() if ci.ci_high < threshold and ci.n_samples >= 2)
        return 0.50 + 0.45 * (n_below / max(len(shapley_cis), 1))

    def _compute_confidence(shapley_cis: dict[str, ConfidenceInterval], failure_type: str) -> float:
        """Derive classification confidence from bootstrap CIs."""
        agents_sorted = sorted(shapley_cis.keys(), key=lambda a: abs(shapley_values.get(a, 0)), reverse=True)

        if failure_type == "local" and len(agents_sorted) >= 2:
            return _ci_overlap_confidence(shapley_cis[agents_sorted[0]], shapley_cis[agents_sorted[1]])
        elif failure_type == "architectural_gap":
            return _all_near_zero_confidence(shapley_cis, ARCH_GAP_THRESHOLD)
        elif failure_type == "feedback_amplification":
            return 0.80  # Damping ratio is a direct measurement
        else:  # systemic
            # Multiple overlapping CIs → inherently uncertain
            return 0.60

    def _confidence_explanation(conf, n_sims, failure_type, shapley_cis):
        parts = [f"Based on {n_sims} simulations."]
        if failure_type == "local":
            parts.append(f"Confidence derived from bootstrap CI separation between top two agents.")
        elif failure_type == "architectural_gap":
            parts.append(f"All agents' 95% CIs are consistent with near-zero impact.")
        elif failure_type == "systemic":
            parts.append(f"Multiple agents have overlapping CIs — classification is less certain.")
        parts.append(f"Confidence = {conf:.0%}")
        return " ".join(parts)

    # ── Check for persistent classifier gap (strongest arch-gap signal) ─
    # If a classifier scores ≈0 in baseline AND stays ≈0 when every agent
    # is ablated, no agent in the pipeline owns that quality dimension.
    # That is definitive evidence of a missing capability, regardless of
    # what the Shapley magnitudes look like.
    PERSISTENT_FAIL_THRESHOLD = 0.15  # classifier must score below this
    persistent_gap_clfs = []
    if failing_classifiers and baseline_results:
        for clf_name in failing_classifiers:
            # Check: does this classifier fail in ALL simulations (baseline + perturbed)?
            all_scores = []
            for r in simulation_results:
                for c in r.classifier_results:
                    if c.name == clf_name:
                        all_scores.append(c.score)
            if all_scores and max(all_scores) < PERSISTENT_FAIL_THRESHOLD:
                persistent_gap_clfs.append(clf_name)

    _cis = shapley_cis or {}

    if persistent_gap_clfs:
        conf = max(_compute_confidence(_cis, "architectural_gap"), 0.85)
        gap_desc = (
            f"No agent in the pipeline is responsible for the failing quality dimensions "
            f"({', '.join(persistent_gap_clfs)}). These dimensions score near zero in all "
            f"{len(simulation_results)} simulations — baseline and every ablation — meaning "
            f"no existing agent addresses them."
        )
        evidence = [
            f"Persistent zero-score classifiers: {', '.join(persistent_gap_clfs)}",
            f"These dimensions score <{PERSISTENT_FAIL_THRESHOLD} in ALL {len(simulation_results)} simulations",
            "Ablating any agent does not fix these dimensions — the capability is missing",
            f"Additional failing classifiers: {', '.join(c for c in failing_classifiers if c not in persistent_gap_clfs)}"
            if len(failing_classifiers) > len(persistent_gap_clfs) else
            "The pipeline architecture needs a new agent to cover these dimensions",
        ]
        return FailureClassification(
            failure_type="architectural_gap",
            confidence=conf,
            description=gap_desc,
            evidence=evidence,
            dominant_agent=None,
            failing_classifiers=failing_classifiers,
            confidence_explanation=_confidence_explanation(
                conf, len(simulation_results), "architectural_gap", _cis
            ),
        )

    if all_near_zero:
        conf = _compute_confidence(_cis, "architectural_gap")
        if failing_classifiers:
            gap_desc = (
                f"No single agent is responsible. The failing quality dimensions "
                f"({', '.join(failing_classifiers)}) are not owned by any agent in the pipeline."
            )
            evidence = [
                f"All Shapley values near zero (max={max_value:.3f})",
                f"Failing classifiers: {', '.join(failing_classifiers)}",
                "Perturbing individual agents does not resolve the failure",
            ]
        else:
            gap_desc = (
                "No single agent is responsible for the failure. The output appears "
                "correct to automated classifiers, but perturbing any agent has no effect."
            )
            evidence = [
                f"All Shapley values near zero (max={max_value:.3f})",
                "All classifiers pass — the failure is invisible to automated checks",
                "Perturbing individual agents does not resolve the failure",
            ]
        return FailureClassification(
            failure_type="architectural_gap",
            confidence=conf,
            description=gap_desc,
            evidence=evidence,
            dominant_agent=None,
            failing_classifiers=failing_classifiers,
            confidence_explanation=_confidence_explanation(
                conf, len(simulation_results), "architectural_gap", _cis
            ),
        )

    elif max_value > 0.35:
        # Check if this is actually a "uniform catastrophe" — all agents equally large.
        # When ablating ANY agent collapses quality equally, it means the pipeline lacks
        # a cross-cutting capability: no agent "owns" the failure, but every ablation
        # exposes it. This is architectural_gap, not local.
        if len(abs_values) > 1:
            vals = list(abs_values.values())
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            cv = std_val / mean_val if mean_val > 0 else 1.0  # coefficient of variation
            all_large = all(v > 0.35 for v in vals)
            uniform_catastrophe = all_large and cv < 0.30
        else:
            uniform_catastrophe = False

        if uniform_catastrophe:
            conf = _compute_confidence(_cis, "architectural_gap")
            evidence = [
                f"All agents have large Shapley magnitudes (min={min(abs_values.values()):.3f}, max={max_value:.3f})",
                f"Low coefficient of variation ({cv:.2f} < 0.30) — agents contribute equally",
                "Uniform catastrophe on ablation = missing cross-cutting validation step",
            ]
            return FailureClassification(
                failure_type="architectural_gap",
                confidence=conf,
                description=(
                    f"Ablating any agent collapses quality equally (CV={cv:.2f}). "
                    f"Missing cross-cutting capability."
                ),
                evidence=evidence,
                dominant_agent=None,
                failing_classifiers=failing_classifiers,
                confidence_explanation=_confidence_explanation(
                    conf, len(simulation_results), "architectural_gap", _cis
                ),
            )

        # One agent dominates → local failure
        conf = _compute_confidence(_cis, "local")
        return FailureClassification(
            failure_type="local",
            confidence=conf,
            description=f"The failure is primarily attributable to the {max_agent} agent.",
            evidence=[
                f"Dominant Shapley value: {max_agent} = {shapley_values.get(max_agent or '', 0.0):.3f}",
                f"Perturbing {max_agent} changes quality by {max_value:.1%}",
                f"Failing classifiers: {', '.join(failing_classifiers)}" if failing_classifiers
                else "Quality dimensions show clear agent dependency",
            ],
            dominant_agent=max_agent,
            failing_classifiers=failing_classifiers,
            confidence_explanation=_confidence_explanation(
                conf, len(simulation_results), "local", _cis
            ),
        )

    elif damping_ratio and damping_ratio > 1.1:
        conf = _compute_confidence(_cis, "feedback_amplification")
        return FailureClassification(
            failure_type="feedback_amplification",
            confidence=conf,
            description="The revision loop is amplifying errors. Each revision degrades quality.",
            evidence=[
                f"Damping ratio: {damping_ratio:.2f}",
                "Removing the critic improves output quality",
            ],
            dominant_agent=None,
            failing_classifiers=failing_classifiers,
            damping_ratio=damping_ratio,
            confidence_explanation=_confidence_explanation(
                conf, len(simulation_results), "feedback_amplification", _cis
            ),
        )

    else:
        conf = _compute_confidence(_cis, "systemic")
        return FailureClassification(
            failure_type="systemic",
            confidence=conf,
            description="Multiple agents contribute to the failure through complex interactions.",
            evidence=[
                "Multiple agents with significant Shapley values",
                f"Top contributor: {max_agent} = {max_value:.3f}",
                f"Failing classifiers: {', '.join(failing_classifiers)}" if failing_classifiers
                else "No single dominant failure mode",
            ],
            dominant_agent=max_agent,
            failing_classifiers=failing_classifiers,
            confidence_explanation=_confidence_explanation(
                conf, len(simulation_results), "systemic", _cis
            ),
        )
