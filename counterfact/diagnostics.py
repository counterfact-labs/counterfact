"""
Diagnostic engine — the central orchestrator.

This module ties everything together. It coordinates:
  1. Eval suite (structural + consistency checks)
  2. Monte Carlo perturbation simulations (real pipeline re-execution)
  3. Attribution (LOO → Shapley fallback)
  4. Failure classification
  5. Recommendation generation + evaluation

The main entry point is run_full_diagnostic(), which runs the complete
pipeline and returns a DiagnosticReport.

Dependencies: evals, perturbation, attribution, recommendations, classifiers, types
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from counterfact.types import (
    SimulationResult,
    FailureClassification,
    Recommendation,
    EvaluationResult,
    EvalSuite,
    ConfidenceInterval,
)
from counterfact.evals import run_eval_suite
from counterfact.perturbation import run_monte_carlo
from counterfact.attribution import (
    compute_shapley_values,
    compute_per_classifier_loo,
    classify_failure,
    compute_bootstrap_ci,
)
from counterfact.recommendations import (
    generate_recommendations,
    evaluate_recommendation,
    extract_empirical_fixes,
    detect_coverage_gaps,
    rank_recommendations,
)
from counterfact.classifiers import ClassifierRegistry

if TYPE_CHECKING:
    from counterfact.graph import CounterfactualGraph


# ═════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC REPORT
# The final output of a complete diagnostic run.
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic report containing all analysis results.

    This is what you get back from run_full_diagnostic(). It contains:
      - Eval results (Tier 1 + 2 checks)
      - Shapley attribution values
      - Failure classification
      - Recommended fixes
      - Evaluation of whether the fixes would work
    """
    query: str
    domain: str
    baseline_quality: float
    shapley_values: dict[str, float]
    per_classifier_shapley: dict[str, dict[str, float]]
    classification: FailureClassification
    recommendations: list[Recommendation]
    evaluations: list[EvaluationResult]
    num_simulations: int
    simulation_results: list[SimulationResult]
    simulation_results_summary: dict
    eval_suite: Optional[EvalSuite] = None
    attribution_method: str = "loo"
    shapley_cis: dict[str, ConfidenceInterval] = field(default_factory=dict)
    baseline_quality_ci: Optional[ConfidenceInterval] = None
    seed: Optional[int] = None
    _trace: Optional[list[dict]] = field(default=None, repr=False, init=False)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "domain": self.domain,
            "baseline_quality": round(self.baseline_quality, 3),
            "shapley_values": {k: round(v, 4) for k, v in self.shapley_values.items()},
            "shapley_cis": {k: v.to_dict() for k, v in self.shapley_cis.items()},
            "baseline_quality_ci": self.baseline_quality_ci.to_dict() if self.baseline_quality_ci else None,
            "attribution_method": self.attribution_method,
            "per_classifier_shapley": {
                clf: {agent: round(v, 4) for agent, v in agents.items()}
                for clf, agents in self.per_classifier_shapley.items()
            },
            "classification": self.classification.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "num_simulations": self.num_simulations,
            "simulation_details": [s.to_dict() for s in self.simulation_results],
            "simulation_results_summary": self.simulation_results_summary,
            "eval_suite": self.eval_suite.to_dict() if self.eval_suite else None,
            "seed": self.seed,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        from counterfact.export import to_json
        return to_json(self, path)

    def to_markdown(self, path: Optional[str] = None) -> str:
        from counterfact.export import to_markdown
        return to_markdown(self, path)

    def to_html(self, path: Optional[str] = None) -> str:
        from counterfact.export import to_html
        return to_html(self, path)


# ═════════════════════════════════════════════════════════════════════════
# QUALITY GATE
# Skip full diagnostics if the output is already high quality.
# ═════════════════════════════════════════════════════════════════════════

QUALITY_GATE_THRESHOLD = 0.8


def _make_no_failure_classification(
    baseline_quality: float,
    baseline_results: list[SimulationResult],
) -> FailureClassification:
    """
    Create a 'no_failure' classification when baseline quality is high.

    If the pipeline's output passes quality checks, there's no failure
    to diagnose — skip the expensive attribution step.
    """
    passing_classifiers = []
    if baseline_results and baseline_results[0].classifier_results:
        for clf in baseline_results[0].classifier_results:
            scores = []
            for r in baseline_results:
                for c in r.classifier_results:
                    if c.name == clf.name:
                        scores.append(c.score)
            if scores:
                passing_classifiers.append(f"{clf.name} ({np.mean(scores):.2f})")

    return FailureClassification(
        failure_type="no_failure",
        confidence=min(0.95, baseline_quality),
        description=(
            f"Pipeline output passes quality checks (baseline quality = {baseline_quality:.2f}). "
            f"No significant failure detected."
        ),
        evidence=[
            f"Baseline quality {baseline_quality:.3f} ≥ threshold {QUALITY_GATE_THRESHOLD}",
            f"Classifier scores: {', '.join(passing_classifiers)}",
            "No attribution performed — output quality is sufficient.",
        ],
        dominant_agent=None,
        failing_classifiers=[],
        confidence_explanation=(
            f"Baseline quality {baseline_quality:.2f} exceeds the {QUALITY_GATE_THRESHOLD} "
            f"threshold. All classifiers indicate acceptable output quality."
        ),
    )


# ═════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# Runs the complete diagnostic pipeline with real re-execution.
# ═════════════════════════════════════════════════════════════════════════


def run_full_diagnostic(
    graph: "CounterfactualGraph",
    input_state: dict,
    sources: str = "",
    domain: str = "rag",
    num_simulations: int = 30,
    quality_gate: float = QUALITY_GATE_THRESHOLD,
    progress_callback: Optional[Callable] = None,
    registry: Optional[ClassifierRegistry] = None,
    llm_fn: Optional[Callable] = None,
    run_evals: bool = True,
    seed: Optional[int] = None,
) -> DiagnosticReport:
    """
    Run the complete diagnostic pipeline with real re-execution.

    Steps:
      0. Run the pipeline once to get the baseline trace
      1. Eval suite — run structural + consistency checks on the trace
      2. Monte Carlo — ablate each agent and actually re-run the pipeline
      3. Quality gate — skip attribution if baseline quality is high
      4. Shapley attribution — real coalition re-execution for each agent subset
      5. Failure classification — classify the type of failure
      6. Recommendations — generate fix suggestions

    Args:
        graph: The compiled CounterfactualGraph (must have build recipe)
        input_state: The input dict to invoke the pipeline with
        sources: Source documents used by the pipeline
        domain: Classifier domain ("rag" or "decision")
        num_simulations: Number of Monte Carlo simulations
        quality_gate: Baseline quality threshold (skip attribution if above)
        progress_callback: Optional callback(current, total, status)
        registry: Custom ClassifierRegistry (uses default if None)
        llm_fn: Custom LLM function for classifiers (prompt, temp) -> str
        run_evals: Whether to run the eval suite first
        seed: Random seed for reproducibility

    Returns:
        DiagnosticReport with attribution, classification, and recommendations
    """
    query = input_state.get("query", str(input_state)[:200])

    # ── Step 0: Get baseline trace ───────────────────────────────────
    # Run the pipeline once to capture the execution trace for evals.
    baseline_result = graph.invoke(input_state)
    baseline_trace = graph.get_trace()
    output_text = ""
    if isinstance(baseline_result, dict):
        for key in ["final_output", "output", "response", "answer", "result"]:
            if key in baseline_result and isinstance(baseline_result[key], str):
                output_text = baseline_result[key]
                break
    if not output_text:
        output_text = str(baseline_result)[:500]

    # ── Step 1: Run eval suite (optional) ────────────────────────────
    eval_suite = None
    if run_evals:
        tiers = [1, 2] if llm_fn else [1]
        try:
            eval_suite = run_eval_suite(
                trace=baseline_trace,
                final_output=output_text,
                llm_fn=llm_fn,
                tiers=tiers,
            )
        except (ValueError, Exception):
            # Eval suite failure shouldn't block diagnostics
            eval_suite = None

    # ── Step 2: Monte Carlo simulations (real re-execution) ──────────
    sim_results = run_monte_carlo(
        graph=graph,
        input_state=input_state,
        sources=sources,
        domain=domain,
        num_simulations=num_simulations,
        progress_callback=progress_callback,
        registry=registry,
        llm_fn=llm_fn,
        seed=seed,
    )

    # ── Step 3: Quality gate ─────────────────────────────────────────
    baseline_results = [r for r in sim_results if r.is_baseline]
    baseline_scores = [r.quality_score for r in baseline_results]
    baseline_quality = float(np.mean(baseline_scores)) if baseline_scores else 0.5

    agents = graph.get_node_names()
    zero_attribution = {agent: 0.0 for agent in agents}

    if baseline_quality >= quality_gate:
        # Output quality is high — no failure to diagnose
        classification = _make_no_failure_classification(baseline_quality, baseline_results)

        summary = _build_summary(
            sim_results, baseline_scores, baseline_quality, agents,
            "quality_gate", True, baseline_results,
        )

        report = DiagnosticReport(
            query=query,
            domain=domain,
            baseline_quality=baseline_quality,
            shapley_values=zero_attribution,
            per_classifier_shapley={},
            classification=classification,
            recommendations=[],
            evaluations=[],
            num_simulations=num_simulations,
            simulation_results=sim_results,
            simulation_results_summary=summary,
            eval_suite=eval_suite,
            attribution_method="quality_gate",
            seed=seed,
        )
        report._trace = baseline_trace
        return report

    # ── Step 4: Attribution (Shapley values via real re-execution) ─────
    baseline_quality_ci = compute_bootstrap_ci(baseline_scores) if baseline_scores else None

    # Build a trace-like structure from node names for attribution functions
    trace_for_attribution = [
        {"node": agent, "status": "pass", "reasoning": ""}
        for agent in agents
    ]

    attribution, cis, per_clf_shapley = compute_shapley_values(
        sim_results, trace_for_attribution,
        graph=graph,
        input_state=input_state,
        sources=sources,
        domain=domain,
        registry=registry,
    )
    attribution_method = "shapley"
    if not per_clf_shapley:
        per_clf_shapley = compute_per_classifier_loo(sim_results, trace_for_attribution)

    # ── Step 5: Classify failure ─────────────────────────────────────
    classification = classify_failure(attribution, sim_results, trace_for_attribution, per_clf_shapley, cis)

    # ── Step 5b: Failure-Focused Attribution (Architectural Gaps) ────
    if (classification.failure_type == "architectural_gap"
            and classification.failing_classifiers
            and per_clf_shapley):
        gap_clf_names = classification.failing_classifiers
        failure_focused = {}
        for agent in attribution:
            per_agent_deltas = []
            for clf_name in gap_clf_names:
                clf_dict = per_clf_shapley.get(clf_name, {})
                per_agent_deltas.append(clf_dict.get(agent, 0.0))
            ff_val = float(np.mean(per_agent_deltas)) if per_agent_deltas else 0.0
            failure_focused[agent] = (ff_val * 0.95) + (attribution[agent] * 0.05)

        attribution = failure_focused

        baseline_mean_q = float(np.mean(baseline_scores)) if baseline_scores else 0.5
        for agent in attribution:
            raw_deltas = []
            for r in sim_results:
                if r.perturbation and r.perturbation.agent == agent and r.perturbation.strategy == "ablate":
                    ff_score = 0
                    for clf_name in gap_clf_names:
                        for c in r.classifier_results:
                            if c.name == clf_name:
                                b_scores = [
                                    cc.score for br in baseline_results
                                    for cc in br.classifier_results if cc.name == clf_name
                                ]
                                b_mean = float(np.mean(b_scores)) if b_scores else 0.0
                                ff_score += (c.score - b_mean)
                    ff_score = ff_score / len(gap_clf_names) if gap_clf_names else 0.0
                    total_delta = r.quality_score - baseline_mean_q
                    raw_deltas.append((ff_score * 0.95) + (total_delta * 0.05))

            if raw_deltas:
                cis[agent] = compute_bootstrap_ci(raw_deltas)
            else:
                cis[agent] = ConfidenceInterval(attribution[agent], attribution[agent], attribution[agent], 0)

    # ── Step 6: Generate recommendations (deterministic-first) ────────

    # 6a: Extract empirical fixes from simulation data
    empirical_fixes = extract_empirical_fixes(
        sim_results, baseline_quality, trace_for_attribution,
    )

    # 6b: Detect coverage gaps and generate add-agent recommendations
    gap_fixes = []
    if classification.failure_type == "architectural_gap":
        gap_fixes = detect_coverage_gaps(
            per_clf_shapley, classification.failing_classifiers,
            sim_results, trace_for_attribution,
        )

    # 6c: Combine deterministic candidates
    all_candidates = empirical_fixes + gap_fixes

    if not all_candidates:
        all_candidates = generate_recommendations(
            classification, attribution, trace_for_attribution, query, domain, llm_fn,
        )

    # 6d: Rank by measured improvement
    recommendations = rank_recommendations(all_candidates)

    # ── Step 7: Evaluate top recommendation (optional, LLM-based) ────
    evaluations = []
    if recommendations and llm_fn and not any(
        r.measurement_confidence == "measured" for r in recommendations
    ):
        eval_result = evaluate_recommendation(
            recommendations[0], trace_for_attribution, query, output_text, sources,
            domain, num_eval_runs=3, registry=registry, llm_fn=llm_fn,
        )
        evaluations.append(eval_result)

    # Build summary
    summary = _build_summary(
        sim_results, baseline_scores, baseline_quality,
        list(attribution.keys()), attribution_method, False, baseline_results,
    )

    report = DiagnosticReport(
        query=query,
        domain=domain,
        baseline_quality=baseline_quality,
        shapley_values=attribution,
        per_classifier_shapley=per_clf_shapley,
        classification=classification,
        recommendations=recommendations,
        evaluations=evaluations,
        num_simulations=num_simulations,
        simulation_results=sim_results,
        simulation_results_summary=summary,
        eval_suite=eval_suite,
        attribution_method=attribution_method,
        shapley_cis=cis,
        baseline_quality_ci=baseline_quality_ci,
        seed=seed,
    )
    report._trace = baseline_trace
    return report


def _build_summary(
    sim_results, baseline_scores, baseline_quality,
    agents, attribution_method, quality_gate_passed, baseline_results,
) -> dict:
    """Build a summary dict for the diagnostic report."""
    return {
        "total_simulations": len(sim_results),
        "baseline_runs": len(baseline_scores),
        "perturbation_runs": len(sim_results) - len(baseline_scores),
        "baseline_quality_mean": round(baseline_quality, 3),
        "baseline_quality_std": round(float(np.std(baseline_scores)), 3) if baseline_scores else 0,
        "agents_analyzed": agents,
        "attribution_method": attribution_method,
        "quality_gate_passed": quality_gate_passed,
        "classifiers_used": (
            [c.name for c in baseline_results[0].classifier_results]
            if baseline_results and baseline_results[0].classifier_results else []
        ),
        "total_llm_calls": (
            len(sim_results) * len(sim_results[0].classifier_results)
            if sim_results and sim_results[0].classifier_results else 0
        ),
    }
