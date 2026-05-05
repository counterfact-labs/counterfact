"""
Recommendation engine for multi-agent pipeline fixes.

Deterministic-first architecture: instead of asking an LLM to imagine
fixes, we extract them directly from existing simulation data.

Fix generation hierarchy (most reliable → least reliable):
  1. Empirical extraction — enhancement/ablation deltas from simulations
  2. Coverage gap detection — per-classifier Shapley matrix analysis
  3. Agent spec generation (4-tier):
     a. Template lookup — pre-built agent templates
     b. Classifier inversion — flip classifier logic into agent logic
     c. Enhancement diff — extract spec from before/after examples
     d. LLM generation — last resort
  4. Legacy LLM generation — backward-compatible fallback

Dependencies: types only (LLM function is injected via llm_fn)
"""

import json
import math
from typing import Callable, Optional

import numpy as np

from counterfact.types import (
    AgentSpec,
    FailureClassification,
    Recommendation,
    EvaluationResult,
    SimulationResult,
)


# ═════════════════════════════════════════════════════════════════════════
# AGENT TEMPLATE LIBRARY
# Pre-built agent specs for known quality dimensions.
# ═════════════════════════════════════════════════════════════════════════


AGENT_TEMPLATES: dict[str, AgentSpec] = {
    "premise_validity": AgentSpec(
        name="premise_validator",
        position="before_synthesizer",
        function=(
            "Check if the query's premises are supported by retrieved sources. "
            "Flag unsupported assumptions and add warnings to state."
        ),
        input_keys=["query", "retrieved_docs"],
        output_keys=["premise_warnings", "validated_query"],
        prompt_template=(
            "Check if the following query contains assumptions that are "
            "contradicted by the source documents: {query}\n"
            "Sources: {sources}\n"
            "If any premise is invalid, explain why and rewrite the query."
        ),
        source_tier="template",
    ),
    "evidence_sufficiency": AgentSpec(
        name="evidence_verifier",
        position="before_decision_agent",
        function=(
            "Verify customer claims against company billing/order records "
            "before any refund decision is made."
        ),
        input_keys=["extracted_claim", "customer_info"],
        output_keys=["verified_claims", "unverified_claims", "verification_result"],
        prompt_template=(
            "Verify each factual claim against company records: {claims}\n"
            "Records: {records}\n"
            "For each claim, mark as VERIFIED or UNVERIFIED with evidence."
        ),
        source_tier="template",
    ),
    "regulatory_compliance": AgentSpec(
        name="compliance_filter",
        position="after_output",
        function=(
            "Screen output for regulatory violations (investment advice, "
            "forward-looking predictions, unauthorized recommendations)."
        ),
        input_keys=["final_output"],
        output_keys=["filtered_output", "compliance_flags"],
        prompt_template=(
            "Review this financial analysis for regulatory compliance: {output}\n"
            "Remove any buy/sell recommendations or investment advice. "
            "Flag any forward-looking predictions."
        ),
        source_tier="template",
    ),
    "audience_fit": AgentSpec(
        name="audience_adapter",
        position="after_synthesizer",
        function=(
            "Rewrite output to match target audience. Replace jargon "
            "with accessible language for non-technical readers."
        ),
        input_keys=["generation", "target_audience"],
        output_keys=["adapted_output"],
        prompt_template=(
            "Rewrite for {audience}: {text}\n"
            "Replace technical jargon with consumer-friendly language."
        ),
        source_tier="template",
    ),
}


# ═════════════════════════════════════════════════════════════════════════
# CLASSIFIER INVERSION TABLE
# Maps each classifier to an action spec for a new agent.
# ═════════════════════════════════════════════════════════════════════════


CLASSIFIER_INVERSIONS: dict[str, dict[str, str]] = {
    "factuality": {
        "failure_condition": "output contains fabricated facts not in sources",
        "action": "verify each factual claim against sources before output",
    },
    "attributability": {
        "failure_condition": "claims made without attribution to sources",
        "action": "add source references to every significant claim",
    },
    "internal_consistency": {
        "failure_condition": "output contradicts itself",
        "action": "check for self-contradictions and resolve them",
    },
    "causal_coherence": {
        "failure_condition": "causal links not supported by evidence",
        "action": "verify cause-effect claims against source evidence",
    },
    "policy_compliance": {
        "failure_condition": "decision violates company policy",
        "action": "check decision against policy rules before approval",
    },
    "reasoning_soundness": {
        "failure_condition": "reasoning is flawed or jumps to conclusions",
        "action": "validate logical chain from evidence to conclusion",
    },
    "decision_consistency": {
        "failure_condition": "decision seems arbitrary relative to precedent",
        "action": "compare against precedent decisions for similar cases",
    },
    "completeness": {
        "failure_condition": "key factors were ignored in the decision",
        "action": "enumerate all required factors from policy and verify each is addressed",
    },
}


# ═════════════════════════════════════════════════════════════════════════
# 1. EMPIRICAL FIX EXTRACTION
# Extract fix candidates directly from perturbation data. No LLM needed.
# ═════════════════════════════════════════════════════════════════════════


def extract_empirical_fixes(
    simulation_results: list[SimulationResult],
    baseline_quality: float,
    trace: list[dict],
    improvement_threshold: float = 0.10,
) -> list[Recommendation]:
    """
    Extract fix candidates directly from perturbation simulation data.

    Sources:
      A. Enhancement perturbations that improved quality → "modify this agent"
      B. Ablation perturbations that improved quality → "remove/limit this agent"
      C. Damping ratio > 1.0 → "cap the revision loop"

    All returned recommendations have measured (not estimated) improvements.
    """
    recommendations: list[Recommendation] = []

    # Get unique agents from trace
    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry.get("node") != "output"
    ))

    # ── Source A: Enhancement-derived fixes ───────────────────────────
    for agent in agents:
        enhance_scores = [
            r.quality_score for r in simulation_results
            if r.perturbation and r.perturbation.agent == agent
            and r.perturbation.strategy == "enhance"
        ]
        if not enhance_scores:
            continue

        enhancement_delta = float(np.mean(enhance_scores)) - baseline_quality
        if enhancement_delta > improvement_threshold:
            # Get the enhanced output preview for evidence
            enhance_results = [
                r for r in simulation_results
                if r.perturbation and r.perturbation.agent == agent
                and r.perturbation.strategy == "enhance"
                and r.perturbed_output
            ]
            preview = enhance_results[0].perturbed_output[:300] if enhance_results else ""

            recommendations.append(Recommendation(
                title=f"Enhance {agent.replace('_', ' ').title()} Agent",
                description=(
                    f"Enhancing the {agent} improved quality by "
                    f"+{enhancement_delta:.2f}. "
                    f"Enhanced output preview: {preview[:150]}..."
                    if preview else
                    f"Enhancing the {agent} improved quality by "
                    f"+{enhancement_delta:.2f}."
                ),
                intervention_type="modify_agent",
                target_agent=agent,
                estimated_failure_reduction=enhancement_delta,
                complexity="medium",
                priority=0,  # Will be re-ranked
                evidence_source="empirical",
                measurement_confidence="measured",
            ))

    # ── Source B: Ablation-derived fixes ──────────────────────────────
    for agent in agents:
        ablate_scores = [
            r.quality_score for r in simulation_results
            if r.perturbation and r.perturbation.agent == agent
            and r.perturbation.strategy == "ablate"
        ]
        if not ablate_scores:
            continue

        ablation_delta = float(np.mean(ablate_scores)) - baseline_quality
        if ablation_delta > improvement_threshold:
            # Removing this agent HELPS — it's harmful
            recommendations.append(Recommendation(
                title=f"Remove or Limit {agent.replace('_', ' ').title()} Agent",
                description=(
                    f"Removing the {agent} improved quality by "
                    f"+{ablation_delta:.2f}. This agent may be "
                    f"introducing errors or unnecessary complexity."
                ),
                intervention_type="remove_loop" if "critic" in agent else "restructure",
                target_agent=agent,
                estimated_failure_reduction=ablation_delta,
                complexity="low",
                priority=0,
                evidence_source="empirical",
                measurement_confidence="measured",
            ))

    # ── Source C: Damping ratio fix ───────────────────────────────────
    synthesizer_count = sum(1 for e in trace if e["node"] == "synthesizer")
    has_revisions = synthesizer_count > 1

    if has_revisions:
        critic_ablate_scores = [
            r.quality_score for r in simulation_results
            if r.perturbation and r.perturbation.agent == "critic"
            and r.perturbation.strategy == "ablate"
        ]
        baseline_scores = [r.quality_score for r in simulation_results if r.is_baseline]

        if critic_ablate_scores and baseline_scores:
            damping_ratio = float(
                float(np.mean(critic_ablate_scores)) / max(float(np.mean(baseline_scores)), 0.01)
            )

            if damping_ratio > 1.0:
                # Determine loop cap based on severity
                if damping_ratio > 1.3:
                    cap_desc = "Remove the revision loop entirely"
                    cap_title = "Remove Revision Loop"
                elif damping_ratio > 1.1:
                    cap_desc = "Cap the revision loop at 1 iteration"
                    cap_title = "Cap Revision Loop (1 iteration)"
                else:
                    cap_desc = "Cap the revision loop at 2 iterations"
                    cap_title = "Cap Revision Loop (2 iterations)"

                improvement = float(np.mean(critic_ablate_scores)) - baseline_quality
                recommendations.append(Recommendation(
                    title=cap_title,
                    description=(
                        f"{cap_desc}. Damping ratio = {damping_ratio:.2f} — "
                        f"each revision degrades quality. "
                        f"Removing the critic improves quality by +{improvement:.2f}."
                    ),
                    intervention_type="remove_loop",
                    target_agent="critic",
                    estimated_failure_reduction=max(0, improvement),
                    complexity="low",
                    priority=0,
                    evidence_source="empirical",
                    measurement_confidence="measured",
                ))

    return recommendations


# ═════════════════════════════════════════════════════════════════════════
# 2. COVERAGE GAP DETECTION
# Identify uncovered quality dimensions from per-classifier Shapley.
# ═════════════════════════════════════════════════════════════════════════


def detect_coverage_gaps(
    per_clf_shapley: dict[str, dict[str, float]],
    failing_classifiers: list[str],
    simulation_results: list[SimulationResult],
    trace: list[dict],
    coverage_threshold: float = 0.10,
) -> list[Recommendation]:
    """
    Identify uncovered quality dimensions and generate add-agent
    recommendations with specific placement and function specs.

    A quality dimension is "uncovered" when:
      1. Its classifier is failing (below threshold)
      2. All agents have near-zero Shapley for that classifier
         (no agent's output affects it)
    """
    if not per_clf_shapley or not failing_classifiers:
        return []

    recommendations: list[Recommendation] = []
    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry.get("node") != "output"
    ))

    for clf_name in failing_classifiers:
        if clf_name not in per_clf_shapley:
            continue

        agent_values = per_clf_shapley[clf_name]

        # Check if ALL agents have near-zero Shapley for this classifier
        all_near_zero = all(
            abs(agent_values.get(a, 0.0)) < coverage_threshold
            for a in agents
        )

        if not all_near_zero:
            continue  # Some agent owns this dimension — not a gap

        # ── This is a confirmed gap: generate add-agent recommendation ──

        # Find which agent's enhancement helped most on this classifier
        best_enhancer = _find_best_enhancer(
            clf_name, agents, simulation_results,
        )

        # Determine placement
        if best_enhancer:
            placement = {"adjacent_to": best_enhancer}
        else:
            placement = {"position": "before_output"}

        # Generate agent spec via 4-tier fallback
        agent_spec = generate_agent_spec(
            clf_name, simulation_results, trace,
        )

        # Compute estimated improvement from enhancement data
        est_improvement = _estimate_gap_fix_improvement(
            clf_name, simulation_results, agents,
        )

        recommendations.append(Recommendation(
            title=f"Add {clf_name.replace('_', ' ').title()} Agent",
            description=(
                f"No existing agent covers the {clf_name} quality dimension. "
                f"All agents have near-zero Shapley for this classifier. "
                f"Insert a {agent_spec.name} agent "
                f"{agent_spec.position.replace('_', ' ')}."
            ),
            intervention_type="add_agent",
            target_agent=None,
            estimated_failure_reduction=est_improvement,
            complexity="medium",
            priority=0,
            agent_spec=agent_spec,
            placement=placement,
            evidence_source="coverage_gap",
            measurement_confidence="measured" if est_improvement > 0 else "estimated",
        ))

    return recommendations


def _find_best_enhancer(
    classifier_name: str,
    agents: list[str],
    simulation_results: list[SimulationResult],
) -> Optional[str]:
    """Find which agent's enhancement most improved the target classifier."""
    best_agent = None
    best_delta = 0.0

    # Get baseline scores for this classifier
    baseline_results = [r for r in simulation_results if r.is_baseline]
    baseline_clf_scores = []
    for r in baseline_results:
        for c in r.classifier_results:
            if c.name == classifier_name:
                baseline_clf_scores.append(c.score)
    baseline_mean = float(np.mean(baseline_clf_scores)) if baseline_clf_scores else 0.5

    for agent in agents:
        enhance_clf_scores = []
        for r in simulation_results:
            if (r.perturbation and r.perturbation.agent == agent
                    and r.perturbation.strategy == "enhance"):
                for c in r.classifier_results:
                    if c.name == classifier_name:
                        enhance_clf_scores.append(c.score)

        if enhance_clf_scores:
            delta = float(np.mean(enhance_clf_scores)) - baseline_mean
            if delta > best_delta:
                best_delta = delta
                best_agent = agent

    return best_agent


def _estimate_gap_fix_improvement(
    classifier_name: str,
    simulation_results: list[SimulationResult],
    agents: list[str],
) -> float:
    """Estimate improvement from fixing a gap, using enhancement data."""
    baseline_results = [r for r in simulation_results if r.is_baseline]
    baseline_clf_scores = []
    for r in baseline_results:
        for c in r.classifier_results:
            if c.name == classifier_name:
                baseline_clf_scores.append(c.score)
    baseline_mean = float(np.mean(baseline_clf_scores)) if baseline_clf_scores else 0.5

    # Find best enhancement improvement for this classifier
    best_delta = 0.0
    for agent in agents:
        enhance_clf_scores = []
        for r in simulation_results:
            if (r.perturbation and r.perturbation.agent == agent
                    and r.perturbation.strategy == "enhance"):
                for c in r.classifier_results:
                    if c.name == classifier_name:
                        enhance_clf_scores.append(c.score)

        if enhance_clf_scores:
            delta = float(np.mean(enhance_clf_scores)) - baseline_mean
            best_delta = max(best_delta, delta)

    # If no enhancement data available, estimate from the gap size:
    # a dedicated agent should recover ~85% of the deficit
    if best_delta == 0.0 and baseline_mean < 0.5:
        best_delta = (1.0 - baseline_mean) * 0.85

    return round(best_delta, 4)


# ═════════════════════════════════════════════════════════════════════════
# 3. AGENT SPEC GENERATION (4-TIER FALLBACK)
# Generate a spec for a new agent to cover an uncovered quality dimension.
# ═════════════════════════════════════════════════════════════════════════


def generate_agent_spec(
    gap_classifier: str,
    simulation_results: list[SimulationResult],
    trace: list[dict],
    llm_fn: Optional[Callable] = None,
) -> AgentSpec:
    """
    Generate a spec for a new agent to cover an uncovered quality dimension.
    Tries four approaches in order of reliability:

      Tier 1: Template lookup (instant, deterministic)
      Tier 2: Classifier inversion (mechanical, general)
      Tier 3: Enhancement diff extraction (empirical, handles novel cases)
      Tier 4: LLM generation (last resort)
    """
    # Tier 1: Template lookup
    spec = _template_lookup(gap_classifier)
    if spec:
        return spec

    # Tier 2: Classifier inversion
    spec = _invert_classifier(gap_classifier)
    if spec:
        return spec

    # Tier 3: Enhancement diff extraction
    spec = _extract_from_enhancement_diff(gap_classifier, simulation_results, trace)
    if spec:
        return spec

    # Tier 4: LLM generation (last resort)
    if llm_fn:
        spec = _llm_generate_spec(gap_classifier, trace, llm_fn)
        if spec:
            return spec

    # Ultimate fallback: generic spec
    return _generic_spec(gap_classifier)


def _template_lookup(classifier_name: str) -> Optional[AgentSpec]:
    """Tier 1: Look up a pre-built agent template."""
    return AGENT_TEMPLATES.get(classifier_name)


def _invert_classifier(classifier_name: str) -> Optional[AgentSpec]:
    """
    Tier 2: Mechanically invert a classifier's evaluation logic
    into an agent's action logic.
    """
    if classifier_name not in CLASSIFIER_INVERSIONS:
        return None

    inv = CLASSIFIER_INVERSIONS[classifier_name]
    return AgentSpec(
        name=f"{classifier_name}_validator",
        position="before_output",
        function=f"Prevent: {inv['failure_condition']}. Action: {inv['action']}.",
        input_keys=["query", "current_output", "sources"],
        output_keys=["validated_output", "validation_flags"],
        source_tier="inversion",
    )


def _extract_from_enhancement_diff(
    gap_classifier: str,
    simulation_results: list[SimulationResult],
    trace: list[dict],
) -> Optional[AgentSpec]:
    """
    Tier 3: Extract agent spec from enhancement simulation diffs.

    Compares baseline output to the best enhancement output.
    The diff IS the functional spec for the new agent.
    """
    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry.get("node") != "output"
    ))

    # Find which agent's enhancement most improved this classifier
    best_agent = _find_best_enhancer(gap_classifier, agents, simulation_results)
    if not best_agent:
        return None

    # Get baseline output preview
    baseline_outputs = [
        r.perturbed_output for r in simulation_results
        if r.is_baseline and r.perturbed_output
    ]
    baseline_preview = baseline_outputs[0][:300] if baseline_outputs else ""

    # Get enhanced output preview
    enhanced_outputs = [
        r.perturbed_output for r in simulation_results
        if r.perturbation and r.perturbation.agent == best_agent
        and r.perturbation.strategy == "enhance"
        and r.perturbed_output
    ]
    enhanced_preview = enhanced_outputs[0][:300] if enhanced_outputs else ""

    if not enhanced_preview:
        return None

    return AgentSpec(
        name=f"{gap_classifier}_agent",
        position=f"adjacent_to_{best_agent}",
        function=(
            f"Transform output to improve {gap_classifier.replace('_', ' ')} quality. "
            f"Based on empirical enhancement data showing +improvement "
            f"when {best_agent} was enhanced."
        ),
        input_keys=["current_output", "query", "sources"],
        output_keys=["improved_output"],
        baseline_example=baseline_preview,
        enhanced_example=enhanced_preview,
        source_tier="enhancement_diff",
    )


def _llm_generate_spec(
    gap_classifier: str,
    trace: list[dict],
    llm_fn: Callable,
) -> Optional[AgentSpec]:
    """Tier 4: Use an LLM to generate an agent spec. Last resort."""
    trace_summary = ", ".join(
        entry["node"] for entry in trace if entry.get("node") != "output"
    )

    prompt = f"""You are designing a new agent to add to a multi-agent pipeline.

CURRENT PIPELINE: {trace_summary}
UNCOVERED QUALITY DIMENSION: {gap_classifier}

The pipeline is failing on the '{gap_classifier}' quality dimension because no
existing agent covers it. Design a new agent to fix this.

Respond with ONLY valid JSON:
{{"name": "...", "position": "before_X or after_X", "function": "1-2 sentence description",
  "input_keys": ["key1", "key2"], "output_keys": ["out1"]}}"""

    try:
        response = llm_fn(prompt, 0.2)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return AgentSpec(
            name=data.get("name", f"{gap_classifier}_agent"),
            position=data.get("position", "before_output"),
            function=data.get("function", f"Cover {gap_classifier} dimension"),
            input_keys=data.get("input_keys", []),
            output_keys=data.get("output_keys", []),
            source_tier="llm",
        )
    except (json.JSONDecodeError, ValueError, RuntimeError, Exception):
        return None


def _generic_spec(gap_classifier: str) -> AgentSpec:
    """Ultimate fallback: produce a generic spec for an unknown classifier."""
    return AgentSpec(
        name=f"{gap_classifier}_validator",
        position="before_output",
        function=f"Validate output quality on the {gap_classifier.replace('_', ' ')} dimension.",
        input_keys=["query", "current_output"],
        output_keys=["validated_output"],
        source_tier="generic",
    )


# ═════════════════════════════════════════════════════════════════════════
# 4. MEASURED RANKING
# Rank fix candidates by their measured improvement.
# ═════════════════════════════════════════════════════════════════════════


def rank_recommendations(
    recommendations: list[Recommendation],
) -> list[Recommendation]:
    """
    Rank recommendations by measured improvement.
    No additional LLM calls — uses data from the simulation phase.

    Sorting priority:
      1. Measured improvements before estimated ones
      2. Higher failure reduction first
      3. Lower complexity preferred
    """
    if not recommendations:
        return []

    complexity_order = {"low": 0, "medium": 1, "high": 2}

    ranked = sorted(
        recommendations,
        key=lambda r: (
            r.measurement_confidence == "measured",  # measured first
            r.estimated_failure_reduction,            # higher improvement
            -complexity_order.get(r.complexity, 1),   # lower complexity
        ),
        reverse=True,
    )

    for i, rec in enumerate(ranked):
        rec.priority = i + 1

    return ranked


# ═════════════════════════════════════════════════════════════════════════
# LEGACY: LLM-GENERATED RECOMMENDATIONS
# Kept for backward compatibility. Used as fallback when deterministic
# approaches produce no candidates.
# ═════════════════════════════════════════════════════════════════════════


def generate_recommendations(
    classification: FailureClassification,
    shapley_values: dict[str, float],
    trace: list[dict],
    query: str = "",
    domain: str = "rag",
    llm_fn: Optional[Callable] = None,
) -> list[Recommendation]:
    """
    Generate recommended fixes based on the diagnostic analysis.

    Uses an LLM to produce context-specific recommendations. If the LLM
    fails, falls back to rule-based defaults.

    This is the legacy API — kept for backward compatibility.
    The new deterministic pipeline uses extract_empirical_fixes() and
    detect_coverage_gaps() instead.
    """
    # If no LLM available, use fallback rules
    if llm_fn is None:
        return _fallback_recommendations(classification, shapley_values, domain)

    # Build context for the LLM
    trace_summary = ", ".join(
        f"{e['node']}({e.get('status', '?')})" for e in trace
    )
    shapley_str = ", ".join(
        f"{k}: {v:.3f}" for k, v in shapley_values.items()
    )

    prompt = f"""You are a multi-agent systems architect. Based on the following diagnostic analysis, recommend 2-3 specific fixes.

DOMAIN: {domain}
PIPELINE: {trace_summary}
FAILURE TYPE: {classification.failure_type}
DESCRIPTION: {classification.description}
FAILING CLASSIFIERS: {', '.join(classification.failing_classifiers)}
SHAPLEY VALUES: {shapley_str}
QUERY THAT TRIGGERED FAILURE: {query[:200]}

For each recommendation, provide:
- title: short name for the fix
- description: detailed explanation (2-3 sentences)
- intervention_type: one of "add_agent", "modify_agent", "remove_loop", "restructure"
- target_agent: which agent to modify (or null for "add_agent")
- estimated_failure_reduction: 0.0 to 1.0 estimate
- complexity: "low", "medium", or "high"

Respond with ONLY a JSON array of recommendation objects. No other text."""

    try:
        response = llm_fn(prompt, 0.2)

        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        recs_data = json.loads(text)
        if not isinstance(recs_data, list):
            recs_data = [recs_data]

        recs = []
        for i, r in enumerate(recs_data[:3]):
            recs.append(Recommendation(
                title=r.get("title", f"Recommendation {i+1}"),
                description=r.get("description", ""),
                intervention_type=r.get("intervention_type", "modify_agent"),
                target_agent=r.get("target_agent"),
                estimated_failure_reduction=float(r.get("estimated_failure_reduction", 0.5)),
                complexity=r.get("complexity", "medium"),
                priority=i + 1,
                evidence_source="llm",
                measurement_confidence="estimated",
            ))
        return recs

    except (json.JSONDecodeError, ValueError, RuntimeError):
        return _fallback_recommendations(classification, shapley_values, domain)


def _fallback_recommendations(
    classification: FailureClassification,
    shapley_values: dict[str, float],
    domain: str,
) -> list[Recommendation]:
    """
    Rule-based fallback recommendations for each failure type.

    These are used when the LLM is unavailable or fails to generate
    valid recommendations. They cover the most common fixes for each
    failure pattern.
    """
    recs = []

    if classification.failure_type == "architectural_gap":
        if domain == "decision":
            recs.append(Recommendation(
                title="Add Evidence Verification Agent",
                description=(
                    "Insert an agent that verifies customer claims against "
                    "company records before any decision is made."
                ),
                intervention_type="add_agent",
                target_agent=None,
                estimated_failure_reduction=0.85,
                complexity="medium",
                priority=1,
                evidence_source="template",
            ))
        else:
            recs.append(Recommendation(
                title="Add Premise Validator Agent",
                description=(
                    "Insert an agent that validates query premises against "
                    "retrieved evidence before synthesis."
                ),
                intervention_type="add_agent",
                target_agent=None,
                estimated_failure_reduction=0.85,
                complexity="medium",
                priority=1,
                evidence_source="template",
            ))

    elif classification.failure_type == "local":
        target = classification.dominant_agent or "unknown"
        recs.append(Recommendation(
            title=f"Upgrade {target.title()} Agent",
            description=(
                f"The {target} is the primary failure source. Enhance with "
                f"better validation and error handling."
            ),
            intervention_type="modify_agent",
            target_agent=target,
            estimated_failure_reduction=0.75,
            complexity="medium",
            priority=1,
        ))

    elif classification.failure_type == "feedback_amplification":
        recs.append(Recommendation(
            title="Cap Revision Loop",
            description=(
                "Limit the revision loop to 1 iteration. Quality degrades "
                "with additional revisions."
            ),
            intervention_type="remove_loop",
            target_agent="critic",
            estimated_failure_reduction=0.6,
            complexity="low",
            priority=1,
        ))

    elif classification.failure_type == "systemic":
        recs.append(Recommendation(
            title="Add Validation Checkpoint",
            description=(
                "Insert a validation checkpoint between the agents with "
                "highest interaction effects to catch cascading errors."
            ),
            intervention_type="restructure",
            target_agent=None,
            estimated_failure_reduction=0.5,
            complexity="high",
            priority=1,
        ))

    return recs


# ═════════════════════════════════════════════════════════════════════════
# RECOMMENDATION EVALUATION (kept for deeper validation)
# Test whether a recommendation would actually improve things.
# ═════════════════════════════════════════════════════════════════════════


def evaluate_recommendation(
    rec: Recommendation,
    trace: list[dict],
    query: str,
    output_text: str,
    sources: str = "",
    domain: str = "rag",
    num_eval_runs: int = 5,
    registry=None,
    llm_fn: Optional[Callable] = None,
) -> EvaluationResult:
    """
    Evaluate a recommendation by simulating its effect.

    We test the fix by:
      1. Measuring baseline quality (how bad is it now?)
      2. Simulating the fix using an LLM
      3. Measuring fixed quality (how good is it after the fix?)
      4. Computing the improvement and checking for regressions
    """
    from counterfact.classifiers import ClassifierRegistry, get_default_registry

    if llm_fn is None:
        raise ValueError("llm_fn is required for evaluating recommendations.")

    reg = registry or get_default_registry()

    # ── Step 1: Measure baseline quality ─────────────────────────────
    baseline_scores = []
    for _ in range(num_eval_runs):
        clf_results = reg.run_all(query, output_text, sources, domain)
        baseline_scores.append(ClassifierRegistry.aggregate_quality(clf_results))

    baseline_failure_rate = sum(
        1 for s in baseline_scores if s < 0.5
    ) / len(baseline_scores)

    # ── Step 2: Simulate the fix ─────────────────────────────────────
    fix_prompt = f"""Apply the following fix to this pipeline output:
FIX: {rec.title} — {rec.description}
ORIGINAL OUTPUT: {output_text[:500]}
QUERY: {query}

Generate an improved output that incorporates this fix. Be specific and rigorous."""

    fixed_output = llm_fn(fix_prompt, 0.3)

    # ── Step 3: Measure fixed quality ────────────────────────────────
    fixed_scores = []
    for _ in range(num_eval_runs):
        clf_results = reg.run_all(query, fixed_output, sources, domain)
        fixed_scores.append(ClassifierRegistry.aggregate_quality(clf_results))

    fixed_failure_rate = sum(
        1 for s in fixed_scores if s < 0.5
    ) / len(fixed_scores)

    # ── Step 4: Compute improvement metrics ──────────────────────────
    failure_reduction = baseline_failure_rate - fixed_failure_rate

    regression_count = sum(
        1 for b, f in zip(baseline_scores, fixed_scores) if f < b - 0.1
    )

    diffs = [b - f for b, f in zip(baseline_scores, fixed_scores)]
    ci_low = float(failure_reduction - 1.96 * np.std(diffs) / math.sqrt(num_eval_runs))
    ci_high = float(failure_reduction + 1.96 * np.std(diffs) / math.sqrt(num_eval_runs))

    # ── Step 5: Verdict ──────────────────────────────────────────────
    if failure_reduction > 0.3 and regression_count < num_eval_runs * 0.05:
        verdict = "recommended"
    elif failure_reduction > 0.1:
        verdict = "caution"
    else:
        verdict = "not_recommended"

    return EvaluationResult(
        recommendation=rec,
        baseline_failure_rate=baseline_failure_rate,
        fixed_failure_rate=fixed_failure_rate,
        failure_reduction=failure_reduction,
        regression_count=regression_count,
        confidence_interval=(ci_low, ci_high),
        verdict=verdict,
    )
