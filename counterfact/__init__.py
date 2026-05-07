"""
counterfact — Deterministic, evidence-driven diagnostics for multi-agent pipelines.

Replace:
    from langgraph.graph import StateGraph, END
With:
    from counterfact import StateGraph, END

Everything works the same, plus you get:
    compiled.get_trace()       — automatic execution tracing
    compiled.eval(...)         — ground-truth-free evaluation
    compiled.diagnose(...)     — real counterfactual analysis via pipeline re-execution

Counterfactual analysis works by actually re-running your pipeline with
agents ablated (replaced by no-ops), not by simulating outputs with an LLM.

Each module can also be used independently:
    from counterfact.evals import run_structural_checks
    from counterfact.discovery import discover_pipeline
    from counterfact.prompt_analysis import analyze_prompt
    from counterfact.tool_tracing import ToolTracer
"""

# ─── Core (always available, lightweight) ────────────────────────────────

# ─── Attribution (Shapley/LOO math) ──────────────────────────────────────
from counterfact.attribution import (
    classify_failure,
    compute_loo_attribution,
    compute_per_classifier_loo,
    compute_per_classifier_shapley,  # backward-compat alias
    compute_shapley_values,
    is_loo_inconclusive,
)

# ─── Classifiers (quality scoring, usable independently) ─────────────────
from counterfact.classifiers import (
    ClassifierRegistry,
    get_default_registry,
    register_classifier,
    set_llm_caller,
)

# ─── Full diagnostics (orchestrator) ─────────────────────────────────────
from counterfact.diagnostics import DiagnosticReport, run_full_diagnostic

# ─── Discovery (AI pipeline analysis, usable independently) ─────────────
from counterfact.discovery import discover_pipeline

# ─── Evals (ground-truth-free checks, usable independently) ─────────────
from counterfact.evals import (
    check_duplicate_agents,
    check_empty_outputs,
    check_error_status,
    check_faithfulness,
    check_grounding,
    check_inter_agent_coherence,
    check_latency_anomalies,
    check_output_length_anomalies,
    # Thinking model checks
    check_plan_completeness,
    check_schema_violations,
    check_tool_error_rate,
    check_tool_redundancy,
    run_consistency_checks,
    run_eval_suite,
    run_structural_checks,
)
from counterfact.graph import END, START, CounterfactualGraph, StateGraph

# ─── Optimizer (single-objective quality maximization) ────────────────────
from counterfact.optimizer import (
    OptimizationResult,
    SearchSpace,
    optimize_pipeline,
)
from counterfact.optimizer import (
    TrialResult as OptTrialResult,
)

# ─── Perturbation (Monte Carlo simulation) ───────────────────────────────
from counterfact.perturbation import (
    generate_perturbations,
    run_monte_carlo,
)

# ─── Prompt Analysis (thinking model evaluation) ─────────────────────────
from counterfact.prompt_analysis import (
    analyze_prompt,
    check_plan_quality,
    detect_conflicting_sections,
    detect_dead_sections,
    parse_prompt_sections,
    run_prompt_section_attribution,
)

# ─── Recommendations (fix generation + evaluation) ───────────────────────
from counterfact.recommendations import (
    AGENT_TEMPLATES,
    CLASSIFIER_INVERSIONS,
    detect_coverage_gaps,
    evaluate_recommendation,
    extract_empirical_fixes,
    generate_agent_spec,
    generate_recommendations,
    rank_recommendations,
)

# ─── Tool Tracing (thinking model tool call capture) ─────────────────────
from counterfact.tool_tracing import (
    ToolTracer,
    perturb_tool_result,
    tool_calls_to_trace,
)
from counterfact.tracing import TracingContext

# ─── Types (all shared data classes) ─────────────────────────────────────
from counterfact.types import (
    AgentProfile,
    AgentSpec,
    ClassifierFn,
    ClassifierResult,
    EvalResult,
    EvalSuite,
    EvaluationResult,
    FailureClassification,
    FixConstraint,
    Perturbation,
    PerturbationPlan,
    PlanStep,
    PromptAnalysisResult,
    # Thinking model types
    PromptSection,
    Recommendation,
    SimulationResult,
    ToolCall,
    TraceEntry,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "StateGraph", "CounterfactualGraph", "END", "START", "TracingContext",
    # Types
    "TraceEntry", "ClassifierResult", "EvalResult", "EvalSuite",
    "Perturbation", "SimulationResult", "FailureClassification",
    "Recommendation", "EvaluationResult", "AgentProfile", "PerturbationPlan",
    "AgentSpec", "FixConstraint",
    "PromptSection", "ToolCall", "PlanStep", "PromptAnalysisResult",
    "ClassifierFn",
    # Evals
    "run_eval_suite", "run_structural_checks", "run_consistency_checks",
    "check_empty_outputs", "check_error_status", "check_schema_violations",
    "check_latency_anomalies", "check_output_length_anomalies",
    "check_duplicate_agents", "check_faithfulness",
    "check_inter_agent_coherence", "check_grounding",
    "check_plan_completeness", "check_tool_error_rate", "check_tool_redundancy",
    # Classifiers
    "ClassifierRegistry", "register_classifier", "set_llm_caller",
    "get_default_registry",
    # Discovery
    "discover_pipeline",
    # Attribution
    "compute_loo_attribution", "compute_shapley_values",
    "compute_per_classifier_loo", "compute_per_classifier_shapley",
    "classify_failure", "is_loo_inconclusive",
    # Perturbation
    "generate_perturbations", "run_monte_carlo",
    # Recommendations
    "generate_recommendations", "evaluate_recommendation",
    "extract_empirical_fixes", "detect_coverage_gaps",
    "generate_agent_spec", "rank_recommendations",
    "AGENT_TEMPLATES", "CLASSIFIER_INVERSIONS",
    # Diagnostics
    "DiagnosticReport", "run_full_diagnostic",
    # Prompt Analysis (thinking models)
    "analyze_prompt", "parse_prompt_sections", "check_plan_quality",
    "run_prompt_section_attribution", "detect_dead_sections",
    "detect_conflicting_sections",
    # Tool Tracing (thinking models)
    "ToolTracer", "tool_calls_to_trace", "perturb_tool_result",
    # Optimizer
    "optimize_pipeline", "SearchSpace", "OptimizationResult", "OptTrialResult",
]
