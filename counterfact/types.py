"""
Shared data types for the counterfact library.

This module is the foundation of the entire library — every other module
imports from here, but this module imports from nothing. This keeps the
dependency graph clean and makes it possible to use any single module
in isolation.

All types are plain dataclasses with a to_dict() method for JSON
serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ─── Statistics Types ───────────────────────────────────────────────────

@dataclass
class ConfidenceInterval:
    """
    Bootstrap confidence interval for a metric (e.g. Shapley value).
    """
    mean: float
    ci_low: float
    ci_high: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "n_samples": self.n_samples,
        }


# ─── Trace Types ─────────────────────────────────────────────────────────
# These represent a single recorded execution step in a multi-agent pipeline.

@dataclass
class TraceEntry:
    """
    A single recorded node execution in a pipeline trace.

    Every time an agent (node) in the pipeline runs, we capture what it
    received, what it produced, how long it took, and whether it succeeded.
    """
    node: str                    # Name of the agent/node (e.g. "retriever")
    input: dict                  # Summary of what the node received
    output: dict                 # Summary of what the node produced
    status: str = "pass"         # "pass" or "error"
    reasoning: str = ""          # Human-readable explanation of what happened
    duration_ms: float = 0.0     # How long this node took to execute

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "reasoning": self.reasoning,
            "duration_ms": round(self.duration_ms, 2),
        }


# ─── Classifier Types ───────────────────────────────────────────────────
# Quality classifiers score pipeline outputs on specific dimensions.

@dataclass
class ClassifierResult:
    """
    Result from a single quality classifier (e.g. "factuality": 0.85).

    Each classifier evaluates one quality dimension. The score is 0-1 where
    1.0 = perfect and 0.0 = complete failure. Weight controls how much this
    dimension matters in the aggregate quality score.
    """
    name: str                    # Classifier name (e.g. "factuality")
    score: float                 # 0.0 (fail) to 1.0 (pass)
    reasoning: str               # Why this score was given
    weight: float = 1.0          # Importance weight for aggregation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "reasoning": self.reasoning,
            "weight": self.weight,
        }


# ─── Eval Types ──────────────────────────────────────────────────────────
# Evals are ground-truth-free checks that identify pipeline issues.

@dataclass
class EvalResult:
    """
    Result of a single evaluation check.

    Each check looks at one aspect of health (e.g. "empty outputs",
    "latency spikes"). Severity is "info", "warning", or "critical".
    """
    check_name: str              # What was checked (e.g. "empty_output")
    passed: bool                 # Did this check pass?
    severity: str = "warning"    # "info", "warning", or "critical"
    message: str = ""            # Human-readable description of the finding
    agent: Optional[str] = None  # Which agent this finding applies to (if any)
    details: dict = field(default_factory=dict)  # Extra structured data

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "agent": self.agent,
            "details": self.details,
        }


@dataclass
class EvalSuite:
    """
    Aggregated results from running a full evaluation suite.

    Groups results by tier (1 = structural, 2 = consistency) and provides
    a summary of overall pipeline health.
    """
    results: list[EvalResult] = field(default_factory=list)
    tier_1_results: list[EvalResult] = field(default_factory=list)
    tier_2_results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no critical issues were found."""
        return not any(
            r.severity == "critical" and not r.passed
            for r in self.results
        )

    @property
    def num_issues(self) -> int:
        """Count of checks that did NOT pass."""
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "num_issues": self.num_issues,
            "total_checks": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }


# ─── Perturbation Types ─────────────────────────────────────────────────
# Perturbations simulate "what if" scenarios by modifying agent outputs.

@dataclass
class Perturbation:
    """
    A single perturbation to apply at an agent boundary.

    Only one strategy is supported:
      - "ablate": Remove the agent from the pipeline entirely.
        Its slot produces no output; downstream agents receive ""
        as their input for that step.

    Degrade and enhance strategies were removed — they have no well-defined
    meaning in the Shapley/LOO attribution framework.
    """
    agent: str                   # Which agent to perturb
    strategy: str                # "ablate"
    description: str             # Human-readable explanation
    magnitude: float             # Always 1.0 for ablation

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "strategy": self.strategy,
            "description": self.description,
            "magnitude": self.magnitude,
        }


@dataclass
class SimulationResult:
    """
    Result of a single Monte Carlo simulation run.

    Each simulation either runs the pipeline normally (baseline) or with
    one agent's output perturbed, then measures the quality of the final
    output using classifiers.
    """
    simulation_id: int
    perturbation: Optional[Perturbation]       # None for baseline runs
    quality_score: float                        # Aggregated quality (0–1)
    classifier_results: list[ClassifierResult] = field(default_factory=list)
    perturbed_output: str = ""                  # The output after perturbation
    is_baseline: bool = False                   # True if this is a normal run
    agent_traces: list[dict] = field(default_factory=list) # Traces of agents in this sim

    def to_dict(self) -> dict:
        return {
            "simulation_id": self.simulation_id,
            "perturbation": self.perturbation.to_dict() if self.perturbation else None,
            "quality_score": round(self.quality_score, 3),
            "classifier_results": [c.to_dict() for c in self.classifier_results],
            "perturbed_output_preview": self.perturbed_output[:200] if self.perturbed_output else "",
            "is_baseline": self.is_baseline,
            "agent_traces": self.agent_traces,
        }


# ─── Classification Types ───────────────────────────────────────────────
# After running simulations, we classify what kind of failure we're seeing.

@dataclass
class FailureClassification:
    """
    Classification of the failure type found in a pipeline.

    Types:
      - "no_failure": Pipeline is working correctly
      - "local": One specific agent is causing the problem
      - "architectural_gap": No agent is responsible — the pipeline is missing
        a capability (e.g. no premise-checking agent)
      - "feedback_amplification": A revision loop is making things worse
      - "systemic": Multiple agents interact to cause failure
    """
    failure_type: str
    confidence: float                                    # 0–1 confidence
    description: str                                     # What's going wrong
    evidence: list[str]                                  # Supporting evidence
    dominant_agent: Optional[str] = None                 # Which agent (for local)
    failing_classifiers: list[str] = field(default_factory=list)
    damping_ratio: Optional[float] = None                # For feedback loops
    confidence_explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "evidence": self.evidence,
            "dominant_agent": self.dominant_agent,
            "failing_classifiers": self.failing_classifiers,
            "damping_ratio": self.damping_ratio,
            "confidence_explanation": self.confidence_explanation,
        }


# ─── Agent Specification Types ──────────────────────────────────────────
# Specs for new agents to add to the pipeline.

@dataclass
class AgentSpec:
    """
    Specification for a new agent to add to the pipeline.

    Generated by the deterministic fix pipeline when a coverage gap is
    detected. Contains everything needed to implement the agent:
    name, position, function description, I/O keys, and optionally a
    ready-to-use prompt template or before/after examples.

    source_tier indicates how the spec was generated:
      - "template": from the pre-built template library
      - "inversion": mechanically inverted from a classifier
      - "enhancement_diff": extracted from enhancement simulation data
      - "llm": generated by an LLM (last resort)
    """
    name: str                           # e.g. "premise_validator"
    position: str                       # e.g. "before_synthesizer", "after_output"
    function: str                       # What the agent does (1-2 sentences)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    prompt_template: str = ""           # Ready-to-use prompt (from template)
    baseline_example: str = ""          # What bad output looks like (from enhancement diff)
    enhanced_example: str = ""          # What good output looks like (from enhancement diff)
    source_tier: str = ""               # "template", "inversion", "enhancement_diff", "llm"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "position": self.position,
            "function": self.function,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "source_tier": self.source_tier,
        }
        if self.prompt_template:
            d["prompt_template"] = self.prompt_template[:500]
        if self.baseline_example:
            d["baseline_example"] = self.baseline_example[:300]
        if self.enhanced_example:
            d["enhanced_example"] = self.enhanced_example[:300]
        return d


@dataclass
class FixConstraint:
    """
    A constraint the fix must satisfy, derived from diagnostic evidence.

    Constraints are generated automatically from the failing classifiers
    and Shapley values. They ensure that recommended fixes actually
    target the identified quality gaps.
    """
    source: str                # Where this constraint comes from (e.g. "classifier:premise_validity")
    requirement: str           # What the fix must do
    priority: str              # "must" or "should"
    target_dimension: str      # Which quality dimension this addresses

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "requirement": self.requirement,
            "priority": self.priority,
            "target_dimension": self.target_dimension,
        }


# ─── Recommendation Types ───────────────────────────────────────────────
# Prescriptive fixes suggested by the diagnostic engine.

@dataclass
class Recommendation:
    """
    A recommended fix for a diagnosed pipeline problem.

    Intervention types:
      - "add_agent": Add a new agent to the pipeline
      - "modify_agent": Change an existing agent's behavior
      - "remove_loop": Remove or limit a feedback loop
      - "restructure": Change the pipeline topology

    Evidence sources (new):
      - "empirical": derived from perturbation simulation data
      - "coverage_gap": derived from per-classifier Shapley matrix
      - "template": from pre-built agent template library
      - "llm": generated by LLM (legacy/fallback)
    """
    title: str
    description: str
    intervention_type: str             # "add_agent", "modify_agent", etc.
    target_agent: Optional[str]        # Which agent to modify (None for "add")
    estimated_failure_reduction: float  # How much this should help (0–1)
    complexity: str                    # "low", "medium", "high"
    priority: int                      # 1 = highest priority
    # ── New fields (all optional with defaults for backward compat) ──
    agent_spec: Optional['AgentSpec'] = None        # For add_agent fixes
    placement: Optional[dict] = None                # {"after": "X", "before": "Y"}
    evidence_source: str = "llm"                    # "empirical", "coverage_gap", "template", "llm"
    measurement_confidence: str = "estimated"        # "measured" or "estimated"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "title": self.title,
            "description": self.description,
            "intervention_type": self.intervention_type,
            "target_agent": self.target_agent,
            "estimated_failure_reduction": round(self.estimated_failure_reduction, 2),
            "complexity": self.complexity,
            "priority": self.priority,
            "evidence_source": self.evidence_source,
            "measurement_confidence": self.measurement_confidence,
        }
        if self.agent_spec:
            d["agent_spec"] = self.agent_spec.to_dict()
        if self.placement:
            d["placement"] = self.placement
        return d


@dataclass
class EvaluationResult:
    """
    Result of evaluating whether a recommendation actually works.

    We test recommendations by simulating the fix and measuring
    quality before and after.
    """
    recommendation: Recommendation
    baseline_failure_rate: float        # How often the pipeline fails now
    fixed_failure_rate: float           # How often it would fail after the fix
    failure_reduction: float            # Improvement (baseline - fixed)
    regression_count: int               # How many cases got worse
    confidence_interval: tuple[float, float]
    verdict: str                        # "recommended", "caution", "not_recommended"

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation.to_dict(),
            "baseline_failure_rate": round(self.baseline_failure_rate, 3),
            "fixed_failure_rate": round(self.fixed_failure_rate, 3),
            "failure_reduction": round(self.failure_reduction, 3),
            "regression_count": self.regression_count,
            "confidence_interval": [
                round(self.confidence_interval[0], 3),
                round(self.confidence_interval[1], 3),
            ],
            "verdict": self.verdict,
        }


# ─── Discovery Types ────────────────────────────────────────────────────
# The discovery agent analyzes unknown pipelines and produces a plan.

@dataclass
class AgentProfile:
    """
    Inferred profile of an agent discovered from traces.

    The discovery agent analyzes traces to figure out what each agent
    does, what it expects as input/output, and how to best test it.
    """
    name: str                           # Agent name from traces
    inferred_role: str                  # e.g. "retriever", "synthesizer", "validator"
    description: str                    # What this agent appears to do
    input_schema: dict = field(default_factory=dict)   # Expected input keys/types
    output_schema: dict = field(default_factory=dict)  # Expected output keys/types
    estimated_importance: float = 0.5   # How critical this agent seems (0–1)
    suggested_perturbations: list[str] = field(default_factory=list)  # e.g. ["ablate", "degrade"]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inferred_role": self.inferred_role,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "estimated_importance": round(self.estimated_importance, 2),
            "suggested_perturbations": self.suggested_perturbations,
        }


@dataclass
class PerturbationPlan:
    """
    A complete plan for how to test a newly discovered pipeline.

    This is the output of the discovery agent — it tells the perturbation
    engine what to test and how, so we don't need manual configuration
    for every new pipeline.
    """
    pipeline_description: str                # Summary of the pipeline
    agent_profiles: list[AgentProfile] = field(default_factory=list)
    perturbations: list[Perturbation] = field(default_factory=list)
    suggested_classifiers: list[str] = field(default_factory=list)  # Classifier names to use
    domain: str = "general"                  # Inferred domain
    confidence: float = 0.5                  # How confident the discovery agent is

    def to_dict(self) -> dict:
        return {
            "pipeline_description": self.pipeline_description,
            "agent_profiles": [a.to_dict() for a in self.agent_profiles],
            "perturbations": [p.to_dict() for p in self.perturbations],
            "suggested_classifiers": self.suggested_classifiers,
            "domain": self.domain,
            "confidence": round(self.confidence, 2),
        }

# ─── Thinking Model Types ───────────────────────────────────────────────
# These support evaluation of models that receive long system prompts
# and design tool call sequences on the fly.

@dataclass
class PromptSection:
    """
    A parsed section of a system prompt.

    Long system prompts (e.g., 15 pages) are segmented into semantic sections
    so we can attribute output quality to specific instructions. Each section
    is a logical unit of instructions that can be independently ablated.

    Categories:
      - "instruction": direct behavioral instructions (e.g., "Always cite sources")
      - "constraint": limitations and guardrails (e.g., "Never give medical advice")
      - "methodology": prescribed workflows (e.g., "Step 1: analyze, Step 2: plan")
      - "format": output formatting rules (e.g., "Return JSON with these fields")
      - "context": background information (e.g., "You are a financial analyst")
    """
    index: int                    # Position in the prompt (0-indexed)
    title: str                    # Short descriptive title for this section
    content: str                  # The actual text of this section
    category: str = "instruction" # "instruction", "constraint", "methodology", "format", "context"
    importance: float = 0.5       # Estimated importance (0–1), set by attribution
    char_start: int = 0           # Character offset where this section starts in the original prompt
    char_end: int = 0             # Character offset where this section ends

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "content": self.content[:200] + ("..." if len(self.content) > 200 else ""),
            "category": self.category,
            "importance": round(self.importance, 3),
            "char_range": [self.char_start, self.char_end],
        }


@dataclass
class ToolCall:
    """
    A recorded tool invocation in a thinking model pipeline.

    When a thinking model executes tool calls, each call is recorded with
    its inputs, outputs, timing, and whether it succeeded. This is the
    thinking-model equivalent of a TraceEntry.
    """
    tool_name: str               # Name of the tool called (e.g., "search_docs")
    tool_input: dict             # Arguments passed to the tool
    tool_output: dict            # What the tool returned
    step_index: int = 0          # Position in the execution sequence
    duration_ms: float = 0.0     # How long the tool call took
    status: str = "success"      # "success" or "error"
    error_message: str = ""      # Error details if status == "error"
    reasoning: str = ""          # Model's reasoning for making this call (from CoT)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": {k: str(v)[:200] for k, v in self.tool_output.items()}
                           if isinstance(self.tool_output, dict) else str(self.tool_output)[:200],
            "step_index": self.step_index,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error_message": self.error_message[:200],
            "reasoning": self.reasoning[:200],
        }


@dataclass
class PlanStep:
    """
    A planned step in a thinking model's execution plan.

    Before executing tool calls, thinking models often generate a plan.
    Each step describes what tool to call, with what arguments, and why.
    """
    step_index: int              # Position in the plan
    tool_name: str               # Which tool to call
    description: str             # What this step accomplishes
    depends_on: list[int] = field(default_factory=list)  # Indices of steps this depends on
    reasoning: str = ""          # Why this step is needed

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "description": self.description,
            "depends_on": self.depends_on,
            "reasoning": self.reasoning[:200],
        }


@dataclass
class PromptAnalysisResult:
    """
    Complete result of analyzing a thinking model's system prompt.

    This is the output of prompt_analysis.analyze_prompt() — it tells you
    which sections of the prompt matter, which are ignored, and which
    conflict with each other.
    """
    sections: list[PromptSection] = field(default_factory=list)
    section_attributions: dict[int, float] = field(default_factory=dict)  # section index -> Shapley value
    dead_sections: list[int] = field(default_factory=list)                # Indices of ignored sections
    conflicting_pairs: list[tuple[int, int]] = field(default_factory=list)  # Pairs of conflicting sections
    plan_quality_score: float = 0.0     # 0–1 score of plan quality
    plan_quality_details: dict = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "num_sections": len(self.sections),
            "sections": [s.to_dict() for s in self.sections],
            "section_attributions": {str(k): round(v, 4) for k, v in self.section_attributions.items()},
            "dead_sections": self.dead_sections,
            "conflicting_pairs": [list(p) for p in self.conflicting_pairs],
            "plan_quality_score": round(self.plan_quality_score, 3),
            "plan_quality_details": self.plan_quality_details,
            "confidence": round(self.confidence, 3),
        }


# Type alias for classifier functions — these take (query, output, sources)
# and return a ClassifierResult.
ClassifierFn = Callable[[str, str, str], ClassifierResult]
