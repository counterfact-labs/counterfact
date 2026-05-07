"""
AI-powered pipeline discovery agent.

This module solves the cold-start problem: when a third-party user wants
to use counterfact on their pipeline, we don't know anything about it.
The discovery agent analyzes traces and/or pipeline descriptions to
automatically figure out:

  1. What each agent does (role inference)
  2. What data each agent expects and produces (schema inference)
  3. How important each agent is (importance estimation)
  4. How to best test each agent (perturbation strategy)
  5. Which quality classifiers to use (classifier suggestion)

The output is a PerturbationPlan — a complete, machine-readable spec
that tells the perturbation engine what to test and how.

Dependencies: types only (LLM function is injected)
"""

import json
from typing import Any, Callable, Optional

from counterfact.types import (
    AgentProfile,
    Perturbation,
    PerturbationPlan,
)

# ═════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# The discover_pipeline function orchestrates the entire discovery flow.
# ═════════════════════════════════════════════════════════════════════════


def discover_pipeline(
    traces: Optional[list[list[dict]]] = None,
    pipeline_description: Optional[str] = None,
    llm_fn: Optional[Callable[[str, float], str]] = None,
) -> PerturbationPlan:
    """
    Analyze a pipeline and produce a complete perturbation plan.

    This is the main entry point. Feed it either:
      - traces: a list of execution traces (each trace is a list of
        trace entry dicts from running the pipeline)
      - pipeline_description: a natural language description of the
        pipeline and its agents
      - both: traces + description for best results

    At least one of traces or pipeline_description must be provided.
    An llm_fn is always required — we need an LLM to understand the
    pipeline structure.

    Returns a PerturbationPlan that can be passed directly to the
    perturbation engine.

    # PROMPT TUNING NOTE: The quality of the discovery depends heavily
    # on the prompts in the sub-functions. Each one has tuning notes.
    # Start by tuning infer_agent_roles, since everything else depends
    # on it getting the roles right.
    """
    if traces is None and pipeline_description is None:
        raise ValueError(
            "At least one of 'traces' or 'pipeline_description' must be provided."
        )
    if llm_fn is None:
        raise ValueError(
            "llm_fn is required — the discovery agent needs an LLM to "
            "analyze the pipeline."
        )

    # ── Step 1: Infer agent roles ────────────────────────────────────
    # Figure out what each agent does from the traces / description
    agent_profiles = infer_agent_roles(
        traces=traces,
        pipeline_description=pipeline_description,
        llm_fn=llm_fn,
    )

    # ── Step 2: Infer data schemas ───────────────────────────────────
    # Figure out input/output schemas from trace data
    if traces:
        agent_profiles = infer_schemas(agent_profiles, traces)

    # ── Step 3: Estimate importance ──────────────────────────────────
    # Guess which agents are most critical (before we run perturbations)
    agent_profiles = estimate_importance(agent_profiles, llm_fn)

    # ── Step 4: Design perturbation strategies ───────────────────────
    # Decide what perturbations to run for each agent
    perturbations = suggest_perturbation_strategies(agent_profiles, llm_fn)

    # ── Step 5: Suggest classifiers ──────────────────────────────────
    # Recommend which quality classifiers to use for this pipeline
    suggested_classifiers = suggest_classifiers(agent_profiles, llm_fn)

    # ── Step 6: Infer domain ─────────────────────────────────────────
    # Classify what kind of pipeline this is (RAG, decision, etc.)
    domain = infer_domain(agent_profiles, llm_fn)

    # ── Step 7: Compute overall confidence ───────────────────────────
    # How confident are we in this discovery?
    confidence = _compute_discovery_confidence(
        traces=traces,
        pipeline_description=pipeline_description,
        agent_profiles=agent_profiles,
    )

    # Build the summary description
    agent_names = [p.name for p in agent_profiles]
    description = (
        f"Pipeline with {len(agent_profiles)} agents: {', '.join(agent_names)}. "
        f"Domain: {domain}."
    )
    if pipeline_description:
        description = pipeline_description[:200] + " | " + description

    return PerturbationPlan(
        pipeline_description=description,
        agent_profiles=agent_profiles,
        perturbations=perturbations,
        suggested_classifiers=suggested_classifiers,
        domain=domain,
        confidence=confidence,
    )


# ═════════════════════════════════════════════════════════════════════════
# STEP 1: ROLE INFERENCE
# ═════════════════════════════════════════════════════════════════════════


def infer_agent_roles(
    traces: Optional[list[list[dict]]] = None,
    pipeline_description: Optional[str] = None,
    llm_fn: Optional[Callable] = None,
) -> list[AgentProfile]:
    """
    Infer what each agent does from traces and/or descriptions.

    This is the most important step — everything else depends on getting
    the roles right. We use the LLM to classify each agent into a role
    like "retriever", "synthesizer", "validator", etc.

    # PROMPT TUNING NOTE: This is the highest-priority prompt to tune.
    # Key issues to watch for:
    #   - Agents with ambiguous roles (e.g. a "processor" that does multiple things)
    #   - Domain-specific agent types the LLM hasn't seen before
    #   - Agents that appear in some traces but not others (conditional paths)
    #
    # Tuning strategy:
    #   1. Run on 5-10 diverse pipelines and log the inferred roles
    #   2. Compare to human-labeled roles
    #   3. Add examples of tricky cases to the prompt's few-shot section
    #   4. Repeat until >90% accuracy on role classification
    """
    if llm_fn is None:
        raise ValueError("llm_fn required for role inference.")

    # Collect agent information from traces
    agent_info = _extract_agent_info_from_traces(traces) if traces else {}

    # Build context for the LLM
    context_parts = []

    if agent_info:
        context_parts.append("OBSERVED AGENTS IN TRACES:")
        for name, info in agent_info.items():
            context_parts.append(
                f"\n  Agent: {name}"
                f"\n    Appears {info['count']} time(s)"
                f"\n    Input keys: {info['input_keys']}"
                f"\n    Output keys: {info['output_keys']}"
                f"\n    Sample input: {info['sample_input'][:200]}"
                f"\n    Sample output: {info['sample_output'][:200]}"
            )

    if pipeline_description:
        context_parts.append(f"\nPIPELINE DESCRIPTION:\n{pipeline_description}")

    context = "\n".join(context_parts)

    # PROMPT TUNING: This is the master role inference prompt.
    # Add few-shot examples here for better accuracy on tricky pipelines.
    prompt = f"""You are analyzing a multi-agent AI pipeline to understand what each agent does.

{context}

For each agent, determine:
1. inferred_role: one of "retriever", "synthesizer", "validator", "critic", "reviewer", "decision_maker", "classifier", "extractor", "formatter", "router", "other"
2. description: what this agent appears to do (1-2 sentences)

Common patterns to look for:
- Agents with "doc" or "retriev" in their input/output → likely retriever
- Agents that produce long text from shorter inputs → likely synthesizer
- Agents that produce short judgments/scores → likely critic or validator
- Agents that extract structured data from text → likely extractor
- Agents that make approve/deny decisions → likely decision_maker

Respond with ONLY a JSON array of objects, one per agent:
[{{"name": "agent_name", "inferred_role": "role", "description": "what it does"}}]"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_json_response(response)
        if not isinstance(parsed, list):
            parsed = [parsed]

        profiles = []
        for item in parsed:
            profiles.append(AgentProfile(
                name=item.get("name", "unknown"),
                inferred_role=item.get("inferred_role", "other"),
                description=item.get("description", ""),
            ))
        return profiles

    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: create basic profiles from trace data
        if agent_info:
            return [
                AgentProfile(name=name, inferred_role="other", description="Role unknown (inference failed)")
                for name in agent_info.keys()
            ]
        return []


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: SCHEMA INFERENCE
# ═════════════════════════════════════════════════════════════════════════


def infer_schemas(
    agent_profiles: list[AgentProfile],
    traces: list[list[dict]],
) -> list[AgentProfile]:
    """
    Infer input/output schemas from trace data.

    This doesn't need an LLM — we just look at what keys appear in the
    actual trace data. By looking across multiple traces, we can identify
    which keys are always present (required) vs sometimes present (optional).
    """
    # Collect observed keys across all traces
    input_keys_by_agent: dict[str, dict[str, int]] = {}
    output_keys_by_agent: dict[str, dict[str, int]] = {}
    trace_count_by_agent: dict[str, int] = {}

    for trace in traces:
        for entry in trace:
            node = entry.get("node", "unknown")
            trace_count_by_agent[node] = trace_count_by_agent.get(node, 0) + 1

            # Count input key appearances
            inp = entry.get("input", {})
            if isinstance(inp, dict):
                if node not in input_keys_by_agent:
                    input_keys_by_agent[node] = {}
                for key in inp.keys():
                    input_keys_by_agent[node][key] = input_keys_by_agent[node].get(key, 0) + 1

            # Count output key appearances
            out = entry.get("output", {})
            if isinstance(out, dict):
                if node not in output_keys_by_agent:
                    output_keys_by_agent[node] = {}
                for key in out.keys():
                    output_keys_by_agent[node][key] = output_keys_by_agent[node].get(key, 0) + 1

    # Update profiles with inferred schemas
    for profile in agent_profiles:
        name = profile.name
        count = trace_count_by_agent.get(name, 0)

        if name in input_keys_by_agent and count > 0:
            # Keys that appear in all traces are "required", others are "optional"
            profile.input_schema = {
                key: "required" if freq >= count else "optional"
                for key, freq in input_keys_by_agent[name].items()
            }

        if name in output_keys_by_agent and count > 0:
            profile.output_schema = {
                key: "required" if freq >= count else "optional"
                for key, freq in output_keys_by_agent[name].items()
            }

    return agent_profiles


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: IMPORTANCE ESTIMATION
# ═════════════════════════════════════════════════════════════════════════


def estimate_importance(
    agent_profiles: list[AgentProfile],
    llm_fn: Callable,
) -> list[AgentProfile]:
    """
    Estimate how important each agent is to the pipeline.

    This is a rough pre-test estimate — real importance is determined
    by Shapley attribution later. But having a prior helps us allocate
    more simulation budget to agents that are likely to matter.

    # PROMPT TUNING NOTE: This prompt should be calibrated so that:
    #   - Retrievers and synthesizers get 0.7-0.9 (high importance)
    #   - Formatters and routers get 0.2-0.4 (low importance)
    #   - Validators/critics get 0.5-0.7 (medium importance)
    #   - Unknown agents get 0.5 (neutral)
    """
    # Build a summary of all agents for context
    agent_summary = "\n".join(
        f"- {p.name} (role: {p.inferred_role}): {p.description}"
        for p in agent_profiles
    )

    # PROMPT TUNING: Adjust importance calibration here.
    prompt = f"""You are estimating how important each agent is in a pipeline.

AGENTS:
{agent_summary}

For each agent, estimate importance from 0.0 (trivial) to 1.0 (critical):
- Retrievers/data sources: usually 0.7-0.9 (pipeline depends on their data)
- Synthesizers/generators: usually 0.7-0.9 (create the main output)
- Validators/critics: usually 0.5-0.7 (quality gate)
- Formatters/routers: usually 0.2-0.4 (cosmetic/routing)

Respond with ONLY a JSON object mapping agent name to importance:
{{"agent_name": 0.0-1.0, ...}}"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_json_response(response)

        for profile in agent_profiles:
            if profile.name in parsed:
                profile.estimated_importance = float(parsed[profile.name])

    except (json.JSONDecodeError, ValueError, KeyError, RuntimeError, Exception):
        # Fallback: use role-based defaults
        role_defaults = {
            "retriever": 0.8, "synthesizer": 0.85, "validator": 0.6,
            "critic": 0.6, "reviewer": 0.5, "decision_maker": 0.8,
            "classifier": 0.5, "extractor": 0.7, "formatter": 0.3,
            "router": 0.3, "other": 0.5,
        }
        for profile in agent_profiles:
            profile.estimated_importance = role_defaults.get(
                profile.inferred_role, 0.5
            )

    return agent_profiles


# ═════════════════════════════════════════════════════════════════════════
# STEP 4: PERTURBATION STRATEGY SUGGESTION
# ═════════════════════════════════════════════════════════════════════════


def suggest_perturbation_strategies(
    agent_profiles: list[AgentProfile],
    llm_fn: Callable,
) -> list[Perturbation]:
    """
    Suggest which perturbation strategies to use for each agent.

    Different agent types need different perturbation approaches:
      - Retrievers: ablate (no docs), degrade (irrelevant docs)
      - Synthesizers: degrade (hallucinate), enhance (add validation)
      - Validators: ablate (skip validation), enhance (stricter checks)

    # PROMPT TUNING NOTE: This prompt controls perturbation design.
    # Key things to tune:
    #   - Perturbation descriptions should be specific to the role
    #   - Magnitudes should reflect how impactful each perturbation is
    #   - The LLM should suggest role-appropriate degradation modes
    #
    # Tuning strategy:
    #   1. Run perturbations with these descriptions
    #   2. Check if the LLM-generated perturbed outputs actually differ
    #   3. If not different enough, make descriptions more explicit
    #   4. If too different, add constraints to the descriptions
    """
    agent_summary = "\n".join(
        f"- {p.name} (role: {p.inferred_role}, importance: {p.estimated_importance:.1f}): {p.description}"
        for p in agent_profiles
    )

    # PROMPT TUNING: This prompt designs the perturbation strategies.
    prompt = f"""You are designing perturbation strategies to test each agent in a pipeline.

AGENTS:
{agent_summary}

For each agent, suggest perturbation strategies. Each strategy should describe
exactly HOW to perturb this specific agent given its role.

For each agent, provide up to 3 perturbations:
- ablate: what happens when this agent is removed
- degrade: how this specific agent would fail (be specific to its role)
- enhance: how this agent could be improved (be specific to its role)

Respond with ONLY a JSON array:
[{{"agent": "name", "strategy": "ablate|degrade|enhance", "description": "specific description", "magnitude": 0.0-1.0}}]"""

    try:
        response = llm_fn(prompt, 0.2)
        parsed = _parse_json_response(response)
        if not isinstance(parsed, list):
            parsed = [parsed]

        perturbations = []
        for item in parsed:
            perturbations.append(Perturbation(
                agent=item.get("agent", "unknown"),
                strategy=item.get("strategy", "ablate"),
                description=item.get("description", ""),
                magnitude=float(item.get("magnitude", 0.5)),
            ))

        # Also update agent profiles with suggested perturbations
        for profile in agent_profiles:
            profile.suggested_perturbations = [
                p.strategy for p in perturbations if p.agent == profile.name
            ]

        return perturbations

    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: generate default perturbations
        from counterfact.perturbation import generate_perturbations as gen_default
        # Build a minimal trace from agent profiles
        fake_trace = [{"node": p.name} for p in agent_profiles]
        return gen_default(fake_trace)


# ═════════════════════════════════════════════════════════════════════════
# STEP 5: CLASSIFIER SUGGESTION
# ═════════════════════════════════════════════════════════════════════════


def suggest_classifiers(
    agent_profiles: list[AgentProfile],
    llm_fn: Callable,
) -> list[str]:
    """
    Suggest which quality classifiers to use for this pipeline.

    Different pipelines need different quality dimensions. A RAG pipeline
    needs factuality and attributability checks. A decision pipeline needs
    policy compliance and evidence sufficiency checks.

    # PROMPT TUNING NOTE: The classifier suggestions should map to
    # actual classifiers in the ClassifierRegistry. If the LLM suggests
    # classifiers that don't exist, we need to either:
    #   1. Add them to the registry, or
    #   2. Map the suggestion to the closest existing classifier
    #
    # Tuning strategy:
    #   1. Collect classifier suggestions from 10+ diverse pipelines
    #   2. Group similar suggestions into categories
    #   3. Ensure the registry has a classifier for each common category
    """
    agent_summary = "\n".join(
        f"- {p.name} (role: {p.inferred_role}): {p.description}"
        for p in agent_profiles
    )

    # PROMPT TUNING: Classification suggestion prompt.
    prompt = f"""You are recommending quality classifiers for a multi-agent pipeline.

PIPELINE AGENTS:
{agent_summary}

Available classifier types:
- factuality: checks if output matches source facts
- attributability: checks if claims are traceable to sources
- premise_validity: checks if the query's assumptions are valid
- internal_consistency: checks for self-contradictions
- causal_coherence: checks if reasoning is logical
- regulatory_compliance: checks for inappropriate financial advice
- policy_compliance: checks if decisions follow company policy
- evidence_sufficiency: checks if all evidence was verified
- reasoning_soundness: checks if logic chain is valid
- decision_consistency: checks for arbitrary decisions
- completeness: checks if all factors were considered

Which classifiers should be used for this pipeline? Pick 3-5 that are
most relevant given the agent roles and pipeline purpose.

Respond with ONLY a JSON array of classifier names:
["factuality", "attributability", ...]"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_json_response(response)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
        return ["factuality", "internal_consistency", "causal_coherence"]

    except (json.JSONDecodeError, ValueError):
        # Fallback: suggest general-purpose classifiers
        return ["factuality", "internal_consistency", "causal_coherence"]


# ═════════════════════════════════════════════════════════════════════════
# STEP 6: DOMAIN INFERENCE
# ═════════════════════════════════════════════════════════════════════════


def infer_domain(
    agent_profiles: list[AgentProfile],
    llm_fn: Callable,
) -> str:
    """
    Infer what domain/category this pipeline belongs to.

    This determines which set of classifiers to use by default.
    Common domains: "rag", "decision", "analysis", "general".

    # PROMPT TUNING NOTE: The domain should match domains registered
    # in the ClassifierRegistry. If new domains are added, update the
    # prompt's list of options.
    """
    roles = [f"{p.name}: {p.inferred_role}" for p in agent_profiles]

    # PROMPT TUNING: Domain inference prompt.
    prompt = f"""Based on these agent roles in a pipeline, what domain is this pipeline for?

AGENTS: {', '.join(roles)}

Options:
- "rag": Retrieval-augmented generation (search + synthesize)
- "decision": Decision-making pipeline (analyze + decide)
- "analysis": Data analysis pipeline (process + analyze)
- "general": Other/general purpose

Respond with ONLY one word (the domain name, in quotes): "rag" or "decision" or "analysis" or "general"
"""

    try:
        response = llm_fn(prompt, 0.1).strip().strip('"').strip("'").lower()
        if response in ("rag", "decision", "analysis", "general"):
            return response
        return "general"

    except Exception:
        return "general"


# ═════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════


def _extract_agent_info_from_traces(
    traces: list[list[dict]],
) -> dict[str, dict]:
    """
    Extract summary information about each agent from trace data.

    Collects agent names, input/output key patterns, and sample data
    across all provided traces.
    """
    agent_info: dict[str, dict] = {}

    for trace in traces:
        for entry in trace:
            node = entry.get("node", "unknown")

            if node not in agent_info:
                agent_info[node] = {
                    "count": 0,
                    "input_keys": set(),
                    "output_keys": set(),
                    "sample_input": "",
                    "sample_output": "",
                }

            info = agent_info[node]
            info["count"] += 1

            # Collect input keys
            inp = entry.get("input", {})
            if isinstance(inp, dict):
                info["input_keys"].update(inp.keys())
                # Keep first non-empty sample
                if not info["sample_input"]:
                    info["sample_input"] = json.dumps(inp, default=str)[:300]

            # Collect output keys
            out = entry.get("output", {})
            if isinstance(out, dict):
                info["output_keys"].update(out.keys())
                if not info["sample_output"]:
                    info["sample_output"] = json.dumps(out, default=str)[:300]

    # Convert sets to sorted lists for JSON serialization
    for info in agent_info.values():
        info["input_keys"] = sorted(info["input_keys"])
        info["output_keys"] = sorted(info["output_keys"])

    return agent_info


def _compute_discovery_confidence(
    traces: Optional[list[list[dict]]],
    pipeline_description: Optional[str],
    agent_profiles: list[AgentProfile],
) -> float:
    """
    Compute how confident we are in the discovery results.

    Confidence is higher when:
      - More traces are available (more data = better inference)
      - A description is provided (reduces ambiguity)
      - All agents have clear roles (none classified as "other")
    """
    score = 0.3  # base confidence

    # More traces = more confidence
    if traces:
        n_traces = len(traces)
        if n_traces >= 50:
            score += 0.3
        elif n_traces >= 10:
            score += 0.2
        elif n_traces >= 3:
            score += 0.1

    # Having a description helps
    if pipeline_description and len(pipeline_description) > 50:
        score += 0.15

    # All agents having clear roles is a good sign
    if agent_profiles:
        known_roles = sum(1 for p in agent_profiles if p.inferred_role != "other")
        role_clarity = known_roles / len(agent_profiles)
        score += 0.2 * role_clarity

    # Both traces AND description = high confidence
    if traces and pipeline_description:
        score += 0.05

    return min(0.95, round(score, 2))


def _parse_json_response(text: str) -> Any:
    """
    Parse a JSON response from an LLM, handling markdown code blocks and truncation.
    """
    text = text.strip()

    import re
    # Extract from markdown block if present
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Grab the outermost array or object
        start1, start2 = text.find("["), text.find("{")
        start = min(s for s in (start1, start2) if s != -1) if (start1 != -1 or start2 != -1) else -1
        if start != -1:
            end1, end2 = text.rfind("]"), text.rfind("}")
            end = max(end1, end2)
            if end >= start:
                text = text[start:end+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to repair truncated JSON arrays
        try:
            if text.strip().startswith("[") and not text.strip().endswith("]"):
                last_brace = text.rfind("}")
                if last_brace != -1:
                    repaired = text[:last_brace+1] + "\n]"
                    return json.loads(repaired)
        except Exception:
            pass
        raise e
