"""
Ground-truth-free evaluation engine for multi-agent pipelines.

This module answers: "Is this pipeline healthy?" — without needing
any labeled data or known-correct outputs.

Two tiers of checks:

  Tier 1 — Structural Health (no LLM needed, instant, free):
    - Empty or missing outputs
    - Schema violations
    - Latency anomalies
    - Token count anomalies
    - Error/retry patterns

  Tier 2 — Internal Consistency (needs an LLM, sampled):
    - Faithfulness: does the final output follow from the evidence?
    - Inter-agent coherence: do agents contradict each other?
    - Grounding: are final claims traceable to retrieved sources?

Each function is standalone — you can call check_empty_outputs()
without importing anything else from counterfact.
"""

import json
import statistics
from typing import Callable, Optional

from counterfact.types import EvalResult, EvalSuite

# ═════════════════════════════════════════════════════════════════════════
# TIER 1: STRUCTURAL HEALTH CHECKS
# These run instantly with zero external dependencies. No LLM needed.
# ═════════════════════════════════════════════════════════════════════════


def check_empty_outputs(trace: list[dict]) -> list[EvalResult]:
    """
    Check if any agent produced an empty or missing output.

    An agent that returns nothing is almost always a bug — either the
    agent crashed silently, or it received bad input. This is the
    simplest and most reliable check we have.
    """
    results = []
    for entry in trace:
        node = entry.get("node", "unknown")
        output = entry.get("output", {})

        # Check for completely empty output
        is_empty = (
            output is None
            or output == {}
            or (isinstance(output, dict) and all(
                v is None or v == "" or v == [] or v == {}
                for v in output.values()
            ))
        )

        if is_empty:
            results.append(EvalResult(
                check_name="empty_output",
                passed=False,
                severity="critical",
                message=f"Agent '{node}' produced an empty output.",
                agent=node,
                details={"output": output},
            ))
        else:
            results.append(EvalResult(
                check_name="empty_output",
                passed=True,
                severity="info",
                message=f"Agent '{node}' produced non-empty output.",
                agent=node,
            ))

    return results


def check_error_status(trace: list[dict]) -> list[EvalResult]:
    """
    Check if any agent reported an error status.

    This catches agents that explicitly flagged themselves as failed.
    Simple but important — if an agent says it failed, believe it.
    """
    results = []
    for entry in trace:
        node = entry.get("node", "unknown")
        status = entry.get("status", "pass")

        if status == "error":
            error_detail = entry.get("output", {}).get("error", "Unknown error")
            results.append(EvalResult(
                check_name="error_status",
                passed=False,
                severity="critical",
                message=f"Agent '{node}' reported error: {str(error_detail)[:200]}",
                agent=node,
                details={"status": status, "error": str(error_detail)[:500]},
            ))
        else:
            results.append(EvalResult(
                check_name="error_status",
                passed=True,
                severity="info",
                message=f"Agent '{node}' status: {status}",
                agent=node,
            ))

    return results


def check_schema_violations(
    trace: list[dict],
    expected_keys: Optional[dict[str, list[str]]] = None,
) -> list[EvalResult]:
    """
    Check if agent outputs contain the expected keys.

    If expected_keys is provided, we check each agent's output against
    its expected schema. If not provided, we just check that outputs
    have at least one key (a very loose check, but better than nothing).

    Args:
        trace: Pipeline execution trace
        expected_keys: Optional mapping of agent_name -> list of required
                       output keys, e.g. {"retriever": ["doc_ids", "num_docs_found"]}
    """
    results = []
    for entry in trace:
        node = entry.get("node", "unknown")
        output = entry.get("output", {})

        if expected_keys and node in expected_keys:
            # Strict check: verify required keys are present
            required = expected_keys[node]
            missing = [k for k in required if k not in output]
            if missing:
                results.append(EvalResult(
                    check_name="schema_violation",
                    passed=False,
                    severity="warning",
                    message=f"Agent '{node}' missing expected output keys: {missing}",
                    agent=node,
                    details={"missing_keys": missing, "expected": required,
                             "actual_keys": list(output.keys()) if isinstance(output, dict) else []},
                ))
            else:
                results.append(EvalResult(
                    check_name="schema_violation",
                    passed=True,
                    severity="info",
                    message=f"Agent '{node}' output matches expected schema.",
                    agent=node,
                ))
        else:
            # Loose check: just make sure there's something there
            if isinstance(output, dict) and len(output) == 0:
                results.append(EvalResult(
                    check_name="schema_violation",
                    passed=False,
                    severity="warning",
                    message=f"Agent '{node}' has no output keys.",
                    agent=node,
                ))
            else:
                results.append(EvalResult(
                    check_name="schema_violation",
                    passed=True,
                    severity="info",
                    message=f"Agent '{node}' output has keys.",
                    agent=node,
                ))

    return results


def check_latency_anomalies(
    trace: list[dict],
    threshold_factor: float = 5.0,
    absolute_threshold_ms: float = 30000,
) -> list[EvalResult]:
    """
    Flag agents whose latency is abnormally high compared to peers.

    An agent that takes 10x longer than usual might be stuck in a loop,
    processing malformed input, or hallucinating extensively. We detect
    this by comparing each agent's latency to the median and flagging
    anything above threshold_factor * median.

    Also flags any agent exceeding the absolute threshold (default 30s).
    """
    results = []

    # Collect all durations
    durations = {}
    for entry in trace:
        node = entry.get("node", "unknown")
        duration = entry.get("duration_ms", 0)
        if duration > 0:
            durations[node] = duration

    if not durations:
        # No timing data available — can't check
        return [EvalResult(
            check_name="latency_anomaly",
            passed=True,
            severity="info",
            message="No latency data available in trace.",
        )]

    # Calculate median for relative comparison
    all_times = list(durations.values())
    median_time = statistics.median(all_times) if all_times else 0

    for node, duration in durations.items():
        # Check absolute threshold
        if duration > absolute_threshold_ms:
            results.append(EvalResult(
                check_name="latency_anomaly",
                passed=False,
                severity="warning",
                message=f"Agent '{node}' took {duration:.0f}ms (exceeds {absolute_threshold_ms:.0f}ms threshold).",
                agent=node,
                details={"duration_ms": duration, "threshold_ms": absolute_threshold_ms},
            ))
        # Check relative threshold (only meaningful with 2+ agents)
        elif len(all_times) >= 2 and median_time > 0 and duration > threshold_factor * median_time:
            results.append(EvalResult(
                check_name="latency_anomaly",
                passed=False,
                severity="warning",
                message=(
                    f"Agent '{node}' took {duration:.0f}ms "
                    f"({duration/median_time:.1f}x the median of {median_time:.0f}ms)."
                ),
                agent=node,
                details={"duration_ms": duration, "median_ms": median_time,
                         "ratio": round(duration / median_time, 2)},
            ))
        else:
            results.append(EvalResult(
                check_name="latency_anomaly",
                passed=True,
                severity="info",
                message=f"Agent '{node}' latency normal ({duration:.0f}ms).",
                agent=node,
            ))

    return results


def check_output_length_anomalies(
    trace: list[dict],
    min_length: int = 5,
    max_length: int = 50000,
) -> list[EvalResult]:
    """
    Flag agents whose outputs are suspiciously short or long.

    Very short outputs suggest the agent didn't do real work.
    Very long outputs suggest the agent hallucinated or looped.
    """
    results = []
    for entry in trace:
        node = entry.get("node", "unknown")
        output = entry.get("output", {})

        # Estimate output length from text-like fields
        output_text = ""
        if isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, str):
                    output_text += val
        elif isinstance(output, str):
            output_text = output

        length = len(output_text)

        if length > 0 and length < min_length:
            results.append(EvalResult(
                check_name="output_length_anomaly",
                passed=False,
                severity="warning",
                message=f"Agent '{node}' output is suspiciously short ({length} chars).",
                agent=node,
                details={"length": length, "min_expected": min_length},
            ))
        elif length > max_length:
            results.append(EvalResult(
                check_name="output_length_anomaly",
                passed=False,
                severity="warning",
                message=f"Agent '{node}' output is suspiciously long ({length} chars).",
                agent=node,
                details={"length": length, "max_expected": max_length},
            ))
        else:
            results.append(EvalResult(
                check_name="output_length_anomaly",
                passed=True,
                severity="info",
                message=f"Agent '{node}' output length normal ({length} chars).",
                agent=node,
            ))

    return results


def check_duplicate_agents(trace: list[dict]) -> list[EvalResult]:
    """
    Check if any agent appears more times than expected (possible loop).

    A retriever running 5 times is probably a bug or infinite loop.
    A synthesizer running 2-3 times might be normal (revision loop),
    but 10 times is suspicious.
    """
    # Count how many times each agent appears
    counts: dict[str, int] = {}
    for entry in trace:
        node = entry.get("node", "unknown")
        counts[node] = counts.get(node, 0) + 1

    results = []
    for node, count in counts.items():
        if count > 5:
            results.append(EvalResult(
                check_name="duplicate_agent",
                passed=False,
                severity="warning",
                message=f"Agent '{node}' ran {count} times — possible infinite loop.",
                agent=node,
                details={"run_count": count},
            ))
        elif count > 2:
            results.append(EvalResult(
                check_name="duplicate_agent",
                passed=True,
                severity="info",
                message=f"Agent '{node}' ran {count} times (revision loop?).",
                agent=node,
                details={"run_count": count},
            ))
        else:
            results.append(EvalResult(
                check_name="duplicate_agent",
                passed=True,
                severity="info",
                message=f"Agent '{node}' ran {count} time(s).",
                agent=node,
                details={"run_count": count},
            ))

    return results


# ─── Tier 1: Thinking Model Checks ─────────────────────────────────────
# These checks are specific to thinking-model tool-call traces.
# They work on traces produced by tool_tracing.tool_calls_to_trace().


def check_plan_completeness(
    trace: list[dict],
    expected_tools: Optional[list[str]] = None,
) -> list[EvalResult]:
    """
    Check whether a thinking model's execution covered all expected tools.

    If expected_tools is provided, we check that each tool was actually
    called at least once. This catches cases where the model skipped a
    required step in its execution plan.

    If expected_tools is not provided, we only check that the trace
    has at least 2 distinct tool calls (a single call is suspicious).
    """
    results = []

    # Get the set of distinct tools actually called
    called_tools = set(entry.get("node", "") for entry in trace)

    if expected_tools:
        # Check each expected tool was called
        missing = [t for t in expected_tools if t not in called_tools]
        if missing:
            results.append(EvalResult(
                check_name="plan_completeness",
                passed=False,
                severity="warning",
                message=f"Expected tools not called: {missing}",
                details={"missing_tools": missing, "called_tools": list(called_tools)},
            ))
        else:
            results.append(EvalResult(
                check_name="plan_completeness",
                passed=True,
                severity="info",
                message=f"All {len(expected_tools)} expected tools were called.",
                details={"expected": expected_tools, "called": list(called_tools)},
            ))
    else:
        # Basic check: at least 2 distinct tools
        if len(called_tools) < 2:
            results.append(EvalResult(
                check_name="plan_completeness",
                passed=False,
                severity="warning",
                message=f"Only {len(called_tools)} distinct tool(s) called — plan may be incomplete.",
                details={"called_tools": list(called_tools)},
            ))
        else:
            results.append(EvalResult(
                check_name="plan_completeness",
                passed=True,
                severity="info",
                message=f"{len(called_tools)} distinct tools called.",
                details={"called_tools": list(called_tools)},
            ))

    return results


def check_tool_error_rate(
    trace: list[dict],
    threshold: float = 0.5,
) -> list[EvalResult]:
    """
    Check if an abnormal number of tool calls are failing.

    A high tool error rate suggests the model is calling tools with
    bad arguments, or the tools are unreliable. Either way, the
    pipeline output is likely degraded.

    Args:
        trace: Pipeline trace (from tool_calls_to_trace)
        threshold: Error rate above which to flag (0.0–1.0, default 0.5)
    """
    results = []

    if not trace:
        return [EvalResult(
            check_name="tool_error_rate",
            passed=True,
            severity="info",
            message="No tool calls in trace.",
        )]

    total = len(trace)
    errors = sum(1 for entry in trace if entry.get("status") == "error")
    rate = errors / total

    if rate > threshold:
        results.append(EvalResult(
            check_name="tool_error_rate",
            passed=False,
            severity="critical" if rate > 0.8 else "warning",
            message=f"{errors}/{total} tool calls failed ({rate:.0%} error rate).",
            details={"error_count": errors, "total": total, "error_rate": round(rate, 3)},
        ))
    else:
        results.append(EvalResult(
            check_name="tool_error_rate",
            passed=True,
            severity="info",
            message=f"Tool error rate: {rate:.0%} ({errors}/{total}).",
            details={"error_count": errors, "total": total, "error_rate": round(rate, 3)},
        ))

    return results


def check_tool_redundancy(trace: list[dict]) -> list[EvalResult]:
    """
    Check if the same tools are called repeatedly with identical inputs.

    Redundant tool calls waste tokens and latency without adding value.
    This catches cases where the model re-does work it already completed.
    """
    results = []

    if not trace:
        return [EvalResult(
            check_name="tool_redundancy",
            passed=True,
            severity="info",
            message="No tool calls in trace.",
        )]

    # Build fingerprints for each call: (tool_name, sorted_input_keys)
    seen: dict[str, int] = {}
    duplicates = []
    for entry in trace:
        node = entry.get("node", "")
        inp = entry.get("input", {})
        # Create a fingerprint from tool name + input values
        fingerprint = f"{node}:{json.dumps(inp, sort_keys=True)}"

        if fingerprint in seen:
            duplicates.append(node)
        seen[fingerprint] = seen.get(fingerprint, 0) + 1

    redundant_tools = {k: v for k, v in seen.items() if v > 1}

    if redundant_tools:
        # Extract just tool names from fingerprints
        tool_names = [fp.split(":")[0] for fp in redundant_tools.keys()]
        results.append(EvalResult(
            check_name="tool_redundancy",
            passed=False,
            severity="warning",
            message=f"Redundant tool calls detected: {tool_names}",
            details={
                "redundant_tools": tool_names,
                "duplicate_count": len(duplicates),
            },
        ))
    else:
        results.append(EvalResult(
            check_name="tool_redundancy",
            passed=True,
            severity="info",
            message="No redundant tool calls detected.",
        ))

    return results


# ═════════════════════════════════════════════════════════════════════════
# TIER 2: INTERNAL CONSISTENCY CHECKS
# These use an LLM to check whether the pipeline's outputs are
# internally consistent — no ground truth needed.
# ═════════════════════════════════════════════════════════════════════════


def check_faithfulness(
    trace: list[dict],
    final_output: str,
    llm_fn: Callable[[str, float], str],
) -> EvalResult:
    """
    Check if the final output is faithful to the evidence in the trace.

    This asks: "Does the final output actually follow from what the
    previous agents produced?" If the retriever found docs about Topic A
    but the final output discusses Topic B, that's a faithfulness violation.

    No ground truth needed — we're checking internal consistency only.

    # PROMPT TUNING NOTE: This prompt may need tuning per domain.
    # Key levers:
    #   - How strictly to define "faithful" (any unsupported claim vs only fabricated ones)
    #   - Whether to consider implicit reasoning as faithful
    #   - Domain-specific language (financial analysis vs general QA)
    """
    # Gather all intermediate outputs from the trace to use as "evidence"
    evidence_parts = []
    for entry in trace:
        node = entry.get("node", "unknown")
        output = entry.get("output", {})
        if isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, str) and len(val) > 20:
                    evidence_parts.append(f"[{node}] {val[:500]}")

    evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no intermediate evidence found)"

    # PROMPT TUNING: This is the core faithfulness prompt.
    # Adjust scoring criteria based on how strict you want to be.
    prompt = f"""You are checking whether a pipeline's final output is faithful to its own intermediate evidence.

INTERMEDIATE EVIDENCE (from pipeline agents):
{evidence[:4000]}

FINAL OUTPUT:
{final_output[:2000]}

Score how faithful the final output is to the intermediate evidence:
- 1.0: All claims in the final output are supported by the intermediate evidence
- 0.7-0.9: Mostly faithful with minor unsupported details
- 0.3-0.6: Notable claims that aren't supported by any evidence
- 0.0: The output contradicts or fabricates information not in the evidence

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_llm_json(response)
        score = parsed.get("score", 0.5)
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        score = 0.5
        reasoning = f"Could not evaluate faithfulness: {str(e)[:200]}"

    return EvalResult(
        check_name="faithfulness",
        passed=score >= 0.7,
        severity="critical" if score < 0.5 else "warning" if score < 0.7 else "info",
        message=f"Faithfulness score: {score:.2f}. {reasoning}",
        details={"score": score, "reasoning": reasoning},
    )


def check_inter_agent_coherence(
    trace: list[dict],
    llm_fn: Callable[[str, float], str],
) -> EvalResult:
    """
    Check if different agents in the trace contradict each other.

    If Agent A says "the company grew 20%" and Agent B says "the company
    shrank 5%", that's an inter-agent contradiction. This check catches
    these inconsistencies.

    # PROMPT TUNING NOTE: Adjust for:
    #   - How to handle legitimate disagreements (critic rejecting synthesis)
    #   - Whether to check ALL agent pairs or just sequential ones
    #   - Sensitivity to numerical vs qualitative contradictions
    """
    # Collect outputs from each agent
    agent_outputs = {}
    for entry in trace:
        node = entry.get("node", "unknown")
        output = entry.get("output", {})
        if isinstance(output, dict):
            text_parts = [
                str(v)[:300] for v in output.values()
                if isinstance(v, str) and len(v) > 10
            ]
            if text_parts:
                agent_outputs[node] = " | ".join(text_parts)

    if len(agent_outputs) < 2:
        return EvalResult(
            check_name="inter_agent_coherence",
            passed=True,
            severity="info",
            message="Not enough agent outputs to check coherence.",
        )

    # Format agent outputs for the prompt
    formatted = "\n\n".join(
        f"[{name}]: {text[:400]}" for name, text in agent_outputs.items()
    )

    # PROMPT TUNING: This prompt needs to handle legitimate disagreements
    # (e.g., a critic properly rejecting a synthesis) vs actual contradictions.
    prompt = f"""You are checking whether different agents in a pipeline contradict each other.

AGENT OUTPUTS:
{formatted[:4000]}

Check for DIRECT contradictions between agents (e.g., Agent A says X, Agent B says not-X).
Note: A "critic" agent disagreeing with a "synthesizer" is NORMAL (that's its job).
Only flag genuine factual contradictions.

Score:
- 1.0: No contradictions found
- 0.5-0.9: Minor inconsistencies
- 0.0-0.4: Direct factual contradictions between agents

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_llm_json(response)
        score = parsed.get("score", 0.5)
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        score = 0.5
        reasoning = f"Could not evaluate coherence: {str(e)[:200]}"

    return EvalResult(
        check_name="inter_agent_coherence",
        passed=score >= 0.7,
        severity="critical" if score < 0.5 else "warning" if score < 0.7 else "info",
        message=f"Inter-agent coherence score: {score:.2f}. {reasoning}",
        details={"score": score, "reasoning": reasoning},
    )


def check_grounding(
    trace: list[dict],
    final_output: str,
    llm_fn: Callable[[str, float], str],
) -> EvalResult:
    """
    Check if claims in the final output are traceable to retrieved sources.

    This is different from faithfulness — faithfulness checks against ALL
    intermediate outputs, while grounding specifically checks against
    source/evidence documents. This is most useful for RAG pipelines.

    # PROMPT TUNING NOTE: Adjust for:
    #   - What counts as a "source" (retriever docs only vs any agent output)
    #   - Whether to require explicit citations or just content overlap
    #   - Domain-specific source formats
    """
    # Find retrieval/source-like output in the trace
    sources: list[str] = []
    for entry in trace:
        output = entry.get("output", {})
        if isinstance(output, dict):
            # Look for common retrieval output patterns
            for key in ["doc_snippets", "doc_ids", "sources", "documents",
                        "retrieved_docs", "policy_text", "analysis_preview"]:
                if key in output:
                    val = output[key]
                    if isinstance(val, list):
                        sources.extend(str(v)[:500] for v in val)
                    elif isinstance(val, str):
                        sources.append(val[:500])

    if not sources:
        return EvalResult(
            check_name="grounding",
            passed=True,
            severity="info",
            message="No source documents found in trace — skipping grounding check.",
        )

    source_text = "\n---\n".join(sources)

    # PROMPT TUNING: Adjust how strict the grounding check is.
    # Some domains need strict citation, others allow implicit grounding.
    prompt = f"""You are checking whether a pipeline's output is grounded in its source documents.

SOURCE DOCUMENTS (from retrieval):
{source_text[:4000]}

FINAL OUTPUT:
{final_output[:2000]}

Score how well-grounded the output is:
- 1.0: Every claim in the output can be traced to the source documents
- 0.7-0.9: Most claims grounded, with minor additions that are reasonable inferences
- 0.3-0.6: Significant claims not found in any source
- 0.0: The output appears fabricated with no connection to sources

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_llm_json(response)
        score = parsed.get("score", 0.5)
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        score = 0.5
        reasoning = f"Could not evaluate grounding: {str(e)[:200]}"

    return EvalResult(
        check_name="grounding",
        passed=score >= 0.7,
        severity="critical" if score < 0.5 else "warning" if score < 0.7 else "info",
        message=f"Grounding score: {score:.2f}. {reasoning}",
        details={"score": score, "reasoning": reasoning},
    )


# ═════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# Convenience functions to run groups of checks together.
# ═════════════════════════════════════════════════════════════════════════


def run_structural_checks(
    trace: list[dict],
    expected_keys: Optional[dict[str, list[str]]] = None,
) -> list[EvalResult]:
    """
    Run all Tier 1 (structural) checks on a trace.

    These are instant, free, and require no LLM. Run them on every trace.
    Returns a flat list of all check results.
    """
    results = []
    results.extend(check_empty_outputs(trace))
    results.extend(check_error_status(trace))
    results.extend(check_schema_violations(trace, expected_keys))
    results.extend(check_latency_anomalies(trace))
    results.extend(check_output_length_anomalies(trace))
    results.extend(check_duplicate_agents(trace))
    return results


def run_consistency_checks(
    trace: list[dict],
    final_output: str,
    llm_fn: Callable[[str, float], str],
) -> list[EvalResult]:
    """
    Run all Tier 2 (internal consistency) checks on a trace.

    These require an LLM and cost money, so use them on sampled traces
    or when Tier 1 flags something suspicious.
    """
    results = []
    results.append(check_faithfulness(trace, final_output, llm_fn))
    results.append(check_inter_agent_coherence(trace, llm_fn))
    results.append(check_grounding(trace, final_output, llm_fn))
    return results


def run_eval_suite(
    trace: list[dict],
    final_output: str = "",
    llm_fn: Optional[Callable[[str, float], str]] = None,
    tiers: Optional[list[int]] = None,
    expected_keys: Optional[dict[str, list[str]]] = None,
) -> EvalSuite:
    """
    Run the full evaluation suite on a trace.

    Args:
        trace: Pipeline execution trace
        final_output: The pipeline's final output text
        llm_fn: LLM function for Tier 2 checks (optional — skip Tier 2 if None)
        tiers: Which tiers to run, e.g. [1], [1, 2], or [2]. Default: [1, 2]
        expected_keys: Optional schema expectations per agent

    Returns:
        EvalSuite with all results grouped by tier
    """
    if tiers is None:
        tiers = [1, 2] if llm_fn else [1]

    suite = EvalSuite()

    # Tier 1: structural checks (always fast and free)
    if 1 in tiers:
        tier_1 = run_structural_checks(trace, expected_keys)
        suite.tier_1_results = tier_1
        suite.results.extend(tier_1)

    # Tier 2: consistency checks (requires LLM)
    if 2 in tiers:
        if llm_fn is None:
            raise ValueError(
                "Tier 2 (consistency) checks require an llm_fn. "
                "Pass llm_fn or use tiers=[1] for structural-only."
            )
        if not final_output:
            raise ValueError(
                "Tier 2 checks require final_output text to evaluate."
            )
        tier_2 = run_consistency_checks(trace, final_output, llm_fn)
        suite.tier_2_results = tier_2
        suite.results.extend(tier_2)

    return suite


# ─── Helpers ─────────────────────────────────────────────────────────────


def _parse_llm_json(text: str) -> dict:
    """
    Parse a JSON response from an LLM, handling common formatting issues.

    LLMs often wrap JSON in markdown code blocks or add extra text.
    This function handles those cases gracefully.
    """
    import json
    import re

    text = text.strip()

    # Strip markdown code block wrapper
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
        # Clamp score to valid range
        if "score" in result:
            result["score"] = max(0.0, min(1.0, float(result["score"])))
        return result
    except (json.JSONDecodeError, ValueError):
        # Try to extract a score from freeform text
        nums = re.findall(r"0\.\d+|1\.0|0|1", text)
        if nums:
            return {"score": float(nums[0]), "reasoning": text[:200]}
        return {"score": 0.5, "reasoning": f"Could not parse: {text[:200]}"}
