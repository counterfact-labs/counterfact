"""
Tool call tracing for thinking model pipelines.

Thinking models design and execute tool call sequences on the fly.
This module provides:

  1. ToolTracer — wraps tool functions to record inputs, outputs, and timing
  2. tool_calls_to_trace() — converts tool call records to standard trace
     format, so ALL existing attribution/eval machinery works on them
  3. perturb_tool_result() — simulates modified tool results for
     tool-boundary attribution

The key insight: tool calls are the observable boundaries in thinking
model pipelines — our equivalent of agent boundaries in explicit
multi-agent systems. By intercepting tool calls, we can reuse all
the existing perturbation/attribution/eval infrastructure.

Dependencies: types only
"""

import functools
import json
import time
from typing import Any, Callable, Optional

from counterfact.types import ToolCall

# ═════════════════════════════════════════════════════════════════════════
# TOOL TRACER
# Wrap tool functions to automatically record inputs, outputs, timing.
# ═════════════════════════════════════════════════════════════════════════


class ToolTracer:
    """
    Wraps tool functions to record all invocations for later analysis.

    Usage:
        tracer = ToolTracer()

        # Wrap your tools
        search = tracer.wrap(search_docs, "search_docs")
        calculate = tracer.wrap(calculator, "calculator")

        # Use them normally — they work exactly the same
        results = search(query="test")
        answer = calculate(expression="2+2")

        # Get the recorded tool calls
        calls = tracer.get_calls()
        trace = tracer.to_trace()  # Standard trace format
    """

    def __init__(self):
        self._calls: list[ToolCall] = []
        self._step_counter = 0

    def wrap(self, fn: Callable, tool_name: str) -> Callable:
        """
        Wrap a tool function to record its invocations.

        The wrapped function works exactly like the original — same inputs,
        same outputs — but records each call for analysis.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            step = self._step_counter
            self._step_counter += 1

            # Build input record
            tool_input: dict[str, Any] = {}
            if args:
                tool_input["args"] = [str(a)[:500] for a in args]
            tool_input.update({k: str(v)[:500] for k, v in kwargs.items()})

            try:
                result = fn(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Build output record
                tool_output = _normalize_output(result)

                self._calls.append(ToolCall(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output,
                    step_index=step,
                    duration_ms=duration_ms,
                    status="success",
                ))

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self._calls.append(ToolCall(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)[:500]},
                    step_index=step,
                    duration_ms=duration_ms,
                    status="error",
                    error_message=str(e)[:500],
                ))
                raise  # Re-raise so calling code still gets the error

        return wrapper

    def get_calls(self) -> list[ToolCall]:
        """Get all recorded tool calls in order."""
        return list(self._calls)

    def clear(self):
        """Clear all recorded calls (e.g., between test runs)."""
        self._calls.clear()
        self._step_counter = 0

    def to_trace(self) -> list[dict]:
        """
        Convert recorded tool calls to standard trace format.

        This is the bridge between thinking-model tool calls and the
        existing counterfact evaluation/attribution infrastructure.
        All existing checks (empty outputs, latency anomalies, etc.)
        work on this format.
        """
        return tool_calls_to_trace(self._calls)


# ═════════════════════════════════════════════════════════════════════════
# TRACE CONVERSION
# Convert tool calls to standard trace format for existing infrastructure.
# ═════════════════════════════════════════════════════════════════════════


def tool_calls_to_trace(calls: list[ToolCall]) -> list[dict]:
    """
    Convert a list of ToolCall records to the standard trace format.

    Each tool call becomes a TraceEntry-compatible dict with:
      - node: the tool name
      - input: what was passed to the tool
      - output: what the tool returned
      - status: "pass" or "error"
      - duration_ms: how long the call took

    This lets ALL existing counterfact infrastructure work on tool calls:
      - evals: check_empty_outputs, check_latency_anomalies, etc.
      - attribution: compute_loo_attribution, compute_shapley_values
      - perturbation: generate_perturbations, run_monte_carlo
    """
    trace = []
    for call in calls:
        trace.append({
            "node": call.tool_name,
            "input": call.tool_input,
            "output": call.tool_output,
            "status": "pass" if call.status == "success" else "error",
            "duration_ms": call.duration_ms,
            "reasoning": call.reasoning,
        })
    return trace


# ═════════════════════════════════════════════════════════════════════════
# TOOL RESULT PERTURBATION
# Simulate what happens if a tool returns different results.
# ═════════════════════════════════════════════════════════════════════════


def perturb_tool_result(
    tool_call: ToolCall,
    strategy: str,
    llm_fn: Optional[Callable[[str, float], str]] = None,
) -> dict:
    """
    Generate a perturbed version of a tool's output.

    Strategies:
      - "error": Simulate the tool returning an error
      - "empty": Simulate the tool returning nothing
      - "degrade": Use LLM to generate a degraded version of the output
      - "enhance": Use LLM to generate an improved version of the output

    This is the thinking-model equivalent of apply_perturbation() in
    the agent-based perturbation module. Returns the modified tool output.
    """
    if strategy == "error":
        # Simulate a tool failure
        return {"error": f"Simulated error: {tool_call.tool_name} failed"}

    elif strategy == "empty":
        # Simulate the tool returning nothing
        return {}

    elif strategy == "degrade":
        if llm_fn is None:
            # Without LLM, return a minimal degraded version
            return {"result": "partial result with missing information"}

        # Use LLM to generate a degraded tool result
        prompt = f"""Simulate a degraded version of this tool's output.

TOOL: {tool_call.tool_name}
INPUT: {json.dumps(tool_call.tool_input)[:500]}
ORIGINAL OUTPUT: {json.dumps(tool_call.tool_output)[:800]}

Generate a DEGRADED version that:
- Misses some important information from the original
- Contains some inaccuracies
- Is still somewhat plausible (not completely random)

Respond with ONLY valid JSON matching the original output structure."""

        try:
            response = llm_fn(prompt, 0.4)
            return _parse_json_safe(response)
        except Exception:
            return {"result": "degraded partial result"}

    elif strategy == "enhance":
        if llm_fn is None:
            return dict(tool_call.tool_output)  # Just return original

        prompt = f"""Simulate an enhanced version of this tool's output.

TOOL: {tool_call.tool_name}
INPUT: {json.dumps(tool_call.tool_input)[:500]}
ORIGINAL OUTPUT: {json.dumps(tool_call.tool_output)[:800]}

Generate an ENHANCED version that:
- Includes additional relevant information
- Is more precise and accurate
- Has better structure and completeness

Respond with ONLY valid JSON matching the original output structure."""

        try:
            response = llm_fn(prompt, 0.3)
            return _parse_json_safe(response)
        except Exception:
            return dict(tool_call.tool_output)

    else:
        # Unknown strategy — return original
        return dict(tool_call.tool_output) if isinstance(tool_call.tool_output, dict) else {}


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════


def _normalize_output(result) -> dict:
    """
    Convert any tool return value into a dict for consistent recording.
    """
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        return {"result": result}
    elif isinstance(result, (list, tuple)):
        return {"result": result}
    elif result is None:
        return {}
    else:
        return {"result": str(result)[:1000]}


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown wrappers."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
