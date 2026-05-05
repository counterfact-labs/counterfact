"""
Automatic trace capture for LangGraph node executions.

Wraps node functions to transparently record inputs, outputs, timing,
and status without requiring manual instrumentation.

This module is intentionally lightweight — it depends only on types.py
and the standard library. It should always be safe to import.

Dependencies: types only
"""

import time
import threading
from typing import Any, Callable, Optional

from counterfact.types import TraceEntry


# ═════════════════════════════════════════════════════════════════════════
# TRACING CONTEXT
# Thread-safe container that collects trace entries during a pipeline run.
# ═════════════════════════════════════════════════════════════════════════


class TracingContext:
    """
    Thread-safe context that collects TraceEntry objects during a graph run.

    Usage:
        ctx = TracingContext()
        ctx.record(entry)
        entries = ctx.get_entries()
    """

    def __init__(self):
        self._entries: list[TraceEntry] = []
        self._lock = threading.Lock()

    def record(self, entry: TraceEntry) -> None:
        """Add a trace entry (thread-safe)."""
        with self._lock:
            self._entries.append(entry)

    def get_entries(self) -> list[TraceEntry]:
        """Get a copy of all recorded entries (thread-safe)."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        """Remove all recorded entries (thread-safe)."""
        with self._lock:
            self._entries.clear()

    def to_dicts(self) -> list[dict]:
        """Convert all entries to dicts (for JSON serialization)."""
        return [e.to_dict() for e in self.get_entries()]


# ─── Thread-local storage for the active tracing context ─────────────────
# This lets us automatically capture traces without passing the context
# through every function call.

_trace_local = threading.local()


def get_active_context() -> Optional[TracingContext]:
    """Get the tracing context for the current thread, if any."""
    return getattr(_trace_local, "context", None)


def set_active_context(ctx: Optional[TracingContext]) -> None:
    """Set the tracing context for the current thread."""
    _trace_local.context = ctx


# ═════════════════════════════════════════════════════════════════════════
# NODE WRAPPING
# Wrap node functions to automatically record trace data.
# ═════════════════════════════════════════════════════════════════════════


def _summarize(data: Any, max_len: int = 300) -> dict:
    """
    Create a summary dict of node input/output data for the trace.

    We don't want to store entire large objects — just enough to
    understand what happened. Strings are truncated, lists show their
    length, and everything else becomes a string.
    """
    if isinstance(data, dict):
        summary: dict[str, Any] = {}
        for k, v in data.items():
            if k == "trace":
                continue  # Skip trace accumulation keys
            if isinstance(v, str):
                summary[k] = v[:max_len] if len(v) > max_len else v
            elif isinstance(v, list):
                summary[k] = f"list[{len(v)}]"
            elif isinstance(v, (int, float, bool)):
                summary[k] = v
            else:
                summary[k] = str(v)[:max_len]
        return summary
    return {"value": str(data)[:max_len]}


def wrap_node(name: str, fn: Callable) -> Callable:
    """
    Wrap a node function to automatically record trace data.

    The wrapper intercepts calls to capture:
      - Input state summary
      - Output (return value) summary
      - Execution time
      - Any exceptions (recorded as status="error")
    """

    def wrapped(state: dict) -> dict:
        ctx = get_active_context()

        t0 = time.perf_counter()
        try:
            result = fn(state)
            duration = (time.perf_counter() - t0) * 1000

            if ctx is not None:
                entry = TraceEntry(
                    node=name,
                    input=_summarize(state),
                    output=_summarize(result) if result else {},
                    status="pass",
                    reasoning=f"Node '{name}' completed in {duration:.0f}ms.",
                    duration_ms=duration,
                )
                ctx.record(entry)

            return result

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            if ctx is not None:
                entry = TraceEntry(
                    node=name,
                    input=_summarize(state),
                    output={"error": str(exc)[:300]},
                    status="error",
                    reasoning=f"Node '{name}' failed: {str(exc)[:200]}",
                    duration_ms=duration,
                )
                ctx.record(entry)
            raise

    # Preserve the original function's metadata
    wrapped.__name__ = getattr(fn, "__name__", name)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__wrapped__ = fn  # type: ignore

    return wrapped
