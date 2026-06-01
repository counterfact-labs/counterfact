"""
Module-level helpers for driving counterfact from an external orchestrator.

Currently exposes :func:`diagnose_dataset`, a thin functional wrapper around
:meth:`CounterfactualGraph.diagnose_dataset` for callers who prefer a
function over a method (e.g. an "AI pipeline copilot").

Dependencies: graph (only via TYPE_CHECKING / duck typing at runtime).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from counterfact.diagnostics import DiagnosticReport
    from counterfact.graph import CounterfactualGraph


def diagnose_dataset(
    graph: "CounterfactualGraph",
    inputs: list[dict],
    **diagnose_kwargs: Any,
) -> list["DiagnosticReport"]:
    """Run a full diagnostic for each input state in ``inputs``.

    This is the functional equivalent of
    :meth:`CounterfactualGraph.diagnose_dataset`. Every input triggers a REAL,
    independent diagnostic re-run (no shared simulations or shortcuts across
    inputs); results are returned in input order.

    Args:
        graph: A compiled ``CounterfactualGraph`` (must carry a build recipe).
        inputs: A list of ``input_state`` dicts, one per diagnostic run.
        **diagnose_kwargs: Forwarded verbatim to ``graph.diagnose`` (e.g.
            ``domain``, ``num_simulations``, ``quality_fn``, ``seed``).

    Returns:
        A list of ``DiagnosticReport`` objects, in input order.
    """
    return graph.diagnose_dataset(inputs, **diagnose_kwargs)
