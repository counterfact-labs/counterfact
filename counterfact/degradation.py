"""Removal strategy for counterfactual attribution: ablate, or severely degrade.

When Shapley/LOO attribution "removes" a node from a coalition, the default is
**ablation** — replace it with a no-op. For many agents that is the right
question ("is this agent load-bearing?"). For some modules it is actively
misleading: ablate a retriever and the synthesizer gets no context at all, so
the pipeline structurally collapses. The retriever then trivially dominates
attribution ("it's necessary") while telling you nothing about whether its
*output quality* is what hurts answers. Parsers, rerankers, and context builders
have the same problem.

For those modules we instead apply **one severe, structure-preserving
degradation**: the node still runs and its output keeps its shape (a retriever
still returns a non-empty doc list; a parser still returns its keys), but the
content is destroyed. That simulates the node contributing nothing useful
*without* the structural failure, so its Shapley value reflects quality rather
than mere necessity.

There is no separate diagnostic method and no magnitude sweep — ``diagnose()``
applies this automatically. Which strategy a node gets is decided once, by its
inferred module type (name + output shape): retrievers / rerankers / parsers are
severely degraded; everything else is ablated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from counterfact.graph import CounterfactualGraph

# Module types whose removal should be a severe degrade, not a no-op ablation.
_DEGRADE_TYPES = {"retriever", "reranker", "parser"}

_RETRIEVER_HINTS = ("retriev", "search", "fetch", "lookup", "recall", "context", "doc", "chunk")
_RERANKER_HINTS = ("rerank", "rank", "order", "sort", "score")
_PARSER_HINTS = ("pars", "extract", "structur", "classif", "route", "triage")

# Replacement content for a severely degraded list item — non-empty (so nothing
# downstream sees an empty/missing value) but useless for answering.
_DISTRACTOR = "[content removed: low-relevance placeholder]"


def infer_module_type(node_name: str, sample_value: Any) -> str:
    """Best-effort role for a node, from its name and a sample output.

    Returns ``retriever`` / ``reranker`` / ``parser`` / ``generator``. Name hints
    win; otherwise the output shape decides (list -> retriever, dict -> parser,
    string -> generator).
    """
    name = (node_name or "").lower()
    if any(h in name for h in _RERANKER_HINTS):
        return "reranker"
    if any(h in name for h in _RETRIEVER_HINTS):
        return "retriever"
    if any(h in name for h in _PARSER_HINTS):
        return "parser"
    if isinstance(sample_value, (list, tuple)):
        return "retriever"
    if isinstance(sample_value, dict):
        return "parser"
    return "generator"


def removal_strategy(module_type: str) -> str:
    """``"degrade"`` for structural modules, ``"ablate"`` otherwise."""
    return "degrade" if module_type in _DEGRADE_TYPES else "ablate"


def _destroy(value: Any) -> Any:
    """Return a same-shaped but content-free version of ``value``.

    Structure is preserved so nothing downstream errors on a missing/empty value;
    only the useful content is removed.
    """
    if isinstance(value, list):
        return [_DISTRACTOR for _ in value] if value else value
    if isinstance(value, tuple):
        return tuple(_DISTRACTOR for _ in value) if value else value
    if isinstance(value, dict):
        return {k: "" for k in value} if value else value
    if isinstance(value, str):
        return ""
    return value


def _primary_output_key(output: dict, input_state: dict) -> Optional[str]:
    """The key a node 'produces': among keys it changed, the largest payload."""
    changed = {k: v for k, v in output.items() if input_state.get(k) != v}
    pool = changed or output

    def _size(v: Any) -> int:
        if isinstance(v, (list, tuple, dict)):
            return len(v) * 1000
        if isinstance(v, str):
            return len(v)
        return 0

    best = max(pool.items(), key=lambda kv: _size(kv[1]), default=(None, None))
    return best[0]


def severe_degraded_node(original_fn):
    """Wrap a node so it runs normally, then has its primary output destroyed.

    Used as the 'removal' for structural modules: the node's contribution is
    erased (content-free) while the pipeline stays runnable (shape preserved).
    """

    def _node(state: dict) -> dict:
        out = original_fn(state)
        if not isinstance(out, dict):
            return out
        key = _primary_output_key(out, state)
        if key is not None and key in out:
            return {**out, key: _destroy(out[key])}
        return out

    _node.__name__ = f"severely_degraded_{getattr(original_fn, '__name__', 'node')}"
    return _node


def decide_removals(graph: "CounterfactualGraph", input_state: dict) -> dict:
    """Classify each node's removal strategy: ``{node: "ablate" | "degrade"}``.

    Runs the pipeline once with per-node capture wrappers to get each node's real
    output (the execution trace summarizes list/dict payloads to strings, which
    would defeat shape-based inference), then applies the module-type heuristic.
    This is a single pipeline execution — not a per-node ablation probe.
    """
    recipe = getattr(graph, "_recipe", None)
    if recipe is None:
        return {}

    captured: dict = {}

    def _make_capture(orig, name):
        def _cap(state: dict) -> dict:
            out = orig(state)
            captured[name] = (dict(state), out if isinstance(out, dict) else {})
            return out

        _cap.__name__ = f"capture_{name}"
        return _cap

    probe = graph
    for name in graph.get_node_names():
        orig = recipe.nodes.get(name)
        if orig is not None:
            probe = probe.clone_with_replacement(name, _make_capture(orig, name))
    try:
        probe.invoke({**input_state})
    except Exception:
        # If the capture run fails, fall back to name-only inference below.
        pass

    strategies: dict = {}
    for name in graph.get_node_names():
        inp, out = captured.get(name, ({}, {}))
        key = _primary_output_key(out, inp) if out else None
        sample = out.get(key) if key else None
        strategies[name] = removal_strategy(infer_module_type(name, sample))
    return strategies
