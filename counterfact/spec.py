"""
Neutral, JSON-serializable pipeline specs for external orchestrators.

This module provides a *neutral intermediate representation* (IR) for
counterfact pipelines so that an external "AI pipeline copilot" can:

  1. Build a counterfact pipeline programmatically from a plain spec
     (``build_graph_from_spec``), and
  2. Export an existing counterfact pipeline back to that same spec
     (``CounterfactualGraph.to_spec`` — implemented in ``graph.py``).

The spec is intentionally minimal and (apart from node ``fn`` callables)
JSON-serializable. The shape is::

    {
        "state_schema": <type or "dict">,     # optional, defaults to dict
        "entry_point": "node_name",            # optional
        "finish_point": "node_name",           # optional
        "nodes": [
            {
                "name": "retriever",
                "fn": <callable (state) -> dict>,   # required to build
                "input_keys": ["query"],            # optional, metadata
                "output_keys": ["docs"],            # optional, metadata
            },
            ...
        ],
        "edges": [
            {"from": "retriever", "to": "synthesizer", "conditional": False},
            ...
        ],
    }

Notes:
  - ``state_schema`` may be an actual Python type (e.g. ``dict`` or a
    ``TypedDict``) or the literal string ``"dict"`` (which maps to the
    builtin ``dict``). Any other string is treated as ``dict`` with a
    recorded ``state_schema_name`` so a round-trip can preserve the label.
  - ``fn`` must be a callable ``(state) -> dict`` when building a graph.
    On export it is omitted (callables are not JSON-serializable).
  - Conditional edges in counterfact require a routing function and a
    path map, which are not JSON-serializable. ``build_graph_from_spec``
    therefore supports conditional edges only when an explicit ``path``
    callable is supplied on the edge dict (key ``"path"``); otherwise a
    conditional edge is built as a simple edge and flagged. On export,
    conditional edges are emitted with ``"conditional": True`` and the
    routing function is referenced by name only.

This module deliberately depends only on the standard library and
``counterfact.graph`` (lazily), keeping the IR self-contained.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict


class NodeSpec(TypedDict, total=False):
    """A single node in a neutral pipeline spec.

    Keys:
      name:        Node name (required).
      fn:          Callable ``(state) -> dict`` (required to *build*; omitted
                   on export since callables are not serializable).
      input_keys:  Optional list of state keys this node reads (metadata).
      output_keys: Optional list of state keys this node writes (metadata).
    """

    name: str
    fn: Optional[Callable]
    input_keys: list[str]
    output_keys: list[str]


class EdgeSpec(TypedDict, total=False):
    """A single edge in a neutral pipeline spec.

    Keys:
      from:        Source node name (required). Use the alias ``source`` if
                   ``from`` is awkward in your dict literal — both are read.
      to:          Target node name (required, or alias ``target``).
      conditional: Whether this is a conditional edge (default False).
      path:        Optional routing callable for conditional edges (only used
                   when building; not serializable so omitted on export).
      path_map:    Optional ``{condition: target}`` map for conditional edges.
    """

    from_: str
    to: str
    conditional: bool
    path: Optional[Callable]
    path_map: Optional[dict[str, str]]


class GraphSpec(TypedDict, total=False):
    """The full neutral pipeline spec. See module docstring for the shape."""

    state_schema: Any
    state_schema_name: str
    entry_point: Optional[str]
    finish_point: Optional[str]
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]


def _resolve_state_schema(spec: dict) -> tuple[type, str]:
    """Resolve the ``state_schema`` entry into an actual type.

    Returns (schema_type, schema_name). Accepts an actual type, the string
    ``"dict"``, or any other string (treated as ``dict`` but recording the
    label so a round-trip can preserve it).
    """
    raw = spec.get("state_schema", dict)
    if isinstance(raw, type):
        return raw, getattr(raw, "__name__", "dict")
    if isinstance(raw, str):
        if raw == "dict" or not raw:
            return dict, "dict"
        # Unknown named schema — fall back to dict but keep the label.
        return dict, raw
    if raw is None:
        return dict, "dict"
    # Anything else (e.g. a TypedDict instance) — best effort.
    return dict, "dict"


def _edge_endpoint(edge: dict, *keys: str) -> Optional[str]:
    """Read the first present key from an edge dict (supports aliases)."""
    for k in keys:
        if k in edge and edge[k] is not None:
            return edge[k]
    return None


def build_graph_from_spec(spec: dict) -> "Any":
    """Build a compiled :class:`CounterfactualGraph` from a neutral spec.

    This is the entry point an external orchestrator uses to construct a
    counterfact pipeline programmatically, without importing or subclassing
    ``StateGraph`` directly.

    The returned graph carries a full build recipe, so all counterfactual
    capabilities (``diagnose``, ``clone_with_ablation``, ``to_spec``, …)
    work normally.

    Args:
        spec: A dict following the :class:`GraphSpec` shape (see module
            docstring). Each node must provide a callable ``fn``.

    Returns:
        A compiled ``CounterfactualGraph``.

    Raises:
        ValueError: If the spec is missing required fields, references an
            unknown node in an edge, or a node lacks a callable ``fn``.
    """
    # Lazy import to avoid a circular import (graph imports nothing from here
    # at module load; this function is only called at runtime).
    from counterfact.graph import StateGraph

    nodes = spec.get("nodes") or []
    if not nodes:
        raise ValueError("spec must contain at least one node under 'nodes'.")

    schema, schema_name = _resolve_state_schema(spec)
    graph = StateGraph(schema)

    node_names: set[str] = set()
    # Stash I/O metadata on the recipe so to_spec() can round-trip it.
    io_meta: dict[str, dict[str, list[str]]] = {}

    for node in nodes:
        name = node.get("name")
        if not name:
            raise ValueError(f"Every node needs a 'name'; got: {node!r}")
        fn = node.get("fn")
        if not callable(fn):
            raise ValueError(f"Node '{name}' must provide a callable 'fn' (state)->dict to build.")
        graph.add_node(name, fn)
        node_names.add(name)
        io_meta[name] = {
            "input_keys": list(node.get("input_keys") or []),
            "output_keys": list(node.get("output_keys") or []),
        }

    for edge in spec.get("edges") or []:
        source = _edge_endpoint(edge, "from", "from_", "source")
        target = _edge_endpoint(edge, "to", "target")
        if source is None or target is None:
            raise ValueError(f"Edge needs 'from' and 'to': {edge!r}")
        # Validate endpoints (END/START are allowed as targets/sources).
        from counterfact.graph import END, START

        for endpoint in (source, target):
            if endpoint in (END, START):
                continue
            if endpoint not in node_names:
                raise ValueError(f"Edge references unknown node '{endpoint}'. Known nodes: {sorted(node_names)}")

        if edge.get("conditional"):
            path = edge.get("path")
            if callable(path):
                graph.add_conditional_edges(source, path, edge.get("path_map"))
            else:
                # No routing callable supplied — a conditional edge cannot be
                # reconstructed from data alone, so fall back to a plain edge.
                graph.add_edge(source, target)
        else:
            graph.add_edge(source, target)

    entry = spec.get("entry_point")
    if entry:
        graph.set_entry_point(entry)
    finish = spec.get("finish_point")
    if finish:
        graph.set_finish_point(finish)

    # Record metadata on the recipe for round-tripping via to_spec().
    graph._cf_recipe.state_schema_name = schema_name  # type: ignore[attr-defined]
    graph._cf_recipe.node_io = io_meta  # type: ignore[attr-defined]

    return graph.compile()
