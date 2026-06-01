"""
Drop-in LangGraph wrapper with real counterfactual simulation capabilities.

Usage:
    # Replace:     from langgraph.graph import StateGraph, END
    # With:        from counterfact import StateGraph, END

    graph = StateGraph(MyState)
    graph.add_node("agent_a", my_function)
    graph.add_edge("agent_a", "agent_b")
    ...
    compiled = graph.compile()

    # Standard LangGraph usage — works exactly the same:
    result = compiled.invoke(initial_state)

    # New: run ground-truth-free evals:
    eval_suite = compiled.eval(final_output=result["final_output"])

    # New: counterfactual diagnostics (actually re-runs the pipeline):
    report = compiled.diagnose(
        input_state={"query": "..."},
        domain="rag",
    )

Dependencies:
  - Always: tracing (lightweight)
  - On demand: diagnostics, perturbation, evals (lazy-loaded when called)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from langgraph.graph import END as END
from langgraph.graph import START as START
from langgraph.graph import StateGraph as _LangGraphStateGraph

from counterfact.tracing import (
    TraceEntry,
    TracingContext,
    set_active_context,
    wrap_node,
)

# ═════════════════════════════════════════════════════════════════════════
# BUILD RECIPE
# Stores the full definition of a StateGraph so it can be cloned and
# rebuilt with modifications (e.g., ablated nodes).
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class _EdgeDef:
    """A normal edge: source -> target."""

    source: str
    target: str


@dataclass
class _ConditionalEdgeDef:
    """A conditional edge: source -> routing_fn -> {condition: target}."""

    source: str
    path: Callable
    path_map: Optional[dict[str, str]]


@dataclass
class _BuildRecipe:
    """
    Complete recipe to reconstruct a StateGraph from scratch.

    Stores every add_node, add_edge, set_entry_point, and
    add_conditional_edges call so the graph can be rebuilt
    with node functions swapped out.
    """

    state_schema: Type
    state_schema_kwargs: dict = field(default_factory=dict)
    nodes: dict[str, Callable] = field(default_factory=dict)
    node_kwargs: dict[str, dict] = field(default_factory=dict)
    edges: list[_EdgeDef] = field(default_factory=list)
    conditional_edges: list[_ConditionalEdgeDef] = field(default_factory=list)
    entry_point: Optional[str] = None
    finish_point: Optional[str] = None
    compile_kwargs: dict = field(default_factory=dict)
    # ── Optional metadata for the neutral spec IR (additive) ──
    # Populated by build_graph_from_spec / preserved across clones so that
    # to_spec() can round-trip node I/O keys and the schema label.
    state_schema_name: Optional[str] = None
    node_io: dict[str, dict] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL GRAPH (compiled pipeline with diagnostics)
# ═════════════════════════════════════════════════════════════════════════


class CounterfactualGraph:
    """
    Wraps a compiled LangGraph with eval and diagnostic capabilities.

    Supports all standard compiled-graph operations (invoke, stream, etc.)
    plus new methods for evaluation and counterfactual diagnostics.
    """

    def __init__(
        self,
        compiled_graph,
        tracing_ctx: TracingContext,
        recipe: Optional[_BuildRecipe] = None,
    ):
        self._graph = compiled_graph
        self._tracing_ctx = tracing_ctx
        self._recipe = recipe
        self._last_result: Optional[dict] = None

    # ─── Standard LangGraph compiled-graph methods ───────────────────
    # These are identical to LangGraph's API. We just add trace capture.

    def invoke(
        self,
        input: dict,
        config: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Run the graph synchronously — identical to LangGraph's invoke().
        Additionally captures execution trace automatically.
        """
        self._tracing_ctx.clear()
        set_active_context(self._tracing_ctx)
        try:
            result = self._graph.invoke(input, config=config, **kwargs)
            self._last_result = result
            return result
        finally:
            set_active_context(None)

    def stream(
        self,
        input: dict,
        config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Stream the graph execution — identical to LangGraph's stream().
        Additionally captures execution trace automatically.
        """
        self._tracing_ctx.clear()
        set_active_context(self._tracing_ctx)
        try:
            for chunk in self._graph.stream(input, config=config, **kwargs):
                yield chunk
                self._last_result = chunk
        finally:
            set_active_context(None)

    async def ainvoke(
        self,
        input: dict,
        config: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """Async version of invoke()."""
        self._tracing_ctx.clear()
        set_active_context(self._tracing_ctx)
        try:
            result = await self._graph.ainvoke(input, config=config, **kwargs)
            self._last_result = result
            return result
        finally:
            set_active_context(None)

    async def astream(
        self,
        input: dict,
        config: Optional[dict] = None,
        **kwargs,
    ):
        """Async version of stream()."""
        self._tracing_ctx.clear()
        set_active_context(self._tracing_ctx)
        try:
            async for chunk in self._graph.astream(input, config=config, **kwargs):
                yield chunk
                self._last_result = chunk
        finally:
            set_active_context(None)

    def get_graph(self, **kwargs):
        """Get the underlying graph visualization object."""
        return self._graph.get_graph(**kwargs)

    # ─── Trace access ────────────────────────────────────────────────

    def get_trace(self) -> list[dict]:
        """
        Get the execution trace from the last invoke/stream call.
        Returns a list of trace entry dicts.
        """
        return self._tracing_ctx.to_dicts()

    def get_trace_entries(self) -> list[TraceEntry]:
        """Get the raw TraceEntry objects from the last run."""
        return self._tracing_ctx.get_entries()

    # ─── Pipeline cloning (for counterfactual analysis) ──────────────

    def get_node_names(self) -> list[str]:
        """Get the ordered list of node names in this pipeline."""
        if self._recipe:
            return list(self._recipe.nodes.keys())
        return []

    def clone_with_ablation(self, agent_name: str) -> "CounterfactualGraph":
        """
        Create a new compiled pipeline with one node replaced by a no-op.

        The ablated node passes its input state through unchanged,
        effectively removing that agent's contribution from the pipeline.
        This is the core mechanism for real counterfactual analysis.

        Args:
            agent_name: The node to ablate.

        Returns:
            A new CounterfactualGraph with the ablated node.

        Raises:
            ValueError: If no build recipe is available or agent not found.
        """
        if self._recipe is None:
            raise ValueError(
                "Cannot clone: no build recipe available. "
                "Use counterfact.StateGraph (not raw LangGraph) to enable cloning."
            )
        if agent_name not in self._recipe.nodes:
            raise ValueError(f"Agent '{agent_name}' not found. Available: {list(self._recipe.nodes.keys())}")

        def _noop(state: dict) -> dict:
            """Ablated node — passes state through unchanged."""
            return state

        return self._clone_with_replacement(agent_name, _noop)

    def clone_with_replacement(self, agent_name: str, fn: Callable) -> "CounterfactualGraph":
        """
        Create a new compiled pipeline with one node's function replaced.

        Args:
            agent_name: The node to replace.
            fn: The replacement function (state) -> state.

        Returns:
            A new CounterfactualGraph with the replaced node.
        """
        if self._recipe is None:
            raise ValueError("Cannot clone: no build recipe available. Use counterfact.StateGraph to enable cloning.")
        if agent_name not in self._recipe.nodes:
            raise ValueError(f"Agent '{agent_name}' not found. Available: {list(self._recipe.nodes.keys())}")
        return self._clone_with_replacement(agent_name, fn)

    def _clone_with_replacement(self, agent_name: str, fn: Callable) -> "CounterfactualGraph":
        """Internal: rebuild the graph with one node function swapped."""
        recipe = self._recipe
        assert recipe is not None

        # Build a new StateGraph from the recipe
        new_graph = _LangGraphStateGraph(recipe.state_schema, **recipe.state_schema_kwargs)

        # Add nodes — swap the target node's function
        for name, node_fn in recipe.nodes.items():
            actual_fn = fn if name == agent_name else node_fn
            new_graph.add_node(name, actual_fn, **recipe.node_kwargs.get(name, {}))

        # Add edges
        for edge in recipe.edges:
            new_graph.add_edge(edge.source, edge.target)

        # Add conditional edges
        for cond in recipe.conditional_edges:
            new_graph.add_conditional_edges(cond.source, cond.path, cond.path_map)  # type: ignore[arg-type]

        # Set entry/finish points
        if recipe.entry_point:
            new_graph.set_entry_point(recipe.entry_point)
        if recipe.finish_point:
            new_graph.set_finish_point(recipe.finish_point)

        # Compile
        compiled = new_graph.compile(**recipe.compile_kwargs)
        ctx = TracingContext()

        # Build a new recipe with the swapped function
        new_recipe = _BuildRecipe(
            state_schema=recipe.state_schema,
            state_schema_kwargs=recipe.state_schema_kwargs,
            nodes={name: (fn if name == agent_name else node_fn) for name, node_fn in recipe.nodes.items()},
            node_kwargs=recipe.node_kwargs,
            edges=recipe.edges,
            conditional_edges=recipe.conditional_edges,
            entry_point=recipe.entry_point,
            finish_point=recipe.finish_point,
            compile_kwargs=recipe.compile_kwargs,
            state_schema_name=recipe.state_schema_name,
            node_io=dict(recipe.node_io),
        )

        return CounterfactualGraph(compiled, ctx, new_recipe)

    # ─── Neutral spec import/export (for external orchestrators) ─────

    def to_spec(self) -> dict:
        """Export this pipeline as a neutral, (mostly) JSON-serializable spec.

        This is the inverse of :func:`counterfact.build_graph_from_spec`. It
        lets an external orchestrator import an existing counterfact pipeline
        into its own intermediate representation.

        The returned dict follows the ``GraphSpec`` shape (see
        ``counterfact.spec``)::

            {
                "state_schema": "<type name>",
                "entry_point": ...,
                "finish_point": ...,
                "nodes": [{"name": ..., "fn": None,
                           "input_keys": [...], "output_keys": [...]}],
                "edges": [{"from": ..., "to": ..., "conditional": bool, ...}],
            }

        Notes:
          - Node ``fn`` callables are **not** serializable, so each node's
            ``"fn"`` is set to ``None`` and the function is identified by name.
          - Conditional edges are emitted with ``"conditional": True``; their
            routing function is referenced by its ``__name__`` (under
            ``"path_name"``) and the ``path_map`` is included if available.
          - I/O keys are only populated if they were supplied when the graph
            was built via ``build_graph_from_spec`` (otherwise empty lists).

        Returns:
            A dict spec describing this pipeline's topology.

        Raises:
            ValueError: If no build recipe is available.
        """
        if self._recipe is None:
            raise ValueError(
                "Cannot export spec: no build recipe available. "
                "Use counterfact.StateGraph or build_graph_from_spec to enable export."
            )
        recipe = self._recipe

        nodes = []
        for name in recipe.nodes:
            io = recipe.node_io.get(name, {})
            nodes.append(
                {
                    "name": name,
                    "fn": None,  # callables are not serializable
                    "input_keys": list(io.get("input_keys", [])),
                    "output_keys": list(io.get("output_keys", [])),
                }
            )

        edges: list[dict] = []
        for edge in recipe.edges:
            # Skip the START/END sentinel edges that mirror entry_point/finish_point —
            # they are re-created from entry_point/finish_point on rebuild, so the
            # neutral spec should only carry the real topology.
            if edge.source == START and edge.target == recipe.entry_point:
                continue
            if edge.target == END and edge.source == recipe.finish_point:
                continue
            edges.append({"from": edge.source, "to": edge.target, "conditional": False})
        for cond in recipe.conditional_edges:
            targets = sorted(set(cond.path_map.values())) if isinstance(cond.path_map, dict) else []
            edges.append(
                {
                    "from": cond.source,
                    "to": targets[0] if targets else None,
                    "conditional": True,
                    "path_name": getattr(cond.path, "__name__", None),
                    "path_map": dict(cond.path_map) if cond.path_map else None,
                    "targets": targets,
                }
            )

        return {
            "state_schema": recipe.state_schema_name or getattr(recipe.state_schema, "__name__", "dict"),
            "entry_point": recipe.entry_point,
            "finish_point": recipe.finish_point,
            "nodes": nodes,
            "edges": edges,
        }

    # ─── Apply a recommendation (returns a NEW graph) ────────────────

    def apply_recommendation(self, rec: "Any") -> "CounterfactualGraph":
        """Return a NEW graph with a :class:`Recommendation` applied.

        The graph is rebuilt from a mutated copy of the build recipe — the
        current graph is never modified. Supported ``rec.intervention_type``
        values:

          - ``"add_agent"``: Insert a new node described by ``rec.agent_spec``
            at ``rec.placement`` (``{"after": X}`` and/or ``{"before": Y}``).
            Edges are rewired so the new node sits between them. **If no
            implementation is supplied, the new node is a passthrough stub**
            (``state -> {}``) — it is wired into the topology but performs no
            work. Provide a real function by attaching it as
            ``rec._impl_fn`` on the recommendation before calling, or by
            using :meth:`clone_with_replacement` afterwards.
          - ``"modify_agent"``: Replace ``rec.target_agent``'s function. Uses
            ``rec._impl_fn`` if present, otherwise a **passthrough stub**
            (``state -> state``, i.e. a no-op like ablation).
          - ``"remove_loop"``: Drop conditional/back edges that route into
            ``rec.target_agent`` (the loop's re-entry point).
          - ``"restructure"``: Raises :class:`NotImplementedError` — generic
            topology rewrites are too open-ended to apply automatically.

        Args:
            rec: A ``Recommendation`` (see ``counterfact.types``).

        Returns:
            A new compiled ``CounterfactualGraph``.

        Raises:
            ValueError: If no recipe is available, or the recommendation
                references unknown/missing agents.
            NotImplementedError: For ``"restructure"`` interventions.
        """
        if self._recipe is None:
            raise ValueError("Cannot apply recommendation: no build recipe available.")
        itype = rec.intervention_type
        if itype == "add_agent":
            return self._apply_add_agent(rec)
        if itype == "modify_agent":
            return self._apply_modify_agent(rec)
        if itype == "remove_loop":
            return self._apply_remove_loop(rec)
        if itype == "restructure":
            raise NotImplementedError(
                "apply_recommendation does not support 'restructure': generic "
                "topology rewrites are too open-ended to apply automatically. "
                "Rebuild the pipeline from a spec via build_graph_from_spec instead."
            )
        raise ValueError(f"Unknown intervention_type: {itype!r}")

    @staticmethod
    def _passthrough_stub(state: dict) -> dict:
        """A no-op node added by apply_recommendation when no impl is given.

        Returns the state unchanged so the inserted node is a true passthrough.
        (Returning ``{}`` would *overwrite* the state to empty under an untyped
        ``dict`` schema, wiping upstream output — that is ablation semantics, not
        insertion. A real implementation is supplied via ``rec._impl_fn`` or
        attached afterward with ``clone_with_replacement``.)
        """
        return dict(state)

    def _copy_recipe(self) -> _BuildRecipe:
        """Return a shallow-but-safe copy of this graph's recipe for mutation."""
        recipe = self._recipe
        assert recipe is not None
        return _BuildRecipe(
            state_schema=recipe.state_schema,
            state_schema_kwargs=dict(recipe.state_schema_kwargs),
            nodes=dict(recipe.nodes),
            node_kwargs={k: dict(v) for k, v in recipe.node_kwargs.items()},
            edges=list(recipe.edges),
            conditional_edges=list(recipe.conditional_edges),
            entry_point=recipe.entry_point,
            finish_point=recipe.finish_point,
            compile_kwargs=dict(recipe.compile_kwargs),
            state_schema_name=recipe.state_schema_name,
            node_io={k: dict(v) for k, v in recipe.node_io.items()},
        )

    def _apply_add_agent(self, rec: "Any") -> "CounterfactualGraph":
        spec = rec.agent_spec
        if spec is None:
            raise ValueError("add_agent recommendation requires rec.agent_spec.")
        recipe = self._copy_recipe()
        new_name = spec.name
        if new_name in recipe.nodes:
            raise ValueError(f"Node '{new_name}' already exists in the pipeline.")

        impl = getattr(rec, "_impl_fn", None)
        recipe.nodes[new_name] = impl if callable(impl) else self._passthrough_stub
        recipe.node_kwargs[new_name] = {}
        recipe.node_io[new_name] = {
            "input_keys": list(getattr(spec, "input_keys", []) or []),
            "output_keys": list(getattr(spec, "output_keys", []) or []),
        }

        placement = rec.placement or {}
        after = placement.get("after")
        before = placement.get("before")

        # Rewire simple edges to splice the new node in.
        if after is not None and before is not None:
            # Replace any direct after->before edge with after->new->before.
            recipe.edges = [e for e in recipe.edges if not (e.source == after and e.target == before)]
            recipe.edges.append(_EdgeDef(source=after, target=new_name))
            recipe.edges.append(_EdgeDef(source=new_name, target=before))
        elif after is not None:
            # Insert after `after`: new node takes over `after`'s outgoing edges.
            outgoing = [e for e in recipe.edges if e.source == after]
            recipe.edges = [e for e in recipe.edges if e.source != after]
            recipe.edges.append(_EdgeDef(source=after, target=new_name))
            for e in outgoing:
                recipe.edges.append(_EdgeDef(source=new_name, target=e.target))
            if not outgoing and recipe.finish_point == after:
                recipe.edges.append(_EdgeDef(source=after, target=new_name))
                recipe.finish_point = new_name
        elif before is not None:
            # Insert before `before`: new node takes over `before`'s incoming edges.
            incoming = [e for e in recipe.edges if e.target == before]
            recipe.edges = [e for e in recipe.edges if e.target != before]
            for e in incoming:
                recipe.edges.append(_EdgeDef(source=e.source, target=new_name))
            recipe.edges.append(_EdgeDef(source=new_name, target=before))
            if not incoming and recipe.entry_point == before:
                recipe.edges.append(_EdgeDef(source=new_name, target=before))
                recipe.entry_point = new_name
        else:
            raise ValueError("add_agent requires rec.placement with 'after' and/or 'before'.")

        return _compile_recipe(recipe)

    def _apply_modify_agent(self, rec: "Any") -> "CounterfactualGraph":
        target = rec.target_agent
        if not target:
            raise ValueError("modify_agent recommendation requires rec.target_agent.")
        if target not in (self._recipe.nodes if self._recipe else {}):
            raise ValueError(
                f"Agent '{target}' not found. Available: {list(self._recipe.nodes) if self._recipe else []}"
            )
        impl = getattr(rec, "_impl_fn", None)
        # Reuse the well-tested clone path for a single-function swap.
        return self.clone_with_replacement(target, impl if callable(impl) else (lambda state: state))

    def _apply_remove_loop(self, rec: "Any") -> "CounterfactualGraph":
        target = rec.target_agent
        if not target:
            raise ValueError("remove_loop recommendation requires rec.target_agent.")
        recipe = self._copy_recipe()

        # Drop conditional edges whose routing can land back on the target,
        # and any plain back-edge feeding into the target.
        def _routes_to_target(cond: _ConditionalEdgeDef) -> bool:
            if cond.source == target:
                return True
            if isinstance(cond.path_map, dict) and target in cond.path_map.values():
                return True
            return False

        recipe.conditional_edges = [c for c in recipe.conditional_edges if not _routes_to_target(c)]
        # Remove plain edges that point back into the target from a later node
        # (heuristic: any edge into target other than the entry wiring).
        recipe.edges = [e for e in recipe.edges if not (e.target == target and e.source != recipe.entry_point)]
        # Removing a loop can leave the loop source with no outgoing edge,
        # which would make the graph non-terminating. If a node now dangles
        # (no outgoing plain/conditional edge and isn't the finish point),
        # route it to END so the rebuilt graph still compiles and terminates.
        sources_with_out = {e.source for e in recipe.edges}
        sources_with_out |= {c.source for c in recipe.conditional_edges}
        for name in list(recipe.nodes):
            if name in sources_with_out or recipe.finish_point == name:
                continue
            recipe.edges.append(_EdgeDef(source=name, target=END))
        return _compile_recipe(recipe)

    # ─── Dataset-level diagnosis ─────────────────────────────────────

    def diagnose_dataset(self, inputs: list[dict], **diagnose_kwargs: Any) -> list["Any"]:
        """Run :meth:`diagnose` over a list of input states.

        This performs a REAL, full diagnostic re-run for every input — there
        are no shortcuts or shared simulations across inputs. It is a simple
        sequential loop; for large datasets an external orchestrator may wish
        to parallelise at its own layer.

        Args:
            inputs: A list of ``input_state`` dicts, one per diagnostic run.
            **diagnose_kwargs: Forwarded verbatim to :meth:`diagnose`
                (e.g. ``domain``, ``num_simulations``, ``quality_fn``).

        Returns:
            A list of ``DiagnosticReport`` objects, in input order.
        """
        return [self.diagnose(inp, **diagnose_kwargs) for inp in inputs]

    # ─── Eval (ground-truth-free checks) ─────────────────────────────

    def eval(
        self,
        final_output: str = "",
        llm_fn: Optional[Callable] = None,
        tiers: Optional[list[int]] = None,
        expected_keys: Optional[dict[str, list[str]]] = None,
    ):
        """
        Run ground-truth-free evaluation on the last execution.

        This runs structural health checks (Tier 1) and optionally
        internal consistency checks (Tier 2) on the trace.

        Args:
            final_output: The pipeline's final output text
            llm_fn: LLM function for Tier 2 checks (optional)
            tiers: Which tiers to run, e.g. [1], [1, 2]. Default: [1]
            expected_keys: Optional schema expectations per agent

        Returns:
            EvalSuite with all check results
        """
        # Lazy import — evals module is only loaded when needed
        from counterfact.evals import run_eval_suite

        trace = self.get_trace()
        if not trace:
            raise ValueError("No trace available. Call invoke() or stream() before eval().")

        return run_eval_suite(
            trace=trace,
            final_output=final_output,
            llm_fn=llm_fn,
            tiers=tiers,
            expected_keys=expected_keys,
        )

    # ─── Full diagnostics (real re-execution) ────────────────────────

    def diagnose(
        self,
        input_state: dict,
        domain: str = "rag",
        num_simulations: int = 30,
        quality_gate: float = 0.8,
        progress_callback: Optional[Callable] = None,
        registry=None,
        llm_fn: Optional[Callable] = None,
        run_evals: bool = True,
        seed: Optional[int] = None,
        quality_fn: Optional[Callable[[str, dict], float]] = None,
    ):
        """
        Run counterfactual diagnostics by actually re-executing the pipeline.

        This is the core diagnostic method. It:
          1. Runs the pipeline as-is (baseline)
          2. For each agent, ablates it (replaces with no-op) and re-runs
          3. Scores each run with quality classifiers
          4. Computes attribution (LOO → Shapley if inconclusive)
          5. Classifies the failure type
          6. Generates recommendations

        Unlike LLM-simulated counterfactuals, this actually re-runs your
        pipeline code with real perturbations.

        Args:
            input_state: The input dict to invoke the pipeline with
            domain: Classifier domain ("rag" or "decision")
            num_simulations: Number of Monte Carlo simulations
            quality_gate: Skip attribution if baseline quality exceeds this
            progress_callback: Optional callback(current, total, status)
            registry: Custom classifier registry (uses default if None)
            llm_fn: LLM function for classifiers (prompt, temp) -> str
            run_evals: Whether to run structural eval checks first
            seed: Random seed for reproducibility
            quality_fn: Optional custom quality scorer
                ``(output_text: str, full_state: dict) -> float`` returning a
                value in [0, 1]. When provided, it is used as the quality
                metric for every simulation INSTEAD OF the classifier
                aggregate (e.g. to drive attribution from a labeled eval set).
                Classifiers still run for per-classifier diagnostics unless
                you pass an empty registry.

        Returns:
            DiagnosticReport with attribution, classification, and recommendations

        Raises:
            ValueError: If no build recipe is available (can't clone pipeline)
        """
        if self._recipe is None:
            raise ValueError(
                "Cannot run diagnostics: no build recipe available. "
                "Use counterfact.StateGraph (not raw LangGraph) to enable diagnostics."
            )

        # Lazy import — diagnostics module is only loaded when needed
        from counterfact.diagnostics import run_full_diagnostic

        return run_full_diagnostic(
            graph=self,
            input_state=input_state,
            domain=domain,
            num_simulations=num_simulations,
            quality_gate=quality_gate,
            progress_callback=progress_callback,
            registry=registry,
            llm_fn=llm_fn,
            run_evals=run_evals,
            seed=seed,
            quality_fn=quality_fn,
        )

    # ─── Pass-through for any other compiled-graph attributes ────────

    def __getattr__(self, name: str) -> Any:
        """Forward any unrecognized attributes to the underlying compiled graph."""
        return getattr(self._graph, name)


# ═════════════════════════════════════════════════════════════════════════
# STATE GRAPH (drop-in replacement for langgraph.graph.StateGraph)
# ═════════════════════════════════════════════════════════════════════════


class StateGraph(_LangGraphStateGraph):
    """
    Drop-in replacement for langgraph.graph.StateGraph.

    All standard methods work identically. The key difference:
    compile() returns a CounterfactualGraph that supports
    get_trace(), eval(), and diagnose() in addition to the
    standard invoke()/stream().

    Stores the full build recipe (nodes, edges, entry points) so
    the pipeline can be cloned with modifications for counterfactual
    analysis.

    Usage:
        from counterfact import StateGraph, END

        graph = StateGraph(MyState)
        graph.add_node("agent", my_fn)
        graph.add_edge("agent", END)
        compiled = graph.compile()

        result = compiled.invoke(initial_state)
        report = compiled.diagnose(input_state=initial_state)
    """

    def __init__(self, state_schema: Type, **kwargs):
        super().__init__(state_schema, **kwargs)
        self._cf_recipe = _BuildRecipe(
            state_schema=state_schema,
            state_schema_kwargs=kwargs,
        )
        self._tracing_ctx = TracingContext()

    def add_node(self, *args: Any, **kwargs: Any) -> "StateGraph":
        """Add a node — wraps the function for automatic trace capture."""
        if len(args) == 2 and callable(args[1]) and isinstance(args[0], str):
            name, fn = args[0], args[1]
            # Store the original (unwrapped) function in the recipe
            self._cf_recipe.nodes[name] = fn
            self._cf_recipe.node_kwargs[name] = {}
            # Wrap for tracing in this build
            wrapped = wrap_node(name, fn)
            super().add_node(name, wrapped, **kwargs)
        elif "action" in kwargs and isinstance(args[0], str):
            name = args[0]
            fn = kwargs.pop("action")
            self._cf_recipe.nodes[name] = fn
            self._cf_recipe.node_kwargs[name] = {}
            wrapped = wrap_node(name, fn)
            super().add_node(name, action=wrapped, **kwargs)
        else:
            super().add_node(*args, **kwargs)
            if args and isinstance(args[0], str):
                # Best-effort: try to extract the function
                name = args[0]
                if len(args) > 1 and callable(args[1]):
                    self._cf_recipe.nodes[name] = args[1]  # pragma: no cover
                self._cf_recipe.node_kwargs[name] = {}
        return self

    def add_edge(self, source: str, target: str, **kwargs) -> "StateGraph":  # type: ignore[override]
        """Add an edge — records it in the build recipe."""
        super().add_edge(source, target, **kwargs)
        self._cf_recipe.edges.append(_EdgeDef(source=source, target=target))
        return self

    def add_conditional_edges(  # type: ignore[override]
        self, source: str, path: Callable, path_map: Any = None, **kwargs
    ) -> "StateGraph":  # type: ignore[override]
        """Add conditional edges — records them in the build recipe."""
        call_kwargs = kwargs.copy()
        if path_map is not None:
            call_kwargs["path_map"] = path_map
        super().add_conditional_edges(source, path, **call_kwargs)
        self._cf_recipe.conditional_edges.append(
            _ConditionalEdgeDef(
                source=source,
                path=path,
                path_map=path_map,
            )
        )
        return self

    def set_entry_point(self, key: str) -> "StateGraph":
        """Set the entry point — records it in the build recipe."""
        super().set_entry_point(key)
        self._cf_recipe.entry_point = key
        return self

    def set_finish_point(self, key: str) -> "StateGraph":
        """Set the finish point — records it in the build recipe."""
        super().set_finish_point(key)
        self._cf_recipe.finish_point = key
        return self

    def compile(  # type: ignore[override]
        self,
        checkpointer: Any = None,
        *,
        cache: Any = None,
        store: Any = None,
        interrupt_before: Any = None,
        interrupt_after: Any = None,
        debug: bool = False,
        name: Any = None,
        **kwargs: Any,
    ) -> CounterfactualGraph:
        """
        Compile the graph into an executable CounterfactualGraph.

        Returns a CounterfactualGraph that supports all standard
        operations plus eval and counterfactual diagnostics via
        real pipeline re-execution.
        """
        compile_kwargs = {
            k: v
            for k, v in {
                "checkpointer": checkpointer,
                "cache": cache,
                "store": store,
                "interrupt_before": interrupt_before,
                "interrupt_after": interrupt_after,
                "debug": debug,
                "name": name,
                **kwargs,
            }.items()
            if v is not None and v is not False
        }
        self._cf_recipe.compile_kwargs = compile_kwargs

        compiled = super().compile(
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
            **kwargs,
        )
        return CounterfactualGraph(compiled, self._tracing_ctx, self._cf_recipe)


# ═════════════════════════════════════════════════════════════════════════
# RECIPE REBUILDING
# Compile an arbitrary (possibly mutated) _BuildRecipe back into a
# CounterfactualGraph. Used by apply_recommendation for topology changes.
# ═════════════════════════════════════════════════════════════════════════


def _compile_recipe(recipe: _BuildRecipe) -> CounterfactualGraph:
    """Build and compile a CounterfactualGraph from a (possibly mutated) recipe.

    Unlike ``_clone_with_replacement`` (which only swaps a single node's
    function), this reconstructs the entire graph from a recipe whose nodes,
    edges, entry/finish points may all have changed. It routes through the
    counterfact ``StateGraph`` so the resulting graph keeps tracing and a
    fresh, accurate build recipe.
    """
    builder = StateGraph(recipe.state_schema, **recipe.state_schema_kwargs)
    for name, fn in recipe.nodes.items():
        builder.add_node(name, fn, **recipe.node_kwargs.get(name, {}))
    for edge in recipe.edges:
        # The entry/finish sentinel edges are re-created by set_entry_point/
        # set_finish_point below; skip them here so they aren't added twice
        # (duplicate START/END edges corrupt the compiled graph). Any other
        # explicit START/END edges are preserved.
        if edge.source == START and edge.target == recipe.entry_point:
            continue
        if edge.target == END and edge.source == recipe.finish_point:
            continue
        builder.add_edge(edge.source, edge.target)
    for cond in recipe.conditional_edges:
        builder.add_conditional_edges(cond.source, cond.path, cond.path_map)  # type: ignore[arg-type]
    if recipe.entry_point:
        builder.set_entry_point(recipe.entry_point)
    if recipe.finish_point:
        builder.set_finish_point(recipe.finish_point)

    # Carry forward the IR metadata onto the rebuilt recipe.
    builder._cf_recipe.state_schema_name = recipe.state_schema_name
    builder._cf_recipe.node_io = {k: dict(v) for k, v in recipe.node_io.items()}

    return builder.compile(**recipe.compile_kwargs)
