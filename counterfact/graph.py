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

from typing import Any, Callable, Optional, Type
from dataclasses import dataclass, field

from langgraph.graph import StateGraph as _LangGraphStateGraph
from langgraph.graph import END as END, START as START

from counterfact.tracing import (
    TracingContext,
    TraceEntry,
    wrap_node,
    set_active_context,
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
            raise ValueError(
                f"Agent '{agent_name}' not found. "
                f"Available: {list(self._recipe.nodes.keys())}"
            )

        def _noop(state: dict) -> dict:
            """Ablated node — passes state through unchanged."""
            return state

        return self._clone_with_replacement(agent_name, _noop)

    def clone_with_replacement(
        self, agent_name: str, fn: Callable
    ) -> "CounterfactualGraph":
        """
        Create a new compiled pipeline with one node's function replaced.

        Args:
            agent_name: The node to replace.
            fn: The replacement function (state) -> state.

        Returns:
            A new CounterfactualGraph with the replaced node.
        """
        if self._recipe is None:
            raise ValueError(
                "Cannot clone: no build recipe available. "
                "Use counterfact.StateGraph to enable cloning."
            )
        if agent_name not in self._recipe.nodes:
            raise ValueError(
                f"Agent '{agent_name}' not found. "
                f"Available: {list(self._recipe.nodes.keys())}"
            )
        return self._clone_with_replacement(agent_name, fn)

    def _clone_with_replacement(
        self, agent_name: str, fn: Callable
    ) -> "CounterfactualGraph":
        """Internal: rebuild the graph with one node function swapped."""
        recipe = self._recipe
        assert recipe is not None

        # Build a new StateGraph from the recipe
        new_graph = _LangGraphStateGraph(
            recipe.state_schema, **recipe.state_schema_kwargs
        )

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
            nodes={
                name: (fn if name == agent_name else node_fn)
                for name, node_fn in recipe.nodes.items()
            },
            node_kwargs=recipe.node_kwargs,
            edges=recipe.edges,
            conditional_edges=recipe.conditional_edges,
            entry_point=recipe.entry_point,
            finish_point=recipe.finish_point,
            compile_kwargs=recipe.compile_kwargs,
        )

        return CounterfactualGraph(compiled, ctx, new_recipe)

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
            raise ValueError(
                "No trace available. Call invoke() or stream() before eval()."
            )

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
                source=source, path=path,
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
        **kwargs: Any
    ) -> CounterfactualGraph:
        """
        Compile the graph into an executable CounterfactualGraph.

        Returns a CounterfactualGraph that supports all standard
        operations plus eval and counterfactual diagnostics via
        real pipeline re-execution.
        """
        compile_kwargs = {
            k: v for k, v in {
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
            **kwargs
        )
        return CounterfactualGraph(compiled, self._tracing_ctx, self._cf_recipe)
