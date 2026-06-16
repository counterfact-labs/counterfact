"""Adapter: debug an OpenAI Agents SDK system with counterfact.

The OpenAI Agents SDK (`pip install openai-agents`, ``from agents import Agent,
Runner``) builds multi-agent systems where a `Runner` loops over a model,
executing tool calls and following *handoffs* between agents until one agent
produces a final answer. That loop is **dynamic** — the model decides at runtime
which specialist handles a turn.

Counterfactual ablation needs the opposite: a graph it can re-run with one agent
removed. So this adapter expresses an Agents SDK system as a counterfact
``StateGraph`` in which **each agent is a discrete, ablatable node**. The harness
(not the model's internal handoff loop) drives the sequence, which is exactly
what makes "remove agent X and re-run" meaningful.

Three common topologies are supported:

* **Sequential** — a fixed chain of agents (``graph_from_sequential``).
* **Orchestrator + handoffs** — a routing agent that delegates to one of several
  specialists (``graph_from_orchestrator``). The orchestrator becomes a routing
  node; counterfact ablates the specialists (and the orchestrator/finalizer).
* **Agents-as-tools** — a top agent that calls sub-agents as tools. Model it with
  ``graph_from_orchestrator`` (top agent = orchestrator, tool-agents =
  specialists), or wire your own topology from the ``agent_node`` primitive.

The runner is always *injected*: pass any ``runner(agent, input_text) -> result``
callable. It defaults to ``agents.Runner.run_sync`` (lazily imported), but a
fake runner lets you unit-test and run offline case studies without the SDK or
network. The result is read via its ``final_output`` attribute (per the SDK's
``RunResult``), or you can pass an ``output_extractor``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from counterfact.graph import END, CounterfactualGraph, StateGraph

# A runner turns (agent, input_text) into a result carrying a final output.
RunnerFn = Callable[[Any, str], Any]

DEFAULT_INPUT_KEY = "input"
DEFAULT_OUTPUT_KEY = "final_output"
ROUTE_KEY = "_cf_route"

# An agent reference is either an Agent object or a (name, agent) pair.
AgentRef = Union[Any, tuple[str, Any]]


def _default_runner() -> RunnerFn:
    """Return a runner backed by ``agents.Runner.run_sync`` (lazy import)."""
    try:
        from agents import Runner
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "The OpenAI Agents SDK is not installed. Install it with "
            "`pip install 'counterfact[openai-agents]'`, or pass an explicit "
            "`runner=` callable (agent, input_text) -> result."
        ) from exc

    def _run(agent: Any, input_text: str) -> Any:
        return Runner.run_sync(agent, input_text)

    return _run


def _agent_name(agent: Any, fallback: str) -> str:
    """Best-effort display name for an Agent (the SDK exposes ``.name``)."""
    name = getattr(agent, "name", None)
    return str(name) if name else fallback


def _final_text(result: Any) -> str:
    """Extract the final output text from a RunResult (or a raw value)."""
    out = getattr(result, "final_output", result)
    return out if isinstance(out, str) else ("" if out is None else str(out))


def _resolve_refs(agents: Any) -> list[tuple[str, Any]]:
    """Normalize a list/dict of agents into ``[(name, agent), ...]``."""
    pairs: list[tuple[str, Any]] = []
    if isinstance(agents, dict):
        items: Any = agents.items()
    else:
        items = agents
    for i, ref in enumerate(items):
        if isinstance(ref, tuple) and len(ref) == 2:
            name, agent = ref
        else:
            agent = ref
            name = _agent_name(agent, f"agent_{i}")
        pairs.append((str(name), agent))
    return pairs


def agent_node(
    agent: Any,
    *,
    runner: Optional[RunnerFn] = None,
    reads: str = DEFAULT_INPUT_KEY,
    writes: str = DEFAULT_OUTPUT_KEY,
    input_builder: Optional[Callable[[dict], str]] = None,
    output_extractor: Optional[Callable[[Any], Any]] = None,
) -> Callable[[dict], dict]:
    """Wrap a single Agents SDK ``Agent`` as a counterfact node ``fn(state)->dict``.

    The node runs the agent as ONE discrete step and writes its output into
    ``state[writes]``. It deliberately does not rely on the SDK's internal
    handoff auto-loop: ablation requires the harness to drive the sequence so a
    single agent can be removed and the pipeline re-run.

    Args:
        agent: An Agents SDK ``Agent`` (or anything your ``runner`` accepts).
        runner: ``(agent, input_text) -> result``. Defaults to
            ``agents.Runner.run_sync``.
        reads: State key whose value is used as the agent's input text (ignored
            if ``input_builder`` is given).
        writes: State key the agent's output is written to.
        input_builder: Optional ``(state) -> str`` to build the agent input from
            the whole state (e.g. to template multiple fields).
        output_extractor: Optional ``(result) -> value`` to pull a custom value
            out of the run result. Defaults to the result's ``final_output``.

    Returns:
        A ``(state) -> dict`` node function suitable for ``StateGraph.add_node``.
    """
    run = runner or _default_runner()

    def _node(state: dict) -> dict:
        text = input_builder(state) if input_builder is not None else str(state.get(reads, ""))
        result = run(agent, text)
        value = output_extractor(result) if output_extractor is not None else _final_text(result)
        return {**state, writes: value}

    _node.__name__ = f"agent_{_agent_name(agent, 'step')}"
    return _node


def graph_from_sequential(
    agents: Any,
    *,
    runner: Optional[RunnerFn] = None,
    input_key: str = DEFAULT_INPUT_KEY,
    output_key: str = DEFAULT_OUTPUT_KEY,
) -> CounterfactualGraph:
    """Build a counterfact graph from a fixed chain of agents.

    The first agent reads ``input_key``; every agent writes ``output_key`` and
    the next agent reads it, so the chain threads one running answer. Each agent
    is an independent ablatable node.

    Args:
        agents: Ordered list of ``Agent`` objects or ``(name, agent)`` pairs
            (names must be unique). A dict ``{name: agent}`` also works.
        runner: Injected runner (see :func:`agent_node`).
        input_key / output_key: Entry/answer state keys.

    Returns:
        A compiled :class:`CounterfactualGraph` ready for ``.diagnose(...)``.
    """
    pairs = _resolve_refs(agents)
    if not pairs:
        raise ValueError("graph_from_sequential needs at least one agent.")

    graph = StateGraph(dict)
    for i, (name, agent) in enumerate(pairs):
        reads = input_key if i == 0 else output_key
        graph.add_node(name, agent_node(agent, runner=runner, reads=reads, writes=output_key))

    graph.set_entry_point(pairs[0][0])
    for (src, _), (dst, _) in zip(pairs, pairs[1:]):
        graph.add_edge(src, dst)
    graph.add_edge(pairs[-1][0], END)
    return graph.compile()


def _default_route_selector(specialist_names: list[str]) -> Callable[[Any, dict], str]:
    """Pick a specialist from the orchestrator's output.

    Default policy: the orchestrator's final output should *name* the chosen
    specialist (case-insensitive substring match). Falls back to the result's
    ``last_agent.name`` (set by the SDK after a handoff), then to the first
    specialist so the graph always routes somewhere.
    """

    def _select(result: Any, state: dict) -> str:
        text = _final_text(result).lower()
        for name in specialist_names:
            if name.lower() in text:
                return name
        last = getattr(getattr(result, "last_agent", None), "name", None)
        if last in specialist_names:
            return last
        return specialist_names[0]

    return _select


def graph_from_orchestrator(
    orchestrator: Any,
    specialists: dict,
    *,
    runner: Optional[RunnerFn] = None,
    finalizer: Any = None,
    input_key: str = DEFAULT_INPUT_KEY,
    output_key: str = DEFAULT_OUTPUT_KEY,
    orchestrator_name: str = "orchestrator",
    finalizer_name: str = "finalizer",
    route_selector: Optional[Callable[[Any, dict], str]] = None,
) -> CounterfactualGraph:
    """Build a counterfact graph from an orchestrator-with-handoffs system.

    Topology built::

        orchestrator --(route)--> specialist_i --> [finalizer] --> END

    The orchestrator runs as a *routing* node: it picks one specialist (via
    ``route_selector``) and records the choice in ``state``. A conditional edge
    then dispatches to that specialist. The specialist's output flows to an
    optional finalizer agent (e.g. a writer/QA agent) and then to the answer.

    Every agent — orchestrator, each specialist, and the finalizer — is an
    independent ablatable node, so ``.diagnose(...)`` attributes the failure to
    the specific agent responsible.

    Args:
        orchestrator: The routing/triage ``Agent``.
        specialists: ``{name: Agent}`` the orchestrator can hand off to.
        runner: Injected runner (see :func:`agent_node`).
        finalizer: Optional ``Agent`` that composes the final answer from the
            chosen specialist's output. If ``None``, the specialist's output is
            the final answer.
        input_key / output_key: Entry/answer state keys.
        orchestrator_name / finalizer_name: Node names for those two roles.
        route_selector: ``(result, state) -> specialist_name``. Defaults to
            naming-based selection (see :func:`_default_route_selector`).

    Returns:
        A compiled :class:`CounterfactualGraph` ready for ``.diagnose(...)``.
    """
    if not specialists:
        raise ValueError("graph_from_orchestrator needs at least one specialist.")
    names = list(specialists.keys())
    select = route_selector or _default_route_selector(names)
    run = runner  # passed through to agent_node (None -> default runner)
    orch_run = run or _default_runner()

    graph = StateGraph(dict)

    # Orchestrator node: run the routing agent and record the chosen route.
    def _orchestrate(state: dict) -> dict:
        result = orch_run(orchestrator, str(state.get(input_key, "")))
        route = select(result, state)
        if route not in specialists:
            route = names[0]
        return {**state, ROUTE_KEY: route, "_orchestrator_output": _final_text(result)}

    _orchestrate.__name__ = f"orchestrator_{_agent_name(orchestrator, orchestrator_name)}"
    graph.add_node(orchestrator_name, _orchestrate)

    # Specialist nodes — each reads the original input and writes the answer.
    for name, agent in specialists.items():
        graph.add_node(name, agent_node(agent, runner=run, reads=input_key, writes=output_key))

    graph.set_entry_point(orchestrator_name)

    # Conditional dispatch from the orchestrator to the chosen specialist.
    # When the orchestrator itself is ablated (no-op), ROUTE_KEY is absent, so
    # fall back to the first specialist to keep the graph runnable.
    def _route(state: dict) -> str:
        return state.get(ROUTE_KEY, names[0])

    _route.__name__ = "route_to_specialist"
    graph.add_conditional_edges(orchestrator_name, _route, {n: n for n in names})

    if finalizer is not None:
        graph.add_node(
            finalizer_name,
            agent_node(finalizer, runner=run, reads=output_key, writes=output_key),
        )
        for name in names:
            graph.add_edge(name, finalizer_name)
        graph.add_edge(finalizer_name, END)
    else:
        for name in names:
            graph.add_edge(name, END)

    return graph.compile()
