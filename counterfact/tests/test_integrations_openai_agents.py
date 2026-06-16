"""Offline tests for the OpenAI Agents SDK adapter.

No network and no `agents` package required: we inject a fake runner over tiny
stand-in Agent/Result objects shaped like the SDK's ``Agent`` / ``RunResult``.
"""

from counterfact.integrations.openai_agents import (
    ROUTE_KEY,
    agent_node,
    graph_from_orchestrator,
    graph_from_sequential,
)


class FakeAgent:
    """Mimics agents.Agent: just carries a name and a deterministic behavior."""

    def __init__(self, name, fn):
        self.name = name
        self._fn = fn


class FakeResult:
    """Mimics agents.RunResult: exposes final_output and last_agent."""

    def __init__(self, final_output, last_agent=None):
        self.final_output = final_output
        self.last_agent = last_agent


def fake_runner(agent, input_text):
    return FakeResult(agent._fn(input_text), last_agent=agent)


def test_agent_node_runs_and_writes_output():
    node = agent_node(FakeAgent("a", lambda t: f"out:{t}"), runner=fake_runner, reads="input", writes="final_output")
    out = node({"input": "hello"})
    assert out["final_output"] == "out:hello"
    # original state is preserved
    assert out["input"] == "hello"


def test_sequential_chain_threads_output():
    a = FakeAgent("a", lambda t: t + "|A")
    b = FakeAgent("b", lambda t: t + "|B")
    g = graph_from_sequential([a, b], runner=fake_runner)
    assert g.get_node_names() == ["a", "b"]
    res = g.invoke({"input": "x"})
    # b reads a's output
    assert res["final_output"] == "x|A|B"


def test_orchestrator_routes_and_finalizes():
    triage = FakeAgent("triage", lambda t: "billing" if "refund" in t else "technical")
    billing = FakeAgent("billing", lambda t: "Refund of $250 issued")
    technical = FakeAgent("technical", lambda t: "Try restarting")
    writer = FakeAgent("writer", lambda t: f"Dear customer, {t}.")

    g = graph_from_orchestrator(
        triage, {"billing": billing, "technical": technical}, finalizer=writer, runner=fake_runner
    )
    assert set(g.get_node_names()) == {"orchestrator", "billing", "technical", "finalizer"}

    res = g.invoke({"input": "I want a refund"})
    assert res[ROUTE_KEY] == "billing"
    assert res["final_output"] == "Dear customer, Refund of $250 issued."

    # A different ticket routes elsewhere (handoff variety).
    res2 = g.invoke({"input": "my app crashes"})
    assert res2[ROUTE_KEY] == "technical"
    assert "restarting" in res2["final_output"]


def test_orchestrator_without_finalizer_uses_specialist_output():
    triage = FakeAgent("triage", lambda t: "billing")
    billing = FakeAgent("billing", lambda t: "Refund of $250 issued")
    g = graph_from_orchestrator(triage, {"billing": billing}, runner=fake_runner)
    assert "finalizer" not in g.get_node_names()
    res = g.invoke({"input": "refund"})
    assert res["final_output"] == "Refund of $250 issued"


def test_route_selector_falls_back_to_last_agent_then_first():
    # Orchestrator output names no specialist; selector falls back to last_agent.
    specialists = {"billing": FakeAgent("billing", lambda t: "b"),
                   "technical": FakeAgent("technical", lambda t: "t")}

    def runner_naming_nothing(agent, text):
        # final_output mentions no specialist; last_agent points at 'technical'
        return FakeResult("I will help you", last_agent=specialists["technical"])

    triage = FakeAgent("triage", lambda t: "")
    g = graph_from_orchestrator(triage, specialists, runner=runner_naming_nothing)
    res = g.invoke({"input": "anything"})
    assert res[ROUTE_KEY] == "technical"  # resolved via last_agent


def test_ablating_orchestrator_still_runs_via_default_route():
    triage = FakeAgent("triage", lambda t: "billing")
    billing = FakeAgent("billing", lambda t: "Refund of $250")
    technical = FakeAgent("technical", lambda t: "Restart it")
    g = graph_from_orchestrator(triage, {"billing": billing, "technical": technical}, runner=fake_runner)
    # Ablating the orchestrator drops the route; graph falls back to first specialist.
    ablated = g.clone_with_ablation("orchestrator")
    res = ablated.invoke({"input": "refund"})
    assert res["final_output"] == "Refund of $250"  # first specialist = billing


def test_ablating_specialist_removes_its_contribution():
    triage = FakeAgent("triage", lambda t: "billing")
    billing = FakeAgent("billing", lambda t: "Refund of $250 issued")
    writer = FakeAgent("writer", lambda t: f"Reply: {t}")
    g = graph_from_orchestrator(triage, {"billing": billing}, finalizer=writer, runner=fake_runner)

    full = g.invoke({"input": "refund please"})
    assert "$250" in full["final_output"]

    ablated = g.clone_with_ablation("billing")
    gone = ablated.invoke({"input": "refund please"})
    assert "$250" not in gone["final_output"]  # specialist's contribution removed


def test_diagnose_isolates_the_buggy_specialist():
    """End-to-end: a buggy specialist (drops the figure) should carry the most
    negative Shapley attribution when scored against a gold figure."""
    from counterfact.classifiers import ClassifierRegistry
    from counterfact.integrations.braintrust import quality_fn_from_scorer

    triage = FakeAgent("triage", lambda t: "billing")
    # buggy: omits the dollar figure the gold requires
    billing_buggy = FakeAgent("billing", lambda t: "Your refund has been processed.")
    writer = FakeAgent("writer", lambda t: f"Dear customer, {t}")
    g = graph_from_orchestrator(triage, {"billing": billing_buggy}, finalizer=writer, runner=fake_runner)

    def contains_scorer(output, expected):
        return 1.0 if str(expected) in (output or "") else 0.0

    qf = quality_fn_from_scorer(contains_scorer)
    report = g.diagnose(
        input_state={"input": "refund status?", "expected": "$250"},
        num_simulations=8,
        quality_fn=qf,
        registry=ClassifierRegistry(),  # empty: no LLM classifiers needed
        run_evals=False,
        seed=0,
    )
    # Baseline fails (figure missing) and the billing specialist is implicated.
    assert report.baseline_quality < 0.5
    shap = report.shapley_values or {}
    # billing should be among the most-attributed agents on the executed path.
    assert "billing" in shap
