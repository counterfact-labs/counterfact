"""Tests for the external-orchestrator capabilities:

- build_graph_from_spec / CounterfactualGraph.to_spec  (neutral IR)
- CounterfactualGraph.apply_recommendation              (recipe mutation)
- diagnose_dataset (module fn + method)                 (dataset diagnosis)
- quality_fn threading through diagnose/run_monte_carlo (pluggable quality)
"""

from unittest.mock import MagicMock, patch

import pytest

from counterfact import (
    END,
    AgentSpec,
    Recommendation,
    build_graph_from_spec,
    diagnose_dataset,
)
from counterfact.graph import CounterfactualGraph, StateGraph
from counterfact.perturbation import run_monte_carlo

# ═════════════════════════════════════════════════════════════════════════
# 1. build_graph_from_spec
# ═════════════════════════════════════════════════════════════════════════


def _linear_spec():
    return {
        "state_schema": "dict",
        "entry_point": "a",
        "finish_point": "b",
        "nodes": [
            {
                "name": "a",
                "fn": lambda s: {"out": s.get("out", "") + "A"},
                "input_keys": ["out"],
                "output_keys": ["out"],
            },
            {
                "name": "b",
                "fn": lambda s: {"out": s.get("out", "") + "B"},
                "input_keys": ["out"],
                "output_keys": ["out"],
            },
        ],
        "edges": [{"from": "a", "to": "b", "conditional": False}],
    }


def test_build_graph_from_spec_runs():
    compiled = build_graph_from_spec(_linear_spec())
    assert isinstance(compiled, CounterfactualGraph)
    assert compiled.get_node_names() == ["a", "b"]
    res = compiled.invoke({"out": ""})
    assert res["out"] == "AB"


def test_build_graph_from_spec_carries_recipe():
    compiled = build_graph_from_spec(_linear_spec())
    # Recipe present -> cloning/diagnose are possible.
    ablated = compiled.clone_with_ablation("a")
    assert ablated.invoke({"out": ""})["out"] == "B"


def test_build_graph_from_spec_to_end_edge():
    spec = _linear_spec()
    spec["edges"].append({"from": "b", "to": END, "conditional": False})
    compiled = build_graph_from_spec(spec)
    assert compiled.invoke({"out": ""})["out"] == "AB"


def test_build_graph_from_spec_aliases():
    spec = _linear_spec()
    spec["edges"] = [{"source": "a", "target": "b"}]
    compiled = build_graph_from_spec(spec)
    assert compiled.invoke({"out": ""})["out"] == "AB"


def test_build_graph_from_spec_errors():
    with pytest.raises(ValueError, match="at least one node"):
        build_graph_from_spec({"nodes": []})

    with pytest.raises(ValueError, match="callable 'fn'"):
        build_graph_from_spec({"nodes": [{"name": "a", "fn": None}]})

    with pytest.raises(ValueError, match="needs a 'name'"):
        build_graph_from_spec({"nodes": [{"fn": lambda s: {}}]})

    with pytest.raises(ValueError, match="unknown node"):
        build_graph_from_spec(
            {
                "nodes": [{"name": "a", "fn": lambda s: {}}],
                "edges": [{"from": "a", "to": "ghost"}],
            }
        )

    with pytest.raises(ValueError, match="'from' and 'to'"):
        build_graph_from_spec(
            {
                "nodes": [{"name": "a", "fn": lambda s: {}}],
                "edges": [{"from": "a"}],
            }
        )


def test_build_graph_from_spec_conditional_with_path():
    spec = {
        "entry_point": "a",
        "finish_point": "b",
        "nodes": [
            {"name": "a", "fn": lambda s: {"out": "A"}},
            {"name": "b", "fn": lambda s: {"out": s.get("out", "") + "B"}},
        ],
        "edges": [
            {
                "from": "a",
                "to": "b",
                "conditional": True,
                "path": lambda s: "b",
                "path_map": {"b": "b"},
            }
        ],
    }
    compiled = build_graph_from_spec(spec)
    # conditional edge recorded in the recipe
    assert len(compiled._recipe.conditional_edges) == 1


def test_build_graph_from_spec_conditional_without_path_falls_back():
    spec = _linear_spec()
    spec["edges"] = [{"from": "a", "to": "b", "conditional": True}]
    compiled = build_graph_from_spec(spec)
    # Falls back to a plain edge (no routing callable available).
    assert len(compiled._recipe.conditional_edges) == 0
    assert compiled.invoke({"out": ""})["out"] == "AB"


# ═════════════════════════════════════════════════════════════════════════
# 2. to_spec  (inverse of build_graph_from_spec)
# ═════════════════════════════════════════════════════════════════════════


def test_to_spec_round_trip():
    original = build_graph_from_spec(_linear_spec())
    spec = original.to_spec()

    assert spec["state_schema"] == "dict"
    assert spec["entry_point"] == "a"
    assert spec["finish_point"] == "b"
    assert {n["name"] for n in spec["nodes"]} == {"a", "b"}
    # I/O keys round-tripped from the build.
    node_a = next(n for n in spec["nodes"] if n["name"] == "a")
    assert node_a["input_keys"] == ["out"]
    assert node_a["fn"] is None  # callables not serialized
    assert {(e["from"], e["to"]) for e in spec["edges"]} == {("a", "b")}

    # Re-supply fns and rebuild — topology is preserved.
    for node in spec["nodes"]:
        node["fn"] = lambda s, _n=node["name"]: {"out": s.get("out", "") + _n.upper()}
    rebuilt = build_graph_from_spec(spec)
    assert rebuilt.get_node_names() == ["a", "b"]


def test_to_spec_conditional_edge_exported():
    graph = StateGraph(dict)
    graph.add_node("a", lambda s: {"out": "A"})
    graph.add_node("b", lambda s: {"out": "B"})

    def router(s):
        return "b"

    graph.add_conditional_edges("a", router, path_map={"b": "b"})
    graph.set_entry_point("a")
    graph.set_finish_point("b")
    compiled = graph.compile()

    spec = compiled.to_spec()
    cond = [e for e in spec["edges"] if e["conditional"]]
    assert len(cond) == 1
    assert cond[0]["from"] == "a"
    assert cond[0]["path_name"] == "router"
    assert cond[0]["path_map"] == {"b": "b"}


def test_to_spec_requires_recipe():
    compiled = build_graph_from_spec(_linear_spec())
    compiled._recipe = None
    with pytest.raises(ValueError, match="no build recipe"):
        compiled.to_spec()


# ═════════════════════════════════════════════════════════════════════════
# 3. apply_recommendation
# ═════════════════════════════════════════════════════════════════════════


def _base_graph():
    return build_graph_from_spec(_linear_spec())


def test_apply_recommendation_add_agent_stub_between():
    g = _base_graph()
    rec = Recommendation(
        title="add validator",
        description="insert validator",
        intervention_type="add_agent",
        target_agent=None,
        estimated_failure_reduction=0.3,
        complexity="low",
        priority=1,
        agent_spec=AgentSpec(name="v", position="between", function="validate"),
        placement={"after": "a", "before": "b"},
    )
    new_g = g.apply_recommendation(rec)
    # original untouched
    assert g.get_node_names() == ["a", "b"]
    assert "v" in new_g.get_node_names()
    # stub contributes nothing -> output still "AB" (a then b run, v is no-op)
    assert new_g.invoke({"out": ""})["out"] == "AB"


def test_apply_recommendation_add_agent_with_impl():
    g = _base_graph()
    rec = Recommendation(
        "add",
        "d",
        "add_agent",
        None,
        0.3,
        "low",
        1,
        agent_spec=AgentSpec(name="v", position="between", function="f"),
        placement={"after": "a", "before": "b"},
    )
    rec._impl_fn = lambda s: {"out": s.get("out", "") + "V"}
    new_g = g.apply_recommendation(rec)
    # a -> v -> b  =>  "A" + "V" + "B"
    assert new_g.invoke({"out": ""})["out"] == "AVB"


def test_apply_recommendation_add_agent_after_only():
    g = _base_graph()
    rec = Recommendation(
        "add",
        "d",
        "add_agent",
        None,
        0.3,
        "low",
        1,
        agent_spec=AgentSpec(name="v", position="after", function="f"),
        placement={"after": "a"},
    )
    rec._impl_fn = lambda s: {"out": s.get("out", "") + "V"}
    new_g = g.apply_recommendation(rec)
    assert new_g.invoke({"out": ""})["out"] == "AVB"


def test_apply_recommendation_add_agent_errors():
    g = _base_graph()
    # missing agent_spec
    rec = Recommendation("t", "d", "add_agent", None, 0.1, "low", 1, placement={"after": "a"})
    with pytest.raises(ValueError, match="requires rec.agent_spec"):
        g.apply_recommendation(rec)
    # duplicate name
    rec2 = Recommendation(
        "t",
        "d",
        "add_agent",
        None,
        0.1,
        "low",
        1,
        agent_spec=AgentSpec(name="a", position="x", function="f"),
        placement={"after": "a"},
    )
    with pytest.raises(ValueError, match="already exists"):
        g.apply_recommendation(rec2)
    # no placement
    rec3 = Recommendation(
        "t",
        "d",
        "add_agent",
        None,
        0.1,
        "low",
        1,
        agent_spec=AgentSpec(name="v", position="x", function="f"),
        placement={},
    )
    with pytest.raises(ValueError, match="placement"):
        g.apply_recommendation(rec3)


def test_apply_recommendation_modify_agent_stub_is_passthrough():
    g = _base_graph()
    rec = Recommendation("t", "d", "modify_agent", "a", 0.2, "low", 1)
    new_g = g.apply_recommendation(rec)
    # a becomes a no-op (state -> state), so "A" not added => "B" only
    assert new_g.invoke({"out": ""})["out"] == "B"


def test_apply_recommendation_modify_agent_with_impl():
    g = _base_graph()
    rec = Recommendation("t", "d", "modify_agent", "a", 0.2, "low", 1)
    rec._impl_fn = lambda s: {"out": s.get("out", "") + "Z"}
    new_g = g.apply_recommendation(rec)
    assert new_g.invoke({"out": ""})["out"] == "ZB"


def test_apply_recommendation_modify_agent_errors():
    g = _base_graph()
    with pytest.raises(ValueError, match="requires rec.target_agent"):
        g.apply_recommendation(Recommendation("t", "d", "modify_agent", None, 0.1, "low", 1))
    with pytest.raises(ValueError, match="not found"):
        g.apply_recommendation(Recommendation("t", "d", "modify_agent", "ghost", 0.1, "low", 1))


def test_apply_recommendation_remove_loop():
    # Build a graph with a back-edge / conditional loop into "b".
    graph = StateGraph(dict)
    graph.add_node("a", lambda s: {"out": "A"})
    graph.add_node("b", lambda s: {"out": s.get("out", "") + "B"})
    graph.add_edge("a", "b")
    graph.add_conditional_edges("b", lambda s: "b", path_map={"loop": "b", "done": END})
    graph.set_entry_point("a")
    compiled = graph.compile()

    rec = Recommendation("t", "remove loop", "remove_loop", "b", 0.4, "medium", 1)
    new_g = compiled.apply_recommendation(rec)
    # The conditional edge routing back into "b" is dropped.
    assert len(new_g._recipe.conditional_edges) == 0


def test_apply_recommendation_remove_loop_requires_target():
    g = _base_graph()
    with pytest.raises(ValueError, match="requires rec.target_agent"):
        g.apply_recommendation(Recommendation("t", "d", "remove_loop", None, 0.1, "low", 1))


def test_apply_recommendation_restructure_not_implemented():
    g = _base_graph()
    rec = Recommendation("t", "d", "restructure", None, 0.1, "high", 1)
    with pytest.raises(NotImplementedError, match="restructure"):
        g.apply_recommendation(rec)


def test_apply_recommendation_unknown_type_and_no_recipe():
    g = _base_graph()
    with pytest.raises(ValueError, match="Unknown intervention_type"):
        g.apply_recommendation(Recommendation("t", "d", "frobnicate", None, 0.1, "low", 1))
    g._recipe = None
    with pytest.raises(ValueError, match="no build recipe"):
        g.apply_recommendation(Recommendation("t", "d", "modify_agent", "a", 0.1, "low", 1))


# ═════════════════════════════════════════════════════════════════════════
# 4. diagnose_dataset
# ═════════════════════════════════════════════════════════════════════════


def test_diagnose_dataset_method_runs_per_input():
    g = _base_graph()
    calls = []

    def fake_diagnose(inp, **kw):
        calls.append((inp, kw))
        return f"report-for-{inp['out']}"

    g.diagnose = fake_diagnose  # type: ignore[assignment]
    reports = g.diagnose_dataset([{"out": "1"}, {"out": "2"}], domain="rag", num_simulations=5)
    assert reports == ["report-for-1", "report-for-2"]
    assert len(calls) == 2
    assert calls[0][1]["domain"] == "rag"
    assert calls[0][1]["num_simulations"] == 5


def test_diagnose_dataset_module_fn_delegates():
    g = _base_graph()
    g.diagnose = lambda inp, **kw: inp["out"]  # type: ignore[assignment]
    assert diagnose_dataset(g, [{"out": "x"}, {"out": "y"}]) == ["x", "y"]


# ═════════════════════════════════════════════════════════════════════════
# 5. quality_fn threading
# ═════════════════════════════════════════════════════════════════════════


def test_quality_fn_overrides_classifier_aggregate():
    g = _base_graph()

    # Registry that would otherwise score 0.0; quality_fn must override it.
    fake_registry = MagicMock()
    fake_registry.run_all.return_value = []

    captured = {}

    def quality_fn(output_text, full_state):
        captured["output_text"] = output_text
        captured["full_state"] = full_state
        return 0.42

    results = run_monte_carlo(
        graph=g,
        input_state={"out": ""},
        domain="rag",
        num_simulations=4,
        registry=fake_registry,
        quality_fn=quality_fn,
    )
    assert results, "expected simulation results"
    # Every sim's quality must be the quality_fn value, not the (empty) aggregate.
    assert all(r.quality_score == 0.42 for r in results)
    # quality_fn received the output text and a state dict.
    assert isinstance(captured["full_state"], dict)


@patch("counterfact.diagnostics.run_monte_carlo")
def test_diagnose_threads_quality_fn(mock_mc):
    from counterfact.types import ClassifierResult, SimulationResult

    mock_mc.return_value = [
        SimulationResult(
            simulation_id=0,
            perturbation=None,
            quality_score=0.95,
            classifier_results=[ClassifierResult("c1", 0.95, "")],
            is_baseline=True,
        )
    ]
    g = _base_graph()
    qf = lambda out, state: 0.95  # noqa: E731
    g.diagnose({"out": ""}, quality_gate=0.8, run_evals=False, quality_fn=qf)
    # quality_fn forwarded into run_monte_carlo.
    assert mock_mc.call_args.kwargs["quality_fn"] is qf
