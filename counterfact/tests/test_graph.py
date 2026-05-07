import asyncio
from unittest.mock import patch

import pytest

from counterfact.graph import StateGraph


def test_stategraph_build_recipe():
    # Test that adding nodes and edges builds the recipe correctly
    graph = StateGraph(dict)

    def my_fn(state):
        return {"a": 1}

    graph.add_node("agent", my_fn)
    # Test alternate add_node signature
    graph.add_node("agent2", action=my_fn)
    graph.add_node("agent3", my_fn)

    graph.add_edge("agent", "agent2")
    graph.add_conditional_edges("agent2", lambda x: "agent3", path_map={"a": "agent3"})

    graph.set_entry_point("agent")
    graph.set_finish_point("agent3")

    compiled = graph.compile()

    # Check recipe
    recipe = compiled._recipe
    assert recipe is not None
    assert "agent" in recipe.nodes
    assert "agent2" in recipe.nodes
    assert recipe.entry_point == "agent"
    assert recipe.finish_point == "agent3"
    assert len(recipe.edges) >= 1
    assert len(recipe.conditional_edges) == 1

def test_counterfactual_graph_invoke():
    graph = StateGraph(dict)
    graph.add_node("agent", lambda x: {"k": x.get("k", 0) + 1})
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()

    res = compiled.invoke({"k": 1})
    assert res == {"k": 2}

    trace = compiled.get_trace()
    assert len(trace) > 0
    assert trace[0]["node"] == "agent"

    entries = compiled.get_trace_entries()
    assert len(entries) == len(trace)

def test_counterfactual_graph_stream():
    graph = StateGraph(dict)
    graph.add_node("agent", lambda x: {"k": x.get("k", 0) + 1})
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()

    chunks = list(compiled.stream({"k": 1}))
    assert len(chunks) > 0
    assert chunks[-1] == {"agent": {"k": 2}}



def test_counterfactual_graph_ainvoke():
    graph = StateGraph(dict)
    def my_fn(x):
        return {"k": x.get("k", 0) + 1}
    graph.add_node("agent", action=my_fn)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()

    res = asyncio.run(compiled.ainvoke({"k": 1}))
    assert res == {"k": 2}

def test_counterfactual_graph_astream():
    graph = StateGraph(dict)
    def my_fn(x):
        return {"k": x.get("k", 0) + 1}
    graph.add_node("agent", action=my_fn)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()

    async def run_stream():
        chunks = []
        async for chunk in compiled.astream({"k": 1}):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run_stream())

    assert len(chunks) > 0
    assert chunks[-1] == {"agent": {"k": 2}}

def test_counterfactual_graph_get_graph():
    graph = StateGraph(dict)
    graph.add_node("agent", lambda x: x)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()
    assert compiled.get_graph() is not None
    assert compiled.get_node_names() == ["agent"]

def test_counterfactual_graph_getattr():
    graph = StateGraph(dict)
    graph.add_node("agent", lambda x: {"k": 1})
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()
    # The underlying graph has .name
    assert compiled.name is not None

    # Hit line 210: get_node_names without recipe
    compiled._recipe = None
    assert compiled.get_node_names() == []

def test_add_node_best_effort():
    # Hit lines 500-506: add_node without explicit function arg
    graph = StateGraph(dict)

    class MyRunnable:
        name = "runnable"
        def __call__(self, state):
            return state

    # Add an object that is not a function string + callable
    # Passing string + dict (len(args)==2, callable is False)
    graph.add_node("noncallable", {"k": lambda x: 1})

    graph.set_entry_point("noncallable")
    compiled = graph.compile()
    assert len(compiled.get_node_names()) == 0

def test_clone_conditional_edges():
    # Hit lines 294-299 and 523 (then)
    graph = StateGraph(dict)
    graph.add_node("agent1", lambda x: {"k": 1})
    graph.add_node("agent2", lambda x: {"k": 2})
    graph.add_conditional_edges("agent1", lambda x: "agent2", path_map={"a": "agent2"})
    graph.set_entry_point("agent1")
    graph.set_finish_point("agent2")
    compiled = graph.compile()

    # Clone to hit the conditional edge rebuilding
    cloned = compiled.clone_with_ablation("agent2")
    assert "agent2" in cloned.get_node_names()

def test_clone_with_ablation():
    graph = StateGraph(dict)
    graph.add_node("agent1", lambda x: {"output": x.get("output", "") + "A"})
    graph.add_node("agent2", lambda x: {"output": x.get("output", "") + "B"})
    graph.add_edge("agent1", "agent2")
    graph.set_entry_point("agent1")
    graph.set_finish_point("agent2")
    compiled = graph.compile()

    # baseline
    res1 = compiled.invoke({"output": ""})
    assert res1["output"] == "AB"

    # ablate agent1
    ablated = compiled.clone_with_ablation("agent1")
    res2 = ablated.invoke({"output": ""})
    assert res2["output"] == "B" # 'A' is missing

    # test errors
    with pytest.raises(ValueError):
        compiled.clone_with_ablation("agent3")

    compiled._recipe = None
    with pytest.raises(ValueError):
        compiled.clone_with_ablation("agent1")

def test_clone_with_replacement():
    graph = StateGraph(dict)
    graph.add_node("agent1", lambda x: {"output": "A"})
    graph.set_entry_point("agent1")
    graph.set_finish_point("agent1")
    compiled = graph.compile()

    replaced = compiled.clone_with_replacement("agent1", lambda x: {"output": "C"})
    res = replaced.invoke({})
    assert res["output"] == "C"

    with pytest.raises(ValueError):
        compiled.clone_with_replacement("agent3", lambda x: x)

    compiled._recipe = None
    with pytest.raises(ValueError):
        compiled.clone_with_replacement("agent1", lambda x: x)

@patch("counterfact.evals.run_eval_suite")
def test_eval(mock_eval):
    graph = StateGraph(dict)
    graph.add_node("agent", lambda x: {"output": "A"})
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()

    with pytest.raises(ValueError, match="No trace available"):
        compiled.eval()

    compiled.invoke({})
    compiled.eval(final_output="A")
    mock_eval.assert_called_once()

@patch("counterfact.diagnostics.run_full_diagnostic")
def test_diagnose(mock_diag):
    graph = StateGraph(dict)
    graph.add_node("agent", lambda x: {"output": "A"})
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    compiled = graph.compile()

    compiled.diagnose({"q": "hi"})
    mock_diag.assert_called_once()

    compiled._recipe = None
    with pytest.raises(ValueError, match="no build recipe available"):
        compiled.diagnose({"q": "hi"})
