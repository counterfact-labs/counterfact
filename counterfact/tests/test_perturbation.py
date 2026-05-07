from unittest.mock import MagicMock, patch

from counterfact.perturbation import (
    _extract_final_output,
    _run_pipeline_safe,
    generate_perturbations,
    generate_perturbations_from_graph,
    run_coalition,
    run_monte_carlo,
)


def test_generate_perturbations_from_graph():
    mock_graph = MagicMock()
    mock_graph.get_node_names.return_value = ["agent1", "agent2"]

    perts = generate_perturbations_from_graph(mock_graph)
    assert len(perts) == 2
    assert perts[0].agent == "agent1"
    assert perts[0].strategy == "ablate"
    assert perts[1].agent == "agent2"

def test_generate_perturbations():
    trace = [
        {"node": "agent1"},
        {"node": "agent2"},
        {"node": "agent1"}, # duplicate
        {"node": "output"}, # ignored
    ]
    perts = generate_perturbations(trace)
    assert len(perts) == 2
    assert perts[0].agent == "agent1"
    assert perts[1].agent == "agent2"

def test_run_pipeline_safe():
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"a": 1}
    mock_graph.get_trace.return_value = [{"node": "agent1"}]

    res, trace = _run_pipeline_safe(mock_graph, {})
    assert res == {"a": 1}
    assert trace == [{"node": "agent1"}]

    # Error case
    mock_graph.invoke.side_effect = ValueError("crash")
    res, trace = _run_pipeline_safe(mock_graph, {})
    assert "_error" in res
    assert "crash" in res["_error"]
    assert trace == [{"node": "agent1"}]

def test_extract_final_output():
    # Error
    assert "PIPELINE ERROR" in _extract_final_output({"_error": "crash"})

    # Common keys
    assert _extract_final_output({"final_output": "hello"}) == "hello"
    assert _extract_final_output({"response": "world"}) == "world"

    # Fallback to string
    res = _extract_final_output({"a": "this is a long string that should be picked up"})
    assert res == "this is a long string that should be picked up"

    # Fallback to str(result)
    res = _extract_final_output({"a": "short", "b": 1})
    assert "short" in res
    assert "1" in res

def test_run_coalition():
    mock_graph = MagicMock()
    mock_graph.get_node_names.return_value = ["agent1", "agent2", "agent3"]

    # clone_with_ablation returns a new graph mock
    mock_cloned_graph = MagicMock()
    mock_graph.clone_with_ablation.return_value = mock_cloned_graph

    mock_cloned_graph.invoke.return_value = {"output": "result"}
    mock_cloned_graph.get_trace.return_value = [{"node": "agent2"}]

    # coalition keeps agent1 and agent2, ablates agent3
    coalition = frozenset(["agent1", "agent2"])

    out, trace = run_coalition(mock_graph, coalition, {})

    assert out == "result"
    assert trace == [{"node": "agent2"}]
    mock_graph.clone_with_ablation.assert_called_once_with("agent3")

@patch("counterfact.classifiers.get_default_registry")
def test_run_monte_carlo(mock_get_registry):
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry

    # Classifier results
    from counterfact.types import ClassifierResult
    mock_registry.run_all.return_value = [ClassifierResult("c1", 0.8, "")]
    # Aggregate quality

    mock_graph = MagicMock()
    mock_graph.get_node_names.return_value = ["agent1", "agent2"]

    mock_graph.invoke.return_value = {"output": "result"}
    mock_graph.get_trace.return_value = [
        {"node": "agent1", "status": "pass", "input": {}, "output": {}},
        {"node": "output", "status": "pass"}
    ]

    mock_cloned = MagicMock()
    mock_cloned.invoke.return_value = {"output": "result2"}
    mock_cloned.get_trace.return_value = [
        {"node": "agent1", "status": "pass", "input": {}, "output": {}},
        {"node": "output", "status": "pass"}
    ]
    mock_graph.clone_with_ablation.return_value = mock_cloned

    cb = MagicMock()

    results = run_monte_carlo(
        mock_graph,
        {"query": "q"},
        num_simulations=10, # 1 baseline (max(3, 1) = 3) -> wait 10 // 10 is 1. max(3, 1) = 3 baselines
        progress_callback=cb,
        seed=42,
    )

    # 3 baselines + 7 ablation runs (divided by 2 agents -> 3 runs each? 7 // 2 = 3 sims per agent)
    # wait: 3 baselines + 2 agents * 3 sims = 9 total runs.
    assert len(results) == 9

    baselines = [r for r in results if r.is_baseline]
    assert len(baselines) == 3

    ablations = [r for r in results if not r.is_baseline]
    assert len(ablations) == 6

    assert cb.called
