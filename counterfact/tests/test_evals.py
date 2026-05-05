from unittest.mock import patch
from counterfact.evals import (
    check_empty_outputs,
    check_error_status,
    check_schema_violations,
    check_latency_anomalies,
    check_output_length_anomalies,
    check_faithfulness,
    check_inter_agent_coherence,
    check_grounding,
    run_eval_suite,
)
from counterfact.types import EvalResult, EvalSuite

def test_check_empty_outputs():
    trace = [
        {"node": "a", "output": {"x": 1}},
        {"node": "b", "output": {}},
        {"node": "c", "output": {"x": None, "y": ""}},
        {"node": "d", "output": None},
    ]
    results = check_empty_outputs(trace)
    assert len(results) == 4
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[2].passed is False
    assert results[3].passed is False

def test_check_error_status():
    trace = [
        {"node": "a", "status": "pass"},
        {"node": "b", "status": "error", "output": {"error": "failed"}},
    ]
    results = check_error_status(trace)
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False

def test_check_schema_violations():
    schema = {"x": "str", "y": "int"}
    trace = [
        {"node": "a", "output": {"x": "hello", "y": 1}},
        {"node": "b", "output": {"x": "hello", "z": 2}}, # missing y, extra z
        {"node": "c", "output": "not_a_dict"},
    ]
    results = check_schema_violations(trace, {"a": schema, "b": schema, "c": schema})
    assert len(results) == 3
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[2].passed is False

def test_check_latency_anomalies():
    trace = [
        {"node": "a", "duration_ms": 100},
        {"node": "b", "duration_ms": 150},
        {"node": "c", "duration_ms": 200},
        {"node": "d", "duration_ms": 1000}, # anomaly
    ]
    results = check_latency_anomalies(trace, threshold_factor=2.0)
    assert len(results) == 4
    assert results[0].passed is True
    assert results[3].passed is False

def test_check_plan_completeness_expected_tools():
    from counterfact.evals import check_plan_completeness
    trace = [{"node": "a"}, {"node": "b"}]
    
    # No expected tools, > 1 distinct
    res = check_plan_completeness(trace)
    assert res[0].passed is True
    
    # Missing tools
    res = check_plan_completeness(trace, expected_tools=["a", "c"])
    assert res[0].passed is False
    assert "c" in res[0].details["missing_tools"]

    # All expected tools
    res = check_plan_completeness(trace, expected_tools=["a", "b"])
    assert res[0].passed is True

    # No expected tools, 1 distinct
    res = check_plan_completeness([{"node": "a"}, {"node": "a"}])
    assert res[0].passed is False

def test_check_output_length_anomalies():
    trace = [
        {"node": "a", "output": "short length"},
        {"node": "b", "output": "short length"},
        {"node": "c", "output": "short length"},
        {"node": "d", "output": "very long " * 10000}, # anomaly
    ]
    results = check_output_length_anomalies(trace, max_length=1000)
    assert len(results) == 4
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is True
    assert results[3].passed is False

def test_check_duplicate_agents():
    from counterfact.evals import check_duplicate_agents
    trace = [{"node": "a"} for _ in range(6)]
    results = check_duplicate_agents(trace)
    assert len(results) == 1
    assert results[0].passed is False

def test_check_plan_completeness():
    from counterfact.evals import check_plan_completeness
    trace = [
        {"node": "a", "output": {"steps": ["step1", "step2", "step3"]}},
        {"node": "b", "output": {"step": "step1"}},
        {"node": "c", "output": {"step": "step2"}},
    ]
    results = check_plan_completeness(trace)
    assert len(results) >= 1
    
    trace_good = [
        {"node": "a", "output": {"steps": ["step1", "step2"]}},
        {"node": "b", "output": {"step": "step1"}},
        {"node": "c", "output": {"step": "step2"}},
    ]
    assert all(r.passed for r in check_plan_completeness(trace_good))

def test_check_tool_error_rate():
    from counterfact.evals import check_tool_error_rate
    from counterfact.types import ToolCall
    trace = [
        {
            "node": "a", 
            "tool_calls": [
                ToolCall("t1", {}, {}, status="error").__dict__,
                ToolCall("t2", {}, {}, status="success").__dict__,
            ]
        }
    ]
    results = check_tool_error_rate(trace, threshold=0.1)
    assert len(results) > 0

def test_check_tool_redundancy():
    from counterfact.evals import check_tool_redundancy
    from counterfact.types import ToolCall
    trace = [
        {
            "node": "a", 
            "tool_calls": [
                ToolCall("search", {"q": "test"}, {}).__dict__,
                ToolCall("search", {"q": "test"}, {}).__dict__, # redundant
            ]
        }
    ]
    results = check_tool_redundancy(trace)
    assert len(results) > 0

def test_check_faithfulness():
    def mock_llm(p, t):
        return '{"score": 0.9, "reasoning": "pass"}'
    result = check_faithfulness([{"node": "a", "output": {"text": "something very long here..."}}], "final output", mock_llm)
    assert result.passed is True

def test_check_inter_agent_coherence():
    def mock_llm(p, t):
        return '{"contradiction": true, "reasoning": "test"}'
    trace = [
        {"node": "a", "output": {"text": "something very long here..."}}, 
        {"node": "b", "output": {"text": "something else very long..."}}
    ]
    result = check_inter_agent_coherence(trace, mock_llm)
    assert result.passed is False

def test_check_grounding():
    def mock_llm(p, t):
        return '{"score": 0.9, "reasoning": "pass"}'
    result = check_grounding([{"node": "a", "output": {"sources": "something very long here..."}}], "final test", mock_llm)
    assert isinstance(result, EvalResult)

@patch("counterfact.evals.check_faithfulness")
@patch("counterfact.evals.check_inter_agent_coherence")
@patch("counterfact.evals.check_grounding")
def test_run_eval_suite_full(mock_grounding, mock_coherence, mock_faith):
    mock_faith.return_value = EvalResult("f", True, "info", "m")
    mock_coherence.return_value = EvalResult("c", True, "info", "m")
    mock_grounding.return_value = EvalResult("g", True, "info", "m")
    
    trace = [
        {"node": "a", "output": {"x": 1}, "status": "pass", "duration_ms": 100, "tokens": 10},
    ]
    suite = run_eval_suite(trace, final_output="out", tiers=[1, 2], llm_fn=lambda p, t: "")
    assert suite.passed is True
    assert len(suite.tier_2_results) > 0

def test_run_eval_suite():
    trace = [
        {"node": "a", "output": {"x": 1}, "status": "pass", "duration_ms": 100, "tokens": 10},
    ]
    suite = run_eval_suite(trace)
    assert isinstance(suite, EvalSuite)
    assert len(suite.results) > 0
    assert suite.passed is True
