import pytest
from counterfact.tool_tracing import (
    ToolTracer,
    perturb_tool_result,
    _normalize_output,
    _parse_json_safe
)
from counterfact.types import ToolCall

def test_tool_tracer_success():
    tracer = ToolTracer()
    
    def my_tool(x):
        return {"val": x * 2}
        
    wrapped = tracer.wrap(my_tool, "my_tool")
    res = wrapped(x=5)
    
    assert res == {"val": 10}
    calls = tracer.get_calls()
    assert len(calls) == 1
    assert calls[0].tool_name == "my_tool"
    assert calls[0].tool_input == {"x": "5"}
    assert calls[0].tool_output == {"val": 10}
    assert calls[0].status == "success"
    
def test_tool_tracer_error():
    tracer = ToolTracer()
    
    def bad_tool():
        raise ValueError("Oops")
        
    wrapped = tracer.wrap(bad_tool, "bad_tool")
    
    with pytest.raises(ValueError):
        wrapped()
        
    calls = tracer.get_calls()
    assert len(calls) == 1
    assert calls[0].status == "error"
    assert "Oops" in calls[0].error_message
    
def test_tracer_clear():
    tracer = ToolTracer()
    def t(): return 1
    wrapped = tracer.wrap(t, "t")
    wrapped()
    assert len(tracer.get_calls()) == 1
    tracer.clear()
    assert len(tracer.get_calls()) == 0

def test_tool_calls_to_trace():
    tracer = ToolTracer()
    def t(): return "hi"
    wrapped = tracer.wrap(t, "t")
    wrapped()
    trace = tracer.to_trace()
    assert len(trace) == 1
    assert trace[0]["node"] == "t"
    assert trace[0]["output"] == {"result": "hi"}

def test_perturb_tool_result():
    call = ToolCall(tool_name="t", tool_input={}, tool_output={"x": 1}, step_index=0, duration_ms=10, status="success")
    
    assert "error" in perturb_tool_result(call, "error")
    assert perturb_tool_result(call, "empty") == {}
    assert "partial" in perturb_tool_result(call, "degrade")["result"]
    assert perturb_tool_result(call, "enhance") == {"x": 1}
    assert perturb_tool_result(call, "unknown") == {"x": 1}

def test_perturb_tool_result_llm():
    call = ToolCall(tool_name="t", tool_input={}, tool_output={"x": 1}, step_index=0, duration_ms=10, status="success")
    def mock_llm(p, t): return '{"x": 2}'
    
    res_degrade = perturb_tool_result(call, "degrade", llm_fn=mock_llm)
    assert res_degrade == {"x": 2}
    
    res_enhance = perturb_tool_result(call, "enhance", llm_fn=mock_llm)
    assert res_enhance == {"x": 2}

def test_normalize_output():
    assert _normalize_output({"a": 1}) == {"a": 1}
    assert _normalize_output("text") == {"result": "text"}
    assert _normalize_output([1, 2]) == {"result": [1, 2]}
    assert _normalize_output(None) == {}
    assert _normalize_output(123) == {"result": "123"}

def test_parse_json_safe():
    assert _parse_json_safe("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _parse_json_safe("{\"b\": 2}") == {"b": 2}
