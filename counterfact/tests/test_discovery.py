import pytest
from counterfact.discovery import (
    discover_pipeline,
    infer_schemas,
    estimate_importance,
    suggest_perturbation_strategies,
    suggest_classifiers,
    infer_domain,
    _extract_agent_info_from_traces,
    _compute_discovery_confidence,
    _parse_json_response
)
from counterfact.types import AgentProfile

def test_discover_pipeline_no_args():
    with pytest.raises(ValueError, match="At least one"):
        discover_pipeline()

def test_discover_pipeline_no_llm():
    with pytest.raises(ValueError, match="llm_fn is required"):
        discover_pipeline(traces=[[{"node": "a"}]])

def mock_llm_fn(prompt, temp):
    p = prompt.lower()
    if "what domain" in p:
        return '"rag"'
    if "estimating how important" in p:
        return '{"agent1": 0.9}'
    if "perturbation strategies" in p:
        return '[{"agent": "agent1", "strategy": "ablate", "description": "desc", "magnitude": 1.0}]'
    if "recommending quality classifiers" in p:
        return '["factuality"]'
    if "understand what each agent does" in p:
        return '[{"name": "agent1", "inferred_role": "retriever", "description": "desc"}]'
    return "{}"

def test_discover_pipeline():
    traces = [[{"node": "agent1", "input": {"q": "hi"}, "output": {"d": "docs"}}]]
    plan = discover_pipeline(traces=traces, llm_fn=mock_llm_fn)
    assert plan.domain == "rag"
    assert len(plan.agent_profiles) == 1
    assert plan.agent_profiles[0].name == "agent1"
    assert plan.agent_profiles[0].inferred_role == "retriever"
    assert plan.agent_profiles[0].estimated_importance == 0.9
    assert plan.suggested_classifiers == ["factuality"]
    assert len(plan.perturbations) == 1

def test_infer_schemas():
    profiles = [AgentProfile(name="agent1", inferred_role="other", description="")]
    traces = [
        [{"node": "agent1", "input": {"a": 1, "b": 2}, "output": {"x": 1}}],
        [{"node": "agent1", "input": {"a": 1}, "output": {"x": 1}}]
    ]
    profiles = infer_schemas(profiles, traces)
    assert profiles[0].input_schema["a"] == "required"
    assert profiles[0].input_schema["b"] == "optional"
    assert profiles[0].output_schema["x"] == "required"

def test_estimate_importance_fallback():
    def broken_llm(p, t): raise ValueError("err")
    profiles = [AgentProfile(name="agent1", inferred_role="retriever", description="")]
    res = estimate_importance(profiles, broken_llm)
    assert res[0].estimated_importance == 0.8

def test_suggest_perturbation_strategies_fallback():
    def broken_llm(p, t): raise ValueError("err")
    profiles = [AgentProfile(name="agent1", inferred_role="retriever", description="")]
    res = suggest_perturbation_strategies(profiles, broken_llm)
    assert len(res) > 0
    assert res[0].agent == "agent1"

def test_suggest_classifiers_fallback():
    def broken_llm(p, t): raise ValueError("err")
    profiles = [AgentProfile(name="agent1", inferred_role="retriever", description="")]
    res = suggest_classifiers(profiles, broken_llm)
    assert "factuality" in res

def test_infer_domain_fallback():
    def broken_llm(p, t): raise ValueError("err")
    profiles = [AgentProfile(name="agent1", inferred_role="retriever", description="")]
    res = infer_domain(profiles, broken_llm)
    assert res == "general"

def test_extract_agent_info_from_traces():
    traces = [[{"node": "a", "input": {"x": 1}, "output": {"y": 2}}]]
    info = _extract_agent_info_from_traces(traces)
    assert "a" in info
    assert info["a"]["count"] == 1
    assert "x" in info["a"]["input_keys"]
    assert "y" in info["a"]["output_keys"]

def test_compute_discovery_confidence():
    profiles = [AgentProfile(name="a", inferred_role="retriever", description="")]
    traces = [[]] * 15 # 15 traces -> +0.2
    conf = _compute_discovery_confidence(traces, "a good description of pipeline", profiles)
    # base 0.3 + 0.2 (traces) + 0.2 (role clarity) + 0.05 (both) = 0.75
    assert 0.74 < conf < 0.76

def test_parse_json_response():
    assert _parse_json_response("```json\n[1, 2]\n```") == [1, 2]
    assert _parse_json_response("here is it: [1, 2] bye") == [1, 2]
    assert _parse_json_response("```\n{\"a\": 1}\n```") == {"a": 1}
    # repair truncated
    assert _parse_json_response("[{\"a\": 1}, {\"a\": 2") == [{"a": 1}]
