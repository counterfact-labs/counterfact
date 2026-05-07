from counterfact.prompt_analysis import (
    _ablate_section,
    _compute_analysis_confidence,
    _parse_json_response,
    analyze_prompt,
    check_plan_quality,
    detect_conflicting_sections,
    detect_dead_sections,
    parse_prompt_sections,
    parse_system_prompt,
    run_prompt_section_attribution,
    score_few_shot_quality,
)
from counterfact.types import PlanStep, PromptSection


def test_parse_prompt_sections_llm():
    def mock_llm(p, t): return '[{"title": "t1", "content": "hello", "category": "instruction"}]'
    sections = parse_prompt_sections("hello world", mock_llm)
    assert len(sections) == 1
    assert sections[0].title == "t1"
    assert sections[0].content == "hello"
    assert sections[0].char_start == 0
    assert sections[0].char_end == 5

def test_parse_prompt_sections_fallback():
    def mock_llm(p, t): raise ValueError("err")
    sections = parse_prompt_sections("## Heading\nSome text\n\n## Another\nMore text", mock_llm)
    assert len(sections) == 2
    assert sections[0].title == "Heading"
    assert "Some text" in sections[0].content

def test_parse_system_prompt_double_newline():
    sections = parse_system_prompt("Para 1\n\nPara 2 Example:\nInput: 1 Output: 2")
    assert len(sections) == 2
    assert sections[0].category == "instruction"
    assert sections[1].category == "context"  # Example heuristic

def test_check_plan_quality():
    steps = [PlanStep(step_index=1, tool_name="t1", description="d1")]
    secs = [PromptSection(index=0, title="t", content="c", category="methodology", char_start=0, char_end=1)]
    def mock_llm(p, t): return '{"completeness": 1.0, "efficiency": 1.0, "adherence": 1.0}'

    res = check_plan_quality(steps, secs, mock_llm)
    assert res.passed is True
    assert res.details["score"] == 1.0

def test_check_plan_quality_empty():
    res = check_plan_quality([], [], lambda p, t: "")
    assert res.passed is False
    assert "No plan steps" in res.message

def test_run_prompt_section_attribution():
    secs = [PromptSection(index=0, title="t", content="c", category="instruction", char_start=0, char_end=1)]

    def mock_llm(p, t):
        if "simulating" in p:
            return "ablated out"
        return '{"score": 0.5}'

    attr = run_prompt_section_attribution("prompt", secs, "q", "out", mock_llm)
    assert attr[0] == 1.0 # Normalized

def test_ablate_section():
    sec1 = PromptSection(index=0, title="t", content="BBB", category="c", char_start=3, char_end=6)
    assert _ablate_section("AAABBBCCC", sec1) == "AAACCC"

    sec2 = PromptSection(index=0, title="t", content="BBB", category="c", char_start=0, char_end=0)
    assert _ablate_section("AAABBBCCC", sec2) == "AAACCC"

def test_detect_dead_sections():
    secs = [PromptSection(index=0, title="t", content="c", category="instruction", char_start=0, char_end=1)]
    def mock_llm(p, t): return '{"verdict": "not_followed"}'

    dead = detect_dead_sections(secs, ["out1"], mock_llm)
    assert dead == [0]

def test_detect_conflicting_sections():
    secs = [
        PromptSection(index=0, title="t1", content="c1", category="instruction", char_start=0, char_end=1),
        PromptSection(index=1, title="t2", content="c2", category="constraint", char_start=1, char_end=2)
    ]
    def mock_llm(p, t): return '[{"pair": 1, "verdict": "conflict"}]'

    conflicts = detect_conflicting_sections(secs, mock_llm)
    assert conflicts == [(0, 1)]

def test_analyze_prompt():
    def mock_llm(p, t):
        if "identifying its logical sections" in p.lower():
            return '[{"title": "t1", "content": "c", "category": "instruction"}]'
        if "evaluating" in p:
            return '{"score": 1.0}'
        if "simulating" in p:
            return "sim"
        if "lost by removing" in p.lower():
            return '{"score": 0.5}'
        if "evidence" in p:
            return '{"verdict": "followed"}'
        if "contradictions" in p:
            return '[]'
        return "{}"

    res = analyze_prompt("prompt", "query", "out", mock_llm)
    assert len(res.sections) == 1
    assert 0 in res.section_attributions

def test_score_few_shot_quality():
    s1 = PromptSection(index=0, title="Examples", content="Output: 1", category="instruction", char_start=0, char_end=1)
    res1 = score_few_shot_quality([s1])
    assert res1["score"] == 0.5  # < 3
    assert res1["count"] == 1

    s2 = PromptSection(index=1, title="Examples", content="Output: 1\nOutput: 2\nOutput: 3\nOutput: 4", category="instruction", char_start=0, char_end=1)
    res2 = score_few_shot_quality([s2])
    assert res2["score"] == 1.0  # 3-10

    s3 = PromptSection(index=2, title="Rules", content="No examples here", category="instruction", char_start=0, char_end=1)
    res3 = score_few_shot_quality([s3])
    assert res3["has_examples"] is False

def test_compute_analysis_confidence():
    assert _compute_analysis_confidence(5, 5, True) == 0.8
    assert _compute_analysis_confidence(1, 1, False) == 0.3

def test_parse_json_response():
    assert _parse_json_response("```json\n[1, 2]\n```") == [1, 2]
    assert _parse_json_response("{\"a\": 1}") == {"a": 1}
