from counterfact.recommendations import (
    _estimate_gap_fix_improvement,
    _fallback_recommendations,
    _find_best_enhancer,
    _invert_classifier,
    detect_coverage_gaps,
    extract_empirical_fixes,
    generate_agent_spec,
    generate_recommendations,
    rank_recommendations,
)
from counterfact.types import ClassifierResult, FailureClassification, Perturbation, Recommendation, SimulationResult


def _make_sim(s_id, q_score, pert_agent=None, pert_strategy=None, clf_results=None, is_baseline=False, p_output=""):
    pert = Perturbation(agent=pert_agent, strategy=pert_strategy, description="", magnitude=1.0) if pert_agent else None
    return SimulationResult(
        simulation_id=s_id,
        perturbation=pert,
        quality_score=q_score,
        classifier_results=clf_results or [],
        perturbed_output=p_output,
        is_baseline=is_baseline
    )

def test_extract_empirical_fixes():
    trace = [
        {"node": "retriever", "output": "doc"},
        {"node": "critic", "output": "rev"}
    ]

    sims = [
        _make_sim(0, 0.5, is_baseline=True),  # baseline
        # A: Enhancement fix (retriever)
        _make_sim(1, 0.8, pert_agent="retriever", pert_strategy="enhance", p_output="better doc"),
        # B: Ablation fix (critic)
        _make_sim(2, 0.9, pert_agent="critic", pert_strategy="ablate"),
    ]

    recs = extract_empirical_fixes(sims, 0.5, trace, improvement_threshold=0.10)
    # Critic ablation and Retriever enhance
    assert len(recs) == 2
    rec_enhance = next(r for r in recs if r.intervention_type == "modify_agent")
    assert rec_enhance.target_agent == "retriever"
    assert round(rec_enhance.estimated_failure_reduction, 2) == 0.3

    rec_ablate = next(r for r in recs if r.intervention_type == "restructure" or r.intervention_type == "remove_loop")
    assert rec_ablate.target_agent == "critic"
    assert round(rec_ablate.estimated_failure_reduction, 2) == 0.4

def test_extract_empirical_fixes_damping():
    trace = [
        {"node": "retriever", "output": "doc"},
        {"node": "synthesizer", "output": "v1"},
        {"node": "critic", "output": "feedback"},
        {"node": "synthesizer", "output": "v2"}
    ]

    sims_1 = [
        _make_sim(0, 0.5, is_baseline=True),  # baseline 0.5
        _make_sim(1, 0.8, pert_agent="critic", pert_strategy="ablate") # > 1.3
    ]
    recs_1 = extract_empirical_fixes(sims_1, 0.5, trace, improvement_threshold=0.10)
    assert any("entirely" in r.description for r in recs_1)

    sims_2 = [
        _make_sim(0, 0.5, is_baseline=True),
        _make_sim(1, 0.6, pert_agent="critic", pert_strategy="ablate") # 1.2 -> > 1.1
    ]
    recs_2 = extract_empirical_fixes(sims_2, 0.5, trace, improvement_threshold=0.10)
    assert any("1 iteration" in r.description for r in recs_2)

    sims_3 = [
        _make_sim(0, 0.5, is_baseline=True),
        _make_sim(1, 0.53, pert_agent="critic", pert_strategy="ablate") # 1.06 -> > 1.0
    ]
    recs_3 = extract_empirical_fixes(sims_3, 0.5, trace, improvement_threshold=0.10)
    assert any("2 iterations" in r.description for r in recs_3)

def test_detect_coverage_gaps():
    trace = [{"node": "agent1"}]
    sims = [
        _make_sim(0, 0.5, is_baseline=True, clf_results=[ClassifierResult(name="factuality", score=0.4, reasoning="")]),
        _make_sim(1, 0.8, pert_agent="agent1", pert_strategy="enhance", clf_results=[ClassifierResult(name="factuality", score=0.6, reasoning="")])
    ]

    # Gap exists because factuality is low AND shapley is near zero
    per_clf_shapley = {
        "factuality": {"agent1": 0.05} # below threshold 0.10
    }
    failing = ["factuality"]

    recs = detect_coverage_gaps(per_clf_shapley, failing, sims, trace)
    assert len(recs) == 1
    assert recs[0].intervention_type == "add_agent"
    assert recs[0].agent_spec is not None

def test_generate_agent_spec_template():
    spec = generate_agent_spec("premise_validity", [], [])
    assert spec.name == "premise_validator"

def test_generate_agent_spec_inversion():
    spec = generate_agent_spec("factuality", [], [])
    assert spec.name == "factuality_validator"

def test_generate_agent_spec_diff():
    # Force generic by using an unknown classifier
    spec = generate_agent_spec("unknown_metric", [], [])
    assert spec.name == "unknown_metric_validator"

def test_rank_recommendations():
    recs = [
        Recommendation(title="R1", description="", intervention_type="", target_agent="", estimated_failure_reduction=0.1, complexity="low", priority=0, evidence_source="", measurement_confidence="estimated"),
        Recommendation(title="R2", description="", intervention_type="", target_agent="", estimated_failure_reduction=0.5, complexity="high", priority=0, evidence_source="", measurement_confidence="measured"),
        Recommendation(title="R3", description="", intervention_type="", target_agent="", estimated_failure_reduction=0.5, complexity="low", priority=0, evidence_source="", measurement_confidence="measured"),
    ]
    ranked = rank_recommendations(recs)
    assert ranked[0].title == "R3"  # measured, high reduction, low complexity
    assert ranked[1].title == "R2"  # measured, high reduction, high complexity
    assert ranked[2].title == "R1"  # estimated, low reduction

def test_generate_recommendations_fallback():
    clf = FailureClassification(failure_type="architectural_gap", description="", failing_classifiers=[], evidence=[], confidence="high")
    recs = generate_recommendations(clf, {}, [], domain="decision")
    assert len(recs) > 0
    assert recs[0].title == "Add Evidence Verification Agent"

def test_generate_recommendations_llm():
    def mock_llm(prompt, temp):
        return '[{"title": "LLM Fix", "description": "desc", "intervention_type": "add_agent", "target_agent": null, "estimated_failure_reduction": 0.9, "complexity": "low"}]'

    clf = FailureClassification(failure_type="toxic_loop", description="", failing_classifiers=[], evidence=[], confidence="high")
    recs = generate_recommendations(clf, {}, [], domain="decision", llm_fn=mock_llm)
    assert len(recs) == 1
    assert recs[0].title == "LLM Fix"

def test__find_best_enhancer():
    # Helper should find the best enhancer for a gap
    sims = [
        _make_sim(0, 0.5, is_baseline=True, clf_results=[ClassifierResult("c1", 0.5, "")]),
        _make_sim(1, 0.6, pert_agent="a", pert_strategy="enhance", clf_results=[ClassifierResult("c1", 0.8, "")]),
        _make_sim(2, 0.55, pert_agent="b", pert_strategy="enhance", clf_results=[ClassifierResult("c1", 0.6, "")]),
    ]
    agent = _find_best_enhancer("c1", ["a", "b"], sims)
    assert agent == "a"

def test__estimate_gap_fix_improvement():
    sims = [
        _make_sim(0, 0.5, is_baseline=True, clf_results=[ClassifierResult("c1", 0.5, "")]),
        _make_sim(1, 0.6, pert_agent="a", pert_strategy="enhance", clf_results=[ClassifierResult("c1", 0.8, "")]),
    ]
    imp = _estimate_gap_fix_improvement("c1", sims, ["a", "b"])
    assert imp > 0.0

def test__invert_classifier():
    spec = _invert_classifier("factuality")
    assert spec is not None
    assert spec.name == "factuality_validator"
    assert _invert_classifier("unknown") is None

def test__fallback_recommendations():
    clf_local = FailureClassification(failure_type="local", description="", failing_classifiers=[], evidence=[], confidence="high", dominant_agent="prompt")
    recs = _fallback_recommendations(clf_local, {}, "decision")
    assert any("prompt" in r.title.lower() for r in recs)

    clf_systemic = FailureClassification(failure_type="systemic", description="", failing_classifiers=[], evidence=[], confidence="high")
    recs = _fallback_recommendations(clf_systemic, {}, "decision")
    assert any("checkpoint" in r.title.lower() for r in recs)

def test_evaluate_recommendation():
    from counterfact.recommendations import evaluate_recommendation
    rec = Recommendation(title="Fix", description="Do this", intervention_type="add", target_agent="", estimated_failure_reduction=0.5, complexity="low", priority=1, evidence_source="", measurement_confidence="estimated")

    def mock_llm(prompt, temp):
        return "Fixed output"

    class MockRegistry:
        def run_all(self, *args, **kwargs):
            return [ClassifierResult("c1", 0.9 if args[1] == "Fixed output" else 0.1, "")]

    registry = MockRegistry()
    res = evaluate_recommendation(rec, "query", "bad output", [], llm_fn=mock_llm, registry=registry, num_eval_runs=2)
    assert res.verdict == "recommended"
    assert res.failure_reduction > 0.5

    # Test caution verdict
    class MockRegistryCaution:
        def __init__(self):
            self.calls = 0

        def run_all(self, *args, **kwargs):
            self.calls += 1
            # baseline runs 10 times, fixed runs 10 times
            # baseline failures: 10/10
            # fixed failures: 8/10 (reduction is 0.2) -> Caution
            is_fixed = args[1] == "Fixed output"
            if not is_fixed:
                return [ClassifierResult("c1", 0.1, "")]
            else:
                # First two fixed outputs pass
                if self.calls <= 12:
                    return [ClassifierResult("c1", 0.9, "")]
                return [ClassifierResult("c1", 0.1, "")]

    res2 = evaluate_recommendation(rec, "query", "bad output", [], llm_fn=mock_llm, registry=MockRegistryCaution(), num_eval_runs=10)
    assert res2.verdict == "caution"

def test__extract_from_enhancement_diff():
    from counterfact.recommendations import _extract_from_enhancement_diff
    trace = [{"node": "a"}, {"node": "b"}]
    sims = [
        _make_sim(0, 0.5, is_baseline=True, clf_results=[ClassifierResult("c1", 0.5, "")], p_output="base"),
        _make_sim(1, 0.6, pert_agent="a", pert_strategy="enhance", clf_results=[ClassifierResult("c1", 0.8, "")], p_output="enhanced"),
    ]
    spec = _extract_from_enhancement_diff("c1", sims, trace)
    assert spec is not None
    assert spec.name == "c1_agent"
    assert spec.baseline_example == "base"
    assert spec.enhanced_example == "enhanced"

def test__llm_generate_spec():
    from counterfact.recommendations import _llm_generate_spec
    def mock_llm(prompt, temp):
        return '{"name": "test_agent", "position": "after_a", "function": "do testing", "input_keys": ["a"], "output_keys": ["b"]}'

    spec = _llm_generate_spec("c1", [{"node": "a"}], mock_llm)
    assert spec is not None
    assert spec.name == "test_agent"

    # Test fallback on invalid JSON
    def bad_llm(prompt, temp):
        return 'not json'
    spec2 = _llm_generate_spec("c1", [{"node": "a"}], bad_llm)
    assert spec2 is None
