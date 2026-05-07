from unittest.mock import MagicMock, patch

import pytest

from counterfact.classifiers import (
    ClassifierRegistry,
    _call_llm,
    _classify_attributability,
    _classify_causal_coherence,
    _classify_completeness,
    _classify_decision_consistency,
    _classify_evidence_sufficiency,
    _classify_factuality,
    _classify_internal_consistency,
    _classify_policy_compliance,
    _classify_premise_validity,
    _classify_reasoning_soundness,
    _classify_regulatory_compliance,
    _parse_classifier_response,
    aggregate_quality,
    get_default_registry,
    register_classifier,
    run_classifiers,
    set_llm_caller,
)
from counterfact.types import ClassifierResult


def test_registry_register_and_get():
    registry = ClassifierRegistry()

    def my_clf(q, o, s):
        return ClassifierResult("my_clf", 0.9, "ok")

    registry.register(my_clf, "custom_domain")

    clfs = registry.get("custom_domain")
    assert len(clfs) == 1
    assert clfs[0] is my_clf

    # Fallback to rag if unknown
    rag_clfs = registry.get("unknown_domain")
    # By default, a new registry has no rag classifiers unless added.
    assert len(rag_clfs) == 0


def test_registry_run_all():
    registry = ClassifierRegistry()

    def clf1(q, o, s): return ClassifierResult("c1", 0.8, "r1")
    def clf2(q, o, s): return ClassifierResult("c2", 0.6, "r2")

    registry.register(clf1, "test")
    registry.register(clf2, "test")

    results = registry.run_all("q", "o", "s", "test")
    assert len(results) == 2
    assert results[0].score == 0.8
    assert results[1].score == 0.6


def test_aggregate_quality():
    results = [
        ClassifierResult("c1", 0.8, "", weight=2.0),
        ClassifierResult("c2", 0.2, "", weight=1.0),
    ]
    # (0.8*2 + 0.2*1) / 3 = 1.8 / 3 = 0.6
    assert ClassifierRegistry.aggregate_quality(results) == pytest.approx(0.6)

    # Empty list
    assert ClassifierRegistry.aggregate_quality([]) == 0.5


def test_global_registry_wrappers():
    def dummy_clf(q, o, s): return ClassifierResult("dummy", 1.0, "")

    register_classifier(dummy_clf, "global_test")
    results = run_classifiers("q", "o", "s", "global_test")
    assert len(results) == 1
    assert results[0].name == "dummy"

    agg = aggregate_quality(results)
    assert agg == 1.0

    assert get_default_registry() is not None


def test_llm_caller():
    # Test not set error
    set_llm_caller(None)
    with pytest.raises(RuntimeError, match="No LLM caller configured"):
        _call_llm("prompt")

    # Test valid call
    mock_llm = MagicMock(return_value="response")
    set_llm_caller(mock_llm)
    res = _call_llm("my prompt", 0.5)
    assert res == "response"
    mock_llm.assert_called_with("my prompt", 0.5)


class TestParseClassifierResponse:
    def test_clean_json(self):
        res = _parse_classifier_response('{"score": 0.85, "reasoning": "ok"}')
        assert res["score"] == 0.85
        assert res["reasoning"] == "ok"

    def test_markdown_json(self):
        res = _parse_classifier_response('```json\n{"score": 0.7, "reasoning": "ok"}\n```')
        assert res["score"] == 0.7

    def test_markdown_no_lang(self):
        res = _parse_classifier_response('```\n{"score": 0.6}\n```')
        assert res["score"] == 0.6

    def test_clamp_score(self):
        assert _parse_classifier_response('{"score": 1.5}')["score"] == 1.0
        assert _parse_classifier_response('{"score": -0.5}')["score"] == 0.0

    def test_fallback_number(self):
        res = _parse_classifier_response("I give this a 0.42 because reasons.")
        assert res["score"] == 0.42

    def test_fallback_one(self):
        res = _parse_classifier_response("Score is 1.0")
        assert res["score"] == 1.0

    def test_fallback_default(self):
        res = _parse_classifier_response("No numbers here at all!")
        assert res["score"] == 0.5


class TestBuiltinClassifiers:
    @patch("counterfact.classifiers._call_llm")
    def test_all_llm_classifiers(self, mock_call):
        mock_call.return_value = '{"score": 0.9, "reasoning": "test"}'

        # RAG
        assert _classify_factuality("q", "o", "s").score == 0.9
        assert _classify_attributability("q", "o", "s").score == 0.9
        assert _classify_premise_validity("q", "o", "s").score == 0.9
        assert _classify_internal_consistency("q", "o", "s").score == 0.9
        assert _classify_causal_coherence("q", "o", "s").score == 0.9
        assert _classify_regulatory_compliance("q", "o", "s").score == 0.9

        # Decision
        assert _classify_policy_compliance("q", "o", "s").score == 0.9
        assert _classify_reasoning_soundness("q", "o", "s").score == 0.9
        assert _classify_decision_consistency("q", "o", "s").score == 0.9
        assert _classify_completeness("q", "o", "s").score == 0.9

    def test_evidence_sufficiency(self):
        # Deterministic classifier
        res_high = _classify_evidence_sufficiency("q", "system confirms and billing records show", "s")
        assert res_high.score == 1.0

        res_med = _classify_evidence_sufficiency("q", "verified against", "s")
        assert res_med.score == 0.5

        res_low = _classify_evidence_sufficiency("q", "just trusting the customer", "s")
        assert res_low.score == 0.0
