"""Offline tests for the Braintrust adapter.

No network and no `braintrust`/`autoevals` packages required: we use plain
callables shaped like autoevals scorers and plain dicts shaped like Braintrust
dataset records.
"""

import pytest

from counterfact.integrations.braintrust import (
    cases_from_dataset,
    quality_fn_from_scorer,
)


class FakeScore:
    """Mimics an autoevals Score: a .score in [0, 1] plus metadata."""

    def __init__(self, score, name="fake", metadata=None):
        self.score = score
        self.name = name
        self.metadata = metadata or {}


def test_quality_fn_unwraps_score_object():
    def scorer(output, expected):
        return FakeScore(1.0 if expected in output else 0.0)

    qf = quality_fn_from_scorer(scorer)
    assert qf("the answer is $250", {"expected": "$250"}) == 1.0
    assert qf("no figure here", {"expected": "$250"}) == 0.0


def test_quality_fn_accepts_bare_float():
    qf = quality_fn_from_scorer(lambda output, expected: 0.7)
    assert qf("anything", {"expected": "x"}) == 0.7


def test_quality_fn_none_score_is_zero():
    qf = quality_fn_from_scorer(lambda output, expected: FakeScore(None))
    assert qf("anything", {"expected": "x"}) == 0.0


def test_quality_fn_passes_input_when_requested():
    seen = {}

    def scorer(output, expected, input):
        seen["input"] = input
        return 1.0

    qf = quality_fn_from_scorer(scorer, input_key="question", pass_input=True)
    qf("out", {"expected": "e", "question": "what?"})
    assert seen["input"] == "what?"


def test_quality_fn_missing_expected_errors_by_default():
    qf = quality_fn_from_scorer(lambda output, expected: 1.0)
    with pytest.raises(KeyError):
        qf("out", {"no_expected": 1})


def test_quality_fn_missing_expected_zero_mode():
    qf = quality_fn_from_scorer(lambda output, expected: 1.0, on_missing_expected="zero")
    assert qf("out", {}) == 0.0


def test_cases_from_dataset_maps_fields_and_embeds_gold():
    dataset = [
        {"input": {"question": "Q1"}, "expected": "$250", "id": "a"},
        {"input": {"question": "Q2"}, "expected": "$99"},
    ]
    cases = cases_from_dataset(dataset, embed_expected_as="expected")
    assert cases[0]["id"] == "a"
    assert cases[0]["gold"] == "$250"
    # gold embedded into the input state so quality_fn can read it during re-runs
    assert cases[0]["input"]["expected"] == "$250"
    assert cases[0]["input"]["question"] == "Q1"
    # missing id falls back to positional index
    assert cases[1]["id"] == 1


def test_cases_from_dataset_scalar_input():
    cases = cases_from_dataset([{"input": "raw text", "expected": "y"}])
    assert cases[0]["input"]["input"] == "raw text"
    assert cases[0]["gold"] == "y"


def test_cases_from_dataset_attribute_style_records():
    class Record:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    cases = cases_from_dataset([Record(input={"q": "Q"}, expected="z", id="r1")])
    assert cases[0]["id"] == "r1"
    assert cases[0]["gold"] == "z"
    assert cases[0]["input"]["expected"] == "z"


def test_round_trip_dataset_then_score():
    """A dataset case feeds a quality_fn directly — the gold embedded by
    cases_from_dataset is exactly what quality_fn_from_scorer reads."""
    cases = cases_from_dataset([{"input": {"q": "refund?"}, "expected": "$250"}])
    qf = quality_fn_from_scorer(lambda output, expected: 1.0 if expected in output else 0.0)
    state = cases[0]["input"]
    assert qf("we issued $250", state) == 1.0
    assert qf("nothing", state) == 0.0
