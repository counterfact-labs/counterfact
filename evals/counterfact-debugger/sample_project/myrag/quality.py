"""Quality classifier for the QA pipeline: does the answer contain a number?

This is a pure-string check (no LLM / no network), so diagnosis is deterministic.
Expose ``build_registry`` so the counterfact debugger can score with the metric
that actually reflects the failure the user reported.
"""
import re

from counterfact.classifiers import ClassifierRegistry
from counterfact.types import ClassifierResult


def has_number(query, output, sources):
    ok = bool(re.search(r"\d", output or ""))
    return ClassifierResult(
        name="has_number",
        score=1.0 if ok else 0.1,
        reasoning="answer contains a numeric figure" if ok else "answer has no number",
        weight=1.5,
    )


def build_registry():
    reg = ClassifierRegistry()
    reg.register(has_number, domain="finance")
    return reg
