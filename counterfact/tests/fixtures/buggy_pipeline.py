"""Deterministic, no-LLM fixture pipeline for the counterfact-debugger skill tests.

A 3-node pipeline (retriever -> parser -> responder) where the parser has a bug:
it strips digits, so the final answer never contains the figure the query asks for.
``build`` returns the buggy pipeline; ``build_fixed`` returns the corrected one.

No network or API key is required: the only classifier (``has_number``) is a pure
string check, so this exercises the full diagnose path deterministically in CI.

Used by test_skill_counterfact_debugger.py via the skill's cf_diagnose.py runner
(``--factory buggy_pipeline:build``), which is why it lives as an importable module.
"""
from counterfact import END, StateGraph
from counterfact.classifiers import ClassifierRegistry
from counterfact.types import ClassifierResult

DATA = {"revenue": "$32,765 million"}


def retriever(state):
    return {"raw": DATA.get(state.get("metric_key", ""), "")}


def _parser(strip_digits):
    def parser(state):
        raw = state.get("raw", "")
        if strip_digits:  # BUG: destroys the numeric answer
            cleaned = "".join(c for c in raw if not c.isdigit() and c not in "$,")
        else:
            cleaned = raw
        return {"parsed": cleaned.strip()}

    return parser


def responder(state):
    parsed = state.get("parsed", "")
    return {"output": f"The answer is {parsed}." if parsed else "No answer."}


def _build(strip_digits):
    g = StateGraph(dict)
    g.add_node("retriever", retriever)
    g.add_node("parser", _parser(strip_digits))
    g.add_node("responder", responder)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "parser")
    g.add_edge("parser", "responder")
    g.add_edge("responder", END)
    return g.compile()


def build():
    """The buggy pipeline (parser strips digits)."""
    return _build(strip_digits=True)


def build_fixed():
    """The corrected pipeline (parser preserves the figure)."""
    return _build(strip_digits=False)


def _has_number(query, output, sources):
    import re

    ok = bool(re.search(r"\d", output or ""))
    return ClassifierResult(
        name="has_number",
        score=1.0 if ok else 0.1,
        reasoning="contains a digit" if ok else "no numeric value in output",
        weight=1.5,
    )


def build_registry():
    reg = ClassifierRegistry()
    reg.register(_has_number, domain="finance")
    return reg
