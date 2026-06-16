"""Quality classifiers for the FinanceBench pipeline + build_registry().

Three dimensions, matching the case study:
  - accuracy  (LLM, weight 2.0): does the output contain the correct figure?
  - precision (pure regex, weight 1.5): is the figure EXACT, not rounded/converted?
  - grounding (LLM, weight 1.0): are all claims traceable to the source document?

Stateless: accuracy/precision resolve ground truth from data.GROUND_TRUTH keyed by
the query string (which counterfact passes to classifiers), and grounding checks
against the canonical data.CASH_FLOW directly — because diagnose() does not forward
a `sources` string to classifiers, so relying on the `sources` arg would score
grounding against an empty document.
"""
from __future__ import annotations

import json
import re

from counterfact.classifiers import ClassifierRegistry
from counterfact.types import ClassifierResult

from .data import CASH_FLOW, GROUND_TRUTH
from .llm import call


def _gt_for(query: str) -> str:
    return GROUND_TRUTH.get(query, "")


def accuracy_clf(query, output, sources) -> ClassifierResult:
    gt = _gt_for(query)
    if not output or len(output) < 10:
        return ClassifierResult(name="accuracy", score=0.1, reasoning="No output", weight=2.0)
    if not gt:
        return ClassifierResult(name="accuracy", score=0.5, reasoning="No ground truth for query", weight=2.0)
    prompt = f"""Does this output contain the correct answer?

CORRECT: {gt}
OUTPUT: {output}

Score:
- 1.0 = states the exact figure ({gt})
- 0.5 = close/rounded (within ~5% but not exact)
- 0.0 = wrong figure or no figure

JSON only: {{"score": 0.0, "reasoning": "..."}}"""
    try:
        m = re.search(r"\{[^}]+\}", call(prompt, max_tokens=120), re.DOTALL)
        if m:
            d = json.loads(m.group())
            return ClassifierResult(name="accuracy", score=float(d["score"]),
                                    reasoning=d.get("reasoning", ""), weight=2.0)
    except Exception:
        pass
    return ClassifierResult(name="accuracy", score=0.5, reasoning="parse error", weight=2.0)


def precision_clf(query, output, sources) -> ClassifierResult:
    gt = _gt_for(query)
    if not output or len(output) < 10:
        return ClassifierResult(name="precision", score=0.1, reasoning="No output", weight=1.5)
    m = re.search(r"[\d,]+", gt)
    gt_num = m.group() if m else ""
    gt_plain = gt_num.replace(",", "")
    if gt_num and (gt_num in output or gt_plain in output):
        return ClassifierResult(name="precision", score=1.0, reasoning=f"Exact {gt}", weight=1.5)
    if "billion" in output.lower():
        return ClassifierResult(name="precision", score=0.2, reasoning="Converted to billions", weight=1.5)
    if gt_plain:
        r100 = f"{round(int(gt_plain), -2):,}"
        r10 = f"{round(int(gt_plain), -1):,}"
        if r100 in output or r10 in output:
            return ClassifierResult(name="precision", score=0.4, reasoning="Rounded", weight=1.5)
    return ClassifierResult(name="precision", score=0.3, reasoning="Unclear precision", weight=1.5)


def grounding_clf(query, output, sources) -> ClassifierResult:
    if not output or len(output) < 10:
        return ClassifierResult(name="grounding", score=0.5, reasoning="No output", weight=1.0)
    # Use the canonical document, not the (empty) `sources` arg.
    source_doc = sources if sources and len(sources) > 20 else CASH_FLOW
    prompt = f"""Check whether this analysis contains claims not supported by the source.

SOURCE DOCUMENT:
{source_doc}

OUTPUT:
{output}

Look specifically for:
- Industry benchmarks or peer comparisons (NOT in the source)
- Projected or forecasted figures (NOT in the source)
- Rounded figures that differ from the source values

Score:
- 1.0 = every specific claim traces to the source
- 0.5 = core answer from source but includes unsourced context
- 0.0 = contains fabricated or unverifiable figures

JSON only: {{"score": 0.0, "reasoning": "..."}}"""
    try:
        m = re.search(r"\{[^}]+\}", call(prompt, max_tokens=150), re.DOTALL)
        if m:
            d = json.loads(m.group())
            return ClassifierResult(name="grounding", score=float(d["score"]),
                                    reasoning=d.get("reasoning", ""), weight=1.0)
    except Exception:
        pass
    return ClassifierResult(name="grounding", score=0.5, reasoning="parse error", weight=1.0)


def build_registry() -> ClassifierRegistry:
    """ClassifierRegistry with accuracy/precision/grounding under 'financebench'."""
    reg = ClassifierRegistry()
    reg.register(accuracy_clf, "financebench")
    reg.register(precision_clf, "financebench")
    reg.register(grounding_clf, "financebench")
    return reg
