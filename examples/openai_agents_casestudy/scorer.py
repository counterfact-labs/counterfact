"""A local scorer with the autoevals / Braintrust scorer interface.

Braintrust evals score outputs with callables shaped like
``scorer(output, expected, ...) -> Score`` where ``Score.score`` is in [0, 1].
This module mirrors that exactly so the case study reads like real Braintrust
usage and plugs straight into ``counterfact.integrations.braintrust``.

To go live, replace ``refund_amount_scorer`` with a real autoevals scorer, e.g.::

    from autoevals import Levenshtein            # or NumericDiff, Factuality, ...
    qf = quality_fn_from_scorer(Levenshtein())

The semantics here ("the exact dollar figure must survive into the reply") match
how a financial-support team would actually grade these answers.
"""

from __future__ import annotations


class Score:
    """Minimal stand-in for autoevals.Score."""

    def __init__(self, name: str, score: float, metadata: dict | None = None):
        self.name = name
        self.score = score
        self.metadata = metadata or {}


def refund_amount_scorer(output: str, expected: str, input: str | None = None) -> Score:
    """Score 1.0 iff the expected dollar figure appears in the reply, else 0.0."""
    present = bool(expected) and str(expected) in (output or "")
    return Score(
        name="refund_amount_present",
        score=1.0 if present else 0.0,
        metadata={"expected": expected, "found": present},
    )
