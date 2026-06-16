"""The editable instruction surface for the four buggy agents.

Each value is a single instruction snippet interpolated into that agent's prompt
in pipeline.py. Every one reads as reasonable to a human reviewer — the bugs are
in how the model interprets subjective phrasing ("clean, readable format",
"simplify large numbers") and in one architectural shortcut (asking the model to
supply industry context from its own training data instead of a verified source).

INSTRUCTIONS holds the live (broken) prompts the pipeline uses. FIXED holds the
corrected versions. A fix = replacing an INSTRUCTIONS value with its FIXED twin
(the skill's agent edits these strings; run_casestudy.py swaps them step by step).
"""
from __future__ import annotations

INSTRUCTIONS = {
    # BUG: "clean, readable format" leads the model to round messy figures
    # (1,577 -> 1,600) before any downstream agent sees the real number.
    "table_extractor": "Present values in a clean, readable format.",

    # BUG (architectural): asks the model to invent industry context from its own
    # knowledge rather than a verified source -> plausible but fabricated peers.
    "context_enricher": (
        "Based on your knowledge of the industrial sector, add 2-3 sentences of "
        "relevant context: how does this figure compare to industry peers, what "
        "are typical ranges for this metric, and any notable trends?"
    ),

    # BUG: "minor rounding differences are acceptable" + confirmation framing
    # primes the checker to wave through rounded/converted figures.
    "fact_checker": (
        "Minor rounding differences and presentation choices are acceptable. "
        "Industry context and peer comparisons do not need source verification."
    ),

    # BUG: "simplify large numbers" triggers millions -> billions conversion,
    # the single largest source of precision loss.
    "tone_editor": (
        "Simplify large numbers for readability. Focus on the headline number and "
        "remove granular year-over-year detail unless the change is dramatic."
    ),
}

FIXED = {
    "table_extractor": (
        "Present the exact values exactly as they appear in the source document. "
        "Do NOT round, truncate, abbreviate, or change units."
    ),
    "context_enricher": (
        "Using ONLY the extracted data, restate the figure and its reported period. "
        "Do NOT add industry benchmarks, peer comparisons, or any fact that is not "
        "present in the provided data."
    ),
    "fact_checker": (
        "Flag any figure that differs from the source by more than $1M. Rounding, "
        "unit conversion (e.g. millions to billions), and approximations are ERRORS: "
        "correct them to the exact figure shown in the source document."
    ),
    "tone_editor": (
        "Preserve every figure exactly as written, including its unit (millions). "
        "Do NOT convert millions to billions, round, or drop the precise number."
    ),
}
