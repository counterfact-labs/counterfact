"""A small, deterministic RAG pipeline built with counterfact.StateGraph.

retriever -> reranker -> synthesizer. State is a dict carrying ``query`` in and a
``final_output`` answer out, with intermediate ``docs`` (a ranked list of passages).

The synthesizer reads only the top ``TOP_K`` passages, modeling an LLM with a
limited context budget — so *where* the relevant passage is ranked decides whether
the answer is correct. That is what makes ranking quality (the reranker) matter
even though ablating the reranker looks harmless.
"""

from __future__ import annotations

from counterfact.graph import END, StateGraph

# How many passages the synthesizer actually reads (context budget).
TOP_K = 2

# A tiny corpus. Each query's relevant passage carries the gold figure; the rest
# are plausible-looking distractors from other parts of the same filing.
_CORPUS = {
    "capex": "FY2018 capital expenditure (purchases of PP&E) was $1,577 million.",
    "dividends": "Dividends paid in FY2018 totaled $3,193 million.",
    "buyback": "Repurchases of common stock in FY2018 were $4,870 million.",
    "da": "Depreciation and amortization for FY2018 was $1,488 million.",
    "ocf": "Net cash provided by operating activities in FY2018 was $6,439 million.",
}
_DISTRACTORS = [
    "The company operates across industrial, healthcare, and consumer segments.",
    "Net income including noncontrolling interest was $5,363 million.",
    "The board reaffirmed its commitment to returning cash to shareholders.",
    "Total assets were reported in the consolidated balance sheet.",
    "Management discussed macroeconomic headwinds in the outlook section.",
]


def _topic(query: str) -> str:
    q = query.lower()
    if "capital expenditure" in q or "capex" in q or "pp&e" in q:
        return "capex"
    if "dividend" in q:
        return "dividends"
    if "repurchase" in q or "buyback" in q or "common stock" in q:
        return "buyback"
    if "depreciation" in q or "amortization" in q:
        return "da"
    if "operating activities" in q or "operating cash" in q:
        return "ocf"
    return "capex"


def retriever(state: dict) -> dict:
    """Return ranked passages: the relevant one first, then distractors.

    Models a decent embedding search — the relevant passage is already near the
    top, so the reranker is not *fixing* a broken retrieval; it is *preserving* a
    good ordering. That is why ablating it (pass-through) looks harmless while
    degrading its quality is not.
    """
    topic = _topic(state.get("query", ""))
    docs = [_CORPUS[topic], *_DISTRACTORS]
    return {**state, "docs": docs}


def reranker(state: dict) -> dict:
    """Order passages best-first. Here it preserves the retriever's good order."""
    return {**state, "docs": list(state.get("docs", []))}


def synthesizer(state: dict) -> dict:
    """Answer from the top-K passages only (context budget)."""
    context = state.get("docs", [])[:TOP_K]
    for passage in context:
        # The gold figure is the dollar amount in the relevant passage.
        if "$" in passage:
            return {**state, "final_output": f"Based on the filing: {passage}"}
    return {**state, "final_output": "I could not find the figure in the retrieved context."}


def build():
    """Factory: a compiled counterfact graph (retriever -> reranker -> synthesizer)."""
    g = StateGraph(dict)
    g.add_node("retriever", retriever)
    g.add_node("reranker", reranker)
    g.add_node("synthesizer", synthesizer)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "reranker")
    g.add_edge("reranker", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()
