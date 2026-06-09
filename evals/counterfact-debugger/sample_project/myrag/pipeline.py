"""A small 3-node financial-QA pipeline.

retriever -> polisher -> responder

Symptom (the thing the user reports): when asked for a specific dollar figure, the
final answer never contains the number — it comes back like "The 2018 revenue was
about that amount." The team doesn't know which of the three agents is responsible.

NOTE: this is built with LangGraph. counterfact's diagnose needs the build recipe
that counterfact.StateGraph captures, so diagnosing this requires the one-line swap
(`from counterfact import StateGraph`).
"""
from langgraph.graph import END, StateGraph

# A tiny stand-in "filing store". The retriever looks figures up here.
FILINGS = {
    "3m_2018_revenue": "$32,765 million",
    "3m_2018_net_income": "$5,349 million",
}


def retriever(state):
    """Compose a correct draft answer from the looked-up figure."""
    figure = FILINGS.get(state.get("metric_key", ""), "")
    metric = state.get("metric_key", "the metric").replace("_", " ")
    draft = f"The {metric} was {figure}." if figure else "No figure found."
    return {"answer": draft}


def polisher(state):
    """Tidy the draft into a smoother sentence before returning it.

    Intended to remove clutter — but it strips out digits and currency symbols,
    so it silently destroys the very figure the user asked for.
    """
    answer = state.get("answer", "")
    cleaned = "".join(ch for ch in answer if not ch.isdigit() and ch not in "$,")
    cleaned = cleaned.replace("  ", " ").replace(" .", ".")
    return {"answer": cleaned.strip()}


def responder(state):
    """Return the (polished) answer as the final output."""
    return {"output": state.get("answer", "No figure found.")}


def build():
    """Return the compiled QA pipeline. No-arg factory for tooling."""
    g = StateGraph(dict)
    g.add_node("retriever", retriever)
    g.add_node("polisher", polisher)
    g.add_node("responder", responder)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "polisher")
    g.add_edge("polisher", "responder")
    g.add_edge("responder", END)
    return g.compile()


if __name__ == "__main__":
    app = build()
    out = app.invoke({"metric_key": "3m_2018_revenue", "query": "What was 3M's 2018 revenue?"})
    print(out["output"])  # -> "The 3m 2018 revenue was million."  (number is gone)
