"""The 8-agent financial-RAG pipeline (counterfact StateGraph).

    query_parser -> doc_retriever -> table_extractor -> context_enricher ->
    synthesizer -> fact_checker -> tone_editor -> output_formatter

build() is a no-arg factory returning the COMPILED graph, so the skill can drive
it via `--factory financebench_casestudy.pipeline:build`. The four buggy agents read
their key instruction from prompts.INSTRUCTIONS (the editable fix surface).
Synthesizer uses Sonnet; the rest use Haiku.
"""
from __future__ import annotations

import json
from typing import TypedDict

from counterfact import END, StateGraph

from .llm import HAIKU, SONNET, call
from .prompts import INSTRUCTIONS


class State(TypedDict):
    """Declared channels so every key (incl. source_document) persists across
    nodes and survives ablation no-ops. An untyped ``dict`` schema would drop
    keys a node doesn't re-emit."""
    query: str
    source_document: str
    parsed_query: str
    retrieved_sections: str
    extracted_data: str
    enriched_context: str
    analysis: str


def query_parser(state: dict) -> dict:
    """Classify the question. Emergent bug: the model sometimes labels a
    straight lookup as 'estimation', nudging the synthesizer to approximate."""
    prompt = f"""Parse this financial question into structured components.

QUESTION: {state['query']}

Extract:
- company, metric, period
- query_type: "lookup" (retrieve a specific reported figure) or
  "estimation" (compute or approximate a figure)

Respond with JSON only:
{{"company": "...", "metric": "...", "period": "...", "query_type": "..."}}"""
    return {"parsed_query": call(prompt)}


def doc_retriever(state: dict) -> dict:
    """Return the relevant filing section (deterministic, correct)."""
    return {"retrieved_sections": state["source_document"]}


def table_extractor(state: dict) -> dict:
    prompt = f"""Extract the requested financial data from the table below.

QUERY: {state.get('parsed_query', state['query'])}

FINANCIAL TABLE:
{state.get('retrieved_sections', state['source_document'])}

Instructions:
1. Identify the line item that best matches the requested metric.
2. Extract values for all available years.
3. {INSTRUCTIONS['table_extractor']}
4. Note which specific line item you matched.

Respond with JSON only:
{{"line_item": "...", "values": {{"2018": "...", "2017": "...", "2016": "..."}}, "unit": "millions"}}"""
    return {"extracted_data": call(prompt)}


def context_enricher(state: dict) -> dict:
    prompt = f"""You are a financial analyst adding context to a data extraction.

EXTRACTED DATA:
{state.get('extracted_data', '')}

COMPANY AND METRIC:
{state.get('parsed_query', state['query'])}

{INSTRUCTIONS['context_enricher']}

Output the original extracted data followed by your context paragraph."""
    return {"enriched_context": call(prompt, max_tokens=500)}


def synthesizer(state: dict) -> dict:
    """Generate the answer (Sonnet). Working correctly."""
    qt = "lookup"
    try:
        qt = json.loads(state.get("parsed_query", "{}")).get("query_type", "lookup")
    except Exception:
        pass
    style = ("Provide the exact figure as reported." if qt == "lookup"
             else "Provide your best estimate or approximate figure based on the data.")
    prompt = f"""Answer this financial question based on the provided data.

QUESTION: {state['query']}

EXTRACTED DATA:
{state.get('extracted_data', 'None')}

INDUSTRY CONTEXT:
{state.get('enriched_context', 'None')}

APPROACH: {style}

Write a 2-paragraph professional response. Include the specific figure,
year-over-year context from the data, and industry comparisons if the
context provides them."""
    return {"analysis": call(prompt, model=SONNET, max_tokens=600)}


def fact_checker(state: dict) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {}
    prompt = f"""Review this financial analysis for accuracy against the source.

ANALYSIS:
{analysis}

SOURCE DOCUMENT:
{state['source_document']}

Verification guidelines:
- Check that key financial figures are consistent with the source.
- {INSTRUCTIONS['fact_checker']}
- Focus on whether the analysis would mislead a reader about the figures.

Output the verified analysis with any corrections applied. Preserve structure and tone."""
    return {"analysis": call(prompt, max_tokens=600)}


def tone_editor(state: dict) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {}
    prompt = f"""Edit this financial analysis for a senior executive audience.

ANALYSIS:
{analysis}

Style guidelines:
- Lead with the most important finding.
- {INSTRUCTIONS['tone_editor']}
- Keep any peer or industry comparisons.
- Maximum 2 concise paragraphs.

Output the edited analysis only."""
    return {"analysis": call(prompt, max_tokens=400)}


def output_formatter(state: dict) -> dict:
    """Format the output (does not alter numbers). Working correctly."""
    analysis = state.get("analysis", "")
    if not analysis:
        return {}
    prompt = f"""Format this financial analysis as a clean, structured response.
Do not modify any numbers, claims, or substance.

ANALYSIS:
{analysis}

Add a header with the company name and metric, and "Source: Company 10-K Filing"
at the end. Output the formatted response only."""
    return {"analysis": call(prompt, max_tokens=400)}


_NODES = [
    ("query_parser", query_parser), ("doc_retriever", doc_retriever),
    ("table_extractor", table_extractor), ("context_enricher", context_enricher),
    ("synthesizer", synthesizer), ("fact_checker", fact_checker),
    ("tone_editor", tone_editor), ("output_formatter", output_formatter),
]


def build():
    """Return the compiled 8-agent pipeline (no-arg factory)."""
    g = StateGraph(State)
    for name, fn in _NODES:
        g.add_node(name, fn)
    g.set_entry_point("query_parser")
    edges = [n for n, _ in _NODES]
    for a, b in zip(edges, edges[1:]):
        g.add_edge(a, b)
    g.add_edge("output_formatter", END)
    return g.compile()
