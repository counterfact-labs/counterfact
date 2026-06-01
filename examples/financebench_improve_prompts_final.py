"""
FinanceBench Step 4: Fix table extractor (final step)
=====================================================

Runs all previous fixes + table extractor fix.
Uses the same cache and evaluation as the incremental script.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/financebench_step4.py
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

from counterfact import StateGraph, END
from counterfact.classifiers import ClassifierRegistry
from counterfact.types import ClassifierResult

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("ERROR: Set ANTHROPIC_API_KEY"); sys.exit(1)
        import anthropic
        _client = anthropic.Anthropic(api_key=key)
    return _client


def call(prompt, model=HAIKU, temperature=0.0, max_tokens=1000):
    key = (prompt, model, temperature, max_tokens)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    r = get_client().messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    text = r.content[0].text
    with _cache_lock:
        _cache[key] = text
    return text

_cache = {}
_cache_lock = threading.Lock()

CASH_FLOW = """3M Company and Subsidiaries
Consolidated Statement of Cash Flows
Years ended December 31 (Millions)
                                                    2018        2017        2016
Cash Flows from Operating Activities
  Net income including noncontrolling interest     $5,363      $4,869      $5,058
  Depreciation and amortization                     1,488       1,544       1,474
  Stock-based compensation expense                    282         324         297
  Gain on sale of businesses                         (545)       (586)       (111)
  Changes in assets and liabilities
    Accounts receivable                              (285)       (402)        (68)
    Inventories                                      (509)       (387)       (174)
    Accounts payable                                   408         255          32
    Accrued income taxes                               149         (43)          3
    Other net                                           88         (35)        (72)
  Net cash provided by operating activities         6,439       5,539       5,939

Cash Flows from Investing Activities
  Purchases of property, plant and equipment (PP&E) (1,577)     (1,373)     (1,420)
  Proceeds from sale of PP&E and other assets           75         549         118
  Acquisitions, net of cash acquired                  (169)     (2,024)       (16)
  Purchases of marketable securities                (2,152)     (1,985)     (1,410)
  Net cash used in investing activities             (3,823)     (4,833)     (2,728)

Cash Flows from Financing Activities
  Change in short-term debt                           (869)      1,338        (478)
  Repayment of debt                                 (1,034)     (1,625)       (512)
  Repurchases of common stock                       (4,870)    (2,068)     (3,753)
  Dividends paid                                    (3,193)    (2,803)     (2,678)
"""

QUERIES = [
    {"query": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Use the cash flow statement.",
     "ground_truth": "$1,577 million", "short": "CapEx"},
    {"query": "What was 3M's depreciation and amortization expense in FY2018 (in USD millions)?",
     "ground_truth": "$1,488 million", "short": "D&A"},
    {"query": "What was 3M's net cash provided by operating activities in FY2018 (in USD millions)?",
     "ground_truth": "$6,439 million", "short": "OpCF"},
    {"query": "How much did 3M pay in dividends in FY2018 (in USD millions)?",
     "ground_truth": "$3,193 million", "short": "Dividends"},
    {"query": "What was 3M's total stock repurchase amount in FY2018 (in USD millions)?",
     "ground_truth": "$4,870 million", "short": "Buybacks"},
]
_tls = threading.local()


class State(TypedDict):
    query: str
    source_document: str
    parsed_query: str
    retrieved_sections: str
    extracted_data: str
    enriched_context: str
    analysis: str


# ═══════════════════════════════════════════════════════════════════
# ALL AGENTS — all fixed versions
# ═══════════════════════════════════════════════════════════════════

def query_parser(state: State) -> dict:
    prompt = f"""Parse this financial question into structured components.

QUESTION: {state['query']}

Extract:
- company: the company name
- metric: what financial metric is being asked about
- period: the fiscal period
- query_type: classify as "lookup" (retrieving a specific reported figure)
  or "estimation" (computing/approximating a figure not directly reported)

Respond with JSON only:
{{"company": "...", "metric": "...", "period": "...", "query_type": "..."}}"""
    return {"parsed_query": call(prompt)}


def doc_retriever(state: State) -> dict:
    return {"retrieved_sections": state["source_document"]}


def table_extractor_fixed(state: State) -> dict:
    """FIXED: 'Present exact values as reported' instead of 'clean, readable format'"""
    prompt = f"""Extract the requested financial data from the table below.

QUERY: {state.get('parsed_query', state['query'])}

FINANCIAL TABLE:
{state.get('retrieved_sections', state['source_document'])}

Instructions:
1. Identify the line item that best matches the requested metric.
2. Extract values for all available years.
3. Present exact values as reported in the source document. Do NOT round.
4. Note which specific line item you matched.

Respond with JSON only:
{{"line_item": "...", "values": {{"2018": "...", "2017": "...", "2016": "..."}}, "unit": "millions"}}"""
    return {"extracted_data": call(prompt)}


def context_enricher_fixed(state: State) -> dict:
    prompt = f"""You are a financial analyst adding context to a data extraction.

EXTRACTED DATA:
{state.get('extracted_data', '')}

COMPANY AND METRIC:
{state.get('parsed_query', state['query'])}

Using ONLY the data provided above and in the source document, add 2-3
sentences of context:
- Year-over-year trends visible in the extracted data
- Relationship to other figures in the source document
- Do NOT add industry benchmarks, peer comparisons, or any data not
  explicitly present in the source

Output the original extracted data followed by your context paragraph."""
    return {"enriched_context": call(prompt, max_tokens=500)}


def synthesizer(state: State) -> dict:
    qt = "lookup"
    try:
        qt = json.loads(state.get("parsed_query", "{}")).get("query_type", "lookup")
    except Exception:
        pass
    style = ("Provide the exact figure as reported." if qt == "lookup" else
             "Provide your best estimate or approximate figure based on the data.")
    prompt = f"""Answer this financial question based on the provided data.

QUESTION: {state['query']}

EXTRACTED DATA:
{state.get('extracted_data', 'None')}

INDUSTRY CONTEXT:
{state.get('enriched_context', 'None')}

APPROACH: {style}

Write a 2-paragraph professional response. Include the specific figure,
year-over-year context from the data, and industry comparisons if
the context provides them."""
    return {"analysis": call(prompt, model=SONNET, max_tokens=600)}


def fact_checker_fixed(state: State) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {}
    prompt = f"""Review this financial analysis for accuracy against the
source document.

ANALYSIS:
{analysis}

SOURCE DOCUMENT:
{state['source_document']}

Verification rules:
- Identify any discrepancies between the analysis and the source
- Flag any dollar figure that differs from the source by more than $1 million
- Flag any claim, benchmark, or comparison not explicitly present in the
  source document
- Numbers must match the source exactly — do not accept rounded versions

List each discrepancy found. Then output the corrected analysis with
all figures matching the source document exactly."""
    return {"analysis": call(prompt, max_tokens=600)}


def tone_editor_fixed(state: State) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {}
    prompt = f"""Edit this financial analysis for a senior executive audience.

ANALYSIS:
{analysis}

Style guidelines:
- Lead with the most important finding
- Preserve all exact figures and their original units (do NOT convert
  between millions and billions)
- Focus on strategic implications
- Keep year-over-year comparisons that show meaningful trends
- Maximum 2 concise paragraphs

Output the edited analysis only."""
    return {"analysis": call(prompt, max_tokens=400)}


def output_formatter(state: State) -> dict:
    analysis = state.get("analysis", "")
    if not analysis:
        return {}
    prompt = f"""Format this financial analysis as a clean, structured response.
Do not modify any numbers, claims, or substance.

ANALYSIS:
{analysis}

Add:
- A header with the company name and metric
- "Source: Company 10-K Filing" at the end

Output the formatted response only."""
    return {"analysis": call(prompt, max_tokens=400)}


# ═══════════════════════════════════════════════════════════════════
# PIPELINE + CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════

def build_pipeline():
    g = StateGraph(State)
    for name, fn in [
        ("query_parser", query_parser), ("doc_retriever", doc_retriever),
        ("table_extractor", table_extractor_fixed),
        ("context_enricher", context_enricher_fixed),
        ("synthesizer", synthesizer), ("fact_checker", fact_checker_fixed),
        ("tone_editor", tone_editor_fixed), ("output_formatter", output_formatter),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("query_parser")
    g.add_edge("query_parser", "doc_retriever")
    g.add_edge("doc_retriever", "table_extractor")
    g.add_edge("table_extractor", "context_enricher")
    g.add_edge("context_enricher", "synthesizer")
    g.add_edge("synthesizer", "fact_checker")
    g.add_edge("fact_checker", "tone_editor")
    g.add_edge("tone_editor", "output_formatter")
    g.add_edge("output_formatter", END)
    return g.compile()


def accuracy_clf(query, output, sources):
    if not output or len(output) < 10:
        return ClassifierResult(name="accuracy", score=0.0,
                                reasoning="No output", weight=2.0)
    prompt = f"""Does this contain the correct answer?

CORRECT: {_tls.gt}
OUTPUT: {output}

Score:
- 1.0 = exact figure ({_tls.gt})
- 0.5 = close/rounded (within ~5% but not exact)
- 0.0 = wrong figure or no figure

Respond with just the number."""
    try:
        return ClassifierResult(name="accuracy",
                                score=float(call(prompt, max_tokens=5).strip()),
                                reasoning="LLM judge", weight=2.0)
    except Exception:
        return ClassifierResult(name="accuracy", score=0.0,
                                reasoning="Parse error", weight=2.0)


def precision_clf(query, output, sources):
    if not output or len(output) < 10:
        return ClassifierResult(name="precision", score=0.1,
                                reasoning="No output", weight=1.5)
    gt_match = re.search(r'[\d,]+', _tls.gt)
    gt_num = gt_match.group() if gt_match else ""
    gt_plain = gt_num.replace(',', '')
    if gt_num in output or gt_plain in output:
        return ClassifierResult(name="precision", score=1.0,
                                reasoning=f"Exact {_tls.gt}", weight=1.5)
    if "billion" in output.lower():
        return ClassifierResult(name="precision", score=0.2,
                                reasoning="Converted to billions", weight=1.5)
    if gt_plain:
        r100 = f"{round(int(gt_plain), -2):,}"
        r10 = f"{round(int(gt_plain), -1):,}"
        if r100 in output or r10 in output:
            return ClassifierResult(name="precision", score=0.4,
                                    reasoning="Rounded", weight=1.5)
    return ClassifierResult(name="precision", score=0.3,
                            reasoning="Unclear precision", weight=1.5)


def grounding_clf(query, output, sources):
    if not output or len(output) < 10:
        return ClassifierResult(name="grounding", score=0.1,
                                reasoning="No output", weight=1.0)
    prompt = f"""Does this analysis contain ONLY claims that are supported
by the source document?

OUTPUT: {output}
SOURCE: {sources[0] if sources else 'None'}

Check for:
- Industry benchmarks or peer comparisons not in the source
- Fabricated statistics or percentages
- Projections or estimates not derived from the source data

Score:
- 1.0 = all claims traceable to source
- 0.5 = core answer sourced but includes unsourced context
- 0.0 = contains fabricated figures

Respond with just the number."""
    try:
        return ClassifierResult(name="grounding",
                                score=float(call(prompt, max_tokens=5).strip()),
                                reasoning="LLM judge", weight=1.0)
    except Exception:
        return ClassifierResult(name="grounding", score=0.0,
                                reasoning="Parse error", weight=1.0)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  STEP 4: All fixes + table extractor")
    print("=" * 72)

    try:
        call("Say OK.", max_tokens=5)
        print(f"\n  ✓ API connected ({SONNET} + {HAIKU})")
    except Exception as e:
        print(f"\n  ERROR: {e}"); sys.exit(1)

    pipeline = build_pipeline()
    reg = ClassifierRegistry()
    reg.register(accuracy_clf, "financebench")
    reg.register(precision_clf, "financebench")
    reg.register(grounding_clf, "financebench")

    def run_one(q):
        _tls.gt = q["ground_truth"]
        init: State = {
            "query": q["query"], "source_document": CASH_FLOW,
            "parsed_query": "", "retrieved_sections": "",
            "extracted_data": "", "enriched_context": "", "analysis": "",
        }
        result = pipeline.invoke(init)
        report = pipeline.diagnose(
            input_state=init, domain="financebench",
            num_simulations=30, registry=reg, seed=42,
        )
        return q, result, report

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(run_one, q): q for q in QUERIES}
        for future in as_completed(futures):
            try:
                q, result, report = future.result()
                print(f"  [{q['short']}] baseline={report.baseline_quality:.3f}  "
                      f"output: {result['analysis'][:80].replace(chr(10),' ')}...")
                results.append((q, result, report))
            except Exception as e:
                print(f"  ERROR: {e}")

    order = {q["short"]: i for i, q in enumerate(QUERIES)}
    results.sort(key=lambda x: order[x[0]["short"]])

    baselines = [r.baseline_quality for _, _, r in results]
    avg = sum(baselines) / len(baselines)
    exact = sum(1 for _, _, r in results if r.baseline_quality > 0.7)

    print(f"\n{'═'*72}")
    print(f"  STEP 4 RESULTS")
    print(f"{'═'*72}")
    print(f"\n  Avg Quality: {avg:.3f}")
    print(f"  Exact answers: {exact}/5")

    print(f"\n  {'Query':12s} {'Quality':>8s}")
    print(f"  {'─'*30}")
    for q, _, report in results:
        print(f"  {q['short']:12s} {report.baseline_quality:8.3f}")

    all_sv = {}
    for _, _, report in results:
        for agent, val in report.shapley_values.items():
            all_sv.setdefault(agent, []).append(val)
    avg_sv = sorted({a: sum(vs)/len(vs) for a, vs in all_sv.items()}.items(),
                    key=lambda x: x[1])
    print(f"\n  Agent                  Avg Shapley")
    print(f"  {'─'*45}")
    for a, v in avg_sv:
        print(f"  {a:22s} {'+' if v >= 0 else ''}{v:.3f}")

    print(f"\n{'═'*72}")


if __name__ == "__main__":
    main()
