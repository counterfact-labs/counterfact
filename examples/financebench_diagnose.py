"""
FinanceBench Competitive Analysis: 8-Agent Pipeline Diagnosis
==============================================================

8-agent financial RAG pipeline with subtle prompt faults.
Every prompt reads as reasonable — the bugs are in LLM interpretation
of subjective instructions and architectural decisions.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/financebench_competitive.py
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


# ═══════════════════════════════════════════════════════════════════
# SOURCE DATA
# ═══════════════════════════════════════════════════════════════════

CASH_FLOW = """
3M Company and Subsidiaries
Consolidated Statement of Cash Flows
Years ended December 31 (Millions)
                                                    2018        2017        2016
Cash Flows from Operating Activities
  Net income including noncontrolling interest     $5,363      $4,869      $5,058
  Depreciation and amortization                     1,488       1,544       1,474
  Net cash provided by operating activities         6,439       6,240       6,662

Cash Flows from Investing Activities
  Purchases of property, plant and equipment (PP&E) (1,577)     (1,373)     (1,420)
  Proceeds from sale of PP&E and other assets          262          49          58
  Acquisitions, net of cash acquired                    13      (2,023)        (16)

Cash Flows from Financing Activities
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
_tls = threading.local()  # thread-local ground truth


# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════

class State(TypedDict):
    query: str
    source_document: str
    parsed_query: str
    retrieved_sections: str
    extracted_data: str
    enriched_context: str
    analysis: str


# ═══════════════════════════════════════════════════════════════════
# STAGE 1 AGENTS — data preparation
# Every prompt reads as reasonable.
# ═══════════════════════════════════════════════════════════════════

def query_parser(state: State) -> dict:
    """Parse the question. Prompt looks normal.
    SUBTLE BUG: Claude sometimes classifies 'expenditure' questions as
    estimation/forecasting due to training data associations. No explicit
    rule forces this — it's emergent LLM behavior."""
    prompt = f"""Parse this financial question into structured components.

QUESTION: {state['query']}

Extract:
- company: the company name
- metric: what financial metric is being asked about
- period: the fiscal period
- query_type: classify as "lookup" (retrieving a specific reported figure)
  or "estimation" (computing or approximating a figure)

Respond with JSON only:
{{"company": "...", "metric": "...", "period": "...", "query_type": "..."}}"""
    return {"parsed_query": call(prompt)}


def doc_retriever(state: State) -> dict:
    """Returns the correct document section."""
    return {"retrieved_sections": state["source_document"]}


def table_extractor(state: State) -> dict:
    """Extract data. Prompt looks normal.
    SUBTLE BUG: 'clean, readable format' causes Claude to sometimes
    round numbers it considers messy (1,577 -> 1,580 or 1,600).
    Whether it rounds depends on the specific number."""
    prompt = f"""Extract the requested financial data from the table below.

QUERY: {state.get('parsed_query', state['query'])}

FINANCIAL TABLE:
{state.get('retrieved_sections', state['source_document'])}

Instructions:
1. Identify the line item that best matches the requested metric.
2. Extract values for all available years.
3. Present values in a clean, readable format.
4. Note which specific line item you matched.

Respond with JSON only:
{{"line_item": "...", "values": {{"2018": "...", "2017": "...", "2016": "..."}}, "unit": "millions"}}"""
    return {"extracted_data": call(prompt)}


def context_enricher(state: State) -> dict:
    """Add industry context. Prompt looks normal.
    SUBTLE BUG: architectural — asks the LLM to generate industry context
    from its own knowledge rather than a verified database. Claude produces
    plausible but unverifiable benchmarks."""
    prompt = f"""You are a financial analyst adding industry context to
a data extraction.

EXTRACTED DATA:
{state.get('extracted_data', '')}

COMPANY AND METRIC:
{state.get('parsed_query', state['query'])}

Based on your knowledge of the industrial sector, add 2-3 sentences of
relevant context:
- How does this figure compare to industry peers?
- What are typical ranges for this metric in the sector?
- Any notable trends?

Output the original extracted data followed by your context paragraph."""
    return {"enriched_context": call(prompt, max_tokens=500)}


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 AGENTS — progressive refinement of 'analysis'
# ═══════════════════════════════════════════════════════════════════

def synthesizer(state: State) -> dict:
    """Generate the analysis. Working correctly."""
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


def fact_checker(state: State) -> dict:
    """Verify claims. Prompt sounds perfectly reasonable.
    SUBTLE BUGS:
    1. 'minor rounding differences are acceptable' — subjective, Claude
       interprets this loosely for financial data
    2. 'If the key claims check out, respond with the verified analysis' —
       confirmation framing primes Claude to confirm rather than reject"""
    analysis = state.get("analysis", "")
    if not analysis:
        return {}

    prompt = f"""Review this financial analysis for accuracy against the
source document.

ANALYSIS:
{analysis}

SOURCE DOCUMENT:
{state['source_document']}

Verification guidelines:
- Check that key financial figures are consistent with the source
- Minor rounding differences and presentation choices are acceptable
- Focus on material accuracy — would this analysis mislead a reader
  about the company's financial position?
- Industry context and peer comparisons do not need source verification

If the key claims check out, output the verified analysis with any
minor corrections applied. Preserve the overall structure and tone."""
    return {"analysis": call(prompt, max_tokens=600)}


def tone_editor(state: State) -> dict:
    """Edit for executives. Prompt sounds normal.
    SUBTLE BUG: 'simplify large numbers for readability' causes Claude to
    convert millions to billions. 'focus on the headline number' causes it
    to drop YoY comparisons."""
    analysis = state.get("analysis", "")
    if not analysis:
        return {}

    prompt = f"""Edit this financial analysis for a senior executive audience.

ANALYSIS:
{analysis}

Style guidelines:
- Lead with the most important finding
- Simplify large numbers for readability
- Focus on the headline number and strategic implications
- Remove granular year-over-year details unless the change is dramatic
- Keep any peer or industry comparisons
- Maximum 2 concise paragraphs

Output the edited analysis only."""
    return {"analysis": call(prompt, max_tokens=400)}


def output_formatter(state: State) -> dict:
    """Format the output. Working correctly."""
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
# PIPELINE
# ═══════════════════════════════════════════════════════════════════

def build_pipeline():
    g = StateGraph(State)
    for name, fn in [
        ("query_parser", query_parser), ("doc_retriever", doc_retriever),
        ("table_extractor", table_extractor), ("context_enricher", context_enricher),
        ("synthesizer", synthesizer), ("fact_checker", fact_checker),
        ("tone_editor", tone_editor), ("output_formatter", output_formatter),
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


# ═══════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════

def accuracy_clf(query, output, sources) -> ClassifierResult:
    if not output or len(output) < 10:
        return ClassifierResult(name="accuracy", score=0.1,
                                reasoning="No output", weight=2.0)
    prompt = f"""Does this contain the correct answer?

CORRECT: {_tls.gt}
OUTPUT: {output}

Score:
- 1.0 = exact figure ({_tls.gt})
- 0.5 = close/rounded (within ~5% but not exact)
- 0.0 = wrong figure or no figure

JSON only: {{"score": 0.0, "reasoning": "..."}}"""
    try:
        r = call(prompt, max_tokens=100)
        m = re.search(r'\{[^}]+\}', r, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return ClassifierResult(name="accuracy", score=float(d["score"]),
                                    reasoning=d.get("reasoning", ""), weight=2.0)
    except Exception:
        pass
    return ClassifierResult(name="accuracy", score=0.5, reasoning="error", weight=2.0)


def precision_clf(query, output, sources) -> ClassifierResult:
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
                                    reasoning=f"Rounded", weight=1.5)
    return ClassifierResult(name="precision", score=0.3,
                            reasoning="Unclear precision", weight=1.5)


def grounding_clf(query, output, sources) -> ClassifierResult:
    if not output or len(output) < 10:
        return ClassifierResult(name="grounding", score=0.5,
                                reasoning="No output", weight=2.0)
    prompt = f"""Check if this financial analysis contains claims not in the source.

SOURCE DOCUMENT:
{sources}

OUTPUT:
{output}

Check specifically for:
- Industry benchmarks or peer comparisons (NOT in the source)
- Projected or forecasted figures (NOT in the source)
- Rounded figures that differ from source values

Score:
- 1.0 = every specific claim traces to the source
- 0.5 = core answer from source but includes unsourced context
- 0.0 = contains fabricated or unverifiable figures

JSON only: {{"score": 0.0, "reasoning": "..."}}"""
    try:
        r = call(prompt, max_tokens=150)
        m = re.search(r'\{[^}]+\}', r, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return ClassifierResult(name="grounding", score=float(d["score"]),
                                    reasoning=d.get("reasoning", ""), weight=2.0)
    except Exception:
        pass
    return ClassifierResult(name="grounding", score=0.5, reasoning="error", weight=2.0)


# ═══════════════════════════════════════════════════════════════════
# MAIN — 5-query competitive analysis
# ═══════════════════════════════════════════════════════════════════

def print_report(report):
    print(f"\n  Baseline Quality: {report.baseline_quality:.3f}")
    sv = sorted(report.shapley_values.items(), key=lambda x: x[1])
    print("  Agent                  Shapley    95% CI")
    print("  " + "─" * 55)
    for a, v in sv:
        ci = report.shapley_cis.get(a)
        cs = f"[{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]" if ci and ci.n_samples >= 2 else "—"
        print(f"  {a:22s} {'+' if v>=0 else ''}{v:.3f}  {cs:>22s}  {'█'*int(abs(v)*20)}")
    if report.per_classifier_shapley:
        print("  Per-Classifier:")
        for clf, avs in report.per_classifier_shapley.items():
            items = sorted(avs.items(), key=lambda x: x[1])
            s = ", ".join(f"{a}={v:+.3f}" for a, v in items[:2])
            s += " ... " + ", ".join(f"{a}={v:+.3f}" for a, v in items[-2:])
            print(f"    {clf:16s}: {s}")
    cls = report.classification
    print(f"  Failure: {cls.failure_type} ({cls.confidence:.0%})")
    if cls.dominant_agent:
        print(f"  Dominant: {cls.dominant_agent}")
    if report.recommendations:
        for i, r in enumerate(report.recommendations, 1):
            print(f"  Rec {i}: [{r.intervention_type}] {r.target_agent}: {r.description}")


def run_query(qi, q, rdir):
    """Run a single query diagnosis. Thread-safe."""
    _tls.gt = q["ground_truth"]

    pipeline = build_pipeline()
    reg = ClassifierRegistry()
    reg.register(accuracy_clf, "financebench")
    reg.register(precision_clf, "financebench")
    reg.register(grounding_clf, "financebench")

    init: State = {
        "query": q["query"], "source_document": CASH_FLOW,
        "parsed_query": "", "retrieved_sections": "",
        "extracted_data": "", "enriched_context": "", "analysis": "",
    }

    print(f"\n  [{q['short']}] Starting baseline...")
    result = pipeline.invoke(init)
    print(f"  [{q['short']}] Output: {result['analysis'][:100]}...")
    print(f"  [{q['short']}] Ground truth: {q['ground_truth']}")

    print(f"  [{q['short']}] Running diagnosis (30 simulations)...")
    report = pipeline.diagnose(
        input_state=init, domain="financebench",
        num_simulations=30, registry=reg, seed=42,
    )

    # Print per-query results
    sv = sorted(report.shapley_values.items(), key=lambda x: x[1])
    worst = sv[0]
    print(f"  [{q['short']}] Done. Baseline={report.baseline_quality:.3f}, "
          f"worst={worst[0]}({worst[1]:+.3f})")

    report.to_json(os.path.join(rdir, f"q{qi}_{q['short'].lower()}.json"))
    return q, report


def main():
    print("=" * 72)
    print(f"  COUNTERFACT COMPETITIVE ANALYSIS — {len(QUERIES)} queries (parallel)")
    print("=" * 72)

    try:
        call("Say OK.", max_tokens=5)
        print(f"\n  ✓ API connected ({SONNET} + {HAIKU})")
    except Exception as e:
        print(f"\n  ERROR: {e}"); sys.exit(1)

    rdir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(rdir, exist_ok=True)

    reports = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(run_query, qi, q, rdir): q
            for qi, q in enumerate(QUERIES, 1)
        }
        for future in as_completed(futures):
            try:
                q, report = future.result()
                reports.append((q, report))
            except Exception as e:
                print(f"  ERROR: {futures[future]['short']}: {e}")

    # Sort by original query order
    order = {q["short"]: i for i, q in enumerate(QUERIES)}
    reports.sort(key=lambda x: order[x[0]["short"]])

    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print("  AGGREGATE RESULTS")
    print(f"{'═'*72}")
    print(f"\n  Queries: {len(reports)}")
    print(f"  Avg baseline: {sum(r.baseline_quality for _,r in reports)/len(reports):.3f}")

    all_shapley = {}
    all_per_clf = {}
    for _, report in reports:
        for agent, val in report.shapley_values.items():
            all_shapley.setdefault(agent, []).append(val)
        if report.per_classifier_shapley:
            for clf, avs in report.per_classifier_shapley.items():
                for agent, val in avs.items():
                    all_per_clf.setdefault(clf, {}).setdefault(agent, []).append(val)

    avg_sv = {a: sum(vs)/len(vs) for a, vs in all_shapley.items()}
    sv = sorted(avg_sv.items(), key=lambda x: x[1])
    print("\n  Agent                  Avg Shapley")
    print("  " + "─" * 45)
    for a, v in sv:
        print(f"  {a:22s} {'+' if v>=0 else ''}{v:.3f}  {'█'*int(abs(v)*20)}")

    if all_per_clf:
        print("\n  Avg Per-Classifier:")
        for clf, avs in all_per_clf.items():
            avgs = sorted({a: sum(vs)/len(vs) for a,vs in avs.items()}.items(), key=lambda x: x[1])
            s = ", ".join(f"{a}={v:+.3f}" for a,v in avgs[:2])
            s += " ... " + ", ".join(f"{a}={v:+.3f}" for a,v in avgs[-2:])
            print(f"    {clf:16s}: {s}")

    print("\n  Per-Query Summary:")
    print(f"  {'Query':12s} {'Baseline':>8s} {'Worst Agent':>28s} {'Best Agent':>28s}")
    print("  " + "─" * 76)
    for q, r in reports:
        sv = sorted(r.shapley_values.items(), key=lambda x: x[1])
        print(f"  {q['short']:12s} {r.baseline_quality:8.3f} {sv[0][0]+'='+f'{sv[0][1]:+.3f}':>28s} {sv[-1][0]+'='+f'{sv[-1][1]:+.3f}':>28s}")

    # Per-query detail
    for q, r in reports:
        print(f"\n  ── {q['short']} ──")
        print_report(r)

    print(f"\n  Reports: {os.path.abspath(rdir)}/")
    print("═" * 72)


if __name__ == "__main__":
    main()
