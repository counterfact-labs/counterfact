"""
FinanceBench Incremental Fix: Iterative Debug Cycle
====================================================

Demonstrates an interactive debugging workflow:
  Step 0: Evaluate the broken pipeline (baseline)
  Step 1: Fix context enricher (worst aggregate Shapley: -0.145)
  Step 2: Fix tone editor (worst precision: -0.132)
  Step 3: Fix fact checker (grounding issues)
  Summary: Compare quality at each step

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/financebench_incremental.py
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
# AGENTS — unchanged across all steps
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


def table_extractor(state: State) -> dict:
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
# AGENTS — with broken/fixed variants
# ═══════════════════════════════════════════════════════════════════

# --- Context Enricher ---
def context_enricher_broken(state: State) -> dict:
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


# --- Fact Checker ---
def fact_checker_broken(state: State) -> dict:
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


# --- Tone Editor ---
def tone_editor_broken(state: State) -> dict:
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


# ═══════════════════════════════════════════════════════════════════
# PIPELINE BUILDER — accepts agent variants
# ═══════════════════════════════════════════════════════════════════

def build_pipeline(ce_fn, fc_fn, te_fn):
    g = StateGraph(State)
    for name, fn in [
        ("query_parser", query_parser), ("doc_retriever", doc_retriever),
        ("table_extractor", table_extractor), ("context_enricher", ce_fn),
        ("synthesizer", synthesizer), ("fact_checker", fc_fn),
        ("tone_editor", te_fn), ("output_formatter", output_formatter),
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
# EVALUATION — run all 5 queries in parallel, return avg quality
# ═══════════════════════════════════════════════════════════════════

def evaluate_step(step_name, ce_fn, fc_fn, te_fn):
    """Run pipeline on all 5 queries and return avg baseline quality."""
    pipeline = build_pipeline(ce_fn, fc_fn, te_fn)
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
                results.append(future.result())
            except Exception as e:
                print(f"    ERROR: {e}")

    # Sort by original order
    order = {q["short"]: i for i, q in enumerate(QUERIES)}
    results.sort(key=lambda x: order[x[0]["short"]])

    # Print summary
    baselines = [r.baseline_quality for _, _, r in results]
    avg = sum(baselines) / len(baselines)

    print(f"\n  {'Query':12s} {'Quality':>8s}  {'Output (first 80 chars)'}")
    print(f"  {'─'*80}")
    for q, result, report in results:
        out = result["analysis"][:80].replace('\n', ' ')
        print(f"  {q['short']:12s} {report.baseline_quality:8.3f}  {out}...")

    # Aggregate Shapley
    all_sv = {}
    for _, _, report in results:
        for agent, val in report.shapley_values.items():
            all_sv.setdefault(agent, []).append(val)
    avg_sv = sorted({a: sum(vs)/len(vs) for a, vs in all_sv.items()}.items(),
                    key=lambda x: x[1])

    print(f"\n  Avg Quality: {avg:.3f}")
    print(f"  Exact answers: {sum(1 for _,_,r in results if r.baseline_quality > 0.7)}/5")
    print(f"\n  Agent                  Avg Shapley")
    print(f"  {'─'*45}")
    for a, v in avg_sv:
        bar = '█' * int(abs(v) * 20)
        print(f"  {a:22s} {'+' if v >= 0 else ''}{v:.3f}  {bar}")

    # Aggregate per-classifier
    all_per_clf = {}
    for _, _, report in results:
        if report.per_classifier_shapley:
            for clf, avs in report.per_classifier_shapley.items():
                for agent, val in avs.items():
                    all_per_clf.setdefault(clf, {}).setdefault(agent, []).append(val)
    if all_per_clf:
        print(f"\n  Per-Classifier (worst agent):")
        for clf, avs in all_per_clf.items():
            avgs = sorted({a: sum(vs)/len(vs) for a, vs in avs.items()}.items(),
                          key=lambda x: x[1])
            print(f"    {clf:12s}: {avgs[0][0]}={avgs[0][1]:+.3f}")

    return avg, results


# ═══════════════════════════════════════════════════════════════════
# MAIN — incremental fix cycle
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  COUNTERFACT — INCREMENTAL DEBUG CYCLE")
    print("=" * 72)

    try:
        call("Say OK.", max_tokens=5)
        print(f"\n  ✓ API connected ({SONNET} + {HAIKU})")
    except Exception as e:
        print(f"\n  ERROR: {e}"); sys.exit(1)

    steps = [
        ("Step 0: Broken pipeline (baseline)",
         context_enricher_broken, fact_checker_broken, tone_editor_broken),
        ("Step 1: Fix context enricher (worst agent, Shapley: -0.145)",
         context_enricher_fixed, fact_checker_broken, tone_editor_broken),
        ("Step 2: + Fix tone editor (worst precision, Shapley: -0.132)",
         context_enricher_fixed, fact_checker_broken, tone_editor_fixed),
        ("Step 3: + Fix fact checker (grounding issues)",
         context_enricher_fixed, fact_checker_fixed, tone_editor_fixed),
    ]

    step_results = []
    for label, ce_fn, fc_fn, te_fn in steps:
        print(f"\n{'═'*72}")
        print(f"  {label}")
        print(f"{'═'*72}")
        avg, results = evaluate_step(label, ce_fn, fc_fn, te_fn)
        step_results.append((label, avg, results))

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  INCREMENTAL IMPROVEMENT SUMMARY")
    print(f"{'═'*72}")
    print(f"\n  {'Step':55s} {'Quality':>8s}  {'Exact':>6s}  {'Delta':>8s}")
    print(f"  {'─'*80}")
    prev = None
    for label, avg, results in step_results:
        exact = sum(1 for _, _, r in results if r.baseline_quality > 0.7)
        delta = f"+{avg - prev:.3f}" if prev is not None else "—"
        print(f"  {label:55s} {avg:8.3f}  {exact:>4d}/5  {delta:>8s}")
        prev = avg

    total_delta = step_results[-1][1] - step_results[0][1]
    print(f"\n  Total improvement: {step_results[0][1]:.3f} → {step_results[-1][1]:.3f}"
          f" (+{total_delta:.3f}, +{total_delta/step_results[0][1]*100:.0f}%)")

    # Save
    rdir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(rdir, exist_ok=True)
    summary = {
        "steps": [
            {"label": label, "avg_quality": avg,
             "exact_answers": sum(1 for _, _, r in results if r.baseline_quality > 0.7),
             "per_query": {q["short"]: r.baseline_quality for q, _, r in results}}
            for label, avg, results in step_results
        ]
    }
    with open(os.path.join(rdir, "incremental_fixes.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Report: {os.path.abspath(os.path.join(rdir, 'incremental_fixes.json'))}")
    print("═" * 72)


if __name__ == "__main__":
    main()
