"""
LLM-as-Debugger Baseline: Can an LLM diagnose pipeline failures from traces?
==============================================================================

Runs the broken pipeline on all 5 queries, collects full traces and
classifier scores, then asks Claude to diagnose which agents to fix.

Compares the LLM's diagnosis against the Shapley-based attribution.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/financebench_llm_debugger.py
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

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


class State(TypedDict):
    query: str
    source_document: str
    parsed_query: str
    retrieved_sections: str
    extracted_data: str
    enriched_context: str
    analysis: str


# ═══════════════════════════════════════════════════════════════════
# AGENTS (broken versions, with trace capture)
# ═══════════════════════════════════════════════════════════════════

def run_pipeline_with_trace(query_info):
    """Run the broken pipeline and capture each agent's prompt and output."""
    state: State = {
        "query": query_info["query"], "source_document": CASH_FLOW,
        "parsed_query": "", "retrieved_sections": "",
        "extracted_data": "", "enriched_context": "", "analysis": "",
    }
    trace = []

    # 1. Query Parser
    prompt1 = f"""Parse this financial question into structured components.

QUESTION: {state['query']}

Extract:
- company: the company name
- metric: what financial metric is being asked about
- period: the fiscal period
- query_type: classify as "lookup" or "estimation"

Respond with JSON only:
{{"company": "...", "metric": "...", "period": "...", "query_type": "..."}}"""
    out1 = call(prompt1)
    state["parsed_query"] = out1
    trace.append({"agent": "query_parser", "prompt": prompt1, "output": out1})

    # 2. Doc Retriever
    state["retrieved_sections"] = state["source_document"]
    trace.append({"agent": "doc_retriever", "prompt": "(returns source document)", "output": "(cash flow statement)"})

    # 3. Table Extractor
    prompt3 = f"""Extract the requested financial data from the table below.

QUERY: {state['parsed_query']}

FINANCIAL TABLE:
{state['retrieved_sections']}

Instructions:
1. Identify the line item that best matches the requested metric.
2. Extract values for all available years.
3. Present values in a clean, readable format.
4. Note which specific line item you matched.

Respond with JSON only:
{{"line_item": "...", "values": {{"2018": "...", "2017": "...", "2016": "..."}}, "unit": "millions"}}"""
    out3 = call(prompt3)
    state["extracted_data"] = out3
    trace.append({"agent": "table_extractor", "prompt": prompt3, "output": out3})

    # 4. Context Enricher
    prompt4 = f"""You are a financial analyst adding industry context to a data extraction.

EXTRACTED DATA:
{state['extracted_data']}

COMPANY AND METRIC:
{state['parsed_query']}

Based on your knowledge of the industrial sector, add 2-3 sentences of
relevant context:
- How does this figure compare to industry peers?
- What are typical ranges for this metric in the sector?
- Any notable trends?

Output the original extracted data followed by your context paragraph."""
    out4 = call(prompt4, max_tokens=500)
    state["enriched_context"] = out4
    trace.append({"agent": "context_enricher", "prompt": prompt4, "output": out4})

    # 5. Synthesizer
    qt = "lookup"
    try:
        qt = json.loads(state.get("parsed_query", "{}")).get("query_type", "lookup")
    except Exception:
        pass
    style = ("Provide the exact figure as reported." if qt == "lookup" else
             "Provide your best estimate or approximate figure based on the data.")
    prompt5 = f"""Answer this financial question based on the provided data.

QUESTION: {state['query']}

EXTRACTED DATA:
{state['extracted_data']}

INDUSTRY CONTEXT:
{state['enriched_context']}

APPROACH: {style}

Write a 2-paragraph professional response."""
    out5 = call(prompt5, model=SONNET, max_tokens=600)
    state["analysis"] = out5
    trace.append({"agent": "synthesizer", "prompt": prompt5, "output": out5})

    # 6. Fact Checker
    prompt6 = f"""Review this financial analysis for accuracy against the source document.

ANALYSIS:
{state['analysis']}

SOURCE DOCUMENT:
{state['source_document']}

Verification guidelines:
- Check that key financial figures are consistent with the source
- Minor rounding differences and presentation choices are acceptable
- Focus on material accuracy
- Industry context and peer comparisons do not need source verification

If the key claims check out, output the verified analysis with any
minor corrections applied."""
    out6 = call(prompt6, max_tokens=600)
    state["analysis"] = out6
    trace.append({"agent": "fact_checker", "prompt": prompt6, "output": out6})

    # 7. Tone Editor
    prompt7 = f"""Edit this financial analysis for a senior executive audience.

ANALYSIS:
{state['analysis']}

Style guidelines:
- Lead with the most important finding
- Simplify large numbers for readability
- Focus on the headline number and strategic implications
- Remove granular year-over-year details unless the change is dramatic
- Maximum 2 concise paragraphs

Output the edited analysis only."""
    out7 = call(prompt7, max_tokens=400)
    state["analysis"] = out7
    trace.append({"agent": "tone_editor", "prompt": prompt7, "output": out7})

    # 8. Output Formatter
    prompt8 = f"""Format this financial analysis as a clean, structured response.
Do not modify any numbers, claims, or substance.

ANALYSIS:
{state['analysis']}

Add:
- A header with the company name and metric
- "Source: Company 10-K Filing" at the end

Output the formatted response only."""
    out8 = call(prompt8, max_tokens=400)
    state["analysis"] = out8
    trace.append({"agent": "output_formatter", "prompt": prompt8, "output": out8})

    # Classify
    gt = query_info["ground_truth"]
    gt_match = re.search(r'[\d,]+', gt)
    gt_num = gt_match.group() if gt_match else ""
    gt_plain = gt_num.replace(',', '')
    final = state["analysis"]

    # Accuracy
    acc_prompt = f"""Does this contain the correct answer?
CORRECT: {gt}
OUTPUT: {final}
Score: 1.0 = exact, 0.5 = close/rounded, 0.0 = wrong.
Respond with just the number."""
    try:
        acc = float(call(acc_prompt, max_tokens=5).strip())
    except Exception:
        acc = 0.0

    # Precision
    if gt_num in final or gt_plain in final:
        prec = 1.0
        prec_reason = "exact"
    elif "billion" in final.lower():
        prec = 0.2
        prec_reason = "converted to billions"
    else:
        prec = 0.3
        prec_reason = "unclear"

    # Grounding
    gnd_prompt = f"""Does this analysis contain ONLY claims supported by the source?
OUTPUT: {final}
SOURCE: {CASH_FLOW}
Score: 1.0 = all sourced, 0.5 = core answer sourced but unsourced context, 0.0 = fabricated.
Respond with just the number."""
    try:
        gnd = float(call(gnd_prompt, max_tokens=5).strip())
    except Exception:
        gnd = 0.0

    scores = {"accuracy": acc, "precision": prec, "grounding": gnd}

    return query_info, state, trace, scores


def main():
    print("=" * 72)
    print("  LLM-AS-DEBUGGER BASELINE")
    print("  Can an LLM diagnose pipeline failures from traces alone?")
    print("=" * 72)

    try:
        call("Say OK.", max_tokens=5)
        print(f"\n  ✓ API connected ({SONNET} + {HAIKU})")
    except Exception as e:
        print(f"\n  ERROR: {e}"); sys.exit(1)

    # Run all 5 queries in parallel
    print("\n  Running broken pipeline on 5 queries...")
    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(run_pipeline_with_trace, q): q for q in QUERIES}
        for future in as_completed(futures):
            try:
                q, state, trace, scores = future.result()
                print(f"  [{q['short']}] acc={scores['accuracy']:.1f} prec={scores['precision']:.1f} gnd={scores['grounding']:.1f}")
                results.append((q, state, trace, scores))
            except Exception as e:
                print(f"  ERROR: {e}")

    # Sort by original order
    order = {q["short"]: i for i, q in enumerate(QUERIES)}
    results.sort(key=lambda x: order[x[0]["short"]])

    # Build the mega-prompt with all traces
    trace_text = ""
    for q, state, trace, scores in results:
        trace_text += f"\n{'='*60}\n"
        trace_text += f"QUERY: {q['query']}\n"
        trace_text += f"CORRECT ANSWER: {q['ground_truth']}\n"
        trace_text += f"PIPELINE OUTPUT: {state['analysis'][:200]}\n"
        trace_text += f"SCORES: accuracy={scores['accuracy']}, precision={scores['precision']}, grounding={scores['grounding']}\n"
        trace_text += f"\nAGENT TRACE:\n"
        for step in trace:
            trace_text += f"\n--- {step['agent']} ---\n"
            trace_text += f"Prompt (key instruction): {step['prompt'][:300]}\n"
            trace_text += f"Output: {step['output'][:300]}\n"

    diagnosis_prompt = f"""You are debugging a multi-agent pipeline that produces wrong answers.

Below are the full execution traces for 5 queries through an 8-agent pipeline.
Each query has a correct answer, the pipeline's output, classifier scores
(accuracy, precision, grounding on 0-1 scale), and the prompt/output for each agent.

Your task:
1. Identify which agents are causing the failures
2. Rank them by severity (most harmful first)
3. For each, explain what specific prompt instruction is causing the issue
4. Suggest the exact prompt fix

Be specific: name the agent, quote the problematic instruction, and state
which quality dimension (accuracy, precision, or grounding) it degrades.

{trace_text}

Provide your diagnosis as a ranked list of agents to fix, most important first.
For each agent, state:
- Agent name
- Problematic instruction (quote it)
- Which quality dimension it degrades
- Suggested fix"""

    print(f"\n{'='*72}")
    print("  Sending all 5 traces to Claude for diagnosis...")
    print(f"  Prompt size: {len(diagnosis_prompt):,} characters")
    print(f"{'='*72}")

    diagnosis = call(diagnosis_prompt, model=SONNET, max_tokens=2000, temperature=0.0)

    print(f"\n  LLM DIAGNOSIS:")
    print(f"  {'─'*60}")
    for line in diagnosis.split('\n'):
        print(f"  {line}")

    # Print comparison
    print(f"\n{'='*72}")
    print("  COMPARISON: LLM diagnosis vs. Shapley attribution")
    print(f"{'='*72}")
    print("""
  Shapley attribution (from counterfact):
    1. tone_editor       -0.088  (worst on precision: -0.236)
    2. context_enricher  -0.061  (worst on accuracy after fix)
    3. table_extractor   -0.047  (rounds source figures)
    4. fact_checker       +0.065  (but permissive prompt)

  Key Shapley insight:
    Fixing context_enricher alone makes quality WORSE (-0.022)
    because tone_editor is still converting M to B.
    The tool catches this interaction; does the LLM?
""")

    # Save
    rdir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "llm_debugger_baseline.txt"), "w") as f:
        f.write("LLM DIAGNOSIS:\n\n")
        f.write(diagnosis)
        f.write("\n\nSHAPLEY ATTRIBUTION:\n")
        f.write("1. tone_editor -0.088\n2. context_enricher -0.061\n3. table_extractor -0.047\n")
    print(f"  Saved: {os.path.abspath(os.path.join(rdir, 'llm_debugger_baseline.txt'))}")
    print("=" * 72)


if __name__ == "__main__":
    main()
