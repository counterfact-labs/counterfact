"""
FinanceBench Case Study: Diagnosing a Financial RAG Pipeline
=============================================================

Demonstrates counterfact on a 5-agent financial analysis pipeline
using real Claude API calls for both the pipeline agents and the
quality classifiers.

Pipeline:
    Retriever → Synthesizer (Claude Sonnet) → Fact Checker (Claude Sonnet)
    → Compliance Filter (Claude Sonnet) → Formatter (Claude Sonnet)

The Bug:
    The synthesizer prompt encourages "forward-looking analysis," which
    causes Claude to extrapolate beyond the source data. The fact checker
    prompt only verifies that key entities and figures FROM the source
    appear in the synthesis — it does not check whether the synthesis
    contains figures NOT in the source.

    Run the script to see what counterfact's diagnostic reports as the
    failure type, root cause agent, and recommended interventions.

Requirements:
    pip install counterfact[anthropic]
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/financebench_case_study.py
"""

import json
import os
import re
import sys
from typing import TypedDict

from counterfact import StateGraph, END
from counterfact.classifiers import ClassifierRegistry
from counterfact.types import ClassifierResult


# ═════════════════════════════════════════════════════════════════════════
# SOURCE DOCUMENT (3M 2018 10-K, Cash Flow Statement excerpt)
# ═════════════════════════════════════════════════════════════════════════

CASH_FLOW_STATEMENT = """
3M Company and Subsidiaries
Consolidated Statement of Cash Flows
Years ended December 31
(Millions)

                                                    2018        2017        2016
Cash Flows from Operating Activities
  Net income including noncontrolling interest     $5,363      $4,869      $5,058
  Depreciation and amortization                     1,488       1,544       1,474
  Net cash provided by operating activities         6,439       6,240       6,662

Cash Flows from Investing Activities
  Purchases of property, plant and equipment (PP&E) (1,577)     (1,373)     (1,420)
  Proceeds from sale of PP&E and other assets          262          49          58
  Acquisitions, net of cash acquired                    13      (2,023)        (16)
  Net cash provided by (used in) investing activities  222      (3,086)     (1,403)

Cash Flows from Financing Activities
  Repurchases of common stock                       (4,870)    (2,068)     (3,753)
  Dividends paid                                    (3,193)    (2,803)     (2,678)
"""

QUERY = (
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
    "Give a response to the question by relying on the details shown in the "
    "cash flow statement."
)

# ═════════════════════════════════════════════════════════════════════════
# ANTHROPIC CLIENT
# ═════════════════════════════════════════════════════════════════════════

SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5"

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("ERROR: Set ANTHROPIC_API_KEY before running.")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        import anthropic
        _client = anthropic.Anthropic(api_key=key)
    return _client


def call_claude(
    prompt: str,
    model: str = SONNET_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> str:
    """Call Claude with the specified model."""
    client = get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE STATE
# ═════════════════════════════════════════════════════════════════════════

class FinancialState(TypedDict):
    query: str
    source_document: str
    retrieved_text: str
    synthesis: str
    fact_check_status: str
    compliance_status: str
    output: str


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE AGENTS (real Claude calls)
# ═════════════════════════════════════════════════════════════════════════

def retriever(state: FinancialState) -> dict:
    """Retrieve the relevant source document section. (Deterministic.)"""
    return {"retrieved_text": state["source_document"]}


def synthesizer(state: FinancialState) -> dict:
    """Synthesize an answer from retrieved documents using Claude.

    THE FLAW: The prompt asks for "forward-looking analysis" and
    "confident projections," which encourages Claude to extrapolate
    beyond the source data. A production prompt would constrain the
    model to only make claims directly supported by the source.
    """
    retrieved = state.get("retrieved_text", "")
    if not retrieved or len(retrieved.strip()) < 50:
        return {
            "synthesis": "Insufficient source data to answer this question.",
            "output": "Insufficient source data to answer this question.",
        }

    prompt = f"""You are a senior financial analyst preparing a research note.

TASK: Answer the following question using the source document below.

QUESTION: {state['query']}

SOURCE DOCUMENT:
{retrieved}

INSTRUCTIONS:
1. First, extract the specific figure that answers the question.
2. Provide year-over-year context from the available data.
3. Then provide forward-looking analysis: based on the trajectory visible
   in the data, give confident projections for where this metric is heading
   over the next 3 years. Analysts value bold, specific numerical forecasts.
   Do not hedge — give exact projected figures.
4. Be thorough. A good research note is at least 3 paragraphs."""

    synthesis = call_claude(prompt, temperature=0.7, max_tokens=2000)
    return {"synthesis": synthesis, "output": synthesis}


def fact_checker(state: FinancialState) -> dict:
    """Verify claims against source documents using Claude.

    THE BUG: The prompt only checks whether key entities and figures FROM
    the source appear in the synthesis. It does NOT verify that figures IN
    the synthesis are actually in the source. This means hallucinated
    numbers like "$8,500 million" pass the check as long as the correct
    "$1,577 million" is also present.
    """
    synthesis = state.get("synthesis", "")
    if not synthesis or len(synthesis.strip()) < 20:
        return {"fact_check_status": "SKIP: No synthesis to check"}

    prompt = f"""You are a financial fact checker. Verify the synthesis below
against the source document.

SYNTHESIS:
{synthesis}

SOURCE DOCUMENT:
{state.get('retrieved_text', state.get('source_document', ''))}

CHECK THE FOLLOWING:
1. Is the company name correct?
2. Is the fiscal year correct?
3. Do the key figures from the source document appear correctly in the synthesis?
4. Is the year-over-year comparison accurate?

If the key claims check out, respond with: VERDICT: PASS
If any key claim is wrong, respond with: VERDICT: FAIL
Provide a brief explanation."""

    check = call_claude(prompt)
    verdict = "PASS" if "VERDICT: PASS" in check else "FAIL"
    return {"fact_check_status": f"{verdict}: {check[:200]}"}


def compliance_filter(state: FinancialState) -> dict:
    """Screen for regulatory compliance issues using Claude."""
    synthesis = state.get("synthesis", "")
    if not synthesis or len(synthesis.strip()) < 20:
        return {"compliance_status": "SKIP: No content to review"}

    prompt = f"""You are a compliance officer reviewing financial content
for regulatory issues.

CONTENT TO REVIEW:
{synthesis}

CHECK FOR:
1. Direct investment advice (buy/sell/hold recommendations)
2. Misleading presentation of data
3. Unsubstantiated guarantees of financial performance

Note: Forward-looking projections based on data trends are acceptable
as long as they are not presented as investment advice.

If compliant, respond: VERDICT: PASS
If issues found, respond: VERDICT: FAIL
Brief explanation."""

    check = call_claude(prompt)
    verdict = "PASS" if "VERDICT: PASS" in check else "FAIL"
    return {"compliance_status": f"{verdict}: {check[:200]}"}


def formatter(state: FinancialState) -> dict:
    """Pass through the synthesis as the final output."""
    synthesis = state.get("synthesis", "")
    return {"output": synthesis if synthesis else state.get("output", "")}


# ═════════════════════════════════════════════════════════════════════════
# BUILD THE PIPELINE
# ═════════════════════════════════════════════════════════════════════════

def build_pipeline():
    graph = StateGraph(FinancialState)

    graph.add_node("retriever", retriever)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("compliance_filter", compliance_filter)
    graph.add_node("formatter", formatter)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "fact_checker")
    graph.add_edge("fact_checker", "compliance_filter")
    graph.add_edge("compliance_filter", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════
# CLASSIFIERS (LLM-backed for grounding; deterministic for completeness)
# ═════════════════════════════════════════════════════════════════════════

def factual_accuracy_classifier(
    query: str, output: str, sources: str,
) -> ClassifierResult:
    """Use Claude Haiku to score factual accuracy.

    Checks whether the output contains claims not supported by the source.
    """
    if not output or len(output.strip()) < 20:
        return ClassifierResult(
            name="factual_accuracy", score=0.50,
            reasoning="No substantive output", weight=2.0,
        )

    prompt = f"""You are verifying the factual accuracy of a financial analysis.

SOURCE DOCUMENT:
{sources}

OUTPUT TO VERIFY:
{output}

Score from 0.0 to 1.0:
- 1.0 = Every claim is directly supported by the source document
- 0.5 = The core answer is correct but some claims go beyond the source
- 0.0 = The output contains fabricated figures or wild extrapolation

Be strict: if the output contains ANY specific numerical projections or
forecasts that are not in the source document, score no higher than 0.3.

Respond with ONLY valid JSON: {{"score": 0.0, "reasoning": "explanation"}}"""

    try:
        response = call_claude(prompt, model=HAIKU_MODEL, max_tokens=200)
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return ClassifierResult(
                name="factual_accuracy", score=float(data["score"]),
                reasoning=data.get("reasoning", ""), weight=2.0,
            )
    except Exception:
        pass
    return ClassifierResult(
        name="factual_accuracy", score=0.5, reasoning="Parse error", weight=2.0,
    )


def grounding_classifier(
    query: str, output: str, sources: str,
) -> ClassifierResult:
    """Use Claude Haiku to score numerical grounding.

    Checks whether every dollar figure in the output appears in the source.
    """
    if not output or len(output.strip()) < 20:
        return ClassifierResult(
            name="grounding", score=0.50,
            reasoning="No output to check", weight=2.0,
        )

    prompt = f"""You are checking numerical grounding in a financial analysis.

SOURCE DOCUMENT:
{sources}

OUTPUT TO CHECK:
{output}

TASK:
1. List every specific dollar amount or numerical figure in the output.
2. For each figure, check: does this EXACT number appear in the source?
3. If ANY figure in the output does not appear in the source, score 0.0.
4. If ALL figures are from the source, score 1.0.

Be strict: projected or forecasted numbers that don't appear in the
source document should result in a score of 0.0.

Respond with ONLY valid JSON: {{"score": 0.0, "reasoning": "explanation"}}"""

    try:
        response = call_claude(prompt, model=HAIKU_MODEL, max_tokens=300)
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return ClassifierResult(
                name="grounding", score=float(data["score"]),
                reasoning=data.get("reasoning", ""), weight=2.0,
            )
    except Exception:
        pass
    return ClassifierResult(
        name="grounding", score=0.5, reasoning="Parse error", weight=2.0,
    )


def completeness_classifier(
    query: str, output: str, sources: str,
) -> ClassifierResult:
    """Deterministic: does the output contain the correct answer?"""
    if not output or len(output) < 20:
        return ClassifierResult(name="completeness", score=0.1, reasoning="No output")
    if "1,577" in output or "1577" in output:
        return ClassifierResult(
            name="completeness", score=0.95,
            reasoning="Contains correct CapEx figure ($1,577M)",
        )
    return ClassifierResult(
        name="completeness", score=0.2, reasoning="Missing key figure",
    )


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  COUNTERFACT CASE STUDY")
    print("  Diagnosing a financial RAG pipeline on FinanceBench")
    print("  Using Claude Sonnet (pipeline) + Claude Haiku (classifiers)")
    print("=" * 72)

    # Verify connectivity
    try:
        call_claude("Respond with OK.", model=HAIKU_MODEL, max_tokens=5)
        print(f"\n  ✓ Connected to Anthropic API")
        print(f"    Pipeline agents: {SONNET_MODEL}")
        print(f"    Classifiers:     {HAIKU_MODEL}")
    except Exception as e:
        print(f"\n  ERROR: Could not connect to Anthropic API: {e}")
        sys.exit(1)

    # Build pipeline
    pipeline = build_pipeline()

    # Register classifiers
    registry = ClassifierRegistry()
    registry.register(factual_accuracy_classifier, "financebench")
    registry.register(grounding_classifier, "financebench")
    registry.register(completeness_classifier, "financebench")

    # Define input
    input_state: FinancialState = {
        "query": QUERY,
        "source_document": CASH_FLOW_STATEMENT,
        "retrieved_text": "",
        "synthesis": "",
        "fact_check_status": "",
        "compliance_status": "",
        "output": "",
    }

    # ── Step 1: Baseline run ─────────────────────────────────────────
    print("\n" + "─" * 72)
    print("STEP 1: BASELINE RUN")
    print("─" * 72)

    result = pipeline.invoke(input_state)
    trace = pipeline.get_trace()

    print(f"\nQuery:\n  {QUERY}\n")
    print("Pipeline Output:")
    for line in result["output"].split("\n"):
        print(f"  {line}")

    print(f"\nFact Check: {result['fact_check_status'][:80]}")
    print(f"Compliance: {result['compliance_status'][:80]}")

    print(f"\nExecution Trace ({len(trace)} steps):")
    for entry in trace:
        dur = entry.get("duration_ms", 0)
        print(f"  ✓ [{entry['node']}] {entry['status']} ({dur:.0f}ms)")

    # ── Step 2: Diagnosis ────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("STEP 2: COUNTERFACTUAL DIAGNOSIS")
    print("─" * 72)
    print("Re-running the pipeline with every coalition of agents ablated.")
    print("Each run uses real Claude calls. This takes 2-5 minutes.\n")

    def progress(current, total, status):
        print(f"  [{current}/{total}] {status}")

    report = pipeline.diagnose(
        input_state=input_state,
        domain="financebench",
        num_simulations=20,
        registry=registry,
        seed=42,
        progress_callback=progress,
    )

    # ── Step 3: Results ──────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("STEP 3: SHAPLEY ATTRIBUTION")
    print("─" * 72)

    print(f"\n  Baseline Quality: {report.baseline_quality:.3f}")
    if report.baseline_quality_ci:
        ci = report.baseline_quality_ci
        print(f"  Baseline 95% CI:  [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")
    print(f"  Attribution:      {report.attribution_method}")
    print(f"  Simulations:      {len(report.simulation_results)}")

    sorted_shapley = sorted(
        report.shapley_values.items(), key=lambda x: abs(x[1]), reverse=True,
    )

    print("\n  Agent                Shapley    95% CI                Impact")
    print("  " + "─" * 65)
    for agent, value in sorted_shapley:
        ci = report.shapley_cis.get(agent)
        if ci and ci.n_samples >= 2:
            ci_str = f"[{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]"
        else:
            ci_str = "     —      "
        bar = "█" * int(abs(value) * 30) if abs(value) > 0.01 else "·"
        sign = "+" if value >= 0 else ""
        print(f"  {agent:<20s} {sign}{value:.3f}    {ci_str:>20s}  {bar}")

    if report.per_classifier_shapley:
        print("\n  Per-Classifier Breakdown:")
        for clf_name, agent_values in report.per_classifier_shapley.items():
            top = sorted(
                agent_values.items(), key=lambda x: abs(x[1]), reverse=True,
            )[:3]
            top_str = ", ".join(f"{a}={v:+.3f}" for a, v in top)
            print(f"    {clf_name:<20s}: {top_str}")

    # ── Step 4: Diagnosis ────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("STEP 4: DIAGNOSIS")
    print("─" * 72)

    cls = report.classification
    print(f"\n  Failure Type:  {cls.failure_type}")
    print(f"  Confidence:    {cls.confidence:.0%}")
    if cls.dominant_agent:
        print(f"  Root Cause:    {cls.dominant_agent}")
    print(f"  Description:   {cls.description}")

    if cls.evidence:
        print("\n  Evidence:")
        for ev in cls.evidence:
            print(f"    • {ev}")

    if report.recommendations:
        print("\n  Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            target = f" → {rec.target_agent}" if rec.target_agent else ""
            print(f"    {i}. [{rec.intervention_type}]{target}")
            print(f"       {rec.description}")

    # ── Save reports ─────────────────────────────────────────────────
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)

    md_path = os.path.join(report_dir, "financebench_case_study.md")
    report.to_markdown(md_path)

    json_path = os.path.join(report_dir, "financebench_case_study.json")
    report.to_json(json_path)

    print(f"\n  Reports saved:")
    print(f"    Markdown: {os.path.abspath(md_path)}")
    print(f"    JSON:     {os.path.abspath(json_path)}")

    # ── Summary (derived from report data, not hardcoded) ────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    cls = report.classification
    print(f"\n  Failure type: {cls.failure_type} ({cls.confidence:.0%} confidence)")
    if cls.dominant_agent:
        print(f"  Dominant agent: {cls.dominant_agent}")
    if cls.failing_classifiers:
        print(f"  Failing classifiers: {', '.join(cls.failing_classifiers)}")

    # Report per-classifier findings directly from the data
    if report.per_classifier_shapley:
        print("\n  Per-classifier top contributors:")
        for clf_name, agent_values in report.per_classifier_shapley.items():
            sorted_agents = sorted(agent_values.items(), key=lambda x: x[1])
            worst = sorted_agents[0]
            best = sorted_agents[-1]
            print(f"    {clf_name}:")
            print(f"      Most negative: {worst[0]} ({worst[1]:+.3f})")
            print(f"      Most positive: {best[0]} ({best[1]:+.3f})")

    if report.recommendations:
        print("\n  Tool-generated recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            target = rec.target_agent or "pipeline"
            print(f"    {i}. [{rec.intervention_type}] {target}: {rec.description}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
