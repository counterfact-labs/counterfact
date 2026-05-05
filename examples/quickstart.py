"""
Counterfact Quickstart Example
==============================

Demonstrates counterfactual diagnosis on a 3-agent RAG pipeline using
a real LLM (Claude Haiku) for synthesis.

Pipeline:
    Retriever → Summarizer (LLM) → Fact Checker

The Scenario:
    A user asks "What is the surface temperature of Venus?"
    The retriever has a bug — it matches on keywords too loosely and
    returns facts about Venus's *atmosphere* and *proximity to the Sun*
    but nothing about surface temperature. The LLM summarizer does its
    best with the irrelevant context and hallucinates a plausible answer.
    The fact checker can't catch the problem because it only validates
    structural consistency, not factual coverage.

    This is a classic RAG failure: retrieval miss → hallucinated
    synthesis → insufficient validation.

    Counterfact diagnoses this by computing Shapley values: re-running
    the pipeline with every coalition of agents, measuring real quality
    changes, and deriving bootstrap confidence intervals.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   (or place in ../.env)

Run:
    python examples/quickstart.py
"""

import os
import sys
from pathlib import Path

from counterfact import StateGraph, END
from counterfact.classifiers import ClassifierRegistry
from counterfact.types import ClassifierResult

# ─── Load API Key ────────────────────────────────────────────────────────

def load_anthropic_key() -> str:
    """Load Anthropic API key from environment or ../.env file."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        env_path = Path(__file__).resolve().parent.parent.parent / "counterfactual-debugger" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["ANTHROPIC_API_KEY"] = key
                    break
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found")
    return key


# ─── Knowledge Base ──────────────────────────────────────────────────────
# Simulates a document store. Note: no fact contains "surface temperature".

KNOWLEDGE_BASE = [
    {"id": "venus-1", "text": "Venus is the second planet from the Sun."},
    {"id": "venus-2", "text": "Venus has a thick atmosphere composed mainly of carbon dioxide."},
    {"id": "venus-3", "text": "The atmospheric pressure on Venus is 92 times that of Earth."},
    {"id": "venus-4", "text": "Venus rotates in the opposite direction to most planets."},
    {"id": "mars-1",  "text": "Mars has the largest volcano in the solar system, Olympus Mons."},
    {"id": "earth-1", "text": "Earth is the third planet from the Sun and has liquid water."},
]

# ─── Pipeline State ──────────────────────────────────────────────────────

from typing import TypedDict

class PipelineState(TypedDict):
    query: str
    retrieved_facts: list[dict]
    summary: str
    validation: str
    output: str

# ─── Node Functions ──────────────────────────────────────────────────────

def retriever(state: PipelineState) -> dict:
    """Search the knowledge base for facts relevant to the query.

    BUG: Matches too broadly on any keyword > 3 chars. For the query
    "surface temperature of Venus", this returns facts about Venus's
    atmosphere — but none containing actual temperature data.
    """
    import re as _re
    query_lower = state["query"].lower()
    words = [_re.sub(r'[^\w]', '', w) for w in query_lower.split()]
    results = []
    for fact in KNOWLEDGE_BASE:
        if any(w in fact["text"].lower() for w in words if len(w) > 3):
            results.append(fact)
    return {"retrieved_facts": results}


# The summarizer is created dynamically in main() with the Anthropic client.
# See _make_summarizer() below.


def fact_checker(state: PipelineState) -> dict:
    """Validate the summary against retrieved facts.

    Only checks structural consistency (are the cited facts present?).
    Cannot detect that the ANSWER to the query is not actually in the sources.
    """
    summary = state.get("summary", "")
    facts = state.get("retrieved_facts", [])

    if not summary or summary == "No relevant information found.":
        return {"validation": "FAIL: No content to validate.", "output": summary}

    fact_texts = [f["text"] for f in facts]
    cited = sum(1 for ft in fact_texts if ft in summary)

    if cited == len(fact_texts):
        validation = f"PASS: All {cited} source facts are cited in the summary."
    else:
        validation = f"PARTIAL: {cited}/{len(fact_texts)} facts cited."

    return {"validation": validation, "output": summary}


def _make_summarizer(client):
    """Create a summarizer node that uses Claude Haiku."""

    def summarizer(state: PipelineState) -> dict:
        """Use Claude Haiku to synthesize retrieved facts into an answer."""
        facts = state.get("retrieved_facts", [])
        query = state.get("query", "")

        if not facts:
            return {"summary": "No relevant information found.", "output": "No relevant information found."}

        facts_text = "\n".join(f"- {f['text']}" for f in facts)
        prompt = (
            f"You are a helpful assistant. Answer the user's question using ONLY "
            f"the provided facts. If the facts don't contain the answer, say so.\n\n"
            f"Facts:\n{facts_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely in 2-3 sentences."
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0.7,   # Non-zero for variance across runs
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.content[0].text
        return {"summary": summary, "output": summary}

    return summarizer


def _deterministic_summarizer(state: PipelineState) -> dict:
    """Fallback summarizer when LLM is unavailable."""
    facts = state.get("retrieved_facts", [])
    if not facts:
        return {"summary": "No relevant information found.", "output": "No relevant information found."}

    answer_parts = ["Based on available knowledge:"]
    for fact in facts:
        answer_parts.append(f"- {fact['text']}")
    answer_parts.append("")
    answer_parts.append(
        "Given Venus's thick CO2 atmosphere and proximity to the Sun, "
        "its surface temperature is likely extremely high."
    )
    summary = "\n".join(answer_parts)
    return {"summary": summary, "output": summary}


# ─── Build Pipeline ──────────────────────────────────────────────────────

def build_pipeline(client=None):
    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever)
    if client is not None:
        graph.add_node("summarizer", _make_summarizer(client))
    else:
        graph.add_node("summarizer", _deterministic_summarizer)
    graph.add_node("fact_checker", fact_checker)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "fact_checker")
    graph.add_edge("fact_checker", END)

    return graph.compile()


# ─── Custom Classifiers (no LLM needed) ──────────────────────────────────

def completeness_classifier(query: str, output: str, sources: str) -> ClassifierResult:
    """Does the output contain substantive content?"""
    if not output or len(output) < 20:
        return ClassifierResult(name="completeness", score=0.1, reasoning="Output is empty or too short.")
    if "No relevant information" in output:
        return ClassifierResult(name="completeness", score=0.2, reasoning="Pipeline produced no answer.")
    word_count = len(output.split())
    score = min(1.0, word_count / 30)
    return ClassifierResult(name="completeness", score=score, reasoning=f"Output contains {word_count} words.")


def answer_relevance_classifier(query: str, output: str, sources: str) -> ClassifierResult:
    """Does the output actually answer the specific question asked?

    For "What is the surface temperature of Venus?", a good answer must
    contain a specific temperature value for Venus. Vague statements
    like "likely extremely high" are insufficient.
    """
    output_lower = output.lower()

    import re
    has_specific_temp = bool(re.search(r'\d+\s*°?[CFK]', output))
    mentions_venus_temp = has_specific_temp and "venus" in output_lower
    mentions_temperature = "temperature" in output_lower

    if mentions_venus_temp:
        return ClassifierResult(
            name="answer_relevance", score=0.9,
            reasoning="Output provides a specific temperature value for Venus.",
        )
    elif mentions_temperature and ("high" in output_lower or "hot" in output_lower):
        return ClassifierResult(
            name="answer_relevance", score=0.3,
            reasoning="Mentions temperature but only vaguely ('extremely high'). No specific value.",
            weight=1.5,
        )
    elif mentions_temperature:
        return ClassifierResult(
            name="answer_relevance", score=0.2,
            reasoning="Output mentions temperature but provides no useful information.",
            weight=1.5,
        )
    else:
        return ClassifierResult(
            name="answer_relevance", score=0.1,
            reasoning="Output does not address the temperature question at all.",
            weight=1.5,
        )


def source_coverage_classifier(query: str, output: str, sources: str) -> ClassifierResult:
    """Do the retrieved sources contain the information needed to answer?"""
    import re
    has_temp_data = bool(re.search(r'\d+\s*°?[CFK]', output))
    mentions_surface = "surface" in output.lower()

    if has_temp_data and mentions_surface:
        return ClassifierResult(
            name="source_coverage", score=0.9,
            reasoning="Sources contain specific temperature data relevant to the query.",
        )
    elif "atmosphere" in output.lower() or "carbon dioxide" in output.lower():
        return ClassifierResult(
            name="source_coverage", score=0.3,
            reasoning="Sources discuss Venus's atmosphere but lack temperature measurements.",
        )
    else:
        return ClassifierResult(
            name="source_coverage", score=0.1,
            reasoning="Sources contain no information relevant to the query.",
        )


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  COUNTERFACT QUICKSTART")
    print("  Diagnosing a retrieval failure in a 3-agent RAG pipeline")
    print("=" * 72)

    # Try to set up LLM-based pipeline; fall back to deterministic if unavailable
    client = None
    try:
        api_key = load_anthropic_key()
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Test connectivity
        client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        print(f"\n  ✓ Using Claude for synthesis (non-deterministic → meaningful CIs)")
    except (SystemExit, Exception) as e:
        client = None
        print(f"\n  ⚠ LLM unavailable ({type(e).__name__}), using deterministic summarizer")
        print(f"    (Set ANTHROPIC_API_KEY for LLM-based diagnosis with real CIs)")

    # Build the pipeline
    pipeline = build_pipeline(client)

    # Set up custom classifiers
    registry = ClassifierRegistry()
    registry.register(completeness_classifier, "demo")
    registry.register(answer_relevance_classifier, "demo")
    registry.register(source_coverage_classifier, "demo")

    # Define input
    input_state: PipelineState = {
        "query": "What is the surface temperature of Venus?",
        "retrieved_facts": [],
        "summary": "",
        "validation": "",
        "output": "",
    }

    # ─── Step 1: Run the pipeline ────────────────────────────────────
    print("\n" + "─" * 72)
    print("STEP 1: BASELINE RUN")
    print("─" * 72)

    result = pipeline.invoke(input_state)
    trace = pipeline.get_trace()

    print(f"\nQuery: {input_state['query']}")
    print(f"\nPipeline Output:\n{result['output']}")
    print(f"\nValidation: {result['validation']}")
    print(f"\nExecution Trace ({len(trace)} steps):")
    for entry in trace:
        print(f"  [{entry['node']}] {entry['status']} ({entry.get('duration_ms', 0):.0f}ms)")

    # ─── Step 2: Run counterfactual diagnosis ────────────────────────
    print("\n" + "─" * 72)
    print("STEP 2: COUNTERFACTUAL DIAGNOSIS")
    print("Running Shapley attribution with real pipeline re-execution...")
    print("─" * 72)

    report = pipeline.diagnose(
        input_state=input_state,
        domain="demo",
        num_simulations=12,
        registry=registry,
        seed=42,
    )

    # ─── Step 3: Show Monte Carlo simulation details ─────────────────
    print("\n" + "─" * 72)
    print("STEP 3: MONTE CARLO SIMULATIONS (ablate one agent at a time)")
    print("─" * 72)

    for sim in report.simulation_results:
        if sim.is_baseline:
            label = "BASELINE (all agents active)"
        else:
            label = f"ABLATE {sim.perturbation.agent}"

        print(f"\n  Simulation #{sim.simulation_id}: {label}")
        print(f"  Quality Score: {sim.quality_score:.3f}")

        output_preview = sim.perturbed_output.replace('\n', ' ')[:80]
        print(f"  Output: {output_preview}...")

        for clf in sim.classifier_results:
            bar = "█" * int(clf.score * 20) + "░" * (20 - int(clf.score * 20))
            print(f"    {clf.name:>20s}: {clf.score:.2f} {bar}  {clf.reasoning[:55]}")

    # ─── Step 4: Shapley attribution with CIs ────────────────────────
    print("\n" + "─" * 72)
    print("STEP 4: SHAPLEY ATTRIBUTION (with bootstrap confidence intervals)")
    print("─" * 72)

    print(f"\nBaseline Quality: {report.baseline_quality:.3f}")
    print(f"Attribution Method: {report.attribution_method}")

    print(f"\nShapley Values (marginal contribution of each agent):")
    for agent, value in sorted(report.shapley_values.items(), key=lambda x: abs(x[1]), reverse=True):
        bar_width = int(abs(value) * 40)
        direction = "◄" if value < 0 else "►"
        bar = "█" * bar_width
        sign = "+" if value >= 0 else ""

        # Show confidence interval if available
        ci = report.shapley_cis.get(agent)
        if ci and ci.n_samples >= 2:
            ci_str = f"  95% CI: [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]  (n={ci.n_samples})"
        else:
            ci_str = ""

        print(f"  {agent:>15s}: {sign}{value:.3f}  {direction}{bar}{ci_str}")

    if report.per_classifier_shapley:
        print(f"\nPer-Classifier Shapley Breakdown:")
        for clf_name, agent_values in report.per_classifier_shapley.items():
            print(f"\n  {clf_name}:")
            for agent, value in sorted(agent_values.items(), key=lambda x: abs(x[1]), reverse=True):
                sign = "+" if value >= 0 else ""
                print(f"    {agent:>15s}: {sign}{value:.3f}")

    # ─── Step 5: Show diagnosis ──────────────────────────────────────
    print("\n" + "─" * 72)
    print("STEP 5: DIAGNOSIS")
    print("─" * 72)

    cls = report.classification
    print(f"\nFailure Type: {cls.failure_type}")
    print(f"Confidence: {cls.confidence:.0%}")
    print(f"Description: {cls.description}")

    if cls.confidence_explanation:
        print(f"Confidence Basis: {cls.confidence_explanation}")

    print(f"\nEvidence:")
    for ev in cls.evidence:
        print(f"  • {ev}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. [{rec.intervention_type}] {rec.description}")
            if rec.target_agent:
                print(f"     Target: {rec.target_agent}")

    print(f"\nSimulation Summary:")
    summary = report.simulation_results_summary
    print(f"  Total simulations: {summary['total_simulations']}")
    print(f"  Baseline runs: {summary['baseline_runs']}")
    print(f"  Perturbation runs: {summary['perturbation_runs']}")
    print(f"  Agents analyzed: {', '.join(summary['agents_analyzed'])}")
    print(f"  Classifiers used: {', '.join(summary['classifiers_used'])}")

    print("\n" + "=" * 72)
    print("  KEY INSIGHT: Counterfact computes Shapley values by actually")
    print("  re-running the pipeline with agents ablated. The LLM-based")
    print("  summarizer introduces natural variance, giving meaningful")
    print("  bootstrap confidence intervals on the attribution scores.")
    print("=" * 72)


if __name__ == "__main__":
    main()


"""
EXPECTED OUTPUT:

────────────────────────────────────────────────────────────────────────
STEP 1: BASELINE RUN
────────────────────────────────────────────────────────────────────────

Query: What is the surface temperature of Venus?

Pipeline Output:
Based on available knowledge:
- Venus is the second planet from the Sun.
- Venus has a thick atmosphere composed mainly of carbon dioxide.
- The atmospheric pressure on Venus is 92 times that of Earth.
- Venus rotates in the opposite direction to most planets.

Given Venus's thick CO2 atmosphere and proximity to the Sun, its surface temperature is likely extremely high.

Validation: PASS: All 4 source facts are cited in the summary.

Execution Trace (3 steps):
  [retriever] pass (0ms)
  [summarizer] pass (0ms)
  [fact_checker] pass (0ms)

────────────────────────────────────────────────────────────────────────
STEP 2: COUNTERFACTUAL DIAGNOSIS
Running Shapley attribution with real pipeline re-execution...
────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────
STEP 3: MONTE CARLO SIMULATIONS (ablate one agent at a time)
────────────────────────────────────────────────────────────────────────

  Simulation #0: BASELINE (all agents active)
  Quality Score: 0.500
  Output: Based on available knowledge: - Venus is the second planet from the Sun. - Venus...
            completeness: 1.00 ████████████████████  Output contains 63 words.
        answer_relevance: 0.30 ██████░░░░░░░░░░░░░░  Mentions temperature but only vaguely ('extremely high'
         source_coverage: 0.30 ██████░░░░░░░░░░░░░░  Sources discuss Venus's atmosphere but lack temperature

  Simulation #1: BASELINE (all agents active)
  Quality Score: 0.500
  Output: Based on available knowledge: - Venus is the second planet from the Sun. - Venus...
            completeness: 1.00 ████████████████████  Output contains 63 words.
        answer_relevance: 0.30 ██████░░░░░░░░░░░░░░  Mentions temperature but only vaguely ('extremely high'
         source_coverage: 0.30 ██████░░░░░░░░░░░░░░  Sources discuss Venus's atmosphere but lack temperature

  Simulation #2: BASELINE (all agents active)
  Quality Score: 0.500
  Output: Based on available knowledge: - Venus is the second planet from the Sun. - Venus...
            completeness: 1.00 ████████████████████  Output contains 63 words.
        answer_relevance: 0.30 ██████░░░░░░░░░░░░░░  Mentions temperature but only vaguely ('extremely high'
         source_coverage: 0.30 ██████░░░░░░░░░░░░░░  Sources discuss Venus's atmosphere but lack temperature

  Simulation #3: ABLATE fact_checker, retriever
  Quality Score: 0.129
  Output: No relevant information found....
            completeness: 0.20 ████░░░░░░░░░░░░░░░░  Pipeline produced no answer.
        answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output does not address the temperature question at all
         source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  Sources contain no information relevant to the query.

  Simulation #4: ABLATE retriever
  Quality Score: 0.129
  Output: No relevant information found....
            completeness: 0.20 ████░░░░░░░░░░░░░░░░  Pipeline produced no answer.
        answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output does not address the temperature question at all
         source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  Sources contain no information relevant to the query.

  Simulation #5: ABLATE summarizer
  Quality Score: 0.100
  Output: ...
            completeness: 0.10 ██░░░░░░░░░░░░░░░░░░  Output is empty or too short.
        answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output does not address the temperature question at all
         source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  Sources contain no information relevant to the query.

  Simulation #6: ABLATE fact_checker
  Quality Score: 0.500
  Output: Based on available knowledge: - Venus is the second planet from the Sun. - Venus...
            completeness: 1.00 ████████████████████  Output contains 63 words.
        answer_relevance: 0.30 ██████░░░░░░░░░░░░░░  Mentions temperature but only vaguely ('extremely high'
         source_coverage: 0.30 ██████░░░░░░░░░░░░░░  Sources discuss Venus's atmosphere but lack temperature

  Simulation #7: ABLATE fact_checker, summarizer
  Quality Score: 0.100
  Output: ...
            completeness: 0.10 ██░░░░░░░░░░░░░░░░░░  Output is empty or too short.
        answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output does not address the temperature question at all
         source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  Sources contain no information relevant to the query.

  Simulation #8: ABLATE retriever, summarizer
  Quality Score: 0.100
  Output: ...
            completeness: 0.10 ██░░░░░░░░░░░░░░░░░░  Output is empty or too short.
        answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output does not address the temperature question at all
         source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  Sources contain no information relevant to the query.

────────────────────────────────────────────────────────────────────────
STEP 4: SHAPLEY ATTRIBUTION (with bootstrap confidence intervals)
────────────────────────────────────────────────────────────────────────

Baseline Quality: 0.500
Attribution Method: shapley

Shapley Values (marginal contribution of each agent):
       summarizer: +0.495  ►███████████████████  95% CI: [+0.248, +0.710]  (n=6)
        retriever: +0.438  ►█████████████████  95% CI: [+0.190, +0.652]  (n=6)
     fact_checker: +0.067  ►██  95% CI: [+0.000, +0.133]  (n=6)

Per-Classifier Shapley Breakdown:

  completeness:
         summarizer: +0.533
          retriever: +0.433
       fact_checker: +0.033

  answer_relevance:
          retriever: +0.444
         summarizer: +0.444
       fact_checker: +0.111

  source_coverage:
          retriever: +0.444
         summarizer: +0.444
       fact_checker: +0.111

────────────────────────────────────────────────────────────────────────
STEP 5: DIAGNOSIS
────────────────────────────────────────────────────────────────────────

Failure Type: local
Confidence: 56%
Description: The failure is primarily attributable to the summarizer agent.
Confidence Basis: Based on 9 simulations. Confidence derived from bootstrap CI separation between top two agents. Confidence = 56%

Evidence:
  • Dominant Shapley value: summarizer = 0.495
  • Perturbing summarizer changes quality by 49.5%
  • Failing classifiers: answer_relevance, source_coverage

Recommendations:
  1. [modify_agent] The summarizer is the primary failure source. Enhance with better validation and error handling.
     Target: summarizer

Simulation Summary:
  Total simulations: 9
  Baseline runs: 3
  Perturbation runs: 6
  Agents analyzed: retriever, summarizer, fact_checker
  Classifiers used: completeness, answer_relevance, source_coverage

========================================================================
  KEY INSIGHT: Counterfact computes Shapley values by actually
  re-running the pipeline with agents ablated. The LLM-based
  summarizer introduces natural variance, giving meaningful
  bootstrap confidence intervals on the attribution scores.
========================

────────────────────────────────────────────────────────────────────────
STEP 6: STATIC EVALUATION (counterfact eval)
────────────────────────────────────────────────────────────────────────

╭────────────────────╮
│ counterfact eval ✓ │
╰─ Ground-truth-free─╯

  Trace: 3 events from trace.json
  Agents: retriever, summarizer, fact_checker
  Tiers: 1

                               Evaluation Results                               
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                 ┃ Severity ┃ Status ┃ Details                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ empty_output          │   info   │ ✓ PASS │ Agent 'retriever' produced       │
│                       │          │        │ non-empty output.                │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ empty_output          │   info   │ ✓ PASS │ Agent 'summarizer' produced      │
│                       │          │        │ non-empty output.                │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ empty_output          │   info   │ ✓ PASS │ Agent 'fact_checker' produced    │
│                       │          │        │ non-empty output.                │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ error_status          │   info   │ ✓ PASS │ Agent 'retriever' status: pass   │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ error_status          │   info   │ ✓ PASS │ Agent 'summarizer' status: pass  │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ error_status          │   info   │ ✓ PASS │ Agent 'fact_checker' status:     │
│                       │          │        │ pass                             │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ schema_violation      │   info   │ ✓ PASS │ Agent 'retriever' output has     │
│                       │          │        │ keys.                            │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ schema_violation      │   info   │ ✓ PASS │ Agent 'summarizer' output has    │
│                       │          │        │ keys.                            │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ schema_violation      │   info   │ ✓ PASS │ Agent 'fact_checker' output has  │
│                       │          │        │ keys.                            │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ latency_anomaly       │   info   │ ✓ PASS │ Agent 'retriever' latency normal │
│                       │          │        │ (0ms).                           │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ output_length_anomaly │   info   │ ✓ PASS │ Agent 'retriever' output length  │
│                       │          │        │ normal (7 chars).                │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ output_length_anomaly │   info   │ ✓ PASS │ Agent 'summarizer' output length │
│                       │          │        │ normal (600 chars).              │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ output_length_anomaly │   info   │ ✓ PASS │ Agent 'fact_checker' output      │
│                       │          │        │ length normal (350 chars).       │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ duplicate_agent       │   info   │ ✓ PASS │ Agent 'retriever' ran 1 time(s). │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ duplicate_agent       │   info   │ ✓ PASS │ Agent 'summarizer' ran 1         │
│                       │          │        │ time(s).                         │
├───────────────────────┼──────────┼────────┼──────────────────────────────────┤
│ duplicate_agent       │   info   │ ✓ PASS │ Agent 'fact_checker' ran 1       │
│                       │          │        │ time(s).                         │
└───────────────────────┴──────────┴────────┴──────────────────────────────────┘
╭────────────────────────────────── Summary ───────────────────────────────────╮
│ Passed: 16/16  |  Score: 100.0%                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

"""
