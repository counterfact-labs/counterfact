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
from pathlib import Path

from typing import TypedDict
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


# ─── Build Pipeline ──────────────────────────────────────────────────────

def build_pipeline(client):
    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever)
    graph.add_node("summarizer", _make_summarizer(client))
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


def make_llm_classifiers(client):
    """Factory to create LLM-backed classifiers for diagnostic simulation."""

    def answer_relevance_classifier(query: str, output: str, sources: str) -> ClassifierResult:
        """Use Claude to score how specifically the output answers the query."""
        if not output or "No relevant information" in output:
            return ClassifierResult(name="answer_relevance", score=0.1, reasoning="Output produced no answer.", weight=1.5)
            
        prompt = (
            f"Query: {query}\n"
            f"Output: {output}\n\n"
            f"Does the output specifically answer the query? If the query asks for a specific value (like temperature), "
            f"does the output provide that exact value? Vague statements like 'extremely high' are insufficient and should score low.\n"
            f"Score 0.0 to 1.0.\n"
            f"Respond with ONLY valid JSON: {{\"score\": 0.0-1.0, \"reasoning\": \"brief explanation\"}}"
        )
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307", max_tokens=150, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            import re
            match = re.search(r'```(?:json)?(.*?)```', response.content[0].text, re.DOTALL)
            text = match.group(1) if match else response.content[0].text
            parsed = json.loads(text.strip())
            return ClassifierResult(
                name="answer_relevance", score=float(parsed.get("score", 0.5)), 
                reasoning=parsed.get("reasoning", "Parsed"), weight=1.5
            )
        except Exception as e:
            return ClassifierResult(name="answer_relevance", score=0.5, reasoning=f"LLM Error: {str(e)}", weight=1.5)

    def source_coverage_classifier(query: str, output: str, sources: str) -> ClassifierResult:
        """Use Claude to score Grounding (is the output hallucinated?)."""
        if not output:
            return ClassifierResult(name="source_coverage", score=0.1, reasoning="No output")
            

        prompt = (
            f"Sources: {sources}\n"
            f"Output: {output}\n\n"
            f"Is the output completely grounded in the provided sources? "
            f"Score 1.0 if every claim is backed by the sources OR if the output correctly states the information is not found in the sources. "
            f"Score 0.0 if the output fabricates or hallucinates specific information not present in the sources.\n"
            f"Respond with ONLY valid JSON: {{\"score\": 0.0-1.0, \"reasoning\": \"brief explanation\"}}"
        )
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307", max_tokens=150, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            import re
            match = re.search(r'```(?:json)?(.*?)```', response.content[0].text, re.DOTALL)
            text = match.group(1) if match else response.content[0].text
            parsed = json.loads(text.strip())
            return ClassifierResult(
                name="source_coverage", score=float(parsed.get("score", 0.5)), 
                reasoning=parsed.get("reasoning", "Parsed")
            )
        except Exception as e:
            return ClassifierResult(name="source_coverage", score=0.5, reasoning=f"LLM Error: {str(e)}")
            
    return answer_relevance_classifier, source_coverage_classifier


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  COUNTERFACT QUICKSTART")
    print("  Diagnosing a retrieval failure in a 3-agent RAG pipeline")
    print("=" * 72)

    api_key = load_anthropic_key()
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    # Test connectivity
    try:
        client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        print("\n  ✓ Using Claude for synthesis and grounding checks")
    except Exception as e:
        print(f"\n  ERROR: Could not connect to Anthropic API ({type(e).__name__}).")
        print("  Please check your ANTHROPIC_API_KEY and network connection.")
        import sys
        sys.exit(1)

    # Build the pipeline
    pipeline = build_pipeline(client)

    # Set up custom classifiers
    ans_rel, src_cov = make_llm_classifiers(client)
    registry = ClassifierRegistry()
    registry.register(completeness_classifier, "demo")
    registry.register(ans_rel, "demo")
    registry.register(src_cov, "demo")

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

    print("\nShapley Values (marginal contribution of each agent):")
    for agent, value in sorted(report.shapley_values.items(), key=lambda x: abs(x[1]), reverse=True):
        bar_width = int(abs(value) * 20)
        if value < 0:
            bar = " " * (20 - bar_width) + "█" * bar_width + "│"
        else:
            bar = " " * 20 + "│" + "█" * bar_width
        sign = "+" if value >= 0 else ""

        # Show confidence interval if available
        ci = report.shapley_cis.get(agent)
        if ci and ci.n_samples >= 2:
            ci_str = f"  95% CI: [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]  (n={ci.n_samples})"
        else:
            ci_str = ""

        print(f"  {agent:>15s}: {sign}{value:.3f}  {bar}{ci_str}")

    if report.per_classifier_shapley:
        print("\nPer-Classifier Shapley Breakdown:")
        for clf_name, agent_values in report.per_classifier_shapley.items():
            print(f"\n  {clf_name}:")
            for agent, value in sorted(agent_values.items(), key=lambda x: abs(x[1]), reverse=True):
                bar_width = int(abs(value) * 20)
                if value < 0:
                    bar = " " * (20 - bar_width) + "█" * bar_width + "│"
                else:
                    bar = " " * 20 + "│" + "█" * bar_width
                sign = "+" if value >= 0 else ""
                print(f"    {agent:>15s}: {sign}{value:.3f}  {bar}")

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

    print("\nEvidence:")
    for ev in cls.evidence:
        print(f"  • {ev}")

    if report.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. [{rec.intervention_type}] {rec.description}")
            if rec.target_agent:
                print(f"     Target: {rec.target_agent}")

    print("\nSimulation Summary:")
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

    ========================================================================
      COUNTERFACT QUICKSTART
      Diagnosing a retrieval failure in a 3-agent RAG pipeline
    ========================================================================
    
      ✓ Using Claude for synthesis (non-deterministic → meaningful CIs)
    
    ────────────────────────────────────────────────────────────────────────
    STEP 1: BASELINE RUN
    ────────────────────────────────────────────────────────────────────────
    
    Query: What is the surface temperature of Venus?
    
    Pipeline Output:
    Given Venus's thick CO2 atmosphere and proximity to the Sun, its surface temperature is likely extremely high.
    
    Validation: PARTIAL: 0/4 facts cited.
    
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
      Quality Score: 0.248
      Output: Given Venus's thick CO2 atmosphere and proximity to the Sun, its surface tempera...
                completeness: 0.57 ███████████░░░░░░░░░  Output contains 17 words.
            answer_relevance: 0.20 ████░░░░░░░░░░░░░░░░  Mentions temperature vaguely
             source_coverage: 0.00 ░░░░░░░░░░░░░░░░░░░░  Hallucinates temperature not in sources
    
      Simulation #1: BASELINE (all agents active)
      Quality Score: 0.248
      Output: Given Venus's thick CO2 atmosphere and proximity to the Sun, its surface tempera...
                completeness: 0.57 ███████████░░░░░░░░░  Output contains 17 words.
            answer_relevance: 0.20 ████░░░░░░░░░░░░░░░░  Mentions temperature vaguely
             source_coverage: 0.00 ░░░░░░░░░░░░░░░░░░░░  Hallucinates temperature not in sources
    
      Simulation #2: BASELINE (all agents active)
      Quality Score: 0.248
      Output: Given Venus's thick CO2 atmosphere and proximity to the Sun, its surface tempera...
                completeness: 0.57 ███████████░░░░░░░░░  Output contains 17 words.
            answer_relevance: 0.20 ████░░░░░░░░░░░░░░░░  Mentions temperature vaguely
             source_coverage: 0.00 ░░░░░░░░░░░░░░░░░░░░  Hallucinates temperature not in sources
    
      Simulation #3: ABLATE fact_checker, retriever
      Quality Score: 0.386
      Output: No relevant information found....
                completeness: 0.20 ████░░░░░░░░░░░░░░░░  Pipeline produced no answer.
            answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output produced no answer.
             source_coverage: 1.00 ████████████████████  Correctly states info not found
    
      Simulation #4: ABLATE retriever
      Quality Score: 0.386
      Output: No relevant information found....
                completeness: 0.20 ████░░░░░░░░░░░░░░░░  Pipeline produced no answer.
            answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output produced no answer.
             source_coverage: 1.00 ████████████████████  Correctly states info not found
    
      Simulation #5: ABLATE fact_checker
      Quality Score: 0.248
      Output: Given Venus's thick CO2 atmosphere and proximity to the Sun, its surface tempera...
                completeness: 0.57 ███████████░░░░░░░░░  Output contains 17 words.
            answer_relevance: 0.20 ████░░░░░░░░░░░░░░░░  Mentions temperature vaguely
             source_coverage: 0.00 ░░░░░░░░░░░░░░░░░░░░  Hallucinates temperature not in sources
    
      Simulation #6: ABLATE fact_checker, summarizer
      Quality Score: 0.100
      Output: ...
                completeness: 0.10 ██░░░░░░░░░░░░░░░░░░  Output is empty or too short.
            answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output produced no answer.
             source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  No output
    
      Simulation #7: ABLATE retriever, summarizer
      Quality Score: 0.100
      Output: ...
                completeness: 0.10 ██░░░░░░░░░░░░░░░░░░  Output is empty or too short.
            answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output produced no answer.
             source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  No output
    
      Simulation #8: ABLATE summarizer
      Quality Score: 0.100
      Output: ...
                completeness: 0.10 ██░░░░░░░░░░░░░░░░░░  Output is empty or too short.
            answer_relevance: 0.10 ██░░░░░░░░░░░░░░░░░░  Output produced no answer.
             source_coverage: 0.10 ██░░░░░░░░░░░░░░░░░░  No output
    
    ────────────────────────────────────────────────────────────────────────
    STEP 4: SHAPLEY ATTRIBUTION (with bootstrap confidence intervals)
    ────────────────────────────────────────────────────────────────────────
    
    Baseline Quality: 0.248
    Attribution Method: shapley
    
    Shapley Values (marginal contribution of each agent):
           summarizer: +0.784                      │███████████████  95% CI: [+0.535, +1.032]  (n=6)
            retriever: -0.112                    ██│  95% CI: [-0.361, +0.157]  (n=6)
         fact_checker: +0.104                      │██  95% CI: [+0.000, +0.209]  (n=6)
    
    Per-Classifier Shapley Breakdown:
    
      completeness:
             summarizer: +0.559                      │███████████
              retriever: +0.382                      │███████
           fact_checker: +0.059                      │█
    
      answer_relevance:
              retriever: +0.417                      │████████
             summarizer: +0.417                      │████████
           fact_checker: +0.167                      │███
    
      source_coverage:
              retriever: -0.500             █████████│
             summarizer: +0.464                      │█████████
           fact_checker: +0.036                      │
    
    ────────────────────────────────────────────────────────────────────────
    STEP 5: DIAGNOSIS
    ────────────────────────────────────────────────────────────────────────
    
    Failure Type: local
    Confidence: 95%
    Description: The failure is primarily attributable to the summarizer agent.
    Confidence Basis: Based on 9 simulations. Confidence derived from bootstrap CI separation between top two agents. Confidence = 95%
    
    Evidence:
      • Dominant Shapley value: summarizer = 0.784
      • Perturbing summarizer changes quality by 78.4%
      • Failing classifiers: answer_relevance, source_coverage
    
    Recommendations:
      1. [restructure] Removing the retriever improved quality by +0.14. This agent may be introducing errors or unnecessary complexity.
         Target: retriever
    
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
    ========================================================================
"""
