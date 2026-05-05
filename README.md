# counterfact

**Deterministic, evidence-driven diagnostics for multi-agent AI systems.**

`counterfact` is a drop-in replacement for [LangGraph](https://github.com/langchain-ai/langgraph) that automatically instruments your multi-agent pipelines with execution tracing, ground-truth-free evaluation, and real counterfactual analysis.

## How It Works

Counterfactual analysis works by **actually re-running your pipeline** with agents ablated (replaced by no-ops). For each agent, we remove it, re-execute the full pipeline, and measure how output quality changes. This gives real, not simulated, attribution scores.

```
Baseline:     [Retriever] → [Synthesizer] → [Critic] → output (quality: 0.85)
Ablate Retriever:  [no-op] → [Synthesizer] → [Critic] → output (quality: 0.31)
Ablate Synthesizer: [Retriever] → [no-op] → [Critic] → output (quality: 0.42)
Ablate Critic: [Retriever] → [Synthesizer] → [no-op] → output (quality: 0.78)

→ Retriever has the highest impact on quality (Shapley: -0.54)
```

## Key Features

- **Drop-in Integration** — Replace `from langgraph.graph import StateGraph` with `from counterfact import StateGraph`. Everything works the same, plus you get diagnostics.
- **Real Counterfactual Analysis** — Actually re-runs your pipeline with agents ablated. No LLM simulation, no guessing.
- **Ground-Truth-Free Evals** — Structural and consistency checks that don't require labeled data: empty outputs, schema violations, latency anomalies, inter-agent coherence, and more.
- **Shapley Attribution** — Shapley value and leave-one-out analysis to identify *which agent* caused a pipeline failure.
- **Failure Classification** — Automatic categorization: local failure, systemic failure, architectural gap, feedback amplification.
- **Actionable Recommendations** — Evidence-based fix suggestions derived from real simulation data.

## Installation

```bash
pip install counterfact
```

With optional provider support (needed for quality classifiers):

```bash
pip install counterfact[google]      # Google Gemini support
pip install counterfact[anthropic]   # Anthropic Claude support
pip install counterfact[all]         # All providers + CLI
```

> **Try it now**: Run `python examples/quickstart.py` for a self-contained demo. (Set `ANTHROPIC_API_KEY` to see the engine use LLM-based grounding to catch a hallucination, or run without a key to see a graceful deterministic fallback).

## Quick Start

```python
from counterfact import StateGraph, END

# Define your pipeline exactly as you would with LangGraph
graph = StateGraph(dict)
graph.add_node("retriever", retriever_fn)
graph.add_node("synthesizer", synthesizer_fn)
graph.add_edge("retriever", "synthesizer")
graph.add_edge("synthesizer", END)
graph.set_entry_point("retriever")

pipeline = graph.compile()

# Run your pipeline (works exactly like LangGraph)
result = pipeline.invoke({"query": "What is quantum computing?"})

# Get the execution trace (automatic, zero instrumentation)
trace = pipeline.get_trace()

# Run ground-truth-free evaluation (no LLM needed for Tier 1)
eval_suite = pipeline.eval(final_output=result["output"])

# Run full counterfactual diagnosis
# This ACTUALLY RE-RUNS the pipeline with each agent ablated
report = pipeline.diagnose(
    input_state={"query": "What is quantum computing?"},
    domain="rag",
    num_simulations=30,
    llm_fn=my_llm_caller,  # needed for quality classifiers
)

print(report.to_markdown())
```

## Architecture

```
counterfact/
├── graph.py            # Drop-in StateGraph with tracing + clone_with_ablation
├── tracing.py          # Execution trace capture
├── types.py            # Shared data types
├── evals.py            # Ground-truth-free evaluation checks
├── classifiers.py      # Quality scoring with pluggable classifiers
├── discovery.py        # Pipeline structure analysis
├── attribution.py      # Shapley value & LOO attribution
├── perturbation.py     # Real pipeline re-execution engine
├── diagnostics.py      # Full diagnostic orchestrator
├── recommendations.py  # Evidence-based fix generation
├── optimizer.py        # Pipeline optimization engine
├── prompt_analysis.py  # Thinking-model prompt evaluation
├── tool_tracing.py     # Tool call capture and analysis
├── export.py           # Report export (markdown, JSON, HTML)
├── llm.py              # LLM abstraction layer
├── async_engine.py     # Async execution engine
└── cli.py              # CLI (eval + discover commands)
```

## CLI

The CLI supports evaluation and discovery. Full counterfactual diagnostics require the actual pipeline and are available via the Python API.

```bash
# Run ground-truth-free evaluation checks on a saved trace
counterfact eval trace.json --domain rag

# Enable Tier 2 checks with an LLM provider
counterfact eval trace.json --provider anthropic

# Discover pipeline topology from raw logs
counterfact discover logs.txt
```

## Development

```bash
# Clone the repo
git clone https://github.com/counterfact-labs/counterfact.git
cd counterfact

# Install in development mode
pip install -e ".[all]"

# Run tests
pytest --cov=counterfact --cov-fail-under=95

# Lint
ruff check counterfact/
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## About

Built by [Counterfact Labs](https://github.com/counterfact-labs) — the causal intelligence layer for AI systems.
