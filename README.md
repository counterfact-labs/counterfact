# counterfact

[![CI](https://github.com/counterfact-labs/counterfact/actions/workflows/ci.yml/badge.svg)](https://github.com/counterfact-labs/counterfact/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

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
- **Smart removal (ablate or degrade)** — When full ablation is too blunt (e.g. a retriever, whose removal just collapses the pipeline), `diagnose` automatically *severely degrades* that module instead — destroying its output content while preserving its shape — so its attribution reflects quality, not mere necessity. See [Removal strategy](#removal-strategy-ablate-or-degrade).
- **Ground-Truth-Free Evals** — Structural and consistency checks that don't require labeled data: empty outputs, schema violations, latency anomalies, inter-agent coherence, and more.
- **Shapley Attribution** — Shapley value and leave-one-out analysis to identify *which agent* caused a pipeline failure.
- **Failure Classification** — Automatic categorization: local failure, systemic failure, architectural gap, feedback amplification.
- **Actionable Recommendations** — Evidence-based fix suggestions derived from real simulation data.

## Installation

`counterfact` is not yet published to PyPI — install it from GitHub:

```bash
pip install "counterfact @ git+https://github.com/counterfact-labs/counterfact.git"
```

With optional provider support (needed for quality classifiers):

```bash
# Google Gemini support
pip install "counterfact[google] @ git+https://github.com/counterfact-labs/counterfact.git"
# Anthropic Claude support
pip install "counterfact[anthropic] @ git+https://github.com/counterfact-labs/counterfact.git"
# All providers + CLI
pip install "counterfact[all] @ git+https://github.com/counterfact-labs/counterfact.git"
```

> **Try it now**: Run `python examples/quickstart.py` for a self-contained demo. (Requires `ANTHROPIC_API_KEY` to run LLM-based grounding and catch hallucinations).

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
├── degradation.py      # Removal strategy: ablate, or severely degrade structural modules
├── perturbation.py     # Real pipeline re-execution engine
├── diagnostics.py      # Full diagnostic orchestrator
├── recommendations.py  # Evidence-based fix generation
├── spec.py             # Neutral graph spec IR (build_graph_from_spec)
├── orchestration.py    # Dataset-level diagnosis for external orchestrators
├── optimizer.py        # Pipeline optimization engine
├── prompt_analysis.py  # Thinking-model prompt evaluation
├── tool_tracing.py     # Tool call capture and analysis
├── export.py           # Report export (markdown, JSON, HTML)
├── llm.py              # LLM abstraction layer
├── async_engine.py     # Async execution engine
├── cli.py              # CLI (eval + discover commands)
└── integrations/       # Optional adapters: OpenAI Agents SDK, Braintrust
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

## Agent Skill

counterfact ships as a portable [Agent Skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills)
so a coding agent (Claude Code, the Agent SDK, claude.ai) can debug a failing pipeline
for you — instrument it, run real counterfactual ablation, interpret the Shapley
attribution, and propose + verify a fix.

```
skills/counterfact-debugger/
├── SKILL.md                  # when-to-use + the 6-step debugging workflow
├── reference/                # failure taxonomy, reading attribution, report schema
└── scripts/
    ├── cf_diagnose.py        # factory + inputs -> runs diagnose, writes JSON + markdown
    ├── llm_fn.py             # Anthropic/Google LLM caller from env
    └── verify.py             # compares two reports to confirm a fix moved the attribution
```

Point the runner at a factory that returns your compiled pipeline and the input(s) that fail:

```bash
python skills/counterfact-debugger/scripts/cf_diagnose.py \
  --factory myapp.pipeline:build \
  --inputs cases.json \
  --domain rag --num-simulations 30 --out report.json
```

Install it by copying `skills/counterfact-debugger/` into your agent's skills directory
(e.g. `~/.claude/skills/`). See `SKILL.md` for the full workflow.

**Worked case study:** [`examples/financebench_casestudy/CASE_STUDY.md`](examples/financebench_casestudy/CASE_STUDY.md)
walks the skill through diagnosing a real 8-agent financial-RAG pipeline on FinanceBench
questions — finding the one agent (of four plausible suspects) that actually causes the
failure, fixing it (0/5 → 5/5 exact answers), and showing where an LLM reading the traces
gets it wrong. Fully reproducible: `PYTHONPATH=examples python -m financebench_casestudy.run_casestudy`.

## Removal strategy (ablate or degrade)

To attribute a failure, `diagnose` "removes" each node from coalitions and measures the quality
change. Pure ablation (a no-op) answers "is this module load-bearing?" For a retriever, parser,
or reranker that is uninformative — remove it and the pipeline structurally collapses, so the
module trivially dominates attribution without telling you whether its *quality* is what hurts
answers.

So `diagnose` chooses the removal **per node, automatically**: most agents are ablated, but a
structural module (retriever / reranker / parser) is instead **severely degraded** — it still
runs and its output keeps its shape (a retriever still returns a non-empty doc list), but the
content is destroyed. Its Shapley value then reflects quality, not mere necessity. No extra
method or configuration:

```python
report = pipeline.diagnose(input_state={"query": "..."}, quality_fn=my_quality_fn)
report.simulation_results_summary["removal_strategies"]
# -> {"retriever": "degrade", "reranker": "degrade", "synthesizer": "ablate"}
```

The choice is made by inferred module type (name + output shape). See
`skills/counterfact-debugger/reference/ablation-vs-degradation.md`.

## Integrations

counterfact is framework-neutral. Beyond the drop-in LangGraph `StateGraph`, it
ships optional adapters for other agent frameworks and eval platforms. These are
**additive** — importing them never changes the core API, and existing LangGraph
pipelines are unaffected.

### OpenAI Agents SDK

Wrap an [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) system
(sequential, orchestrator + handoffs, or agents-as-tools) so each agent becomes an
ablatable node. counterfact drives the agents as discrete steps — not the SDK's
internal handoff loop — which is what makes "remove agent X and re-run" meaningful.

```bash
pip install "counterfact[openai-agents] @ git+https://github.com/counterfact-labs/counterfact.git"
```

```python
from agents import Runner
from counterfact.integrations.openai_agents import graph_from_orchestrator

graph = graph_from_orchestrator(
    triage_agent,                                  # the routing/handoff agent
    {"billing": billing_agent, "tech": tech_agent},  # specialists it hands off to
    finalizer=responder_agent,                     # composes the final reply
)
report = graph.diagnose(input_state={"input": ticket}, quality_fn=my_quality_fn)
```

### Braintrust

Use a [Braintrust](https://www.braintrust.dev/) / `autoevals` scorer as the quality
metric that drives Shapley attribution, and pull eval cases straight from a
Braintrust dataset — so your attribution reflects the *same* scorer your evals use.

```bash
pip install "counterfact[braintrust] @ git+https://github.com/counterfact-labs/counterfact.git"
```

```python
from autoevals import Factuality
from counterfact.integrations.braintrust import (
    quality_fn_from_scorer, cases_from_dataset, load_braintrust_dataset,
)

quality_fn = quality_fn_from_scorer(Factuality(), pass_input=True, input_key="input")
cases = cases_from_dataset(load_braintrust_dataset("support", "refunds-eval"))
reports = graph.diagnose_dataset([c["input"] for c in cases], quality_fn=quality_fn)
```

**Worked case studies** (all offline-reproducible, no API keys):

- [`examples/openai_agents_casestudy/`](examples/openai_agents_casestudy/CASE_STUDY.md) — OpenAI Agents SDK **orchestrator-with-handoffs** support system scored by a Braintrust-style scorer. counterfact isolates a downstream agent that silently strips the answer (exonerating the obvious suspect) and fixes it 0/5 → 5/5.
- [`examples/rag_degradation_casestudy/`](examples/rag_degradation_casestudy/CASE_STUDY.md) — a **RAG retriever → reranker → synthesizer** pipeline where pure ablation structurally fails 30/105 coalition runs (and calls the reranker irrelevant), while `diagnose`'s automatic severe-degradation keeps every run live and measures all three modules' contributions.
- [`examples/agents_as_tools_casestudy/`](examples/agents_as_tools_casestudy/CASE_STUDY.md) — OpenAI Agents SDK **agents-as-tools**, attributing a wrong answer to the specific sub-agent tool that caused it.

## Development

```bash
# Clone the repo
git clone https://github.com/counterfact-labs/counterfact.git
cd counterfact

# Install in development mode
pip install -e ".[all]"

# Run tests
pytest --cov=counterfact --cov-fail-under=50

# Lint
ruff check counterfact/
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## About

Built by [Counterfact Labs](https://github.com/counterfact-labs) — the causal intelligence layer for AI systems.
