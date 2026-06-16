# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Smart removal strategy in `diagnose` (ablate or severely degrade)** — when
  attribution removes a node, structural modules (retrievers, rerankers, parsers)
  are now **severely degraded** rather than ablated: the node still runs and keeps
  its output shape, but its content is destroyed. This avoids the structural
  collapse a no-op causes for such modules, so their Shapley reflects quality
  rather than mere necessity. The choice is automatic, by inferred module type
  (`counterfact.degradation`), with no new method or configuration; the per-node
  strategy is reported under `simulation_results_summary["removal_strategies"]`.
  The `counterfact-debugger` skill applies this automatically (see
  `reference/ablation-vs-degradation.md`).
- **Two more worked case studies** — `examples/rag_degradation_casestudy/` (retriever
  pipeline where pure ablation structurally fails but auto-degradation keeps runs
  live) and `examples/agents_as_tools_casestudy/` (OpenAI Agents SDK agents-as-tools).

- **OpenAI Agents SDK adapter** (`counterfact.integrations.openai_agents`) — wrap
  a sequential, orchestrator-with-handoffs, or agents-as-tools system so each
  agent becomes an ablatable counterfact node. The runner is injected (defaults
  to `agents.Runner.run_sync`), so systems can be diagnosed offline with a fake
  runner. Install with `counterfact[openai-agents]`.
- **Braintrust adapter** (`counterfact.integrations.braintrust`) — adapt a
  Braintrust/`autoevals` scorer into counterfact's `quality_fn` to drive Shapley
  attribution from the same metric your evals use, and convert Braintrust
  datasets into counterfact cases (`cases_from_dataset`, `load_braintrust_dataset`).
  Install with `counterfact[braintrust]`.
- **Worked case study** (`examples/openai_agents_casestudy/`) — offline, deterministic
  walk-through that debugs an OpenAI Agents SDK orchestrator-with-handoffs support
  system scored by a Braintrust-style scorer (0/5 → 5/5).

Both adapters are additive: the core API and existing LangGraph workflows are
unchanged.

## [0.1.0] - 2026-05-07

### Added

- **Drop-in `StateGraph` replacement** for LangGraph with automatic tracing and counterfactual diagnostics.
- **Shapley value attribution** with bootstrap confidence intervals for rigorous per-agent contribution analysis.
- **Leave-One-Out (LOO) attribution** as a fast initial estimator with automatic escalation to Shapley when inconclusive.
- **Failure classification engine** — classifies failures as `local`, `architectural_gap`, `feedback_amplification`, or `systemic`.
- **Monte Carlo simulation engine** with real pipeline re-execution (not LLM-simulated counterfactuals).
- **Pluggable classifier registry** with built-in classifiers for RAG and decision-making domains.
- **Ground-truth-free evaluation suite** (Tier 1: structural checks, Tier 2: LLM-based consistency checks).
- **Automated recommendation engine** with empirical fix suggestions and agent specifications.
- **Async execution engine** for concurrent LLM simulations.
- **CLI** (`counterfact eval`, `counterfact discover`) for trace analysis.
- **Pipeline topology discovery** from raw logs.
- **Prompt analysis** module for detecting anti-patterns, including prompt-section attribution for thinking models.
- **Tool-call tracing** for capturing and analyzing tool usage in thinking-model pipelines.
- **Pipeline optimizer** for single-objective quality maximization over a search space.
- **Neutral graph spec IR** (`build_graph_from_spec`) and **dataset-level diagnosis** (`diagnose_dataset`) for external orchestrators.
- **LLM response caching** for reproducible, cost-efficient diagnostic runs.
- **Export utilities** for diagnostic reports in Markdown, JSON, and HTML.
- **Portable Agent Skill** (`skills/counterfact-debugger/`) that drives the full instrument → ablate → attribute → fix → verify workflow.
- Support for Google Gemini and Anthropic Claude as LLM providers.
