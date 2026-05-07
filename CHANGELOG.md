# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Prompt analysis** module for detecting anti-patterns.
- **LLM response caching** for reproducible, cost-efficient diagnostic runs.
- **Export utilities** for Markdown diagnostic reports.
- Support for Google Gemini and Anthropic Claude as LLM providers.
