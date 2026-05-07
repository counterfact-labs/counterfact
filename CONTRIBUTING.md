# Contributing to Counterfact

Thank you for your interest in contributing to Counterfact! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/counterfact-labs/counterfact.git
cd counterfact

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[all]"

# Install development dependencies
pip install ruff mypy pytest pytest-cov pytest-asyncio
```

## Running Tests

```bash
# Run the full test suite with coverage
pytest --cov=counterfact --cov-report=term-missing

# Run a specific test file
pytest counterfact/tests/test_graph.py -v

# Run tests matching a pattern
pytest -k "test_shapley" -v
```

## Code Quality Checks

All of these must pass before submitting a PR:

```bash
# Linting
ruff check counterfact/

# Type checking
mypy counterfact/

# Tests with coverage threshold
pytest --cov=counterfact --cov-fail-under=50
```

You can run all checks at once with:

```bash
bash run_checks.sh
```

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- All public functions and classes must have docstrings (Google-style).
- Type hints are required for all function signatures.
- Imports are organized: stdlib → third-party → local.

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`.
2. **Write tests** for any new functionality. We aim for high coverage.
3. **Run all checks** (`ruff`, `mypy`, `pytest`) before submitting.
4. **Write a clear PR description** explaining what changed and why.
5. **Keep PRs focused** — one logical change per PR.

## Architecture Overview

Before contributing, familiarize yourself with the module structure:

```
counterfact/
├── graph.py          # Drop-in StateGraph replacement (core)
├── tracing.py        # Thread-safe execution trace capture
├── types.py          # Shared dataclasses
├── diagnostics.py    # Diagnostic orchestrator
├── perturbation.py   # Pipeline re-execution with ablations
├── attribution.py    # Shapley values & failure classification
├── classifiers.py    # Pluggable quality classifiers
├── recommendations.py # Automated fix suggestions
├── evals.py          # Ground-truth-free evaluation checks
├── async_engine.py   # Async equivalents of core functions
├── cli.py            # Command-line interface
└── tests/            # Test suite
```

**Key design principles:**
- `graph.py` is the public entry point — keep it lightweight.
- Heavy modules (`diagnostics`, `evals`) are lazily imported.
- All types are defined centrally in `types.py`.
- LLM functions are always injected, never hardcoded.

## Reporting Bugs

Please use [GitHub Issues](https://github.com/counterfact-labs/counterfact/issues) to report bugs. Include:
- Python version and OS
- Minimal reproduction steps
- Expected vs. actual behavior
- Full traceback if applicable

## Feature Requests

We welcome feature requests! Please open an issue with the `enhancement` label and describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
