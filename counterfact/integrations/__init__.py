"""Optional adapters that connect counterfact to external agent frameworks
and eval platforms.

These modules are intentionally **not** imported by ``counterfact/__init__.py``
so that the core package keeps working with zero extra dependencies (and the
existing LangGraph workflow is completely unaffected). Import the adapter you
need explicitly:

    from counterfact.integrations.openai_agents import graph_from_orchestrator
    from counterfact.integrations.braintrust import quality_fn_from_scorer

Each adapter lazily imports its third-party dependency only when you actually
use the framework-backed default (e.g. the real ``agents.Runner`` or the
``braintrust`` client), so the adapters can be unit-tested and demoed offline
by injecting your own runner / scorer.
"""
