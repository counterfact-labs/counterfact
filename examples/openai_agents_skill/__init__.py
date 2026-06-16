"""Offline, reproducible case study: debugging an OpenAI Agents SDK
orchestrator-with-handoffs system using counterfact + a Braintrust-style scorer.

Everything here runs with no network and no API keys. ``agents_shim`` provides a
tiny deterministic stand-in for ``from agents import Agent, Runner`` so the
*adapter* code path (``counterfact.integrations.openai_agents``) is identical to
real SDK usage — to go live, swap ``agents_shim`` for the real ``agents``
package and drop the injected ``runner=``. The scorer in ``scorer`` mirrors the
autoevals ``scorer(output, expected) -> Score`` interface used by Braintrust.

Run it:

    PYTHONPATH=examples python -m openai_agents_skill.run_casestudy
"""
