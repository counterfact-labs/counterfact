"""Offline case study: the OpenAI Agents SDK *agents-as-tools* pattern.

A top "assistant" agent answers a question by calling sub-agents as tools, then a
composer writes the reply:

    assistant --> [ fact_lookup tool ] --> [ unit_converter tool ] --> composer

counterfact runs each tool-agent as a discrete, ablatable node (it does not rely
on the model's own tool-calling loop), so it can attribute a wrong answer to the
specific sub-agent tool that caused it. Here one tool (the unit converter) has a
bug; counterfact's attribution finds it among the tools that all "ran fine."

Deterministic and offline (reuses the agents_shim from the sibling
openai_agents_skill package):

    PYTHONPATH=examples python -m agents_as_tools_skill.run_casestudy
"""
