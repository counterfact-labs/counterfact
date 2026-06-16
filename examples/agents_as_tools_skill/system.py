"""The agents-as-tools system, wired for counterfact via the sequential adapter.

In the OpenAI Agents SDK, "agents as tools" means a top agent invokes sub-agents
as callable tools. counterfact diagnoses that by running each tool-agent as a
discrete, ablatable step in a chain:

    assistant --> fact_lookup --> unit_converter --> composer

Each node reads the running answer and writes it forward, so ablating any one
tool removes exactly that tool's contribution. The bug lives in ``unit_converter``
(buggy instruction converts feet->meters with the wrong factor); ``FIXED`` holds
the corrected instruction the debugging loop can swap in.
"""

from __future__ import annotations

import re

from counterfact.integrations.openai_agents import graph_from_sequential

# Reuse the deterministic SDK stand-in from the orchestrator case study.
from openai_agents_skill.agents_shim import Agent, Runner

BUGGY_CONVERTER = "convert:buggy"
FIXED_CONVERTER = "convert:fixed"

INSTRUCTIONS = {"unit_converter": BUGGY_CONVERTER}
FIXED = {"unit_converter": FIXED_CONVERTER}
FIXABLE = ["fact_lookup", "unit_converter", "composer"]


def _assistant(instructions: str, query: str) -> str:
    # The planner just passes the question through to the tool chain.
    return query


def _fact_lookup(instructions: str, text: str) -> str:
    # Looks up the height in feet AND the correct metric conversion (330 m).
    return f"{text} || feet:1083 || meters:330"


def _unit_converter(instructions: str, text: str) -> str:
    m = re.search(r"feet:(\d+)", text)
    if not m:
        return text
    feet = int(m.group(1))
    # Correct factor is 0.3048 m/ft; the buggy instruction uses 0.5. The converter
    # appends its own figure, which the composer prefers over the looked-up value.
    factor = 0.3048 if instructions == FIXED_CONVERTER else 0.5
    return f"{text} || converted:{round(feet * factor)}"


def _composer(instructions: str, text: str) -> str:
    # Prefer the converter's figure; fall back to the looked-up metric value.
    m = re.search(r"converted:(\d+)", text) or re.search(r"meters:(\d+)", text)
    meters = m.group(1) if m else "unknown"
    return f"The tower is approximately {meters} meters tall."


_BEHAVIORS = {
    "assistant": _assistant,
    "fact_lookup": _fact_lookup,
    "unit_converter": _unit_converter,
    "composer": _composer,
}


def build_system(instructions: dict | None = None):
    """Build the compiled counterfact graph for the agents-as-tools chain."""
    instr = {**INSTRUCTIONS, **(instructions or {})}
    agents = [
        Agent("assistant", "assistant:v1", _BEHAVIORS["assistant"]),
        Agent("fact_lookup", "lookup:v1", _BEHAVIORS["fact_lookup"]),
        Agent("unit_converter", instr["unit_converter"], _BEHAVIORS["unit_converter"]),
        Agent("composer", "composer:v1", _BEHAVIORS["composer"]),
    ]
    return graph_from_sequential(agents, runner=Runner.run_sync)
