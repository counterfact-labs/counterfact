"""A tiny deterministic stand-in for the OpenAI Agents SDK.

This mirrors just enough of ``from agents import Agent, Runner`` for an offline,
reproducible case study — no model calls, no network. The shapes match the real
SDK where it matters for the adapter:

  * ``Agent`` has a ``name`` (used by counterfact as the node name) and
    ``instructions`` (which we mutate to apply a "fix").
  * ``Runner.run_sync(agent, input_text)`` returns a ``RunResult`` with a
    ``final_output`` string and a ``last_agent`` reference.

The only non-SDK part is ``behavior``: a pure function ``(instructions, input)
-> str`` that deterministically stands in for the model. In a real system the
model produces ``final_output`` from ``instructions`` + ``input``; here we make
that mapping explicit and deterministic so the case study is byte-stable.

To run against the real SDK instead, delete this module, do
``from agents import Agent, Runner``, define real ``Agent(...)`` objects, and
drop the ``runner=`` argument in ``system.py`` (the adapter then defaults to
``agents.Runner.run_sync``).
"""

from __future__ import annotations

from typing import Callable


class Agent:
    """Deterministic stand-in for agents.Agent."""

    def __init__(self, name: str, instructions: str, behavior: Callable[[str, str], str]):
        self.name = name
        self.instructions = instructions
        self._behavior = behavior

    def run(self, input_text: str) -> str:
        return self._behavior(self.instructions, input_text)


class RunResult:
    """Stand-in for agents.RunResult (the object Runner.run_sync returns)."""

    def __init__(self, final_output: str, last_agent: Agent):
        self.final_output = final_output
        self.last_agent = last_agent


class Runner:
    """Stand-in for agents.Runner."""

    @staticmethod
    def run_sync(agent: Agent, input_text: str) -> RunResult:
        return RunResult(agent.run(input_text), last_agent=agent)
