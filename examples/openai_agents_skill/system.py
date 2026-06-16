"""The system under test: a customer-support triage built with the OpenAI
Agents SDK pattern (orchestrator + handoffs), wrapped for counterfact.

Topology (orchestrator + handoffs)::

    triage --(handoff)--> {billing | technical | account} --> compliance_editor --> reply

The case study dataset is all *billing refund* tickets, so the orchestrator
hands every ticket to the ``billing`` specialist, then a ``compliance_editor``
finalizes the customer-facing reply.

The planted bug is subtle and realistic: the ``billing`` specialist correctly
states the exact refund amount, but the downstream ``compliance_editor`` —
trying to be "policy-compliant" — rewrites the reply into a generic
acknowledgement that **drops the dollar figure**. The naive suspect is the
billing specialist ("it owns the amount"); the actual culprit is the downstream
editor. counterfact's ablation tells them apart.

``INSTRUCTIONS`` holds the current (buggy) instruction per editable agent;
``FIXED`` holds the corrected instruction the debugging loop can swap in.
"""

from __future__ import annotations

import re

from counterfact.integrations.openai_agents import graph_from_orchestrator

from .agents_shim import Agent, Runner

# Which specialist handles a ticket. Deterministic keyword routing stands in for
# the orchestrator model's handoff decision.
_CATEGORIES = ("billing", "technical", "account")


def _triage_behavior(instructions: str, ticket: str) -> str:
    text = ticket.lower()
    if any(k in text for k in ("refund", "charge", "charged", "invoice", "billing")):
        return "billing"
    if any(k in text for k in ("crash", "error", "bug", "login", "password")):
        return "technical"
    return "account"


_AMOUNT_RE = re.compile(r"\$[\d,]+\.\d{2}")


def _extract_amount(ticket: str) -> str:
    m = _AMOUNT_RE.search(ticket)
    return m.group(0) if m else "the charged amount"


def _billing_behavior(instructions: str, ticket: str) -> str:
    """Billing specialist — correctly resolves the ticket WITH the exact amount."""
    amount = _extract_amount(ticket)
    return f"Refund of {amount} approved for this order; it will post in 5-7 business days."


def _technical_behavior(instructions: str, ticket: str) -> str:
    return "Cleared the session cache and reset the login token; please sign in again."


def _account_behavior(instructions: str, ticket: str) -> str:
    return "Updated the account profile and verified the contact email on file."


# ── The editable agent instructions (current = buggy) ──────────────────────
BUGGY_COMPLIANCE = "compliance:strip"
FIXED_COMPLIANCE = "compliance:preserve"

INSTRUCTIONS = {
    "billing": "billing:v1",
    "technical": "technical:v1",
    "account": "account:v1",
    "compliance_editor": BUGGY_COMPLIANCE,
}

FIXED = {
    # The one corrected instruction the debugging loop should discover it needs.
    "compliance_editor": FIXED_COMPLIANCE,
}

# Agents the developer is allowed to edit (the orchestrator's routing is fixed).
FIXABLE = ["billing", "technical", "account", "compliance_editor"]


def _compliance_behavior(instructions: str, resolution: str) -> str:
    """Finalizer — composes the customer reply from the specialist resolution.

    BUGGY (``compliance:strip``): rewrites into a generic, figure-free
    acknowledgement "for compliance" — silently dropping the dollar amount.
    FIXED (``compliance:preserve``): keeps the specialist's specifics verbatim.
    """
    if instructions == FIXED_COMPLIANCE:
        return f"Hello, thanks for reaching out. {resolution} This was handled in line with our refund policy."
    # Buggy: strips the specifics.
    return "Hello, thanks for reaching out. Your request has been reviewed and processed in accordance with our policy. A confirmation email will follow shortly."


_BEHAVIORS = {
    "triage": _triage_behavior,
    "billing": _billing_behavior,
    "technical": _technical_behavior,
    "account": _account_behavior,
    "compliance_editor": _compliance_behavior,
}


def build_system(instructions: dict | None = None):
    """Build the compiled counterfact graph for the support system.

    Args:
        instructions: Optional override of editable-agent instructions (defaults
            to the current, buggy ``INSTRUCTIONS``). The debugging loop passes a
            patched copy here to apply a fix.

    Returns:
        A compiled ``CounterfactualGraph`` (orchestrator + handoffs).
    """
    instr = {**INSTRUCTIONS, **(instructions or {})}

    triage = Agent("triage", "triage:v1", _BEHAVIORS["triage"])
    specialists = {
        cat: Agent(cat, instr[cat], _BEHAVIORS[cat]) for cat in _CATEGORIES
    }
    compliance = Agent("compliance_editor", instr["compliance_editor"], _BEHAVIORS["compliance_editor"])

    return graph_from_orchestrator(
        triage,
        specialists,
        finalizer=compliance,
        runner=Runner.run_sync,          # swap for the real agents.Runner to go live
        finalizer_name="compliance_editor",
    )
