# Debugging an OpenAI Agents SDK system with counterfact and Braintrust

This case study runs the [`counterfact-debugger`](../../skills/counterfact-debugger/)
skill against a customer-support system built with the OpenAI Agents SDK
orchestrator-with-handoffs pattern and scored with a Braintrust scorer. It is
offline and deterministic (no API keys, no network), so it reproduces exactly:

```bash
PYTHONPATH=examples python -m openai_agents_skill.run_casestudy
```

## The system

A `triage` agent reads each support ticket and hands off to one of three
specialists. A `compliance_editor` then writes the final customer reply:

```
triage --handoff--> { billing | technical | account } --> compliance_editor --> reply
```

The evaluation set is five billing-refund tickets in the usual Braintrust shape
(`input` plus `expected`), so the orchestrator sends every ticket to the
`billing` specialist. The grading rule is the one a billing team would use: the
exact refund amount has to appear in the customer reply. It is implemented as a
Braintrust/autoevals-style scorer, `refund_amount_scorer(output, expected) -> Score`.

## The symptom

On the baseline pipeline, 0 of 5 replies are correct. Each one is a polite
acknowledgement that leaves out the dollar figure:

> Hello, thanks for reaching out. Your request has been reviewed and processed
> in accordance with our policy. A confirmation email will follow shortly.

## Where the bug could be

Several agents could be responsible. The `triage` agent might have routed to the
wrong specialist; the `billing` specialist owns the refund amount and the amount
is missing; the `account` and `technical` specialists could be interfering; or
the `compliance_editor`, the last agent to touch the text, could be at fault.

Asked to debug from the transcript, an LLM usually blames the `billing` agent on
the grounds that it is responsible for the amount and produced a reply without
it. That is also the first place a human on call tends to look. It is the wrong
answer here.

## What counterfact found

counterfact does not infer the cause from the transcript. It removes one agent
at a time, re-runs the pipeline, and measures how the Braintrust score changes.
Using the scorer as the `quality_fn`, averaged over the failing cases
(`num_simulations=16`, `seed=42`):

| agent | Shapley |
|---|---|
| `compliance_editor` | −0.49 |
| `billing` | +0.00 |
| `account` | +0.15 |
| `technical` | +0.15 |
| `orchestrator` | +0.20 |

The `billing` agent has a non-negative score, so removing it never improves the
result. It produces the correct amount; the reason it does not lift the score is
that something downstream throws the amount away.

That something is the `compliance_editor`, with a score of −0.49. Removing it
raises quality. Its instruction to make the reply policy-compliant rewrites the
specialist's answer into a generic template and drops the exact figure the
billing agent produced. The agent that names the amount is fine; the agent that
erases it is the bug.

## The fix

The debugging loop corrects the agent with the most negative score,
`compliance_editor`, by changing its instruction to keep the specialist's
figures. Re-running the evaluation gives 5 of 5, in a single round.

## Why the trace alone is not enough

The final transcript shows no sign that the `compliance_editor` had the figure
and discarded it. By the time you read the reply, the amount is gone, and the
natural reading is that the billing agent never produced it. Removing the editor
and re-running the pipeline is what shows the figure was present all along. That
is the part counterfact adds on top of an ordinary observability trace.

## Pointing this at a real system

Two changes turn this offline demo into a live debugging session:

1. Agents: delete `agents_shim.py`, import `Agent` and `Runner` from `agents`,
   define real `Agent(...)` objects, and drop the injected `runner=` in
   `system.py` (the adapter then defaults to `agents.Runner.run_sync`). See
   [`counterfact/integrations/openai_agents.py`](../../counterfact/integrations/openai_agents.py).
2. Scorer and dataset: replace `refund_amount_scorer` with a real autoevals
   scorer and load cases from a Braintrust project:

   ```python
   from autoevals import Factuality
   from counterfact.integrations.braintrust import (
       quality_fn_from_scorer, cases_from_dataset, load_braintrust_dataset,
   )

   cases = cases_from_dataset(load_braintrust_dataset("support", "refunds-eval"))
   quality_fn = quality_fn_from_scorer(Factuality(), pass_input=True, input_key="input")
   ```

The rest, including the ablation engine, the Shapley attribution, and the fix
loop, is the same as in the LangGraph workflow.
