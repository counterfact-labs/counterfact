# Debugging an agents-as-tools system with counterfact

This case study covers the third OpenAI Agents SDK pattern: agents-as-tools, where
a top agent calls sub-agents as callable tools. It is offline and deterministic:

```bash
PYTHONPATH=examples python -m agents_as_tools_skill.run_casestudy
```

## The system

An assistant answers a question by calling two sub-agent tools and a composer:

```
assistant --> fact_lookup (tool) --> unit_converter (tool) --> composer
```

counterfact runs each tool-agent as a discrete, ablatable node rather than relying
on the model's own tool-calling loop, which is what lets it remove one tool and
re-run to measure that tool's contribution.

## The symptom

Asked "How tall is the tower in meters?", the system answers **542 meters**. The
correct answer is **330 meters** (1083 feet). Every tool ran without error, so the
trace alone does not say which one is wrong.

## What counterfact found

Ablating each tool-agent and re-running gives:

| tool | Shapley |
|---|---|
| `fact_lookup` | +0.50 |
| `assistant` | +0.00 |
| `composer` | -0.25 |
| `unit_converter` | -0.25 |

`fact_lookup` is positive: it supplies the correct figures, so removing it hurts.
The negative scores fall on `unit_converter` and the `composer` that surfaces its
output, which is the signature of a tool whose presence makes the answer worse.
The debugging loop fixes the most implicated editable tool, `unit_converter`
(its instruction used a wrong feet-to-meters factor), and the answer becomes
**330 meters**, confirming the attribution.

## Why this is the agents-as-tools case

The same counterfact machinery handles all three OpenAI Agents SDK patterns. The
sequential and orchestrator-with-handoffs patterns are covered in
[`openai_agents_skill`](../openai_agents_skill/CASE_STUDY.md); this one shows that
a top agent calling sub-agents as tools is just another wiring of ablatable nodes.
For a tool that is retrieval-like (a search sub-agent), prefer graded degradation
over ablation; see [`rag_degradation_skill`](../rag_degradation_skill/CASE_STUDY.md).
