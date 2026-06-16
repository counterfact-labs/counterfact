# Ablation vs severe degradation

`diagnose` attributes a failure by **removing** each node from coalitions and measuring how
quality moves (Shapley/LOO). How a node is "removed" is chosen automatically per node — you do
not configure it.

## The problem with ablating everything

Ablation replaces a node with a no-op. For most agents that answers the right question: *is
this node load-bearing?* For some modules it does not.

Ablate a **retriever** and the synthesizer gets no context at all, so the whole pipeline
collapses to a degenerate state. The retriever's Shapley value is huge and positive, but all it
tells you is "the pipeline needs a retriever" — not whether *retrieval quality* is what is
dragging answers down. Parsers, rerankers, and context builders have the same failure mode:
removing them breaks the run rather than revealing a quality effect, so they trivially dominate
attribution and drown out the node that actually matters.

## What diagnose does instead

For those modules, `diagnose` applies **one severe, structure-preserving degradation** as the
"removal":

- the node still runs, so the pipeline stays runnable;
- its output keeps its **shape** — a retriever still returns a non-empty doc list, a parser
  still returns its keys;
- the **content** is destroyed (docs replaced with low-relevance placeholders, parsed values
  blanked).

This simulates the module contributing nothing useful *without* the structural collapse, so its
Shapley value reflects the quality it contributes, comparable to the other agents.

## How the strategy is chosen

Per node, by inferred **module type** (from its name and the shape of its output, captured in a
single pipeline run):

| inferred type | removal |
|---|---|
| retriever (name hints `retriev/search/fetch/lookup/context/doc/chunk`, or list output) | **degrade** |
| reranker (`rerank/rank/order/sort/score`, list output) | **degrade** |
| parser (`pars/extract/structur/classif/route/triage`, dict output) | **degrade** |
| generator (everything else, e.g. a synthesizer/writer) | **ablate** |

No magnitude sweep and no separate method — there is exactly one severe level, and `diagnose`
is the only entry point. The choice is reported: read
`report.simulation_results_summary["removal_strategies"]` (a `{node: "ablate"|"degrade"}` map),
and `cf_diagnose.py` prints which modules were degraded.

## Reading the result

A structural module that comes back with a **large** Shapley under degradation is genuinely a
quality lever (its content quality changes the answer). One that comes back **near zero** is
load-bearing but quality-insensitive on these cases — pure ablation could not have told you the
difference, because it would have collapsed the pipeline either way.

If your pipeline names a structural module unconventionally (so it is inferred as a generator
and ablated), rename it to include a hint above, or treat a suspiciously dominant ablation
result on that node as a sign to look closer.
