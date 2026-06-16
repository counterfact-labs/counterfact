# Why diagnose degrades a retriever instead of ablating it

This case study shows the removal strategy `diagnose` uses on a retrieval pipeline.
It is offline and deterministic (no API keys), so it reproduces exactly:

```bash
PYTHONPATH=examples python -m rag_degradation_casestudy.run_casestudy
PYTHONPATH=examples python -m rag_degradation_casestudy.make_report
```

## The system

A standard three-stage pipeline. The synthesizer answers from the top-2 retrieved
passages, so it genuinely depends on retrieval: with no context there is nothing
to answer from.

```
retriever --> reranker --> synthesizer  (reads top 2)
```

## The problem with ablating everything

To attribute a failure, `diagnose` removes each node from coalitions and measures
the quality change. If every removal is a plain ablation (a no-op), removing the
retriever leaves the synthesizer with no passages and the run **structurally
fails**. Across the five cases, pure ablation produces this:

| node | pure-ablation Shapley |
|---|---|
| retriever | +0.50 |
| synthesizer | +0.50 |
| reranker | +0.00 |

with **30 of 105 coalition runs ending in a pipeline error**. The retriever's
score only says "the pipeline needs a retriever," and the reranker reads as
irrelevant (removing it is a harmless pass-through), so the attribution is both
distorted by errors and blind to ranking quality.

## What diagnose does instead

`diagnose` classifies each node by type and removes it accordingly: the retriever
and reranker are **severely degraded** (they still run and return a non-empty doc
list, but the content is replaced with low-relevance placeholders), while the
synthesizer (a generator) is ablated. Re-running the same cases:

| node | auto Shapley | strategy |
|---|---|---|
| retriever | +0.33 | degrade |
| reranker | +0.33 | degrade |
| synthesizer | +0.33 | ablate |

with **0 of 105 structural failures**. Every coalition run stays live, so the
retriever and reranker are measured as real quality contributors rather than
collapsing the pipeline. The reranker, invisible under pure ablation (+0.00), now
shows the contribution it actually makes.

## Takeaway

Removing a structural module by no-op answers "is it load-bearing" and, when the
rest of the pipeline depends on it, just breaks the run. Severely degrading it
(destroy the content, keep the shape) answers the more useful question, how much
the module's output quality is worth, without the structural failure. `diagnose`
makes that choice automatically, by inferred module type; the strategy it used is
in `report.simulation_results_summary["removal_strategies"]`. See
[`reference/ablation-vs-degradation.md`](../../skills/counterfact-debugger/reference/ablation-vs-degradation.md).
