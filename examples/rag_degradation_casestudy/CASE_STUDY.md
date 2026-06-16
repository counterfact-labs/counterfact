# When ablation misleads: finding the quality lever in a RAG pipeline

This case study shows why counterfact supports graded degradation, not just
ablation. It runs the [`counterfact-debugger`](../../skills/counterfact-debugger/)
skill against a three-stage RAG pipeline and is offline and deterministic (no API
keys), so it reproduces exactly:

```bash
PYTHONPATH=examples python -m rag_degradation_casestudy.run_casestudy
PYTHONPATH=examples python -m rag_degradation_casestudy.make_report
```

## The system

A standard retrieval pipeline:

```
retriever --> reranker --> synthesizer
```

The retriever returns ranked passages, the reranker orders them, and the
synthesizer reads only the top 2 passages (a stand-in for an LLM with a limited
context budget) to write the answer. The eval rule is that the answer must
contain the exact figure from the filing.

This pipeline passes its eval on all five questions. So the question is not "what
is broken." It is the question teams actually face once a pipeline works: which
module's quality is the real lever, and where is the pipeline fragile?

## Ablation gives a misleading answer

Replacing each node with a no-op and re-running (the default counterfactual)
produces this Shapley attribution:

| node | ablation Shapley |
|---|---|
| retriever | +0.50 |
| synthesizer | +0.50 |
| reranker | +0.00 |

Read literally, this says the retriever and synthesizer matter and the reranker
is dead weight. The reranker scores zero because removing it is a harmless
pass-through: the retriever already happens to return the relevant passage near
the top on these cases, so skipping the reranker changes nothing. A team acting
on this would stop tuning the reranker and pour effort into the retriever.

## Graded degradation gives the right answer

Instead of removing each module, degrade its output across a range of magnitudes
(magnitude 1.0 is full ablation) and watch how answer quality responds:

| node | classification | what the curve shows |
|---|---|---|
| retriever | structural | quality holds under partial degradation and only falls when the retriever is fully removed |
| reranker | quality_driver | decaying the ranking even slightly pushes the relevant passage out of the top-2 window, and answers fail |
| synthesizer | quality_driver | degrading the written answer drops the figure |

The reranker, which ablation rated a flat zero, is in fact a quality driver. The
retriever, which ablation rated important, is merely structural: as long as it
returns the relevant passage somewhere, its ranking quality does not change the
answer; the reranker is what decides whether the relevant passage lands inside
the synthesizer's context window.

This is the distinction ablation cannot make. A retriever and a reranker can both
show up as "necessary" (or, worse, the reranker as "irrelevant") under ablation,
while only one of them is where answer quality is actually won or lost.

## Why this matters

For a team running evals on a retrieval system, the practical takeaway is that
the lever is ranking quality, not retrieval recall, on these cases. counterfact
reaches that conclusion by re-running the real pipeline under controlled
degradation, not by reading a trace or trusting a single on/off ablation.

## How to run it on a real system

`graph.diagnose_sensitivity(...)` (or the skill runner with `--sensitivity`)
takes the same factory and inputs as ordinary diagnosis. Built-in degraders are
selected per module type (drop ranked items for a retriever, decay order for a
reranker, drop sentences for a generator); override any of them per node with a
custom degrader `(value, magnitude, rng) -> value`. See
[`reference/ablation-vs-degradation.md`](../../skills/counterfact-debugger/reference/ablation-vs-degradation.md).
