# Ablation vs graded degradation

## The problem with pure ablation

Ablation replaces a node with a no-op and re-runs the pipeline. That answers one question:
*is this node load-bearing?* For many modules that is exactly the wrong question.

Ablate a **retriever** and the synthesizer gets no context at all, so the whole pipeline
structurally collapses. The retriever's Shapley value is huge and positive, but all it tells
you is "the pipeline needs a retriever" — not whether *retrieval quality* is what is dragging
your answers down. The same applies to parsers, query rewriters, routers, and context builders:
removing them breaks the run rather than revealing a quality effect.

## Degradation generalizes ablation

A degrader takes a node's output and a **magnitude** in `[0, 1]` and returns a progressively
worse version. `magnitude=0` is unchanged; `magnitude=1.0` is true ablation (the node
contributes nothing). So ablation is just the endpoint of a spectrum. By sweeping several
magnitudes and re-running the real pipeline at each, you get a **dose-response curve** for every
node, which is far more informative than a single on/off ablation.

Run it with the skill runner:
```bash
python scripts/cf_diagnose.py --factory ... --inputs ... --sensitivity \
  --magnitudes 0.25,0.5,0.75,1.0 --out sensitivity.json
```
or directly in code:
```python
report = graph.diagnose_sensitivity(input_state, quality_fn=my_quality_fn,
                                     magnitudes=(0.25, 0.5, 0.75, 1.0), seed=42)
```

## The four classifications

Each node is labeled from the shape of its curve:

| class | curve shape | meaning | what to do |
|---|---|---|---|
| **quality_driver** | quality falls smoothly as the node degrades | the node's *output quality* drives the answer | invest here (better retrieval/ranking/prompt) |
| **structural** | flat under partial degradation, collapses only at full removal (or errors) | the node is required to run, but ablation is the blunt, uninformative signal | look elsewhere for quality wins |
| **harmful** | quality *rises* as the node degrades/removed | the node is actively hurting the answer | fix its instruction or remove it |
| **robust** | quality barely moves even at full degradation | low impact on this metric | ignore for this failure |

The key discrimination: a retriever that comes back **structural** means "needed, but partial
quality loss does not hurt — your problem is elsewhere," whereas **quality_driver** means
"retrieval quality is exactly what is limiting answers." Pure ablation cannot tell these apart;
both look like a big positive Shapley.

## The degrader library

Built-in degraders are auto-selected by an inferred module type (name + output shape):

- **retriever** (list output) → `drop_items` — keep only a `(1-magnitude)` prefix of ranked docs.
- **reranker** (list, name hints `rank`/`rerank`/`order`) → `shuffle_relevance` — push good items toward the back.
- **parser** (dict output) → `drop_fields` — drop a fraction of extracted fields.
- **generator** (string output) → `drop_sentences` — drop a fraction of sentences.

Others available to import from `counterfact.sensitivity`: `inject_distractors` (replace a
fraction of retrieved items with off-topic filler — quality loss without quantity loss) and
`truncate_text`.

## Custom degraders

When the built-ins do not model your failure (e.g. you want *hard* distractors specific to your
domain, or to degrade a particular state key), supply your own. A degrader is
`(value, magnitude, rng) -> value`:

```python
# my_degraders.py
from counterfact.sensitivity import inject_distractors

def build():
    return {
        "retriever": inject_distractors("A plausible but wrong passage about a different company."),
    }
```
```bash
python scripts/cf_diagnose.py --factory ... --inputs ... --sensitivity \
  --degraders my_degraders:build
```
Use `target_keys={node: "state_key"}` (in code) to force which output key is degraded; by
default the node's largest changed output is used.

## When to reach for which

- Default to **ablation** (`diagnose`) for agent chains where each step writes prose and you
  want Shapley/coalition attribution and a failure classification.
- Reach for **degradation** (`--sensitivity`) whenever a suspect is a retriever, reranker,
  parser, router, or context builder — anything whose full removal would just break the run.
- If an ablation run prints the `HINT:` that the top agent's removal looks structural, that is
  the signal to re-run with `--sensitivity`. Degradation subsumes ablation, so when in doubt it
  is the safer lens.
