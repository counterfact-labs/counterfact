# Reading the attribution — Shapley values, CIs, and when NOT to trust them

Attribution answers "how much does each agent contribute to (or subtract from) output
quality?" It is computed by **really re-running the pipeline** with agents ablated, not by
asking an LLM. But the numbers are estimates from a finite number of simulations, so read
them with the guardrails below.

## The fields

- `shapley_values: {agent -> float}` — marginal contribution of each agent to quality.
  - **Positive** = the agent helps; ablating it lowers quality.
  - **Negative** = the agent *hurts*; the pipeline does better without it (a real bug signal).
  - Magnitude = how much quality moves (roughly in quality-score units).
- `shapley_cis: {agent -> {ci_low, ci_high, n_samples}}` — 95% bootstrap confidence interval
  for each Shapley value. **This is the field that decides whether a number is trustworthy.**
- `attribution_method` — `"loo"` (leave-one-out), `"shapley"`, or `"quality_gate"`. The engine
  starts with the cheaper LOO and falls back to full Shapley when LOO is inconclusive; seeing
  `"shapley"` often means the agents interact (order/coalition matters). `"quality_gate"` means
  baseline quality was already high enough that attribution was skipped (paired with
  `failure_type: "no_failure"`) — common in an *after-the-fix* report and a good success signal.
- `per_classifier_shapley: {classifier -> {agent -> float}}` — the same attribution broken
  down by quality dimension. Use it to learn *which* aspect of quality the culprit harms.
- `baseline_quality` / `baseline_quality_ci` — quality of the pipeline as-is.

## Guardrails — do NOT name a culprit when…

1. **The CI straddles zero.** If `ci_low < 0 < ci_high` for an agent, its sign is not
   established. You cannot say it helps or hurts. Raise `--num-simulations` and re-run.
2. **The top two CIs overlap.** If the #1 and #2 agents' confidence intervals overlap, you
   can't claim #1 is the dominant cause over #2. Either report both as candidates or gather
   more simulations until they separate.
3. **`n_samples` is tiny.** A CI built from 2–3 samples is noise. Prefer `n_samples >= 5`
   per agent before acting; that usually means `--num-simulations` of 30+ for a 3-agent
   pipeline, more for larger graphs.
4. **The quality metric is weak.** If you fell back to built-in `--domain rag` classifiers on
   a non-RAG pipeline, low-magnitude values may just mean the metric can't see the failure.
   Scaffold a targeted classifier (see `report-schema.md`) before trusting small deltas.

## A worked reading (from the quickstart fixture)

```
summarizer:   +0.784   95% CI: [+0.535, +1.032]  (n=6)
retriever:    -0.112    95% CI: [-0.361, +0.157]  (n=6)
fact_checker: +0.104    95% CI: [+0.000, +0.209]  (n=6)
```
- `summarizer` is clearly the highest-impact agent (CI well above zero, no overlap with the
  others) → `failure_type: local`, dominant agent = summarizer.
- `retriever`'s CI **straddles zero** ([-0.361, +0.157]) → do **not** claim the retriever
  helps or hurts; its effect is not established at this sample size.
- Per-classifier breakdown shows `source_coverage` for the retriever is `-0.500`: the
  retriever specifically hurts grounding even though its overall effect is ambiguous. That's
  the kind of dimension-specific signal `per_classifier_shapley` exists to surface.

## "Dominant" is not always the culprit

`classification.dominant_agent` is the agent with the largest **magnitude** Shapley value.
That is not always the thing to fix:

- A strongly **negative** agent actively *hurts* — the pipeline scores higher with it
  ablated. This is the clearest "fix me / remove me" signal, even if its magnitude is
  smaller than some positive agent's.
- A strongly **positive** agent is essential, but in a *failing* pipeline it can still be
  the culprit because it does its essential job badly (e.g. a summarizer that hallucinates
  has high positive impact *and* is the problem).

So don't reflexively edit `dominant_agent`. Check the sign, then disambiguate with
`per_classifier_shapley` and the actual `simulation_details` outputs: if ablating an agent
*raises* quality, fixing or removing that agent is the lead. If quality only comes from one
positive agent but the output is still bad, that agent's internal behavior is the lead.

## When attribution is inconclusive

If the headline classification is low-confidence, the top CIs overlap, or
`attribution_method` flips to `shapley` with still-wide intervals: **raise
`--num-simulations`** (50–100) and re-run before drawing conclusions. More simulations
narrow the CIs. Guessing past an inconclusive result is exactly the failure mode this tool
exists to prevent.
