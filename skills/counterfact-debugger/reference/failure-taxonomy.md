# Failure taxonomy — `classification.failure_type`

`diagnose` classifies every failure into one of five types. The type tells you the
*shape* of the fix. Branch on it before touching the Shapley numbers.

| `failure_type` | What it means | Fix shape |
|---|---|---|
| `no_failure` | Baseline quality is already high (above the quality gate ~0.8). Attribution was skipped. | Nothing to fix. If the user disagrees, the quality metric (classifiers) is too lenient — strengthen it and re-run. |
| `local` | One agent dominates the quality drag; removing/ablating it changes quality far more than any other. `dominant_agent` is set. | Fix that **one** agent: its prompt, its retrieval, its tool use, its parsing. Smallest-change-first. |
| `systemic` | Multiple agents interact to cause the failure — no single dominant culprit; contributions are spread or entangled. | Look at shared context: a common prompt fragment, a state key several agents mishandle, or an ordering problem. Fixing one agent in isolation usually won't move the metric. |
| `architectural_gap` | **No** existing agent is responsible — the pipeline is missing a capability. E.g. nothing checks the premise, nothing verifies a number, nothing grounds the answer. | **Add** an agent/node, don't edit one. `recommendations` with `intervention_type="add_agent"` usually accompany this. |
| `feedback_amplification` | A revision/critic loop is making output worse over iterations. `damping_ratio` is set (>1 means error grows each pass). | Break or damp the loop: cap iterations, fix the critic's signal, or change the stop condition. |

## How to act on each

- **`local`** → Read `per_classifier_shapley[<failing classifier>]` to see *how* the
  dominant agent fails (relevance? grounding? completeness?), then edit accordingly. This
  is the easy, high-confidence case.
- **`systemic`** → Don't trust a single-agent edit. Inspect the simulation outputs in the
  report (`simulation_details`) for coalitions: which *combinations* being ablated recover
  quality? That points at the interacting pair.
- **`architectural_gap`** → The recommendation engine often emits an `AgentSpec` (name,
  position, I/O keys, prompt template). Use it as the scaffold for the new node, then
  re-diagnose to confirm the gap closed.
- **`feedback_amplification`** → Check `damping_ratio`. Reducing loop iterations is the
  cheapest test; if quality recovers, the loop was the problem.

## Confidence

`classification.confidence` (0–1) and `confidence_explanation` tell you how much to trust
the type. Low confidence usually means too few simulations or an under-powered quality
metric. Prefer raising `--num-simulations` over acting on a low-confidence classification.
See `reading-attribution.md` for the statistical guardrails.
