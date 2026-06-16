---
name: counterfact-debugger
description: >-
  Diagnose WHICH agent is responsible for a failing or flaky multi-agent / LangGraph
  pipeline by running real counterfactual ablation (Shapley attribution), then propose
  and verify a fix. Use when a user has a multi-step LLM pipeline (RAG, agent chain,
  LangGraph StateGraph) that produces wrong, hallucinated, empty, or inconsistent output
  and wants to know which node to blame and how to fix it — rather than guessing from logs.
  Triggers: "which agent is breaking my pipeline", "debug my LangGraph", "my RAG keeps
  hallucinating and I don't know why", "attribute this failure", "counterfactual analysis",
  "is my retriever the problem", "does retrieval/reranking quality matter here".
---

# Counterfact Debugger

Find the agent responsible for a multi-agent pipeline failure using **real counterfactual
analysis**: the pipeline is re-executed with each agent ablated (replaced by a no-op), and
quality changes are measured to compute Shapley attribution. This is ground truth, not an
LLM guessing from a trace.

## When this applies

The pipeline is built (or can be built) with `counterfact.StateGraph` — a drop-in
replacement for `langgraph.graph.StateGraph`. Diagnosis **re-runs the live pipeline**, so a
static trace file is not enough; you need runnable code. If the user only has logs/traces,
fall back to `counterfact eval <trace.json>` (weaker, structural-only) and say so.

## Workflow

Work through these steps. Do not skip the verification step — an unverified fix is a guess.
Before diagnosing a real pipeline, skim `reference/practical-limits.md` — the four frictions
(factory extraction, destructive re-runs, weak metrics, imperfect LangGraph parity) decide
whether this is safe and worth it.

### 0. Safety preflight (do not skip on real systems)
Diagnosis **re-executes the whole pipeline ~`num_simulations` times per case**. Before that:
- Check for **side-effecting nodes** — DB writes, payments, emails, rate-limited/paid APIs.
  If present, point the factory at a sandbox/mock; never diagnose against production.
- Preview cost/behavior with one execution:
  `python scripts/cf_diagnose.py --factory ... --inputs ... --dry-run`.
- After the langgraph→counterfact swap, **parity-check**: confirm one invocation still
  produces the same output it did before the swap (catches unsupported features).

### 1. Confirm prerequisites
- `counterfact` is importable in the project's Python env (`python -c "import counterfact"`).
  If not (it is not yet on PyPI, so install from git):
  `pip install "counterfact[anthropic] @ git+https://github.com/counterfact-labs/counterfact.git"` (or `[all]`).
- An LLM key is set for quality classifiers: `ANTHROPIC_API_KEY` (preferred) or
  `GOOGLE_API_KEY`. Without it, only structural classifiers run — note the weaker signal.

### 2. Get a pipeline factory + sample inputs
Diagnosis needs a **compiled graph carrying a build recipe**. Establish two things:

- **Factory**: a `module:function` that returns a *compiled* `CounterfactualGraph` with no
  required args, e.g. `myapp.pipeline:build`. If the project imports
  `from langgraph.graph import StateGraph`, do the one-line swap to
  `from counterfact import StateGraph` first — everything else is identical. If the graph is
  built inline (not in a reusable function), extract that build into a small factory function.
- **Inputs**: a JSON file with the failing `input_state` dict, or a list of them for a
  dataset run. Use the actual case(s) the user says are broken.

A graph built with raw LangGraph has **no recipe** and `diagnose` will raise — the swap is
mandatory, not optional.

### 3. Pick the quality metric — and confirm it with the user
Attribution is **only as good as the quality metric**: the classifier defines what counts as a
"correct" answer, and a wrong definition silently produces a confident-but-wrong diagnosis.
This is the user's call, not yours — do not silently guess it. In order of preference:

1. **Project's own** `ClassifierRegistry` — pass via `--registry module:function`. **The
   `--domain` must match the domain the classifiers are registered under** (e.g. a registry
   that does `register(fn, domain="finance")` needs `--domain finance`). A mismatch means
   zero classifiers run and attribution is silent noise; the runner now warns/auto-corrects,
   but set it right. If `classifiers_used` is empty in the report, the metric didn't run.
2. **Custom scaffold** — if there's no project classifier that clearly matches the reported
   symptom, write 1–2 small classifiers `(query, output, sources) -> ClassifierResult`
   targeting the actual failure. See `reference/report-schema.md`.
3. **Built-in domain** — `--domain rag` or `--domain decision`. Weakest on non-RAG pipelines.

**Confirm before running whenever you had to scaffold or fall back (cases 2–3):** state the
metric you propose — i.e. what you will treat as a "good" answer (e.g. "correct only if the
answer contains the exact dollar figure asked for") — and ask the user to confirm. The user
may approve it, refine it, or **specify their own preferred definition**; if they describe one,
**implement that as the classifier** and diagnose with it. When you reuse a project classifier
that clearly matches the symptom (case 1), you needn't block — but still **name the metric**
you're using so the user can correct it. The diagnosis is only trustworthy once the metric is
the one the user actually cares about.

### 4. Run the diagnosis
```bash
python scripts/cf_diagnose.py \
  --factory myapp.pipeline:build \
  --inputs cases.json \
  --domain rag \
  --num-simulations 30 \
  --seed 42 \
  --out report.json
```
This writes `report.json` (machine-readable) and `report.md` (human-readable). Read both.
Bump `--num-simulations` (50–100) if attribution comes back inconclusive.

> **Ablate vs degrade is automatic — you don't choose.** When attribution removes a node,
> most agents are ablated (replaced by a no-op), but a **structural module** (retriever,
> reranker, parser, context builder) would just collapse the pipeline if fully ablated — the
> huge Shapley would say "this is necessary," not whether its *output quality* is what hurts
> answers. So `diagnose` instead **severely degrades** those modules: the node still runs and
> its output keeps its shape (a retriever still returns a non-empty doc list), but the content
> is destroyed. Their Shapley then reflects quality, not mere necessity. The choice is made by
> inferred module type; `report.simulation_results_summary["removal_strategies"]` shows which
> nodes were `ablate`d vs `degrade`d (the runner also prints this). See
> `reference/ablation-vs-degradation.md`.

### 5. Interpret — do not over-read the numbers
Read `reference/failure-taxonomy.md` and `reference/reading-attribution.md`, then:
- Branch on `classification.failure_type`: `local` / `systemic` / `architectural_gap` /
  `feedback_amplification` / `no_failure`. Each implies a different fix shape.
- **Respect the statistics.** Do not name a culprit whose Shapley CI (`shapley_cis`) straddles
  zero or overlaps the runner-up's CI. If `attribution_method` fell back to Shapley because LOO
  was inconclusive, or the top two overlap, raise `--num-simulations` and re-run rather than guess.
- Use `per_classifier_shapley` to see *which quality dimension* the culprit hurts.
- Treat `recommendations` as leads, not orders — translate them into a concrete code edit.

### 6. Propose the fix, then VERIFY
Apply the smallest change that addresses the attributed agent. Then re-diagnose and confirm
the attributed agent's negative contribution shrank and baseline quality rose:
```bash
python scripts/verify.py --baseline report.json --candidate report_after.json
```
(Generate `report_after.json` by re-running step 4 after the edit.) Report the before/after
deltas. If the culprit's score didn't move, the fix was wrong — say so and iterate.

## Reference (read on demand)
- `reference/practical-limits.md` — the four frictions and how to handle them. Read first.
- `reference/failure-taxonomy.md` — what each `failure_type` means and the fix it implies.
- `reference/reading-attribution.md` — Shapley values, bootstrap CIs, dominant≠culprit, the
  inconclusive case.
- `reference/ablation-vs-degradation.md` — why structural modules (retrievers, parsers) are
  severely degraded instead of ablated, and how the strategy is chosen.
- `reference/report-schema.md` — `DiagnosticReport` fields, classifier signature, examples.

## Scripts
- `scripts/cf_diagnose.py` — load factory + inputs, wire `llm_fn`, run `diagnose` (which
  auto-ablates or severely-degrades each node by type), emit reports.
- `scripts/llm_fn.py` — default Anthropic/Google LLM caller from env (importable + standalone).
- `scripts/verify.py` — compare two diagnose reports to confirm a fix moved the attribution.
