# Practical limits & frictions — read this before diagnosing a real pipeline

Counterfactual ablation is powerful but it is not free and not universal. These are the
four things that bite in practice and how to handle each. Surface the relevant ones to the
user rather than barging ahead.

## 1. The factory requirement

`diagnose` needs a no-arg `module:function` that returns a *compiled counterfact graph*.
Real pipelines are often built inline in a script, or need config/secrets/clients.

- **Mitigation:** extract the build into a small wrapper. If it needs config, bake a safe
  default into the wrapper: `def build(): return build_app(load_config("staging"))`.
- If the graph is constructed with live clients (DB, vector store, LLM), the factory is also
  where you swap in **mocked or staging** clients (see friction #2).
- If you genuinely cannot produce a runnable factory (e.g. you only have logs), you cannot
  run `diagnose`. Fall back to `counterfact eval <trace.json>` for structural checks and say
  the attribution step isn't available.

## 2. Re-execution has side effects and cost — the dangerous one

Diagnosis re-runs the **entire pipeline ~`num_simulations` times per case**. If any node
writes to a database, sends email/Slack, charges a card, or hits a rate-limited/paid API,
those effects happen on every re-run. There is no built-in sandbox.

- **Always preview first:** `python cf_diagnose.py --factory ... --inputs ... --dry-run`
  runs the pipeline exactly once and prints how many executions a full run would incur.
- **Point the factory at a sandbox:** mock or stub side-effecting nodes (network, writes,
  payments) so re-runs are safe and cheap. The factory is the seam for this.
  - **LangGraph state gotcha when mocking:** a node that returns `{}` drops prior state keys
    under an untyped `dict` schema (this is LangGraph's own behavior, not counterfact's). A
    mocked/stubbed node must return `None` (or the keys it's responsible for), never `{}`, or
    it will silently wipe the answer and skew the diagnosis.
- **Scale `--num-simulations` to cost:** start at ~24–30; only raise it if attribution is
  inconclusive (see `reading-attribution.md`).
- **Never run a diagnosis against production resources** without confirming with the user.

## 3. Attribution is only as good as the quality metric

Shapley values are computed from a quality score. A weak or absent metric yields confident-
looking noise.

- **Domain/registry must match.** If you pass `--registry` whose classifiers are registered
  under `domain="finance"`, you must pass `--domain finance`. A mismatch means zero
  classifiers run and every output scores a flat default — meaningless attribution. The
  runner now warns/auto-corrects, but check `classifiers_used` in the report is non-empty.
- **Built-in `--domain rag` classifiers are weak on non-RAG pipelines** and need an LLM key.
  Prefer scaffolding a tiny targeted classifier (`report-schema.md`) that measures the actual
  failure the user reported.
- **Smell test:** if all Shapley CIs straddle zero or the failure type is a low-confidence
  `systemic` tie, suspect the metric before believing "no single cause."

## 4. "Drop-in" LangGraph replacement isn't total

counterfact's `StateGraph` mirrors LangGraph's, but advanced features may not round-trip:
conditional edges (need an explicit routing callable in the spec), checkpointing/persistence,
async nodes, streaming, and subgraphs.

- **Parity-check after the swap:** invoke the pipeline once post-swap (or use `--dry-run`)
  and confirm it runs without error and produces the same output it did before the swap.
  If it diverges or errors, the graph uses something the swap doesn't preserve.
- **Conditional routing:** if the pipeline branches, ensure the conditional edges survived
  the swap; a branch silently flattened to a linear edge changes what you're diagnosing.
- If parity fails, report exactly which feature broke instead of trusting the diagnosis.
