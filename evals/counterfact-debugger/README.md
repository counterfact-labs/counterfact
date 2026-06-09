# counterfact-debugger skill eval

A **rerunnable behavioral eval** for the `counterfact-debugger` Agent Skill. It tests the
skill the way it is actually used: a real agent, given a natural-language bug report and the
skill, must diagnose and fix a broken pipeline on its own. This is the tier-4 test that the
unit tests (`counterfact/tests/test_skill_*.py`) can't cover.

## Scenarios

- **`basic`** (default) — a plain broken pipeline (`sample_project/`). Grades that the agent
  diagnoses, fixes, and verifies. The bug (`polisher` overwrites a good answer and strips the
  figure) is *ablation-discoverable*: removing it restores quality.
- **`side_effects`** (`sample_project_side_effects/`) — adds a `notifier` node that pages an
  on-call channel (appends to `$MYRAG_OUTBOX`, default `prod_outbox.log`) on **every** run.
  A naive full diagnosis fires it ~11×. This scenario ALSO grades that the agent **contained**
  the side effect (redirected/mocked it, or previewed with `--dry-run`) — testing friction #2.
- **`no_classifier`** (`sample_project_no_classifier/`) — ships **no** `quality.py`, so the
  agent must build the quality metric itself. The prompt supplies the user's preferred
  definition of "correct" (the answer must contain the exact figure) — standing in for the
  confirmation answer a headless run can't ask for. Tests that the agent implements a
  **user-specified** metric and diagnoses with it, rather than silently guessing one.

```bash
evals/counterfact-debugger/run_eval.sh                          # basic, 1x
N=5 evals/counterfact-debugger/run_eval.sh                      # basic, pass rate
SCENARIO=side_effects N=5 evals/counterfact-debugger/run_eval.sh
```

## What it does

`run_eval.sh`:
1. Copies `sample_project/` (a deliberately broken LangGraph QA pipeline) into a fresh temp
   workspace and installs the skill into `.claude/skills/`.
2. Runs `claude -p` headless with a user-style prompt, letting the agent drive the skill.
3. Grades the **result** with `grade.py` against ground truth.

The sample bug is *ablation-discoverable*: a `polisher` node overwrites a correct answer and
strips the figure. Ground truth: removing/fixing `polisher` restores quality to 1.0. The
project is built with `from langgraph.graph import ...` on purpose, so a passing run must
also perform the counterfact swap.

## Grading (deterministic, outcome-based)

`grade.py` passes only if all criteria hold — checked independently of the agent's narration:
- **skill_used** — a diagnose report (with `shapley_values`) exists in the workspace.
- **swap_done** — `pipeline.py` now imports `counterfact.StateGraph`, not LangGraph.
- **fix_works** — re-running diagnosis on the agent's *edited* code yields baseline quality
  ≥ 0.9 (the grader computes this itself, redirecting any side effect to a scratch file).
- **side_effects_contained** (`side_effects` scenario only, `--max-side-effects N`) — the
  production outbox has ≤ N entries, proving the agent didn't blast the real side effect via
  an unmocked full diagnosis. Counted before the grader's own re-run, which is redirected.

`agent_blamed_agent` is reported for insight but not gated (tiny-fixture attribution can be
noisy; outcome is what matters).

## Run it

```bash
evals/counterfact-debugger/run_eval.sh        # one run
N=5 evals/counterfact-debugger/run_eval.sh    # 5 runs, prints pass rate
```

Requires the `claude` CLI, the repo `.venv` (with `counterfact` installed), and an
`ANTHROPIC_API_KEY` (auto-loaded from `../counterfactual-debugger/.env` if present).

**This is an LLM eval: non-deterministic and token-consuming.** A single pass is evidence,
not proof — run it a few times and watch the pass rate. Transcripts and verdicts land in
`results/` (gitignored). Failed runs keep their workspace path for inspection.

## Design principle: grade outcomes, not artifacts

A skill eval must grade **observable outcomes**, never the agent's intermediate steps or its
tidiness. An earlier version of this grader had a ~60% false-negative rate because it checked:
- a leftover `report.json` (but tidy agents delete their artifacts when done), and
- whether the import swap persisted (but reverting the swap after diagnosing is *correct* —
  it leaves the user's pipeline as they had it).

The fix: `fix_works` **invokes** the edited pipeline and checks the answer is correct (works
whether or not the swap was reverted); `skill_used` accepts a report **or** transcript
evidence; `swap_done` is informational only. Validated to still fail genuine misses (careless
side effects, unfixed bug, fixed-without-the-skill). When an eval and the agent disagree,
suspect the eval first.

## What past runs exposed (and fed back into the skill)

- A **silent domain/registry mismatch**: `--domain rag` with a `finance`-registered classifier
  ran zero classifiers and produced noise. → `cf_diagnose.py` now warns/auto-corrects.
- **dominant ≠ culprit**: the highest-magnitude agent isn't always the one to fix. → added
  guidance in `reference/reading-attribution.md` and fixed `verify.py`'s blame logic.
- **LangGraph state gotcha** (not a counterfact bug — reproduces on raw LangGraph): a node
  returning `{}` drops prior state keys under an untyped `dict` schema, while returning `None`
  preserves them. Matters when mocking a node for safe diagnosis. → documented in
  `reference/practical-limits.md`.
