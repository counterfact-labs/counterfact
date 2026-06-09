# counterfact-debugger skill eval

A **rerunnable behavioral eval** for the `counterfact-debugger` Agent Skill. It tests the
skill the way it is actually used: a real agent, given a natural-language bug report and the
skill, must diagnose and fix a broken pipeline on its own — the tier-4 test that the unit
tests (`counterfact/tests/test_skill_*.py`) can't cover.

## The eval (`run_eval_financebench.sh`)

A headless agent is given the **broken 8-agent FinanceBench pipeline**
(`examples/financebench_skill/`) + the skill, and a user-style report ("the answers are
missing the exact figures"). It must diagnose with real counterfactual ablation and fix the
responsible agents' prompts. `grade_financebench.py` then runs the agent's **edited** pipeline
on the FinanceBench queries and checks the exact-answer count recovered (broken baseline = 0/5).

```bash
N=3 FB_SIMS=12 evals/counterfact-debugger/run_eval_financebench.sh
```

Requires the `claude` CLI, the repo `.venv` (with `counterfact` installed), and an
`ANTHROPIC_API_KEY`. **This is an LLM eval: non-deterministic and token-consuming** — a single
pass is evidence, not proof; run it a few times and watch the pass rate. The runs share the
persistent on-disk LLM cache (`examples/financebench_skill/llm.py`), so repeats and restarts
are cheap. Transcripts/verdicts land in `results/` (gitignored); failed runs keep their
workspace for inspection. A heartbeat line reports elapsed time + cache growth while the
agent's (otherwise hidden) `claude -p` session runs.

Latest result: **2/3 reached 5/5 exact answers**; the one miss was a single-turn headless
cutoff (the agent backgrounded the slow diagnosis), not a misdiagnosis — it doesn't apply to
interactive use.

## Grading (deterministic, outcome-based)

`grade_financebench.py` passes only if both hold — checked independently of the agent's narration:
- **skill_used** — the agent produced a diagnose report (with `shapley_values`), or the
  transcript describes the counterfactual diagnosis.
- **fix_works** — the agent's edited pipeline now produces ≥ `--min-exact` exact answers
  (the grader runs it itself; broken baseline is 0/5).

## Design principle: grade outcomes, not artifacts

A skill eval must grade **observable outcomes**, never the agent's intermediate steps or its
tidiness. An earlier grader had a high false-negative rate because it relied on leftover report
files (tidy agents delete them) and on the import swap persisting (reverting it after diagnosing
is *correct*). The fix: run the agent's edited pipeline and check the answers; treat skill usage
as "a diagnose report OR transcript evidence." When an eval and the agent disagree, suspect the
eval first.

## What past runs exposed (and fed back into the skill)

- **Silent domain/registry mismatch**: `--domain` not matching the registry's domain ran zero
  classifiers and produced noise → `cf_diagnose.py` now warns/auto-corrects.
- **dominant ≠ culprit**: the highest-magnitude agent isn't always the one to fix → guidance
  in `reference/reading-attribution.md` + fixed `verify.py`'s blame logic.
- **LangGraph state gotcha** (upstream behavior, not a counterfact bug): a node returning `{}`
  drops prior state keys under an untyped `dict` schema, while `None` preserves them — matters
  when mocking a node for safe diagnosis → documented in `reference/practical-limits.md`.
