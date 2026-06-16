#!/usr/bin/env bash
# Behavioral eval on the REAL 8-agent FinanceBench pipeline.
#
# A headless agent is given the broken pipeline + the counterfact-debugger skill and
# a user-style bug report ("answers are missing the exact figures"). It must diagnose
# with real counterfactual ablation, fix the responsible agents' prompts, and improve
# the pipeline. grade_financebench.py then runs the agent's EDITED pipeline on the
# FinanceBench queries and checks exact-answer count recovered.
#
# This is the real-class proof: the skill on a genuine multi-agent LLM pipeline, not a
# deterministic toy. It is expensive and non-deterministic. The agent is told to use a
# reduced simulation count (FB_SIMS) to bound cost.
#
# Usage:  N=3 evals/counterfact-debugger/run_eval_financebench.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SKILL="$REPO/skills/counterfact-debugger"
PKG="$REPO/examples/financebench_casestudy"
N="${N:-1}"
FB_SIMS="${FB_SIMS:-12}"          # reduced sims per query to bound eval cost
MIN_EXACT="${MIN_EXACT:-3}"       # exact answers (of 5) required to pass
RESULTS="$HERE/results/financebench"

# Use the repo venv if present, otherwise rely on python/python3 already on PATH.
if [ -x "$REPO/.venv/bin/python" ]; then
  export PATH="$REPO/.venv/bin:$PATH"
  PY="$REPO/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "ERROR: set ANTHROPIC_API_KEY (preferred) or GOOGLE_API_KEY before running this eval." >&2
  exit 1
fi
if ! command -v claude >/dev/null; then
  echo "ERROR: the 'claude' CLI is required for this behavioral eval (it drives a headless agent)." >&2
  exit 1
fi

PROMPT="My financial-QA pipeline in the financebench_casestudy/ package answers questions about
3M's 2018 10-K, but the answers come back with rounded figures (\$1.6 billion instead of
\$1,577 million) and fabricated peer comparisons. Eight agents, all passing in the trace,
and I can't tell which ones to fix. Use the counterfact debugger skill in
.claude/skills/counterfact-debugger to diagnose which agents are responsible and fix them.
The factory is financebench_casestudy.pipeline:build, the classifier registry is
financebench_casestudy.quality:build_registry, the queries are in cases.json, and the classifier
domain is 'financebench'. The four editable agents' instructions live in
financebench_casestudy/prompts.py — fix the prompts, do not rewrite the pipeline.

IMPORTANT — finish within this single session:
- Run every command SYNCHRONOUSLY in the foreground and BLOCK until it returns. The Bash
  tool waits for the command to finish and returns its output — so just run cf_diagnose and
  wait. Do NOT use background execution, do NOT use a Monitor, and do NOT schedule a wakeup
  or 'wait for it to fire later'. There is no later — this is one shot. If you find yourself
  about to wait for a background task, instead run it in the foreground and block on it.
- The full diagnosis is slow. You do NOT need all 5 queries: identify the culprit by
  diagnosing just 1-2 queries at --num-simulations ${FB_SIMS} (the fix generalizes). A
  single representative query is enough to find the agent that converts millions to billions.
- Then edit the responsible prompt(s) in financebench_casestudy/prompts.py and confirm the fix
  by invoking the pipeline once per query (cheap) — the answers should contain the exact
  figures like \$1,577 million."

mkdir -p "$RESULTS"
pass=0
for i in $(seq 1 "$N"); do
  WORK="$(mktemp -d)"
  cp -R "$PKG" "$WORK/financebench_casestudy"
  cp "$PKG/cases.json" "$WORK/cases.json" 2>/dev/null || true
  mkdir -p "$WORK/.claude/skills"
  cp -R "$SKILL" "$WORK/.claude/skills/counterfact-debugger"

  echo "=== [financebench] run $i/$N | sims=$FB_SIMS | workspace: $WORK ==="
  # Heartbeat: the agent's own progress bar is trapped inside its `claude -p` session,
  # so emit a liveness line here every 30s (elapsed + shared-cache growth) so the run is
  # watchable from the outside.
  CACHE_DIR="${FB_LLM_CACHE:-$HOME/.cache/financebench_casestudy}"
  ( t0=$SECONDS
    while true; do
      sleep 30
      calls=$(ls "$CACHE_DIR"/*.txt 2>/dev/null | wc -l | tr -d ' ')
      reports=$(ls "$WORK"/*.json 2>/dev/null | grep -c -v cases.json)
      echo "  [run $i/$N] working… $(( (SECONDS-t0)/60 ))m$(( (SECONDS-t0)%60 ))s elapsed | ${calls} cached calls | ${reports} diagnose reports"
    done ) &
  HB=$!
  ( cd "$WORK" && PYTHONPATH="$WORK" claude -p "$PROMPT" \
      --allowedTools "Bash Read Edit Write Glob Grep" \
      --dangerously-skip-permissions \
      > "$RESULTS/transcript_$i.txt" 2>&1 )
  kill "$HB" 2>/dev/null

  echo "--- grading run $i (running edited pipeline on the queries) ---"
  PYTHONPATH="$WORK" "$PY" "$HERE/grade_financebench.py" \
      --workspace "$WORK" --transcript "$RESULTS/transcript_$i.txt" \
      --min-exact "$MIN_EXACT" | tee "$RESULTS/verdict_$i.json"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    pass=$((pass+1)); echo "run $i: PASS"
  else
    echo "run $i: FAIL  (workspace kept: $WORK)"
  fi
  echo
done

echo "================================================"
echo "counterfact-debugger eval [financebench]: $pass/$N passed"
echo "================================================"
[ "$pass" -eq "$N" ]
