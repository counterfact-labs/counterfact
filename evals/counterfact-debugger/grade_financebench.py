"""Outcome grader for the FinanceBench behavioral eval.

Grades whether the agent actually IMPROVED the real 8-agent pipeline, by running
the agent's EDITED pipeline on the FinanceBench queries and counting exact answers
(answers containing the precise dollar figure). The broken baseline scores ~0/5;
a real fix recovers several. Independent of anything the agent narrates.

Criteria (all must hold):
  - skill_used  — transcript shows the agent ran the counterfactual diagnosis.
  - fix_works   — the edited pipeline now produces >= --min-exact exact answers.

Usage:
  python grade_financebench.py --workspace DIR --transcript FILE [--min-exact 3] [--queries 5]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SKILL_EVIDENCE = re.compile(r"shapley|counterfactual|ablat|cf_diagnose|attribution", re.I)


def _ran_diagnosis(workspace: Path) -> bool:
    """Robust skill-usage signal: a diagnose report (with shapley_values) was produced.
    More reliable than transcript keywords — a `claude -p` final message can be just a
    'waiting…' note if the agent was mid-task when the turn ended."""
    for p in workspace.rglob("*.json"):
        if ".claude" in p.parts:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "shapley_values" in d:
            return True
    return False


def _exact_answers(workspace: Path, n_queries: int) -> tuple[int | None, list, str]:
    """Invoke the agent's edited pipeline on each query; count exact figures."""
    code = f"""
import sys, json
sys.path.insert(0, {str(workspace)!r})
from financebench_skill import data
from financebench_skill.pipeline import build
import re
n = {n_queries}
results = []
for q in data.QUERIES[:n]:
    out = build().invoke(data.make_input_state(q["query"])).get("analysis", "")
    gt = data.GROUND_TRUTH[q["query"]]
    num = re.search(r"[\\d,]+", gt).group()
    exact = (num in out) or (num.replace(",", "") in out)
    results.append({{"short": q["short"], "exact": bool(exact), "out": out[:120]}})
print("<<<" + json.dumps(results) + ">>>")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, [], f"pipeline invoke failed: {proc.stderr.strip()[-300:]}"
    m = re.search(r"<<<(.*)>>>", proc.stdout, re.DOTALL)
    if not m:
        return None, [], f"no result marker: {proc.stdout.strip()[-200:]}"
    results = json.loads(m.group(1))
    return sum(r["exact"] for r in results), results, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--min-exact", type=int, default=3, help="exact answers required to pass")
    ap.add_argument("--queries", type=int, default=5)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    transcript = ""
    if args.transcript and Path(args.transcript).exists():
        transcript = Path(args.transcript).read_text()
    # Skill usage = produced a diagnose report OR the transcript describes the diagnosis.
    skill_used = _ran_diagnosis(workspace) or bool(_SKILL_EVIDENCE.search(transcript))

    exact, detail_rows, detail = _exact_answers(workspace, args.queries)
    fix_works = exact is not None and exact >= args.min_exact

    criteria = {"skill_used": skill_used, "fix_works": fix_works}
    verdict = {
        "pass": all(criteria.values()),
        "criteria": criteria,
        "exact_answers": exact,
        "min_exact": args.min_exact,
        "n_queries": args.queries,
        "per_query": detail_rows,
        "detail": detail,
    }
    print(json.dumps(verdict, indent=2))
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
