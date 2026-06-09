"""Deterministic grader for the counterfact-debugger skill eval.

Grades the agent's RESULT, not its narration, and measures OUTCOMES that survive
legitimate agent choices (deleting report artifacts, reverting the import swap, etc.):

  - fix_works  (hard) — INVOKE the agent's edited pipeline on the failing case and check
                        the answer now contains the figure. Invoking works whether the
                        pipeline ends on counterfact OR raw LangGraph, so it doesn't punish
                        an agent that reverted the swap after diagnosing. Any side effect is
                        redirected to a scratch path so grading doesn't page on-call.
  - skill_used (hard) — evidence the agent actually used the debugger: a diagnose report
                        (with shapley_values) was produced, OR the final transcript describes
                        the counterfactual diagnosis (shapley / ablation / cf_diagnose).
  - side_effects_contained (hard, --max-side-effects only) — the production outbox has <= N
                        entries, proving the agent didn't blast a real side effect via an
                        unmocked full diagnosis. Counted BEFORE the grader's own invoke.

  - swap_done  (informational, NOT gated) — whether pipeline.py currently uses counterfact.
                        Reverting the swap after diagnosing is fine, so this never fails a run.

Exit code 0 only if all hard criteria pass. Prints a JSON verdict to stdout.

Usage:
    python grade.py --workspace DIR --skill-scripts DIR [--transcript FILE]
                    [--max-side-effects N] [--outbox NAME]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_DEFAULT_CASE = {"metric_key": "3m_2018_revenue", "query": "What was 3M's 2018 revenue?"}
_SKILL_EVIDENCE = re.compile(r"shapley|counterfactual|ablat|cf_diagnose|attribution", re.I)


def _find_report(workspace: Path) -> dict | None:
    """Find a diagnose report JSON the agent produced (has shapley_values), if any."""
    for p in sorted(workspace.rglob("*.json")):
        if ".claude" in p.parts:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "shapley_values" in data:
            return {"path": str(p), "data": data}
    return None


def _first_case(workspace: Path) -> dict:
    cases = workspace / "cases.json"
    if cases.exists():
        try:
            data = json.loads(cases.read_text())
            return data[0] if isinstance(data, list) else data
        except Exception:
            pass
    return _DEFAULT_CASE


def _invoke_output(workspace: Path) -> tuple[str | None, str]:
    """Build the agent's pipeline and invoke it once on the failing case.

    Runs in a subprocess so the workspace's myrag package imports cleanly and any
    side effect is redirected to a scratch file. Returns (output_text, detail).
    """
    import subprocess

    case = json.dumps(_first_case(workspace))
    code = (
        "import sys, json, os\n"
        f"sys.path.insert(0, {str(workspace)!r})\n"
        "import myrag.pipeline as m\n"
        f"out = m.build().invoke(json.loads({case!r}))\n"
        "print('<<<' + str(out.get('output', '') if isinstance(out, dict) else out) + '>>>')\n"
    )
    env = {**__import__("os").environ, "MYRAG_OUTBOX": str(workspace / "_grader_scratch.log")}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return None, f"invoke failed: {proc.stderr.strip()[-200:]}"
    m = re.search(r"<<<(.*)>>>", proc.stdout, re.DOTALL)
    return (m.group(1) if m else proc.stdout.strip()), "ok"


def _swap_done(workspace: Path) -> bool:
    pipeline = workspace / "myrag" / "pipeline.py"
    if not pipeline.exists():
        return False
    return re.search(r"from\s+counterfact\s+import|import\s+counterfact", pipeline.read_text()) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--skill-scripts", required=True)  # kept for interface stability
    ap.add_argument("--transcript", default=None, help="agent final-message file (skill_used signal)")
    ap.add_argument("--max-side-effects", type=int, default=None,
                    help="If set, require the production outbox to have <= N entries.")
    ap.add_argument("--outbox", default="prod_outbox.log")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()

    # --- side effects: count BEFORE we invoke anything (our invoke is redirected) ---
    side_effects = None
    side_effects_contained = True
    if args.max_side_effects is not None:
        outbox = workspace / args.outbox
        side_effects = len(outbox.read_text().splitlines()) if outbox.exists() else 0
        side_effects_contained = side_effects <= args.max_side_effects

    # --- fix_works: invoke the edited pipeline, check the figure is present ---
    output, detail = _invoke_output(workspace)
    has_number = bool(output and re.search(r"\d", output))
    fix_works = has_number

    # --- skill_used: report artifact OR transcript describes the diagnosis ---
    report = _find_report(workspace)
    transcript = ""
    if args.transcript and Path(args.transcript).exists():
        transcript = Path(args.transcript).read_text()
    skill_used = report is not None or bool(_SKILL_EVIDENCE.search(transcript))

    blamed = None
    if report:
        cls = report["data"].get("classification") or {}
        blamed = cls.get("dominant_agent")

    criteria = {"skill_used": skill_used, "fix_works": fix_works}
    if args.max_side_effects is not None:
        criteria["side_effects_contained"] = side_effects_contained

    verdict = {
        "pass": all(criteria.values()),
        "criteria": criteria,
        "swap_done_informational": _swap_done(workspace),
        "final_output": output,
        "agent_blamed_agent": blamed,
        "side_effects_fired": side_effects,
        "max_side_effects": args.max_side_effects,
        "detail": detail,
        "report_found": report["path"] if report else None,
    }
    print(json.dumps(verdict, indent=2))
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
