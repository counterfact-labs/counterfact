"""
verify.py — confirm a fix actually moved the attribution.

Counterfactual diagnosis attributes a failure to an agent. After you edit that
agent and re-run cf_diagnose.py to produce a second report, this script compares
the two and tells you whether the fix did what it was supposed to:

  - baseline quality went UP
  - the previously-blamed agent's Shapley contribution moved toward zero
    (i.e. it is no longer dragging quality down)
  - no other agent's contribution got dramatically worse (regression check)

Usage:
    python verify.py --baseline report.json --candidate report_after.json

Exit code 0 if the fix is an improvement on the headline metrics, 1 otherwise.
This is a heuristic gate, not proof — read the deltas it prints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        print(f"error: report not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def _blamed_agent(report: dict) -> str | None:
    """The agent to hold responsible — i.e. the one we'd try to fix.

    Prefer the most-NEGATIVE Shapley contributor: a negative value means the agent
    actively hurts quality (the pipeline does better without it), which is the
    clearest culprit signal. Only fall back to `dominant_agent` (largest magnitude,
    often an essential/positive agent) or the least-positive agent when nothing is
    negative.
    """
    shap = report.get("shapley_values") or {}
    if shap:
        agent, value = min(shap.items(), key=lambda kv: kv[1])
        if value < 0:
            return agent
    dom = (report.get("classification") or {}).get("dominant_agent")
    if dom:
        return dom
    return min(shap.items(), key=lambda kv: kv[1])[0] if shap else None


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two counterfact diagnose reports.")
    p.add_argument("--baseline", required=True, help="diagnose report JSON from before the fix")
    p.add_argument("--candidate", required=True, help="diagnose report JSON from after the fix")
    args = p.parse_args()

    base, cand = _load(args.baseline), _load(args.candidate)

    bq_before = base.get("baseline_quality", 0.0)
    bq_after = cand.get("baseline_quality", 0.0)
    dq = bq_after - bq_before

    agent = _blamed_agent(base)
    shap_before = (base.get("shapley_values") or {}).get(agent, 0.0) if agent else 0.0
    shap_after = (cand.get("shapley_values") or {}).get(agent, 0.0) if agent else 0.0
    # "Improved" = contribution moved toward zero from a negative drag.
    drag_reduced = abs(shap_after) < abs(shap_before) if shap_before < 0 else True

    print("=" * 64)
    print("COUNTERFACT FIX VERIFICATION")
    print("=" * 64)
    print(f"Blamed agent (baseline):  {agent}")
    print(f"Baseline quality:         {bq_before:.3f} -> {bq_after:.3f}  ({dq:+.3f})")
    print(f"{agent} Shapley:          {shap_before:+.3f} -> {shap_after:+.3f}")
    print(f"Failure type:             "
          f"{(base.get('classification') or {}).get('failure_type')} -> "
          f"{(cand.get('classification') or {}).get('failure_type')}")

    # Regression check: did any agent get a much worse drag than before?
    sb = base.get("shapley_values") or {}
    sc = cand.get("shapley_values") or {}
    regressions = []
    for a in set(sb) | set(sc):
        before, after = sb.get(a, 0.0), sc.get(a, 0.0)
        if after < before - 0.1 and after < 0:  # got materially more negative
            regressions.append((a, before, after))
    if regressions:
        print("\nPossible new regressions (agent: before -> after):")
        for a, before, after in regressions:
            print(f"  ! {a}: {before:+.3f} -> {after:+.3f}")

    quality_up = dq > 0.01
    improved = quality_up and drag_reduced and not regressions

    print("\nVerdict:", "IMPROVED ✓" if improved else "NOT CLEARLY IMPROVED ✗")
    if not improved:
        if not quality_up:
            print("  - baseline quality did not rise meaningfully")
        if not drag_reduced:
            print(f"  - {agent}'s negative contribution did not shrink")
        if regressions:
            print("  - another agent regressed; the fix may have shifted the problem")
    print("=" * 64)
    sys.exit(0 if improved else 1)


if __name__ == "__main__":
    main()
