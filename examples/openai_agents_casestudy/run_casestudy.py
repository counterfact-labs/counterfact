"""Drive the counterfact skill over the OpenAI Agents SDK support system.

Pipeline of the case study itself:

  1. Load the eval set (Braintrust dataset shape) and a Braintrust-style scorer.
  2. Measure the baseline pass rate (it fails: the figure is stripped).
  3. Diagnose each failing case with real counterfactual ablation, using the
     scorer as the quality metric, and average the Shapley attribution.
  4. Apply the corrected instruction to the most-implicated (most-negative
     Shapley) editable agent, then re-evaluate.
  5. Repeat until the eval set passes, then write a JSON + Markdown report.

Run:

    PYTHONPATH=examples python -m openai_agents_casestudy.run_casestudy

Deterministic (no network, fixed seed): the same numbers every run.
"""

from __future__ import annotations

import json
import os

from counterfact.classifiers import ClassifierRegistry
from counterfact.integrations.braintrust import (
    cases_from_dataset,
    quality_fn_from_scorer,
)

from .scorer import refund_amount_scorer
from .system import FIXABLE, FIXED, INSTRUCTIONS, build_system

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
SIMS = 16
SEED = 42
MAX_FIXES = 4


def _load_cases():
    with open(os.path.join(HERE, "cases.json")) as f:
        records = json.load(f)
    # Braintrust dataset records -> counterfact cases (gold embedded in input state).
    return cases_from_dataset(records)


def _evaluate(instructions, cases, quality_fn):
    """Return (pass_rate, per_case_results) for the given instruction set."""
    graph = build_system(instructions)
    results = []
    passed = 0
    for case in cases:
        out = graph.invoke({**case["input"]})
        final = out.get("final_output", "")
        score = quality_fn(final, case["input"])
        ok = score >= 1.0
        passed += ok
        results.append({"id": case["id"], "score": score, "passed": bool(ok), "final_output": final})
    return passed / len(cases), results


def _diagnose_failing(instructions, cases, quality_fn, case_results):
    """Average Shapley attribution across the currently-failing cases."""
    failing = [c for c, r in zip(cases, case_results) if not r["passed"]]
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    for case in failing:
        graph = build_system(instructions)
        report = graph.diagnose(
            input_state={**case["input"]},
            num_simulations=SIMS,
            quality_fn=quality_fn,
            registry=ClassifierRegistry(),  # scorer drives attribution; no LLM classifiers
            run_evals=False,
            seed=SEED,
        )
        for node, val in (report.shapley_values or {}).items():
            agg[node] = agg.get(node, 0.0) + val
            counts[node] = counts.get(node, 0) + 1
    return {node: agg[node] / counts[node] for node in agg}


def _pick_agent_to_fix(shapley):
    """Most-negative-Shapley editable agent: the one whose presence most hurts."""
    candidates = [(node, s) for node, s in shapley.items() if node in FIXABLE]
    if not candidates:
        return None, None
    node, score = min(candidates, key=lambda kv: kv[1])
    return node, score


def run():
    cases = _load_cases()
    quality_fn = quality_fn_from_scorer(refund_amount_scorer)

    timeline = []
    instructions = dict(INSTRUCTIONS)

    base_rate, base_results = _evaluate(instructions, cases, quality_fn)
    print(f"Baseline: {base_rate * 100:.0f}% of tickets include the refund amount "
          f"({sum(r['passed'] for r in base_results)}/{len(cases)})")

    rate, results = base_rate, base_results
    fixes = []
    for _ in range(MAX_FIXES):
        if rate >= 1.0:
            break
        shapley = _diagnose_failing(instructions, cases, quality_fn, results)
        agent, score = _pick_agent_to_fix(shapley)
        timeline.append({"shapley": {k: round(v, 3) for k, v in shapley.items()},
                         "picked": agent, "picked_shapley": round(score, 3) if score is not None else None})
        print(f"  Diagnosis: Shapley = {timeline[-1]['shapley']}  ->  fix '{agent}' (Shapley {score:+.2f})")
        if agent is None or agent not in FIXED:
            print(f"  No corrected instruction available for '{agent}'. Stopping.")
            break
        instructions[agent] = FIXED[agent]
        fixes.append(agent)
        rate, results = _evaluate(instructions, cases, quality_fn)
        print(f"  After fixing '{agent}': {rate * 100:.0f}% "
              f"({sum(r['passed'] for r in results)}/{len(cases)})")

    report = {
        "system": "openai-agents-sdk orchestrator+handoffs (offline, deterministic)",
        "scorer": "braintrust-style refund_amount_present",
        "n_cases": len(cases),
        "baseline_pass_rate": round(base_rate, 3),
        "final_pass_rate": round(rate, 3),
        "fixes_applied": fixes,
        "diagnosis_timeline": timeline,
        "final_case_results": [{"id": r["id"], "passed": r["passed"]} for r in results],
    }
    os.makedirs(REPORTS, exist_ok=True)
    json_path = os.path.join(REPORTS, "openai_agents_casestudy.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(REPORTS, "openai_agents_casestudy.md")
    with open(md_path, "w") as f:
        f.write(_render_markdown(report))

    print(f"\nFinal: {rate * 100:.0f}% (fixed: {', '.join(fixes) or 'none'})")
    print(f"Wrote {json_path}\n      {md_path}")
    return report


def _render_markdown(report) -> str:
    lines = [
        "# OpenAI Agents SDK + Braintrust — counterfact case study (generated)",
        "",
        f"- System: {report['system']}",
        f"- Scorer: `{report['scorer']}`",
        f"- Cases: {report['n_cases']}",
        f"- Baseline pass rate: **{report['baseline_pass_rate'] * 100:.0f}%**",
        f"- Final pass rate: **{report['final_pass_rate'] * 100:.0f}%**",
        f"- Fixes applied: {', '.join(f'`{a}`' for a in report['fixes_applied']) or 'none'}",
        "",
        "## Diagnosis timeline",
        "",
    ]
    for i, step in enumerate(report["diagnosis_timeline"], 1):
        lines.append(f"**Round {i}** — picked `{step['picked']}` "
                     f"(Shapley {step['picked_shapley']:+.2f})")
        lines.append("")
        lines.append("| agent | Shapley |")
        lines.append("|---|---|")
        for node, val in sorted(step["shapley"].items(), key=lambda kv: kv[1]):
            lines.append(f"| `{node}` | {val:+.3f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    run()


if __name__ == "__main__":
    main()
