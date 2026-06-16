"""Diagnose the agents-as-tools chain: which sub-agent tool caused the wrong answer?

The assistant calls two tool-agents (fact_lookup, unit_converter) and a composer
writes the reply. The answer is wrong (542 m instead of 330 m). All three tools
"ran fine", so which is at fault? counterfact ablates each tool-agent, re-runs the
chain, and attributes the failure. It then fixes the implicated tool and confirms.

Run: PYTHONPATH=examples python -m agents_as_tools_casestudy.run_casestudy
"""

from __future__ import annotations

import json
import os

from counterfact.classifiers import ClassifierRegistry

from .system import FIXABLE, FIXED, build_system

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
GOLD = "330"
QUERY = "How tall is the tower in meters?"
SIMS = 16
SEED = 42


def _quality_fn(output_text, state):
    return 1.0 if GOLD in (output_text or "") else 0.0


def run():
    baseline = build_system().invoke({"input": QUERY}).get("final_output", "")
    print(f"Question: {QUERY}")
    print(f"Baseline answer: {baseline!r}  (correct = {GOLD} meters)\n")

    report = build_system().diagnose(
        input_state={"input": QUERY},
        num_simulations=SIMS,
        quality_fn=_quality_fn,
        registry=ClassifierRegistry(),
        run_evals=False,
        seed=SEED,
    )
    shap = report.shapley_values or {}
    print("Attribution (Shapley; negative = the tool's presence hurts the answer):")
    for node, val in sorted(shap.items(), key=lambda kv: kv[1]):
        print(f"  {node:16} {val:+.3f}")

    culprit = min(((n, v) for n, v in shap.items() if n in FIXABLE), key=lambda kv: kv[1], default=(None, 0))[0]
    print(f"\nMost implicated fixable tool: {culprit}")

    fixed_answer = ""
    if culprit in FIXED:
        fixed_answer = build_system(FIXED).invoke({"input": QUERY}).get("final_output", "")
        print(f"After fixing '{culprit}': {fixed_answer!r}")

    report_out = {
        "system": "openai-agents-sdk agents-as-tools (offline, deterministic)",
        "pattern": "assistant -> fact_lookup(tool) -> unit_converter(tool) -> composer",
        "query": QUERY,
        "gold": f"{GOLD} meters",
        "baseline_answer": baseline,
        "baseline_quality": round(report.baseline_quality, 3),
        "shapley_values": {k: round(v, 4) for k, v in shap.items()},
        "culprit": culprit,
        "fixed_answer": fixed_answer,
    }
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "agents_as_tools_casestudy.json")
    with open(path, "w") as f:
        json.dump(report_out, f, indent=2)
    print(f"\nWrote {path}")
    return report_out


def main():
    run()


if __name__ == "__main__":
    main()
