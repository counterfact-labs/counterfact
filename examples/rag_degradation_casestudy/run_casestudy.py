"""Show why diagnose severely degrades a retriever instead of ablating it.

The pipeline answers from the top-K retrieved passages, so removing the retriever
entirely (a no-op) leaves the synthesizer with no context and the run structurally
fails. We diagnose the same cases two ways and compare:

  1. Forced pure ablation — every node removed by a no-op (the old behavior).
  2. Auto (default) — diagnose decides per node; structural modules (retriever,
     reranker) are severely degraded instead of ablated.

The point: pure ablation turns the retriever's coalitions into pipeline errors
(structural failures), so its attribution only says "necessary." Auto-degradation
keeps every run live, so the retriever gets a clean, comparable contribution.

Run: PYTHONPATH=examples python -m rag_degradation_casestudy.run_casestudy
"""

from __future__ import annotations

import json
import os

from counterfact.classifiers import ClassifierRegistry

from .pipeline import TOP_K, build

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
SIMS = 24
SEED = 42


def _load_cases():
    with open(os.path.join(HERE, "cases.json")) as f:
        return json.load(f)


def _quality_fn(output_text, state):
    # Gold is carried under a leading-underscore key so it is never mistaken for
    # the pipeline's output (counterfact ignores _-prefixed state keys).
    return 1.0 if state.get("_gold", "\0") in (output_text or "") else 0.0


def _diagnose(case, force_ablation):
    graph = build()
    if force_ablation:
        # Pin every node to plain ablation (the pre-degradation behavior).
        graph._removals = {n: "ablate" for n in graph.get_node_names()}
    return graph.diagnose(
        input_state={"query": case["query"], "_gold": case["gold"]},
        num_simulations=SIMS,
        quality_fn=_quality_fn,
        registry=ClassifierRegistry(),
        run_evals=False,
        quality_gate=1.01,  # the pipeline passes; force attribution anyway
        seed=SEED,
    )


def _aggregate(cases, force_ablation):
    """Average Shapley per node + count coalition runs that structurally failed."""
    agg, counts, errors, total = {}, {}, 0, 0
    strategies = {}
    for case in cases:
        report = _diagnose(case, force_ablation)
        strategies = report.simulation_results_summary.get("removal_strategies", {})
        for node, val in (report.shapley_values or {}).items():
            agg[node] = agg.get(node, 0.0) + val
            counts[node] = counts.get(node, 0) + 1
        for sim in report.simulation_results:
            total += 1
            if "PIPELINE ERROR" in (sim.perturbed_output or ""):
                errors += 1
    shapley = {n: round(agg[n] / counts[n], 3) for n in agg}
    return shapley, errors, total, strategies


def run():
    cases = _load_cases()
    print(f"RAG pipeline (retriever -> reranker -> synthesizer), synthesizer reads top-{TOP_K}.")
    print(f"{len(cases)} cases.\n")

    abl_shapley, abl_errors, abl_total, _ = _aggregate(cases, force_ablation=True)
    auto_shapley, auto_errors, auto_total, strategies = _aggregate(cases, force_ablation=False)

    print(f"{'node':14} {'pure-ablation Shapley':>22} {'auto Shapley':>14} {'auto strategy':>16}")
    for n in ["retriever", "reranker", "synthesizer"]:
        print(f"{n:14} {abl_shapley.get(n, 0.0):>+22.3f} {auto_shapley.get(n, 0.0):>+14.3f} "
              f"{strategies.get(n, ''):>16}")
    print(f"\nStructural failures (pipeline errors) across coalition runs:")
    print(f"  pure ablation: {abl_errors}/{abl_total}")
    print(f"  auto (degrade structural modules): {auto_errors}/{auto_total}")

    report = {
        "system": "RAG retriever -> reranker -> synthesizer (offline, deterministic)",
        "top_k": TOP_K,
        "n_cases": len(cases),
        "removal_strategies": strategies,
        "pure_ablation": {"shapley": abl_shapley, "structural_failures": abl_errors, "runs": abl_total},
        "auto_degrade": {"shapley": auto_shapley, "structural_failures": auto_errors, "runs": auto_total},
        "lesson": (
            "Removing the retriever entirely (pure ablation) leaves the synthesizer with no "
            "context and the run structurally fails, so the retriever's attribution only says "
            "'necessary'. diagnose instead severely degrades the retriever (and reranker): the "
            "run stays live and their contribution is measured as a real quality effect."
        ),
    }
    os.makedirs(REPORTS, exist_ok=True)
    json_path = os.path.join(REPORTS, "rag_degradation_casestudy.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {json_path}")
    return report


def main():
    run()


if __name__ == "__main__":
    main()
