"""Contrast pure ablation with graded degradation on the RAG pipeline.

The pipeline passes its eval, so this is a *fragility* question: which module's
quality actually controls the answer? We run both lenses on the same cases:

  1. Ablation (``diagnose``) — Shapley from replacing each node with a no-op.
  2. Degradation (``diagnose_sensitivity``) — the dose-response classification.

and print them side by side. The point: ablation calls the reranker irrelevant
(removing it is a harmless pass-through), while degradation shows it is a quality
driver (decaying its ranking pushes the relevant passage out of the synthesizer's
top-k and answers fail).

Run: PYTHONPATH=examples python -m rag_degradation_skill.run_casestudy
"""

from __future__ import annotations

import json
import os
from collections import Counter

from counterfact.classifiers import ClassifierRegistry

from .pipeline import TOP_K, build

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
SIMS = 24
SEED = 42
MAGNITUDES = (0.25, 0.5, 0.75, 1.0)


def _load_cases():
    with open(os.path.join(HERE, "cases.json")) as f:
        return json.load(f)


def _quality_fn(output_text, state):
    # Gold is carried under a leading-underscore key so it is never mistaken for
    # the pipeline's output (counterfact ignores _-prefixed state keys).
    return 1.0 if state.get("_gold", "\0") in (output_text or "") else 0.0


def _ablation_shapley(cases):
    """Average per-node Shapley from pure ablation across the cases."""
    agg, counts = {}, {}
    for case in cases:
        state = {"query": case["query"], "_gold": case["gold"]}
        report = build().diagnose(
            input_state=state,
            num_simulations=SIMS,
            quality_fn=_quality_fn,
            registry=ClassifierRegistry(),
            run_evals=False,
            quality_gate=1.01,  # the pipeline passes; force attribution anyway
            seed=SEED,
        )
        for node, val in (report.shapley_values or {}).items():
            agg[node] = agg.get(node, 0.0) + val
            counts[node] = counts.get(node, 0) + 1
    return {n: agg[n] / counts[n] for n in agg}


def _degradation(cases):
    """Average per-node sensitivity + majority classification across the cases."""
    sens, partial, classes, curves = {}, {}, {}, {}
    for case in cases:
        state = {"query": case["query"], "_gold": case["gold"]}
        report = build().diagnose_sensitivity(
            state,
            quality_fn=_quality_fn,
            registry=ClassifierRegistry(),
            magnitudes=MAGNITUDES,
            seed=SEED,
        )
        for n in report.nodes:
            sens.setdefault(n.node, []).append(n.sensitivity)
            partial.setdefault(n.node, []).append(n.partial_sensitivity)
            classes.setdefault(n.node, []).append(n.classification)
            curves.setdefault(n.node, []).append([q for _, q in n.curve])
    out = {}
    for node in sens:
        mean_curve = [round(sum(col) / len(col), 3) for col in zip(*curves[node])]
        out[node] = {
            "sensitivity": round(sum(sens[node]) / len(sens[node]), 3),
            "partial_sensitivity": round(sum(partial[node]) / len(partial[node]), 3),
            "classification": Counter(classes[node]).most_common(1)[0][0],
            "mean_curve": mean_curve,
        }
    return out


def run():
    cases = _load_cases()
    print(f"RAG pipeline (retriever -> reranker -> synthesizer), synthesizer reads top-{TOP_K}.")
    print(f"{len(cases)} cases. Baseline passes; asking which module's quality drives the answer.\n")

    ablation = _ablation_shapley(cases)
    degradation = _degradation(cases)

    nodes = ["retriever", "reranker", "synthesizer"]
    header = f"{'node':14} {'ablation Shapley':>17} {'degradation class':>20} {'max quality drop':>17}"
    print(header)
    for n in nodes:
        ab = ablation.get(n, 0.0)
        dg = degradation.get(n, {})
        drop = max(dg.get("sensitivity", 0.0), dg.get("partial_sensitivity", 0.0))
        print(f"{n:14} {ab:>+17.3f} {dg.get('classification',''):>20} {drop:>17.3f}")

    report = {
        "system": "RAG retriever -> reranker -> synthesizer (offline, deterministic)",
        "top_k": TOP_K,
        "n_cases": len(cases),
        "magnitudes": list(MAGNITUDES),
        "ablation_shapley": {n: round(ablation.get(n, 0.0), 4) for n in nodes},
        "degradation": {n: degradation.get(n, {}) for n in nodes},
        "lesson": (
            "Ablation calls the reranker irrelevant (removing it is a harmless pass-through, "
            "Shapley ~0), but graded degradation classifies it a quality_driver: decaying its "
            "ranking pushes the relevant passage out of the synthesizer's top-k and answers "
            "fail. The retriever is structural (only full removal hurts)."
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
