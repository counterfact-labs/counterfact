"""Tests for graded-degradation sensitivity analysis."""

import random

from counterfact import END, StateGraph
from counterfact.classifiers import ClassifierRegistry
from counterfact.sensitivity import (
    default_degrader_for,
    drop_fields,
    drop_items,
    drop_sentences,
    infer_module_type,
    inject_distractors,
    shuffle_relevance,
    truncate_text,
)

RNG = random.Random(0)


# ── Degrader unit tests ────────────────────────────────────────────────────

def test_drop_items_magnitude_endpoints():
    d = drop_items()
    items = [1, 2, 3, 4]
    assert d(items, 0.0, RNG) == [1, 2, 3, 4]   # unchanged
    assert d(items, 1.0, RNG) == []             # ablation-equivalent
    assert d(items, 0.5, RNG) == [1, 2]         # keep prefix
    assert d("not a list", 1.0, RNG) == "not a list"


def test_truncate_text_endpoints():
    d = truncate_text()
    assert d("abcdefgh", 0.0, RNG) == "abcdefgh"
    assert d("abcdefgh", 1.0, RNG) == ""
    assert d("abcdefgh", 0.5, RNG) == "abcd"


def test_inject_distractors_replaces_fraction():
    d = inject_distractors(distractor="JUNK")
    out = d(["a", "b", "c", "d"], 0.5, random.Random(1))
    assert out.count("JUNK") == 2
    assert d(["a", "b"], 0.0, RNG) == ["a", "b"]


def test_shuffle_relevance_reverses_at_full():
    d = shuffle_relevance()
    out = d([1, 2, 3, 4], 1.0, RNG)
    assert out[0] == 4 and out[-1] == 1  # good items pushed to the back
    assert d([1, 2, 3], 0.0, RNG) == [1, 2, 3]


def test_drop_sentences_and_fields():
    assert drop_sentences()("One. Two. Three. Four.", 1.0, RNG) == ""
    out = drop_fields()({"a": 1, "b": 2, "c": 3, "d": 4}, 0.5, random.Random(2))
    assert len(out) == 2


def test_degrader_names_are_readable():
    assert drop_items().__name__ == "drop_items"
    assert shuffle_relevance().__name__ == "shuffle_relevance"


# ── Inference / selection ──────────────────────────────────────────────────

def test_infer_module_type_by_name_and_shape():
    assert infer_module_type("doc_retriever", None) == "retriever"
    assert infer_module_type("reranker", [1, 2]) == "reranker"
    assert infer_module_type("query_parser", {}) == "parser"
    assert infer_module_type("writer", "prose") == "generator"
    assert infer_module_type("mystery", [1, 2, 3]) == "retriever"  # shape fallback


def test_default_degrader_selection():
    assert default_degrader_for("retriever", ["d1"]).__name__ == "drop_items"
    assert default_degrader_for("reranker", ["d1", "d2"]).__name__ == "shuffle_relevance"
    assert default_degrader_for("parser", {"k": 1}).__name__ == "drop_fields"
    assert default_degrader_for("generator", "text").__name__ == "drop_sentences"


# ── End-to-end on a small RAG-ish pipeline ─────────────────────────────────

GOLD = "$1,577 million"


def _build_pipeline(top_k=None):
    docs = [f"{GOLD} is 3M FY2018 capex.", "weather", "stock tips", "sports", "recipes"]

    def retriever(state):
        return {**state, "docs": list(docs)}

    def reranker(state):
        return {**state, "docs": state.get("docs", [])}

    def synthesizer(state):
        ctx = state.get("docs", [])
        if top_k is not None:
            ctx = ctx[:top_k]
        hit = any(GOLD in d for d in ctx)
        return {**state, "final_output": (f"Answer: {GOLD}." if hit else "Figure not found.")}

    g = StateGraph(dict)
    g.add_node("retriever", retriever)
    g.add_node("reranker", reranker)
    g.add_node("synthesizer", synthesizer)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "reranker")
    g.add_edge("reranker", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def _qf(out, state):
    return 1.0 if GOLD in (out or "") else 0.0


def test_retriever_is_structural_when_synth_scans_all():
    # synth scans every doc, so dropping the prefix keeps the relevant (first) doc
    # until full ablation -> structural, not quality-driven.
    report = _build_pipeline(top_k=None).diagnose_sensitivity(
        {"query": "capex?"}, quality_fn=_qf, registry=ClassifierRegistry(),
        magnitudes=(0.25, 0.5, 1.0), seed=0,
    )
    by_node = {n.node: n for n in report.nodes}
    assert by_node["retriever"].classification == "structural"
    assert by_node["retriever"].target_key == "docs"
    assert by_node["retriever"].module_type == "retriever"


def test_reranker_degradation_drives_quality_with_topk_synth():
    # With a top-2 synthesizer, decaying the ranking buries the relevant doc out
    # of the window -> graded degradation reveals a quality driver that pure
    # ablation could not (ablating reranker just passes docs through unchanged).
    report = _build_pipeline(top_k=2).diagnose_sensitivity(
        {"query": "capex?"},
        quality_fn=_qf,
        registry=ClassifierRegistry(),
        degraders={"reranker": shuffle_relevance()},
        magnitudes=(0.25, 0.5, 0.75, 1.0),
        seed=0,
    )
    by_node = {n.node: n for n in report.nodes}
    assert by_node["reranker"].classification in ("quality_driver", "structural")
    assert by_node["reranker"].partial_sensitivity > 0.0  # hurt before full magnitude


def test_harmful_node_detected():
    # A node that corrupts a correct upstream answer should read as harmful:
    # degrading/removing it improves quality.
    def good(state):
        return {**state, "final_output": f"Answer: {GOLD}."}

    def corrupter(state):
        return {**state, "final_output": "Answer: approximately $1.6 billion."}

    g = StateGraph(dict)
    g.add_node("good", good)
    g.add_node("corrupter", corrupter)
    g.set_entry_point("good")
    g.add_edge("good", "corrupter")
    g.add_edge("corrupter", END)
    report = g.compile().diagnose_sensitivity(
        {"query": "capex?"}, quality_fn=_qf, registry=ClassifierRegistry(),
        magnitudes=(0.5, 1.0), seed=0,
    )
    by_node = {n.node: n for n in report.nodes}
    assert by_node["corrupter"].classification == "harmful"
    assert by_node["corrupter"].sensitivity < 0  # removing it raised quality


def test_report_serialization_and_ranking():
    report = _build_pipeline(top_k=None).diagnose_sensitivity(
        {"query": "capex?"}, quality_fn=_qf, registry=ClassifierRegistry(),
        magnitudes=(0.5, 1.0), seed=0,
    )
    d = report.to_dict()
    assert "nodes" in d and d["most_sensitive"] is not None
    assert report.most_sensitive() is report.ranked()[0]
    md = report.to_markdown()
    assert "Sensitivity analysis" in md and "retriever" in md
