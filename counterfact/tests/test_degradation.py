"""Tests for the ablate-vs-severely-degrade removal strategy."""

from counterfact import END, StateGraph
from counterfact.classifiers import ClassifierRegistry
from counterfact.degradation import (
    _destroy,
    decide_removals,
    infer_module_type,
    removal_strategy,
    severe_degraded_node,
)

# ── module-type inference + strategy ───────────────────────────────────────

def test_infer_module_type_by_name_and_shape():
    assert infer_module_type("doc_retriever", None) == "retriever"
    assert infer_module_type("reranker", [1, 2]) == "reranker"
    assert infer_module_type("query_parser", {}) == "parser"
    assert infer_module_type("writer", "prose") == "generator"
    assert infer_module_type("mystery", [1, 2, 3]) == "retriever"  # shape fallback
    assert infer_module_type("mystery", {"k": 1}) == "parser"
    assert infer_module_type("mystery", "text") == "generator"


def test_removal_strategy_mapping():
    assert removal_strategy("retriever") == "degrade"
    assert removal_strategy("reranker") == "degrade"
    assert removal_strategy("parser") == "degrade"
    assert removal_strategy("generator") == "ablate"


# ── structure-preserving destroy ───────────────────────────────────────────

def test_destroy_preserves_shape():
    out = _destroy(["a", "b", "c"])
    assert isinstance(out, list) and len(out) == 3 and all(out)  # same length, non-empty
    assert "a" not in out  # content gone

    d = _destroy({"x": 1, "y": 2})
    assert set(d.keys()) == {"x", "y"} and all(v == "" for v in d.values())  # keys kept

    assert _destroy("hello") == ""
    assert _destroy([]) == []        # empty stays empty
    assert _destroy(42) == 42        # scalars unchanged


def test_severe_degraded_node_runs_then_destroys():
    def retriever(state):
        return {**state, "docs": ["relevant doc", "d2", "d3"]}

    node = severe_degraded_node(retriever)
    out = node({"query": "q"})
    assert len(out["docs"]) == 3            # shape preserved (non-empty)
    assert "relevant doc" not in out["docs"]  # content destroyed


# ── decide_removals on a real pipeline ─────────────────────────────────────

def _rag(top_k=2):
    docs = ["GOLD-42 is the answer.", "noise1", "noise2", "noise3"]

    def retriever(s):
        return {**s, "docs": list(docs)}

    def reranker(s):
        return {**s, "docs": s.get("docs", [])}

    def synthesizer(s):
        ctx = s.get("docs", [])[:top_k]
        return {**s, "final_output": ("GOLD-42 found" if any("GOLD-42" in d for d in ctx) else "not found")}

    g = StateGraph(dict)
    g.add_node("retriever", retriever)
    g.add_node("reranker", reranker)
    g.add_node("synthesizer", synthesizer)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "reranker")
    g.add_edge("reranker", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def test_decide_removals_classifies_structural_vs_generator():
    strategies = decide_removals(_rag(), {"query": "q"})
    assert strategies["retriever"] == "degrade"
    assert strategies["reranker"] == "degrade"
    assert strategies["synthesizer"] == "ablate"  # generator


def test_clone_with_ablation_respects_strategy():
    g = _rag()
    g._removals = {"retriever": "degrade", "synthesizer": "ablate"}

    # Degrade: retriever still produces a non-empty docs list, but useless content.
    degraded = g.clone_with_ablation("retriever")
    res = degraded.invoke({"query": "q"})
    assert len(res["docs"]) == 4           # shape preserved (no structural collapse)
    assert not any("GOLD-42" in d for d in res["docs"])  # content destroyed

    # Ablate: synthesizer is a no-op (final_output not produced).
    ablated = g.clone_with_ablation("synthesizer")
    res2 = ablated.invoke({"query": "q"})
    assert "GOLD-42 found" not in res2.get("final_output", "")


def test_clone_with_ablation_defaults_to_ablate_without_strategy():
    g = _rag()  # no _removals set
    ablated = g.clone_with_ablation("retriever")
    res = ablated.invoke({"query": "q"})
    # Pure ablation (no-op): retriever does not set docs -> empty downstream.
    assert res.get("docs", []) == []


# ── end-to-end diagnose uses degradation, no structural collapse ───────────

def test_diagnose_degrades_retriever_without_collapse():
    g = _rag(top_k=2)

    def qf(out, state):
        return 1.0 if "GOLD-42 found" in (out or "") else 0.0

    report = g.diagnose(
        input_state={"query": "q"},
        num_simulations=12,
        quality_fn=qf,
        registry=ClassifierRegistry(),
        run_evals=False,
        quality_gate=1.01,  # pipeline passes; force attribution
        seed=0,
    )
    strategies = report.simulation_results_summary["removal_strategies"]
    assert strategies["retriever"] == "degrade"
    # Attribution computed, finite, and the structural retriever is implicated.
    assert "retriever" in report.shapley_values
    assert all(abs(v) < 10 for v in report.shapley_values.values())  # no blow-up
