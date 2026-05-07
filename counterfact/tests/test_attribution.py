import numpy as np
import pytest

from counterfact.attribution import classify_failure, compute_bootstrap_ci, compute_shapley_values
from counterfact.types import ClassifierResult, ConfidenceInterval, Perturbation, SimulationResult


def test_compute_bootstrap_ci_empty():
    ci = compute_bootstrap_ci([])
    assert ci.mean == 0.0
    assert ci.n_samples == 0

def test_compute_bootstrap_ci_single():
    ci = compute_bootstrap_ci([0.5])
    assert ci.mean == 0.5
    assert ci.n_samples == 1

def test_compute_bootstrap_ci_multiple():
    np.random.seed(42)
    ci = compute_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=100)
    assert ci.mean == pytest.approx(0.3)
    assert ci.n_samples == 5

def _make_sim(sim_id, agents_ablated="", is_baseline=False, quality=0.5, clf_scores=None):
    if clf_scores is None:
        clf_scores = {"c1": quality}
    clfs = [ClassifierResult(k, v, "ok", 1.0) for k, v in clf_scores.items()]
    pert = None if is_baseline else Perturbation(agent=agents_ablated, strategy="ablate", description="", magnitude=1.0)
    return SimulationResult(sim_id, pert, quality, clfs, "out", is_baseline, [])

def _make_trace(agents):
    return [{"node": a} for a in agents]

def test_compute_shapley_values_basic():
    # 2 agents: A, B. Full coalition = {A, B}
    # Empty = {}
    # A ablated = {B}
    # B ablated = {A}
    trace = _make_trace(["A", "B"])

    sims = [
        _make_sim(0, is_baseline=True, quality=1.0, clf_scores={"c1": 1.0}), # {A, B}
        _make_sim(1, "A", quality=0.5, clf_scores={"c1": 0.5}), # {B}
        _make_sim(2, "B", quality=0.5, clf_scores={"c1": 0.5}), # {A}
        _make_sim(3, "A, B", quality=0.0, clf_scores={"c1": 0.0}), # {}
    ]

    shapley, cis, per_clf = compute_shapley_values(sims, trace)

    # Total value = 1.0. Marginal of A = ({A} - {}) + ({A,B} - {B}) / 2 = (0.5 - 0) + (1.0 - 0.5) / 2 = 0.5
    assert shapley["A"] == pytest.approx(0.5)
    assert shapley["B"] == pytest.approx(0.5)

    assert "c1" in per_clf
    assert per_clf["c1"]["A"] == pytest.approx(0.5)
    assert per_clf["c1"]["B"] == pytest.approx(0.5)

def test_shapley_asymmetric_two_agents():
    """Asymmetric 2-agent case — the Shapley values should differ.

    v({}) = 0.0, v({A}) = 0.3, v({B}) = 0.7, v({A,B}) = 0.9

    φ(A) = ½[v({A}) - v({})] + ½[v({A,B}) - v({B})]
         = ½(0.3) + ½(0.2) = 0.25
    φ(B) = ½[v({B}) - v({})] + ½[v({A,B}) - v({A})]
         = ½(0.7) + ½(0.6) = 0.65

    After normalization by sum(|φ|) = 0.9:
      A = 0.25/0.9 ≈ 0.2778, B = 0.65/0.9 ≈ 0.7222
    """
    trace = _make_trace(["A", "B"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.9),   # {A, B} → 0.9
        _make_sim(1, "A", quality=0.7),                 # {B}   → 0.7
        _make_sim(2, "B", quality=0.3),                 # {A}   → 0.3
        _make_sim(3, "A, B", quality=0.0),              # {}    → 0.0
    ]
    shapley, cis, _ = compute_shapley_values(sims, trace)

    assert shapley["A"] == pytest.approx(0.25 / 0.9, abs=1e-4)
    assert shapley["B"] == pytest.approx(0.65 / 0.9, abs=1e-4)
    # B should dominate since it contributes more in both orderings
    assert shapley["B"] > shapley["A"]


def test_shapley_three_agents_exact():
    """3-agent case with all 2^3 coalitions — verify exact Shapley formula.

    v({})=0, v({A})=0.2, v({B})=0.1, v({C})=0,
    v({A,B})=0.5, v({A,C})=0.3, v({B,C})=0.2, v({A,B,C})=0.8

    φ(A) = 1/3(0.2) + 1/6(0.4) + 1/6(0.3) + 1/3(0.6) = 23/60
    φ(B) = 1/3(0.1) + 1/6(0.3) + 1/6(0.2) + 1/3(0.5) = 17/60
    φ(C) = 1/3(0.0) + 1/6(0.1) + 1/6(0.1) + 1/3(0.3) = 8/60
    """
    trace = _make_trace(["A", "B", "C"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.8),    # {A,B,C}
        _make_sim(1, "A", quality=0.2),                  # {B,C}
        _make_sim(2, "B", quality=0.3),                  # {A,C}
        _make_sim(3, "C", quality=0.5),                  # {A,B}
        _make_sim(4, "A, B", quality=0.0),               # {C}
        _make_sim(5, "A, C", quality=0.1),               # {B}
        _make_sim(6, "B, C", quality=0.2),               # {A}
        _make_sim(7, "A, B, C", quality=0.0),            # {}
    ]
    shapley, cis, _ = compute_shapley_values(sims, trace)

    total_raw = 23 / 60 + 17 / 60 + 8 / 60  # = 0.8
    assert shapley["A"] == pytest.approx((23 / 60) / total_raw, abs=1e-4)
    assert shapley["B"] == pytest.approx((17 / 60) / total_raw, abs=1e-4)
    assert shapley["C"] == pytest.approx((8 / 60) / total_raw, abs=1e-4)

    # Ordering should be A > B > C
    assert shapley["A"] > shapley["B"] > shapley["C"]


def test_shapley_efficiency_axiom():
    """Pre-normalization Shapley values must sum to v(N) - v(∅).

    This is the efficiency axiom: the total "pie" is fully distributed.
    """
    trace = _make_trace(["A", "B", "C"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.8),
        _make_sim(1, "A", quality=0.2),
        _make_sim(2, "B", quality=0.3),
        _make_sim(3, "C", quality=0.5),
        _make_sim(4, "A, B", quality=0.0),
        _make_sim(5, "A, C", quality=0.1),
        _make_sim(6, "B, C", quality=0.2),
        _make_sim(7, "A, B, C", quality=0.0),
    ]
    shapley, _, _ = compute_shapley_values(sims, trace)

    # After normalization, absolute values should sum to 1.0
    assert sum(abs(v) for v in shapley.values()) == pytest.approx(1.0, abs=1e-6)


def test_shapley_sparse_loo_only():
    """With only LOO data (no multi-agent ablations), Shapley degrades to LOO.

    Only v(N), v(N\\{i}) for each i, and v({}) are available.
    The only computable marginals are at k = N-1.

    v({A,B,C})=0.8, v({B,C})=0.2, v({A,C})=0.3, v({A,B})=0.5, v({})=0.0
    LOO marginals: A=0.6, B=0.5, C=0.3 → normalized: A=0.6/1.4, B=0.5/1.4, C=0.3/1.4
    """
    trace = _make_trace(["A", "B", "C"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.8),    # {A,B,C}
        _make_sim(1, "A", quality=0.2),                  # {B,C}
        _make_sim(2, "B", quality=0.3),                  # {A,C}
        _make_sim(3, "C", quality=0.5),                  # {A,B}
        # No multi-agent ablations — only single-agent LOO
    ]
    shapley, _, _ = compute_shapley_values(sims, trace)

    # With only LOO data, marginals are only from the top coalition:
    # A: v({A,B,C}) - v({B,C}) = 0.6
    # B: v({A,B,C}) - v({A,C}) = 0.5
    # C: v({A,B,C}) - v({A,B}) = 0.3
    total = 0.6 + 0.5 + 0.3  # = 1.4
    assert shapley["A"] == pytest.approx(0.6 / total, abs=1e-4)
    assert shapley["B"] == pytest.approx(0.5 / total, abs=1e-4)
    assert shapley["C"] == pytest.approx(0.3 / total, abs=1e-4)


def test_shapley_negative_contributions():
    """An agent that hurts quality should get a negative Shapley value.

    v({})=0.0, v({A})=0.6, v({B})=0.0, v({A,B})=0.4
    Adding B to {A} reduces quality: v({A,B}) - v({A}) = -0.2

    φ(A) = ½(0.6) + ½(0.4) = 0.5
    φ(B) = ½(0.0) + ½(-0.2) = -0.1

    Normalized by sum(|φ|) = 0.6: A ≈ 0.833, B ≈ -0.167
    """
    trace = _make_trace(["A", "B"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.4),   # {A,B}
        _make_sim(1, "A", quality=0.0),                 # {B}
        _make_sim(2, "B", quality=0.6),                 # {A}
        _make_sim(3, "A, B", quality=0.0),              # {}
    ]
    shapley, _, _ = compute_shapley_values(sims, trace)

    assert shapley["B"] < 0, "Agent B hurts quality and should have negative Shapley"
    assert shapley["A"] > 0
    assert shapley["A"] == pytest.approx(0.5 / 0.6, abs=1e-4)
    assert shapley["B"] == pytest.approx(-0.1 / 0.6, abs=1e-4)


def test_shapley_per_classifier_asymmetric():
    """Per-classifier Shapley values should differ when agents affect different classifiers.

    Agent A primarily affects c1, agent B primarily affects c2.
    """
    trace = _make_trace(["A", "B"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.8, clf_scores={"c1": 0.9, "c2": 0.7}),  # {A,B}
        _make_sim(1, "A", quality=0.4, clf_scores={"c1": 0.2, "c2": 0.6}),               # {B}
        _make_sim(2, "B", quality=0.5, clf_scores={"c1": 0.8, "c2": 0.2}),               # {A}
        _make_sim(3, "A, B", quality=0.0, clf_scores={"c1": 0.0, "c2": 0.0}),            # {}
    ]
    _, _, per_clf = compute_shapley_values(sims, trace)

    assert "c1" in per_clf
    assert "c2" in per_clf
    # A should dominate c1 (removing A tanks c1: 0.9 → 0.2)
    assert abs(per_clf["c1"]["A"]) > abs(per_clf["c1"]["B"])
    # B should dominate c2 (removing B tanks c2: 0.7 → 0.2)
    assert abs(per_clf["c2"]["B"]) > abs(per_clf["c2"]["A"])


def test_compute_per_classifier_loo():
    from counterfact.attribution import compute_per_classifier_loo
    trace = _make_trace(["A", "B"])
    sims = [
        _make_sim(0, is_baseline=True, quality=0.8, clf_scores={"c1": 0.8, "c2": 0.9}),
        _make_sim(1, "A", quality=0.5, clf_scores={"c1": 0.5, "c2": 0.9}),
        _make_sim(2, "B", quality=0.5, clf_scores={"c1": 0.8, "c2": 0.5}),
    ]
    per_clf = compute_per_classifier_loo(sims, trace)
    assert "c1" in per_clf
    assert "c2" in per_clf
    assert per_clf["c1"]["A"] < 0
    assert per_clf["c2"]["B"] < 0

def test_compute_shapley_values_missing_agents():
    assert compute_shapley_values([], []) == ({}, {}, {})

def test_compute_loo_attribution():
    from counterfact.attribution import compute_loo_attribution
    trace = [{"node": "a"}, {"node": "b"}, {"node": "output"}]
    sims = [
        _make_sim(0, is_baseline=True, quality=0.8),
        _make_sim(1, "a", quality=0.5), # removing a hurts -> a is helpful (value > 0 is bad, wait LOO is ablate - baseline)
        _make_sim(2, "b", quality=0.9), # removing b helps -> b is harmful
    ]
    # LOO = ablate - baseline
    # a: 0.5 - 0.8 = -0.3
    # b: 0.9 - 0.8 = 0.1
    attr = compute_loo_attribution(sims, trace)
    assert attr["a"] == pytest.approx(-0.3)
    assert attr["b"] == pytest.approx(0.1)
    assert "output" not in attr

def test_is_loo_inconclusive():
    from counterfact.attribution import is_loo_inconclusive
    assert is_loo_inconclusive({}) is True
    assert is_loo_inconclusive({"a": 0.01, "b": -0.01}) is True
    assert is_loo_inconclusive({"a": 0.5, "b": 0.0}) is False
    assert is_loo_inconclusive({"a": 0.01, "b": 0.06}, threshold=0.04) is False

def test_classify_failure_local():
    trace = _make_trace(["a", "b"])
    sims = [_make_sim(0, is_baseline=True, quality=0.5)]
    attribution = {"a": 0.9, "b": 0.1}
    cis = {"a": ConfidenceInterval(0.9, 0.8, 1.0, 10), "b": ConfidenceInterval(0.1, 0.0, 0.2, 10)}
    per_clf = {"c1": {"a": 0.9, "b": 0.1}}

    cls = classify_failure(attribution, sims, trace, per_clf, cis)
    assert cls.failure_type == "local"
    assert "a" in cls.description

def test_classify_failure_architectural():
    trace = _make_trace(["a", "b", "c"])
    sims = [_make_sim(0, is_baseline=True, quality=0.5)]
    attribution = {"a": 0.33, "b": 0.33, "c": 0.33}
    cis = {
        "a": ConfidenceInterval(0.33, 0.3, 0.4, 10),
        "b": ConfidenceInterval(0.33, 0.3, 0.4, 10),
        "c": ConfidenceInterval(0.33, 0.3, 0.4, 10),
    }
    per_clf = {"c1": {"a": 0.33, "b": 0.33, "c": 0.33}}

    cls = classify_failure(attribution, sims, trace, per_clf, cis)
    assert cls.failure_type == "systemic"

def test_classify_failure_architectural_gap():
    trace = _make_trace(["a", "b", "c"])
    sims = [_make_sim(0, is_baseline=True, quality=0.5)]
    attribution = {"a": 0.1, "b": 0.1, "c": 0.1}
    cis = {
        "a": ConfidenceInterval(0.1, 0.05, 0.15, 10),
        "b": ConfidenceInterval(0.1, 0.05, 0.15, 10),
        "c": ConfidenceInterval(0.1, 0.05, 0.15, 10),
    }
    per_clf = {"c1": {"a": 0.1, "b": 0.1, "c": 0.1}}

    cls = classify_failure(attribution, sims, trace, per_clf, cis)
    assert cls.failure_type == "architectural_gap"

def test_classify_failure_feedback_amplification():
    trace = [{"node": "synthesizer"}, {"node": "critic"}, {"node": "synthesizer"}]
    # Baseline 0.5, but ablating critic gets 0.8 => damping_ratio = 1.6 > 1.05
    sims = [
        _make_sim(0, is_baseline=True, quality=0.5),
        _make_sim(1, "critic", quality=0.8),
    ]
    attribution = {"synthesizer": 0.3, "critic": 0.3}
    cis = {"synthesizer": ConfidenceInterval(0.3, 0.2, 0.4, 10), "critic": ConfidenceInterval(0.3, 0.2, 0.4, 10)}
    per_clf = {"c1": {"synthesizer": 0.3, "critic": 0.3}}

    cls = classify_failure(attribution, sims, trace, per_clf, cis)
    assert cls.failure_type == "feedback_amplification"
