import pytest
import numpy as np
from counterfact.types import ClassifierResult, SimulationResult, Perturbation, ConfidenceInterval
from counterfact.attribution import compute_bootstrap_ci, compute_shapley_values, classify_failure

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
    from counterfact.types import Perturbation
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

def test_compute_shapley_values_missing_agents():
    assert compute_shapley_values([], []) == ({}, {}, {})

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
