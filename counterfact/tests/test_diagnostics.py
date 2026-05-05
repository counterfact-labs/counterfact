from unittest.mock import MagicMock, patch

from counterfact.types import (
    SimulationResult,
    ClassifierResult,
    FailureClassification,
    Perturbation,
    Recommendation,
    ConfidenceInterval,
)
from counterfact.diagnostics import (
    DiagnosticReport,
    run_full_diagnostic,
    _make_no_failure_classification,
    _build_summary,
)

# ─── Data models ─────────────────────────────────────────────────────────

def test_diagnostic_report_serialization():
    report = DiagnosticReport(
        query="q",
        domain="d",
        baseline_quality=0.9,
        shapley_values={"a": 0.5},
        per_classifier_shapley={"c1": {"a": 0.5}},
        classification=FailureClassification("local", 0.9, "desc", ["ev"], "a", []),
        recommendations=[Recommendation("desc", 0.5, "measured", "local", 0.5, "low", 1)],
        evaluations=[],
        num_simulations=10,
        simulation_results=[],
        simulation_results_summary={},
        baseline_quality_ci=ConfidenceInterval(0.9, 0.8, 1.0, 5),
        shapley_cis={"a": ConfidenceInterval(0.5, 0.4, 0.6, 5)},
        seed=42,
    )

    d = report.to_dict()
    assert d["query"] == "q"
    assert d["baseline_quality"] == 0.9
    assert d["classification"]["failure_type"] == "local"
    assert "a" in d["shapley_cis"]
    assert d["baseline_quality_ci"]["mean"] == 0.9
    
    with patch("counterfact.export.to_json", return_value="json"):
        assert report.to_json() == "json"
        
    with patch("counterfact.export.to_markdown", return_value="md"):
        assert report.to_markdown() == "md"
        
    with patch("counterfact.export.to_html", return_value="html"):
        assert report.to_html() == "html"


# ─── Helper functions ───────────────────────────────────────────────────

def test_make_no_failure_classification():
    baseline_results = [
        SimulationResult(
            simulation_id=0,
            perturbation=None,
            quality_score=0.9,
            classifier_results=[ClassifierResult("c1", 0.9, "")],
            is_baseline=True
        )
    ]
    cls = _make_no_failure_classification(0.9, baseline_results)
    assert cls.failure_type == "no_failure"
    assert cls.confidence == 0.9
    assert "passes quality checks" in cls.description


def test_build_summary():
    sims = [MagicMock(classifier_results=[MagicMock()]), MagicMock(classifier_results=[MagicMock()])]
    baselines = [MagicMock(quality_score=0.9, classifier_results=[MagicMock(name="c1")])]
    
    summary = _build_summary(sims, [0.9], 0.9, ["a"], "loo", True, baselines)
    assert summary["total_simulations"] == 2
    assert summary["baseline_runs"] == 1
    assert summary["baseline_quality_mean"] == 0.9


# ─── Full Diagnostic Run ────────────────────────────────────────────────

@patch("counterfact.diagnostics.run_eval_suite")
@patch("counterfact.diagnostics.run_monte_carlo")
@patch("counterfact.diagnostics.compute_shapley_values")
@patch("counterfact.diagnostics.classify_failure")
@patch("counterfact.diagnostics.extract_empirical_fixes")
@patch("counterfact.diagnostics.rank_recommendations")
def test_run_full_diagnostic_quality_gate(
    mock_rank, mock_extract, mock_classify, mock_shapley, mock_mc, mock_eval
):
    # Setup graph mock
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"output": "text"}
    mock_graph.get_trace.return_value = [{"node": "a", "status": "pass"}]
    mock_graph.get_node_names.return_value = ["a"]

    # Mock Monte Carlo to return high quality
    sim = SimulationResult(
        simulation_id=0,
        perturbation=None,
        quality_score=0.95,
        classifier_results=[ClassifierResult("c1", 0.95, "")],
        is_baseline=True
    )
    mock_mc.return_value = [sim]

    report = run_full_diagnostic(mock_graph, {"query": "hi"}, quality_gate=0.8, run_evals=False)
    
    # Should skip attribution
    mock_shapley.assert_not_called()
    assert report.classification.failure_type == "no_failure"
    assert report.baseline_quality == 0.95
    assert report.attribution_method == "quality_gate"


@patch("counterfact.diagnostics.run_eval_suite")
@patch("counterfact.diagnostics.run_monte_carlo")
@patch("counterfact.diagnostics.compute_shapley_values")
@patch("counterfact.diagnostics.compute_per_classifier_loo")
@patch("counterfact.diagnostics.classify_failure")
@patch("counterfact.diagnostics.extract_empirical_fixes")
@patch("counterfact.diagnostics.detect_coverage_gaps")
@patch("counterfact.diagnostics.rank_recommendations")
def test_run_full_diagnostic_full_path(
    mock_rank, mock_detect, mock_extract, mock_classify, mock_loo, mock_shapley, mock_mc, mock_eval
):
    # Setup graph mock
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = "text output directly" # hit output extraction edge case
    mock_graph.get_trace.return_value = [{"node": "a", "status": "pass"}]
    mock_graph.get_node_names.return_value = ["a"]

    # Eval suite
    mock_eval.return_value = MagicMock()

    # Monte Carlo returns low quality
    sim = SimulationResult(
        simulation_id=0,
        perturbation=None,
        quality_score=0.4,
        classifier_results=[ClassifierResult("c1", 0.4, "")],
        is_baseline=True
    )
    mock_mc.return_value = [sim]

    # Shapley
    mock_shapley.return_value = ({"a": 0.5}, {"a": ConfidenceInterval(0.5, 0.4, 0.6, 5)}, {})
    mock_loo.return_value = {"c1": {"a": 0.5}}

    # Classify
    cls = FailureClassification("architectural_gap", 0.9, "", [], None, ["c1"])
    mock_classify.return_value = cls

    # Recommendations
    rec = Recommendation("desc", 0.5, "measured", "local", 0.5, "low", 1)
    mock_rank.return_value = [rec]
    mock_extract.return_value = []
    mock_detect.return_value = [rec]

    # Run
    report = run_full_diagnostic(mock_graph, {"q": "hi"}, quality_gate=0.8, run_evals=True)
    
    # Should call everything
    mock_shapley.assert_called_once()
    mock_classify.assert_called_once()
    mock_detect.assert_called_once()
    assert report.baseline_quality == 0.4
    assert report.attribution_method == "shapley"
    assert len(report.recommendations) == 1

@patch("counterfact.diagnostics.run_eval_suite")
@patch("counterfact.diagnostics.run_monte_carlo")
@patch("counterfact.diagnostics.compute_shapley_values")
@patch("counterfact.diagnostics.compute_per_classifier_loo")
@patch("counterfact.diagnostics.classify_failure")
@patch("counterfact.diagnostics.extract_empirical_fixes")
@patch("counterfact.diagnostics.detect_coverage_gaps")
@patch("counterfact.diagnostics.rank_recommendations")
@patch("counterfact.diagnostics.generate_recommendations")
@patch("counterfact.diagnostics.evaluate_recommendation")
def test_run_full_diagnostic_evaluate_rec(
    mock_eval_rec, mock_gen, mock_rank, mock_detect, mock_extract, mock_classify, mock_loo, mock_shapley, mock_mc, mock_eval
):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"output": "text"}
    mock_graph.get_trace.return_value = [{"node": "a", "status": "pass"}]
    mock_graph.get_node_names.return_value = ["a"]

    sim = SimulationResult(
        simulation_id=0, perturbation=None, quality_score=0.4,
        classifier_results=[], is_baseline=True
    )
    mock_mc.return_value = [sim]
    
    mock_shapley.return_value = ({"a": 0.5}, {"a": ConfidenceInterval(0.5, 0.4, 0.6, 5)}, {})
    mock_classify.return_value = FailureClassification("local", 0.9, "", [], "a", [])
    
    mock_extract.return_value = []
    mock_detect.return_value = []
    
    # Generate falls back
    rec = Recommendation("desc", 0.5, "estimated", "local", 0.5, "low", 1)
    mock_gen.return_value = [rec]
    mock_rank.return_value = [rec]
    
    mock_eval_rec.return_value = MagicMock()

    # Pass llm_fn so it evaluates
    report = run_full_diagnostic(mock_graph, {"q": "hi"}, quality_gate=0.8, llm_fn=lambda p, t: "a", run_evals=False)
    
    mock_eval_rec.assert_called_once()
    assert len(report.evaluations) == 1


@patch("counterfact.diagnostics.run_eval_suite")
@patch("counterfact.diagnostics.run_monte_carlo")
@patch("counterfact.diagnostics.compute_shapley_values")
def test_run_full_diagnostic_eval_suite_failure(mock_shapley, mock_mc, mock_eval):
    # Test that eval suite throwing an exception doesn't crash the pipeline
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"output": "text"}
    mock_eval.side_effect = Exception("Eval failed")
    
    mock_mc.return_value = [SimulationResult(simulation_id=0, perturbation=None, quality_score=0.9, classifier_results=[], is_baseline=True)]
    mock_shapley.return_value = ({"a": 0.5}, {"a": ConfidenceInterval(0.5, 0.4, 0.6, 5)}, {})
    
    report = run_full_diagnostic(mock_graph, {"q": "hi"}, run_evals=True)
    assert report.eval_suite is None
    assert report.classification.failure_type == "no_failure"


@patch("counterfact.diagnostics.run_eval_suite")
@patch("counterfact.diagnostics.run_monte_carlo")
@patch("counterfact.diagnostics.compute_shapley_values")
@patch("counterfact.diagnostics.classify_failure")
@patch("counterfact.diagnostics.extract_empirical_fixes")
@patch("counterfact.diagnostics.detect_coverage_gaps")
@patch("counterfact.diagnostics.rank_recommendations")
def test_run_full_diagnostic_failure_focused_attribution(
    mock_rank, mock_detect, mock_extract, mock_classify, mock_shapley, mock_mc, mock_eval
):
    # Setup graph mock
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"output": "text"}
    mock_graph.get_trace.return_value = [{"node": "a", "status": "pass"}]
    mock_graph.get_node_names.return_value = ["a"]

    # Monte Carlo returns low quality to avoid quality gate
    sim_baseline = SimulationResult(
        simulation_id=0, perturbation=None, quality_score=0.4,
        classifier_results=[ClassifierResult("c1", 0.4, "")], is_baseline=True
    )
    sim_ablate = SimulationResult(
        simulation_id=1, perturbation=Perturbation("a", "ablate", "", 1.0), quality_score=0.3,
        classifier_results=[ClassifierResult("c1", 0.3, "")], is_baseline=False
    )
    mock_mc.return_value = [sim_baseline, sim_ablate]

    # Shapley values
    mock_shapley.return_value = ({"a": 0.5}, {"a": ConfidenceInterval(0.5, 0.4, 0.6, 5)}, {"c1": {"a": 0.5}})

    # Classify failure as architectural gap with a failing classifier
    cls = FailureClassification("architectural_gap", 0.9, "", [], None, ["c1"])
    mock_classify.return_value = cls

    mock_rank.return_value = []
    mock_extract.return_value = []
    mock_detect.return_value = []

    # Run
    report = run_full_diagnostic(mock_graph, {"q": "hi"}, quality_gate=0.8, run_evals=False)
    
    # Assert failure focused logic was applied (shapley values should still exist)
    assert report.classification.failure_type == "architectural_gap"
    assert "a" in report.shapley_values
    assert report.attribution_method == "shapley"

