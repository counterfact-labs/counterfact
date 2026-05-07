import json

from counterfact.diagnostics import DiagnosticReport
from counterfact.export import _ascii_bar, _format_ci, to_html, to_json, to_markdown
from counterfact.types import AgentSpec, ConfidenceInterval, FailureClassification, Recommendation


def get_mock_report():
    return DiagnosticReport(
        query="test query",
        domain="rag",
        baseline_quality=0.8,
        shapley_values={"agent1": 0.5, "agent2": -0.2},
        shapley_cis={
            "agent1": ConfidenceInterval(mean=0.5, ci_low=0.4, ci_high=0.6, n_samples=10),
            "agent2": ConfidenceInterval(mean=-0.2, ci_low=-0.3, ci_high=-0.1, n_samples=10)
        },
        baseline_quality_ci=ConfidenceInterval(mean=0.8, ci_low=0.7, ci_high=0.9, n_samples=10),
        per_classifier_shapley={},
        classification=FailureClassification(
            failure_type="toxic_loop",
            description="desc",
            failing_classifiers=[],
            evidence=["ev1", "ev2"],
            confidence=0.9
        ),
        recommendations=[
            Recommendation(
                title="Fix agent1",
                description="desc",
                intervention_type="modify_agent",
                target_agent="agent1",
                estimated_failure_reduction=0.3,
                complexity="low",
                priority=1,
                evidence_source="empirical",
                measurement_confidence="measured",
                agent_spec=AgentSpec(name="agent1", position="before", function="do this")
            )
        ],
        evaluations=[],
        num_simulations=10,
        simulation_results=[],
        simulation_results_summary={},
        eval_suite=None,
        attribution_method="loo",
        seed=42
    )

def test_to_json(tmp_path):
    report = get_mock_report()
    res = to_json(report)
    data = json.loads(res)
    assert data["query"] == "test query"
    assert data["baseline_quality"] == 0.8

    # Test file output
    file_path = tmp_path / "report.json"
    to_json(report, str(file_path))
    assert file_path.exists()

def test_format_ci():
    ci = ConfidenceInterval(mean=0.5, ci_low=0.4, ci_high=0.6, n_samples=10)
    assert _format_ci(ci) == "+0.500 [+0.400, +0.600]"
    assert _format_ci(ci, percentage=True) == "+50.0% [+40.0%, +60.0%]"
    assert _format_ci(None) == "N/A"

    ci_empty = ConfidenceInterval(mean=0, ci_low=0, ci_high=0, n_samples=0)
    assert _format_ci(ci_empty) == "N/A"

def test_ascii_bar():
    assert _ascii_bar(0.5, 0.0, 1.0, width=10) == "█████░░░░░"
    assert _ascii_bar(1.5, 0.0, 1.0, width=10) == "██████████"
    assert _ascii_bar(-0.5, 0.0, 1.0, width=10) == "░░░░░░░░░░"
    assert _ascii_bar(0.5, 0.5, 0.5, width=5) == "█████"

def test_to_markdown(tmp_path):
    report = get_mock_report()
    md = to_markdown(report)

    assert "Counterfact Diagnostic Report" in md
    assert "test query" in md
    assert "rag" in md
    assert "toxic_loop" in md
    assert "agent1" in md
    assert "agent2" in md
    assert "Modify Agent" in md

    file_path = tmp_path / "report.md"
    to_markdown(report, str(file_path))
    assert file_path.exists()

def test_to_html(tmp_path):
    report = get_mock_report()
    html = to_html(report)

    assert "<!DOCTYPE html>" in html
    assert "<body>" in html
    assert "Counterfact Diagnostic Report" in html

    file_path = tmp_path / "report.html"
    to_html(report, str(file_path))
    assert file_path.exists()
