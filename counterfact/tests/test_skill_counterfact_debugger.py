"""End-to-end test for the counterfact-debugger Agent Skill.

Exercises the skill's bundled scripts exactly as the skill drives them:

  1. cf_diagnose.py on a buggy pipeline      -> detects a failure, low quality
  2. cf_diagnose.py on the fixed pipeline    -> no failure, high quality
  3. verify.py comparing the two reports     -> reports IMPROVED (exit 0)

Runs the scripts as subprocesses (the real entry points), using the deterministic
no-LLM fixture so no API key or network is needed. This guards the skill plumbing —
factory loading, registry wiring, report I/O, and the verify gate — against drift.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "skills" / "counterfact-debugger" / "scripts"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not _SCRIPTS.exists(), reason="counterfact-debugger skill scripts not present"
)


def _run_diagnose(factory, out_path, tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"metric_key": "revenue", "query": "What was revenue?"}]))
    env = dict(os.environ, PYTHONPATH=str(_FIXTURES))
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPTS / "cf_diagnose.py"),
            "--factory", factory,
            "--inputs", str(cases),
            "--registry", "buggy_pipeline:build_registry",
            "--domain", "finance",
            "--num-simulations", "24",
            "--seed", "42",
            "--no-llm",
            "--out", str(out_path),
        ],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"cf_diagnose failed:\n{proc.stderr}"
    return json.loads(out_path.read_text())


def test_diagnose_detects_failure_and_verify_confirms_fix(tmp_path):
    # 1. Buggy pipeline: quality should be low and a failure detected.
    before_path = tmp_path / "report.json"
    before = _run_diagnose("buggy_pipeline:build", before_path, tmp_path)
    assert before["baseline_quality"] < 0.5
    assert before["classification"]["failure_type"] != "no_failure"
    assert before["shapley_values"], "expected per-agent attribution"
    # The companion markdown report is written alongside the JSON.
    assert before_path.with_suffix(".md").exists()

    # 2. Fixed pipeline: quality should recover.
    after_path = tmp_path / "report_after.json"
    after = _run_diagnose("buggy_pipeline:build_fixed", after_path, tmp_path)
    assert after["baseline_quality"] > before["baseline_quality"]

    # 3. verify.py should report the fix as an improvement (exit 0).
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPTS / "verify.py"),
            "--baseline", str(before_path),
            "--candidate", str(after_path),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"verify did not confirm improvement:\n{proc.stdout}\n{proc.stderr}"
    assert "IMPROVED" in proc.stdout


def test_diagnose_rejects_non_counterfact_factory(tmp_path):
    """A factory that doesn't return a counterfact graph fails fast with a clear error."""
    bad = tmp_path / "bad_factory.py"
    bad.write_text("def build():\n    return object()\n")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"query": "x"}))
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPTS / "cf_diagnose.py"),
            "--factory", "bad_factory:build",
            "--inputs", str(cases),
            "--no-llm",
            "--out", str(tmp_path / "r.json"),
        ],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0
    assert "diagnose" in proc.stderr.lower() or "recipe" in proc.stderr.lower()
