#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────────
# Counterfact CI checks — runs the same checks as GitHub Actions.
#
# Usage:
#   bash run_checks.sh
#
# This script auto-detects dev tools from common venv locations.
# To install dev deps in your own venv:
#   pip install -e ".[all,dev]"
# ──────────────────────────────────────────────────────────────────

# Auto-detect a venv with the dev tools; fall back to whatever is on PATH.
VENV_BIN=""
for candidate in ".venv/bin" "venv/bin"; do
    if [ -x "$candidate/ruff" ] && [ -x "$candidate/mypy" ] && [ -x "$candidate/pytest" ]; then
        VENV_BIN="$candidate"
        break
    fi
done

if [ -n "$VENV_BIN" ]; then
    RUFF="$VENV_BIN/ruff"; MYPY="$VENV_BIN/mypy"; PYTEST="$VENV_BIN/pytest"
    echo "Using tools from: $VENV_BIN"
elif command -v ruff >/dev/null && command -v mypy >/dev/null && command -v pytest >/dev/null; then
    RUFF="ruff"; MYPY="mypy"; PYTEST="pytest"
    echo "Using tools from: PATH"
else
    echo "ERROR: Could not find ruff, mypy, and pytest (in .venv/, venv/, or on PATH)."
    echo "Install dev deps: pip install -e \".[all,dev]\""
    exit 1
fi

echo "======================================"
echo "Running Ruff Linter..."
echo "======================================"
"$RUFF" check counterfact/

echo "======================================"
echo "Running Mypy Type Checker..."
echo "======================================"
"$MYPY" counterfact/

echo "======================================"
echo "Running Pytest & Coverage..."
echo "======================================"
"$PYTEST" --cov=counterfact --cov-fail-under=50

echo "======================================"
echo "✅ ALL CHECKS PASSED SUCCESSFULLY!"
echo "======================================"
