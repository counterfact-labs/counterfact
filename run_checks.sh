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

# Auto-detect venv with dev tools
VENV_BIN=""
for candidate in \
    ".venv/bin" \
    "../counterfactual-debugger/.venv/bin" \
    "venv/bin"; do
    if [ -x "$candidate/ruff" ] && [ -x "$candidate/mypy" ] && [ -x "$candidate/pytest" ]; then
        VENV_BIN="$candidate"
        break
    fi
done

if [ -z "$VENV_BIN" ]; then
    echo "ERROR: Could not find ruff, mypy, pytest in any known venv."
    echo "Install dev deps: pip install -e \".[all,dev]\""
    exit 1
fi

echo "======================================"
echo "Using tools from: $VENV_BIN"
echo "======================================"

echo "======================================"
echo "Running Ruff Linter..."
echo "======================================"
"$VENV_BIN/ruff" check counterfact/

echo "======================================"
echo "Running Mypy Type Checker..."
echo "======================================"
"$VENV_BIN/mypy" counterfact/

echo "======================================"
echo "Running Pytest & Coverage..."
echo "======================================"
"$VENV_BIN/pytest" --cov=counterfact --cov-fail-under=50

echo "======================================"
echo "✅ ALL CHECKS PASSED SUCCESSFULLY!"
echo "======================================"
