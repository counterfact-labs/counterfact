#!/bin/bash
set -e

# Activate virtual environment if necessary
# . ../counterfactual-debugger/.venv/bin/activate

echo "======================================"
echo "Running Ruff Linter..."
echo "======================================"
ruff check .

echo "======================================"
echo "Running Mypy Type Checker..."
echo "======================================"
mypy .

echo "======================================"
echo "Running Pytest & Coverage..."
echo "======================================"
pytest --cov=counterfact --cov-fail-under=50

echo "======================================"
echo "✅ ALL CHECKS PASSED SUCCESSFULLY!"
echo "======================================"
