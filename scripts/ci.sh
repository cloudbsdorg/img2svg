#!/usr/bin/env bash
# ci.sh - local CI script that mirrors the Jenkinsfile pipeline.
#
# Usage:  ./scripts/ci.sh [pytest-flags...]
#
# Runs the same checks Jenkins runs in a clean shell. Useful for
# reproducing CI failures locally and as a pre-push sanity check.
# Any extra arguments are forwarded to pytest (e.g. -k, -m, --lf).
set -euo pipefail

# Anchor to project root so the script can be invoked from anywhere.
cd "$(dirname "$0")/.."

echo "==> Environment"
echo "OS:     $(uname -s)"
echo "Python: $(python3 --version)"
echo "uv:     $(uv --version)"

echo "==> Setup"
uv sync --all-extras

echo "==> Lint"
uv run ruff check
uv run ruff format --check
uv run mypy src/

echo "==> Test"
uv run pytest \
    --cov=img2svg \
    --cov-fail-under=80 \
    --junitxml=build/junit.xml \
    --json-report \
    --json-report-file=build/report.json \
    "$@"

echo "==> Build Docs"
uv run mkdocs build --strict

# Package step is intentionally omitted locally. The Jenkinsfile
# runs `uv build` only on the main branch; invoke it by hand if you
# want a local wheel.

echo "CI pipeline complete."
