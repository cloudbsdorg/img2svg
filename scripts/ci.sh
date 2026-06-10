#!/usr/bin/env bash
# ci.sh - local CI script that mirrors the Jenkinsfile pipeline.
#
# Usage:  ./scripts/ci.sh [--slow] [pytest-flags...]
#
# Runs the same checks Jenkins runs in a clean shell. Useful for
# reproducing CI failures locally and as a pre-push sanity check.
# Pass --slow to additionally run the slow integration tests (real
# YOLO + real vtracer). Any extra arguments are forwarded to pytest
# (e.g. -k, --lf).
set -euo pipefail

# Anchor to project root so the script can be invoked from anywhere.
cd "$(dirname "$0")/.."

# Parse flags. The only flag we recognize is --slow; everything else
# is forwarded to pytest verbatim.
RUN_SLOW=0
PYTEST_EXTRA=()
for arg in "$@"; do
    case "$arg" in
        --slow)
            RUN_SLOW=1
            ;;
        *)
            PYTEST_EXTRA+=("$arg")
            ;;
    esac
done

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
# Override pyproject's default `-m not slow` when --slow is passed, so
# the full suite (including slow integration tests) runs. When --slow
# is not passed, the addopts already filter to fast tests.
if [ "$RUN_SLOW" -eq 1 ]; then
    uv run pytest \
        --cov=img2svg \
        --cov-fail-under=80 \
        -m slow \
        --junitxml=build/junit.xml \
        --json-report \
        --json-report-file=build/report.json \
        "${PYTEST_EXTRA[@]}"
else
    uv run pytest \
        --cov=img2svg \
        --cov-fail-under=80 \
        --junitxml=build/junit.xml \
        --json-report \
        --json-report-file=build/report.json \
        "${PYTEST_EXTRA[@]}"
fi

echo "==> Build Docs"
uv run mkdocs build --strict

# Package step is intentionally omitted locally. The Jenkinsfile
# runs `uv build` only on the main branch; invoke it by hand if you
# want a local wheel.

echo "CI pipeline complete."
