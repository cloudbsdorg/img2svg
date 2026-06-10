"""Structural tests for the Jenkinsfile and the local CI script."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JENKINSFILE = PROJECT_ROOT / "Jenkinsfile"
CI_SCRIPT = PROJECT_ROOT / "scripts" / "ci.sh"
GITHUB_DIR = PROJECT_ROOT / ".github"
GITHUB_WORKFLOWS = GITHUB_DIR / "workflows"

REQUIRED_STAGES: tuple[str, ...] = (
    "Setup",
    "Lint",
    "Test",
    "Build Docs",
    "Package",
)

REQUIRED_JENKINS_COMMANDS: tuple[str, ...] = (
    "uv sync",
    "uv run pytest",
    "uv run mkdocs build",
)


def test_jenkinsfile_exists() -> None:
    assert JENKINSFILE.is_file(), f"missing {JENKINSFILE}"


def test_jenkinsfile_has_required_stages() -> None:
    text = JENKINSFILE.read_text(encoding="utf-8")
    for stage in REQUIRED_STAGES:
        single = f"stage('{stage}')"
        double = f'stage("{stage}")'
        assert single in text or double in text, (
            f"Jenkinsfile is missing required stage {single!r}"
        )


def test_jenkinsfile_contains_required_commands() -> None:
    text = JENKINSFILE.read_text(encoding="utf-8")
    for needle in REQUIRED_JENKINS_COMMANDS:
        assert needle in text, f"Jenkinsfile is missing required command: {needle!r}"


def test_ci_script_exists_and_is_executable() -> None:
    assert CI_SCRIPT.is_file(), f"missing {CI_SCRIPT}"
    mode = CI_SCRIPT.stat().st_mode
    assert mode & 0o111, (
        f"{CI_SCRIPT} must be executable (got mode {oct(mode & 0o777)}, "
        f"need at least one x bit set)"
    )


def test_ci_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(CI_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"bash -n {CI_SCRIPT} failed (rc={result.returncode}): "
        f"{result.stderr.strip()}"
    )


def test_ci_script_uses_strict_mode() -> None:
    text = CI_SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text, (
        "ci.sh must use 'set -euo pipefail' for fail-fast / undefined-var / "
        "pipe-failure safety"
    )


def test_ci_script_syncs_all_extras() -> None:
    text = CI_SCRIPT.read_text(encoding="utf-8")
    assert "uv sync --all-extras" in text, (
        "ci.sh must run 'uv sync --all-extras' to install dev dependencies"
    )


def test_ci_script_runs_pytest() -> None:
    text = CI_SCRIPT.read_text(encoding="utf-8")
    assert "pytest" in text, "ci.sh must run pytest"


def test_no_github_workflows_directory() -> None:
    assert not GITHUB_WORKFLOWS.is_dir(), (
        f"Found {GITHUB_WORKFLOWS} — this project uses Jenkins, not GitHub Actions"
    )
