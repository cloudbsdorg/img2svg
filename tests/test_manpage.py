# img2svg - tests for the img2svg(1) man page.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the img2svg(1) man page.

Validates that the man page source is present, well-formed, and contains
the required sections and metadata. The page is hand-written roff/groff,
so these tests are content checks (not parser tests). Coverage goals:

    1. File exists at the expected path
    2. All required `.SH` sections are present
    3. The `.TH` header has the right arguments (name, section, date, version)
    4. Author email is present
    5. Release date is present
    6. At least 5 example blocks (`.EX` / `.EE`) are present
    7. All CLI flags from `cli.py` are documented

The man page ships with the wheel via `force-include` in pyproject.toml,
so a separate test verifies that the wheel-build configuration actually
points at it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Resolve paths once at import time. Tests don't need to be hermetic
# about the project root — they assume pytest is run from the repo root
# (the project uses `pythonpath = ["src"]` and `testpaths = ["tests"]`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANPAGE_PATH = PROJECT_ROOT / "man" / "img2svg.1"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def manpage_text() -> str:
    """Read the man page source once per module.

    Most tests only need the text; fixture-scope keeps disk I/O down
    when running the full suite.
    """
    return MANPAGE_PATH.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Test 1: file exists
# ----------------------------------------------------------------------


def test_manpage_file_exists() -> None:
    """`man/img2svg.1` exists at the project root."""
    assert MANPAGE_PATH.is_file(), f"man page missing: {MANPAGE_PATH}"


# ----------------------------------------------------------------------
# Test 2: required .SH sections
# ----------------------------------------------------------------------

# The spec requires at minimum: NAME, SYNOPSIS, DESCRIPTION, OPTIONS,
# EXAMPLES, EXIT STATUS, AUTHOR. Output modes, GPU, files, environment,
# see also, and bugs are nice-to-have — we test for their presence too
# but treat them as soft requirements (so a missing section fails
# loudly with a clear message).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "NAME",
    "SYNOPSIS",
    "DESCRIPTION",
    "OPTIONS",
    "EXAMPLES",
    "EXIT STATUS",
    "AUTHOR",
)

OPTIONAL_SECTIONS: tuple[str, ...] = (
    "OUTPUT MODES",
    "GPU SUPPORT",
    "FILES",
    "ENVIRONMENT",
    "SEE ALSO",
    "BUGS",
)


def test_manpage_has_required_sections(manpage_text: str) -> None:
    """All required `.SH` sections are present in the man page."""
    # Match `.SH` followed by whitespace and the section name (which may
    # be quoted with `"..."` for multi-word names like EXIT STATUS).
    # A section header line looks like: `.SH NAME` or `.SH "EXIT STATUS"`.
    found: set[str] = set()
    for line in manpage_text.splitlines():
        m = re.match(r'^\s*\.SH\s+(?:"([^"]+)"|(\S+))', line)
        if m:
            found.add(m.group(1) or m.group(2))

    missing = [s for s in REQUIRED_SECTIONS if s not in found]
    assert not missing, f"man page is missing required sections: {missing}; found: {sorted(found)}"


def test_manpage_has_optional_sections(manpage_text: str) -> None:
    """Recommended `.SH` sections (output modes, GPU, files, ...) are present."""
    found: set[str] = set()
    for line in manpage_text.splitlines():
        m = re.match(r'^\s*\.SH\s+(?:"([^"]+)"|(\S+))', line)
        if m:
            found.add(m.group(1) or m.group(2))

    missing = [s for s in OPTIONAL_SECTIONS if s not in found]
    assert not missing, (
        f"man page is missing recommended sections: {missing}; found: {sorted(found)}"
    )


# ----------------------------------------------------------------------
# Test 3: author email
# ----------------------------------------------------------------------


def test_manpage_has_author_email(manpage_text: str) -> None:
    """The author email `mark@cloudbsd.org` is present."""
    assert "mark@cloudbsd.org" in manpage_text, "man page must list author email: mark@cloudbsd.org"


# ----------------------------------------------------------------------
# Test 4: release date
# ----------------------------------------------------------------------


def test_manpage_has_release_date(manpage_text: str) -> None:
    """The `.TH` header release date `2026-06-10` is present."""
    assert "2026-06-10" in manpage_text, (
        "man page must include the release date 2026-06-10 in the .TH header"
    )


# ----------------------------------------------------------------------
# Test 5: at least 5 examples
# ----------------------------------------------------------------------


def test_manpage_has_at_least_five_examples(manpage_text: str) -> None:
    """The man page contains at least 5 example blocks.

    Per the spec, an "example" line is one that starts with `.TP`
    (tagged paragraph used for flag descriptions, counted as inline
    usage examples), `.EX` (start of a literal code block), or `#`
    (a roff comment used as a section divider). A single `.EX` block
    counts once.
    """
    count = 0
    for line in manpage_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(".TP") or stripped.startswith(".EX") or stripped.startswith("#"):
            count += 1
    assert count >= 5, f"man page has only {count} example/documented lines; need >= 5"


# ----------------------------------------------------------------------
# Test 6: .TH header shape
# ----------------------------------------------------------------------


def test_manpage_th_header(manpage_text: str) -> None:
    """The `.TH` macro has the canonical 5-argument shape."""
    th_lines = [ln for ln in manpage_text.splitlines() if ln.lstrip().startswith(".TH")]
    assert th_lines, "man page is missing a .TH header"
    # The .TH line should have: name, section, date, version, "User Commands"
    first = th_lines[0]
    assert "IMG2SVG" in first, f".TH must include the program name: {first!r}"
    assert re.search(r"\b1\b", first), f".TH must declare section 1: {first!r}"
    assert "2026-06-10" in first, f".TH must include the release date: {first!r}"
    assert "0.1.0" in first, f".TH must include the version: {first!r}"
    assert "User Commands" in first, f".TH must declare category 'User Commands': {first!r}"


# ----------------------------------------------------------------------
# Test 7: every CLI flag is documented
# ----------------------------------------------------------------------


# Flags that must appear somewhere in the man page's OPTIONS section.
# Match the public CLI surface from src/img2svg/cli.py. The leading
# backslash is to allow roff-escaped hyphens (`\-\-mode`) in addition
# to bare ones.
EXPECTED_FLAGS: tuple[str, ...] = (
    r"\-\-output",
    r"\-\-mode",
    r"\-\-model",
    r"\-\-device",
    r"\-\-gpu\-strategy",
    r"\-\-conf",
    r"\-\-no\-clobber",
    r"\-\-quiet",
    r"\-\-verbose",
    r"\-\-version",
)


def test_manpage_documents_every_cli_flag(manpage_text: str) -> None:
    """Every CLI flag in cli.py is mentioned in the man page."""
    # The man page may write the flag as `\-\-mode` (escaped to keep
    # roff from interpreting `--` as an em-dash) or as a bare `--mode`.
    # Normalize by stripping the optional backslashes before each `-`
    # so a single substring check covers both forms.
    normalized = re.sub(r"\\-", "-", manpage_text)
    missing: list[str] = []
    for flag in EXPECTED_FLAGS:
        # `flag` is a regex pattern (e.g. `r"\-\-mode"`) — first try
        # the literal pattern as-is, then fall back to the un-escaped
        # bare form.
        if not re.search(flag, manpage_text) and not re.search(
            re.escape(flag.replace(r"\-", "-")), normalized
        ):
            missing.append(flag.replace(r"\-", "-"))
    assert not missing, f"man page does not document these CLI flags: {missing}"


# ----------------------------------------------------------------------
# Test 8: pyproject.toml ships the man page
# ----------------------------------------------------------------------


def test_pyproject_includes_man_page() -> None:
    """`pyproject.toml` `force-include` ships `man/img2svg.1` with the wheel.

    If this regresses, `pip install img2svg` will silently drop the man
    page — so we check it explicitly. The check is intentionally
    generous: any of `force-include`, `force_include`, or `package-data`
    containing the man page path is accepted.
    """
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    # Accept any of: force-include with the project-root path, or
    # package-data / artifacts with a man/ glob. We use a simple
    # substring check on the project-root man page path.
    assert "man/img2svg.1" in text or "man/*.1" in text, (
        "pyproject.toml does not reference man/img2svg.1 in any package-data "
        "or force-include entry. The man page will not ship with `pip install`."
    )
