"""Tests for the img2svg README.

The README is the project's landing page on PyPI, GitHub, and the
mkdocs site. These tests guard against accidental structural drift:
a future refactor that drops the "Installation" section, removes the
author email, or forgets the Mermaid diagram should fail loudly here
before it ships.

Coverage goals:

    1. README.md exists at the project root
    2. All required top-level sections are present
    3. The Mermaid architecture diagram is included
    4. Author and license metadata are intact
    5. Output modes section documents all four modes

The README is rendered on GitHub and in mkdocs, so the tests do
structural checks (substring, regex on headings) rather than full
Markdown parsing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Resolve paths once at import time. Tests don't need to be hermetic
# about the project root — they assume pytest is run from the repo root
# (the project uses `pythonpath = ["src"]` and `testpaths = ["tests"]`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Read the README source once per module.

    Most tests only need the text; module-scope keeps disk I/O down
    when the full suite runs.
    """
    return README_PATH.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Test 1: file exists
# ----------------------------------------------------------------------


def test_readme_file_exists() -> None:
    """`README.md` exists at the project root."""
    assert README_PATH.is_file(), f"README missing: {README_PATH}"


# ----------------------------------------------------------------------
# Test 2: required ## sections
# ----------------------------------------------------------------------

# Section names that MUST appear as level-2 Markdown headings (`## Foo`).
# A heading line in the README looks like `## Title` or `## Title — note`.
# We match the first `## Word(s)` on a line so descriptions and trailing
# dashes don't trip the check.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Installation",
    "Usage",
    "License",
    "Author",
)


def _h2_headings(text: str) -> set[str]:
    """Return the set of level-2 Markdown headings in `text`.

    A heading line starts with `##` followed by whitespace and a name.
    We strip any trailing `:`, `-`, or whitespace so `## Usage: CLI` and
    `## Usage` both register as `"Usage"`.
    """
    found: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            name = m.group(1).strip()
            # Drop trailing punctuation that some headings use as
            # decoration (`## Overview` is fine, `## Overview:` is too).
            name = name.rstrip(":")
            found.add(name)
    return found


def test_readme_has_installation_section(readme_text: str) -> None:
    """The README contains a `## Installation` section."""
    headings = _h2_headings(readme_text)
    assert "Installation" in headings, (
        f"README must include a `## Installation` section; "
        f"found headings: {sorted(headings)}"
    )


def test_readme_has_usage_section(readme_text: str) -> None:
    """The README contains at least one `## Usage` section.

    The current README has two: `## Usage: CLI` and `## Usage: Python API`.
    We match on the leading word so both register.
    """
    headings = _h2_headings(readme_text)
    usage_sections = [h for h in headings if h.startswith("Usage")]
    assert usage_sections, (
        f"README must include a `## Usage` section; "
        f"found headings: {sorted(headings)}"
    )


def test_readme_has_license_section(readme_text: str) -> None:
    """The README contains a `## License` section."""
    headings = _h2_headings(readme_text)
    assert "License" in headings, (
        f"README must include a `## License` section; "
        f"found headings: {sorted(headings)}"
    )


def test_readme_has_author_section(readme_text: str) -> None:
    """The README contains a `## Author` section."""
    headings = _h2_headings(readme_text)
    assert "Author" in headings, (
        f"README must include a `## Author` section; "
        f"found headings: {sorted(headings)}"
    )


# ----------------------------------------------------------------------
# Test 6: Mermaid code block
# ----------------------------------------------------------------------


def test_readme_has_mermaid_diagram(readme_text: str) -> None:
    """The README contains a fenced Mermaid code block.

    The block opens with ```` ```mermaid ```` and closes with a matching
    fence. We look for the opening fence only — that's enough to know
    the diagram is present and will be rendered by mkdocs/GitHub.
    """
    assert "```mermaid" in readme_text, (
        "README must include a fenced ```mermaid code block for the architecture diagram"
    )


# ----------------------------------------------------------------------
# Test 7: author email
# ----------------------------------------------------------------------


def test_readme_has_author_email(readme_text: str) -> None:
    """The author email `mark@cloudbsd.org` is present in the README."""
    assert "mark@cloudbsd.org" in readme_text, (
        "README must list author email: mark@cloudbsd.org"
    )


# ----------------------------------------------------------------------
# Test 8: BSD license mention
# ----------------------------------------------------------------------


def test_readme_has_bsd_license_mention(readme_text: str) -> None:
    """The README mentions the BSD license and includes the SPDX identifier.

    The SPDX form (`BSD-3-Clause`) is what PyPI expects when the
    `License-Expression` metadata is set, so we check for it explicitly
    in addition to the human-friendly "BSD-3-Clause License" wording.
    """
    assert "BSD" in readme_text, "README must mention the BSD license"
    assert "BSD-3-Clause" in readme_text, (
        "README must include the SPDX license identifier 'BSD-3-Clause'"
    )
