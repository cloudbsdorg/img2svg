"""Tests for the example gallery (T30).

These tests pin the shape of the user-facing documentation: each example
file must exist, the markdown code blocks must be in the expected
language, and every committed sample SVG must parse as valid XML.

Six assertions, all structural — no model inference or YOLO calls.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from lxml import etree

# Resolve paths once at import time. The gallery is part of the repo
# and lives at <repo>/examples/, so the layout is fixed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
BASIC_USAGE_MD = EXAMPLES_DIR / "basic_usage.md"
PYTHON_API_MD = EXAMPLES_DIR / "python_api.md"
BEFORE_AFTER_MD = EXAMPLES_DIR / "before_after.md"
SAMPLE_OUTPUTS_DIR = EXAMPLES_DIR / "sample_outputs"
SAMPLE_INPUTS_README = EXAMPLES_DIR / "sample_inputs" / "README.md"


def _count_fenced_blocks(text: str, info_string: str) -> int:
    """Count ```<info_string> fenced code blocks; info string must be the
    first token on the opening fence (so `` ```bash`` matches but
    `` ```bashrc`` does not).
    """
    pattern = re.compile(
        r"^```" + re.escape(info_string) + r"\b",
        flags=re.MULTILINE,
    )
    return len(pattern.findall(text))


def test_basic_usage_md_exists_with_bash_blocks() -> None:
    """basic_usage.md must exist and have 5+ ```bash code blocks."""
    assert BASIC_USAGE_MD.is_file(), f"missing {BASIC_USAGE_MD}"
    text = BASIC_USAGE_MD.read_text(encoding="utf-8")
    count = _count_fenced_blocks(text, "bash")
    assert count >= 5, f"expected >=5 bash blocks, found {count}"


def test_python_api_md_exists_with_python_blocks() -> None:
    """python_api.md must exist and have 3+ ```python code blocks."""
    assert PYTHON_API_MD.is_file(), f"missing {PYTHON_API_MD}"
    text = PYTHON_API_MD.read_text(encoding="utf-8")
    count = _count_fenced_blocks(text, "python")
    assert count >= 3, f"expected >=3 python blocks, found {count}"


def test_before_after_md_exists() -> None:
    """before_after.md must exist (visual comparison document)."""
    assert BEFORE_AFTER_MD.is_file(), f"missing {BEFORE_AFTER_MD}"
    text = BEFORE_AFTER_MD.read_text(encoding="utf-8")
    assert "sample_outputs/" in text, "before_after.md should reference sample_outputs/"
    assert "tests/fixtures/" in text, "before_after.md should reference tests/fixtures/"


def test_sample_outputs_contains_at_least_eight_svgs() -> None:
    """sample_outputs/ must contain at least 8 .svg files."""
    assert SAMPLE_OUTPUTS_DIR.is_dir(), f"missing dir {SAMPLE_OUTPUTS_DIR}"
    svgs = sorted(SAMPLE_OUTPUTS_DIR.glob("*.svg"))
    assert len(svgs) >= 8, f"expected >=8 SVGs, found {len(svgs)}: {[p.name for p in svgs]}"


def test_all_sample_svgs_parse_as_valid_xml() -> None:
    """Every .svg in sample_outputs/ must parse with lxml (valid XML)."""
    assert SAMPLE_OUTPUTS_DIR.is_dir(), f"missing dir {SAMPLE_OUTPUTS_DIR}"
    svgs = sorted(SAMPLE_OUTPUTS_DIR.glob("*.svg"))
    assert svgs, "no SVGs to validate"
    for svg in svgs:
        try:
            etree.parse(str(svg))
        except etree.XMLSyntaxError as exc:  # pragma: no cover - test fails first
            pytest.fail(f"{svg.name} is not valid XML: {exc}")


def test_sample_inputs_readme_exists() -> None:
    """examples/sample_inputs/README.md must exist."""
    assert SAMPLE_INPUTS_README.is_file(), f"missing {SAMPLE_INPUTS_README}"
    text = SAMPLE_INPUTS_README.read_text(encoding="utf-8")
    assert "logo.png" in text, "sample_inputs/README.md should mention logo.png"
    assert "diagram.png" in text, "sample_inputs/README.md should mention diagram.png"
