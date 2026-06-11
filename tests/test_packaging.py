# img2svg - tests for PyPI packaging metadata in pyproject.toml.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for PyPI packaging metadata in pyproject.toml.

The `pyproject.toml` is the single source of truth for the metadata
PyPI displays (description, classifiers, URLs, console script) and for
the wheel build configuration (force-include of man pages and locale
files). These tests guard against accidental structural drift: a future
edit that drops the console script, removes the BSD classifier, or
forgets to ship the man page in the wheel should fail loudly here
before it ships.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

REQUIRED_CLASSIFIERS: tuple[str, ...] = (
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: BSD :: FreeBSD",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
)


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    """Parse pyproject.toml once per module."""
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


# ----------------------------------------------------------------------
# Test 1: description field present
# ----------------------------------------------------------------------


def test_pyproject_has_description(pyproject: dict[str, Any]) -> None:
    """`[project].description` exists and is a non-empty string."""
    project = pyproject["project"]
    assert "description" in project, "[project] must include a 'description' field"
    assert isinstance(project["description"], str), "'description' must be a string"
    assert project["description"].strip(), "'description' must be non-empty"


# ----------------------------------------------------------------------
# Test 2: readme field points at README.md
# ----------------------------------------------------------------------


def test_pyproject_readme_is_readme_md(pyproject: dict[str, Any]) -> None:
    """`[project].readme` is exactly the string 'README.md'."""
    project = pyproject["project"]
    assert "readme" in project, "[project] must include a 'readme' field"
    assert project["readme"] == "README.md", (
        f"[project].readme must be 'README.md', got {project['readme']!r}"
    )


# ----------------------------------------------------------------------
# Test 3: BSD-3-Clause license
# ----------------------------------------------------------------------


def test_pyproject_has_bsd_3_clause_license(pyproject: dict[str, Any]) -> None:
    """`[project].license` declares BSD-3-Clause (PEP 639 inline form supported)."""
    project = pyproject["project"]
    assert "license" in project, "[project] must include a 'license' field"

    license_value = project["license"]
    if isinstance(license_value, dict):
        text = license_value.get("text", "")
        assert "BSD-3-Clause" in text, (
            f"[project].license.text must be 'BSD-3-Clause', got {text!r}"
        )
    else:
        assert "BSD-3-Clause" in str(license_value), (
            f"[project].license must contain 'BSD-3-Clause', got {license_value!r}"
        )


# ----------------------------------------------------------------------
# Test 4: all 11 required classifiers
# ----------------------------------------------------------------------


def test_pyproject_has_all_required_classifiers(pyproject: dict[str, Any]) -> None:
    """`[project].classifiers` includes every one of the 11 required entries."""
    project = pyproject["project"]
    assert "classifiers" in project, "[project] must include a 'classifiers' list"
    classifiers = list(project["classifiers"])

    missing = [c for c in REQUIRED_CLASSIFIERS if c not in classifiers]
    assert not missing, (
        f"pyproject.toml is missing required classifiers: {missing}; found: {classifiers}"
    )


# ----------------------------------------------------------------------
# Test 5: 4 project URLs
# ----------------------------------------------------------------------


def test_pyproject_has_four_project_urls(pyproject: dict[str, Any]) -> None:
    """`[project.urls]` has Homepage, Repository, Issues, Documentation."""
    project = pyproject["project"]
    assert "urls" in project, "[project] must include a 'urls' table"
    urls = project["urls"]

    expected_keys = {"Homepage", "Repository", "Issues", "Documentation"}
    actual_keys = set(urls.keys())
    assert expected_keys.issubset(actual_keys), (
        f"[project.urls] is missing keys: {expected_keys - actual_keys}; "
        f"found: {sorted(actual_keys)}"
    )
    assert len(urls) >= 4, (
        f"[project.urls] must have at least 4 entries, found {len(urls)}: {sorted(urls)}"
    )

    for key, value in urls.items():
        assert isinstance(value, str), f"url {key!r} must be a string"
        assert value.startswith(("http://", "https://")), (
            f"url {key!r} must start with http:// or https://, got {value!r}"
        )


# ----------------------------------------------------------------------
# Test 6: [project.scripts] img2svg = "img2svg.cli:app"
# ----------------------------------------------------------------------


def test_pyproject_registers_img2svg_console_script(
    pyproject: dict[str, Any],
) -> None:
    """`[project.scripts].img2svg` is the `img2svg.cli:app` entry point."""
    project = pyproject["project"]
    assert "scripts" in project, "[project] must include a 'scripts' table"
    scripts = project["scripts"]

    assert "img2svg" in scripts, (
        f"[project.scripts] must register the 'img2svg' console script; found: {sorted(scripts)}"
    )
    assert scripts["img2svg"] == "img2svg.cli:app", (
        f"[project.scripts].img2svg must be 'img2svg.cli:app', got {scripts['img2svg']!r}"
    )

    entry = scripts["img2svg"]
    match = re.fullmatch(r"([\w.]+):(\w+)", entry)
    assert match is not None, (
        f"console script entry {entry!r} must be of the form 'module.path:attr'"
    )


# ----------------------------------------------------------------------
# Test 7: man/img2svg.1 is included in the wheel build
# ----------------------------------------------------------------------


def test_pyproject_includes_man_page_in_wheel(pyproject: dict[str, Any]) -> None:
    """The wheel build target force-includes the man page source path."""
    hatch = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    force_include = hatch.get("force-include", {})

    source_paths = list(force_include.keys())
    man_page_sources = [s for s in source_paths if "img2svg.1" in s]
    assert man_page_sources, (
        f"wheel force-include must include the man page source path; "
        f"force-include keys: {source_paths}"
    )

    man_page_dest = force_include[man_page_sources[0]]
    assert man_page_dest.startswith("img2svg/man/") or man_page_dest == "img2svg/man", (
        f"man page destination must be under img2svg/man/, got {man_page_dest!r}"
    )
