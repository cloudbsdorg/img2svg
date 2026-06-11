# img2svg - pytest configuration and shared fixtures.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Pytest configuration and shared fixtures for img2svg tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Make sure fixtures exist. If not, run the generator.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_GENERATOR = Path(__file__).parent.parent / "scripts" / "gen_test_images.py"


def _ensure_fixtures() -> None:
    if not _FIXTURES_DIR.exists() or not list(_FIXTURES_DIR.glob("*.png")):
        subprocess.check_call([sys.executable, str(_GENERATOR)])


_ensure_fixtures()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return _FIXTURES_DIR


@pytest.fixture
def logo_path() -> Path:
    return _FIXTURES_DIR / "logo.png"


@pytest.fixture
def photo_path() -> Path:
    return _FIXTURES_DIR / "photo.jpg"


@pytest.fixture
def diagram_path() -> Path:
    return _FIXTURES_DIR / "diagram.png"


@pytest.fixture
def line_art_path() -> Path:
    return _FIXTURES_DIR / "line_art.png"


@pytest.fixture
def transparent_path() -> Path:
    return _FIXTURES_DIR / "transparent.png"


@pytest.fixture
def screenshot_path() -> Path:
    return _FIXTURES_DIR / "screenshot.png"


@pytest.fixture
def corrupt_path() -> Path:
    return _FIXTURES_DIR / "corrupt.bin"
