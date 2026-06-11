# img2svg - XDG Base Directory compliant paths for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""XDG Base Directory compliant paths for img2svg.

Provides configuration, data, cache, and system-config directories following
the FreeDesktop XDG Base Directory Specification, with a CloudBSD-specific
system-wide fallback for FreeBSD installations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SYSTEM_CONFIG_DIR = Path("/usr/local/etc/cloudbsd/img2svg")


def config_dir() -> Path:
    """User config directory: $XDG_CONFIG_HOME/img2svg/ (default ~/.config/img2svg/)."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or str(Path.home() / ".config")
    p = Path(base) / "img2svg"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    """User data directory: $XDG_DATA_HOME/img2svg/ (default ~/.local/share/img2svg/)."""
    base = os.environ.get("XDG_DATA_HOME", "").strip() or str(Path.home() / ".local" / "share")
    p = Path(base) / "img2svg"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    """User cache directory: $XDG_CACHE_HOME/img2svg/ (default ~/.cache/img2svg/)."""
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
    p = Path(base) / "img2svg"
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_cache_path(model_name: str = "yolo11x.pt") -> Path:
    """Path to a cached YOLO model inside the XDG cache directory."""
    p = cache_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p / model_name


def system_config_dir() -> Path:
    """System-wide config directory for FreeBSD/CloudBSD installations."""
    return SYSTEM_CONFIG_DIR


def load_config() -> dict:
    """Load img2svg config from <config_dir>/config.toml. Returns empty dict if missing."""
    path = config_dir() / "config.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return {}
