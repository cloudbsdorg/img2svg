# img2svg - enumerations for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Enumerations for img2svg.

Uses a cross-compatible StrEnum pattern (the stdlib `StrEnum` only landed in
Python 3.11; this module supports the project's `requires-python = ">=3.10"`).
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[misc,no-redef]
        """StrEnum shim for Python <3.11."""


class Mode(StrEnum):
    """Output rendering mode."""

    AUTO = "auto"
    LABELS = "labels"
    VISUAL = "visual"
    ANNOTATED = "annotated"
    TRACE = "trace"


class ImageType(StrEnum):
    """Heuristic image type classification."""

    LOGO = "logo"
    PHOTO = "photo"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    LINE_ART = "line_art"
    UNKNOWN = "unknown"


class DeviceStrategy(StrEnum):
    """Strategy for picking the best GPU when multiple are available."""

    AUTO = "auto"
    POWER = "power"
    AVAILABILITY = "availability"


class GpuVendor(StrEnum):
    """GPU hardware vendor."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    CPU = "cpu"
    UNKNOWN = "unknown"
