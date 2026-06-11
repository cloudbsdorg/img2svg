# img2svg - auto-mode selection: map image type to renderer mode, and mode to vtracer preset.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Auto-mode selection: map image type to renderer mode, and mode to vtracer preset."""

from __future__ import annotations

from img2svg.enums import ImageType, Mode
from img2svg.vectorizer import Preset

# Which Mode to use for each ImageType (used when user passes --mode auto).
# Per product constraint, AUTO must never resolve to LABELS or ANNOTATED —
# those are explicit-only modes. SEGMENTED is also explicit-only (it has no
# sensible default and is a power-user mode).
IMAGE_TYPE_TO_MODE: dict[ImageType, Mode] = {
    ImageType.LOGO:       Mode.VISUAL,
    ImageType.PHOTO:      Mode.DETAILED,  # Aggressive: pre-process + high-fidelity trace
    ImageType.DIAGRAM:    Mode.VISUAL,
    ImageType.SCREENSHOT: Mode.VISUAL,
    ImageType.LINE_ART:   Mode.VISUAL,
    ImageType.UNKNOWN:    Mode.VISUAL,
}

# Which vtracer preset to use for each Mode.
MODE_TO_PRESET: dict[Mode, Preset] = {
    Mode.LABELS: "logo",
    Mode.VISUAL: "default",
    Mode.ANNOTATED: "default",
    Mode.TRACE: "photo",
    Mode.POSTER: "poster",
    Mode.DETAILED: "photo_hifi",
    Mode.EDGE: "bw_edge",
    Mode.WATERCOLOR: "watercolor",
    Mode.SEGMENTED: "default",
}


def select_mode(image_type: ImageType, requested: Mode = Mode.AUTO) -> tuple[Mode, str]:
    """Resolve which Mode to use.

    If `requested` is not AUTO, the user override wins and we return
    `(requested, "explicit override")`.

    Otherwise we use the IMAGE_TYPE_TO_MODE table and return
    `(mode, f"auto: {image_type} -> {mode}")`.
    """
    if requested != Mode.AUTO:
        return (requested, "explicit override")
    mode = IMAGE_TYPE_TO_MODE[image_type]
    return (mode, f"auto: {image_type} -> {mode}")


def select_preset(mode: Mode) -> Preset:
    """Return the vtracer preset for a Mode.

    Falls back to 'default' for unknown modes.
    """
    return MODE_TO_PRESET.get(mode, "default")
