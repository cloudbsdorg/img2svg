# img2svg - auto-mode selection: map image type to renderer mode, and mode to vtracer preset.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Auto-mode selection: map image type to renderer mode, and mode to vtracer preset."""

from __future__ import annotations

from img2svg.enums import ImageType, Mode
from img2svg.vectorizer import Preset

# Which Mode to use for each ImageType (used when user passes --mode auto).
IMAGE_TYPE_TO_MODE: dict[ImageType, Mode] = {
    ImageType.LOGO: Mode.LABELS,
    ImageType.PHOTO: Mode.ANNOTATED,
    ImageType.DIAGRAM: Mode.LABELS,
    ImageType.SCREENSHOT: Mode.VISUAL,
    ImageType.LINE_ART: Mode.LABELS,
    ImageType.UNKNOWN: Mode.ANNOTATED,
}

# Which vtracer preset to use for each Mode.
MODE_TO_PRESET: dict[Mode, Preset] = {
    Mode.LABELS: "logo",
    Mode.VISUAL: "default",
    Mode.ANNOTATED: "default",
    Mode.TRACE: "photo",
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
