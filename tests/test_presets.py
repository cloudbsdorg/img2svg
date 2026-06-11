# img2svg - tests for auto-mode selection and mode->preset mapping.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for auto-mode selection and mode→preset mapping."""

from __future__ import annotations

from img2svg.enums import ImageType, Mode
from img2svg.presets import (
    IMAGE_TYPE_TO_MODE,
    MODE_TO_PRESET,
    select_mode,
    select_preset,
)


def test_image_type_to_mode_covers_all_types() -> None:
    for it in ImageType:
        assert it in IMAGE_TYPE_TO_MODE


def test_image_type_to_mode_never_picks_explicit_only_modes() -> None:
    """AUTO must never resolve to LABELS, ANNOTATED, or SEGMENTED.

    Those modes are explicit-only — users opt in by passing the mode flag.
    Regression guard for the user constraint recorded in the photo-quality
    plan (T10).
    """
    forbidden = {Mode.LABELS, Mode.ANNOTATED, Mode.SEGMENTED}
    for image_type, mode in IMAGE_TYPE_TO_MODE.items():
        assert mode not in forbidden, (
            f"AUTO must not resolve {image_type!r} to {mode!r}; "
            f"LABELS/ANNOTATED/SEGMENTED are explicit-only"
        )


def test_image_type_to_mode_photo_uses_detailed() -> None:
    """PHOTO in AUTO must use DETAILED (aggressive pre-process + hi-fi trace)."""
    assert IMAGE_TYPE_TO_MODE[ImageType.PHOTO] == Mode.DETAILED


def test_mode_to_preset_covers_non_auto_modes() -> None:
    for m in Mode:
        if m == Mode.AUTO:
            continue
        assert m in MODE_TO_PRESET


def test_select_mode_explicit_override_wins() -> None:
    """Explicit --mode always wins over auto-mapping."""
    for it in ImageType:
        for m in Mode:
            if m == Mode.AUTO:
                continue
            chosen, reasoning = select_mode(it, m)
            assert chosen == m
            assert reasoning == "explicit override"


def test_select_mode_auto_uses_image_type_table() -> None:
    chosen, reasoning = select_mode(ImageType.PHOTO, Mode.AUTO)
    assert chosen == Mode.DETAILED
    assert "auto" in reasoning.lower()
    assert "photo" in reasoning.lower()


def test_select_mode_auto_for_each_type() -> None:
    """Every ImageType maps to a known Mode in auto mode."""
    for it in ImageType:
        chosen, _ = select_mode(it, Mode.AUTO)
        assert isinstance(chosen, Mode)
        assert chosen != Mode.AUTO  # auto resolves to a concrete mode


def test_select_preset_for_known_modes() -> None:
    assert select_preset(Mode.LABELS) == "logo"
    assert select_preset(Mode.VISUAL) == "default"
    assert select_preset(Mode.ANNOTATED) == "default"
    assert select_preset(Mode.TRACE) == "photo"


def test_select_preset_for_auto_falls_back() -> None:
    """AUTO isn't a vtracer preset; should fall back to 'default'."""
    assert select_preset(Mode.AUTO) == "default"


def test_mode_to_preset_values_are_known() -> None:
    from img2svg.vectorizer import PRESETS

    for mode, preset in MODE_TO_PRESET.items():
        assert preset in PRESETS, f"mode {mode!r} maps to unknown preset {preset!r}"
