# img2svg - tests for the typed exception hierarchy.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the typed exception hierarchy."""

from __future__ import annotations

import pytest

from img2svg import errors


def test_base_class_catches_all() -> None:
    with pytest.raises(errors.Img2SvgError):
        raise errors.UnsupportedFormatError("ICO")


def test_unsupported_format_stores_name() -> None:
    e = errors.UnsupportedFormatError("ICO")
    assert e.format_name == "ICO"
    assert "PNG" in e.supported
    assert "ICO" in e.user_message()


def test_corrupt_image_user_message() -> None:
    e = errors.CorruptImageError("/tmp/bad.png")
    assert "/tmp/bad.png" in e.user_message()
    assert e.path == "/tmp/bad.png"


def test_model_load_user_message() -> None:
    e = errors.ModelLoadError("yolo11x.pt")
    assert "yolo11x.pt" in e.user_message()
    assert "yolo11x.pt" in str(e)


def test_device_unavailable_user_message() -> None:
    e = errors.DeviceUnavailableError("cuda", ["cpu"])
    assert "cuda" in e.user_message()
    assert "cpu" in e.user_message()
    assert "auto" in e.user_message()  # hint


def test_output_path_collision_user_message() -> None:
    e = errors.OutputPathCollisionError("/tmp/out.svg")
    assert "/tmp/out.svg" in e.user_message()
    assert "--force" in e.user_message()


def test_vectorization_error() -> None:
    e = errors.VectorizationError("/tmp/in.png")
    assert "/tmp/in.png" in str(e)


def test_config_error() -> None:
    e = errors.ConfigError("/tmp/cfg.toml", "missing key")
    assert "/tmp/cfg.toml" in str(e)
    assert "missing key" in str(e)


def test_all_have_user_and_dev_message() -> None:
    """Every error class must implement both user_message() and dev_message()."""
    classes = [
        errors.UnsupportedFormatError("X"),
        errors.CorruptImageError("p"),
        errors.ModelLoadError("m"),
        errors.DeviceUnavailableError("cuda", []),
        errors.OutputPathCollisionError("p"),
        errors.VectorizationError("p"),
        errors.ConfigError("p", "r"),
    ]
    for e in classes:
        assert isinstance(e.user_message(), str)
        assert isinstance(e.dev_message(), str)
        assert e.user_message()  # non-empty
        assert e.dev_message()  # non-empty
