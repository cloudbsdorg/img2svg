# img2svg - tests for the vtracer vectorizer wrapper.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the vtracer vectorizer wrapper. Mocks vtracer."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from img2svg.errors import VectorizationError
from img2svg.vectorizer import PRESETS, VtracerVectorizer


def test_presets_dict_has_five_entries() -> None:
    assert set(PRESETS.keys()) == {"default", "bw", "logo", "poster", "photo"}


def test_each_preset_has_required_keys() -> None:
    required = {
        "colormode",
        "hierarchical",
        "mode",
        "filter_speckle",
        "color_precision",
        "layer_difference",
        "corner_threshold",
        "length_threshold",
        "max_iterations",
        "splice_threshold",
        "path_precision",
    }
    for name, params in PRESETS.items():
        assert required.issubset(params.keys()), (
            f"preset {name!r} missing keys: {required - params.keys()}"
        )


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        VtracerVectorizer(preset="nope")  # type: ignore[arg-type]
    assert "nope" in str(exc_info.value)
    assert "default" in str(exc_info.value)


def test_default_preset_is_color_stacked_spline() -> None:
    p = VtracerVectorizer("default").params
    assert p["colormode"] == "color"
    assert p["hierarchical"] == "stacked"
    assert p["mode"] == "spline"


def test_bw_preset_is_binary() -> None:
    p = VtracerVectorizer("bw").params
    assert p["colormode"] == "binary"


def test_vectorize_calls_vtracer_with_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = mock.MagicMock()
    monkeypatch.setattr("img2svg.vectorizer.vtracer.convert_image_to_svg_py", fake)
    v = VtracerVectorizer("logo")
    v.vectorize(tmp_path / "in.png", tmp_path / "out.svg")
    fake.assert_called_once()
    call = fake.call_args
    # Positional: input_path, output_path
    assert call.args[0] == str(tmp_path / "in.png")
    assert call.args[1] == str(tmp_path / "out.svg")
    # Keyword: all preset params
    assert call.kwargs["colormode"] == "color"
    assert call.kwargs["filter_speckle"] == 8
    assert call.kwargs["color_precision"] == 8


def test_vectorize_raises_vectorization_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise(*a: object, **kw: object) -> None:
        raise RuntimeError("vtracer exploded")

    monkeypatch.setattr("img2svg.vectorizer.vtracer.convert_image_to_svg_py", _raise)
    v = VtracerVectorizer("default")
    with pytest.raises(VectorizationError) as exc_info:
        v.vectorize(tmp_path / "in.png", tmp_path / "out.svg")
    assert "in.png" in str(exc_info.value)
    assert "vtracer exploded" in str(exc_info.value)


def test_each_preset_has_distinct_params() -> None:
    """At minimum, default, bw, logo, and photo should have at least one differing param."""
    assert PRESETS["default"]["colormode"] != PRESETS["bw"]["colormode"]
    assert PRESETS["default"]["filter_speckle"] != PRESETS["logo"]["filter_speckle"]
    assert PRESETS["default"]["layer_difference"] != PRESETS["photo"]["layer_difference"]
