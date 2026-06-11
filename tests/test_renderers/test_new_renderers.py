# img2svg - tests for the new preset-renderer classes and registry wiring.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the new preset-renderer classes (T6/T7/T8/T9/T16) and the
expanded :data:`RENDERER_REGISTRY` / :data:`IMAGE_TYPE_TO_MODE` /
:data:`MODE_TO_PRESET` tables.

Every test that touches vtracer uses ``Mock(VtracerVectorizer)`` so the
native binary is never invoked. Real photos from ``tests/fixtures/photo.jpg``
or ``tests/fixtures/logo.png`` are used as end-to-end smoke inputs.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from img2svg.enums import ImageType, Mode
from img2svg.loader import LoadedImage
from img2svg.pipeline import RENDERER_REGISTRY
from img2svg.presets import IMAGE_TYPE_TO_MODE, MODE_TO_PRESET
from img2svg.renderers.base import Renderer
from img2svg.renderers.detailed import DetailedRenderer
from img2svg.renderers.edge import EdgeRenderer
from img2svg.renderers.poster import PosterRenderer
from img2svg.renderers.segmented import SegmentedRenderer
from img2svg.renderers.watercolor import WatercolorRenderer
from img2svg.svg_builder import SVGDocument

_FAKE_VTRACER_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <path d="M0 0 L10 10" fill="#ff0000"/>
</svg>
"""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_loaded_image(width: int = 20, height: int = 20) -> LoadedImage:
    """Build a tiny RGB LoadedImage backed by a PIL image."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[5:15, 5:15] = [255, 0, 0]
    pil = Image.fromarray(arr)
    return LoadedImage(
        pil_image=pil,
        format="PNG",
        original_mode="RGB",
        has_alpha=False,
        width=width,
        height=height,
    )


def _make_svg_doc() -> SVGDocument:
    return SVGDocument(100, 100, title="t", desc="d")


def _fake_vectorize_side_effect(body: str):
    """Return a `vectorize` side_effect that writes `body` to the output path."""

    def _side_effect(inp: str, outp: str) -> None:
        Path(outp).write_text(body, encoding="utf-8")

    return _side_effect


# ----------------------------------------------------------------------
# Renderer class structure
# ----------------------------------------------------------------------


class TestNewRendererClasses:
    """The 5 new preset-renderer classes are all Renderer subclasses."""

    def test_poster_renderer_is_renderer_subclass(self) -> None:
        assert issubclass(PosterRenderer, Renderer)

    def test_detailed_renderer_is_renderer_subclass(self) -> None:
        assert issubclass(DetailedRenderer, Renderer)

    def test_edge_renderer_is_renderer_subclass(self) -> None:
        assert issubclass(EdgeRenderer, Renderer)

    def test_watercolor_renderer_is_renderer_subclass(self) -> None:
        assert issubclass(WatercolorRenderer, Renderer)

    def test_segmented_renderer_is_renderer_subclass(self) -> None:
        assert issubclass(SegmentedRenderer, Renderer)

    @pytest.mark.parametrize(
        ("renderer_cls", "expected_preset"),
        [
            (PosterRenderer, "poster"),
            (DetailedRenderer, "photo_hifi"),
            (EdgeRenderer, "bw_edge"),
            (WatercolorRenderer, "watercolor"),
            (SegmentedRenderer, "default"),
        ],
    )
    def test_preset_name_classvar(self, renderer_cls: type[Renderer], expected_preset: str) -> None:
        """Each renderer declares the correct vtracer preset as a ClassVar."""
        assert renderer_cls.preset_name == expected_preset

    def test_all_renderers_can_be_constructed(self) -> None:
        """Each new renderer can be instantiated with the standard 4-arg signature."""
        svg = _make_svg_doc()
        image = _make_loaded_image()
        for cls in (
            PosterRenderer,
            DetailedRenderer,
            EdgeRenderer,
            WatercolorRenderer,
        ):
            r = cls(svg, image, [], None)
            assert r.svg is svg
            assert r.image is image
            assert r.detections == []
        sr = SegmentedRenderer(svg, image, [], None)
        assert sr._segmentation_result is None


# ----------------------------------------------------------------------
# End-to-end render with mocked vtracer
# ----------------------------------------------------------------------


class TestRenderWithMockedVtracer:
    """Each new preset renderer produces a non-empty SVG when vtracer is stubbed."""

    @pytest.mark.parametrize(
        "mode",
        [Mode.POSTER, Mode.DETAILED, Mode.EDGE, Mode.WATERCOLOR],
    )
    def test_preset_renderer_runs_with_mock(
        self, tmp_path: Path, logo_path: Path, mode: Mode
    ) -> None:
        """The 4 single-call preset renderers all go through vtracer exactly once."""
        from img2svg.models import ConversionOptions
        from img2svg.pipeline import Pipeline

        out_svg = tmp_path / f"out_{mode.value}.svg"

        mock_det = mock.MagicMock()
        mock_det.detect.return_value = []
        mock_det.device = "cpu"

        with (
            mock.patch("img2svg.pipeline.get_detector", return_value=mock_det),
            mock.patch("img2svg.renderers.visual.VtracerVectorizer") as mock_vec,
        ):
            mock_vec.return_value.vectorize.side_effect = _fake_vectorize_side_effect(
                _FAKE_VTRACER_SVG
            )
            result = Pipeline(ConversionOptions(mode=mode)).run(logo_path, out_svg)

        assert out_svg.exists()
        assert result.sidecar.mode_used == mode
        assert mock_vec.call_count == 1
        assert mock_vec.call_args.kwargs.get("preset") == RENDERER_REGISTRY[mode].preset_name


# ----------------------------------------------------------------------
# RENDERER_REGISTRY wiring
# ----------------------------------------------------------------------


class TestRendererRegistry:
    """``RENDERER_REGISTRY`` covers all 9 concrete modes (AUTO is resolved first)."""

    def test_registry_size(self) -> None:
        """Exactly 9 entries (one per concrete Mode; AUTO is deliberately absent)."""
        assert len(RENDERER_REGISTRY) == 9

    def test_registry_has_all_concrete_modes(self) -> None:
        for mode in Mode:
            if mode == Mode.AUTO:
                assert mode not in RENDERER_REGISTRY, "AUTO must be resolved via select_mode()"
                continue
            assert mode in RENDERER_REGISTRY, f"missing registry entry for {mode!r}"

    def test_registry_wires_new_renderers(self) -> None:
        """The 5 new renderers are registered to their respective Modes."""
        assert RENDERER_REGISTRY[Mode.POSTER] is PosterRenderer
        assert RENDERER_REGISTRY[Mode.DETAILED] is DetailedRenderer
        assert RENDERER_REGISTRY[Mode.EDGE] is EdgeRenderer
        assert RENDERER_REGISTRY[Mode.WATERCOLOR] is WatercolorRenderer
        assert RENDERER_REGISTRY[Mode.SEGMENTED] is SegmentedRenderer


# ----------------------------------------------------------------------
# IMAGE_TYPE_TO_MODE — AUTO never picks explicit-only modes
# ----------------------------------------------------------------------


class TestImageTypeToMode:
    """``IMAGE_TYPE_TO_MODE`` must cover every ImageType and avoid explicit-only modes."""

    def test_covers_all_image_types(self) -> None:
        for it in ImageType:
            assert it in IMAGE_TYPE_TO_MODE

    def test_no_explicit_only_modes(self) -> None:
        """AUTO must never resolve to LABELS, ANNOTATED, or SEGMENTED."""
        forbidden = {Mode.LABELS, Mode.ANNOTATED, Mode.SEGMENTED}
        for image_type, mode in IMAGE_TYPE_TO_MODE.items():
            assert mode not in forbidden, f"AUTO must not resolve {image_type!r} to {mode!r}"

    def test_photo_uses_detailed(self) -> None:
        assert IMAGE_TYPE_TO_MODE[ImageType.PHOTO] == Mode.DETAILED


# ----------------------------------------------------------------------
# MODE_TO_PRESET — every concrete Mode has a vtracer preset
# ----------------------------------------------------------------------


class TestModeToPreset:
    """``MODE_TO_PRESET`` covers all 9 non-AUTO modes with valid preset names."""

    def test_covers_all_non_auto_modes(self) -> None:
        for m in Mode:
            if m == Mode.AUTO:
                continue
            assert m in MODE_TO_PRESET

    def test_preset_values_are_known(self) -> None:
        from img2svg.vectorizer import PRESETS

        for mode, preset in MODE_TO_PRESET.items():
            assert preset in PRESETS, f"mode {mode!r} maps to unknown preset {preset!r}"

    @pytest.mark.parametrize(
        ("mode", "expected_preset"),
        [
            (Mode.LABELS, "logo"),
            (Mode.VISUAL, "default"),
            (Mode.ANNOTATED, "default"),
            (Mode.TRACE, "photo"),
            (Mode.POSTER, "poster"),
            (Mode.DETAILED, "photo_hifi"),
            (Mode.EDGE, "bw_edge"),
            (Mode.WATERCOLOR, "watercolor"),
            (Mode.SEGMENTED, "default"),
        ],
    )
    def test_specific_mode_to_preset(self, mode: Mode, expected_preset: str) -> None:
        assert MODE_TO_PRESET[mode] == expected_preset
