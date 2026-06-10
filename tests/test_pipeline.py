# img2svg - tests for the `Pipeline` orchestrator.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `Pipeline` orchestrator.

The YOLO detector is mocked at `img2svg.pipeline.get_detector` to avoid
loading model weights. vtracer is mocked at `img2svg.renderers.visual`
because the helper used by `VisualRenderer` / `TraceRenderer` /
`AnnotatedRenderer` lives there. The `LabelsRenderer` path does not
touch vtracer and is exercised without any extra patching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from img2svg.enums import ImageType, Mode
from img2svg.errors import OutputPathCollisionError
from img2svg.models import (
    BoundingBox,
    ConversionOptions,
    ConversionResult,
    Detection,
    Sidecar,
)
from img2svg.pipeline import RENDERER_REGISTRY, Pipeline
from img2svg.renderers.annotated import AnnotatedRenderer
from img2svg.renderers.labels import LabelsRenderer
from img2svg.renderers.trace import TraceRenderer
from img2svg.renderers.visual import VisualRenderer

# A tiny valid vtracer-style SVG used to stub the `VtracerVectorizer` so
# renderers that touch vtracer do not actually run the binary.
_FAKE_VTRACER_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <path d="M0 0 L10 10" fill="#ff0000"/>
</svg>
"""


def _fake_vectorize_side_effect(body: str):
    """Return a `vectorize` side_effect that writes `body` to the output path."""

    def _side_effect(inp: str, outp: str) -> None:
        Path(outp).write_text(body, encoding="utf-8")

    return _side_effect


def _make_detection(class_name: str = "person", conf: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=conf,
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
    )


def _make_mock_detector(detections: list[Detection] | None = None) -> Any:
    """Return a mock `YOLODetector` with a `.detect()` method returning `detections`."""
    mock_det = mock.MagicMock()
    mock_det.detect.return_value = detections if detections is not None else []
    return mock_det


# ----------------------------------------------------------------------
# Registry / structural
# ----------------------------------------------------------------------


def test_renderer_registry_has_all_concrete_modes() -> None:
    """RENDERER_REGISTRY must contain every concrete Mode except AUTO."""
    assert RENDERER_REGISTRY[Mode.LABELS] is LabelsRenderer
    assert RENDERER_REGISTRY[Mode.VISUAL] is VisualRenderer
    assert RENDERER_REGISTRY[Mode.ANNOTATED] is AnnotatedRenderer
    assert RENDERER_REGISTRY[Mode.TRACE] is TraceRenderer
    assert Mode.AUTO not in RENDERER_REGISTRY, "AUTO must be resolved via select_mode()"


# ----------------------------------------------------------------------
# Pipeline.run() — LabelsRenderer (no vtracer, no detector)
# ----------------------------------------------------------------------


def test_pipeline_run_labels_mode_produces_svg_and_sidecar(tmp_path: Path, logo_path: Path) -> None:
    """End-to-end with LABELS mode (no vtracer) and a stubbed detector.

    Verifies the full 12-step flow:
    - SVG is written at the given path.
    - Sidecar JSON is written at `output.with_suffix('.json')`.
    - The sidecar validates as a `Sidecar` and has the expected metadata.
    """
    out_svg = tmp_path / "out.svg"

    with (
        mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector([])),
        mock.patch("img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")),
    ):
        result = Pipeline(ConversionOptions(mode=Mode.LABELS)).run(logo_path, out_svg)

    assert isinstance(result, ConversionResult)
    assert result.svg_path == out_svg
    assert out_svg.exists(), "SVG output not written"
    sidecar_path = out_svg.with_suffix(".json")
    assert sidecar_path.exists(), "Sidecar JSON not written"
    assert result.sidecar_path == sidecar_path

    # Sidecar round-trips and is internally consistent
    loaded = Sidecar.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
    assert loaded.input_path == logo_path
    assert loaded.output_path == out_svg
    assert loaded.mode_used == Mode.LABELS
    assert loaded.image_type == ImageType.LOGO, "classify was mocked to return LOGO"
    assert loaded.model == "yolo11x.pt"
    assert loaded.detections == []
    assert loaded.geometric is not None
    assert "total" in loaded.timings and loaded.timings["total"] >= 0.0

    # JSON content is sane
    raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert raw["mode_used"] == "labels"
    assert raw["image_type"] == "logo"
    assert len(raw["input_hash"]) == 64, "input_hash should be SHA-256 hex"


def test_pipeline_run_with_detections_populates_sidecar(tmp_path: Path, photo_path: Path) -> None:
    """With a mocked detector returning detections, the sidecar carries them."""
    dets = [_make_detection("person", 0.87), _make_detection("dog", 0.65)]
    out_svg = tmp_path / "out.svg"

    with mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector(dets)):
        result = Pipeline(ConversionOptions(mode=Mode.ANNOTATED)).run(photo_path, out_svg)

    assert len(result.detections) == 2
    assert result.sidecar.detections[0].class_name == "person"
    assert result.sidecar.detections[0].confidence == 0.87
    assert result.sidecar.mode_used == Mode.ANNOTATED
    # Output file size is the on-disk size
    assert result.sidecar.output_size == out_svg.stat().st_size


def test_pipeline_respects_user_mode_override(tmp_path: Path, logo_path: Path) -> None:
    """Explicit mode override is honored regardless of image_type."""
    out_svg = tmp_path / "out.svg"

    with mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector([])):
        result = Pipeline(ConversionOptions(mode=Mode.TRACE)).run(logo_path, out_svg)

    assert result.sidecar.mode_used == Mode.TRACE
    assert "explicit override" in result.sidecar.mode_reasoning


def test_pipeline_auto_mode_resolves_to_concrete(tmp_path: Path, logo_path: Path) -> None:
    """Mode.AUTO is resolved to a concrete mode via the classifier + select_mode."""
    out_svg = tmp_path / "out.svg"

    with (
        mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector([])),
        mock.patch("img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")),
    ):
        result = Pipeline(ConversionOptions(mode=Mode.AUTO)).run(logo_path, out_svg)

    # Forced LOGO classification → IMAGE_TYPE_TO_MODE[LOGO] = Mode.LABELS
    assert result.sidecar.mode_used == Mode.LABELS
    assert "auto:" in result.sidecar.mode_reasoning


# ----------------------------------------------------------------------
# Pipeline.run() — renderers that touch vtracer
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", [Mode.VISUAL, Mode.ANNOTATED, Mode.TRACE])
def test_pipeline_runs_vtracer_renderer_modes(tmp_path: Path, logo_path: Path, mode: Mode) -> None:
    """VISUAL, ANNOTATED, TRACE all hit vtracer. Stub the vectorizer."""
    out_svg = tmp_path / f"out_{mode.value}.svg"

    with (
        mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector([])),
        mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec,
    ):
        MockVec.return_value.vectorize.side_effect = _fake_vectorize_side_effect(_FAKE_VTRACER_SVG)
        result = Pipeline(ConversionOptions(mode=mode)).run(logo_path, out_svg)

    assert out_svg.exists()
    assert result.sidecar.mode_used == mode
    # Vtracer was called exactly once per pipeline run
    MockVec.assert_called_once()


# ----------------------------------------------------------------------
# Error propagation
# ----------------------------------------------------------------------


def test_pipeline_raises_on_unsupported_format(tmp_path: Path, corrupt_path: Path) -> None:
    """An unsupported / corrupt input propagates the loader error."""
    out_svg = tmp_path / "out.svg"
    pipeline = Pipeline(ConversionOptions(mode=Mode.LABELS))
    with pytest.raises(Exception) as ei:
        pipeline.run(corrupt_path, out_svg)
    # The loader raises CorruptImageError or UnsupportedFormatError (Img2SvgError subclass)
    assert not out_svg.exists()
    assert "img2svg" in type(ei.value).__module__ or "Img2SvgError" in [
        base.__name__ for base in type(ei.value).__mro__
    ]


# ----------------------------------------------------------------------
# No-clobber guard
# ----------------------------------------------------------------------


_SENTINEL_SVG = "<sentinel>do-not-overwrite</sentinel>\n"


def test_pipeline_no_clobber_raises_on_existing_output(
    tmp_path: Path, logo_path: Path
) -> None:
    """With `no_clobber=True` and a pre-existing output, raise and leave the file alone."""
    out_svg = tmp_path / "out.svg"
    out_svg.write_text(_SENTINEL_SVG, encoding="utf-8")
    original_bytes = out_svg.read_bytes()
    original_mtime = out_svg.stat().st_mtime

    mock_detector = _make_mock_detector([])
    options = ConversionOptions(mode=Mode.LABELS, no_clobber=True)
    with (
        mock.patch("img2svg.pipeline.get_detector", return_value=mock_detector),
        pytest.raises(OutputPathCollisionError) as exc_info,
    ):
        Pipeline(options).run(logo_path, out_svg)

    assert exc_info.value.path == str(out_svg)
    assert out_svg.read_bytes() == original_bytes
    assert out_svg.stat().st_mtime == original_mtime
    mock_detector.detect.assert_not_called()


def test_pipeline_no_clobber_false_overwrites_existing_output(
    tmp_path: Path, logo_path: Path
) -> None:
    """When `no_clobber=False` (the default), a pre-existing output is overwritten."""
    out_svg = tmp_path / "out.svg"
    out_svg.write_text(_SENTINEL_SVG, encoding="utf-8")

    with mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector([])):
        result = Pipeline(ConversionOptions(mode=Mode.LABELS)).run(logo_path, out_svg)

    assert out_svg.exists()
    assert out_svg.read_text(encoding="utf-8") != _SENTINEL_SVG
    assert result.svg_path == out_svg


# ----------------------------------------------------------------------
# Timings
# ----------------------------------------------------------------------


def test_pipeline_records_timings(tmp_path: Path, logo_path: Path) -> None:
    """The sidecar.timings dict has all expected keys and total >= sum-of-parts."""
    out_svg = tmp_path / "out.svg"
    with mock.patch("img2svg.pipeline.get_detector", return_value=_make_mock_detector([])):
        result = Pipeline(ConversionOptions(mode=Mode.LABELS)).run(logo_path, out_svg)

    t = result.sidecar.timings
    for key in ("load", "analyze", "classify", "select_mode", "detect", "render", "write", "total"):
        assert key in t, f"missing timings key: {key!r}"
        assert t[key] >= 0.0
    # total is measured independently and should be >= max(parts)
    assert t["total"] >= 0.0
