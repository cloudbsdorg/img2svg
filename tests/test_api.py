"""Tests for the high-level public API (`convert`, `convert_batch`).

The YOLO detector and vtracer are mocked at the same boundaries as the
pipeline tests so the suite stays fast and has no model-dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from img2svg.api import (
    _build_options,
    _expand_glob,
    _resolve_output_dir,
    convert,
    convert_batch,
)
from img2svg.enums import ImageType, Mode
from img2svg.models import (
    BoundingBox,
    ConversionOptions,
    ConversionResult,
    Detection,
    Sidecar,
)


_FAKE_VTRACER_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <path d="M0 0 L10 10" fill="#ff0000"/>
</svg>
"""


def _fake_vectorize_side_effect(body: str):
    def _side_effect(inp: str, outp: str) -> None:
        Path(outp).write_text(body, encoding="utf-8")
    return _side_effect


def _make_mock_detector(detections: list[Detection] | None = None) -> Any:
    mock_det = mock.MagicMock()
    mock_det.detect.return_value = detections if detections is not None else []
    return mock_det


# ----------------------------------------------------------------------
# Public re-exports
# ----------------------------------------------------------------------


def test_public_api_imports() -> None:
    """`from img2svg import convert, convert_batch, ...` works."""
    from img2svg import (  # noqa: F401
        ConversionOptions,
        ConversionResult,
        DeviceStrategy,
        ImageType,
        Mode,
        Sidecar,
        convert,
        convert_batch,
    )

    assert callable(convert)
    assert callable(convert_batch)
    assert Mode.AUTO == "auto"
    assert ImageType.LOGO == "logo"
    assert DeviceStrategy.AUTO == "auto"


# ----------------------------------------------------------------------
# convert()
# ----------------------------------------------------------------------


def test_convert_returns_conversion_result(
    tmp_path: Path, logo_path: Path
) -> None:
    """`convert(input, output)` returns a `ConversionResult` and writes both files."""
    out_svg = tmp_path / "out.svg"

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ):
        result = convert(logo_path, out_svg, mode=Mode.LABELS)

    assert isinstance(result, ConversionResult)
    assert result.svg_path == out_svg
    assert out_svg.exists()
    assert result.sidecar_path.exists()
    # Defaults via kwargs: mode=LABELS overrides AUTO
    assert result.sidecar.mode_used == Mode.LABELS


def test_convert_accepts_options_object(
    tmp_path: Path, logo_path: Path
) -> None:
    """`convert(..., options=ConversionOptions(...))` is honored."""
    out_svg = tmp_path / "out.svg"
    opts = ConversionOptions(mode=Mode.VISUAL, conf=0.5, iou=0.6)

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ), mock.patch(
        "img2svg.renderers.visual.VtracerVectorizer"
    ) as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize_side_effect(
            _FAKE_VTRACER_SVG
        )
        result = convert(logo_path, out_svg, options=opts)

    assert result.sidecar.mode_used == Mode.VISUAL
    MockVec.assert_called_once()


def test_convert_drops_unknown_kwargs(
    tmp_path: Path, logo_path: Path
) -> None:
    """Unknown kwargs are silently ignored; known kwargs override defaults."""
    out_svg = tmp_path / "out.svg"

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ):
        # `not_a_real_field` should not raise
        result = convert(
            logo_path, out_svg,
            mode=Mode.LABELS,
            conf=0.4,
            not_a_real_field="ignored",
        )

    assert result.sidecar.mode_used == Mode.LABELS


def test_convert_vtracer_mode_produces_svg(
    tmp_path: Path, logo_path: Path
) -> None:
    """VISUAL mode hits vtracer and produces a vtracer-output group."""
    out_svg = tmp_path / "out.svg"

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ), mock.patch(
        "img2svg.renderers.visual.VtracerVectorizer"
    ) as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize_side_effect(
            _FAKE_VTRACER_SVG
        )
        convert(logo_path, out_svg, mode=Mode.VISUAL)

    assert out_svg.exists()
    content = out_svg.read_text(encoding="utf-8")
    assert "vtracer-output" in content


# ----------------------------------------------------------------------
# convert_batch()
# ----------------------------------------------------------------------


def test_convert_batch_processes_list(
    tmp_path: Path, logo_path: Path, photo_path: Path
) -> None:
    """`convert_batch([a, b])` returns a list of `ConversionResult`."""
    inputs = [logo_path, photo_path]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ):
        results = convert_batch(inputs, output_dir=out_dir, mode=Mode.LABELS)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, ConversionResult)
        assert r.svg_path.exists()
        assert r.sidecar_path.exists()


def test_convert_batch_glob_string(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A glob string input is expanded and each file is processed."""
    # Copy two fixtures into tmp_path so we can control the glob.
    targets = []
    for name in ("logo.png", "diagram.png"):
        src = fixtures_dir / name
        dst = tmp_path / name
        dst.write_bytes(src.read_bytes())
        targets.append(dst)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ):
        results = convert_batch(str(tmp_path / "*.png"), output_dir=out_dir, mode=Mode.LABELS)

    assert len(results) == 2
    produced = {r.svg_path.name for r in results}
    assert produced == {"logo.svg", "diagram.svg"}


def test_convert_batch_default_output_dir(
    tmp_path: Path, logo_path: Path, photo_path: Path
) -> None:
    """When `output_dir` is omitted, falls back to `inputs[0].parent`."""
    # Place both inputs in the same parent directory.
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    a = in_dir / "a.png"
    b = in_dir / "b.png"
    a.write_bytes(logo_path.read_bytes())
    b.write_bytes(photo_path.read_bytes())

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ):
        results = convert_batch([a, b], mode=Mode.LABELS)

    # Outputs are written next to the inputs.
    assert (in_dir / "a.svg").exists()
    assert (in_dir / "b.svg").exists()
    assert len(results) == 2


def test_convert_batch_creates_output_dir(
    tmp_path: Path, logo_path: Path
) -> None:
    """`convert_batch` creates the output dir if it doesn't exist."""
    inputs = [logo_path]
    out_dir = tmp_path / "new" / "out"  # doesn't exist yet

    with mock.patch(
        "img2svg.pipeline.get_detector", return_value=_make_mock_detector([])
    ), mock.patch(
        "img2svg.pipeline.classify", return_value=(ImageType.LOGO, "forced → LOGO")
    ):
        results = convert_batch(inputs, output_dir=out_dir, mode=Mode.LABELS)

    assert out_dir.exists()
    assert len(results) == 1
    assert results[0].svg_path.parent == out_dir


# ----------------------------------------------------------------------
# Helper-function unit tests
# ----------------------------------------------------------------------


def test_build_options_with_options_passthrough() -> None:
    """If options is provided, kwargs are ignored."""
    opts = ConversionOptions(mode=Mode.LABELS, conf=0.5)
    out = _build_options(opts, {"conf": 0.1, "mode": Mode.VISUAL})
    assert out is opts
    assert out.conf == 0.5
    assert out.mode == Mode.LABELS


def test_build_options_from_kwargs() -> None:
    """If options is None, kwargs are filtered to known fields."""
    out = _build_options(None, {"mode": Mode.TRACE, "conf": 0.3, "junk": "x"})
    assert isinstance(out, ConversionOptions)
    assert out.mode == Mode.TRACE
    assert out.conf == 0.3
    # unknown kwargs are silently dropped
    assert not hasattr(out, "junk")


def test_resolve_output_dir_uses_explicit() -> None:
    explicit = Path("/tmp/explicit")
    assert _resolve_output_dir([Path("/a/b.png")], explicit) == explicit


def test_resolve_output_dir_defaults_to_inputs_parent() -> None:
    inputs = [Path("/a/b/c.png"), Path("/a/b/d.png")]
    assert _resolve_output_dir(inputs, None) == Path("/a/b")


def test_resolve_output_dir_empty_inputs() -> None:
    """Empty input list falls back to cwd."""
    out = _resolve_output_dir([], None)
    assert out == Path.cwd()


def test_expand_glob_returns_sorted_paths(tmp_path: Path) -> None:
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"x")
    pattern = str(tmp_path / "*.png")
    result = _expand_glob(pattern)
    assert [p.name for p in result] == ["a.png", "b.png", "c.png"]
