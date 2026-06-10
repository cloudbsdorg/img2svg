# img2svg - integration tests (real YOLO detector and vtracer against fixtures).
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for img2svg.

These tests exercise the real YOLO detector (downloading the small
``yolo11n.pt`` model on first use) and the real ``vtracer`` binary
against the on-disk fixtures. They are slow and skipped by default;
run them with::

    uv run pytest -m slow

or::

    ./scripts/ci.sh --slow

The full default suite (``uv run pytest``) excludes them via the
``addopts = ["-m", "not slow"]`` in ``pyproject.toml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from img2svg import convert, convert_batch
from img2svg.enums import Mode
from img2svg.models import ConversionOptions
from img2svg.vectorizer import VtracerVectorizer

# All tests in this module are integration tests that hit the real
# heavy dependencies (YOLO model download, vtracer binary). They are
# gated behind a single ``slow`` marker so the default test run stays
# fast.
pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------
# 1. Real YOLO on a synthetic-style fixture
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_real_yolo_detects_in_synthetic(logo_path: Path, tmp_path: Path) -> None:
    """Run a real YOLO inference on the logo fixture using the small model.

    Uses ``yolo11n.pt`` (the smallest, ~5 MB) so the test stays cheap.
    The yolo11x.pt model may be present in the XDG cache from a prior
    end-to-end run, but we explicitly request the nano model here so
    this test does not depend on cache state.

    The test asserts the call returns a list (possibly empty) without
    raising — i.e. the model loaded, the inference ran, and the result
    was structurally valid.
    """
    from img2svg.detector import YOLODetector  # heavy import; keep inside test

    # Skip cleanly if the model is not downloadable (offline CI etc.)
    try:
        detector = YOLODetector(model_name="yolo11n.pt", device_str="cpu")
    except Exception as e:
        pytest.skip(f"yolo11n.pt not available: {e}")

    import numpy as np
    from PIL import Image

    with Image.open(logo_path) as img:
        arr = np.array(img.convert("RGB"))

    detections = detector.detect(arr, conf=0.25, iou=0.7, imgsz=640)
    # Real YOLO returns a list of Detection objects; the assertion is
    # that the call didn't crash and the type contract holds. We don't
    # assert on a specific count — synthetic fixtures may yield 0.
    assert isinstance(detections, list)


# ----------------------------------------------------------------------
# 2. Real vtracer direct invocation
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_real_vtracer_produces_svg(logo_path: Path, tmp_path: Path) -> None:
    """Feed the logo fixture directly through vtracer and assert SVG output.

    This bypasses the img2svg pipeline and uses ``VtracerVectorizer``
    (the production wrapper) with the default preset. The SVG file
    must exist and be non-empty.
    """
    out_svg = tmp_path / "logo.svg"
    VtracerVectorizer(preset="default").vectorize(logo_path, out_svg)

    assert out_svg.exists(), "vtracer did not produce an output SVG"
    assert out_svg.stat().st_size > 0, "vtracer produced an empty SVG"
    # Sanity-check the SVG content starts with a real SVG root.
    head = out_svg.read_text(encoding="utf-8")[:200].lstrip()
    assert head.startswith("<?xml") or head.startswith("<svg"), (
        f"output does not look like SVG; first 200 chars: {head!r}"
    )


# ----------------------------------------------------------------------
# 3. Full public-API pipeline end-to-end
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_full_pipeline_end_to_end(logo_path: Path, tmp_path: Path) -> None:
    """End-to-end through the public ``convert()`` API.

    This exercises: loader → classifier → geometric → YOLO detector
    (small model) → renderer (vtracer for visual modes) → writer
    (atomic SVG + sidecar JSON).

    For modes that don't need vtracer (LABELS), it also exercises the
    real YOLO download path. We pick ``Mode.LABELS`` for this test
    because it is the cheapest full pipeline that still hits YOLO.
    """
    out_svg = tmp_path / "out.svg"

    result = convert(
        logo_path,
        out_svg,
        model="yolo11n.pt",
        device="cpu",
        mode=Mode.LABELS,
    )

    assert out_svg.exists(), "SVG output missing"
    assert out_svg.stat().st_size > 0, "SVG output is empty"
    sidecar = out_svg.with_suffix(".json")
    assert sidecar.exists(), "sidecar JSON missing"
    assert sidecar.stat().st_size > 0, "sidecar JSON is empty"
    # Round-trip the sidecar through Pydantic to confirm schema validity.
    from img2svg.models import Sidecar

    loaded = Sidecar.model_validate_json(sidecar.read_text(encoding="utf-8"))
    assert loaded.mode_used == Mode.LABELS
    assert loaded.model == "yolo11n.pt"
    assert loaded.output_path == out_svg


# ----------------------------------------------------------------------
# 4. Batch with mixed valid + corrupt inputs
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_batch_with_errors(
    logo_path: Path, photo_path: Path, corrupt_path: Path, tmp_path: Path
) -> None:
    """Run a batch with 1 corrupt and 2 valid fixtures; assert 2 succeed.

    ``convert_batch`` with ``continue_on_error=True`` (default) should
    record the corrupt file's failure in its ``errors`` list and
    successfully process the two real images.
    """
    inputs = [corrupt_path, logo_path, photo_path]
    results = convert_batch(
        inputs,
        output_dir=tmp_path,
        model="yolo11n.pt",
        device="cpu",
        mode=Mode.LABELS,
    )

    assert len(results) == 3
    successes = [r for r in results if not r.errors]
    failures = [r for r in results if r.errors]
    assert len(successes) == 2, f"expected 2 successes, got {len(successes)}"
    assert len(failures) == 1, f"expected 1 failure, got {len(failures)}"

    # The two successful SVGs should be on disk.
    for r in successes:
        assert r.svg_path.exists(), f"missing SVG for {r.svg_path}"
    # The corrupt input should have a non-empty error string.
    assert failures[0].errors, "corrupt input did not record an error message"


# ----------------------------------------------------------------------
# 5. All 4 concrete modes on the logo fixture
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "mode",
    [Mode.LABELS, Mode.VISUAL, Mode.ANNOTATED, Mode.TRACE],
)
def test_all_4_modes(logo_path: Path, tmp_path: Path, mode: Mode) -> None:
    """Run every concrete mode (labels, visual, annotated, trace) on the logo.

    Each mode should produce a non-empty SVG file. ``Mode.AUTO`` is
    excluded because it is resolved by the classifier before a concrete
    mode is picked; the 4 concrete modes are the deterministic set
    exercised here.
    """
    out_svg = tmp_path / f"out_{mode.value}.svg"
    opts = ConversionOptions(mode=mode, model="yolo11n.pt", device="cpu")

    convert(logo_path, out_svg, options=opts)

    assert out_svg.exists(), f"SVG not written for mode={mode.value}"
    assert out_svg.stat().st_size > 0, f"empty SVG for mode={mode.value}"
