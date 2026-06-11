# img2svg - tests for segmentation mask extraction helpers.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the segmentation mask extraction helpers in detector.py."""

from __future__ import annotations

import numpy as np

from img2svg.detector import compute_mask_area, extract_polygons, get_tight_bbox


def test_extract_polygons_empty_mask_returns_zeros() -> None:
    """Empty mask: (0, 2) float32 zeros, dtype correct."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    polygons = extract_polygons([mask])
    assert len(polygons) == 1
    assert polygons[0].shape == (0, 2)
    assert polygons[0].dtype == np.float32


def test_extract_polygons_single_object_square() -> None:
    """Single 20x20 square: polygon has corners tracing the square boundary."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    polygons = extract_polygons([mask])
    assert len(polygons) == 1
    poly = polygons[0]
    assert poly.ndim == 2 and poly.shape[1] == 2
    assert poly.dtype == np.float32
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.min() == 30.0 and xs.max() == 49.0
    assert ys.min() == 20.0 and ys.max() == 39.0
    assert len(poly) >= 3


def test_extract_polygons_multiple_disjoint_objects_returns_largest() -> None:
    """Two squares: only the largest is returned (one polygon per mask)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[50:90, 60:80] = 255
    polygons = extract_polygons([mask])
    assert len(polygons) == 1
    poly = polygons[0]
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.min() == 60.0 and xs.max() == 79.0
    assert ys.min() == 50.0 and ys.max() == 89.0


def test_extract_polygons_full_image_mask() -> None:
    """Full-image mask: polygon traces the image perimeter."""
    mask = np.full((50, 60), 255, dtype=np.uint8)
    polygons = extract_polygons([mask])
    assert len(polygons) == 1
    poly = polygons[0]
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.min() == 0.0 and xs.max() == 59.0
    assert ys.min() == 0.0 and ys.max() == 49.0


def test_extract_polygons_list_mixed() -> None:
    """List with empty + populated masks: one output per input, sizes differ."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    populated = np.zeros((40, 40), dtype=np.uint8)
    populated[5:15, 5:15] = 255
    polygons = extract_polygons([empty, populated, empty])
    assert len(polygons) == 3
    assert polygons[0].shape == (0, 2)
    assert polygons[1].shape[0] >= 3
    assert polygons[2].shape == (0, 2)


def test_compute_mask_area_empty() -> None:
    """All-zeros mask: area is 0."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert compute_mask_area(mask) == 0


def test_compute_mask_area_single_object() -> None:
    """10x10 square: area is 100."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 40:50] = 255
    assert compute_mask_area(mask) == 100


def test_compute_mask_area_multiple_disjoint_objects() -> None:
    """Two disjoint squares: area is the sum of both."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 255
    mask[50:70, 50:60] = 255
    assert compute_mask_area(mask) == 300


def test_compute_mask_area_full_image() -> None:
    """Full-image mask: area is H*W."""
    mask = np.full((30, 40), 255, dtype=np.uint8)
    assert compute_mask_area(mask) == 30 * 40


def test_compute_mask_area_binary_threshold() -> None:
    """Mask with values > 0 (not just 255): still counts all nonzero pixels."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:5] = 128
    assert compute_mask_area(mask) == 25


def test_get_tight_bbox_empty() -> None:
    """All-zeros mask: bbox is (0, 0, 0, 0)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert get_tight_bbox(mask) == (0, 0, 0, 0)


def test_get_tight_bbox_single_object() -> None:
    """Square at rows 20-30, cols 40-50: bbox is (40, 20, 49, 29)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 40:50] = 255
    assert get_tight_bbox(mask) == (40, 20, 49, 29)


def test_get_tight_bbox_multiple_disjoint_objects() -> None:
    """Two disjoint squares: bbox spans the combined extent."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[80:90, 80:90] = 255
    assert get_tight_bbox(mask) == (10, 10, 89, 89)


def test_get_tight_bbox_full_image() -> None:
    """Full-image mask: bbox is (0, 0, W-1, H-1)."""
    mask = np.full((30, 40), 255, dtype=np.uint8)
    assert get_tight_bbox(mask) == (0, 0, 39, 29)


def test_get_tight_bbox_single_pixel() -> None:
    """Single pixel at (5, 7): bbox is (7, 5, 7, 5) (zero-area but valid)."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 7] = 255
    assert get_tight_bbox(mask) == (7, 5, 7, 5)


import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from img2svg.detector import trace_region  # noqa: E402
from img2svg.svg_builder import SVG_NS  # noqa: E402


class _FakeVtracer:
    """Drop-in replacement for ``VtracerVectorizer``.

    Writes ``<svg><path d="..."/></svg>`` with ``num_paths`` elements
    to ``output_path``. Records every call — including a copy of the
    input PNG bytes, because the real ``trace_region`` runs vtracer
    inside a ``tempfile.TemporaryDirectory`` context that is cleaned up
    before the caller can re-open the input file.
    """

    def __init__(self, preset: str = "default", num_paths: int = 2) -> None:
        self.preset = preset
        self.num_paths = num_paths
        self.calls: list[dict[str, Any]] = []

    def vectorize(self, input_path: str | Path, output_path: str | Path) -> None:
        self.calls.append(
            {
                "input_path": str(input_path),
                "input_bytes": Path(input_path).read_bytes(),
                "output_path": str(output_path),
                "preset": self.preset,
            }
        )
        svg = ET.Element(f"{{{SVG_NS}}}svg")
        for i in range(self.num_paths):
            path_el = ET.SubElement(svg, f"{{{SVG_NS}}}path")
            path_el.set("d", f"M0,0 L{i},{i} Z")
        ET.ElementTree(svg).write(str(output_path), xml_declaration=True, encoding="utf-8")


def _install_fake_vtracer(monkeypatch: pytest.MonkeyPatch, num_paths: int = 2) -> _FakeVtracer:
    """Swap detector.VtracerVectorizer for a fake. Returns the fake instance."""
    fake = _FakeVtracer(num_paths=num_paths)

    def _factory(
        preset: str = "default", params_override: dict[str, Any] | None = None
    ) -> _FakeVtracer:
        fake.preset = preset
        fake.params_override = params_override
        return fake

    monkeypatch.setattr("img2svg.detector.VtracerVectorizer", _factory)
    return fake


def test_trace_region_returns_paths_and_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: 2 vtracer paths + correct (x1, y1) offset."""
    fake = _install_fake_vtracer(monkeypatch, num_paths=2)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 20:50] = 1

    paths, offset = trace_region(img, mask, preset="photo_hifi")

    assert len(paths) == 2
    assert offset == (20, 10)
    assert len(fake.calls) == 1
    assert fake.calls[0]["preset"] == "photo_hifi"


def test_trace_region_empty_mask_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty mask returns ([], (0, 0)) and never invokes vtracer."""
    fake = _install_fake_vtracer(monkeypatch, num_paths=2)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    paths, offset = trace_region(img, mask)

    assert paths == []
    assert offset == (0, 0)
    # Empty mask fast path must skip vtracer entirely.
    assert fake.calls == []


def test_trace_region_single_pixel_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1-pixel mask still produces a valid 1x1 crop and a 2-path trace."""
    fake = _install_fake_vtracer(monkeypatch, num_paths=2)
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 40] = 1

    paths, offset = trace_region(img, mask, preset="logo")

    assert len(paths) == 2
    assert offset == (40, 25)
    assert fake.calls[0]["preset"] == "logo"


def test_trace_region_default_preset_is_photo_hifi(monkeypatch: pytest.MonkeyPatch) -> None:
    """No preset kwarg → preset='photo_hifi' (the documented default)."""
    fake = _install_fake_vtracer(monkeypatch, num_paths=1)
    img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:5, 0:5] = 1

    paths, offset = trace_region(img, mask)

    assert len(paths) == 1
    assert offset == (0, 0)
    assert fake.calls[0]["preset"] == "photo_hifi"


def test_trace_region_crops_around_mask_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the crop on disk is tight to the bbox (no padding rows/cols)."""
    from io import BytesIO

    from PIL import Image

    fake = _install_fake_vtracer(monkeypatch, num_paths=1)
    img = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
    mask = np.zeros((60, 60), dtype=np.uint8)
    # x1=30, y1=10, x2=49, y2=24 (inclusive)
    mask[10:25, 30:50] = 1

    paths, offset = trace_region(img, mask, preset="default")
    assert offset == (30, 10)
    assert len(paths) == 1

    saved = np.asarray(Image.open(BytesIO(fake.calls[0]["input_bytes"])).convert("RGBA"))
    # Bbox inclusive: x in [30, 49], y in [10, 24] → 20 wide, 15 tall.
    assert saved.shape == (15, 20, 4)


def test_trace_region_return_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return value is exactly (list[str], tuple[int, int])."""
    _install_fake_vtracer(monkeypatch, num_paths=1)
    img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:5, 0:5] = 1

    paths, offset = trace_region(img, mask)
    assert isinstance(paths, list)
    assert all(isinstance(p, str) for p in paths)
    assert isinstance(offset, tuple)
    assert len(offset) == 2
    assert all(isinstance(v, int) for v in offset)


def test_real_photo_path_exists(real_photo_path: Path) -> None:
    """Smoke test: real_photo_path fixture returns a valid image file."""
    assert real_photo_path.exists()
    assert real_photo_path.stat().st_size > 50_000  # > 50KB
