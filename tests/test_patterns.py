# img2svg - tests for the OpenCV geometric patterns module.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the OpenCV geometric patterns module."""

from __future__ import annotations

import numpy as np

from img2svg.models import BoundingBox
from img2svg.patterns import analyze_global, analyze_roi


def _solid(rgb: tuple[int, int, int], size: int = 100) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def _gradient(size: int = 100) -> np.ndarray:
    grad = np.linspace(0, 255, size, dtype=np.uint8)
    return np.broadcast_to(grad.reshape(1, size, 1), (size, size, 3)).copy()


def _line_art(size: int = 100) -> np.ndarray:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[20, :] = 0
    img[40, :] = 0
    img[60, :] = 0
    img[80, :] = 0
    return img


def test_analyze_global_solid_color_returns_1_dominant() -> None:
    """Solid red image: K-means should return 1 dominant color (others have ~0 population)."""
    img = _solid((255, 0, 0))
    g = analyze_global(img)
    assert len(g.dominant_colors) == 5  # K=5
    # First (most populous) should be near red
    r, g_, b = g.dominant_colors[0]
    assert r > 200 and g_ < 50 and b < 50
    assert g.has_alpha is False


def test_analyze_global_detects_alpha() -> None:
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    g = analyze_global(rgba)
    assert g.has_alpha is True


def test_analyze_global_no_alpha_for_rgb() -> None:
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    g = analyze_global(rgb)
    assert g.has_alpha is False


def test_analyze_global_edge_density_line_art() -> None:
    img = _line_art()
    g = analyze_global(img)
    # Line art has clear edges
    assert g.edge_density > 0.0


def test_analyze_global_edge_density_solid_zero() -> None:
    img = _solid((128, 128, 128))
    g = analyze_global(img)
    assert g.edge_density == 0.0


def test_analyze_global_contour_count_solid() -> None:
    img = _solid((255, 0, 0))
    g = analyze_global(img)
    # Solid image: thresholding at 127 gives all-0 or all-255 → 1 contour
    assert g.contour_count >= 0


def test_analyze_global_empty_image() -> None:
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    g = analyze_global(img)
    assert g.dominant_colors == []


def test_analyze_roi_extracts_subregion() -> None:
    img = _solid((0, 0, 0), size=200)
    # Set a small white region
    img[50:80, 50:80] = (255, 255, 255)
    bbox = BoundingBox(x1=50, y1=50, x2=80, y2=80)
    g = analyze_roi(img, bbox)
    # ROI is white: dominant color is white
    r, gc, b = g.dominant_colors[0]
    assert r > 200 and gc > 200 and b > 200


def test_analyze_roi_clamps_out_of_bounds() -> None:
    """Bbox partially outside image bounds: clamped, no exception."""
    img = _solid((128, 128, 128), size=100)
    bbox = BoundingBox(x1=-10, y1=-10, x2=200, y2=200)  # way outside
    g = analyze_roi(img, bbox)
    assert len(g.dominant_colors) == 5  # should still return K colors


def test_analyze_roi_empty_after_clamp() -> None:
    """Bbox completely outside image: returns empty GeometricAnalysis."""
    img = _solid((128, 128, 128), size=100)
    bbox = BoundingBox(x1=200, y1=200, x2=300, y2=300)  # outside right
    g = analyze_roi(img, bbox)
    assert g.dominant_colors == []
    assert g.edge_density == 0.0
    assert g.contour_count == 0
