# img2svg - geometric (non-ML) image analysis using OpenCV.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Geometric (non-ML) image analysis using OpenCV.

Provides:
- analyze_global(image) for whole-image stats
- analyze_roi(image, bbox) for per-detection analysis
"""

from __future__ import annotations

import cv2
import numpy as np

from img2svg.models import BoundingBox, GeometricAnalysis


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 (H, W, 3) to LAB (H, W, 3) float32."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)


def _kmeans_dominant_colors(
    image: np.ndarray, k: int = 5, max_iter: int = 10
) -> list[tuple[int, int, int]]:
    """K-means in LAB color space, returning K RGB centroids sorted by population desc."""
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return []
    h, w = image.shape[:2]
    if image.shape[-1] == 4:
        rgb_image = image[..., :3]
    else:
        rgb_image = image
    pixels = rgb_image.reshape((-1, 3))
    if pixels.shape[0] < k:
        k = max(1, pixels.shape[0])
    pixels_f = pixels.astype(np.float32)
    if pixels_f.shape[1] == 3 and rgb_image.shape[-1] == 3:
        # Convert to LAB for perceptual clustering
        try:
            img_rgb = pixels.reshape((h, w, 3)).astype(np.uint8)
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            pixels_f = lab.reshape((-1, 3)).astype(np.float32)
        except cv2.error:
            pass  # fall through with RGB
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    _, labels, centers = cv2.kmeans(pixels_f, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    # Sort by count desc
    order = np.argsort(-counts)
    out: list[tuple[int, int, int]] = []
    for idx in order:
        center = centers[idx]
        if center.shape[0] == 3 and rgb_image.shape[-1] == 3:
            try:
                lab_arr = np.zeros((1, 1, 3), dtype=np.uint8)
                lab_arr[0, 0] = [int(max(0, min(255, c))) for c in center]
                rgb = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0, 0]
                out.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
            except cv2.error:
                out.append((int(center[0]), int(center[1]), int(center[2])))
        else:
            out.append((int(center[0]), int(center[1]), int(center[2])))
    return out


def _canny_edge_density(image: np.ndarray) -> float:
    """Fraction of pixels that are Canny edge pixels."""
    if image.size == 0:
        return 0.0
    if image.shape[-1] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return float(edges.astype(bool).sum()) / float(edges.size)


def _contour_count(image: np.ndarray) -> int:
    """Number of significant contours found in the image."""
    if image.size == 0:
        return 0
    if image.shape[-1] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter tiny noise
    return sum(1 for c in contours if cv2.contourArea(c) >= 25)


def analyze_global(image: np.ndarray) -> GeometricAnalysis:
    """Whole-image geometric analysis.

    Args:
        image: RGB uint8 array (H, W, 3) or RGBA uint8 (H, W, 4).

    Returns:
        GeometricAnalysis with dominant_colors (K=5), edge_density,
        contour_count, and has_alpha.
    """
    if image.size == 0:
        return GeometricAnalysis(
            dominant_colors=[], has_alpha=image.shape[-1] == 4 if image.ndim == 3 else False
        )
    has_alpha = image.ndim == 3 and image.shape[-1] == 4
    colors = _kmeans_dominant_colors(image, k=5)
    edges = _canny_edge_density(image)
    contours = _contour_count(image)
    return GeometricAnalysis(
        dominant_colors=colors,
        edge_density=edges,
        contour_count=contours,
        has_alpha=has_alpha,
    )


def analyze_roi(image: np.ndarray, bbox: BoundingBox) -> GeometricAnalysis:
    """Per-ROI geometric analysis. Extracts the region and runs the same analysis."""
    h, w = image.shape[:2]
    x1 = max(0, min(w, int(round(bbox.x1))))
    y1 = max(0, min(h, int(round(bbox.y1))))
    x2 = max(0, min(w, int(round(bbox.x2))))
    y2 = max(0, min(h, int(round(bbox.y2))))
    if x2 <= x1 or y2 <= y1:
        return GeometricAnalysis()
    sub = image[y1:y2, x1:x2]
    return analyze_global(sub)
