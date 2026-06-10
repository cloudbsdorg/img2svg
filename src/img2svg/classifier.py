# img2svg - heuristic image type classifier.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Heuristic image type classifier.

Maps an image to one of: LOGO, PHOTO, DIAGRAM, SCREENSHOT, LINE_ART, UNKNOWN.
The heuristic is based on a pre-computed GeometricAnalysis:
  - has_alpha and few colors and many edges     → LOGO
  - many colors and few edges                    → PHOTO
  - many edges and not alpha (low color count)   → DIAGRAM or LINE_ART
  - medium edges, geometric blocks               → SCREENSHOT
  - else                                         → UNKNOWN
"""

from __future__ import annotations

import numpy as np

from img2svg.enums import ImageType
from img2svg.models import GeometricAnalysis
from img2svg.patterns import analyze_global


def classify(image: np.ndarray, analysis: GeometricAnalysis | None = None) -> tuple[ImageType, str]:
    """Classify an image and return (image_type, reasoning_string).

    Args:
        image: RGB or RGBA uint8 numpy array.
        analysis: optional pre-computed GeometricAnalysis. If None, computed.

    Returns:
        Tuple of (ImageType, human-readable reasoning).
    """
    if analysis is None:
        analysis = analyze_global(image)

    n_colors = len([c for c in analysis.dominant_colors if any(v > 0 for v in c)])
    edges = analysis.edge_density
    contours = analysis.contour_count
    has_alpha = analysis.has_alpha

    reasoning_parts: list[str] = []

    # Heuristic rules, ordered by specificity.
    if has_alpha and n_colors <= 3 and edges > 0.3:
        return (
            ImageType.LOGO,
            f"has_alpha + {n_colors} dominant colors + edges={edges:.2f} > 0.3 → LOGO",
        )
    if n_colors >= 4 and edges < 0.1:
        return (
            ImageType.PHOTO,
            f"{n_colors} dominant colors + edges={edges:.2f} < 0.1 → PHOTO",
        )
    if n_colors <= 3 and edges > 0.4:
        return (
            ImageType.LINE_ART,
            f"{n_colors} dominant colors + edges={edges:.2f} > 0.4 → LINE_ART",
        )
    if edges > 0.2 and not has_alpha and n_colors <= 5:
        return (
            ImageType.DIAGRAM,
            f"edges={edges:.2f} > 0.2 + no alpha + {n_colors} colors → DIAGRAM",
        )
    if 5 <= n_colors <= 8 and 0.1 <= edges <= 0.3 and contours >= 4:
        return (
            ImageType.SCREENSHOT,
            f"{n_colors} colors + edges={edges:.2f} in [0.1, 0.3] + {contours} contours → SCREENSHOT",
        )
    return (
        ImageType.UNKNOWN,
        f"no heuristic matched (colors={n_colors}, edges={edges:.2f}, alpha={has_alpha}) → UNKNOWN",
    )
