# img2svg - vtracer wrapper for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""vtracer wrapper for img2svg.

Provides a clean Python API around `vtracer.convert_image_to_svg_py()` with
named presets for the four output modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import vtracer

from img2svg.errors import VectorizationError

Preset = Literal["default", "bw", "logo", "poster", "photo"]

# Each preset maps to vtracer's keyword args.
PRESETS: dict[str, dict] = {
    "default": {
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 4,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3,
    },
    "bw": {
        "colormode": "binary",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 4,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3,
    },
    "logo": {
        # Sharp edges, high color precision for clean geometric shapes.
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 8,
        "color_precision": 8,
        "layer_difference": 24,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3,
    },
    "poster": {
        # Flat color regions, like a poster print.
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 4,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3,
    },
    "photo": {
        # Smooth, high color precision for photorealistic trace.
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 2,
        "color_precision": 6,
        "layer_difference": 8,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3,
    },
}


class VtracerVectorizer:
    """Wrapper around vtracer with named presets."""

    def __init__(self, preset: Preset = "default") -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset: {preset!r}; valid: {list(PRESETS)}")
        self.preset = preset
        self.params = dict(PRESETS[preset])

    def vectorize(self, input_path: str | Path, output_path: str | Path) -> None:
        """Convert `input_path` to an SVG at `output_path` using this preset.

        Raises VectorizationError on any vtracer failure.
        """
        try:
            vtracer.convert_image_to_svg_py(
                str(input_path),
                str(output_path),
                **self.params,
            )
        except Exception as e:
            raise VectorizationError(str(input_path), original=e) from e
