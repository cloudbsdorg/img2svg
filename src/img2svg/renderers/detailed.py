# img2svg - DetailedRenderer: traces the image with the new 'photo_hifi' preset.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""DetailedRenderer: traces the image with the new 'photo_hifi' preset — high
color_precision=8, low filter_speckle=4, fine path_precision=4,
max_iterations=20. Designed to be paired with the pre-processing pipeline
(bilateral + unsharp) for maximum photo fidelity.
"""

from __future__ import annotations

from typing import ClassVar

from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer


class DetailedRenderer(Renderer):
    """Renderer that traces the image with vtracer's 'photo_hifi' preset.

    The 'photo_hifi' preset is tuned for maximum photographic fidelity:
    high color_precision=8 (8-bit color quantization), low filter_speckle=4
    (retain small features), fine path_precision=4 (sub-pixel path
    coordinates), and max_iterations=20 (deep color clustering). Pair with
    the pre-processing pipeline (bilateral + unsharp) for the best results
    on complex photographs.
    """

    preset_name: ClassVar[str] = "photo_hifi"

    def render(self) -> None:
        """Trace the image and embed vtracer paths as a background group."""
        _render_with_vtracer(self, self.preset_name)
