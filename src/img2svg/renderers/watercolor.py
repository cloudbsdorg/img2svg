# img2svg - WatercolorRenderer: traces the image with the 'watercolor' preset.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""WatercolorRenderer: traces the image with the 'watercolor' preset.

Spline mode with low splice_threshold=20 and low corner_threshold=20 for soft,
organic, painterly curves. Best paired with pre-processing (bilateral + CLAHE).
"""

from __future__ import annotations

from typing import ClassVar

from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer


class WatercolorRenderer(Renderer):
    """Renderer that traces the image with vtracer's 'watercolor' preset.

    The 'watercolor' preset is tuned for soft, organic, painterly curves
    suitable for watercolor-style images. Best paired with pre-processing
    (bilateral + CLAHE).
    """

    preset_name: ClassVar[str] = "watercolor"

    def render(self) -> None:
        """Trace the image and embed vtracer paths as a background group."""
        _render_with_vtracer(self, self.preset_name)
