# img2svg - EdgeRenderer: traces the image with the 'bw_edge' vtracer preset.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""EdgeRenderer: traces the image with the 'bw_edge' preset.

Binary colormode + polygon (not spline) mode. Produces line-art style SVGs
with no fills — just outlines. Best paired with pre-processing
(median + Canny) to extract clean edges from photos. The shared
`_render_with_vtracer` helper in `img2svg.renderers.visual` is reused
unchanged; this subclass only fixes the preset name and documents the
intended use case.
"""

from __future__ import annotations

from typing import ClassVar

from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer


class EdgeRenderer(Renderer):
    """Renderer that traces the image with the 'bw_edge' vtracer preset.

    The 'bw_edge' preset uses binary colormode with polygon (not spline) mode,
    producing line-art style SVGs that contain only outlines — no filled
    regions. Best paired with pre-processing (median + Canny) to extract
    clean edges from photos.
    """

    preset_name: ClassVar[str] = "bw_edge"

    def render(self) -> None:
        """Trace the image and embed vtracer paths as a background group."""
        _render_with_vtracer(self, self.preset_name)
