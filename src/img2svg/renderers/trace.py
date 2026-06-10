# img2svg - TraceRenderer: traces the image with vtracer's 'photo' preset.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""TraceRenderer: traces the image with vtracer's 'photo' preset.

Identical to `VisualRenderer` except that vtracer is invoked with the
'photo' preset, which has lower `filter_speckle` and `layer_difference`
values for higher-fidelity photorealistic traces.
"""

from __future__ import annotations

from typing import ClassVar

from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer


class TraceRenderer(Renderer):
    """Renderer that traces the image with vtracer's 'photo' preset.

    The 'photo' preset is tuned for smooth, high-color-fidelity output
    suitable for raster photographs and complex images with many subtle
    color regions.
    """

    preset_name: ClassVar[str] = "photo"

    def render(self) -> None:
        """Trace the image and embed vtracer paths as a background group."""
        _render_with_vtracer(self, self.preset_name)
