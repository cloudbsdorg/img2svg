# img2svg - PosterRenderer: traces the image with vtracer's 'poster' preset.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""PosterRenderer: traces the image with vtracer's 'poster' preset for stylized, limited-color output. The 'poster' preset has color_precision=8 for high color fidelity with stacked layers."""

from __future__ import annotations

from typing import ClassVar

from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer


class PosterRenderer(Renderer):
    """Renderer that traces the image with vtracer's 'poster' preset.

    The 'poster' preset is tuned for stylized, limited-color output suitable
    for posters, logos, and graphics that benefit from reduced color depth
    with stacked layer differentiation.
    """

    preset_name: ClassVar[str] = "poster"

    def render(self) -> None:
        """Trace the image and embed vtracer paths as a background group."""
        _render_with_vtracer(self, self.preset_name)
