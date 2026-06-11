# img2svg - LabelsRenderer: pure semantic SVG output (white background + bbox + label text).
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""`LabelsRenderer` — pure semantic SVG output: white background + bbox + label text.

Used by the `LABELS` mode. Produces an SVG with no rasterized image, just
a solid white background and labeled bounding boxes drawn over it.
"""

from __future__ import annotations

from img2svg.logging import get_logger
from img2svg.renderers.base import Renderer

_logger = get_logger("img2svg.renderers.labels")

_BACKGROUND_FILL = "white"
_BOX_STROKE = "black"
_BOX_STROKE_WIDTH = 2.0
_TEXT_FONT_SIZE = 14.0
_TEXT_FILL = "white"
_TEXT_STROKE = "black"
_TEXT_STROKE_WIDTH = 3.0
_TEXT_OFFSET_Y = 4.0


class LabelsRenderer(Renderer):
    """Render a pure-semantic SVG: background + labeled bounding boxes.

    The output contains:
    - A full-size white `<rect>` covering the image.
    - One `<g id="det_{class_name}_{idx}">` per detection containing:
      - A stroked `<rect>` (no fill) for the bounding box.
      - A `<text>` with the class name and confidence, with a black outline
        (paint-order="stroke") for readability.
    """

    def render(self) -> None:
        """Mutate `self.svg` in place: add background rect + detection groups."""
        w, h = self.image.width, self.image.height
        _logger.debug(
            "LabelsRenderer: %d detection(s) for %dx%d image",
            len(self.detections),
            w,
            h,
        )

        bg = self.svg.add_group(id="background")
        bg.add_rect(
            x=0.0,
            y=0.0,
            w=float(w),
            h=float(h),
            fill=_BACKGROUND_FILL,
            stroke="none",
            stroke_width=None,
        )

        for idx, det in enumerate(self.detections):
            label = f"{det.class_name} {det.confidence:.2f}"
            group = self.svg.add_group(
                id=f"det_{det.class_name}_{idx}",
                **{"data-class": det.class_name, "data-conf": str(det.confidence)},
            )
            bb = det.bbox
            group.add_rect(
                x=bb.x1,
                y=bb.y1,
                w=bb.width,
                h=bb.height,
                fill="none",
                stroke=_BOX_STROKE,
                stroke_width=_BOX_STROKE_WIDTH,
            )
            text_y = max(0.0, bb.y1 - _TEXT_OFFSET_Y)
            group.add_text_with_outline(
                x=bb.x1,
                y=text_y,
                text=label,
                font_size=_TEXT_FONT_SIZE,
                fill=_TEXT_FILL,
                stroke=_TEXT_STROKE,
                stroke_width=_TEXT_STROKE_WIDTH,
            )
