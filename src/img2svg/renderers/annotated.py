"""AnnotatedRenderer: combines vtracer trace (background) with detection overlays (foreground).

Used by the `ANNOTATED` mode (and is the default for `Mode.AUTO` for the
pipeline's composite output). Produces an SVG with two layers:

1. **Background** — a single `<g id="vtracer-output">` group populated by
   the shared vtracer helper (`_render_with_vtracer` with the 'default'
   preset), placed immediately after the base `<title>`/`<desc>`/`<style>`
   children.
2. **Foreground** — one `<g id="det_{class_name}_{idx}">` group per
   detection, each containing a stroked `<rect>` for the bounding box and
   an outlined `<text>` for the label, added in detection order so they
   render on top of the vtracer trace.

The detection groups are appended after the vtracer group, so SVG paint
order places them above the trace. An empty detections list produces the
trace-only background with no detection groups.
"""
from __future__ import annotations

from img2svg.logging import get_logger
from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer

_logger = get_logger("img2svg.renderers.annotated")

_BOX_STROKE = "black"
_BOX_STROKE_WIDTH = 2.0
_TEXT_FONT_SIZE = 14.0
_TEXT_FILL = "white"
_TEXT_STROKE = "black"
_TEXT_STROKE_WIDTH = 3.0
_TEXT_OFFSET_Y = 4.0


class AnnotatedRenderer(Renderer):
    """Renderer that overlays detection labels on top of a vtracer trace.

    The output SVG structure is:

    ```
    <svg>
      <title/><desc/><style/>
      <g id="vtracer-output">…</g>     <!-- background trace -->
      <g id="det_{class_name}_0">      <!-- detection overlays (foreground) -->
        <rect …/>
        <text …/>
      </g>
      <g id="det_{class_name}_1">…</g>
      …
    </svg>
    ```

    The vtracer group is the first user-content element; detection groups
    follow in detection order so they paint on top via SVG's natural
    document order.
    """

    preset_name: str = "default"

    def render(self) -> None:
        """Mutate `self.svg` in place: vtracer trace + per-detection overlays."""
        _logger.debug(
            "AnnotatedRenderer: %d detection(s) for %dx%d image",
            len(self.detections), self.image.width, self.image.height,
        )

        _render_with_vtracer(self, self.preset_name)

        for idx, det in enumerate(self.detections):
            label = f"{det.class_name} {det.confidence:.2f}"
            group = self.svg.add_group(
                id=f"det_{det.class_name}_{idx}",
                **{"data-class": det.class_name, "data-conf": str(det.confidence)},
            )
            bb = det.bbox
            group.add_rect(
                x=bb.x1, y=bb.y1,
                w=bb.width, h=bb.height,
                fill="none", stroke=_BOX_STROKE, stroke_width=_BOX_STROKE_WIDTH,
            )
            text_y = max(0.0, bb.y1 - _TEXT_OFFSET_Y)
            group.add_text_with_outline(
                x=bb.x1, y=text_y, text=label,
                font_size=_TEXT_FONT_SIZE,
                fill=_TEXT_FILL, stroke=_TEXT_STROKE, stroke_width=_TEXT_STROKE_WIDTH,
            )
