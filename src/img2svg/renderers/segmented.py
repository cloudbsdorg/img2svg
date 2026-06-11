# img2svg - SegmentedRenderer: multi-layer editable SVG with YOLO instance segmentation.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""SegmentedRenderer: produces a multi-layer editable SVG. Each YOLO-detected
object becomes its own ``<g id="obj_class_idx">`` group with semantic class
name and confidence. Background is traced separately. Empty detections
falls back to VisualRenderer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np
from lxml import etree
from PIL import Image

from img2svg.detector import SegmentationResult, trace_region
from img2svg.logging import get_logger
from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import _render_with_vtracer
from img2svg.svg_builder import SVG_NS
from img2svg.vectorizer import VtracerVectorizer

if TYPE_CHECKING:
    from img2svg.loader import LoadedImage
    from img2svg.models import Detection, GeometricAnalysis
    from img2svg.svg_builder import SVGDocument

_log = get_logger("img2svg.renderers.segmented")

_BG_FILL: int = 255
_BG_GROUP_ID: str = "background"
_BG_DATA_ROLE_ATTR: str = "data-role"
_BG_DATA_ROLE_VALUE: str = "background"

_OBJ_ID_PREFIX: str = "obj_"
_OBJ_DATA_CLASS_ATTR: str = "data-class"
_OBJ_DATA_CONF_ATTR: str = "data-conf"


def _has_any_region(result: SegmentationResult | None) -> bool:
    """Return True if result is non-None and has at least one non-empty mask.

    A mask counts as non-empty when at least one pixel is greater than
    zero. YOLO returns (H, W) uint8 arrays with values in {0, 1}
    (retina_masks=True) so a strict ``> 0`` test is the correct semantic
    check for "this detection has pixels".
    """
    if result is None:
        return False
    return any((m > 0).any() for m in result.masks)


def _build_background_mask(
    image_shape: tuple[int, ...], masks: list[np.ndarray]
) -> np.ndarray:
    """Build a uint8 mask of background pixels (255 = keep, 0 = fill).

    Starts as all-255 and subtracts each region mask via ``cv2.subtract``
    so overlapping masks (NMS did not dedupe perfectly) collapse cleanly
    to 0. Using ``cv2.subtract`` over plain numpy subtraction guarantees
    unsigned-int8 saturation behavior -- we never go negative even if a
    mask somehow exceeded 255.
    """
    bg_mask = np.ones(image_shape[:2], dtype=np.uint8) * _BG_FILL
    for mask in masks:
        bg_mask = cv2.subtract(bg_mask, mask)
    return bg_mask


def _embed_background_paths(bg_image: np.ndarray, svg: SVGDocument) -> None:
    """Run vtracer on bg_image and embed the resulting paths as one group.

    Writes bg_image to a temp PNG (vtracer needs a filesystem path), runs
    vtracer with the ``default`` preset, parses the output, and embeds
    the ``<path>`` children in a new
    ``<g id="background" data-role="background">`` group. Mirrors
    :func:`img2svg.renderers.visual._embed_vtracer_paths` but operates on
    an explicit image (the white-filled background) rather than the
    renderer's source image.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_png = td_path / "bg.png"
        output_svg = td_path / "bg.svg"
        Image.fromarray(bg_image).save(str(input_png))
        VtracerVectorizer(preset="default").vectorize(input_png, output_svg)
        tree = etree.parse(str(output_svg))
        root = tree.getroot()
        vtracer_paths = list(root.findall(f"{{{SVG_NS}}}path"))

    group = svg.add_group(
        id=_BG_GROUP_ID, **{_BG_DATA_ROLE_ATTR: _BG_DATA_ROLE_VALUE}
    )
    for path_el in vtracer_paths:
        group.element.append(etree.fromstring(etree.tostring(path_el)))


class SegmentedRenderer(Renderer):
    """Renderer that emits a multi-layer SVG using YOLO instance segmentation.

    Output structure (one ``<g>`` per YOLO detection, plus one for
    background):

    ```
    <svg>
      <title/><desc/><style/>
      <g id="background" data-role="background">...</g>
      <g id="obj_<class>_<idx>" data-class="..." data-conf="...">
        <g transform="translate(x1 y1)">...</g>
      </g>
      ...
    </svg>
    ```

    The pipeline calls :meth:`set_segmentation` with the
    :class:`SegmentationResult` produced by :class:`YOLOSegmentor` before
    invoking :meth:`render`. If no segmentation result is set (or every
    mask is empty), :meth:`render` falls back to :class:`VisualRenderer`
    behavior -- a single whole-image vtracer trace.
    """

    preset_name: ClassVar[str] = "default"

    def __init__(
        self,
        svg: SVGDocument,
        image: LoadedImage,
        detections: list[Detection],
        geometric: GeometricAnalysis | None,
    ) -> None:
        """Store render context and initialize the unset segmentation result."""
        super().__init__(svg, image, detections, geometric)
        self._segmentation_result: SegmentationResult | None = None

    def set_segmentation(self, result: SegmentationResult) -> None:
        """Inject a SegmentationResult to drive per-region rendering.

        The pipeline calls this between :class:`YOLOSegmentor` inference
        and :meth:`render` to thread the segmentation masks into the
        renderer without coupling it to YOLO directly.
        """
        self._segmentation_result = result

    def render(self) -> None:
        """Emit the multi-layer SVG, or fall back to whole-image vtracer.

        When ``_segmentation_result`` is missing or all masks are empty,
        delegates to :func:`_render_with_vtracer` for the same whole-image
        trace a :class:`VisualRenderer` would produce.

        Otherwise:

        1. Build a background mask by subtracting each region mask from a
           full-image 255 mask.
        2. White-fill foreground regions in the source image and trace
           the result via vtracer's ``default`` preset.
        3. Embed the background as
           ``<g id="background" data-role="background">``.
        4. For each region (in order of descending confidence):
           - Crop+trace with vtracer's ``photo_hifi`` preset
             (:func:`trace_region`).
           - Embed as ``<g id="obj_{class}_{idx}">`` with an inner
             ``<g transform="translate(...)">`` for path positioning.
           - Skip if the trace produced zero paths.
        """
        result = self._segmentation_result
        if not _has_any_region(result):
            _log.debug("no segmentation regions; falling back to VisualRenderer")
            _render_with_vtracer(self, "default")
            return

        assert result is not None
        image = self.image.np_array

        bg_mask = _build_background_mask(image.shape, result.masks)
        bg_image = image.copy()
        bg_image[bg_mask == 0] = _BG_FILL

        _embed_background_paths(bg_image, self.svg)

        order = sorted(
            range(len(result.boxes)),
            key=lambda i: result.boxes[i].confidence,
            reverse=True,
        )
        for idx, i in enumerate(order):
            det = result.boxes[i]
            mask = result.masks[i]
            paths, offset = trace_region(image, mask, preset="photo_hifi")
            if not paths:
                _log.debug(
                    "skipping empty region: class=%s idx=%d (0 paths)",
                    det.class_name,
                    i,
                )
                continue
            outer = self.svg.add_group(
                id=f"{_OBJ_ID_PREFIX}{det.class_name}_{idx}",
                **{
                    _OBJ_DATA_CLASS_ATTR: det.class_name,
                    _OBJ_DATA_CONF_ATTR: str(det.confidence),
                },
            )
            inner_el = etree.SubElement(
                outer.element,
                f"{{{SVG_NS}}}g",
                transform=f"translate({offset[0]} {offset[1]})",
            )
            for d_attr in paths:
                path_el = etree.SubElement(inner_el, f"{{{SVG_NS}}}path")
                path_el.set("d", d_attr)
