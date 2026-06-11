# img2svg - VisualRenderer: traces the image with vtracer's 'default' preset.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""VisualRenderer: traces the image with vtracer's 'default' preset.

Produces an SVG whose primary content is a single `<g id="vtracer-output">`
group placed immediately after the `<title>` / `<desc>` / `<style>` base
elements. All vtracer `<path>` children are deep-copied into that group.

The actual SVG is written to a temporary file because vtracer requires a
filesystem path. vtracer output is then parsed with lxml and the paths
extracted; we never copy vtracer's `<svg>` wrapper or its attributes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from lxml import etree
from PIL import Image

from img2svg.logging import get_logger
from img2svg.renderers.base import Renderer
from img2svg.svg_builder import SVG_NS
from img2svg.vectorizer import VtracerVectorizer

if TYPE_CHECKING:
    from img2svg.svg_builder import SVGDocument

_log = get_logger("img2svg.renderers.visual")

# Base SVGDocument children that must always precede user content.
_BASE_LOCAL_TAGS: frozenset[str] = frozenset({"title", "desc", "style"})

# Group id used to mark the vtracer trace layer.
_VTRACER_GROUP_ID: str = "vtracer-output"


def _move_to_after_base(root: etree._Element, target: etree._Element) -> None:
    """Move `target` to the position immediately after the last base child.

    Base children are contiguous `<title>`, `<desc>`, `<style>` at the start
    of the SVG document. We position the vtracer group so that it is the
    first user-content element (background layer).
    """
    insert_at = 0
    for child in root:
        if etree.QName(child.tag).localname in _BASE_LOCAL_TAGS:
            insert_at += 1
        else:
            break
    root.remove(target)
    root.insert(insert_at, target)


def _embed_vtracer_paths(svg: SVGDocument, vtracer_svg_path: Path) -> None:
    """Parse vtracer's output SVG and embed its `<path>` children into `svg`.

    All paths are placed inside a single `<g id="vtracer-output">` group,
    positioned as the first user-content element.
    """
    tree = etree.parse(str(vtracer_svg_path))
    vtracer_root = tree.getroot()
    vtracer_paths = list(vtracer_root.findall(f"{{{SVG_NS}}}path"))

    group = svg.add_group(id=_VTRACER_GROUP_ID)
    _move_to_after_base(svg.root, group.element)

    for path_el in vtracer_paths:
        # Deep-copy the path so the new tree owns the elements
        group.element.append(etree.fromstring(etree.tostring(path_el)))


def _render_with_vtracer(renderer: Renderer, preset: str) -> None:
    """Run vtracer with the given preset and embed the output paths.

    Shared implementation used by both `VisualRenderer` and `TraceRenderer`.
    Writes the source image to a temporary PNG (vtracer needs a path),
    runs vtracer, and embeds the resulting paths into the renderer's SVG.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_png = td_path / "input.png"
        output_svg = td_path / "output.svg"
        Image.fromarray(renderer.image.np_array).save(str(input_png))
        VtracerVectorizer(
            preset=preset, params_override=renderer._vtracer_params_override
        ).vectorize(input_png, output_svg)
        _embed_vtracer_paths(renderer.svg, output_svg)
    _log.debug("embedded vtracer %r output as %s", preset, _VTRACER_GROUP_ID)


class VisualRenderer(Renderer):
    """Renderer that traces the image with vtracer's 'default' preset.

    The 'default' preset uses color/stacked/spline with balanced filter
    parameters — a good general-purpose trace suitable for diagrams,
    screenshots, and mixed images.
    """

    preset_name: ClassVar[str] = "default"

    def render(self) -> None:
        """Trace the image and embed vtracer paths as a background group."""
        _render_with_vtracer(self, self.preset_name)
