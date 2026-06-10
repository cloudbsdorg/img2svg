# img2svg - tests for `VisualRenderer` (vtracer default preset).
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `VisualRenderer` (vtracer default preset)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from img2svg.loader import LoadedImage
from img2svg.renderers.base import Renderer
from img2svg.renderers.visual import VisualRenderer
from img2svg.svg_builder import SVG_NS, SVGDocument

# A synthetic vtracer-style SVG output. vtracer writes the trace as a
# bare `<svg>` root with `<path>` children; this is the structure we
# expect to parse and embed.
_KNOWN_VTRACER_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <path d="M0 0 L10 10" fill="#ff0000"/>
  <path d="M5 5 L15 15" fill="#00ff00"/>
</svg>
"""


def _make_image(width: int = 20, height: int = 20) -> LoadedImage:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[5:15, 5:15] = [255, 0, 0]
    pil = Image.fromarray(arr)
    return LoadedImage(
        pil_image=pil,
        format="PNG",
        original_mode="RGB",
        has_alpha=False,
        width=width,
        height=height,
    )


def _make_doc() -> SVGDocument:
    return SVGDocument(100, 100, title="t", desc="d")


def _fake_vectorize(svg_body: str):
    """Return a `vectorize` side_effect that writes svg_body to the output path."""

    def _side_effect(inp: str, outp: str) -> None:
        Path(outp).write_text(svg_body, encoding="utf-8")

    return _side_effect


def _vtracer_group(root):
    """Return the `<g id="vtracer-output">` element, or None."""
    for g in root.findall(f".//{{{SVG_NS}}}g"):
        if g.get("id") == "vtracer-output":
            return g
    return None


def test_visual_renderer_is_subclass_of_renderer() -> None:
    assert issubclass(VisualRenderer, Renderer)


def test_visual_renderer_uses_default_preset() -> None:
    assert VisualRenderer.preset_name == "default"


def test_render_embeds_vtracer_output_group() -> None:
    svg = _make_doc()
    renderer = VisualRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    from lxml import etree

    root = etree.fromstring(svg.to_string().encode("utf-8"))
    assert _vtracer_group(root) is not None, "vtracer-output group missing from SVG"


def test_render_copies_vtracer_paths_into_group() -> None:
    svg = _make_doc()
    renderer = VisualRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    from lxml import etree

    root = etree.fromstring(svg.to_string().encode("utf-8"))
    group = _vtracer_group(root)
    assert group is not None
    paths = group.findall(f"{{{SVG_NS}}}path")
    assert len(paths) == 2
    assert paths[0].get("d") == "M0 0 L10 10"
    assert paths[0].get("fill") == "#ff0000"
    assert paths[1].get("d") == "M5 5 L15 15"
    assert paths[1].get("fill") == "#00ff00"


def test_render_calls_vtracer_vectorizer_with_default_preset() -> None:
    svg = _make_doc()
    renderer = VisualRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    MockVec.assert_called_once_with(preset="default")
    MockVec.return_value.vectorize.assert_called_once()
    args = MockVec.return_value.vectorize.call_args.args
    assert str(args[0]).endswith(".png")
    assert str(args[1]).endswith(".svg")


def test_vtracer_output_group_is_after_title_and_desc() -> None:
    """The `<g id="vtracer-output">` must appear as the first user element,
    immediately after the contiguous title/desc/style base children."""
    svg = _make_doc()
    renderer = VisualRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    from lxml import etree

    root = etree.fromstring(svg.to_string().encode("utf-8"))
    children = list(root)

    vtracer_idx = None
    for i, c in enumerate(children):
        if c.get("id") == "vtracer-output":
            vtracer_idx = i
            break
    assert vtracer_idx is not None, "vtracer-output group not present"

    # Every element preceding the vtracer-output group must be one of
    # the base elements (title / desc / style).
    base_tags = {"title", "desc", "style"}
    for i in range(vtracer_idx):
        tag = etree.QName(children[i].tag).localname
        assert tag in base_tags, f"unexpected element {tag!r} before vtracer-output group"
