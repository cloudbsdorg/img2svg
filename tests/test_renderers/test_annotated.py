# img2svg - tests for `AnnotatedRenderer` (vtracer trace + detection overlays).
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `AnnotatedRenderer` (vtracer trace + detection overlays)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from img2svg.loader import LoadedImage
from img2svg.models import BoundingBox, Detection
from img2svg.renderers.annotated import AnnotatedRenderer
from img2svg.renderers.base import Renderer
from img2svg.svg_builder import SVG_NS, SVGDocument

_KNOWN_VTRACER_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <path d="M0 0 L10 10" fill="#ff0000"/>
  <path d="M5 5 L15 15" fill="#00ff00"/>
</svg>
"""


def _make_image(width: int = 200, height: int = 200) -> LoadedImage:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
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
    return SVGDocument(200, 200, title="t", desc="d")


def _fake_vectorize(svg_body: str):
    """Return a `vectorize` side_effect that writes svg_body to the output path."""

    def _side_effect(inp: str, outp: str) -> None:
        Path(outp).write_text(svg_body, encoding="utf-8")

    return _side_effect


def _det(class_name: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _parse(svg: SVGDocument):
    from lxml import etree

    return etree.fromstring(svg.to_string().encode("utf-8"))


def _vtracer_group(root):
    """Return the `<g id="vtracer-output">` element, or None."""
    for g in root.findall(f".//{{{SVG_NS}}}g"):
        if g.get("id") == "vtracer-output":
            return g
    return None


def _det_groups(root):
    """Return `<g>` elements whose id starts with `det_`."""
    return [g for g in root.findall(f"{{{SVG_NS}}}g") if (g.get("id") or "").startswith("det_")]


def test_annotated_renderer_is_subclass_of_renderer() -> None:
    assert issubclass(AnnotatedRenderer, Renderer)


def test_annotated_renderer_uses_default_preset() -> None:
    assert AnnotatedRenderer.preset_name == "default"


def test_empty_detections_has_vtracer_but_no_detection_groups() -> None:
    svg = _make_doc()
    renderer = AnnotatedRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    root = _parse(svg)
    assert _vtracer_group(root) is not None, "vtracer-output group missing from SVG"
    assert _det_groups(root) == [], "no detection groups expected for empty detections"


def test_single_detection_has_vtracer_and_one_detection_group() -> None:
    svg = _make_doc()
    det = _det("person", 0.87, 10.0, 20.0, 110.0, 220.0)
    renderer = AnnotatedRenderer(svg, _make_image(), [det], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    root = _parse(svg)
    assert _vtracer_group(root) is not None
    groups = _det_groups(root)
    assert len(groups) == 1
    g = groups[0]
    assert g.get("id") == "det_person_0"
    assert g.get("data-class") == "person"
    assert g.get("data-conf") == "0.87"

    rects = g.findall(f"{{{SVG_NS}}}rect")
    texts = g.findall(f"{{{SVG_NS}}}text")
    assert len(rects) == 1
    assert len(texts) == 1
    assert rects[0].get("stroke") == "black"
    assert rects[0].get("fill") == "none"
    assert texts[0].text == "person 0.87"
    assert texts[0].get("paint-order") == "stroke"


def test_two_detections_both_groups_appear_after_vtracer_group() -> None:
    svg = _make_doc()
    dets = [
        _det("cat", 0.90, 10.0, 10.0, 50.0, 50.0),
        _det("dog", 0.80, 60.0, 60.0, 120.0, 120.0),
    ]
    renderer = AnnotatedRenderer(svg, _make_image(), dets, None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    root = _parse(svg)
    assert _vtracer_group(root) is not None

    groups = _det_groups(root)
    assert len(groups) == 2
    assert [g.get("id") for g in groups] == ["det_cat_0", "det_dog_1"]

    children = list(root)
    vtracer_idx = next(i for i, c in enumerate(children) if c.get("id") == "vtracer-output")
    det_indices = [i for i, c in enumerate(children) if (c.get("id") or "").startswith("det_")]
    assert len(det_indices) == 2
    for di in det_indices:
        assert di > vtracer_idx, (
            f"detection group at index {di} appears before vtracer-output at {vtracer_idx}"
        )


def test_vtracer_paths_preserved_inside_vtracer_output_group() -> None:
    """The trace group from `_render_with_vtracer` must still contain the
    vtracer `<path>` children after detection groups are added."""
    svg = _make_doc()
    det = _det("person", 0.5, 1.0, 2.0, 3.0, 4.0)
    renderer = AnnotatedRenderer(svg, _make_image(), [det], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    root = _parse(svg)
    group = _vtracer_group(root)
    assert group is not None
    paths = group.findall(f"{{{SVG_NS}}}path")
    assert len(paths) == 2
    assert paths[0].get("d") == "M0 0 L10 10"
    assert paths[0].get("fill") == "#ff0000"
    assert paths[1].get("d") == "M5 5 L15 15"
    assert paths[1].get("fill") == "#00ff00"


def test_vtracer_vectorizer_called_with_default_preset() -> None:
    """The annotated renderer must invoke vtracer with the 'default' preset,
    matching VisualRenderer behavior."""
    svg = _make_doc()
    renderer = AnnotatedRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    MockVec.assert_called_once_with(preset="default")
    MockVec.return_value.vectorize.assert_called_once()
