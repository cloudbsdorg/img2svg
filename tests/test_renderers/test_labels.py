# img2svg - tests for `LabelsRenderer`.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `LabelsRenderer`."""

from __future__ import annotations

from PIL import Image

from img2svg.loader import LoadedImage
from img2svg.models import BoundingBox, Detection
from img2svg.renderers.labels import LabelsRenderer
from img2svg.svg_builder import SVG_NS, SVGDocument


def _make_image(width: int = 200, height: int = 200) -> LoadedImage:
    pil = Image.new("RGB", (width, height), color="white")
    return LoadedImage(
        pil_image=pil,
        format="PNG",
        original_mode="RGB",
        has_alpha=False,
        width=width,
        height=height,
    )


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


def _det_groups(root):
    """Return `<g>` elements whose id starts with `det_`."""
    return [g for g in root.findall(f"{{{SVG_NS}}}g") if (g.get("id") or "").startswith("det_")]


def test_empty_detections_produces_background_only() -> None:
    svg = SVGDocument(200, 200)
    renderer = LabelsRenderer(svg, _make_image(), [], None)
    renderer.render()

    root = _parse(svg)
    rects = root.findall(f".//{{{SVG_NS}}}rect")
    assert len(rects) == 1
    bg = rects[0]
    assert bg.get("width") == "200.0"
    assert bg.get("height") == "200.0"
    assert bg.get("fill") == "white"
    assert _det_groups(root) == []


def test_single_detection_produces_one_group_with_rect_and_text() -> None:
    svg = SVGDocument(400, 300)
    det = _det("person", 0.87, 10.0, 20.0, 110.0, 220.0)
    renderer = LabelsRenderer(svg, _make_image(400, 300), [det], None)
    renderer.render()

    root = _parse(svg)
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


def test_three_detections_same_class_produces_three_groups() -> None:
    svg = SVGDocument(500, 500)
    dets = [
        _det("cat", 0.90, 10.0, 10.0, 50.0, 50.0),
        _det("cat", 0.80, 60.0, 60.0, 120.0, 120.0),
        _det("cat", 0.70, 130.0, 130.0, 200.0, 200.0),
    ]
    renderer = LabelsRenderer(svg, _make_image(500, 500), dets, None)
    renderer.render()

    root = _parse(svg)
    groups = _det_groups(root)
    assert len(groups) == 3
    ids = [g.get("id") for g in groups]
    assert ids == ["det_cat_0", "det_cat_1", "det_cat_2"]


def test_group_ids_follow_det_class_idx_pattern() -> None:
    svg = SVGDocument(640, 480)
    dets = [
        _det("person", 0.95, 0.0, 0.0, 100.0, 100.0),
        _det("dog", 0.88, 100.0, 100.0, 200.0, 200.0),
        _det("cell phone", 0.75, 200.0, 200.0, 250.0, 250.0),
    ]
    renderer = LabelsRenderer(svg, _make_image(640, 480), dets, None)
    renderer.render()

    root = _parse(svg)
    groups = _det_groups(root)
    assert len(groups) == 3
    for g, det in zip(groups, dets):
        expected_id = f"det_{det.class_name}_{dets.index(det)}"
        assert g.get("id") == expected_id
        assert g.get("data-class") == det.class_name
        assert g.get("data-conf") == str(det.confidence)
