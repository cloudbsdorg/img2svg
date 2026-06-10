# img2svg - tests for the SVG builder.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the SVG builder."""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from img2svg.svg_builder import SVG_NS, SVGDocument


def test_minimal_document_parses() -> None:
    doc = SVGDocument(100, 100)
    s = doc.to_string()
    assert "svg" in s
    root = etree.fromstring(s.encode("utf-8"))
    assert root.tag == f"{{{SVG_NS}}}svg"
    assert root.get("viewBox") == "0 0 100 100"
    assert root.get("width") == "100"
    assert root.get("height") == "100"


def test_title_and_desc_included() -> None:
    doc = SVGDocument(50, 50, title="my image", desc="a description")
    s = doc.to_string()
    root = etree.fromstring(s.encode("utf-8"))
    # First two children should be title and desc
    children = list(root)
    titles = root.findall(f"{{{SVG_NS}}}title")
    descs = root.findall(f"{{{SVG_NS}}}desc")
    assert len(titles) == 1 and titles[0].text == "my image"
    assert len(descs) == 1 and descs[0].text == "a description"
    # Title is first child for accessibility
    assert children[0].tag == f"{{{SVG_NS}}}title"


def test_add_group_creates_g_with_id() -> None:
    doc = SVGDocument(100, 100)
    g = doc.add_group(id="det_person_0", **{"data-class": "person", "data-conf": "0.95"})
    assert g.element.tag == f"{{{SVG_NS}}}g"
    assert g.element.get("id") == "det_person_0"
    assert g.element.get("data-class") == "person"
    assert g.element.get("data-conf") == "0.95"


def test_group_add_rect() -> None:
    doc = SVGDocument(100, 100)
    g = doc.add_group(id="box")
    g.add_rect(10, 20, 30, 40, fill="red", stroke="blue", stroke_width=2.0)
    s = doc.to_string()
    root = etree.fromstring(s.encode("utf-8"))
    rects = root.findall(f".//{{{SVG_NS}}}rect")
    assert len(rects) == 1
    r = rects[0]
    assert r.get("x") == "10" and r.get("y") == "20"
    assert r.get("width") == "30" and r.get("height") == "40"
    assert r.get("fill") == "red"
    assert r.get("stroke") == "blue"
    assert r.get("stroke-width") == "2.0"


def test_group_add_path() -> None:
    doc = SVGDocument(100, 100)
    g = doc.add_group(id="p")
    g.add_path("M 0 0 L 10 10 Z", fill="green", stroke="black", stroke_width=1.0)
    s = doc.to_string()
    root = etree.fromstring(s.encode("utf-8"))
    paths = root.findall(f".//{{{SVG_NS}}}path")
    assert len(paths) == 1
    assert paths[0].get("d") == "M 0 0 L 10 10 Z"
    assert paths[0].get("fill") == "green"


def test_group_add_text() -> None:
    doc = SVGDocument(100, 100)
    g = doc.add_group(id="t")
    g.add_text(5, 15, "hello", font_size=14, fill="purple")
    s = doc.to_string()
    root = etree.fromstring(s.encode("utf-8"))
    texts = root.findall(f".//{{{SVG_NS}}}text")
    assert len(texts) == 1
    assert texts[0].text == "hello"
    assert texts[0].get("fill") == "purple"
    assert texts[0].get("font-size") == "14"


def test_group_add_text_with_outline() -> None:
    doc = SVGDocument(100, 100)
    g = doc.add_group(id="t")
    g.add_text_with_outline(5, 15, "boxed", fill="white", stroke="black", stroke_width=3.0)
    s = doc.to_string()
    root = etree.fromstring(s.encode("utf-8"))
    text = root.find(f".//{{{SVG_NS}}}text")
    assert text.get("paint-order") == "stroke"
    assert text.get("stroke") == "black"
    assert text.get("stroke-width") == "3.0"


def test_write_atomic(tmp_path: Path) -> None:
    doc = SVGDocument(50, 50, title="test")
    target = tmp_path / "out.svg"
    doc.write(target)
    assert target.exists()
    assert not target.with_suffix(".svg.tmp").exists()
    # Parse the written file
    s = target.read_text()
    root = etree.fromstring(s.encode("utf-8"))
    assert root.tag == f"{{{SVG_NS}}}svg"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    doc = SVGDocument(50, 50)
    target = tmp_path / "deep" / "nested" / "out.svg"
    doc.write(target)
    assert target.exists()


def test_no_xmlns_mustache() -> None:
    """The serialized SVG must declare the SVG namespace."""
    doc = SVGDocument(50, 50)
    s = doc.to_string()
    assert SVG_NS in s
    assert "xmlns" in s
