# img2svg - SVG document builder using lxml for clean namespace handling.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""SVG document builder using lxml for clean namespace handling.

Provides a small `SVGDocument` class that wraps an `<svg>` root with helpers
for adding groups, rects, paths, and text. Auto-includes <title> and <desc>
for accessibility (WCAG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {
    "svg": SVG_NS,
    "xlink": XLINK_NS,
}


def _q(local: str) -> str:
    """Return a namespaced tag in Clark notation: {ns}local."""
    return f"{{{SVG_NS}}}{local}"


class SVGGroup:
    """A `<g>` element with helper methods."""

    def __init__(self, doc: SVGDocument, **attrs: Any) -> None:
        self._doc = doc
        self._el = etree.SubElement(doc.root, _q("g"), nsmap={"xlink": XLINK_NS})
        for k, v in attrs.items():
            if v is None:
                continue
            # Use simple unprefixed attributes; SVG accepts them.
            self._el.set(k.replace("_", "-"), str(v))

    def set(self, key: str, value: Any) -> None:
        self._el.set(key.replace("_", "-"), str(value))

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | None = "none",
        stroke: str | None = "black",
        stroke_width: float | None = 1.0,
        **attrs: Any,
    ) -> Any:
        el = etree.SubElement(self._el, _q("rect"))
        el.set("x", str(x))
        el.set("y", str(y))
        el.set("width", str(w))
        el.set("height", str(h))
        if fill is not None:
            el.set("fill", fill)
        if stroke is not None:
            el.set("stroke", stroke)
        if stroke_width is not None:
            el.set("stroke-width", str(stroke_width))
        for k, v in attrs.items():
            if v is not None:
                el.set(k.replace("_", "-"), str(v))
        return el

    def add_path(
        self,
        d: str,
        fill: str | None = "none",
        stroke: str | None = None,
        stroke_width: float | None = None,
        **attrs: Any,
    ) -> Any:
        el = etree.SubElement(self._el, _q("path"))
        el.set("d", d)
        if fill is not None:
            el.set("fill", fill)
        if stroke is not None:
            el.set("stroke", stroke)
        if stroke_width is not None:
            el.set("stroke-width", str(stroke_width))
        for k, v in attrs.items():
            if v is not None:
                el.set(k.replace("_", "-"), str(v))
        return el

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        font_size: float = 12.0,
        fill: str = "black",
        **attrs: Any,
    ) -> Any:
        el = etree.SubElement(self._el, _q("text"))
        el.set("x", str(x))
        el.set("y", str(y))
        el.set("font-size", str(font_size))
        el.set("fill", fill)
        el.text = text
        for k, v in attrs.items():
            if v is not None:
                el.set(k.replace("_", "-"), str(v))
        return el

    def add_text_with_outline(
        self,
        x: float,
        y: float,
        text: str,
        font_size: float = 12.0,
        fill: str = "white",
        stroke: str = "black",
        stroke_width: float = 3.0,
    ) -> Any:
        """Text with a contrasting outline (white fill, black stroke) for readability over images."""
        el = self.add_text(
            x,
            y,
            text,
            font_size=font_size,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            paint_order="stroke",
        )
        return el

    @property
    def element(self) -> Any:
        return self._el


class SVGDocument:
    """An SVG document with helpers for assembling content.

    Auto-includes a <title> and <desc> as the first children for accessibility.
    """

    def __init__(
        self,
        width: int | float,
        height: int | float,
        title: str | None = None,
        desc: str | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.root = etree.Element(
            _q("svg"),
            nsmap={"svg": SVG_NS, "xlink": XLINK_NS},
        )
        self.root.set("width", str(width))
        self.root.set("height", str(height))
        self.root.set("viewBox", f"0 0 {width} {height}")
        self.root.set("xmlns", SVG_NS)
        if title:
            t = etree.SubElement(self.root, _q("title"))
            t.text = title
        if desc:
            d = etree.SubElement(self.root, _q("desc"))
            d.text = desc
        # Embedded CSS for default styling
        self._style = etree.SubElement(self.root, _q("style"))
        self._style.text = (
            "text { font-family: sans-serif; }"
            " .detection { fill: none; stroke: black; stroke-width: 2; }"
        )

    def add_group(self, id: str | None = None, **attrs: Any) -> SVGGroup:
        """Add a <g> child. Pass `id`, `class_` (becomes class), `data_*` attrs as kwargs."""
        if id is not None:
            attrs["id"] = id
        return SVGGroup(self, **attrs)

    def to_string(self, pretty: bool = True) -> str:
        """Serialize to SVG XML string."""
        # lxml forbids xml_declaration=True with encoding="unicode"; the
        # standard workaround is to ask for bytes with a declared encoding
        # and then decode back to a string.
        return etree.tostring(
            self.root, pretty_print=pretty, xml_declaration=True, encoding="UTF-8"
        ).decode("utf-8")

    def write(self, path: str | Path) -> None:
        """Write to `path` (atomic: write to .tmp, then rename)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(self.to_string(pretty=True), encoding="utf-8")
        tmp.replace(target)
