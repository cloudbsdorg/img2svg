"""Tests for `TraceRenderer` (vtracer photo preset)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from img2svg.loader import LoadedImage
from img2svg.renderers.base import Renderer
from img2svg.renderers.trace import TraceRenderer
from img2svg.svg_builder import SVG_NS, SVGDocument

# A different vtracer-style output than the visual fixture, so that
# cross-contamination between the two renderers is detectable.
_KNOWN_VTRACER_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <path d="M1 1 L11 11" fill="#abcdef"/>
</svg>
"""


def _make_image(width: int = 20, height: int = 20) -> LoadedImage:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[5:15, 5:15] = [0, 255, 0]
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


def test_trace_renderer_is_subclass_of_renderer() -> None:
    assert issubclass(TraceRenderer, Renderer)


def test_trace_renderer_uses_photo_preset() -> None:
    assert TraceRenderer.preset_name == "photo"


def test_render_embeds_vtracer_output_group() -> None:
    svg = _make_doc()
    renderer = TraceRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    from lxml import etree
    root = etree.fromstring(svg.to_string().encode("utf-8"))
    assert _vtracer_group(root) is not None, "vtracer-output group missing from SVG"


def test_render_copies_vtracer_paths_into_group() -> None:
    svg = _make_doc()
    renderer = TraceRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    from lxml import etree
    root = etree.fromstring(svg.to_string().encode("utf-8"))
    group = _vtracer_group(root)
    assert group is not None
    paths = group.findall(f"{{{SVG_NS}}}path")
    assert len(paths) == 1
    assert paths[0].get("d") == "M1 1 L11 11"
    assert paths[0].get("fill") == "#abcdef"


def test_render_calls_vtracer_vectorizer_with_photo_preset() -> None:
    svg = _make_doc()
    renderer = TraceRenderer(svg, _make_image(), [], None)

    with mock.patch("img2svg.renderers.visual.VtracerVectorizer") as MockVec:
        MockVec.return_value.vectorize.side_effect = _fake_vectorize(_KNOWN_VTRACER_SVG)
        renderer.render()

    MockVec.assert_called_once_with(preset="photo")
    MockVec.return_value.vectorize.assert_called_once()
    args = MockVec.return_value.vectorize.call_args.args
    assert str(args[0]).endswith(".png")
    assert str(args[1]).endswith(".svg")


def test_visual_and_trace_use_different_presets() -> None:
    """Sanity check: the two renderers must not share the same preset."""
    from img2svg.renderers.visual import VisualRenderer
    assert VisualRenderer.preset_name != TraceRenderer.preset_name
