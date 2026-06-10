# Before / After

What does the SVG actually look like next to the source PNG? These are the two synthetic test fixtures the project ships with, rendered in all four explicit modes.

The full set of eight outputs lives in [`sample_outputs/`](./sample_outputs/). Every SVG was produced by the command in its caption.

## `tests/fixtures/logo.png` — flat-color logo

A 200×200 PNG with three solid-color shapes (red square, blue triangle, black circle) on a white background. The classifier rates it as a photo (alpha=false, multi-color, low edge density) so the `auto` mode would not pick `labels`; the four explicit runs below show what each renderer actually produces.

| Input | Output (mode) | File |
| ----- | ------------- | ---- |
| ![logo input](../../tests/fixtures/logo.png) | Labels | [`logo_labels.svg`](./sample_outputs/logo_labels.svg) |
| ![logo input](../../tests/fixtures/logo.png) | Visual | [`logo_visual.svg`](./sample_outputs/logo_visual.svg) |
| ![logo input](../../tests/fixtures/logo.png) | Annotated | [`logo_annotated.svg`](./sample_outputs/logo_annotated.svg) |
| ![logo input](../../tests/fixtures/logo.png) | Trace | [`logo_trace.svg`](./sample_outputs/logo_trace.svg) |

```bash
uv run img2svg tests/fixtures/logo.png -o examples/sample_outputs/logo_labels.svg     --mode labels
uv run img2svg tests/fixtures/logo.png -o examples/sample_outputs/logo_visual.svg    --mode visual
uv run img2svg tests/fixtures/logo.png -o examples/sample_outputs/logo_annotated.svg --mode annotated
uv run img2svg tests/fixtures/logo.png -o examples/sample_outputs/logo_trace.svg     --mode trace
```

What each mode does for this image:

- **labels** — YOLO found no confident objects, so the output is a clean white background rect with no detection groups. The SVG is ~500 bytes.
- **visual** — vtracer fills the red, blue, and black shapes with smooth Bézier paths. The output looks identical to the input at typical zoom.
- **annotated** — same vtracer fill as `visual`, but with the detection layer ready to draw on top. Still empty for this image because no YOLO classes matched.
- **trace** — vtracer's outline-only preset: thin stroked paths for each shape, no fill.

## `tests/fixtures/diagram.png` — schematic with rectangles and labels

A 200×200 PNG that draws a small schematic: a few labeled rectangles connected by lines, on a white background. Diagrams tend to have many edges and few colors, which is what the `auto` classifier keys on.

| Input | Output (mode) | File |
| ----- | ------------- | ---- |
| ![diagram input](../../tests/fixtures/diagram.png) | Labels | [`diagram_labels.svg`](./sample_outputs/diagram_labels.svg) |
| ![diagram input](../../tests/fixtures/diagram.png) | Visual | [`diagram_visual.svg`](./sample_outputs/diagram_visual.svg) |
| ![diagram input](../../tests/fixtures/diagram.png) | Annotated | [`diagram_annotated.svg`](./sample_outputs/diagram_annotated.svg) |
| ![diagram input](../../tests/fixtures/diagram.png) | Trace | [`diagram_trace.svg`](./sample_outputs/diagram_trace.svg) |

```bash
uv run img2svg tests/fixtures/diagram.png -o examples/sample_outputs/diagram_labels.svg     --mode labels
uv run img2svg tests/fixtures/diagram.png -o examples/sample_outputs/diagram_visual.svg    --mode visual
uv run img2svg tests/fixtures/diagram.png -o examples/sample_outputs/diagram_annotated.svg --mode annotated
uv run img2svg tests/fixtures/diagram.png -o examples/sample_outputs/diagram_trace.svg     --mode trace
```

What each mode does for this image:

- **labels** — still no YOLO hits (the synthetic shapes are too abstract), so the output is the white background. The label renderer is most useful on real photos where YOLO finds something.
- **visual** — vtracer fills the rectangles and lines with solid colors. The result reads like the original diagram, just vectorized.
- **annotated** — same `visual` fill, with the detection overlay in reserve.
- **trace** — outlines only. Useful for line-art sources where you want clean strokes and no fill.

## Picking a mode at a glance

- Source is a real photo with people, cars, animals: try `annotated` or `labels` so the YOLO boxes show up.
- Source is a logo, icon, or flat illustration: `visual` keeps the colors and removes the raster artifacts.
- Source is a screenshot or diagram: `visual` for a clean look, `trace` for an outline-only sketch.
- Unsure: `auto`. The classifier picks `labels` for logo-shaped inputs, `visual` for photos, and `annotated` for diagrams and screenshots.

Sidecar JSON files (`*.json`) sit next to each SVG in `sample_outputs/`. They hold the resolved mode, the model and device used, the image type the classifier chose, and the per-step timings. See [`python_api.md`](./python_api.md) for how to read them from a script.
