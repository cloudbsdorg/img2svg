# Output modes

`img2svg` has ten rendering modes that control how the source image is converted to SVG. `auto` is the default and runs the classifier to pick a mode based on the image. The other nine are explicit overrides for predictable output.

Five of the explicit modes target photographs and other complex raster sources: `poster`, `detailed`, `edge`, `watercolor`, and `segmented`. They live in a separate [Photo modes](photo-modes.md) guide because they share a common pipeline (preprocessing, segmentation, per-mode renderer) that's worth a deep dive of its own. The other four — `labels`, `visual`, `annotated`, `trace` — are covered in full on this page.

## Mode summary

| Mode         | Image type          | What it produces                                  |
|--------------|---------------------|---------------------------------------------------|
| `auto`       | Any                 | Classifier picks `visual` or `detailed`.          |
| `labels`     | Logos, line art     | Bounding boxes with class labels over the image.  |
| `visual`     | Photos, paintings   | Vectorized via vtracer, no overlays.              |
| `annotated`  | Photos with objects | Vectorized via vtracer plus bounding-box labels.  |
| `trace`      | Sketches, line art  | vtracer output only, no image, no labels.         |
| `poster`     | Posters, prints     | Stylized, limited-color vtracer output.           |
| `detailed`   | Photos              | High-fidelity vtracer `photo_hifi` trace.         |
| `edge`       | Sketches, line art  | Line-art, polygon-mode vtracer output.            |
| `watercolor` | Art, illustrations  | Soft, painterly vtracer output.                   |
| `segmented`  | Photos with objects | Multi-layer SVG, one group per YOLO detection.    |

The mode enum is in `img2svg.enums.Mode`. The Python API takes the enum or its string value; the CLI takes the string.

```python
from img2svg import ConversionOptions
from img2svg.enums import Mode

opts = ConversionOptions(mode=Mode.LABELS)        # enum
opts2 = ConversionOptions(mode="labels")          # str — auto-coerced
```

## auto

The default. The classifier inspects alpha, dominant colors, and edge density and picks one of the explicit modes:

- Photos → `detailed` (the high-fidelity photo trace).
- Logos, clean line art, diagrams, screenshots, unknown → `visual`.

`auto` never picks `labels`, `annotated`, `poster`, `edge`, `watercolor`, or `segmented` — those are explicit-only modes. See [Photo modes](photo-modes.md#choosing-a-photo-mode) for the rationale.

The classification is heuristic; if the auto mode picks something unexpected, force the mode explicitly with `--mode`.

## labels

Best for logos, icons, and line art where you want the SVG to know what the objects are. The image is embedded as a background, and YOLO detections are drawn as labeled bounding boxes.

```bash
img2svg logo.png -o logo.svg --mode labels
```

Renderer: `img2svg.renderers.labels.LabelsRenderer`. Each detection becomes a `<g>` element with `data-class` and `data-conf` attributes and a labeled text overlay.

## visual

Best for photos where you want a clean vectorized image without bounding boxes. The image is run through [vtracer](https://github.com/visioncortex/vtracer) and the resulting `<path>` elements are embedded in the SVG. No labels, no overlays.

```bash
img2svg photo.jpg -o photo.svg --mode visual
```

Renderer: `img2svg.renderers.visual.VisualRenderer`. The output is dominated by vtracer's color-quantized paths.

## annotated

Combines `visual` and `labels`. The image is vectorized, and YOLO detections are drawn as labeled bounding boxes on top. The vtracer layer is drawn first (background), then the detection groups (foreground) — paint order follows document order in SVG.

```bash
img2svg group_photo.jpg -o group.svg --mode annotated
```

Renderer: `img2svg.renderers.annotated.AnnotatedRenderer`. Useful for photographs with multiple recognizable subjects (people, cars, animals).

## trace

For line art, sketches, and diagrams where the image itself is the only output. No detection, no labels — just the vectorized paths.

```bash
img2svg sketch.png -o sketch.svg --mode trace
```

Renderer: `img2svg.renderers.trace.TraceRenderer`. The output is a pure vtracer SVG with the embedded image removed.

## Choosing a mode

A simple decision tree:

1. Is it a photo?
   - Yes, and you want a clean vectorized image → `visual`.
   - Yes, and you want to know what's in it → `annotated`.
   - No, continue.
2. Is it a logo, icon, or simple line drawing?
   - Yes → `labels`.
3. Is it a sketch or hand-drawn line art?
   - Yes → `trace`.
4. Not sure? → `auto`.

## Examples side by side

The same source image rendered in each mode:

```bash
img2svg cat.png -o cat.labels.svg   --mode labels
img2svg cat.png -o cat.visual.svg   --mode visual
img2svg cat.png -o cat.annotated.svg --mode annotated
img2svg cat.png -o cat.trace.svg    --mode trace
```

The four outputs differ in:

- **File size**: `trace` is usually smallest; `visual` and `annotated` are similar to each other; `labels` is smallest of all.
- **Editability**: `trace` and `visual` are best for editing in Inkscape or Illustrator (paths only, no overlays).
- **Metadata**: `labels` and `annotated` carry class and confidence metadata on each detection group, useful for downstream tools.

## Forcing a mode from Python

The Python API takes the enum or its string value:

```python
from img2svg import convert, ConversionOptions

# String value
convert("cat.png", "cat.svg", mode="annotated")

# Enum value (for static type checking)
from img2svg.enums import Mode
convert("cat.png", "cat.svg", options=ConversionOptions(mode=Mode.ANNOTATED))
```

## See also

- [Photo modes](photo-modes.md) — the five photo modes (`poster`, `detailed`, `edge`, `watercolor`, `segmented`), the optional preprocessing chain, and the segmentation workflow.
- [Usage](usage.md) — CLI examples for each mode.
- [Architecture](architecture.md) — how modes map to renderer classes (with diagram).
- [Python API](api.md) — `ConversionOptions` reference.
