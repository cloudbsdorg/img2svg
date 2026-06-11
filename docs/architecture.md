# Architecture

`img2svg` is organized as a small core orchestrator (`Pipeline`) that composes a sequence of well-defined stages, with pluggable renderers at the end. This page is the tour of the internals — the pipeline stages, the renderer registry, and the GPU dispatch.

If you are looking for the public API surface, see [Python API](api.md). This page is for contributors and reviewers.

## High-level layout

```
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  CLI / API   │───▶│   Pipeline   │───▶│   Renderer   │───▶  SVG + JSON
   └──────────────┘    └──────────────┘    └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Detector /   │
                       │ Classifier   │
                       └──────────────┘
```

- **CLI / API** — the entry points. The CLI (`img2svg.cli`) is a Typer app. The Python API (`img2svg.api`) exports `convert()` and `convert_batch()`. Both delegate to the same `Pipeline`.
- **Pipeline** — `img2svg.pipeline.Pipeline` owns the 12-step sequence. Constructing one is cheap; pass the same instance to multiple `.run()` calls in a batch to amortize YOLO load time.
- **Detector / Classifier** — `img2svg.detector` wraps Ultralytics YOLO. `img2svg.classifier` is a non-ML heuristic on alpha, dominant colors, and edge density.
- **Renderer** — one of four classes registered in `RENDERER_REGISTRY`. See the [Renderer composition](#renderer-composition) section below.

## Pipeline flowchart

The full sequence from input image to SVG + sidecar:

```mermaid
flowchart LR
    A[Image Input] --> B[Loader]
    B --> P[Preprocessing]
    P --> C[Classifier]
    C --> D[YOLO Detector]
    C --> E[Geometric Analysis]
    D --> S[Segmentation]
    S --> F[Pipeline]
    E --> F
    F --> G[Renderer]
    G --> H[SVG Output]
    F --> I[Sidecar JSON]
```

The pipeline steps inside the `Pipeline.run()` are documented inline in the source with step-marker comments. The high-level flow is:

1. **Load** the image via `img2svg.loader.load_image()`.
2. **Preprocess** — apply the configured OpenCV filter chain (denoise, sharpen, posterize, edge detection). Off by default; opt in with `--preprocess` or the `--denoise` / `--sharpen` shortcuts.
3. **Analyze globally** — dominant colors, edge density, contour count, alpha presence (`img2svg.patterns.analyze_global`).
4. **Classify** the image type (`logo`, `photo`, `diagram`, `screenshot`, `line_art`, `unknown`).
5. **Select mode** — if the user passed `Mode.AUTO`, pick a concrete mode from the classification. Otherwise honor the user's choice.
6. **Detect** objects with YOLO detection (skipped for `Mode.TRACE`).
7. **Segment** — for `Mode.SEGMENTED`, run YOLO instance segmentation and trace each region with vtracer. Falls back to bbox detection on VRAM pressure. Skipped when `--no-seg` is set or for non-segmented modes.
8. (Per-ROI analysis is currently a no-op; the spec defers it.)
9. **Resolve device** — pick the GPU per the strategy, or fall back to CPU.
10. **Build the renderer** from the resolved mode.
11. **Render** the SVG via the renderer's `.render(svg, image, detections, analysis)` method. The `SegmentedRenderer` also receives the `SegmentationResult` via `set_segmentation()`.
12. **Write** the SVG atomically (capped at `max_svg_size_mb`).
13. **Write** the sidecar JSON atomically.
14. **Return** the `ConversionResult`.

`Mode.AUTO` is the only mode that goes through step 5; explicit modes skip it. The pipeline is strict about this — `RENDERER_REGISTRY` does not have an entry for `Mode.AUTO`, so the lookup would fail if step 5 were skipped.

## Renderer composition

Each `Mode` value maps to a renderer class via `img2svg.pipeline.RENDERER_REGISTRY`. `Mode.AUTO` is intentionally absent — the pipeline resolves it first via `select_mode()`.

```mermaid
flowchart TB
    A[Mode.AUTO] --> B[select_mode]
    B --> C[Mode.LABELS]
    B --> D[Mode.VISUAL]
    B --> E[Mode.ANNOTATED]
    B --> F[Mode.TRACE]
    B --> K[Mode.DETAILED]
    B --> L[Mode.POSTER]
    B --> M[Mode.EDGE]
    B --> N[Mode.WATERCOLOR]
    B --> O[Mode.SEGMENTED]
    C --> G[LabelsRenderer]
    D --> H[VisualRenderer]
    E --> I[AnnotatedRenderer]
    F --> J[TraceRenderer]
    K --> P[DetailedRenderer]
    L --> Q[PosterRenderer]
    M --> R[EdgeRenderer]
    N --> S[WatercolorRenderer]
    O --> T[SegmentedRenderer]
```

The ten renderer classes all subclass `img2svg.renderers.base.Renderer`, which defines the contract:

```python
class Renderer:
    def __init__(self, svg: SVGDocument, image: LoadedImage,
                 detections: list[Detection],
                 geometric: GeometricAnalysis) -> None: ...
    def render(self) -> None: ...
```

The constructor stashes the inputs, and `.render()` is the only side-effecting call. This makes renderers trivial to unit-test: build a `SVGDocument` and a fake `LoadedImage`, call the renderer, and assert the SVG tree.

| Mode         | Renderer class                                | What it does                                                |
|--------------|-----------------------------------------------|-------------------------------------------------------------|
| `LABELS`     | `img2svg.renderers.labels.LabelsRenderer`     | Embeds image, draws labeled bounding boxes.                 |
| `VISUAL`     | `img2svg.renderers.visual.VisualRenderer`     | Embeds image, vectorizes with vtracer.                      |
| `ANNOTATED`  | `img2svg.renderers.annotated.AnnotatedRenderer` | vtracer output plus labeled bounding boxes.              |
| `TRACE`      | `img2svg.renderers.trace.TraceRenderer`       | vtracer output only, no image, no labels.                   |
| `POSTER`     | `img2svg.renderers.poster.PosterRenderer`     | vtracer `poster` preset: flat color regions.                |
| `DETAILED`   | `img2svg.renderers.detailed.DetailedRenderer` | vtracer `photo_hifi` preset: high-fidelity photo trace.     |
| `EDGE`       | `img2svg.renderers.edge.EdgeRenderer`         | vtracer `bw_edge` preset: line-art, polygon mode.            |
| `WATERCOLOR` | `img2svg.renderers.watercolor.WatercolorRenderer` | vtracer `watercolor` preset: soft painterly output.     |
| `SEGMENTED`  | `img2svg.renderers.segmented.SegmentedRenderer` | Multi-layer SVG: one `<g>` per YOLO detection.            |

The `VisualRenderer` and `TraceRenderer` share a private helper `_render_with_vtracer()` that lives in `visual.py` and is imported by `trace.py`. The `AnnotatedRenderer` calls the same helper first, then appends detection groups — the vtracer group is the background, the detection groups are the foreground. SVG paint order follows document order, so this layering works out automatically.

The `SegmentedRenderer` is a special case. It accepts a `SegmentationResult` via `set_segmentation()` between the segmentor call and `render()`. The renderer falls back to `VisualRenderer` behavior when no segmentation result is set or every mask is empty, which makes the renderer testable without YOLO. See [Photo modes](photo-modes.md#segmented) for the per-region tracing flow.

## GPU dispatch

The detector is the only stage that touches the GPU. Dispatch happens once per `Pipeline` instance:

```mermaid
flowchart LR
    A[Pipeline.run] --> B{DeviceStrategy}
    B -- power --> C[list_gpus]
    B -- availability --> C
    B -- auto --> C
    C --> D[recommend_gpu]
    D --> E[YOLO .to device]
    E --> F[Inference]
```

`list_gpus()` tries three sources in order:

1. `nvidia-smi` — direct subprocess call, returns NVIDIA GPUs with VRAM and utilization.
2. `rocm-smi` — direct subprocess call, returns AMD GPUs.
3. `torch.cuda` — falls back to PyTorch's CUDA enumeration (works for both NVIDIA CUDA and AMD ROCm builds).

The first non-empty result wins. If all three return nothing, the pipeline runs on CPU.

`recommend_gpu(gpus, strategy)` picks the best device. The `power` strategy picks the largest total VRAM; `availability` picks the largest free VRAM; `auto` returns the first device in index order. On a tie, the smaller index wins (the `key=` lambda uses `-g.index` to break ties toward smaller indices).

The chosen device string is passed to YOLO via `.to(device)`, which lazy-loads the model and moves it to the device on first inference. Subsequent calls in the same `Pipeline` reuse the loaded model.

## Sidecar metadata

The sidecar is a JSON file written next to the SVG with the same name plus `.json`. The schema is fixed by `img2svg.models.Sidecar`:

```json
{
  "version": "0.1.0",
  "input_path": "/path/to/source.png",
  "input_hash": "sha256-hex",
  "output_path": "/path/to/output.svg",
  "output_size": 12345,
  "mode_used": "labels",
  "mode_reasoning": "classifier: logo (4 dominant colors, low edge density)",
  "model": "yolo11x.pt",
  "device": "cuda:0",
  "image_type": "logo",
  "detections": [ ... ],
  "geometric": { ... },
  "timings": {
    "load": 0.012,
    "classify": 0.003,
    "detect": 0.234,
    "render": 0.045,
    "total": 0.310
  },
  "timestamp": "2026-06-10T12:34:56.789+00:00"
}
```

The `timings` dict is built incrementally across the 12 steps, and `total` is set **before** the `Sidecar` is constructed so it gets serialized. This is a load-bearing ordering invariant — see the T19 learning note in the notepad for the full story.

## Concurrency model

`Pipeline` instances are **not** thread-safe. The YOLO detector is module-level cached but holds global CUDA state. To parallelize a batch, run multiple processes (`multiprocessing.Pool`, `concurrent.futures.ProcessPoolExecutor`) with one `Pipeline` per worker. The CLI does not parallelize internally; for that, use the Python API in a worker pool.

Within a single `Pipeline`, calls to `.run()` are sequential. The YOLO model is loaded on first `.run()` and reused for the rest of the instance's lifetime.

## Error handling

The pipeline lets exceptions propagate. The API catches them at the boundary:

- `convert()` re-raises `UnsupportedFormatError` and `CorruptImageError` so the caller can decide what to do.
- `convert_batch()` catches all exceptions, records the message in the result's `errors` list, and continues with the next file (when `continue_on_error=True`).

The CLI maps exceptions to exit codes:

| Exception                       | Exit code |
|---------------------------------|-----------|
| Success / batch with no errors  | 0         |
| Batch with at least one failure | 1         |
| `UnsupportedFormatError`, `CorruptImageError`, `FileNotFoundError`, `OutputPathCollisionError`, `DeviceUnavailableError` | 2 |
| `ModelLoadError`, `ConfigError` | 3         |
| Other `Img2SvgError`            | 1         |

## Module dependency graph

A quick view of the imports between modules. The arrows point from importer to importee.

```
cli ──▶ api, device, enums, errors, gpu, logging, models
api ──▶ enums, logging, models, pipeline
pipeline ──▶ classifier, detector, enums, loader, logging, metadata, models, patterns, presets, renderers.*, svg_builder
renderers.annotated ──▶ renderers.base, renderers.visual (private helper)
renderers.visual ──▶ renderers.base, vectorizer
renderers.trace ──▶ renderers.base, renderers.visual (private helper)
renderers.labels ──▶ renderers.base
```

The renderers are leaf-ish: they depend on `base`, `vectorizer`, and the SVG builder, but nothing depends on them directly other than the pipeline. This makes them easy to unit-test in isolation.

## See also

- [Python API](api.md) — public surface.
- [Output modes](modes.md) — when to use each mode.
- [Photo modes](photo-modes.md) — the five photo modes, the preprocessing chain, and the segmentation workflow.
- [GPU setup](gpu.md) — vendor-specific install and dispatch.
- [Development](development.md) — TDD workflow and code conventions.
