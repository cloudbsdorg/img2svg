# Python API

The `img2svg` package exposes a small, stable Python API. Most callers will only ever use `convert()` (one image) and `convert_batch()` (a list, glob, or directory). The data classes that flow in and out of those functions — `ConversionOptions`, `ConversionResult`, `Sidecar` — are also public and live in `img2svg.models`.

## Import paths

| Symbol              | Import path                |
|---------------------|----------------------------|
| `convert`           | `img2svg.api.convert`      |
| `convert_batch`     | `img2svg.api.convert_batch`|
| `ConversionOptions` | `img2svg.models.ConversionOptions` |
| `ConversionResult`  | `img2svg.models.ConversionResult`  |
| `Sidecar`           | `img2svg.models.Sidecar`   |
| `Detection`         | `img2svg.models.Detection` |
| `RegionInfo`        | `img2svg.models.RegionInfo` |
| `SegmentationResult`| `img2svg.detector.SegmentationResult` |
| `GPUInfo`           | `img2svg.models.GPUInfo`   |
| `BackendSpec`       | `img2svg.models.BackendSpec` |
| `Mode`              | `img2svg.enums.Mode`       |
| `ImageType`         | `img2svg.enums.ImageType`  |
| `DeviceStrategy`    | `img2svg.enums.DeviceStrategy` |
| `GpuVendor`         | `img2svg.enums.GpuVendor`  |
| `BackendType`       | `img2svg.backends.BackendType` |

The `img2svg.api` module re-exports `Pipeline` from `img2svg.pipeline` for advanced callers that want direct control over the orchestrator.

## convert()

```python
from img2svg import convert

result = convert("photo.png", "photo.svg", mode="visual", conf=0.3)
```

`convert()` runs the full pipeline on a single image and returns a `ConversionResult`. The two positional arguments are the input and output paths. All other behavior is configured through either an `options` object or keyword arguments that map to `ConversionOptions` fields.

**Signature**

```python
def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    options: ConversionOptions | None = None,
    **kwargs: object,
) -> ConversionResult: ...
```

**Arguments**

- `input_path` — source image (PNG, JPEG, BMP, WebP, TIFF, or GIF).
- `output_path` — destination `.svg` file.
- `options` — pre-built `ConversionOptions` (overrides kwargs).
- `**kwargs` — any `ConversionOptions` field as a keyword. Unknown kwargs are silently dropped, so passing CLI-style extras will not break the API.

**Returns**

A `ConversionResult` with:

- `svg_path` (`Path`) — the SVG that was written.
- `sidecar_path` (`Path`) — the JSON metadata file next to the SVG.
- `sidecar` (`Sidecar`) — the structured metadata record.
- `detections` (`list[Detection]`) — YOLO detections, in the order they were emitted.
- `errors` (`list[str]`) — non-empty if the conversion failed.

**Raises**

- `img2svg.errors.UnsupportedFormatError` — input format not in the supported set.
- `img2svg.errors.CorruptImageError` — Pillow could not decode the input.

**Example**

```python
from img2svg import convert, ConversionOptions

opts = ConversionOptions(mode="annotated", conf=0.3, device="cuda:0")
result = convert("group_photo.jpg", "group_photo.svg", options=opts)
print(f"wrote {result.svg_path} with {len(result.detections)} detections")
```

## convert_batch()

```python
from img2svg import convert_batch

results = convert_batch("photos/", output_dir="svg-out/", recursive=True)
```

`convert_batch()` runs `convert()` over a list of inputs. The `inputs` argument accepts a list of paths, a glob pattern, a single file path, or a directory. Only files with supported image extensions are processed.

**Signature**

```python
def convert_batch(
    inputs: list[str | Path] | str,
    output_dir: str | Path | None = None,
    *,
    options: ConversionOptions | None = None,
    recursive: bool = False,
    continue_on_error: bool = True,
    show_progress: bool = False,
    **kwargs: object,
) -> list[ConversionResult]: ...
```

**Arguments**

- `inputs` — list, glob, single file, or directory. See [Usage](usage.md#batch-conversion) for examples.
- `output_dir` — destination directory. If `None`, the default is the input file's parent (single file), the directory itself (directory input), or the first resolved path's parent (list/glob).
- `options` — pre-built `ConversionOptions` (overrides kwargs).
- `recursive` — walk input directories recursively.
- `continue_on_error` — if `True` (default), per-file errors are recorded in the result's `errors` field and processing continues. If `False`, the first error re-raises.
- `show_progress` — if `True`, render a Rich progress bar over the batch.
- `**kwargs` — forwarded to `ConversionOptions`.

**Returns**

A list of `ConversionResult` in deterministic input order (sorted by path). Failed files have non-empty `errors`. The list has the same length as the number of inputs that matched the supported-extension filter; non-image files are silently dropped.

**Example with progress**

```python
from img2svg import convert_batch

results = convert_batch(
    "photos/",
    output_dir="svg-out/",
    recursive=True,
    show_progress=True,
    mode="visual",
)

for r in results:
    if r.errors:
        print(f"FAIL: {r.svg_path} - {r.errors[0]}")
    else:
        print(f"OK:   {r.svg_path}")
```

## ConversionOptions

`ConversionOptions` is the user-facing options object. It is a Pydantic v2 `BaseModel` with `use_enum_values=False`, so the `mode` and `gpu_strategy` fields are enum instances (not strings) on the model itself.

| Field           | Type             | Default        | Notes                                    |
|-----------------|------------------|----------------|------------------------------------------|
| `mode`          | `Mode`           | `Mode.AUTO`    | `auto`, `labels`, `visual`, `annotated`, `trace`, `poster`, `detailed`, `edge`, `watercolor`, `segmented`. |
| `model`         | `str`            | `"yolo11x.pt"` | YOLO weights file.                      |
| `device`        | `str`            | `"auto"`       | **Deprecated.** Use `backend` instead.   |
| `backend`       | `BackendSpec`    | `BackendSpec()` | The new structured selector. See [BackendSpec](#backendspec). |
| `conf`          | `float`          | `0.25`         | Confidence threshold, 0.0 to 1.0.       |
| `iou`           | `float`          | `0.7`          | IoU threshold for NMS.                   |
| `gpu_strategy`  | `DeviceStrategy` | `POWER`        | `auto`, `power`, `availability`.        |
| `no_clobber`    | `bool`           | `False`        | Refuse to overwrite existing outputs.    |
| `force_overwrite` | `bool`         | `False`        | Force-overwrite even with `no_clobber`.  |
| `preprocess`    | `list[str]`      | `[]`           | Preprocessing filter names. May be a chain. |
| `denoise`       | `str`            | `""`           | `bilateral`, `nlmeans`, `median`, or empty. |
| `sharpen`       | `str`            | `""`           | `unsharp` or empty.                     |
| `max_colors`    | `int`            | `0`            | Color cap, 0-256 (0 = no cap).          |
| `quality`       | `int`            | `90`           | JPEG-style quality hint, 1-100. Stored in sidecar. |
| `no_preprocess` | `bool`           | `False`        | Disable all preprocessing, overriding any positive choices. |
| `seg_model`     | `str`            | `"yolo11s-seg"`| YOLO segmentation model variant. One of `yolo11n-seg`, `yolo11s-seg`, `yolo11m-seg`, `yolo11l-seg`, `yolo11x-seg`. |
| `no_seg`        | `bool`           | `False`        | Disable segmentation even when the mode would use it. |
| `max_svg_size_mb` | `int`          | `50`           | Hard upper bound on SVG size in MB, 1-1024. If exceeded, the file is deleted and an error is raised. |

The validators on `conf`, `iou`, `max_colors`, `quality`, and `max_svg_size_mb` reject out-of-range values at construction time. The CLI does the same checks via option-level callbacks for friendlier error messages.

See [Photo modes](photo-modes.md) for the deep dive on the new photo modes, the preprocessing chain, and the segmentation workflow.

### The `backend=` parameter (replaces `device=`)

New code should use the structured `BackendSpec` selector via the `backend` field rather than the legacy `device` string. The two are equivalent today; the structured form is forward-compatible with the auto-detect chain and with vendor names that do not fit the legacy `cuda:N` shape (MPS, ROCm, CPU).

```python
from img2svg import ConversionOptions
from img2svg.models import BackendSpec

# Default: auto-detect the best backend.
opts = ConversionOptions()

# Force a specific backend.
opts = ConversionOptions(backend=BackendSpec(requested="cuda"))

# Pick a specific device index on a multi-GPU box.
opts = ConversionOptions(backend=BackendSpec(requested="cuda", index=1))

# AMD discrete GPU on a Linux host with a ROCm PyTorch build.
opts = ConversionOptions(backend=BackendSpec(requested="rocm", index=0))

# Apple Silicon.
opts = ConversionOptions(backend=BackendSpec(requested="mps"))

# CPU fallback.
opts = ConversionOptions(backend=BackendSpec(requested="cpu"))
```

### Deprecation warning on `device=`

The legacy `device` field is still accepted and round-trips into `BackendSpec` under the hood, but emits a `DeprecationWarning` when set explicitly. The shim is a transient aid for callers migrating from the 0.1.x API:

```python
import warnings
from img2svg import ConversionOptions

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    opts = ConversionOptions(device="cuda:0")

assert any(issubclass(w.category, DeprecationWarning) for w in caught)
assert opts.backend.requested == "cuda"
assert opts.backend.index == 0
```

If both `device` and `backend` are passed, the explicit `backend` wins and the deprecation warning still fires. The `device` field will be removed in 0.3.0.

## BackendSpec

`BackendSpec` is a frozen Pydantic v2 model that parses user-supplied device strings into a structured selector. It lives in `img2svg.models` and is the canonical way to address a compute backend from the Python API.

```python
from img2svg.models import BackendSpec

# Plain form
spec = BackendSpec(requested="cuda")

# Indexed form (multi-GPU)
spec = BackendSpec(requested="cuda:1")          # index=1, requested="cuda"

# AMD
spec = BackendSpec(requested="rocm:0")          # index=0, requested="rocm"

# Apple Silicon
spec = BackendSpec(requested="mps")

# CPU fallback
spec = BackendSpec(requested="cpu")

# Auto-detect (default)
spec = BackendSpec(requested="auto")
```

| Field         | Type                                                  | Default      | Notes                          |
|---------------|-------------------------------------------------------|--------------|--------------------------------|
| `requested`   | `Literal["auto", "cuda", "rocm", "mps", "cpu"]`       | `"auto"`     | The selector token.            |
| `index`       | `int \| None`                                         | `None`       | Device index for `cuda`/`rocm`.|

The model is **frozen** — assignment to a field raises `ValidationError`. To produce a new spec, build a fresh one. The `requested` field is a `Literal` so unknown selectors like `"directml"` are rejected at construction time. The `model_construct` escape hatch is reserved for the registry's deferred-validation path; prefer the regular constructor in user code.

The validator splits indexed forms like `"cuda:2"` and `"rocm:0"` into `(base, index)` so the `Literal` type can stay closed:

```python
spec = BackendSpec.model_validate({"requested": "cuda:2"})
assert spec.requested == "cuda"
assert spec.index == 2
```

Only `cuda` and `rocm` support `:N` indexing; `"mps:0"` and `"cpu:0"` are rejected at validation time.

## ConversionResult

The return type of `convert()`. Pydantic v2 `BaseModel` with these fields:

- `svg_path: Path` — the SVG that was written.
- `sidecar_path: Path` — the JSON sidecar path.
- `sidecar: Sidecar` — the structured metadata record.
- `detections: list[Detection]` — YOLO detections in emission order.
- `errors: list[str]` — non-empty if the conversion failed.

## RegionInfo

Per-region metadata carried in the sidecar for `segmented` mode runs. A `RegionInfo` exists for every detected object whose mask has at least one non-zero pixel. Empty masks are dropped before the sidecar is written.

| Field         | Type                          | Notes                                                    |
|---------------|-------------------------------|----------------------------------------------------------|
| `class_id`    | `int`                         | COCO class index from the YOLO model.                    |
| `class_name`  | `str`                         | Human-readable class label (e.g. `person`, `dog`).       |
| `confidence`  | `float`                       | Detection confidence, 0.0 to 1.0.                        |
| `bbox`        | `BoundingBox`                 | Tight bounding box around the mask, in image pixels.     |
| `area_pixels` | `int`                         | Number of pixels in the mask. Always `>= 0`.             |
| `polygon`     | `list[tuple[float, float]]`   | Mask as a list of `(x, y)` vertices. Empty when no mask. |
| `mask_path`   | `str \| None`                 | Reserved for future use; always `None` in the current release. |

`RegionInfo` is a Pydantic v2 `BaseModel`. The list of regions is in `sidecar.regions` (see [Sidecar](#sidecar)) and is empty for non-segmented modes.

```python
from img2svg.models import Sidecar, RegionInfo

sidecar = Sidecar.model_validate_json(open("photo.json").read())
for region in sidecar.regions:
    print(f"{region.class_name} @ {region.area_pixels} px (conf {region.confidence:.2f})")
```

## SegmentationResult

The structured output of `YOLOSegmentor.segment()`. Lives in `img2svg.detector`, not in `img2svg.models`, because it's a YOLO-specific result type that the pipeline threads into `SegmentedRenderer` without leaking YOLO types to the public surface.

| Field         | Type                              | Notes                                              |
|---------------|-----------------------------------|----------------------------------------------------|
| `masks`       | `list[np.ndarray]`                | One `(H, W)` uint8 array per detection, in `{0, 1}`. |
| `boxes`       | `list[Detection]`                 | The same detections the regular YOLO detector returns, in the same order. |
| `polygons`    | `list[list[tuple[float, float]]]` | Mask contours as `(x, y)` vertices.                |

The pipeline populates `sidecar.regions` with one `RegionInfo` per non-empty mask. Empty masks (no pixel above zero) are dropped. The `SegmentationResult` itself is not persisted to the sidecar; only the `RegionInfo` summary is.

## Sidecar

## Sidecar

The JSON metadata written next to every SVG. The fields are stable and machine-readable; tooling can rely on them.

- `version: str` — `img2svg` version that produced the file.
- `input_path: Path` — source image path.
- `input_hash: str` — SHA-256 of the input, hex-encoded.
- `output_path: Path` — destination SVG path.
- `output_size: int` — byte size of the SVG.
- `mode_used: Mode` — the resolved mode (never `AUTO`; the pipeline resolves it).
- `mode_reasoning: str` — human-readable explanation of the mode choice.
- `model: str` — YOLO weights file used.
- `device: str` — runtime device string. **Deprecated for new consumers**; use `backend_resolved` instead.
- `backend_requested: str` — the user's original selector (e.g. `"auto"`, `"cuda:0"`, `"mps"`). Empty string when not set.
- `backend_resolved: str` — the concrete backend the pipeline dispatched to (e.g. `"cuda:0"`, `"mps"`). Empty string when not set.
- `image_type: ImageType` — `logo`, `photo`, `diagram`, `screenshot`, `line_art`, or `unknown`.
- `detections: list[Detection]` — same as `ConversionResult.detections`.
- `geometric: GeometricAnalysis | None` — dominant colors, edge density, contour count, alpha.
- `preprocessing: list[str]` — names of preprocessing filters that ran, in order. Empty when preprocessing was off.
- `regions: list[RegionInfo]` — per-region metadata for `segmented` mode. Empty for non-segmented modes. See [RegionInfo](#regioninfo).
- `model_variant: str` — the segmentation model variant in use (e.g. `yolo11s-seg`). Empty string when no segmentation was performed.
- `timings: dict[str, float]` — per-step timings in seconds (`load`, `classify`, `preprocess`, `segment`, `detect`, `render`, `total`, ...).
- `timestamp: str` — ISO 8601 timestamp of when the conversion ran.

The `backend_requested` / `backend_resolved` pair is the structured successor to the legacy `device` field. New code should read the new pair; the legacy `device` field is preserved for backward compatibility with consumers that already parse the sidecar.

```json
{
  "version": "0.2.0",
  "input_hash": "a1b2c3...",
  "mode_used": "detailed",
  "model": "yolo11x.pt",
  "device": "cuda:0",
  "backend_requested": "cuda",
  "backend_resolved": "cuda:0",
  "image_type": "photo",
  "preprocessing": ["denoise_bilateral", "sharpen_unsharp"],
  "regions": [],
  "model_variant": "",
  "timings": {"load": 0.05, "classify": 0.12, "preprocess": 0.18, "detect": 1.43, "render": 0.08, "total": 1.86},
  "timestamp": "2026-06-10T14:22:08.123456+00:00"
}
```

## Working with the pipeline directly

For advanced callers, `img2svg.api.Pipeline` is the orchestrator class. Construct one with a `ConversionOptions` and call `.run()` per image. The pipeline is cheap to construct and can be reused across a batch to amortize YOLO model load time.

```python
from img2svg import ConversionOptions
from img2svg.api import Pipeline

pipeline = Pipeline(ConversionOptions(mode="visual"))
for src, dst in [("a.png", "a.svg"), ("b.png", "b.svg")]:
    pipeline.run(src, dst)
```

The pipeline follows a strict 12-step sequence. See [Architecture](architecture.md) for the diagram and per-step description.

## Error types

The public exceptions live in `img2svg.errors` and are caught at the API boundary:

- `Img2SvgError` — base class for all library errors.
- `UnsupportedFormatError` — input file extension not in the supported set.
- `CorruptImageError` — Pillow could not decode the input.
- `ModelLoadError` — YOLO weights could not be loaded.
- `ConfigError` — invalid configuration in the user's config file.
- `DeviceUnavailableError` — requested device is not present.
- `OutputPathCollisionError` — `--no-clobber` and the output already exists.

The CLI maps each to a specific exit code (0, 1, 2, or 3). The Python API lets the caller decide what to do — typically `try / except` around the call and fall back to a `ConversionResult` with non-empty `errors`.
