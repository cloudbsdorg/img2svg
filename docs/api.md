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
| `GPUInfo`           | `img2svg.models.GPUInfo`   |
| `Mode`              | `img2svg.enums.Mode`       |
| `ImageType`         | `img2svg.enums.ImageType`  |
| `DeviceStrategy`    | `img2svg.enums.DeviceStrategy` |
| `GpuVendor`         | `img2svg.enums.GpuVendor`  |

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
| `mode`          | `Mode`           | `Mode.AUTO`    | `auto`, `labels`, `visual`, `annotated`, `trace`. |
| `model`         | `str`            | `"yolo11x.pt"` | YOLO weights file.                      |
| `device`        | `str`            | `"auto"`       | `auto`, `cpu`, `cuda`, `cuda:N`, `mps`.  |
| `conf`          | `float`          | `0.25`         | Confidence threshold, 0.0 to 1.0.       |
| `iou`           | `float`          | `0.7`          | IoU threshold for NMS.                   |
| `gpu_strategy`  | `DeviceStrategy` | `POWER`        | `auto`, `power`, `availability`.        |
| `no_clobber`    | `bool`           | `False`        | Refuse to overwrite existing outputs.    |
| `force_overwrite` | `bool`         | `False`        | Force-overwrite even with `no_clobber`.  |
| `palette_size`  | `int`            | `8`            | Number of colors for vectorization, 2-64.|

The validators on `conf`, `iou`, and `palette_size` reject out-of-range values at construction time. The CLI does the same checks via option-level callbacks for friendlier error messages.

## ConversionResult

The return type of `convert()`. Pydantic v2 `BaseModel` with these fields:

- `svg_path: Path` — the SVG that was written.
- `sidecar_path: Path` — the JSON sidecar path.
- `sidecar: Sidecar` — the structured metadata record.
- `detections: list[Detection]` — YOLO detections in emission order.
- `errors: list[str]` — non-empty if the conversion failed.

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
- `device: str` — runtime device string.
- `image_type: ImageType` — `logo`, `photo`, `diagram`, `screenshot`, `line_art`, or `unknown`.
- `detections: list[Detection]` — same as `ConversionResult.detections`.
- `geometric: GeometricAnalysis | None` — dominant colors, edge density, contour count, alpha.
- `timings: dict[str, float]` — per-step timings in seconds (`load`, `classify`, `detect`, `render`, `total`, ...).
- `timestamp: str` — ISO 8601 timestamp of when the conversion ran.

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
