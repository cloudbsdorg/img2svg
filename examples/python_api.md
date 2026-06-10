# Python API

`img2svg` ships as a library too. The two public functions, `convert` and `convert_batch`, mirror the CLI's single-file and batch paths. Both accept a `ConversionOptions` object or any subset of its fields as keyword arguments.

## 1. Single file via `convert()`

The simplest call. `result` is a `ConversionResult` with the output paths, the parsed `Sidecar`, and the raw detection list.

```python
from img2svg import convert

result = convert("tests/fixtures/logo.png", "out/logo.svg")
print(result.svg_path)      # PosixPath('out/logo.svg')
print(result.sidecar_path)  # PosixPath('out/logo.json')
print(result.errors)        # [] on success
```

The output directory is created on demand; `convert` will not overwrite an existing file unless you set `no_clobber=False` (the default) and the path is writable.

## 2. Batch via `convert_batch()`

Pass a glob string, a directory, or an explicit list. A Rich progress bar is available but off by default.

```python
from img2svg import convert_batch

results = convert_batch(
    "tests/fixtures/*.png",
    output_dir="out/batch",
    show_progress=True,
)

for r in results:
    if r.errors:
        print(f"FAIL: {r.svg_path.name}: {r.errors[0]}")
    else:
        print(f"OK:   {r.svg_path.name}")
```

By default, per-file errors are recorded in `result.errors` and the batch continues. Set `continue_on_error=False` to re-raise the first failure instead.

## 3. Custom `ConversionOptions`

Build the options object once and reuse it across many calls. The Pydantic model validates ranges (e.g. `conf` in `[0.0, 1.0]`, `palette_size` in `[2, 64]`).

```python
from img2svg import convert, ConversionOptions

options = ConversionOptions(
    mode="annotated",
    model="yolo11x.pt",
    device="cuda:0",
    conf=0.35,
    gpu_strategy="availability",
    palette_size=12,
)

result = convert("tests/fixtures/diagram.png", "out/diagram.svg", options=options)
```

You can also pass any `ConversionOptions` field as a kwarg to `convert` or `convert_batch`. Unknown kwargs are silently dropped, which keeps CLI-style call sites working without breaking the API.

## 4. Reading the sidecar JSON

Every SVG is paired with a `.json` sidecar holding version, mode, model, device, image type, detections, geometric analysis, and per-step timings. The `result.sidecar` attribute is the parsed Pydantic model, so you can read fields directly without manual JSON loading.

```python
from img2svg import convert
from img2svg.enums import ImageType

result = convert("tests/fixtures/diagram.png", "out/diagram.svg", mode="labels")
sidecar = result.sidecar

print(sidecar.mode_used)        # <Mode.LABELS: 'labels'>
print(sidecar.image_type)       # <ImageType.DIAGRAM: 'diagram'>
print(sidecar.detections)       # [Detection(class_id=..., ...), ...]
print(sidecar.timings["total"]) # 1.77 (seconds)
```

If you only have the path, load the sidecar yourself with `Sidecar.model_validate_json(path.read_text())`. The on-disk JSON is identical to `result.sidecar.model_dump_json()`.
