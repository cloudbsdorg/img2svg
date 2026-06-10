# Usage

The `img2svg` command-line tool covers the common cases; the Python API in [`img2svg.api`](api.md) covers everything else. This page is a tour of the CLI, the most common flag combinations, and the input shapes that `convert_batch` accepts.

## Basic conversion

The simplest invocation takes a single image and writes a single SVG:

```bash
img2svg photo.png -o photo.svg
```

The output is the rendered SVG plus a `photo.svg.json` sidecar with detection metadata. The sidecar is written next to the SVG by default. The long form of the output flag is `--output` (`-o` is the short alias).

## Batch conversion

`img2svg` accepts three batch input shapes and picks the right code path automatically.

A directory (non-recursive by default):

```bash
img2svg photos/ -o svg-out/
```

A glob pattern (recursive when the pattern crosses a directory boundary):

```bash
img2svg "photos/**/*.png" -o svg-out/
```

A list of explicit files (no batch support at the CLI level, but the Python API handles this directly):

```python
from img2svg import convert_batch

convert_batch(["a.png", "b.png"], output_dir="svg-out/")
```

Only files with supported extensions (`.png .jpg .jpeg .bmp .webp .tiff .tif .gif`) are processed. Unsupported files are silently skipped. The case of the extension is ignored, so `FOO.PNG` is accepted.

## Output modes

`--mode` controls how the SVG is rendered. The default `auto` runs the classifier and picks a mode based on the image type. The full set:

| Mode         | When to use                                                  |
|--------------|--------------------------------------------------------------|
| `auto`       | Default. Lets the classifier pick `labels`/`visual`/etc.     |
| `labels`     | Logos and clean line art. Bounding boxes with class labels.  |
| `visual`     | Photos. Vectorizes the image with vtracer.                  |
| `annotated`  | Photos with detectable objects. Vectorize + label overlays.  |
| `trace`      | Sketches and line art. Path-only output, no labels.         |

See [Output modes](modes.md) for examples and the underlying renderer mapping.

```bash
img2svg logo.png -o logo.svg --mode labels
img2svg photo.jpg -o photo.svg --mode visual
img2svg group.png -o group.svg --mode annotated
```

## GPU selection

`--device` picks the runtime device. The default `auto` queries the OS, picks the best GPU per the strategy, and falls back to CPU if no GPU is available.

```bash
img2svg photo.png -o photo.svg --device cpu
img2svg photo.png -o photo.svg --device cuda
img2svg photo.png -o photo.svg --device cuda:0
img2svg photo.png -o photo.svg --device mps
```

`--gpu-strategy` controls which GPU `auto` picks when more than one is present.

| Strategy       | Meaning                                  |
|----------------|------------------------------------------|
| `power`        | Largest total VRAM.                      |
| `availability` | Largest free VRAM right now.             |
| `auto`         | First device, in index order.            |

For a full list of detected devices, run:

```bash
img2svg list-gpus --strategy power
```

## Confidence threshold and model

`--conf` is the YOLO confidence threshold (0.0 to 1.0). Detections below the threshold are dropped. The default is `0.25`, which is a good balance for general photos; lower it to `0.10` to catch more objects at the risk of false positives.

`--model` picks the YOLO weights file. The default is `yolo11x.pt`, which is the largest and most accurate. Smaller models (`yolo11n.pt`, `yolo11s.pt`) are faster but less accurate.

```bash
img2svg photo.png -o photo.svg --conf 0.4 --model yolo11n.pt
```

## Output control

`--no-clobber` refuses to overwrite an existing output file. The CLI exits with code 2 and a clear error message naming the file:

```bash
img2svg photo.png -o existing.svg --no-clobber
```

In a batch, the offending file is recorded as a failure in the result list and processing continues. If you want to force-overwrite, omit `--no-clobber`.

## Verbosity

`--verbose` (or `-v`) turns on debug logging. `--quiet` (or `-q`) suppresses non-essential output. The defaults print the conversion result and a per-file summary in batch mode.

```bash
img2svg photo.png -o photo.svg -v
img2svg photos/ -o svg-out/ -q
```

## Subcommands

The CLI has three top-level subcommands:

- `img2svg convert` (the default when no subcommand is given) — convert one or more images.
- `img2svg list-gpus` — print a table of detected GPUs and the recommended device.
- `img2svg info` — print the version, Python version, OS, and compute devices.

`img2svg --version` prints the version string and exits 0. It is `is_eager` so it works without a subcommand and without a positional input.

## Exit codes

| Code | Meaning                                                |
|------|--------------------------------------------------------|
| 0    | Success (or batch with no failures).                   |
| 1    | Partial failure (batch with at least one bad file).    |
| 2    | Invalid args, unsupported format, file not found.      |
| 3    | Dependency or model load failure.                      |

## Next steps

- For the programmatic interface, see [Python API](api.md).
- For mode selection details, see [Output modes](modes.md).
- For GPU-specific setup, see [GPU setup](gpu.md).
- For a per-flag reference, see `man img2svg` (after install) or `docs/man/img2svg.1` in the source tree.
