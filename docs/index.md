# img2svg

Convert raster images to clean, optimized SVG using YOLO segmentation and [vtracer](https://github.com/visioncortex/vtracer).

`img2svg` reads a PNG, JPEG, BMP, WebP, TIFF, or GIF and writes a hand-tunable SVG plus a JSON sidecar with detection metadata, geometric analysis, and per-step timings. It classifies each image and picks a rendering mode automatically, or you can force a specific mode for predictable output.

## Highlights

- **Ten output modes**: `auto`, `labels`, `visual`, `annotated`, `trace`, `poster`, `detailed`, `edge`, `watercolor`, and `segmented`. The classifier picks a sensible default for `auto`; the photo modes give you a stylistic and fidelity dial for photographs and other complex sources.
- **Auto mode selection**: the classifier inspects the image and routes `photo` to `detailed` and other image types to `visual`. Explicit-only modes (`labels`, `annotated`, `segmented`) are never picked automatically.
- **Optional preprocessing**: an OpenCV filter chain (denoise, sharpen, posterize, edge detection) runs before vtracer for cleaner output on noisy sources.
- **Multi-layer editable SVG**: `segmented` mode runs YOLO instance segmentation, traces each detected object with vtracer, and emits one `<g>` group per object.
- **Batch-friendly**: a single call walks a directory, expands a glob, or processes a list of paths.
- **GPU-aware**: detects NVIDIA, AMD, and Apple Silicon devices, with a recommendation strategy (`power` or `availability`).
- **Atomic writes**: every output is written to a temp file and renamed into place, so partial outputs never appear in the final directory.
- **No surprises**: per-file errors are recorded in the result list instead of aborting the whole batch.

## Installation

Install from PyPI with `pip`, or build from source for the latest changes. See [Installation](installation.md) for platform notes (Linux, FreeBSD, macOS).

```bash
pip install img2svg
```

## Quick start

Convert a single image:

```bash
img2svg photo.png -o photo.svg
```

Convert a whole directory:

```bash
img2svg photos/ -o svg-out/
```

Use a specific mode and device:

```bash
img2svg logo.png -o logo.svg --mode visual --device cuda
```

See [Usage](usage.md) for the full flag reference and batch examples, or jump straight to the [Python API](api.md).

## Documentation map

- [Installation](installation.md): install from PyPI, source, plus platform notes for FreeBSD and macOS.
- [Usage](usage.md): CLI flag reference, batch processing, GPU selection.
- [Python API](api.md): `convert()`, `convert_batch()`, `ConversionOptions`, and friends.
- [Output modes](modes.md): when to use `auto`, `labels`, `visual`, `annotated`, or `trace`.
- [Photo modes](photo-modes.md): the five photo modes (`poster`, `detailed`, `edge`, `watercolor`, `segmented`), the optional preprocessing chain, and the segmentation workflow.
- [GPU setup](gpu.md): NVIDIA CUDA, AMD ROCm, Apple Silicon MPS.
- [Configuration](configuration.md): XDG Base Directory paths and the config file.
- [Troubleshooting](troubleshooting.md): common errors and their fixes.
- [Development](development.md): dev setup, TDD workflow, contributing.
- [Architecture](architecture.md): pipeline internals with diagrams.
- [Changelog](changelog.md): version history.

## Project information

- Source: `https://github.com/cloudbsdorg/img2svg`
- Issues: `https://github.com/cloudbsdorg/img2svg/issues`
- License: BSD 3-Clause. Copyright (c) 2026, REVYTECH, Inc.
- Status: pre-alpha. APIs may change between minor versions.
