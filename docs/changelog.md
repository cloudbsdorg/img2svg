# Changelog

All notable changes to `img2svg` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Five new output modes** for photographs and complex raster sources:
  - `poster` — flat, stylized color regions (vtracer `poster` preset).
  - `detailed` — high-fidelity photo trace (vtracer `photo_hifi` preset). The new `auto` default for `photo` image types.
  - `edge` — line-art output (vtracer `bw_edge` preset in polygon mode).
  - `watercolor` — soft painterly output (vtracer `watercolor` preset).
  - `segmented` — multi-layer editable SVG with one `<g>` per YOLO-detected object.
- **Optional OpenCV preprocessing pipeline** that runs before the renderer. Chain denoise, sharpen, posterize, and edge detection filters via the new `--preprocess`, `--denoise`, and `--sharpen` flags.
- **YOLO instance segmentation** with a dedicated `YOLOSegmentor`. Five model variants: `yolo11n-seg`, `yolo11s-seg` (default), `yolo11m-seg`, `yolo11l-seg`, `yolo11x-seg`. Auto-falls-back to a smaller model on out-of-memory errors.
- **Eight new CLI flags**:
  - `--preprocess` (repeatable) for chaining preprocessing filters.
  - `--denoise` as a single-filter shortcut (`bilateral`, `nlmeans`, `median`).
  - `--sharpen` as a single-filter shortcut (`unsharp`).
  - `--max-colors` to cap the output palette (0-256).
  - `--quality` as a JPEG-style quality hint stored in the sidecar.
  - `--no-preprocess` to disable all preprocessing.
  - `--seg-model` to pick the YOLO segmentation variant.
  - `--no-seg` to disable segmentation even when the active mode would use it.
  - `--max-svg-size` to cap the rendered SVG size in MB (1-1024, default 50).
- **Per-region metadata** in the sidecar JSON via the new `regions` field. Each region carries class, confidence, bounding box, mask area, and polygon vertices.
- **Preprocessing trace** in the sidecar JSON via the new `preprocessing` field (list of filter names that ran, in order).
- **Model variant** in the sidecar JSON via the new `model_variant` field (e.g. `yolo11s-seg`).
- **RegionInfo** and **SegmentationResult** types in the Python API.
- **Photo modes documentation** at `docs/photo-modes.md`, with mode-by-mode deep dives, the preprocessing chain reference, and the segmentation workflow.

### Changed

- The `auto` mode no longer resolves to `labels` or `annotated`. The new mapping is:
  - `photo` → `detailed` (aggressive default).
  - `logo`, `diagram`, `screenshot`, `line_art`, `unknown` → `visual`.
- The `palette_size` field on `ConversionOptions` was replaced with `max_colors` (with proper vtracer wiring) and a new `quality` field.
- The pipeline grows from 12 steps to 14 steps to accommodate the new preprocessing and segmentation stages. See [Architecture](architecture.md) for the updated flowchart.
- The `timings` dict in the sidecar may now include `preprocess` and `segment` entries.

## [0.1.0] — 2026-06-10

### Added

- Initial public release.
- `convert()` and `convert_batch()` Python API for converting single images and batches.
- Typer-based CLI with a default `convert` subcommand plus `list-gpus` and `info` subcommands.
- Five output modes: `auto`, `labels`, `visual`, `annotated`, `trace`.
- GPU detection for NVIDIA (via `nvidia-smi`), AMD (via `rocm-smi`), Apple Silicon (via `torch.backends.mps`), and a PyTorch CUDA fallback.
- GPU recommendation strategies: `power` (largest total VRAM), `availability` (largest free VRAM), `auto` (first device by index).
- YOLO segmentation with `yolo11x.pt` as the default weights; configurable via `--model`.
- vtracer-backed vectorization for `visual`, `annotated`, and `trace` modes.
- Atomic file writes for both SVG and sidecar JSON outputs.
- XDG Base Directory path resolution with a CloudBSD-specific system-wide fallback at `/usr/local/etc/cloudbsd/img2svg/`.
- Per-step timings recorded in the sidecar JSON.
- Per-batch error continuation (default `continue_on_error=True`) with a per-file error summary at the CLI.
- Pre-built wheels for Linux x86_64 (CPU and CUDA), macOS arm64, and macOS x86_64.
- `img2svg info` subcommand printing version, Python, OS, and detected devices.
- Test suite with 200+ tests covering pipeline, API, CLI, renderers, GPU detection, paths, errors, and metadata.
- Pre-commit hooks (Black, isort, Ruff, mypy).
- `pyproject.toml` with hatchling backend and explicit dependency pins.

### Known limitations

- HEIC, AVIF, RAW, and PSD input formats are not supported (only PNG, JPEG, BMP, WebP, TIFF, GIF).
- `Pipeline` is not thread-safe. Use multiple processes to parallelize a batch.
- The classifier is a non-ML heuristic. Auto-mode may pick an unexpected mode for unusual images; force a mode explicitly when in doubt.
- The vendored vtracer is built with the default color palette (8 colors). `palette_size` is exposed on `ConversionOptions` but not yet wired through to vtracer's parameters.
- Per-ROI analysis (originally step 6 in the pipeline spec) is currently a no-op. The renderers receive the global analysis only.
- Documentation is hosted on GitHub Pages; there is no readthedocs mirror yet.

[Unreleased]: https://github.com/cloudbsdorg/img2svg/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cloudbsdorg/img2svg/releases/tag/v0.1.0
