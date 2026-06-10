# Changelog

All notable changes to `img2svg` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Full MkDocs documentation under `docs/`, published at `https://cloudbsdorg.github.io/img2svg/`.
- Man page (`man/img2svg.1`) in roff/groff format.
- `img2svg list-gpus` subcommand with Rich-rendered table and recommendation highlight.

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
