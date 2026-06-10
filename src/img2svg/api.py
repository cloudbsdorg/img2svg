"""High-level public API for img2svg.

This module is the public entry point for the library. The two functions
here — `convert()` and `convert_batch()` — are the only things most
callers need. Both are thin wrappers around `Pipeline`.

`convert_batch()` is intentionally minimal in T19. T20 (Batch enhancement)
will add: per-file error continuation, progress reporting, recursive
directory walking, and glob-based input handling for directory paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from img2svg.logging import get_logger
from img2svg.models import ConversionOptions, ConversionResult
from img2svg.pipeline import Pipeline

_logger = get_logger("img2svg.api")


def _build_options(
    options: ConversionOptions | None, kwargs: dict[str, object]
) -> ConversionOptions:
    """Return `options` if provided, else build a `ConversionOptions` from kwargs.

    Unknown kwargs are silently dropped so that callers can pass CLI-style
    extras without breaking the API. Only fields declared on
    `ConversionOptions.model_fields` are forwarded.
    """
    if options is not None:
        return options
    valid = {k: v for k, v in kwargs.items() if k in ConversionOptions.model_fields}
    return ConversionOptions(**valid)


def _resolve_output_dir(
    inputs: list[Path], output_dir: Path | None
) -> Path:
    """Pick the output directory when one is not provided.

    - For a glob-expanded list, `inputs[0].parent` is the glob's parent.
    - For a list of explicit paths, `inputs[0].parent` is the same.
    Falls back to the current working directory if `inputs` is empty.
    """
    if output_dir is not None:
        return output_dir
    if not inputs:
        return Path.cwd()
    return inputs[0].parent


def _expand_glob(inputs: str) -> list[Path]:
    """Expand a glob pattern string into a sorted list of paths."""
    p = Path(inputs)
    base = p.parent if p.parent != Path() else Path.cwd()
    return sorted(base.glob(p.name))


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    options: ConversionOptions | None = None,
    **kwargs: object,
) -> ConversionResult:
    """Convert a single image to SVG.

    Convenience wrapper around `Pipeline`. Pass either an `options` object
    or any subset of `ConversionOptions` fields as kwargs. Unknown kwargs
    are ignored.

    Args:
        input_path: Source image file.
        output_path: Destination `.svg` file.
        options: Pre-built `ConversionOptions` (overrides kwargs).
        **kwargs: Any `ConversionOptions` field (e.g. `mode="labels"`,
            `conf=0.3`, `device="cpu"`).

    Returns:
        `ConversionResult` with `svg_path`, `sidecar_path`, `sidecar`,
        and `detections`.

    Raises:
        img2svg.errors.UnsupportedFormatError: Input format not supported.
        img2svg.errors.CorruptImageError: Pillow could not decode input.
    """
    opts = _build_options(options, dict(kwargs))
    pipeline = Pipeline(opts)
    in_path = Path(input_path)
    out_path = Path(output_path)
    _logger.info("convert: %s -> %s (mode=%s)", in_path, out_path, opts.mode)
    return pipeline.run(in_path, out_path)


def convert_batch(
    inputs: list[str | Path] | str,
    output_dir: str | Path | None = None,
    *,
    options: ConversionOptions | None = None,
    **kwargs: object,
) -> list[ConversionResult]:
    """Convert a batch of images.

    Basic version (T19). T20 adds error continuation, progress reporting,
    and recursive directory walking.

    Behavior:
      - If `inputs` is a string, it is treated as a glob pattern relative
        to the current working directory.
      - For each input, the output SVG is written into `output_dir` with
        the same stem and a `.svg` extension. If `output_dir` is not
        provided, falls back to `inputs[0].parent` (the glob parent for
        glob input, or the first file's directory for a list).
      - All files are processed; the first failure propagates. T20 will
        change this to per-file error continuation.

    Args:
        inputs: List of input paths, or a glob pattern string.
        output_dir: Directory for output SVGs (default: `inputs[0].parent`).
        options: Pre-built `ConversionOptions` (overrides kwargs).
        **kwargs: Any `ConversionOptions` field.

    Returns:
        List of `ConversionResult` in input order.
    """
    if isinstance(inputs, str):
        paths: list[Path] = _expand_glob(inputs)
    else:
        paths = [Path(p) for p in inputs]

    out_dir = _resolve_output_dir(paths, Path(output_dir) if output_dir is not None else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = _build_options(options, dict(kwargs))
    pipeline = Pipeline(opts)

    results: list[ConversionResult] = []
    for p in paths:
        out = out_dir / (p.stem + ".svg")
        _logger.info("convert_batch: %s -> %s", p, out)
        results.append(pipeline.run(p, out))
    return results
