# img2svg - high-level public API for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""High-level public API for img2svg.

This module is the public entry point for the library. The two functions
here — `convert()` and `convert_batch()` — are the only things most
callers need. Both are thin wrappers around `Pipeline`.

`convert_batch()` accepts any of:
  - a list of paths
  - a glob pattern string (``"img/*.png"``)
  - a single image file path
  - a directory (walked non-recursively by default)

By default only files with supported image extensions
(``.png .jpg .jpeg .bmp .webp .tiff .tif .gif``) are processed. Use
``recursive=True`` to descend into subdirectories, ``continue_on_error=False``
to fail fast on the first error, and ``show_progress=True`` to render a
Rich progress bar over the batch.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from img2svg.enums import ImageType, Mode
from img2svg.logging import get_logger, progress_bar
from img2svg.models import (
    ConversionOptions,
    ConversionResult,
    Sidecar,
)
from img2svg.pipeline import Pipeline

_logger = get_logger("img2svg.api")


# Extensions accepted by `convert_batch` discovery. Comparison is
# case-insensitive via `.lower()` so ``.PNG`` and ``.Jpg`` are honored.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
)
_GLOB_META: frozenset[str] = frozenset("*?[")


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


def _is_glob_pattern(s: str) -> bool:
    """True if `s` contains any glob meta-character (`*`, `?`, `[`)."""
    return any(c in s for c in _GLOB_META)


def _is_supported_image(path: Path) -> bool:
    """True if `path.suffix` is in `SUPPORTED_EXTENSIONS` (case-insensitive)."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _expand_glob(inputs: str) -> list[Path]:
    """Expand a glob pattern string into a sorted list of paths."""
    p = Path(inputs)
    base = p.parent if p.parent != Path() else Path.cwd()
    return sorted(base.glob(p.name))


def _resolve_output_dir(inputs: list[Path], output_dir: Path | None) -> Path:
    # T19-era shim — kept so existing callers/tests keep working. New
    # code should use `_default_output_dir`, which also handles string
    # `inputs`.
    if output_dir is not None:
        return output_dir
    if not inputs:
        return Path.cwd()
    return inputs[0].parent


def _normalize_inputs(
    inputs: list[str | Path] | str,
    recursive: bool,
) -> list[Path]:
    """Resolve `inputs` to a sorted, de-duplicated list of supported image paths.

    The shape of `inputs` may be:
      - ``list[str | Path]``         → use as-is, filter to supported formats
      - ``str`` with glob meta-chars → ``Path.glob()`` (or ``rglob`` if recursive)
      - ``str`` ending in a supported extension → single file
      - ``str`` pointing to an existing directory → walked with ``glob``
        (or ``rglob`` if recursive)

    Non-image files are dropped, duplicates are removed, and the final
    list is sorted for deterministic batch order.
    """
    if isinstance(inputs, str):
        if _is_glob_pattern(inputs):
            p = Path(inputs)
            base = p.parent if p.parent != Path() else Path.cwd()
            iterator = base.rglob(p.name) if recursive else base.glob(p.name)
            candidates: list[Path] = sorted(iterator)
        else:
            p = Path(inputs)
            if p.is_dir():
                iterator = p.rglob("*") if recursive else p.glob("*")
                candidates = sorted(iterator)
            else:
                candidates = [p]
    else:
        candidates = [Path(p) for p in inputs]

    seen: set[Path] = set()
    result: list[Path] = []
    for p in candidates:
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        if resolved in seen:
            continue
        if _is_supported_image(p):
            seen.add(resolved)
            result.append(p)
    result.sort(key=lambda x: str(x))
    return result


def _default_output_dir(inputs: list[str | Path] | str, paths: list[Path]) -> Path:
    """Pick the output directory when one is not provided.

    Resolution order:
      - For a single-file string (no glob chars, not a directory), use
        that file's parent so the SVG lands next to the source.
      - For a directory-string input, use that directory itself.
      - For a glob pattern or an explicit list, use the parent of the
        first resolved path.
      - Empty input list falls back to ``Path.cwd()``.
    """
    if isinstance(inputs, str) and not _is_glob_pattern(inputs):
        p = Path(inputs)
        return p if p.is_dir() else p.parent
    if not paths:
        return Path.cwd()
    return paths[0].parent


def _make_error_result(
    input_path: Path, output_path: Path, error: BaseException
) -> ConversionResult:
    """Build a `ConversionResult` whose `errors` field carries the exception.

    Used by `convert_batch` to record per-file failures without aborting
    the entire batch. The `sidecar` field is a minimal placeholder — the
    `errors` list is the source of truth for failure handling.
    """
    # `sidecar` is a minimal placeholder — the `errors` list is the
    # source of truth for failure handling.
    placeholder = Sidecar(
        version="0.0.0",
        input_path=input_path,
        input_hash="",
        output_path=output_path,
        output_size=0,
        mode_used=Mode.AUTO,
        mode_reasoning="conversion failed before mode resolution",
        model="",
        device="",
        image_type=ImageType.UNKNOWN,
    )
    return ConversionResult(
        svg_path=output_path,
        sidecar_path=output_path.with_suffix(".json"),
        sidecar=placeholder,
        detections=[],
        errors=[str(error) or type(error).__name__],
    )


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
    recursive: bool = False,
    continue_on_error: bool = True,
    show_progress: bool = False,
    **kwargs: object,
) -> list[ConversionResult]:
    """Convert a batch of images.

    `inputs` may be:
      - a list of file paths (each filtered to supported extensions)
      - a glob pattern string (e.g. ``"img/*.png"``)
      - a path to a single image file
      - a path to a directory (walked non-recursively by default)

    In every case, only files with supported image extensions
    (``.png .jpg .jpeg .bmp .webp .tiff .tif .gif``) are processed.
    Outputs are written to ``output_dir`` (or to a sensible default —
    the input file's parent for a single file, the directory itself
    for a directory input, or ``inputs[0].parent`` for a list/glob).

    Args:
        inputs: List of paths, a glob pattern, a single file, or a directory.
        output_dir: Directory for output SVGs. If ``None``, falls back to
            a value derived from ``inputs`` (see above).
        options: Pre-built `ConversionOptions` (overrides kwargs).
        recursive: Walk input directories recursively.
        continue_on_error: If ``True`` (default), per-file errors are
            recorded in the result's ``errors`` field and processing
            continues. If ``False``, the first error re-raises.
        show_progress: If ``True``, render a Rich progress bar over the
            batch via ``img2svg.logging.progress_bar``.
        **kwargs: Any `ConversionOptions` field.

    Returns:
        List of `ConversionResult` in deterministic input order. Failed
        files have non-empty ``errors``.
    """
    paths = _normalize_inputs(inputs, recursive=recursive)

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = _default_output_dir(inputs, paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = _build_options(options, dict(kwargs))
    pipeline = Pipeline(opts)

    results: list[ConversionResult] = []
    total = len(paths)
    progress_cm = (
        progress_bar(total, description="Converting") if show_progress else nullcontext(None)
    )

    with progress_cm as progress:
        for i, p in enumerate(paths, start=1):
            out = out_dir / (p.stem + ".svg")
            _logger.info("Processing %s (%d/%d)", p, i, total)
            try:
                results.append(pipeline.run(p, out))
            except Exception as exc:
                _logger.error("Failed: %s — %s", p, exc)
                if not continue_on_error:
                    raise
                results.append(_make_error_result(p, out, exc))
            finally:
                if progress is not None:
                    progress.advance(progress.task_ids[0], 1)

    successes = sum(1 for r in results if not r.errors)
    failures = total - successes
    _logger.info("Batch complete: %d succeeded, %d failed", successes, failures)
    return results
