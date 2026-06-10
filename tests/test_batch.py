"""Tests for the enhanced `convert_batch()` in `img2svg.api`.

Covers the T20 enhancement: directory walking, glob input, per-file
error continuation, progress bar, and output directory resolution.

The YOLO detector and vtracer are mocked at the same boundaries as
the existing pipeline / API tests so the suite stays fast and has no
model-dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from img2svg.api import convert_batch
from img2svg.enums import ImageType, Mode
from img2svg.models import ConversionResult


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _mock_detector() -> Any:
    """Return a MagicMock detector with empty detections."""
    det = mock.MagicMock()
    det.detect.return_value = []
    return det


class _PatchStack:
    """Tiny RAII wrapper around a list of `mock.patch` objects.

    Usage:
        with _PatchStack([...]) as stack:
            ...  # patches active
    """

    def __init__(self, patches: list[mock._patch]) -> None:
        self._patches = patches

    def __enter__(self) -> "_PatchStack":
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for p in self._patches:
            p.stop()


def _success_patches() -> list[mock._patch]:
    """Patches that make the pipeline run end-to-end with empty detections."""
    return [
        mock.patch(
            "img2svg.pipeline.get_detector", return_value=_mock_detector()
        ),
        mock.patch(
            "img2svg.pipeline.classify",
            return_value=(ImageType.LOGO, "forced → LOGO"),
        ),
    ]


def _copy_real_png(dst: Path, src: Path) -> Path:
    """Copy a valid PNG fixture to `dst`."""
    dst.write_bytes(src.read_bytes())
    return dst


def _write_corrupt_png(dst: Path) -> Path:
    """Write garbage bytes with a `.png` extension.

    The bytes look like a PNG header but are missing the IHDR chunk,
    so PIL raises `Image.UnidentifiedImageError` → `CorruptImageError`
    when `load_image` is called.
    """
    dst.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00garbage")
    return dst


# ----------------------------------------------------------------------
# 1. List of 3 valid files
# ----------------------------------------------------------------------


def test_batch_list_of_three_valid_files(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A list of 3 supported files produces 3 successful results."""
    inputs = [
        _copy_real_png(tmp_path / "a.png", fixtures_dir / "logo.png"),
        _copy_real_png(tmp_path / "b.png", fixtures_dir / "diagram.png"),
        _copy_real_png(tmp_path / "c.png", fixtures_dir / "screenshot.png"),
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(inputs, output_dir=out_dir, mode=Mode.LABELS)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, ConversionResult)
        assert not r.errors
        assert r.svg_path.exists()
        assert r.sidecar_path.exists()
    # Output filenames preserve the input stems.
    assert {r.svg_path.name for r in results} == {"a.svg", "b.svg", "c.svg"}


# ----------------------------------------------------------------------
# 2. Mixed valid + corrupt — error continuation
# ----------------------------------------------------------------------


def test_batch_continues_on_corrupt_file(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A corrupt file produces 1 error result; the other 2 still succeed."""
    good_a = _copy_real_png(tmp_path / "good_a.png", fixtures_dir / "logo.png")
    good_b = _copy_real_png(tmp_path / "good_b.png", fixtures_dir / "diagram.png")
    bad = _write_corrupt_png(tmp_path / "bad.png")
    inputs = [good_a, bad, good_b]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(inputs, output_dir=out_dir, mode=Mode.LABELS)

    assert len(results) == 3
    successes = [r for r in results if not r.errors]
    failures = [r for r in results if r.errors]
    assert len(successes) == 2
    assert len(failures) == 1
    # The error result points at the intended output path of `bad.png`.
    assert "bad" in failures[0].svg_path.name
    # Error message should mention the corrupt image.
    assert failures[0].errors, "errors list must be non-empty"
    assert any("bad" in msg or "corrupt" in msg.lower() for msg in failures[0].errors)


# ----------------------------------------------------------------------
# 3. Directory input (non-recursive)
# ----------------------------------------------------------------------


def test_batch_directory_input_processes_supported_files(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A directory string is walked non-recursively; only images processed."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    _copy_real_png(in_dir / "alpha.png", fixtures_dir / "logo.png")
    _copy_real_png(in_dir / "beta.png", fixtures_dir / "diagram.png")
    (in_dir / "notes.txt").write_text("not an image")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(str(in_dir), output_dir=out_dir, mode=Mode.LABELS)

    assert len(results) == 2
    assert {r.svg_path.name for r in results} == {"alpha.svg", "beta.svg"}


# ----------------------------------------------------------------------
# 4. Recursive directory
# ----------------------------------------------------------------------


def test_batch_recursive_directory_walks_subdirs(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """`recursive=True` descends into subdirectories."""
    root = tmp_path / "root"
    sub = root / "sub" / "deep"
    sub.mkdir(parents=True)
    _copy_real_png(root / "top.png", fixtures_dir / "logo.png")
    _copy_real_png(sub / "nested.png", fixtures_dir / "diagram.png")
    # Unsupported file in subdir (should be ignored)
    (sub / "ignore.txt").write_text("nope")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(
            str(root),
            output_dir=out_dir,
            recursive=True,
            mode=Mode.LABELS,
        )

    assert len(results) == 2
    assert {r.svg_path.name for r in results} == {"top.svg", "nested.svg"}


def test_batch_non_recursive_skips_subdirs(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """`recursive=False` (default) does NOT descend into subdirectories."""
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True)
    _copy_real_png(root / "top.png", fixtures_dir / "logo.png")
    _copy_real_png(sub / "nested.png", fixtures_dir / "diagram.png")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(
            str(root), output_dir=out_dir, mode=Mode.LABELS
        )

    # Only the top-level file is found.
    assert len(results) == 1
    assert results[0].svg_path.name == "top.svg"


# ----------------------------------------------------------------------
# 5. Glob pattern
# ----------------------------------------------------------------------


def test_batch_glob_pattern_matches_only_listed_extensions(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A `*.png` glob pattern picks only PNG files, not JPGs in the same dir."""
    _copy_real_png(tmp_path / "a.png", fixtures_dir / "logo.png")
    _copy_real_png(tmp_path / "b.png", fixtures_dir / "diagram.png")
    _copy_real_png(tmp_path / "ignore.jpg", fixtures_dir / "photo.jpg")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(
            str(tmp_path / "*.png"),
            output_dir=out_dir,
            mode=Mode.LABELS,
        )

    assert len(results) == 2
    assert {r.svg_path.name for r in results} == {"a.svg", "b.svg"}


# ----------------------------------------------------------------------
# 6. Single file string
# ----------------------------------------------------------------------


def test_batch_single_file_string(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A single supported-extension string is treated as one file."""
    img = _copy_real_png(tmp_path / "only.png", fixtures_dir / "logo.png")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(str(img), output_dir=out_dir, mode=Mode.LABELS)

    assert len(results) == 1
    assert results[0].svg_path.name == "only.svg"


# ----------------------------------------------------------------------
# 7. Unsupported format filter
# ----------------------------------------------------------------------


def test_batch_unsupported_format_files_are_filtered(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """Non-image extensions in a list input are dropped before processing."""
    _copy_real_png(tmp_path / "keep.png", fixtures_dir / "logo.png")
    (tmp_path / "skip.txt").write_text("hello")
    (tmp_path / "skip.json").write_text("{}")
    inputs = [
        tmp_path / "keep.png",
        tmp_path / "skip.txt",
        tmp_path / "skip.json",
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        results = convert_batch(inputs, output_dir=out_dir, mode=Mode.LABELS)

    assert len(results) == 1
    assert results[0].svg_path.name == "keep.svg"


# ----------------------------------------------------------------------
# 8. continue_on_error=False re-raises
# ----------------------------------------------------------------------


def test_batch_reraises_when_continue_on_error_false(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """A failure with `continue_on_error=False` propagates the exception."""
    bad = _write_corrupt_png(tmp_path / "bad.png")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        with pytest.raises(Exception):
            convert_batch(
                [bad],
                output_dir=out_dir,
                mode=Mode.LABELS,
                continue_on_error=False,
            )


# ----------------------------------------------------------------------
# 9. show_progress=True uses progress_bar
# ----------------------------------------------------------------------


def test_batch_show_progress_invokes_progress_bar(
    tmp_path: Path,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`show_progress=True` calls `progress_bar` from `img2svg.api`."""
    img = _copy_real_png(tmp_path / "img.png", fixtures_dir / "logo.png")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Build a fake progress object with a real `task_ids` list and
    # a stub `advance` method.
    fake_progress = mock.MagicMock()
    fake_progress.task_ids = [0]
    fake_progress.advance = mock.MagicMock()

    # Build a context-manager mock whose __enter__ returns the progress.
    fake_cm = mock.MagicMock()
    fake_cm.__enter__ = mock.MagicMock(return_value=fake_progress)
    fake_cm.__exit__ = mock.MagicMock(return_value=False)

    progress_mock = mock.MagicMock(return_value=fake_cm)
    monkeypatch.setattr("img2svg.api.progress_bar", progress_mock)

    with _PatchStack(_success_patches()):
        results = convert_batch(
            [img],
            output_dir=out_dir,
            mode=Mode.LABELS,
            show_progress=True,
        )

    assert len(results) == 1
    progress_mock.assert_called_once()
    fake_progress.advance.assert_called_once()


def test_batch_no_progress_bar_when_show_progress_false(
    tmp_path: Path,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`show_progress=False` (default) does NOT invoke `progress_bar`."""
    img = _copy_real_png(tmp_path / "img.png", fixtures_dir / "logo.png")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    progress_mock = mock.MagicMock()
    monkeypatch.setattr("img2svg.api.progress_bar", progress_mock)

    with _PatchStack(_success_patches()):
        results = convert_batch(
            [img], output_dir=out_dir, mode=Mode.LABELS
        )

    assert len(results) == 1
    progress_mock.assert_not_called()


# ----------------------------------------------------------------------
# 10. output_dir override
# ----------------------------------------------------------------------


def test_batch_output_dir_override_creates_and_uses_custom_dir(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """An explicit `output_dir` is created if missing and used for outputs."""
    img = _copy_real_png(tmp_path / "img.png", fixtures_dir / "logo.png")
    custom_out = tmp_path / "custom" / "nested" / "out"
    assert not custom_out.exists()

    with _PatchStack(_success_patches()):
        results = convert_batch(
            [img], output_dir=custom_out, mode=Mode.LABELS
        )

    assert len(results) == 1
    assert custom_out.exists()
    assert custom_out.is_dir()
    assert results[0].svg_path.parent == custom_out
    assert results[0].svg_path.name == "img.svg"


def test_batch_default_output_dir_for_list_input(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """When `output_dir` is None and `inputs` is a list, fall back to
    `inputs[0].parent`."""
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    a = _copy_real_png(in_dir / "a.png", fixtures_dir / "logo.png")
    b = _copy_real_png(in_dir / "b.png", fixtures_dir / "diagram.png")

    with _PatchStack(_success_patches()):
        results = convert_batch([a, b], mode=Mode.LABELS)

    assert len(results) == 2
    assert (in_dir / "a.svg").exists()
    assert (in_dir / "b.svg").exists()
