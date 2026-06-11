# img2svg - tests for the Typer CLI (`img2svg.cli`).
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Typer CLI (`img2svg.cli`).

Covers the 8 spec tests:

    1. `img2svg --help` exits 0
    2. `img2svg --version` prints version, exits 0
    3. `img2svg list-gpus --help` exits 0
    4. `img2svg info --help` exits 0
    5. `img2svg --mode bogus` exits 2 (invalid mode)
    6. `img2svg nonexistent.png` exits 2 (file not found)
    7. `img2svg fixtures/logo.png -o /tmp/out.svg` exits 0 (real conversion)
    8. `img2svg fixtures/ -o /tmp/batch_out/` exits 0 (batch)

YOLO is mocked at the same boundary as the other test suites
(`img2svg.pipeline.get_detector`) so the suite is fast and has no
model-dependency. `pytest-mock` is not installed in the active venv,
so we use the built-in `unittest.mock` with a small RAII helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

from typer.testing import CliRunner

from img2svg.cli import __version__, app
from img2svg.enums import ImageType
from img2svg.models import Detection


def _mock_detector(detections: list[Detection] | None = None) -> Any:
    """Return a MagicMock detector with the given detections (default empty)."""
    det = mock.MagicMock()
    det.detect.return_value = detections if detections is not None else []
    det.device = "cpu"
    return det


class _PatchStack:
    """Tiny RAII wrapper around a list of `mock.patch` objects.

    Usage:
        with _PatchStack([...]) as stack:
            ...  # patches active
    """

    def __init__(self, patches: list[mock._patch]) -> None:
        self._patches = patches

    def __enter__(self) -> _PatchStack:
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for p in self._patches:
            p.stop()


def _success_patches() -> list[mock._patch]:
    """Patches that make the pipeline run end-to-end with empty detections."""
    return [
        mock.patch("img2svg.pipeline.get_detector", return_value=_mock_detector()),
        mock.patch(
            "img2svg.pipeline.classify",
            return_value=(ImageType.LOGO, "forced → LOGO"),
        ),
    ]


runner = CliRunner()


# ----------------------------------------------------------------------
# Test 1: --help exits 0
# ----------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    """`img2svg --help` shows help and exits 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert "img2svg" in result.output
    assert "convert" in result.output
    assert "list-gpus" in result.output
    assert "info" in result.output


# ----------------------------------------------------------------------
# Test 2: --version prints version and exits 0
# ----------------------------------------------------------------------


def test_cli_version_prints_and_exits_zero() -> None:
    """`img2svg --version` prints `img2svg 0.1.0` and exits 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert f"img2svg {__version__}" in result.output


# ----------------------------------------------------------------------
# Test 3: list-gpus --help exits 0
# ----------------------------------------------------------------------


def test_cli_list_gpus_help_exits_zero() -> None:
    """`img2svg list-gpus --help` shows the subcommand help and exits 0."""
    result = runner.invoke(app, ["list-gpus", "--help"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert "list-gpus" in result.output.lower() or "gpu" in result.output.lower()


# ----------------------------------------------------------------------
# Test 4: info --help exits 0
# ----------------------------------------------------------------------


def test_cli_info_help_exits_zero() -> None:
    """`img2svg info --help` shows the subcommand help and exits 0."""
    result = runner.invoke(app, ["info", "--help"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert "info" in result.output.lower() or "version" in result.output.lower()


# ----------------------------------------------------------------------
# Test 5: invalid --mode exits 2
# ----------------------------------------------------------------------


def test_cli_invalid_mode_exits_two() -> None:
    """`img2svg --mode bogus` exits 2 (invalid mode value)."""
    result = runner.invoke(app, ["--mode", "bogus"])
    assert result.exit_code == 2, f"got {result.exit_code}: {result.output}"


# ----------------------------------------------------------------------
# Test 6: missing input file exits 2
# ----------------------------------------------------------------------


def test_cli_nonexistent_input_exits_two() -> None:
    """`img2svg nonexistent.png` exits 2 (file not found)."""
    result = runner.invoke(app, ["nonexistent.png"])
    assert result.exit_code == 2, f"got {result.exit_code}: {result.output}"
    assert "nonexistent" in result.output or "not found" in result.output.lower()


# ----------------------------------------------------------------------
# Test 7: real conversion (single file) exits 0
# ----------------------------------------------------------------------


def test_cli_convert_single_file_exits_zero(tmp_path: Path, fixtures_dir: Path) -> None:
    """`img2svg logo.png -o out.svg` runs a real conversion (YOLO mocked) and exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    assert logo.exists(), f"fixture missing: {logo}"

    with _PatchStack(_success_patches()):
        result = runner.invoke(app, [str(logo), "-o", str(out), "--mode", "labels"])

    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert out.exists(), f"SVG was not written: {out}"
    assert out.with_suffix(".json").exists()


# ----------------------------------------------------------------------
# Test 8: batch conversion (directory) exits 0
# ----------------------------------------------------------------------


def test_cli_convert_batch_directory_exits_zero(tmp_path: Path, fixtures_dir: Path) -> None:
    """`img2svg fixtures/ -o batch_out/` runs a batch (YOLO mocked) and exits 0.

    Uses a small directory of 2 valid PNGs so the test stays fast and
    has no model dependency. The directory is created under tmp_path
    rather than using `fixtures_dir` directly, so the test does not
    pick up `corrupt.bin` (unsupported extension) or all 6 fixtures.
    """
    in_dir = tmp_path / "batch_in"
    in_dir.mkdir()
    (in_dir / "a.png").write_bytes((fixtures_dir / "logo.png").read_bytes())
    (in_dir / "b.png").write_bytes((fixtures_dir / "diagram.png").read_bytes())

    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()

    with _PatchStack(_success_patches()):
        result = runner.invoke(app, [str(in_dir), "-o", str(out_dir), "--mode", "labels"])

    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert (out_dir / "a.svg").exists()
    assert (out_dir / "b.svg").exists()
    assert (out_dir / "a.json").exists()
    assert (out_dir / "b.json").exists()


def test_cli_subcommand_info_runs() -> None:
    """`img2svg info` (subcommand) prints env info and exits 0."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert "img2svg" in result.output
    assert "Python" in result.output
    assert "OS" in result.output


def test_cli_subcommand_info_shows_backend() -> None:
    """`img2svg info` (subcommand) prints the active compute backend."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    assert "Backend:" in result.output
    assert "(torch" in result.output


def test_cli_subcommand_list_gpus_runs() -> None:
    """`img2svg list-gpus` (subcommand) prints a table and exits 0.

    On this CI host there may be no GPU; the stub prints
    "No GPUs detected." or a table — either is fine.
    """
    result = runner.invoke(app, ["list-gpus"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_convert_no_output_for_single_file_exits_two(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """Single-file convert without --output exits 2."""
    logo = fixtures_dir / "logo.png"
    with _PatchStack(_success_patches()):
        result = runner.invoke(app, [str(logo)])
    assert result.exit_code == 2, f"got {result.exit_code}: {result.output}"


def test_cli_app_is_typer_instance() -> None:
    """The exported `app` is a Typer instance and is callable."""
    from typer import Typer

    assert isinstance(app, Typer)
    assert callable(app)


def test_cli_module_exports_version() -> None:
    """`img2svg.cli.__version__` is the canonical version string."""
    assert __version__ == "0.1.0"
    assert isinstance(__version__, str)


# ----------------------------------------------------------------------
# New preprocessing / segmentation / size flags (T1, T5, T14, T20)
# ----------------------------------------------------------------------


def test_cli_preprocess_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--preprocess bilateral --preprocess unsharp` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app,
            [
                str(logo),
                "-o",
                str(out),
                "--mode",
                "labels",
                "--preprocess",
                "bilateral",
                "--preprocess",
                "unsharp",
            ],
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_denoise_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--denoise bilateral` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--denoise", "bilateral"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_sharpen_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--sharpen unsharp` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--sharpen", "unsharp"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_max_colors_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--max-colors 8` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--max-colors", "8"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_quality_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--quality 75` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--quality", "75"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_no_preprocess_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--no-preprocess` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--no-preprocess"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_seg_model_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--seg-model yolo11s-seg` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app,
            [str(logo), "-o", str(out), "--mode", "labels", "--seg-model", "yolo11s-seg"],
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_seg_model_rejects_invalid_value(tmp_path: Path) -> None:
    """`--seg-model yolo99-seg` is rejected with exit code 2 (BadParameter)."""
    result = runner.invoke(app, ["nonexistent.png", "--seg-model", "yolo99-seg"])
    assert result.exit_code == 2, f"got {result.exit_code}: {result.output}"


def test_cli_no_seg_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--no-seg` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--no-seg"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_max_svg_size_flag_passes_through(tmp_path: Path, fixtures_dir: Path) -> None:
    """`--max-svg-size 10` is accepted and the run exits 0."""
    logo = fixtures_dir / "logo.png"
    out = tmp_path / "out.svg"
    with _PatchStack(_success_patches()):
        result = runner.invoke(
            app, [str(logo), "-o", str(out), "--mode", "labels", "--max-svg-size", "10"]
        )
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"


def test_cli_max_svg_size_rejects_out_of_range(tmp_path: Path) -> None:
    """`--max-svg-size 0` is rejected with exit code 2 (below min=1)."""
    result = runner.invoke(app, ["nonexistent.png", "--max-svg-size", "0"])
    assert result.exit_code == 2, f"got {result.exit_code}: {result.output}"


def test_cli_help_documents_all_ten_modes() -> None:
    """`img2svg convert --help` lists all 10 output modes in the --mode help text."""
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    for mode in (
        "auto",
        "labels",
        "visual",
        "annotated",
        "trace",
        "poster",
        "detailed",
        "edge",
        "watercolor",
        "segmented",
    ):
        assert mode in result.output, f"convert --help is missing mode: {mode}"


def test_cli_convert_help_documents_new_flags() -> None:
    """`img2svg convert --help` mentions all 9 new flags."""
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0, f"got {result.exit_code}: {result.output}"
    for flag in (
        "--preprocess",
        "--denoise",
        "--sharpen",
        "--max-colors",
        "--quality",
        "--no-preprocess",
        "--seg-model",
        "--no-seg",
        "--max-svg-size",
    ):
        assert flag in result.output, f"convert --help is missing flag: {flag}"
