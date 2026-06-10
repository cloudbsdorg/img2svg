# img2svg - tests for `img2svg.gpu.print_gpu_recommendation`.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `img2svg.gpu.print_gpu_recommendation` (T22).

Verifies the Rich table output: column presence, recommended-row
highlighting, empty-list fallback, and the OS / PyTorch header lines.
"""

from __future__ import annotations

import pytest
import torch

from img2svg.enums import GpuVendor
from img2svg.gpu import print_gpu_recommendation
from img2svg.models import GPUInfo


def _fake_gpus() -> list[GPUInfo]:
    """Two GPUs designed so POWER and AVAILABILITY pick different winners.

    GPU 0 (NVIDIA RTX 3060): 12000 MB total, 18000 MB free.
    GPU 1 (AMD RX 7900): 24000 MB total, 5000 MB free.

    Names are kept short so Rich doesn't wrap them in the Name column.
    POWER picks GPU 1 (larger total VRAM).
    AVAILABILITY picks GPU 0 (larger free VRAM).
    """
    return [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="RTX 3060",
            vram_total_mb=12000,
            vram_free_mb=18000,
            utilization_pct=50.0,
        ),
        GPUInfo(
            index=1,
            vendor=GpuVendor.AMD,
            name="RX 7900",
            vram_total_mb=24000,
            vram_free_mb=5000,
            utilization_pct=80.0,
        ),
    ]


def _row_containing(output: str, needle: str) -> str:
    """Return the line of output that contains `needle`.

    Asserts exactly one match. Useful for mapping a marker character
    (e.g. "Y" in the Recommended column) back to its row in a Rich
    table that has been captured to stdout.
    """
    matches = [line for line in output.splitlines() if needle in line]
    assert len(matches) == 1, (
        f"expected exactly one line with {needle!r}, got {len(matches)}: {matches!r}"
    )
    return matches[0]


def test_power_strategy_prints_both_gpus(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POWER strategy: both GPUs appear in the table; GPU 1 is recommended."""
    monkeypatch.setattr("img2svg.gpu.list_gpus", lambda: _fake_gpus())
    print_gpu_recommendation("power")
    out = capsys.readouterr().out

    assert "RTX 3060" in out
    assert "RX 7900" in out
    assert "nvidia" in out
    assert "amd" in out

    rec_row = _row_containing(out, "Y")
    assert "RX 7900" in rec_row

    non_rec_row = _row_containing(out, "RTX 3060")
    assert "Y" not in non_rec_row


def test_availability_strategy_picks_different_gpu(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AVAILABILITY strategy: recommended GPU is GPU 0 (more free VRAM)."""
    monkeypatch.setattr("img2svg.gpu.list_gpus", lambda: _fake_gpus())
    print_gpu_recommendation("availability")
    out = capsys.readouterr().out

    rec_row = _row_containing(out, "Y")
    assert "RTX 3060" in rec_row

    non_rec_row = _row_containing(out, "RX 7900")
    assert "Y" not in non_rec_row


def test_empty_gpu_list_prints_no_gpu_detected(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty list → 'No GPU detected on this system.' instead of a table."""
    monkeypatch.setattr("img2svg.gpu.list_gpus", lambda: [])
    print_gpu_recommendation("power")
    out = capsys.readouterr().out

    assert "No GPU detected on this system." in out
    assert "Available GPUs" not in out


def test_os_info_appears_in_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OS name is printed above the table."""
    monkeypatch.setattr("img2svg.gpu.list_gpus", lambda: _fake_gpus())
    print_gpu_recommendation("power")
    out = capsys.readouterr().out

    assert "OS:" in out
    os_line = _row_containing(out, "OS:")
    suffix = os_line.split("OS:", 1)[1].strip()
    assert suffix, f"OS line had no value: {os_line!r}"


def test_torch_version_appears_in_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PyTorch version (or 'N/A') is printed above the table."""
    monkeypatch.setattr("img2svg.gpu.list_gpus", lambda: _fake_gpus())
    print_gpu_recommendation("power")
    out = capsys.readouterr().out

    assert "PyTorch:" in out
    pytorch_line = _row_containing(out, "PyTorch:")
    suffix = pytorch_line.split("PyTorch:", 1)[1].strip()
    assert suffix, f"PyTorch line had no value: {pytorch_line!r}"
    assert suffix == "N/A" or suffix == torch.__version__
