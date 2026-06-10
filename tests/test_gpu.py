# img2svg - tests for GPU enumeration and recommendation.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for GPU enumeration and recommendation."""

from __future__ import annotations

from img2svg.enums import DeviceStrategy, GpuVendor
from img2svg.gpu import recommend_gpu
from img2svg.models import GPUInfo


def _make_gpu(idx: int, vendor: GpuVendor, name: str, total: int, free: int) -> GPUInfo:
    return GPUInfo(
        index=idx,
        vendor=vendor,
        name=name,
        vram_total_mb=total,
        vram_free_mb=free,
    )


def test_recommend_empty() -> None:
    assert recommend_gpu([], DeviceStrategy.POWER) is None
    assert recommend_gpu([], DeviceStrategy.AVAILABILITY) is None


def test_recommend_power_picks_largest_vram() -> None:
    gpus = [
        _make_gpu(0, GpuVendor.NVIDIA, "RTX 3060", 12000, 10000),
        _make_gpu(1, GpuVendor.NVIDIA, "RTX 4090", 24000, 20000),
        _make_gpu(2, GpuVendor.AMD, "RX 7900 XTX", 24000, 18000),
    ]
    rec = recommend_gpu(gpus, DeviceStrategy.POWER)
    assert rec is not None
    assert rec.vram_total_mb == 24000
    # Tie broken by index
    assert rec.index == 1


def test_recommend_availability_picks_most_free() -> None:
    gpus = [
        _make_gpu(0, GpuVendor.NVIDIA, "A", 24000, 5000),  # busy
        _make_gpu(1, GpuVendor.NVIDIA, "B", 24000, 18000),  # mostly free
        _make_gpu(2, GpuVendor.AMD, "C", 16000, 12000),
    ]
    rec = recommend_gpu(gpus, DeviceStrategy.AVAILABILITY)
    assert rec is not None
    assert rec.vram_free_mb == 18000
    assert rec.index == 1


def test_recommend_mixed_vendors() -> None:
    """User has both AMD and NVIDIA. Both should be considered."""
    gpus = [
        _make_gpu(0, GpuVendor.AMD, "RX 7900", 16000, 12000),
        _make_gpu(1, GpuVendor.NVIDIA, "RTX 4090", 24000, 20000),
    ]
    rec_power = recommend_gpu(gpus, DeviceStrategy.POWER)
    rec_avail = recommend_gpu(gpus, DeviceStrategy.AVAILABILITY)
    assert rec_power is not None and rec_power.vendor == GpuVendor.NVIDIA
    assert rec_avail is not None and rec_avail.vendor == GpuVendor.NVIDIA


def test_recommend_single_gpu() -> None:
    gpus = [_make_gpu(0, GpuVendor.NVIDIA, "Only", 8000, 4000)]
    assert recommend_gpu(gpus, DeviceStrategy.POWER) == gpus[0]
    assert recommend_gpu(gpus, DeviceStrategy.AVAILABILITY) == gpus[0]
