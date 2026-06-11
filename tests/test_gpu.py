# img2svg - tests for GPU enumeration and recommendation.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for GPU enumeration and recommendation."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from img2svg.enums import DeviceStrategy, GpuVendor
from img2svg.gpu import (
    _parse_lspci,
    _parse_rocm_smi,
    _parse_rocminfo,
    list_gpus,
    recommend_gpu,
)
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


def test_parse_lspci_detects_amd_igpu(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/lspci")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        gpu_mod.subprocess,
        "check_output",
        lambda *_a, **_k: (
            "c3:00.0 Display controller [0380]: Advanced Micro Devices, Inc. "
            "[AMD/ATI] Strix [Radeon 880M / 890M] [1002:150e] (rev c1)\n"
        ),
    )
    gpus = _parse_lspci()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.AMD
    assert gpus[0].name == "AMD Radeon 880M / 890M"
    assert gpus[0].vram_total_mb == 0
    assert gpus[0].vram_free_mb == 0
    assert gpus[0].utilization_pct is None
    assert gpus[0].compute_capability is None


def test_parse_lspci_detects_nvidia(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/lspci")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        gpu_mod.subprocess,
        "check_output",
        lambda *_a, **_k: (
            "c2:00.0 VGA compatible controller [0300]: NVIDIA Corporation "
            "GB206M [GeForce RTX 5070 Max-Q / Mobile] [10de:2d58] (rev a1)\n"
        ),
    )
    gpus = _parse_lspci()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.NVIDIA
    assert gpus[0].name == "NVIDIA GeForce RTX 5070 Max-Q / Mobile"
    assert gpus[0].vram_total_mb == 0
    assert gpus[0].utilization_pct is None


def test_parse_lspci_filters_audio_device(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/lspci")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        gpu_mod.subprocess,
        "check_output",
        lambda *_a, **_k: (
            "c3:00.0 Display controller [0380]: Advanced Micro Devices, Inc. "
            "[AMD/ATI] Strix [Radeon 880M / 890M] [1002:150e] (rev c1)\n"
            "c3:00.1 Audio device [0403]: Advanced Micro Devices, Inc. "
            "[AMD/ATI] Radeon High Definition Audio Controller [1002:1640]\n"
        ),
    )
    gpus = _parse_lspci()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.AMD
    assert "Audio" not in gpus[0].name


def test_parse_lspci_missing_binary(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: None)  # type: ignore[attr-defined]
    check_output = MagicMock(side_effect=AssertionError("lspci invoked when missing"))
    monkeypatch.setattr(gpu_mod.subprocess, "check_output", check_output)
    assert _parse_lspci() == []


def test_parse_lspci_subprocess_error(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/lspci")  # type: ignore[attr-defined]

    def _raise(*_a: object, **_k: object) -> str:
        raise subprocess.CalledProcessError(1, "lspci")

    monkeypatch.setattr(gpu_mod.subprocess, "check_output", _raise)
    assert _parse_lspci() == []


def test_parse_lspci_no_display_controllers(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/lspci")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        gpu_mod.subprocess,
        "check_output",
        lambda *_a, **_k: (
            "00:00.0 Host bridge [0600]: Intel Corporation Device [8086:a000]\n"
            "01:00.0 Network controller [0280]: Intel Corporation Wi-Fi [8086:2723]\n"
        ),
    )
    assert _parse_lspci() == []


def test_list_gpus_merges_nvidia_and_lspci(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    nvidia_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070",
            vram_total_mb=8151,
            vram_free_mb=7680,
            utilization_pct=12.0,
        )
    ]
    lspci_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="AMD Radeon 880M / 890M",
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
            compute_capability=None,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: nvidia_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: lspci_result)
    gpus = list_gpus()
    assert len(gpus) == 2
    # Both sources assigned index=0, so the stable sort preserves source priority:
    # nvidia-smi first, then lspci. list_gpus() reassigns sequential indices.
    assert gpus[0].vendor == GpuVendor.NVIDIA
    assert gpus[0].vram_total_mb == 8151
    assert gpus[0].index == 0
    assert gpus[1].vendor == GpuVendor.AMD
    assert gpus[1].name == "AMD Radeon 880M / 890M"
    assert gpus[1].vram_total_mb == 0
    assert gpus[1].utilization_pct is None
    assert gpus[1].index == 1


def test_list_gpus_merges_nvidia_and_rocm(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    nvidia_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="RTX 4090",
            vram_total_mb=24000,
            vram_free_mb=20000,
            utilization_pct=15.0,
        )
    ]
    rocm_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="RX 7900 XTX",
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: nvidia_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: rocm_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: [])
    gpus = list_gpus()
    assert len(gpus) == 2
    assert {g.vendor for g in gpus} == {GpuVendor.NVIDIA, GpuVendor.AMD}
    # Stable source priority order; list_gpus() reassigns sequential indices.
    assert gpus[0].vendor == GpuVendor.NVIDIA
    assert gpus[0].index == 0
    assert gpus[1].vendor == GpuVendor.AMD
    assert gpus[1].index == 1


def test_list_gpus_dedup_prefers_nvidia_smi(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    nvidia_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070",
            vram_total_mb=8151,
            vram_free_mb=7680,
            utilization_pct=12.0,
        )
    ]
    lspci_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070",
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
            compute_capability=None,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: nvidia_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: lspci_result)
    gpus = list_gpus()
    assert len(gpus) == 1
    assert gpus[0].vram_total_mb == 8151
    assert gpus[0].utilization_pct == 12.0


def test_list_gpus_dedup_case_insensitive(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    nvidia_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070",
            vram_total_mb=8151,
            vram_free_mb=7680,
        )
    ]
    lspci_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="nvidia geforce rtx 5070",
            vram_total_mb=0,
            vram_free_mb=0,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: nvidia_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: lspci_result)
    gpus = list_gpus()
    assert len(gpus) == 1
    assert gpus[0].vram_total_mb == 8151


def test_list_gpus_dedup_strips_variant_descriptors(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    nvidia_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070 Laptop GPU",
            vram_total_mb=8151,
            vram_free_mb=7696,
            utilization_pct=12.0,
        )
    ]
    lspci_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070 Max-Q / Mobile",
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
            compute_capability=None,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: nvidia_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: lspci_result)
    gpus = list_gpus()
    assert len(gpus) == 1
    assert gpus[0].vram_total_mb == 8151
    assert gpus[0].utilization_pct == 12.0
    assert gpus[0].vram_free_mb == 7696


def test_list_gpus_falls_back_to_torch_when_all_empty(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: [])
    torch_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="torch-discovered",
            vram_total_mb=8000,
            vram_free_mb=4000,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_torch_fallback", lambda: torch_result)
    gpus = list_gpus()
    assert gpus == torch_result


def test_parse_rocm_smi_parses_all_three_subcommands(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocm-smi")  # type: ignore[attr-defined]

    def _fake_check_output(cmd: list[str], *_a: object, **_k: object) -> str:
        joined = " ".join(cmd)
        if "--showproductname" in joined:
            return (
                "GPU[0]\t\t: Card Series: \t\tAMD Radeon 890M Graphics\n"
                "GPU[0]\t\t: Card Model: \t\t0x150e\n"
            )
        if "--showmeminfo" in joined and "--csv" in joined:
            return (
                "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
                "card0,536870912,518139904\n"
            )
        if "--showmemuse" in joined:
            return "GPU[0]\t\t: GPU Memory Allocated (VRAM%): 96\n"
        return ""

    monkeypatch.setattr(gpu_mod.subprocess, "check_output", _fake_check_output)
    gpus = _parse_rocm_smi()
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.vendor == GpuVendor.AMD
    assert gpu.name == "AMD Radeon 890M Graphics"
    assert gpu.vram_total_mb == 512
    assert gpu.vram_free_mb == 18
    assert gpu.utilization_pct == 96.0


def test_parse_rocm_smi_parses_text_vram_fallback(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocm-smi")  # type: ignore[attr-defined]

    def _fake_check_output(cmd: list[str], *_a: object, **_k: object) -> str:
        joined = " ".join(cmd)
        if "--showproductname" in joined:
            return "GPU[0]\t\t: Card Series: \t\tAMD Radeon 890M Graphics\n"
        if "--showmeminfo" in joined and "--csv" in joined:
            return (
                "GPU[0]\t\t: VRAM Total Memory (B): 536870912\n"
                "GPU[0]\t\t: VRAM Total Used Memory (B): 518139904\n"
            )
        if "--showmemuse" in joined:
            return "GPU[0]\t\t: GPU Memory Allocated (VRAM%): 50\n"
        return ""

    monkeypatch.setattr(gpu_mod.subprocess, "check_output", _fake_check_output)
    gpus = _parse_rocm_smi()
    assert len(gpus) == 1
    assert gpus[0].vram_total_mb == 512
    assert gpus[0].vram_free_mb == 18
    assert gpus[0].utilization_pct == 50.0


def test_parse_rocm_smi_returns_empty_when_binary_missing(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: None)  # type: ignore[attr-defined]
    check_output = MagicMock(side_effect=AssertionError("rocm-smi invoked when missing"))
    monkeypatch.setattr(gpu_mod.subprocess, "check_output", check_output)
    assert _parse_rocm_smi() == []


def test_parse_rocm_smi_returns_empty_when_subprocess_fails(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocm-smi")  # type: ignore[attr-defined]

    def _raise(*_a: object, **_k: object) -> str:
        raise subprocess.CalledProcessError(1, "rocm-smi")

    monkeypatch.setattr(gpu_mod.subprocess, "check_output", _raise)
    assert _parse_rocm_smi() == []


def test_parse_rocm_smi_handles_name_only(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocm-smi")  # type: ignore[attr-defined]

    def _fake_check_output(cmd: list[str], *_a: object, **_k: object) -> str:
        if "--showproductname" in " ".join(cmd):
            return "GPU[0]\t\t: Card Series: \t\tAMD Radeon Pro W7900\n"
        raise subprocess.CalledProcessError(1, "rocm-smi")

    monkeypatch.setattr(gpu_mod.subprocess, "check_output", _fake_check_output)
    gpus = _parse_rocm_smi()
    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon Pro W7900"
    assert gpus[0].vram_total_mb == 0
    assert gpus[0].vram_free_mb == 0
    assert gpus[0].utilization_pct is None


def test_parse_rocminfo_extracts_gpu_names(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocminfo")  # type: ignore[attr-defined]
    sample = (
        "ROCk module is loaded\n"
        "\n"
        "***********************\n"
        "Agent 1\n"
        "***********************\n"
        "  Marketing Name:          AMD Ryzen AI 9 HX 370 w/ Radeon 890M\n"
        "  Vendor Name:             AMD\n"
        "\n"
        "***********************\n"
        "Agent 2\n"
        "***********************\n"
        "  Marketing Name:          AMD Radeon 890M Graphics\n"
        "  Vendor Name:             AMD\n"
        "\n"
        "***********************\n"
        "Agent 3\n"
        "***********************\n"
        "  Marketing Name:          AIE-ML\n"
        "  Vendor Name:             AMD\n"
    )
    monkeypatch.setattr(gpu_mod.subprocess, "check_output", lambda *_a, **_k: sample)
    gpus = _parse_rocminfo()
    # APU brand line and AIE-ML are filtered out; only "AMD Radeon 890M Graphics" remains.
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.AMD
    assert gpus[0].name == "AMD Radeon 890M Graphics"
    assert gpus[0].vram_total_mb == 0
    assert gpus[0].vram_free_mb == 0
    assert gpus[0].utilization_pct is None
    assert gpus[0].compute_capability is None


def test_parse_rocminfo_filters_apu_brand_and_npu(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocminfo")  # type: ignore[attr-defined]
    sample = (
        "  Marketing Name:          AMD Ryzen AI 9 HX 370 w/ Radeon 890M\n"
        "  Marketing Name:          AIE-ML\n"
        "  Marketing Name:          CPU\n"
    )
    monkeypatch.setattr(gpu_mod.subprocess, "check_output", lambda *_a, **_k: sample)
    assert _parse_rocminfo() == []


def test_parse_rocminfo_returns_empty_when_binary_missing(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: None)  # type: ignore[attr-defined]
    check_output = MagicMock(side_effect=AssertionError("rocminfo invoked when missing"))
    monkeypatch.setattr(gpu_mod.subprocess, "check_output", check_output)
    assert _parse_rocminfo() == []


def test_parse_rocminfo_returns_empty_when_subprocess_fails(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/rocminfo")  # type: ignore[attr-defined]

    def _raise(*_a: object, **_k: object) -> str:
        raise subprocess.TimeoutExpired("rocminfo", 5)

    monkeypatch.setattr(gpu_mod.subprocess, "check_output", _raise)
    assert _parse_rocminfo() == []


def test_list_gpus_merges_rocm_and_rocminfo(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    rocm_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="AMD Radeon 890M Graphics",
            vram_total_mb=512,
            vram_free_mb=18,
            utilization_pct=96.0,
        )
    ]
    rocminfo_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="AMD Radeon 890M Graphics",
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
            compute_capability=None,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: rocm_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: rocminfo_result)
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: [])
    gpus = list_gpus()
    assert len(gpus) == 1
    # rocm-smi (priority) wins because it has VRAM data.
    assert gpus[0].vram_total_mb == 512
    assert gpus[0].utilization_pct == 96.0


def test_list_gpus_dedup_rocm_and_rocminfo_same_name(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    nvidia_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA GeForce RTX 5070 Laptop GPU",
            vram_total_mb=8151,
            vram_free_mb=7680,
        )
    ]
    rocm_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="AMD Radeon 890M Graphics",
            vram_total_mb=512,
            vram_free_mb=18,
        )
    ]
    rocminfo_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="AMD Radeon 890M Graphics",
            vram_total_mb=0,
            vram_free_mb=0,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: nvidia_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: rocm_result)
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: rocminfo_result)
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: [])
    gpus = list_gpus()
    assert len(gpus) == 2
    # Stable source priority order; list_gpus() reassigns sequential indices.
    assert gpus[0].vendor == GpuVendor.NVIDIA
    assert gpus[0].index == 0
    assert gpus[1].vendor == GpuVendor.AMD
    assert gpus[1].index == 1
    # The AMD entry came from rocm-smi (has VRAM), not rocminfo.
    assert gpus[1].vram_total_mb == 512
    assert gpus[1].vram_free_mb == 18


def test_list_gpus_rocminfo_only_when_rocm_smi_missing(monkeypatch: object) -> None:
    import img2svg.gpu as gpu_mod

    rocminfo_result = [
        GPUInfo(
            index=0,
            vendor=GpuVendor.AMD,
            name="AMD Radeon 890M Graphics",
            vram_total_mb=0,
            vram_free_mb=0,
        )
    ]
    monkeypatch.setattr(gpu_mod, "_parse_nvidia_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocm_smi", lambda: [])
    monkeypatch.setattr(gpu_mod, "_parse_rocminfo", lambda: rocminfo_result)
    monkeypatch.setattr(gpu_mod, "_parse_lspci", lambda: [])
    gpus = list_gpus()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.AMD
    assert gpus[0].name == "AMD Radeon 890M Graphics"


def _install_fake_torch_cuda(
    monkeypatch: object,
    name: str,
    total_mem: int,
    major: int,
    minor: int,
    free_mem: int,
) -> None:
    """Stub out the torch.cuda.* API used by `_torch_fallback`.

    `_torch_fallback` calls `torch.cuda.is_available()`,
    `torch.cuda.device_count()`, `torch.cuda.get_device_properties(i)`,
    and `torch.cuda.mem_get_info(i)`. We install minimal fakes so the
    fallback enters the loop body and reports a single device.
    """
    import img2svg.gpu as gpu_mod

    props = MagicMock()
    props.name = name
    props.total_memory = total_mem
    props.major = major
    props.minor = minor

    monkeypatch.setattr(gpu_mod.torch.cuda, "is_available", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(gpu_mod.torch.cuda, "device_count", lambda: 1)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu_mod.torch.cuda,
        "get_device_properties",
        lambda _i: props,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu_mod.torch.cuda,
        "mem_get_info",
        lambda _i: (free_mem, total_mem),
    )


def test_torch_fallback_vendor_nvidia(monkeypatch: object) -> None:
    """torch.version.hip is None on a CUDA build → vendor must be NVIDIA."""
    import img2svg.gpu as gpu_mod

    _install_fake_torch_cuda(
        monkeypatch,
        name="NVIDIA GeForce RTX 5070",
        total_mem=8 * 1024 * 1024 * 1024,
        major=12,
        minor=0,
        free_mem=7 * 1024 * 1024 * 1024,
    )
    # CUDA build of PyTorch: torch.version.hip is None.
    monkeypatch.setattr(gpu_mod.torch.version, "hip", None)  # type: ignore[attr-defined]
    gpus = gpu_mod._torch_fallback()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.NVIDIA
    assert gpus[0].name == "NVIDIA GeForce RTX 5070"
    assert gpus[0].compute_capability == "12.0"


def test_torch_fallback_vendor_rocm(monkeypatch: object) -> None:
    """torch.version.hip is a non-empty string on a ROCm build → vendor must be AMD."""
    import img2svg.gpu as gpu_mod

    _install_fake_torch_cuda(
        monkeypatch,
        name="AMD Radeon RX 7900 XTX",
        total_mem=24 * 1024 * 1024 * 1024,
        major=11,
        minor=0,
        free_mem=20 * 1024 * 1024 * 1024,
    )
    # ROCm build of PyTorch: torch.version.hip is a version string.
    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu_mod.torch.version, "hip", "6.2.41134"
    )
    gpus = gpu_mod._torch_fallback()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.AMD
    assert gpus[0].name == "AMD Radeon RX 7900 XTX"
    assert gpus[0].compute_capability == "11.0"


def test_torch_fallback_vendor_nvidia_when_hip_empty_string(monkeypatch: object) -> None:
    """Empty-string torch.version.hip is treated as CUDA (not AMD).

    Defensive guard: some PyTorch build configurations may surface
    `torch.version.hip` as an empty string instead of None. The
    `getattr(..., None)` check uses truthiness, so an empty string
    must NOT be classified as AMD.
    """
    import img2svg.gpu as gpu_mod

    _install_fake_torch_cuda(
        monkeypatch,
        name="Test CUDA GPU",
        total_mem=4 * 1024 * 1024 * 1024,
        major=8,
        minor=6,
        free_mem=2 * 1024 * 1024 * 1024,
    )
    monkeypatch.setattr(gpu_mod.torch.version, "hip", "")  # type: ignore[attr-defined]
    gpus = gpu_mod._torch_fallback()
    assert len(gpus) == 1
    assert gpus[0].vendor == GpuVendor.NVIDIA


def test_torch_fallback_returns_empty_when_cuda_unavailable(monkeypatch: object) -> None:
    """torch.cuda.is_available() == False → _torch_fallback returns [] without crashing."""
    import img2svg.gpu as gpu_mod

    monkeypatch.setattr(  # type: ignore[attr-defined]
        gpu_mod.torch.cuda, "is_available", lambda: False
    )
    assert gpu_mod._torch_fallback() == []

