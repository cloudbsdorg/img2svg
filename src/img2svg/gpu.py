# img2svg - GPU enumeration and recommendation for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""GPU enumeration and recommendation for img2svg."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any

import torch
from rich.console import Console
from rich.style import Style
from rich.table import Table

from img2svg.enums import DeviceStrategy, GpuVendor
from img2svg.models import GPUInfo


def _parse_nvidia_smi() -> list[GPUInfo]:
    """Query `nvidia-smi` for GPU info. Returns [] if nvidia-smi is missing."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            total = int(parts[2])
            free = int(parts[3])
            util = float(parts[4]) if parts[4] else None
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                index=idx,
                vendor=GpuVendor.NVIDIA,
                name=name,
                vram_total_mb=total,
                vram_free_mb=free,
                utilization_pct=util,
            )
        )
    return gpus


def _parse_rocm_smi() -> list[GPUInfo]:
    """Query `rocm-smi` for AMD GPU info. Returns [] if missing."""
    if shutil.which("rocm-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showidname", "--showmeminfo", "vram", "--csv"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    gpus: list[GPUInfo] = []
    current: dict[str, Any] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            if current:
                gpus.append(_rocm_dict_to_info(current))
                current = {}
            continue
        if "," in line:
            key, _, val = line.partition(",")
            current[key.strip().lower()] = val.strip()
    if current:
        gpus.append(_rocm_dict_to_info(current))
    return gpus


def _rocm_dict_to_info(d: dict[str, str]) -> GPUInfo:
    """Convert a rocm-smi --csv row dict to GPUInfo."""
    try:
        idx = int(d.get("device", "0"))
    except ValueError:
        idx = 0
    return GPUInfo(
        index=idx,
        vendor=GpuVendor.AMD,
        name=d.get("name", "AMD GPU"),
        vram_total_mb=0,
        vram_free_mb=0,
        utilization_pct=None,
    )


def _torch_fallback() -> list[GPUInfo]:
    """Fallback: enumerate via torch (works for both NVIDIA CUDA and AMD ROCm)."""
    if not torch.cuda.is_available():
        return []
    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
        except (RuntimeError, AttributeError):
            continue
        name = getattr(props, "name", f"GPU {i}")
        total = int(getattr(props, "total_memory", 0)) // (1024 * 1024)
        try:
            free_bytes, _total = torch.cuda.mem_get_info(i)
            free = int(free_bytes) // (1024 * 1024)
        except (RuntimeError, AttributeError):
            free = 0
        try:
            cc = f"{props.major}.{props.minor}"
        except AttributeError:
            cc = None
        vendor = GpuVendor.NVIDIA  # Could be AMD if PyTorch is ROCm build
        gpus.append(
            GPUInfo(
                index=i,
                vendor=vendor,
                name=name,
                vram_total_mb=total,
                vram_free_mb=free,
                compute_capability=cc,
            )
        )
    return gpus


def list_gpus() -> list[GPUInfo]:
    """Enumerate all GPUs on the system.

    Tries (in order): nvidia-smi, rocm-smi, torch CUDA. First non-empty wins.
    """
    nvidia = _parse_nvidia_smi()
    if nvidia:
        return nvidia
    rocm = _parse_rocm_smi()
    if rocm:
        return rocm
    return _torch_fallback()


def recommend_gpu(gpus: list[GPUInfo], strategy: DeviceStrategy) -> GPUInfo | None:
    """Pick the best GPU for the given strategy, or None if no GPUs."""
    if not gpus:
        return None
    if strategy == DeviceStrategy.POWER:
        return max(gpus, key=lambda g: (g.vram_total_mb, -g.index))
    if strategy == DeviceStrategy.AVAILABILITY:
        return max(gpus, key=lambda g: (g.vram_free_mb, -g.index))
    return gpus[0]


def _detect_os() -> str:
    """Return a human-readable OS name. Cross-platform."""
    try:
        return platform.uname().system
    except AttributeError:
        # Windows has no os.uname
        return os.uname().sysname  # type: ignore[attr-defined]


def _torch_version() -> str:
    """Return the installed PyTorch version, or 'N/A' if not importable."""
    try:
        return str(torch.__version__)
    except (ImportError, AttributeError):
        return "N/A"


def print_gpu_recommendation(strategy: str = "power") -> None:
    """Print a Rich table of detected GPUs and the recommended one.

    Args:
        strategy: One of ``"auto"``, ``"power"``, or ``"availability"``.
            Defaults to ``"power"`` (largest total VRAM).

    Behavior:

    - Calls :func:`list_gpus` to enumerate devices.
    - Calls :func:`recommend_gpu` to pick the best device for ``strategy``.
    - Prints OS info and the PyTorch version above the table.
    - Prints a Rich table with columns: Index, Vendor, Name, VRAM Total (MB),
      VRAM Free (MB), Util (%), Recommended.
    - The recommended row is highlighted in bold green; the Recommended
      column shows ``"Y"`` for the recommended GPU and ``" "`` for others.
    - If the GPU list is empty, prints ``"No GPU detected on this system."``
      in yellow instead of an empty table.
    - If ``strategy`` is not a valid :class:`DeviceStrategy`, prints an error
      message in red and returns without raising.

    The table is written to the default Rich :class:`Console` (stdout).
    """
    console = Console()
    console.print(f"[bold]OS:[/bold] {_detect_os()}")
    console.print(f"[bold]PyTorch:[/bold] {_torch_version()}")

    gpus = list_gpus()
    if not gpus:
        console.print("[yellow]No GPU detected on this system.[/yellow]")
        return

    try:
        strategy_enum = DeviceStrategy(strategy)
    except ValueError:
        valid = ", ".join(s.value for s in DeviceStrategy)
        console.print(f"[red]invalid strategy {strategy!r}. Valid strategies: {valid}[/red]")
        return

    recommended = recommend_gpu(gpus, strategy_enum)
    rec_index = recommended.index if recommended is not None else None

    table = Table(title="Available GPUs", title_style="bold", show_lines=False)
    table.add_column("Index", justify="right")
    table.add_column("Vendor")
    table.add_column("Name")
    table.add_column("VRAM Total (MB)", justify="right")
    table.add_column("VRAM Free (MB)", justify="right")
    table.add_column("Util (%)", justify="right")
    table.add_column("Recommended", justify="center")

    rec_style = Style(bold=True, color="green")
    for gpu in sorted(gpus, key=lambda g: g.index):
        is_rec = rec_index == gpu.index
        util_str = f"{gpu.utilization_pct:.0f}" if gpu.utilization_pct is not None else "-"
        table.add_row(
            str(gpu.index),
            gpu.vendor.value,
            gpu.name,
            str(gpu.vram_total_mb),
            str(gpu.vram_free_mb),
            util_str,
            "Y" if is_rec else " ",
            style=rec_style if is_rec else None,
        )

    console.print(table)
