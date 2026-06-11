# img2svg - GPU enumeration and recommendation for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""GPU enumeration and recommendation for img2svg."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess

import torch
from rich.console import Console
from rich.style import Style
from rich.table import Table

from img2svg.enums import DeviceStrategy, GpuVendor
from img2svg.models import GPUInfo

# PCI vendor IDs we recognize as discrete/display GPUs.
# 0x1002 = AMD/ATI, 0x10de = NVIDIA.
_LSPCI_AMD_VENDOR = "1002"
_LSPCI_NVIDIA_VENDOR = "10de"

# Match the device description after the vendor name on a single lspci line.
# Captures the text between the vendor name and the trailing [vendor:device] PCI ID.
_LSPCI_DEVICE_RE = re.compile(
    r":\s*"
    r"(?:Advanced Micro Devices, Inc\.\s*\[AMD/ATI\]|NVIDIA Corporation)"
    r"\s+(.+?)\s+\[(?:1002|10de):[0-9a-fA-F]+\]"
)

# Display-class PCI codes: 0300 = VGA, 0302 = 3D, 0380 = Display controller.
# Filters out sibling devices (audio [0403], USB, etc.) that share the vendor ID.
_LSPCI_DISPLAY_CLASS_RE = re.compile(r"\[\s*(?:0300|0302|0380)\s*\]")

# Inner bracket extraction: e.g. "Strix [Radeon 880M / 890M]" -> "Radeon 880M / 890M".
_LSPCI_INNER_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

# Strip common variant descriptors from a name for dedup comparison. lspci
# reports "GeForce RTX 5070 Max-Q / Mobile" while nvidia-smi reports
# "NVIDIA GeForce RTX 5070 Laptop GPU" — same physical GPU, different
# marketing descriptors. Normalization strips both forms so they match.
_DEDUP_NORMALIZE_RE = re.compile(
    r"\s+(laptop\s+gpu|desktop\s+gpu|max-?q(?:\s*/\s*mobile)?|mobile|desktop)\s*$",
    re.IGNORECASE,
)
_DEDUP_VENDOR_PREFIX_RE = re.compile(
    r"^(nvidia|amd|ati|advanced\s+micro\s+devices)\s+",
    re.IGNORECASE,
)


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


# Match lines like "GPU[0]		: Card Series: 		AMD Radeon 890M Graphics".
# Captures the device index from "GPU[N]" and the marketing name after "Card Series:".
_ROCM_PRODUCT_NAME_RE = re.compile(
    r"GPU\[(\d+)\][^:]*:\s*Card Series:\s*(.+?)\s*$"
)

# Match lines like "GPU[0]		: GPU Memory Allocated (VRAM%): 96".
_ROCM_MEMUSE_RE = re.compile(
    r"GPU\[(\d+)\][^:]*:\s*GPU Memory Allocated \(VRAM%\):\s*(\d+(?:\.\d+)?)"
)

# A GPU-class device name (positive list). Used to filter out APU / NPU / CPU
# entries reported by rocminfo. Matches "Radeon", "Instinct", "Navi", "Vega",
# and the "MI" / "Pro" / "RX " family prefixes that AMD/ROCm uses for
# discrete data-center and consumer GPUs.
_ROCMINFO_GPU_NAME_RE = re.compile(
    r"\b(radeon|instinct|navi|vega|pro|rx\s|mi\d{2,3})\b",
    re.IGNORECASE,
)

# Negative filter: marketing names that mention "Radeon" but aren't a
# discrete GPU (APU brand, NPU, CPU). Checked before the positive filter
# so "AMD Ryzen AI 9 HX 370 w/ Radeon 890M" is rejected.
_ROCMINFO_NON_GPU_RE = re.compile(
    r"\b(ryzen\s+ai|aie(?:-?ml)?|npu|cpu)\b",
    re.IGNORECASE,
)


def _parse_rocm_smi() -> list[GPUInfo]:
    """Query `rocm-smi` for AMD GPU info. Returns [] if missing.

    Uses three separate invocations to stay portable across rocm-smi versions:

    1. ``--showproductname`` (text) — parses ``Card Series: <name>`` lines.
    2. ``--showmeminfo vram --csv`` (CSV) — parses
       ``device,VRAM Total Memory (B),VRAM Total Used Memory (B)`` rows.
       Falls back to text output if the CSV header is missing.
    3. ``--showmemuse`` (text) — parses ``GPU Memory Allocated (VRAM%): <n>`` lines.

    Results are merged by device index into a single :class:`GPUInfo` per
    device. VRAM bytes are converted to MB (``bytes / (1024 * 1024)``).
    Returns ``[]`` if ``rocm-smi`` is missing or any invocation fails.
    """
    if shutil.which("rocm-smi") is None:
        return []
    names, vram, util = _rocm_smi_collect()
    if not names and not vram and not util:
        return []
    by_idx: dict[int, GPUInfo] = {}
    for idx, name in names.items():
        by_idx[idx] = GPUInfo(
            index=idx,
            vendor=GpuVendor.AMD,
            name=name,
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
        )
    for idx, (total_b, used_b) in vram.items():
        gpu = by_idx.get(idx) or GPUInfo(
            index=idx,
            vendor=GpuVendor.AMD,
            name=f"AMD GPU {idx}",
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=None,
        )
        total_mb = max(0, total_b // (1024 * 1024))
        used_mb = max(0, used_b // (1024 * 1024))
        free_mb = max(0, total_mb - used_mb)
        by_idx[idx] = gpu.model_copy(
            update={"vram_total_mb": total_mb, "vram_free_mb": free_mb}
        )
    for idx, pct in util.items():
        if 0.0 <= pct <= 100.0:
            gpu = by_idx.get(idx) or GPUInfo(
                index=idx,
                vendor=GpuVendor.AMD,
                name=f"AMD GPU {idx}",
                vram_total_mb=0,
                vram_free_mb=0,
                utilization_pct=None,
            )
            by_idx[idx] = gpu.model_copy(update={"utilization_pct": pct})
    return sorted(by_idx.values(), key=lambda g: g.index)


def _rocm_smi_collect() -> tuple[
    dict[int, str], dict[int, tuple[int, int]], dict[int, float]
]:
    """Run the three rocm-smi subcommands and parse their output.

    Returns ``(names, vram, util)`` where:

    - ``names`` maps device index to marketing name.
    - ``vram`` maps device index to ``(total_bytes, used_bytes)``.
    - ``util`` maps device index to utilization percentage.

    Missing binaries, non-zero exit codes, and timeouts are swallowed
    (return empty dicts) so a partial install of rocm-smi still yields
    whatever data is available.
    """
    names: dict[int, str] = {}
    vram: dict[int, tuple[int, int]] = {}
    util: dict[int, float] = {}

    # 1. Product name.
    name_out = _rocm_smi_run(["rocm-smi", "--showproductname"])
    if name_out is not None:
        for line in name_out.splitlines():
            m = _ROCM_PRODUCT_NAME_RE.search(line)
            if m:
                idx = int(m.group(1))
                names[idx] = m.group(2).strip()

    # 2. VRAM info. Try CSV first; if the first non-empty line is not a
    # CSV header, fall back to text parsing of the same fields.
    vram_out = _rocm_smi_run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if vram_out is not None:
        vram = _parse_rocm_smi_vram(vram_out)
        if not vram:
            vram = _parse_rocm_smi_vram_text(vram_out)

    # 3. Memory utilization.
    util_out = _rocm_smi_run(["rocm-smi", "--showmemuse"])
    if util_out is not None:
        for line in util_out.splitlines():
            m = _ROCM_MEMUSE_RE.search(line)
            if m:
                idx = int(m.group(1))
                try:
                    util[idx] = float(m.group(2))
                except ValueError:
                    continue

    return names, vram, util


def _rocm_smi_run(cmd: list[str]) -> str | None:
    """Run a rocm-smi subcommand and return stdout, or None on failure."""
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def _parse_rocm_smi_vram(out: str) -> dict[int, tuple[int, int]]:
    """Parse CSV output of ``rocm-smi --showmeminfo vram --csv``.

    Expected header:
    ``device,VRAM Total Memory (B),VRAM Total Used Memory (B)``

    Some rocm-smi versions emit slightly different column orderings or
    include extra columns; we only require the first three.
    Returns ``{device_index: (total_bytes, used_bytes)}``.
    """
    result: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("device"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total = int(parts[1])
            used = int(parts[2])
        except ValueError:
            continue
        # Device identifier may be "card0", "0", "GPU-0", etc. Extract trailing
        # digits for the device index; default to 0 if no digits found.
        digits = re.findall(r"\d+", parts[0])
        idx = int(digits[-1]) if digits else 0
        result[idx] = (total, used)
    return result


def _parse_rocm_smi_vram_text(out: str) -> dict[int, tuple[int, int]]:
    """Parse text output of ``rocm-smi --showmeminfo vram`` (non-CSV fallback).

    Lines look like::
        GPU[0]		: VRAM Total Memory (B): 536870912
        GPU[0]		: VRAM Total Used Memory (B): 518139904
    """
    totals: dict[int, int] = {}
    used_values: dict[int, int] = {}
    line_re = re.compile(
        r"GPU\[(\d+)\][^:]*:\s*"
        r"(VRAM (?:Total(?: Used)? Memory|Used Memory) \(B\))"
        r":\s*(\d+)"
    )
    for line in out.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        idx = int(m.group(1))
        field = m.group(2)
        try:
            value = int(m.group(3))
        except ValueError:
            continue
        if "Used" in field:
            used_values[idx] = value
        else:
            totals[idx] = value
    return {
        idx: (totals[idx], used_values[idx])
        for idx in totals
        if idx in used_values
    }


def _parse_rocminfo() -> list[GPUInfo]:
    """Query ``rocminfo`` for AMD device marketing names.

    rocminfo (a separate ROCm utility) reports a cleaner marketing name
    than lspci (e.g. ``"AMD Radeon 890M Graphics"`` instead of
    ``"AMD Radeon 880M / 890M"``). It does NOT report runtime metrics
    (VRAM, utilization) — entries have ``vram_total_mb=0``,
    ``vram_free_mb=0``, ``utilization_pct=None``,
    ``compute_capability=None``.

    Non-GPU entries (APU brand lines like ``"AMD Ryzen AI 9 HX 370
    w/ Radeon 890M"``, NPUs like ``"AIE-ML"``, CPUs) are filtered out by
    requiring the name to contain a GPU-class indicator. Returns ``[]``
    if ``rocminfo`` is missing or fails.
    """
    if shutil.which("rocminfo") is None:
        return []
    try:
        out = subprocess.check_output(
            ["rocminfo"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    return _parse_rocminfo_text(out)


def _parse_rocminfo_text(out: str) -> list[GPUInfo]:
    """Extract GPU-class marketing names from ``rocminfo`` text output.

    Lines look like::
        Marketing Name:          AMD Radeon 890M Graphics
    """
    gpus: list[GPUInfo] = []
    name_re = re.compile(r"Marketing Name:\s+(.+?)\s*$")
    seen: set[tuple[str, str]] = set()
    for line in out.splitlines():
        m = name_re.search(line)
        if not m:
            continue
        name = m.group(1).strip()
        if not name:
            continue
        if _ROCMINFO_NON_GPU_RE.search(name):
            continue
        if not _ROCMINFO_GPU_NAME_RE.search(name):
            continue
        key = (GpuVendor.AMD.value, name.lower())
        if key in seen:
            continue
        seen.add(key)
        gpus.append(
            GPUInfo(
                index=len(gpus),
                vendor=GpuVendor.AMD,
                name=name,
                vram_total_mb=0,
                vram_free_mb=0,
                utilization_pct=None,
                compute_capability=None,
            )
        )
    return gpus


def _parse_lspci() -> list[GPUInfo]:
    """Detect display GPUs via `lspci -nn`, filtered to AMD/NVIDIA vendor IDs.

    Returns a list of GPUInfo with VRAM and utilization fields zero/unset
    (lspci does not report runtime metrics, and AMD APUs share system RAM).
    Returns [] if lspci is missing or fails.
    """
    if shutil.which("lspci") is None:
        return []
    try:
        out = subprocess.check_output(
            ["lspci", "-nn"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    gpus: list[GPUInfo] = []
    idx = 0
    for line in out.splitlines():
        if _LSPCI_NVIDIA_VENDOR not in line and _LSPCI_AMD_VENDOR not in line:
            continue
        if not _LSPCI_DISPLAY_CLASS_RE.search(line):
            continue
        m = _LSPCI_DEVICE_RE.search(line)
        if not m:
            continue
        desc = m.group(1).strip()
        # Prefer the inner bracket (marketing name) over the silicon code.
        inner = _LSPCI_INNER_BRACKET_RE.search(desc)
        device_name = inner.group(1) if inner else desc
        if _LSPCI_AMD_VENDOR in line:
            vendor = GpuVendor.AMD
            prefix = "AMD "
        else:
            vendor = GpuVendor.NVIDIA
            prefix = "NVIDIA "
        gpus.append(
            GPUInfo(
                index=idx,
                vendor=vendor,
                name=prefix + device_name,
                vram_total_mb=0,
                vram_free_mb=0,
                utilization_pct=None,
                compute_capability=None,
            )
        )
        idx += 1
    return gpus


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
        # Use torch.version.hip to distinguish a ROCm build of PyTorch from
        # a CUDA build. CUDA builds have torch.version.hip == None; ROCm
        # builds have a version string (e.g. "6.2.41134").
        if getattr(torch, "version", None) is not None and getattr(
            torch.version, "hip", None
        ):
            vendor = GpuVendor.AMD
        else:
            vendor = GpuVendor.NVIDIA
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


def _dedup_key(gpu: GPUInfo) -> tuple[str, str]:
    """Return a dedup key for a GPUInfo: (vendor, normalized name)."""
    name = _DEDUP_VENDOR_PREFIX_RE.sub("", gpu.name)
    name = _DEDUP_NORMALIZE_RE.sub("", name)
    return (gpu.vendor, name.lower().strip())


def list_gpus() -> list[GPUInfo]:
    """Enumerate all GPUs on the system.

    Merges results from `nvidia-smi`, `rocm-smi`, `rocminfo`, and `lspci`.
    When the same device is reported by more than one source (e.g. an AMD
    GPU seen by both `rocm-smi` and `rocminfo`), the entry with the most
    detail wins — `nvidia-smi` first, then `rocm-smi` (has VRAM and
    utilization), then `rocminfo` (cleaner marketing name but no runtime
    metrics), then `lspci` (static device tree, no metrics). Earlier
    entries win on dedup conflict. lspci is used as a last-resort static
    fallback for devices that none of the vendor tools report (e.g. headless
    servers with PCI topology but no drivers loaded).
    Falls back to torch CUDA enumeration only when ALL of the above are
    empty (e.g. headless systems with no lspci binary).
    """
    merged: dict[tuple[str, str], GPUInfo] = {}
    lspci_entries = _parse_lspci()
    # lspci names are imprecise (e.g. "Radeon 880M / 890M" for a GPU that
    # rocm-smi/rocminfo report as "Radeon 890M Graphics"). If any higher-
    # priority source already reports a GPU of the same vendor, skip lspci's
    # entries for that vendor so the user sees the GPU exactly once with
    # the highest-quality name and runtime metrics.
    seen_vendors: set[str] = set()
    for source in (_parse_nvidia_smi(), _parse_rocm_smi(), _parse_rocminfo()):
        for gpu in source:
            merged.setdefault(_dedup_key(gpu), gpu)
            seen_vendors.add(gpu.vendor)
    for gpu in lspci_entries:
        if gpu.vendor in seen_vendors:
            continue
        merged.setdefault(_dedup_key(gpu), gpu)
    if merged:
        sorted_gpus = sorted(merged.values(), key=lambda g: g.index)
        # Reassign sequential indices so the table shows unique values per row.
        # (Each source assigns its own device index starting at 0; the merge
        # preserves those, causing duplicate "Index 0" in the table.)
        return [
            gpu.model_copy(update={"index": i})
            for i, gpu in enumerate(sorted_gpus)
        ]
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
        is_rec = recommended is gpu
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
