# img2svg - compute device autodetect for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Compute device autodetect for img2svg.

Priority chain for `auto`:
  1. CUDA — `torch.cuda.is_available()` returns True for BOTH NVIDIA CUDA
     AND AMD ROCm (ROCm uses HIP which exposes the CUDA API).
  2. MPS  — Apple Silicon via `torch.backends.mps.is_available()`.
  3. CPU  — always available.

For explicit requests (`--device cpu`, `--device cuda`, `--device cuda:N`,
`--device mps`, `--device rocm`), we fail hard with `DeviceUnavailableError`
if the requested device is not available — no silent fallback.
"""

from __future__ import annotations

import torch

from img2svg.errors import DeviceUnavailableError


def is_available(device: str) -> bool:
    """Return True if `device` is currently usable, False otherwise."""
    device = device.strip().lower()
    if device == "cpu":
        return True
    if device in ("cuda", "rocm"):
        return bool(torch.cuda.is_available())
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
        except ValueError:
            return False
        return bool(torch.cuda.is_available()) and idx < torch.cuda.device_count()
    if device == "mps":
        try:
            return bool(torch.backends.mps.is_available())
        except AttributeError:
            return False
    return False


def list_available_devices() -> list[str]:
    """Return a list of device strings currently available on this host."""
    devs: list[str] = []
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            devs.append(f"cuda:{i}")
    try:
        if torch.backends.mps.is_available():
            devs.append("mps")
    except AttributeError:
        pass
    devs.append("cpu")
    return devs


def detect_device(requested: str = "auto") -> str:
    """Resolve the requested device to a concrete device string.

    `auto`: try CUDA (covers NVIDIA + AMD ROCm), then MPS, then CPU.
    Explicit requests fail hard with `DeviceUnavailableError`.
    """
    requested_clean = requested.strip().lower()
    if requested_clean in ("auto", ""):
        if torch.cuda.is_available():
            return "cuda:0"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"

    if is_available(requested_clean):
        return requested_clean

    available = list_available_devices()
    raise DeviceUnavailableError(requested_clean, available)
