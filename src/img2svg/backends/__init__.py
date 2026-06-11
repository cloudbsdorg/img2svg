# img2svg - multi-vendor GPU/CPU backend abstraction.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-vendor compute backend abstraction.

This subpackage decouples the rest of the codebase from the specifics of
NVIDIA CUDA, AMD ROCm, Apple MPS, and CPU. Each concrete backend
(``CPUBackend``, ``CudaBackend``, ``MpsBackend``, ...) implements the
:class:`DeviceBackend` protocol declared in :mod:`img2svg.backends.protocol`.

The current public surface is intentionally minimal: a :class:`BackendType`
str-enum (selector key), the :class:`DeviceBackend` protocol, the always-
available :class:`CPUBackend` implementation, and its module-level
``CPU_BACKEND`` singleton. A registry/factory and the GPU-specific backends
are added in later tasks.
"""

from __future__ import annotations

from img2svg.backends.cpu import CPU_BACKEND, CPUBackend
from img2svg.backends.cuda import CUDA_BACKEND, CUDABackend
from img2svg.backends.mps import MPS_BACKEND, MPSBackend
from img2svg.backends.protocol import BackendType, DeviceBackend
from img2svg.backends.rocm import ROCM_BACKEND, ROCMBackend

__all__ = [
    "CPU_BACKEND",
    "CUDA_BACKEND",
    "MPS_BACKEND",
    "ROCM_BACKEND",
    "BackendType",
    "CPUBackend",
    "CUDABackend",
    "DeviceBackend",
    "MPSBackend",
    "ROCMBackend",
]
