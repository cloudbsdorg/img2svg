# img2svg - NVIDIA CUDA compute backend for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""NVIDIA CUDA compute backend.

The :class:`CUDABackend` implements the :class:`DeviceBackend` protocol for
hosts with an NVIDIA GPU and a CUDA-enabled PyTorch build. It is the first
choice in the auto-detect chain (CUDA > ROCM > MPS > CPU): the project's
research shows that an NVIDIA discrete GPU is by far the fastest way to run
YOLO segmentation at the resolutions vtracer cares about.

PyTorch is imported lazily inside each method (never at module level) so this
module is importable on CPU-only hosts, ROCm-only hosts, and macOS — the
import does not require a CUDA toolchain. A missing :mod:`torch` module or a
``torch`` build without CUDA is reported as :meth:`is_available` returning
``False``, never as an import-time error.

Detection model:

* :meth:`is_available` calls :func:`torch.cuda.is_available`. This returns
  ``True`` only when PyTorch was built with CUDA support **and** a driver
  is loaded.
* :meth:`device_count`, :meth:`device_name`, :meth:`total_memory_mb`, and
  :meth:`free_memory_mb` defer to the corresponding ``torch.cuda`` APIs. If
  any call raises (driver disappeared between calls, device was hot-unplugged,
  runtime error inside the CUDA context) the method returns a documented
  fallback (``0`` or a synthetic name) so the auto-detect chain does not
  crash on transient hardware faults.
* :meth:`warmup` calls :func:`torch.cuda.init` to trigger lazy CUDA-context
  creation up front. The first real YOLO forward pass is then freed of
  context-creation latency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from img2svg.enums import GpuVendor

if TYPE_CHECKING:  # pragma: no cover - import-only for type hints
    from img2svg.backends.protocol import BackendType

# Bytes per mebibyte — kept explicit to make the math obvious.
_BYTES_PER_MB = 1024 * 1024


class CUDABackend:
    """NVIDIA CUDA compute backend implementation.

    Implements the :class:`DeviceBackend` protocol for NVIDIA hardware. The
    number of devices is whatever the CUDA runtime reports (typically
    ``0`` on a CPU-only host and ``1+`` on a workstation with one or more
    GPUs). Device indices are backend-local and zero-based, matching the
    convention used by :class:`DeviceBackend`.
    """

    def type(self) -> BackendType:
        """Return the backend's :class:`BackendType` (always ``CUDA``)."""
        # Local import keeps the protocol/types module decoupled from
        # this vendor-specific glue and mirrors the pattern in
        # :class:`CPUBackend`.
        from img2svg.backends.protocol import BackendType

        return BackendType.CUDA

    def is_available(self) -> bool:
        """Return whether an NVIDIA CUDA backend is usable on this host.

        Returns ``True`` only when :mod:`torch` is importable **and** the
        installed build includes CUDA support **and** a driver is loaded.
        Any of these failing returns ``False``; the caller should fall back
        to the next backend in the chain (typically ROCm, then MPS, then
        CPU) rather than treating ``False`` as an error.
        """
        try:
            import torch
        except ImportError:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - defensive
            # Some unusual environments raise on the first probe (driver
            # mismatch, partially-installed CUDA). Treat as unavailable
            # rather than letting the exception bubble.
            return False

    def device_count(self) -> int:
        """Return the number of CUDA devices visible to PyTorch.

        Returns ``0`` when the backend is unavailable (no :mod:`torch`,
        no CUDA build, or no driver) or when the host has no NVIDIA
        devices. Callers MUST treat ``0`` as "no GPU" and fall back to
        the next backend in the chain.
        """
        try:
            import torch
        except ImportError:
            return 0
        try:
            return int(torch.cuda.device_count())
        except Exception:  # pragma: no cover - defensive
            return 0

    def device_name(self, i: int) -> str:
        """Return a human-readable marketing name for CUDA device ``i``.

        Forwards to :func:`torch.cuda.get_device_name`. If the call raises
        (invalid index, runtime error, etc.) returns the synthetic string
        ``f"GPU {i}"`` so callers always get a non-empty label.
        """
        try:
            import torch
        except ImportError:
            return f"GPU {i}"
        try:
            name = torch.cuda.get_device_name(i)
        except (RuntimeError, AttributeError):
            return f"GPU {i}"
        # Some drivers return ``""`` or whitespace for misconfigured GPUs.
        if not isinstance(name, str) or not name.strip():
            return f"GPU {i}"
        return name

    def total_memory_mb(self, i: int) -> int:
        """Return total VRAM in MiB for CUDA device ``i``.

        Forwards to :attr:`torch.cuda.get_device_properties(i).total_memory`
        and converts bytes to mebibytes. Returns ``0`` if the property
        cannot be read (invalid index, runtime error).
        """
        try:
            import torch
        except ImportError:
            return 0
        try:
            props = torch.cuda.get_device_properties(i)
        except (RuntimeError, AttributeError):
            return 0
        total_bytes = int(props.total_memory)
        return total_bytes // _BYTES_PER_MB

    def free_memory_mb(self, i: int) -> int:
        """Return free VRAM in MiB for CUDA device ``i``.

        Uses :func:`torch.cuda.mem_get_info` when available — that is the
        only public PyTorch API that exposes the *currently free* amount
        rather than the cached-reserved amount. Returns ``0`` on any
        failure (older CUDA contexts that do not support ``mem_get_info``,
        runtime errors, invalid index).
        """
        try:
            import torch
        except ImportError:
            return 0
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        except (RuntimeError, AttributeError):
            return 0
        return int(free_bytes) // _BYTES_PER_MB

    def vendor(self) -> GpuVendor:
        """Return the hardware vendor (always :attr:`GpuVendor.NVIDIA`)."""
        return GpuVendor.NVIDIA

    def to_ultralytics_string(self, i: int) -> str:
        """Return the ultralytics device string for CUDA device ``i``.

        Ultralytics expects the form ``"cuda:N"`` (lowercase) where ``N``
        is the zero-indexed CUDA device ordinal. No index-range checking
        is performed here; the registry is expected to call this only
        with valid indices.
        """
        return f"cuda:{i}"

    def warmup(self) -> None:
        """Trigger lazy CUDA-context initialization.

        Calls :func:`torch.cuda.init` if a CUDA build is present. This
        materializes the CUDA context eagerly so the first real YOLO
        forward pass is not penalized by context-creation latency. The
        method is a safe no-op on hosts without CUDA: it never raises
        and never imports :mod:`torch` at module level.
        """
        try:
            import torch
        except ImportError:
            return
        try:
            torch.cuda.init()  # type: ignore[no-untyped-call]
        except Exception:  # pragma: no cover - defensive
            # Eager init is best-effort. If the driver refuses, the
            # subsequent real call will surface the real error.
            return


# Module-level singleton. Other modules (registry, CLI, pipeline) import
# this constant rather than constructing a fresh CUDABackend every time.
# The instance is cheap: methods lazy-import torch on first call.
CUDA_BACKEND = CUDABackend()

__all__ = ["CUDA_BACKEND", "CUDABackend"]
