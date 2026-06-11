# img2svg - AMD ROCm compute backend for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""AMD ROCm compute backend.

The :class:`ROCMBackend` is the AMD counterpart to :class:`CudaBackend`.
It is the second preferred backend in the auto-detect chain (after NVIDIA
CUDA). It shares the same :mod:`torch.cuda` API surface — AMD's ROCm
flavored PyTorch build intentionally re-exports the CUDA namespace as
the only stable vendor-neutral entry point.

The single critical difference from a plain CUDA build is
``torch.version.hip``: it is ``None`` for a CUDA build of PyTorch and a
non-empty string (e.g. ``"6.2.41134"``) for a ROCm build. ``is_available``
relies on this attribute to distinguish the two cases and refuses to claim
a CUDA-only PyTorch as a usable ROCm backend.

Lazy import strategy
--------------------

``torch`` is **not** imported at module load time. Every method that needs
it imports inside the function body and returns a safe default (``False``,
``0``) on :class:`ImportError`. This keeps the module importable in
environments where PyTorch is not installed (e.g. minimal CI images,
documentation builds).

Why ``[ROCm]`` prefix on device names
-------------------------------------

The underlying :func:`torch.cuda.get_device_name` call returns the same
AMD marketing string (e.g. ``"Radeon RX 7900 XT"``) that would appear for
an NVIDIA card. Prefixing with ``"[ROCm] "`` makes the output of
``img2svg list-gpus`` unambiguous when both vendors coexist in the same
host (rare on Linux, but possible in mixed CI fleets).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from img2svg.enums import GpuVendor

if TYPE_CHECKING:  # pragma: no cover - import-only for type hints
    from img2svg.backends.protocol import BackendType


# Bytes per mebibyte — kept as a module constant for symmetry with
# cpu.py and to make the memory math obvious.
_BYTES_PER_MB = 1024 * 1024


class ROCMBackend:
    """AMD ROCm compute backend implementation.

    Implements the :class:`DeviceBackend` protocol. Devices are
    zero-indexed in backend-local space (``ROCMBackend(0)`` is the first
    AMD GPU visible to the ROCm PyTorch runtime). For AMD APUs the
    runtime may report the integrated GPU as device ``0``; the
    recommendation logic in the registry decides whether the device has
    enough free memory to be useful.

    The class is duck-typed against the :class:`DeviceBackend` protocol
    — it does not subclass it explicitly. ``runtime_checkable`` on the
    protocol makes :func:`isinstance` checks work for tests and the
    registry.
    """

    def type(self) -> "BackendType":
        """Return the backend's :class:`BackendType` (always ``ROCM``)."""
        from img2svg.backends.protocol import BackendType

        return BackendType.ROCM

    def is_available(self) -> bool:
        """Return whether ROCm is usable on the current host.

        The two required conditions are:

        1. :func:`torch.cuda.is_available` returns ``True`` — the
           ROCm PyTorch runtime loaded the driver and sees at least
           one device.
        2. :attr:`torch.version.hip` is a non-empty string — the
           PyTorch build is the ROCm flavor, not the plain CUDA
           flavor. A plain CUDA build reports ``hip is None`` even
           on a host with a working NVIDIA driver.

        Returns ``False`` if :mod:`torch` is not importable.
        """
        try:
            import torch
        except ImportError:
            return False
        # The `and` short-circuits cleanly when `torch.cuda.is_available`
        # is False, so we never hit `torch.version.hip` on a system
        # where the CUDA driver failed to initialize.
        return bool(torch.cuda.is_available() and torch.version.hip)

    def device_count(self) -> int:
        """Return the number of AMD GPUs visible to the ROCm runtime.

        Returns ``0`` when :meth:`is_available` is ``False`` so callers
        can treat the return value as "how many usable devices do we
        have" without an extra check.
        """
        if not self.is_available():
            return 0
        try:
            import torch
        except ImportError:
            return 0
        return int(torch.cuda.device_count())

    def device_name(self, i: int) -> str:
        """Return a human-readable name for device ``i``.

        Prefixes the result from :func:`torch.cuda.get_device_name` with
        ``"[ROCm] "`` so multi-vendor listings (rare but possible in
        mixed CI fleets) stay unambiguous.
        """
        if not self.is_available():
            raise RuntimeError(
                "ROCMBackend.device_name called but ROCm is not available"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guard, not a path
            raise RuntimeError("torch is required to query device name") from exc
        name = str(torch.cuda.get_device_name(i))
        return f"[ROCm] {name}"

    def total_memory_mb(self, i: int) -> int:
        """Return total VRAM in MiB for device ``i``.

        Uses :func:`torch.cuda.get_device_properties` and divides the
        ``total_memory`` attribute (bytes) by 1 MiB. On older ROCm
        builds that lack the attribute, falls back to ``0`` rather than
        raising — the registry treats ``0`` as "unknown" and skips the
        device from the recommendation ranking.
        """
        if not self.is_available():
            raise RuntimeError(
                "ROCMBackend.total_memory_mb called but ROCm is not available"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guard, not a path
            raise RuntimeError(
                "torch is required to query device memory"
            ) from exc
        try:
            props = torch.cuda.get_device_properties(i)
        except Exception:
            return 0
        total_bytes = getattr(props, "total_memory", 0) or 0
        return int(total_bytes // _BYTES_PER_MB)

    def free_memory_mb(self, i: int) -> int:
        """Return free VRAM in MiB for device ``i``.

        Uses :func:`torch.cuda.mem_get_info` which returns a
        ``(free_bytes, total_bytes)`` tuple on CUDA 11.4+ and on recent
        ROCm builds. The call may raise on very old runtimes or when the
        device is in a faulted state; in that case we return ``0`` and
        let the registry fall back to :meth:`total_memory_mb` heuristics.
        """
        if not self.is_available():
            raise RuntimeError(
                "ROCMBackend.free_memory_mb called but ROCm is not available"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guard, not a path
            raise RuntimeError(
                "torch is required to query free device memory"
            ) from exc
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        except Exception:
            return 0
        return int(free_bytes // _BYTES_PER_MB)

    def vendor(self) -> GpuVendor:
        """Return the hardware vendor (always :attr:`GpuVendor.AMD`)."""
        return GpuVendor.AMD

    def to_ultralytics_string(self, i: int) -> str:
        """Return the ultralytics device string for device ``i``.

        Ultralytics treats ROCm devices as CUDA devices under the hood
        — the ROCm PyTorch build re-exports the CUDA API — so the
        canonical ultralytics string is ``"cuda:<i>"``. Using the
        ``"rocm:0"`` form would fail at ultralytics' device parsing
        layer.
        """
        if i < 0:
            raise IndexError(
                f"ROCMBackend device index must be non-negative; got {i}"
            )
        return f"cuda:{i}"

    def warmup(self) -> None:
        """No-op for ROCm.

        The ROCm runtime initializes lazily on first kernel launch; we
        deliberately do **not** trigger that here. The first real
        inference call (in the YOLO detection stage) will pay the
        context-creation cost once, and the user has already approved
        that work by running the conversion. Eager warmup would inflate
        ``list-gpus`` latency for users who only wanted to inspect
        available devices.
        """
        return None


# Module-level singleton. Other modules (registry, CLI, pipeline) import
# this constant rather than constructing a fresh ROCMBackend every time.
# Construction is cheap (no torch import happens until the first method
# call) but the singleton pattern matches the rest of the backend
# matrix for consistency.
ROCM_BACKEND = ROCMBackend()

__all__ = ["ROCMBackend", "ROCM_BACKEND"]
