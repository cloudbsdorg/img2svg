# img2svg - Apple Metal Performance Shaders compute backend.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Apple Metal Performance Shaders (MPS) compute backend.

The :class:`MPSBackend` is the Apple Silicon accelerator implementation
of the :class:`DeviceBackend` protocol. It targets macOS hosts running
PyTorch 2.0 or newer with the Metal backend enabled (the default for
``arm64`` macOS wheels).

Design notes:

* **Lazy torch import.** ``torch`` is imported inside each method, not
  at module load. This keeps ``img2svg.backends.mps`` importable on
  non-macOS systems (Linux CI agents, Windows dev boxes) and on macOS
  PyTorch builds compiled without MPS support, where
  ``torch.backends.mps`` may not exist as an attribute.
* **System RAM is shared memory.** Apple Silicon GPUs are integrated
  on-package with the CPU and draw from the same unified memory pool.
  We therefore reuse the same stdlib :mod:`os` ``sysconf`` query that
  :class:`img2svg.backends.cpu.CPUBackend` uses, rather than asking
  PyTorch for VRAM (which is meaningless on MPS — there is none).
* **No warmup work.** The Metal context is created lazily by the first
  real tensor operation, and there is no benefit to triggering it
  eagerly. :meth:`warmup` is a documented no-op kept for symmetry with
  the GPU backends.
* **Single device.** Every Apple Silicon host exposes exactly one MPS
  device (index ``0``) when MPS is available — there are no multi-GPU
  Mac Pros in the wild as of 2026.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING

from img2svg.enums import GpuVendor

if TYPE_CHECKING:  # pragma: no cover - import-only for type hints
    from img2svg.backends.protocol import BackendType

# ``sysconf`` names we need. Defined as module constants so tests can patch
# them and so the same names appear consistently in type stubs if we add
# them later. ``SC_PAGE_SIZE`` and ``SC_PHYS_PAGES`` are POSIX and work on
# macOS the same way they do on Linux.
_SC_PAGE_SIZE = "SC_PAGE_SIZE"
_SC_PHYS_PAGES = "SC_PHYS_PAGES"
_SC_AVPHYS_PAGES = "SC_AVPHYS_PAGES"

# Bytes per mebibyte — kept explicit to make the math obvious.
_BYTES_PER_MB = 1024 * 1024

# Used as a last-resort free-memory estimate when ``os.sysconf`` does not
# know ``SC_AVPHYS_PAGES`` (e.g. some BSDs). Half of total RAM is a
# common conservative default.
_FALLBACK_FREE_RATIO = 2

# Marketing name returned by ``device_name(0)`` when MPS is usable.
# We prefer a stable label over ``platform.processor()`` (which on
# Apple Silicon returns ``"arm"``) so the CLI list is unambiguous.
_APPLE_SILICON_NAME = "Apple Silicon"


def _sysconf_bytes(name: str) -> int | None:
    """Return ``os.sysconf(name) * os.sysconf('SC_PAGE_SIZE')`` or ``None``.

    Mirrors the helper in :mod:`img2svg.backends.cpu` so MPS and CPU
    report the same number on the same host. Returns ``None`` if either
    sysconf call is unsupported (``os.sysconf`` raises
    :class:`ValueError`/returns ``-1`` for unknown names on POSIX, and
    :class:`AttributeError` on non-POSIX platforms).
    """
    try:
        page_size = int(os.sysconf(_SC_PAGE_SIZE))
        count = int(os.sysconf(name))
    except (ValueError, OSError, AttributeError):
        return None
    if page_size <= 0 or count <= 0:
        return None
    return page_size * count


class MPSBackend:
    """Apple Metal Performance Shaders compute backend.

    Implements the :class:`DeviceBackend` protocol for Apple Silicon
    (M-series) Macs running PyTorch with the Metal backend. There is
    exactly one virtual device (index ``0``) when MPS is available;
    when it is not, :meth:`device_count` returns ``0`` and every
    index-taking method raises :class:`IndexError`.

    The backend is safe to instantiate on non-macOS systems and on
    PyTorch builds compiled without MPS support — :meth:`is_available`
    simply returns ``False`` in those cases, and every other method
    short-circuits accordingly.
    """

    #: The single virtual device index exposed by this backend.
    _DEVICE_INDEX = 0

    def type(self) -> BackendType:
        """Return the backend's :class:`BackendType` (always ``MPS``)."""
        # Local import to keep this module importable even if the
        # protocol/types module is still being authored in parallel.
        from img2svg.backends.protocol import BackendType

        return BackendType.MPS

    def is_available(self) -> bool:
        """Return whether the MPS backend is usable on the current host.

        Probes ``torch.backends.mps`` defensively: PyTorch builds for
        non-macOS targets (and some older macOS builds) do not define
        the ``mps`` attribute, and a naive ``torch.backends.mps.is_available()``
        call would raise :class:`AttributeError` on those systems. We
        therefore guard the lookup with ``getattr`` and a broad
        ``except`` so the backend is always importable, always safe
        to instantiate, and never raises from this method.

        The torch import itself is also lazy so importing this module
        does not require a working PyTorch install.
        """
        try:
            import torch
        except ImportError:
            return False
        mps_module = getattr(torch.backends, "mps", None)
        if mps_module is None:
            return False
        try:
            return bool(mps_module.is_available())
        except (AttributeError, RuntimeError):
            # Some PyTorch builds expose ``mps`` but its ``is_available``
            # probe raises (e.g. when the Metal driver is missing on a
            # stripped-down macOS image). Treat as unavailable rather
            # than crashing the auto-detect chain.
            return False

    def device_count(self) -> int:
        """Return the number of MPS devices (0 or 1).

        Apple Silicon has a single GPU per host, so the answer is
        always ``1`` when MPS is available and ``0`` otherwise.
        """
        return 1 if self.is_available() else 0

    def device_name(self, i: int) -> str:
        """Return a human-readable name for device ``i``.

        Returns the stable label :data:`_APPLE_SILICON_NAME` when MPS
        is usable (the exact silicon model — M1, M2 Pro, M3 Max — is
        not exposed through the PyTorch MPS API in a portable way, and
        a stable label is more useful for the CLI's ``list-gpus`` view
        than a model number nobody can sanity-check). Falls back to
        :func:`platform.processor` if the index is invalid; the
        intent is "always return *some* non-empty string" so the CLI
        never has to special-case missing values.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"MPSBackend has 1 device (index {self._DEVICE_INDEX}); requested index {i}"
            )
        if not self.is_available():
            # Mirror the CPU backend's behaviour: return a label even
            # when the backend is not active, so the CLI can render
            # "Apple Silicon (unavailable)" if it wants to.
            name = platform.processor() or ""
            return name.strip() or _APPLE_SILICON_NAME
        return _APPLE_SILICON_NAME

    def total_memory_mb(self, i: int) -> int:
        """Return total unified memory in MiB for device ``i``.

        Apple Silicon uses unified memory shared between CPU and GPU;
        there is no separate VRAM pool to query. We therefore report
        the host's total physical RAM, using the same
        ``os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')``
        formula that :class:`CPUBackend` uses (POSIX, works on macOS).
        On platforms where those sysconf names are not available,
        returns ``0`` so the registry can show "memory: unknown"
        honestly.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"MPSBackend has 1 device (index {self._DEVICE_INDEX}); requested index {i}"
            )
        if sys.platform == "win32":
            # os.sysconf is not present on native Windows. Returning 0
            # is honest: this backend cannot answer the question on
            # this OS, and the platform check also keeps us out of
            # a weird "MPS on Windows" footgun.
            return 0
        total = _sysconf_bytes(_SC_PHYS_PAGES)
        if total is None:
            return 0
        return total // _BYTES_PER_MB

    def free_memory_mb(self, i: int) -> int:
        """Return free unified memory in MiB for device ``i``.

        Uses ``os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')``
        on POSIX (macOS qualifies). On platforms where the kernel does
        not expose ``SC_AVPHYS_PAGES`` (rare; some BSDs, native Windows)
        returns half of the total memory as a best-effort estimate.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"MPSBackend has 1 device (index {self._DEVICE_INDEX}); requested index {i}"
            )
        if sys.platform == "win32":
            total = self.total_memory_mb(i)
            return total // _FALLBACK_FREE_RATIO
        free = _sysconf_bytes(_SC_AVPHYS_PAGES)
        if free is None:
            total = self.total_memory_mb(i)
            return total // _FALLBACK_FREE_RATIO
        return free // _BYTES_PER_MB

    def vendor(self) -> GpuVendor:
        """Return the hardware vendor (always :attr:`GpuVendor.APPLE`)."""
        return GpuVendor.APPLE

    def to_ultralytics_string(self, i: int) -> str:
        """Return the ultralytics device string for device ``i``.

        Ultralytics expects the literal string ``"mps"`` for Apple
        Silicon execution. Unlike CUDA/ROCm, MPS has no index suffix
        because there is only one device.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"MPSBackend has 1 device (index {self._DEVICE_INDEX}); requested index {i}"
            )
        return "mps"

    def warmup(self) -> None:
        """No-op for MPS.

        The Metal device context is created lazily by the first real
        tensor operation. There is no per-process initialization we
        can usefully trigger up-front, and the :class:`CPUBackend`
        contract is that :meth:`warmup` is allowed to do nothing.
        """
        return None


# Module-level singleton. Other modules (registry, CLI, pipeline)
# import this constant rather than constructing a fresh MPSBackend
# every time. The singleton is safe even on systems where MPS is
# unusable — :meth:`is_available` will simply return ``False``.
MPS_BACKEND = MPSBackend()

__all__ = ["MPS_BACKEND", "MPSBackend"]
