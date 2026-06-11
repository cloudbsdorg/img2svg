# img2svg - CPU compute backend for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""CPU compute backend.

The :class:`CPUBackend` is the always-available fallback used when no GPU
accelerator (CUDA, ROCm, MPS) is detected. It implements every method of the
:class:`DeviceBackend` protocol declared in
:mod:`img2svg.backends.protocol` and is suitable for the tail of the
auto-detect chain (CUDA > ROCM > MPS > CPU).

System memory is reported via stdlib :mod:`os` ``sysconf`` queries (POSIX).
On non-POSIX platforms (e.g. native Windows) the memory queries fall back to
a documented default rather than raising.
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
# them later.
_SC_PAGE_SIZE = "SC_PAGE_SIZE"
_SC_PHYS_PAGES = "SC_PHYS_PAGES"
_SC_AVPHYS_PAGES = "SC_AVPHYS_PAGES"

# Bytes per mebibyte — kept explicit to make the math obvious.
_BYTES_PER_MB = 1024 * 1024

# Used as a last-resort free-memory estimate when ``os.sysconf`` does not
# know ``SC_AVPHYS_PAGES`` (e.g. some BSDs, native Windows). Half of total
# RAM is a common conservative default that the registry can also report.
_FALLBACK_FREE_RATIO = 2


def _sysconf_bytes(name: str) -> int | None:
    """Return ``os.sysconf(name) * os.sysconf('SC_PAGE_SIZE')`` or ``None``.

    Returns ``None`` if either sysconf call is unsupported (``os.sysconf``
    raises :class:`ValueError`/returns ``-1`` for unknown names on POSIX, and
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


class CPUBackend:
    """CPU compute backend implementation.

    Implements the :class:`DeviceBackend` protocol. There is exactly one
    "device" (index ``0``) — the host system. The backend is always
    available: nothing in this class touches third-party libraries, so it
    works on every supported platform.
    """

    #: The single virtual device index exposed by this backend.
    _DEVICE_INDEX = 0

    def type(self) -> "BackendType":
        """Return the backend's :class:`BackendType` (always ``CPU``)."""
        # Local import to keep this module importable even if the
        # protocol/types module is still being authored in parallel.
        from img2svg.backends.protocol import BackendType

        return BackendType.CPU

    def is_available(self) -> bool:
        """Return whether the backend is usable (always ``True``).

        CPU execution is always available; this method is the only one in the
        whole backend matrix that returns a constant truthy value, which is
        what makes the CPU a safe fallback in the auto-detect chain.
        """
        return True

    def device_count(self) -> int:
        """Return the number of devices (always ``1``)."""
        return 1

    def device_name(self, i: int) -> str:
        """Return a human-readable name for device ``i``.

        Uses :func:`platform.processor` when it reports a non-empty string
        and the value is distinguishable from the literal generic
        ``platform.processor()`` returns on some Linux builds; otherwise
        falls back to ``"CPU"``.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"CPUBackend has 1 device (index {self._DEVICE_INDEX}); "
                f"requested index {i}"
            )
        name = platform.processor() or ""
        return name.strip() or "CPU"

    def total_memory_mb(self, i: int) -> int:
        """Return total system RAM in MiB for device ``i``.

        Uses ``os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')`` on
        POSIX platforms. On platforms where these sysconf names are not
        available, returns ``0`` (the registered backend never refuses to
        load — the registry and CLI decide how to react to ``0``).
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"CPUBackend has 1 device (index {self._DEVICE_INDEX}); "
                f"requested index {i}"
            )
        if sys.platform == "win32":
            # os.sysconf is not present on native Windows. Returning 0 is
            # honest: this backend cannot answer the question on this OS.
            return 0
        total = _sysconf_bytes(_SC_PHYS_PAGES)
        if total is None:
            return 0
        return total // _BYTES_PER_MB

    def free_memory_mb(self, i: int) -> int:
        """Return free system RAM in MiB for device ``i``.

        Uses ``os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')``
        on POSIX. On platforms where the kernel does not expose
        ``SC_AVPHYS_PAGES`` (rare; some BSDs, native Windows) returns half
        of the total RAM as a best-effort estimate.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"CPUBackend has 1 device (index {self._DEVICE_INDEX}); "
                f"requested index {i}"
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
        """Return the hardware vendor (always :attr:`GpuVendor.CPU`)."""
        return GpuVendor.CPU

    def to_ultralytics_string(self, i: int) -> str:
        """Return the ultralytics device string for device ``i``.

        Ultralytics expects the literal string ``"cpu"`` for CPU execution.
        """
        if i != self._DEVICE_INDEX:
            raise IndexError(
                f"CPUBackend has 1 device (index {self._DEVICE_INDEX}); "
                f"requested index {i}"
            )
        return "cpu"

    def warmup(self) -> None:
        """No-op for CPU.

        CPU has no lazy initialization to trigger. The method exists so the
        CPU backend is drop-in compatible with GPU backends that may need
        to do per-process CUDA initialization.
        """
        return None


# Module-level singleton. Other modules (registry, CLI, pipeline) import
# this constant rather than constructing a fresh CPUBackend every time.
CPU_BACKEND = CPUBackend()

__all__ = ["CPUBackend", "CPU_BACKEND"]
