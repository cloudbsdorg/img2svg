# img2svg - DeviceBackend protocol and BackendType enum.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""DeviceBackend protocol and BackendType enum.

This module defines the vendor-neutral abstraction that all concrete
backends (CPU, CUDA, ROCm, MPS, ...) must implement. The rest of the
codebase depends on the :class:`DeviceBackend` protocol, not on a
specific vendor's PyTorch API.

Design notes:

* :class:`BackendType` is a :class:`enum.StrEnum` (with a Python 3.10
  shim) so selectors round-trip cleanly through CLI / config layers.
* :class:`DeviceBackend` is decorated with :func:`typing.runtime_checkable`
  so tests and factories can use ``isinstance(obj, DeviceBackend)`` for
  structural conformance checks.
* This module deliberately does **not** import ``torch``; it is a pure
  type-level contract. Vendor-specific glue lives in sibling modules
  (e.g. ``cuda.py``, ``rocm.py``, ``mps.py``, ``cpu.py``).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from img2svg.enums import GpuVendor

# `enum.StrEnum` was added in Python 3.11. The project supports 3.10+,
# so fall back to a minimal shim when running on an older interpreter.
# We intentionally do NOT add a `strenum` dependency for this — the
# shim is two lines and keeps the runtime surface identical across
# supported Python versions.
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """StrEnum shim for Python <3.11."""


class BackendType(StrEnum):
    """Selector key for a compute backend.

    Used by the user-facing CLI / config layer (``--device``) and by the
    backend registry to map a string token onto a concrete
    :class:`DeviceBackend` implementation.

    Values are lowercase for ergonomic round-tripping with shell tooling.
    ``AUTO`` is a meta-selector (resolved by the registry at runtime)
    rather than a concrete backend.
    """

    AUTO = "auto"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


@runtime_checkable
class DeviceBackend(Protocol):
    """Vendor-neutral compute backend contract.

    Every backend (CPU, NVIDIA CUDA, AMD ROCm, Apple MPS, ...) implements
    these methods. Methods that take a device index ``i`` are
    zero-indexed in backend-local space (e.g. ``CudaBackend(0)`` is the
    first CUDA device, regardless of how the OS numbers PCI slots).

    The protocol is decorated with :func:`typing.runtime_checkable` so
    ``isinstance(obj, DeviceBackend)`` works in tests and the registry.
    Implementations do not need to subclass this protocol — they only
    need to provide methods with matching names and signatures.
    """

    def type(self) -> BackendType:
        """Return the concrete :class:`BackendType` of this backend.

        For a back-end that can serve either CUDA or ROCm, this returns
        the variant that the host actually supports.
        """
        ...

    def is_available(self) -> bool:
        """Return True if this backend is usable on the current host.

        A backend may be importable but not usable (e.g. the ``torch``
        CUDA build is missing, or the driver is not loaded). Callers
        should treat False as "skip this backend, try the next one".
        """
        ...

    def device_count(self) -> int:
        """Return the number of devices visible to this backend.

        Returns ``0`` when the backend is unavailable or the host has
        no devices. Callers MUST treat 0 as "no GPU" and fall back to
        CPU rather than indexing into an empty list.
        """
        ...

    def device_name(self, i: int) -> str:
        """Return a human-readable marketing name for device ``i``.

        Used for ``list-gpus`` display and for diagnostic logging.
        ``i`` is in the range ``[0, device_count())``.
        """
        ...

    def total_memory_mb(self, i: int) -> int:
        """Return total VRAM/RAM in mebibytes for device ``i``.

        For shared-memory devices (Apple MPS, AMD APUs), reports the
        fraction of system RAM the runtime exposes. Returns ``0`` when
        the value is unknown.
        """
        ...

    def free_memory_mb(self, i: int) -> int:
        """Return free VRAM/RAM in mebibytes for device ``i``.

        Used by the recommendation logic to pick the least-loaded
        device. Returns ``0`` when the value is unknown.
        """
        ...

    def vendor(self) -> GpuVendor:
        """Return the :class:`img2svg.enums.GpuVendor` of this backend.

        Used for display and for vendor-specific capability checks.
        """
        ...

    def to_ultralytics_string(self, i: int) -> str:
        """Translate a device index to the string form ultralytics expects.

        Examples:

        * ``"cuda:0"`` for an NVIDIA CUDA device
        * ``"mps"`` for Apple Silicon
        * ``"cpu"`` for the CPU backend
        * ``"0"`` for ROCm (ultralytics treats ROCm as CUDA; the
          integer form is acceptable)

        ``i`` is in the range ``[0, device_count())``.
        """
        ...

    def warmup(self) -> None:
        """Perform any lazy initialization needed before first use.

        Default is a no-op. Backends with expensive first-call
        initialization (e.g. CUDA context creation) may override to
        trigger that work eagerly, so the first real call doesn't pay
        the latency penalty. Must not raise on unavailable hardware.
        """
        ...
