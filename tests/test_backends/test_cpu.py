# img2svg - tests for the CPU compute backend.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`img2svg.backends.cpu.CPUBackend`."""

from __future__ import annotations

import pytest

from img2svg.backends import CPU_BACKEND, CPUBackend
from img2svg.backends.cpu import CPUBackend as CPUBackendDirect
from img2svg.backends.protocol import BackendType
from img2svg.enums import GpuVendor


def test_is_available_always_true() -> None:
    """CPU is the always-available fallback — every other backend relies on it."""
    backend = CPUBackend()
    assert backend.is_available() is True
    # Idempotent: the result is a constant.
    assert backend.is_available() is True


def test_device_count_is_one() -> None:
    """The CPU backend represents the single host system."""
    assert CPUBackend().device_count() == 1


def test_type_is_cpu() -> None:
    """The backend reports its :class:`BackendType` correctly."""
    assert CPUBackend().type() == BackendType.CPU


def test_to_ultralytics_string() -> None:
    """The ultralytics device string is the literal ``"cpu"``."""
    assert CPUBackend().to_ultralytics_string(0) == "cpu"


def test_total_memory_mb_positive() -> None:
    """Any real system has > 0 MiB of RAM."""
    total = CPUBackend().total_memory_mb(0)
    assert isinstance(total, int)
    assert total > 0


def test_free_memory_mb_non_negative() -> None:
    """Free RAM must be non-negative and not exceed total RAM."""
    backend = CPUBackend()
    free = backend.free_memory_mb(0)
    total = backend.total_memory_mb(0)
    assert isinstance(free, int)
    assert free >= 0
    # Sanity: free cannot exceed total.
    assert free <= total


def test_vendor_is_cpu() -> None:
    """The backend advertises the new :attr:`GpuVendor.CPU` value."""
    assert CPUBackend().vendor() == GpuVendor.CPU


def test_device_name_returns_non_empty_string() -> None:
    """``device_name(0)`` always returns a non-empty string."""
    name = CPUBackend().device_name(0)
    assert isinstance(name, str)
    assert name.strip() != ""


def test_warmup_is_noop() -> None:
    """``warmup()`` returns ``None`` and does not raise."""
    assert CPUBackend().warmup() is None


def test_invalid_index_raises() -> None:
    """Out-of-range index raises :class:`IndexError` for every index-taking method."""
    backend = CPUBackend()
    with pytest.raises(IndexError):
        backend.device_name(1)
    with pytest.raises(IndexError):
        backend.total_memory_mb(5)
    with pytest.raises(IndexError):
        backend.free_memory_mb(-1)
    with pytest.raises(IndexError):
        backend.to_ultralytics_string(2)


def test_module_singleton_is_cpu_backend() -> None:
    """``CPU_BACKEND`` is a usable :class:`CPUBackend` instance."""
    assert isinstance(CPU_BACKEND, CPUBackend)
    assert isinstance(CPU_BACKEND, CPUBackendDirect)
    assert CPU_BACKEND.is_available() is True
    assert CPU_BACKEND.to_ultralytics_string(0) == "cpu"


def test_satisfies_device_backend_protocol() -> None:
    """``CPUBackend`` exposes every method of the :class:`DeviceBackend` protocol."""
    backend = CPUBackend()
    # All nine methods must be present and callable; if a future refactor
    # accidentally drops one, this test fails with AttributeError.
    assert callable(backend.type)
    assert callable(backend.is_available)
    assert callable(backend.device_count)
    assert callable(backend.device_name)
    assert callable(backend.total_memory_mb)
    assert callable(backend.free_memory_mb)
    assert callable(backend.vendor)
    assert callable(backend.to_ultralytics_string)
    assert callable(backend.warmup)
