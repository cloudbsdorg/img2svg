# img2svg - tests for the Apple Metal Performance Shaders backend.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`img2svg.backends.mps.MPSBackend`.

The MPS backend is the Apple Silicon accelerator. These tests run on
**every** host (Linux CI, Windows dev, macOS M-series) and must pass
regardless of whether MPS hardware is actually present. Where the
backend's ``is_available`` probe is exercised, the torch module is
mocked so the test outcome is independent of the local PyTorch build.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest import mock

import pytest

from img2svg.backends import MPS_BACKEND, MPSBackend
from img2svg.backends.mps import MPSBackend as MPSBackendDirect
from img2svg.backends.protocol import BackendType
from img2svg.enums import GpuVendor

# --- 1. is_available when torch is missing or MPS attr is absent --------------


def test_is_available_false_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_available()`` must return ``False`` when torch cannot be imported.

    Simulates a Python environment without PyTorch (e.g. a minimal
    doc-build venv). The backend must never raise from this method.
    """
    # Force ``import torch`` to raise ImportError. We patch the
    # ``import`` machinery so the lazy torch import inside
    # ``MPSBackend.is_available`` sees a missing module.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed (test stub)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert MPSBackend().is_available() is False


def test_is_available_false_when_mps_attribute_missing() -> None:
    """``is_available()`` returns ``False`` when ``torch.backends.mps`` is absent.

    Mirrors the situation on Linux/Windows PyTorch wheels (and older
    macOS builds) where the ``mps`` attribute is simply not defined.
    A naive ``torch.backends.mps.is_available()`` would raise
    :class:`AttributeError`; the backend must swallow that.
    """

    class _FakeBackends:
        # Deliberately no ``mps`` attribute.
        pass

    fake_torch = mock.MagicMock()
    fake_torch.backends = _FakeBackends()

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        assert MPSBackend().is_available() is False


# --- 2. is_available when MPS reports True -----------------------------------


def test_is_available_true_when_torch_reports_available() -> None:
    """``is_available()`` returns ``True`` when ``torch.backends.mps.is_available()`` does.

    Patches a fake ``torch.backends.mps`` whose ``is_available()``
    returns ``True``. The backend must pass that through.
    """
    fake_mps = mock.MagicMock()
    fake_mps.is_available.return_value = True
    fake_torch = mock.MagicMock()
    fake_torch.backends.mps = fake_mps

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        assert MPSBackend().is_available() is True
        fake_mps.is_available.assert_called_once_with()


# --- 3. to_ultralytics_string ------------------------------------------------


def test_to_ultralytics_string_is_mps() -> None:
    """ultralytics expects the literal string ``"mps"`` (no index suffix)."""
    assert MPSBackend().to_ultralytics_string(0) == "mps"


# --- 4. vendor ---------------------------------------------------------------


def test_vendor_is_apple() -> None:
    """The backend reports :attr:`GpuVendor.APPLE`."""
    assert MPSBackend().vendor() == GpuVendor.APPLE


# --- 5. type -----------------------------------------------------------------


def test_type_is_mps() -> None:
    """The backend reports :attr:`BackendType.MPS`."""
    assert MPSBackend().type() == BackendType.MPS


# --- 6. device_count ---------------------------------------------------------


def test_device_count_zero_when_unavailable() -> None:
    """``device_count()`` is 0 when MPS is not usable (the typical Linux case)."""
    # ``is_available`` will return False on this Linux box (no
    # ``torch.backends.mps`` attribute on the CPU wheel).
    assert MPSBackend().device_count() == 0


def test_device_count_one_when_available() -> None:
    """``device_count()`` is 1 when MPS is usable (mocked)."""
    fake_mps = mock.MagicMock()
    fake_mps.is_available.return_value = True
    fake_torch = mock.MagicMock()
    fake_torch.backends.mps = fake_mps

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        assert MPSBackend().device_count() == 1


# --- 7. device_name ----------------------------------------------------------


def test_device_name_returns_non_empty_string() -> None:
    """``device_name(0)`` always returns a non-empty string.

    On a non-macOS host the backend falls back to ``platform.processor()``
    or the stable ``"Apple Silicon"`` label — either way the result is
    a non-empty string. The contract is "never raise, never return
    empty" so the CLI can render the list without special-casing.
    """
    name = MPSBackend().device_name(0)
    assert isinstance(name, str)
    assert name.strip() != ""


# --- bonus: index-out-of-range contract (matches CPU backend) ----------------


def test_invalid_index_raises() -> None:
    """Index-taking methods raise :class:`IndexError` for out-of-range indices."""
    backend = MPSBackend()
    with pytest.raises(IndexError):
        backend.device_name(1)
    with pytest.raises(IndexError):
        backend.total_memory_mb(5)
    with pytest.raises(IndexError):
        backend.free_memory_mb(-1)
    with pytest.raises(IndexError):
        backend.to_ultralytics_string(2)


# --- bonus: protocol surface -------------------------------------------------


def test_satisfies_device_backend_protocol() -> None:
    """``MPSBackend`` exposes every method of the :class:`DeviceBackend` protocol."""
    backend = MPSBackend()
    for method in (
        "type",
        "is_available",
        "device_count",
        "device_name",
        "total_memory_mb",
        "free_memory_mb",
        "vendor",
        "to_ultralytics_string",
        "warmup",
    ):
        assert callable(getattr(backend, method)), f"missing method: {method}"


# --- bonus: module singleton -------------------------------------------------


def test_module_singleton_is_mps_backend() -> None:
    """``MPS_BACKEND`` is a usable :class:`MPSBackend` instance."""
    assert isinstance(MPS_BACKEND, MPSBackend)
    assert isinstance(MPS_BACKEND, MPSBackendDirect)
    # The Linux test host has no MPS, so the singleton reports False —
    # this is the expected cross-platform behaviour.
    assert MPS_BACKEND.is_available() is False
    assert MPS_BACKEND.to_ultralytics_string(0) == "mps"
    assert MPS_BACKEND.vendor() == GpuVendor.APPLE
    assert MPS_BACKEND.type() == BackendType.MPS
    # ``warmup`` is contractually a no-op for the MPS backend.
    assert MPS_BACKEND.warmup() is None


# --- bonus: memory queries on the real host (no torch patching) -------------


def test_total_memory_mb_positive_on_this_host() -> None:
    """Total memory in MiB is positive on any real POSIX host.

    This runs on the same code path that production uses, without
    mocking torch, so it validates the ``os.sysconf`` query end-to-end.
    """
    total = MPSBackend().total_memory_mb(0)
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows
        assert total == 0
    else:
        assert isinstance(total, int)
        assert total > 0


def test_free_memory_mb_non_negative_on_this_host() -> None:
    """Free memory is non-negative and never exceeds total memory."""
    backend = MPSBackend()
    free = backend.free_memory_mb(0)
    total = backend.total_memory_mb(0)
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows
        assert free == total // 2
    else:
        assert isinstance(free, int)
        assert free >= 0
        assert free <= total
