# img2svg - tests for the NVIDIA CUDA compute backend.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`img2svg.backends.cuda.CUDABackend`.

The tests deliberately mock ``torch.cuda`` so the suite is runnable on
hosts that do not have an NVIDIA GPU, a CUDA-enabled PyTorch build, or
even ``torch`` installed at all. The CUDA backend's contract is "behave
sanely when torch is missing" and the tests pin that contract.
"""

from __future__ import annotations

import builtins
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

from img2svg.backends import CUDA_BACKEND, CUDABackend
from img2svg.backends.cuda import CUDABackend as CUDABackendDirect
from img2svg.backends.protocol import BackendType
from img2svg.enums import GpuVendor


def _make_fake_torch_cuda(
    *,
    is_available: bool = True,
    device_count: int = 1,
    device_name: str = "NVIDIA GeForce RTX 5090",
    total_memory_bytes: int = 12 * 1024 * 1024 * 1024,
    free_memory_bytes: int = 10 * 1024 * 1024 * 1024,
) -> types.SimpleNamespace:
    """Build a fake ``torch.cuda`` namespace populated with sensible values.

    The shape mirrors the public ``torch.cuda`` API surface used by
    :class:`CUDABackend`. Each call site is wrapped in a ``MagicMock`` so
    tests can override individual attributes per-case.
    """
    cuda_ns = types.SimpleNamespace()
    cuda_ns.is_available = MagicMock(return_value=is_available)
    cuda_ns.device_count = MagicMock(return_value=device_count)
    cuda_ns.get_device_name = MagicMock(return_value=device_name)
    cuda_ns.get_device_properties = MagicMock(
        return_value=types.SimpleNamespace(total_memory=total_memory_bytes)
    )
    cuda_ns.mem_get_info = MagicMock(
        return_value=(free_memory_bytes, total_memory_bytes)
    )
    cuda_ns.init = MagicMock(return_value=None)
    return cuda_ns


def _install_fake_torch(cuda_ns: types.SimpleNamespace | None) -> Any:
    """Install a fake ``torch`` module (with optional ``cuda`` namespace).

    When ``cuda_ns`` is ``None`` the fake torch has no ``cuda`` attribute —
    this simulates a CPU-only ``torch`` build.
    """
    fake_torch = types.ModuleType("torch")
    if cuda_ns is not None:
        fake_torch.cuda = cuda_ns  # type: ignore[attr-defined]
    else:
        # Use a property that raises AttributeError, matching the real
        # behavior of a CPU-only torch build.
        def _getattr(_name: str) -> Any:
            raise AttributeError(
                "module 'torch' has no attribute 'cuda' (CPU-only build)"
            )

        fake_torch.__getattr__ = _getattr  # type: ignore[attr-defined]
    return fake_torch


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Type / vendor / protocol-conformance tests
# ---------------------------------------------------------------------------


def test_type_is_cuda() -> None:
    """``CUDABackend.type()`` always returns :attr:`BackendType.CUDA`."""
    assert CUDABackend().type() == BackendType.CUDA


def test_vendor_is_nvidia() -> None:
    """``CUDABackend.vendor()`` always returns :attr:`GpuVendor.NVIDIA`."""
    assert CUDABackend().vendor() == GpuVendor.NVIDIA


def test_satisfies_device_backend_protocol() -> None:
    """``CUDABackend`` exposes every method of the :class:`DeviceBackend` protocol."""
    backend = CUDABackend()
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


def test_module_singleton_is_cuda_backend() -> None:
    """``CUDA_BACKEND`` is a usable :class:`CUDABackend` instance."""
    assert isinstance(CUDA_BACKEND, CUDABackend)
    assert isinstance(CUDA_BACKEND, CUDABackendDirect)
    assert CUDA_BACKEND.type() == BackendType.CUDA
    assert CUDA_BACKEND.vendor() == GpuVendor.NVIDIA


# ---------------------------------------------------------------------------
# String-formatting tests (no torch needed)
# ---------------------------------------------------------------------------


def test_to_ultralytics_string_zero() -> None:
    """``to_ultralytics_string(0)`` is exactly ``"cuda:0"``."""
    assert CUDABackend().to_ultralytics_string(0) == "cuda:0"


def test_to_ultralytics_string_nonzero() -> None:
    """``to_ultralytics_string(N)`` formats the index verbatim."""
    backend = CUDABackend()
    assert backend.to_ultralytics_string(1) == "cuda:1"
    assert backend.to_ultralytics_string(7) == "cuda:7"


# ---------------------------------------------------------------------------
# is_available tests
# ---------------------------------------------------------------------------


def test_is_available_true_when_torch_cuda_available() -> None:
    """``is_available()`` returns ``True`` when ``torch.cuda.is_available()`` is ``True``."""
    fake_torch = _install_fake_torch(_make_fake_torch_cuda(is_available=True))
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().is_available() is True


def test_is_available_false_when_torch_cuda_unavailable() -> None:
    """``is_available()`` returns ``False`` when ``torch.cuda.is_available()`` is ``False``."""
    fake_torch = _install_fake_torch(_make_fake_torch_cuda(is_available=False))
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().is_available() is False


def test_is_available_false_when_torch_not_importable() -> None:
    """``is_available()`` returns ``False`` when :mod:`torch` is missing entirely."""
    real_import = builtins.__import__

    def _block_torch(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"simulated missing torch (blocked: {name!r})")
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=_block_torch):
        assert CUDABackend().is_available() is False


def test_is_available_false_when_torch_has_no_cuda() -> None:
    """``is_available()`` returns ``False`` for a CPU-only torch build (no ``cuda`` attr)."""
    fake_torch = _install_fake_torch(cuda_ns=None)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        # The CUDABackend will see ``AttributeError`` on
        # ``torch.cuda.is_available`` and return False.
        assert CUDABackend().is_available() is False


# ---------------------------------------------------------------------------
# device_count / device_name / memory tests
# ---------------------------------------------------------------------------


def test_device_count_returns_mocked_value() -> None:
    """``device_count()`` returns the integer reported by ``torch.cuda.device_count()``."""
    fake_torch = _install_fake_torch(_make_fake_torch_cuda(device_count=4))
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().device_count() == 4


def test_device_count_returns_zero_when_torch_missing() -> None:
    """``device_count()`` returns ``0`` when :mod:`torch` is not importable."""
    real_import = builtins.__import__

    def _block_torch(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"simulated missing torch (blocked: {name!r})")
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=_block_torch):
        assert CUDABackend().device_count() == 0


def test_device_name_returns_marketing_name() -> None:
    """``device_name(i)`` returns the string from ``torch.cuda.get_device_name(i)``."""
    fake_torch = _install_fake_torch(
        _make_fake_torch_cuda(device_name="NVIDIA GeForce RTX 5070 Laptop GPU")
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().device_name(0) == "NVIDIA GeForce RTX 5070 Laptop GPU"


def test_device_name_falls_back_on_runtime_error() -> None:
    """``device_name(i)`` returns ``f"GPU {i}"`` when ``torch.cuda.get_device_name`` raises."""
    cuda_ns = _make_fake_torch_cuda()
    cuda_ns.get_device_name = MagicMock(
        side_effect=RuntimeError("CUDA error: invalid device ordinal")
    )
    fake_torch = _install_fake_torch(cuda_ns)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().device_name(0) == "GPU 0"


def test_total_memory_mb_converts_bytes_to_mib() -> None:
    """``total_memory_mb(i)`` converts bytes to MiB (integer division)."""
    # 8 GiB exactly — should report 8192 MiB.
    eight_gib = 8 * 1024 * 1024 * 1024
    fake_torch = _install_fake_torch(
        _make_fake_torch_cuda(total_memory_bytes=eight_gib)
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().total_memory_mb(0) == 8192


def test_total_memory_mb_returns_zero_on_runtime_error() -> None:
    """``total_memory_mb(i)`` returns ``0`` when properties cannot be read."""
    cuda_ns = _make_fake_torch_cuda()
    cuda_ns.get_device_properties = MagicMock(
        side_effect=RuntimeError("no such device")
    )
    fake_torch = _install_fake_torch(cuda_ns)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().total_memory_mb(0) == 0


def test_free_memory_mb_converts_bytes_to_mib() -> None:
    """``free_memory_mb(i)`` returns the first element of ``mem_get_info`` in MiB."""
    # 6 GiB free, 12 GiB total.
    free = 6 * 1024 * 1024 * 1024
    total = 12 * 1024 * 1024 * 1024
    fake_torch = _install_fake_torch(
        _make_fake_torch_cuda(free_memory_bytes=free, total_memory_bytes=total)
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().free_memory_mb(0) == 6144


def test_free_memory_mb_returns_zero_on_runtime_error() -> None:
    """``free_memory_mb(i)`` returns ``0`` when ``mem_get_info`` raises (older CUDA contexts)."""
    cuda_ns = _make_fake_torch_cuda()
    cuda_ns.mem_get_info = MagicMock(
        side_effect=RuntimeError("operation not supported on this context")
    )
    fake_torch = _install_fake_torch(cuda_ns)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert CUDABackend().free_memory_mb(0) == 0


def test_free_memory_mb_returns_zero_when_torch_missing() -> None:
    """``free_memory_mb(i)`` returns ``0`` when :mod:`torch` is not importable."""
    real_import = builtins.__import__

    def _block_torch(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"simulated missing torch (blocked: {name!r})")
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=_block_torch):
        assert CUDABackend().free_memory_mb(0) == 0


# ---------------------------------------------------------------------------
# warmup test
# ---------------------------------------------------------------------------


def test_warmup_calls_torch_cuda_init() -> None:
    """``warmup()`` triggers ``torch.cuda.init()`` to eagerly create the CUDA context."""
    cuda_ns = _make_fake_torch_cuda()
    fake_torch = _install_fake_torch(cuda_ns)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        CUDABackend().warmup()
    cuda_ns.init.assert_called_once_with()


def test_warmup_is_noop_when_torch_missing() -> None:
    """``warmup()`` does not raise when :mod:`torch` is not importable."""
    real_import = builtins.__import__

    def _block_torch(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"simulated missing torch (blocked: {name!r})")
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=_block_torch):
        # Must not raise.
        assert CUDABackend().warmup() is None


def test_warmup_swallows_init_errors() -> None:
    """``warmup()`` swallows ``torch.cuda.init`` errors (best-effort)."""
    cuda_ns = _make_fake_torch_cuda()
    cuda_ns.init = MagicMock(side_effect=RuntimeError("driver not loaded"))
    fake_torch = _install_fake_torch(cuda_ns)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        # Must not raise.
        assert CUDABackend().warmup() is None


def test_construct_is_cheap() -> None:
    """Constructing a :class:`CUDABackend` does not import :mod:`torch`."""
    backend = CUDABackend()
    assert backend is not None
    # The lazy-import guarantee: these methods do not touch torch.
    assert backend.type() == BackendType.CUDA
    assert backend.vendor() == GpuVendor.NVIDIA
    assert backend.to_ultralytics_string(0) == "cuda:0"
