# img2svg - tests for the AMD ROCm compute backend.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`img2svg.backends.rocm.ROCMBackend`.

The :class:`ROCMBackend` lazy-imports :mod:`torch` inside each method, so the
test strategy is to install a fake ``torch`` module into :data:`sys.modules`
for the duration of each test. The fake exposes the small slice of the
``torch.cuda`` and ``torch.version`` API the backend probes, plus
``version.hip`` which is the single most important attribute for ROCm
detection (``None`` for CUDA builds, a version string for ROCm builds).

Why ``sys.modules`` injection over :func:`unittest.mock.patch`:
the backend does ``import torch`` inside each method body, so a
direct :func:`unittest.mock.patch` on the import statement would have
to be repeated for every test method. Installing a controllable fake
module into :data:`sys.modules` once per test gives the lazy import a
real (if fake) target and keeps the test bodies focused on behavior.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from img2svg.backends import ROCM_BACKEND, ROCMBackend
from img2svg.backends.rocm import ROCMBackend as ROCMBackendDirect
from img2svg.enums import GpuVendor
from img2svg.backends.protocol import BackendType


def _install_fake_torch(
    *,
    cuda_is_available: bool,
    hip: str | None,
    device_count: int = 0,
    device_name: str = "Radeon RX 7900 XT",
    total_memory_bytes: int = 24 * 1024 * 1024 * 1024,
    free_memory_bytes: int = 20 * 1024 * 1024 * 1024,
) -> types.ModuleType:
    """Install a controllable fake ``torch`` module into :data:`sys.modules`.

    Returns the installed module so tests can introspect it. The fake
    is removed (and any prior ``torch`` restored) by the ``_fake_torch``
    fixture's teardown.
    """
    torch_mod = types.ModuleType("torch")
    torch_mod.version = types.SimpleNamespace(hip=hip)
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = MagicMock(return_value=cuda_is_available)
    cuda_mod.device_count = MagicMock(return_value=device_count)
    cuda_mod.get_device_name = MagicMock(return_value=device_name)
    cuda_mod.get_device_properties = MagicMock(
        return_value=types.SimpleNamespace(total_memory=total_memory_bytes)
    )
    cuda_mod.mem_get_info = MagicMock(
        return_value=(free_memory_bytes, total_memory_bytes)
    )
    torch_mod.cuda = cuda_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.cuda"] = cuda_mod
    return torch_mod


@pytest.fixture
def _fake_torch() -> types.ModuleType:
    """Install a fake ``torch`` and restore the real one on teardown.

    Yields the fake module so tests can tweak its attributes
    mid-test (e.g. flip ``cuda.is_available`` to ``False``) before
    calling the backend method under test.
    """
    prior_torch = sys.modules.get("torch")
    prior_torch_cuda = sys.modules.get("torch.cuda")
    try:
        yield _install_fake_torch(
            cuda_is_available=True,
            hip="6.2.41134",
            device_count=1,
        )
    finally:
        if prior_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = prior_torch
        if prior_torch_cuda is None:
            sys.modules.pop("torch.cuda", None)
        else:
            sys.modules["torch.cuda"] = prior_torch_cuda


def test_is_available_false_when_hip_is_none(_fake_torch: Any) -> None:
    """The defining check: a plain CUDA PyTorch build must NOT claim ROCm.

    On a host with a CUDA build of PyTorch, ``torch.version.hip`` is
    ``None``. Even when the CUDA driver is loaded and GPUs are
    visible, the ROCm backend must report itself as unavailable —
    otherwise the auto-detect chain would route AMD-bound work to the
    wrong runtime.
    """
    _fake_torch.version.hip = None
    _fake_torch.cuda.is_available.return_value = True
    backend = ROCMBackend()
    assert backend.is_available() is False


def test_is_available_true_when_hip_and_cuda_available(_fake_torch: Any) -> None:
    """Both conditions met: ROCm PyTorch build + driver loaded + GPU visible."""
    # _fake_torch fixture already sets cuda_is_available=True and hip="6.2.41134"
    backend = ROCMBackend()
    assert backend.is_available() is True


def test_is_available_false_when_cuda_unavailable_even_with_hip(
    _fake_torch: Any,
) -> None:
    """ROCm build present, but no GPU / no driver — backend stays unavailable.

    The reverse case of the defining test: ``torch.version.hip`` is set
    (so it really is a ROCm build) but ``torch.cuda.is_available()``
    returns ``False`` (no driver loaded, or no AMD GPU in the host).
    The ROCm backend must still report ``False`` — a ROCm build without
    a working runtime is no more useful than a CUDA build without one.
    """
    _fake_torch.cuda.is_available.return_value = False
    _fake_torch.version.hip = "6.2.41134"  # still a ROCm build
    backend = ROCMBackend()
    assert backend.is_available() is False


def test_to_ultralytics_string_returns_cuda_prefix() -> None:
    """Ultralytics treats ROCm devices as CUDA — the string is ``"cuda:N"``.

    This test does not need the fake-torch fixture because
    :meth:`to_ultralytics_string` is the only method that does not
    query :mod:`torch` at all.
    """
    assert ROCMBackend().to_ultralytics_string(0) == "cuda:0"
    assert ROCMBackend().to_ultralytics_string(2) == "cuda:2"


def test_to_ultralytics_string_rejects_negative_index() -> None:
    """Defensive: a negative device index is meaningless for the runtime."""
    with pytest.raises(IndexError):
        ROCMBackend().to_ultralytics_string(-1)


def test_vendor_is_amd() -> None:
    """The backend advertises :attr:`GpuVendor.AMD` regardless of runtime state."""
    assert ROCMBackend().vendor() == GpuVendor.AMD


def test_type_is_rocm() -> None:
    """The backend's :class:`BackendType` is :attr:`BackendType.ROCM`."""
    assert ROCMBackend().type() == BackendType.ROCM


def test_device_name_has_rocm_prefix(_fake_torch: Any) -> None:
    """``device_name(0)`` returns a string starting with ``"[ROCm] "``.

    The prefix disambiguates AMD devices from NVIDIA devices in
    ``img2svg list-gpus`` output when both vendors coexist on a host.
    """
    backend = ROCMBackend()
    name = backend.device_name(0)
    assert isinstance(name, str)
    assert name.startswith("[ROCm] ")
    # The vendor-supplied name follows the prefix verbatim.
    assert name == "[ROCm] Radeon RX 7900 XT"


def test_device_name_raises_when_rocm_unavailable() -> None:
    """Calling :meth:`device_name` outside an available ROCm runtime raises.

    Mirrors the contract: methods that depend on live hardware must
    refuse to answer rather than fabricating a label when the backend
    itself reports unavailable.
    """
    # No fake torch installed — `is_available()` returns False because
    # either torch is missing or the real torch is a CUDA build on
    # this system.
    with pytest.raises(RuntimeError, match="not available"):
        ROCMBackend().device_name(0)


def test_device_count_zero_when_unavailable() -> None:
    """``device_count()`` returns ``0`` when the backend is unavailable.

    The contract is "callers MUST treat 0 as no GPU and fall back to
    the next backend" — verified here without any torch mock because
    the real environment is a CUDA build (hip is None) and
    :meth:`is_available` returns ``False``.
    """
    assert ROCMBackend().device_count() == 0


def test_device_count_reflects_torch(_fake_torch: Any) -> None:
    """``device_count()`` forwards to :func:`torch.cuda.device_count` when available."""
    _fake_torch.cuda.device_count.return_value = 3
    assert ROCMBackend().device_count() == 3
    _fake_torch.cuda.device_count.assert_called_once()


def test_total_memory_mb_divides_bytes(_fake_torch: Any) -> None:
    """``total_memory_mb`` converts bytes → MiB using the documented constant."""
    _fake_torch.cuda.get_device_properties.return_value = types.SimpleNamespace(
        total_memory=24 * 1024 * 1024 * 1024  # 24 GiB
    )
    backend = ROCMBackend()
    assert backend.total_memory_mb(0) == 24 * 1024  # 24 GiB == 24576 MiB


def test_free_memory_mb_divides_bytes(_fake_torch: Any) -> None:
    """``free_memory_mb`` returns the free half of the ``mem_get_info`` tuple."""
    _fake_torch.cuda.mem_get_info.return_value = (
        16 * 1024 * 1024 * 1024,  # 16 GiB free
        24 * 1024 * 1024 * 1024,  # 24 GiB total
    )
    backend = ROCMBackend()
    assert backend.free_memory_mb(0) == 16 * 1024


def test_warmup_is_noop() -> None:
    """``warmup()`` is a documented no-op and never raises."""
    assert ROCMBackend().warmup() is None
    # Calling it twice is also fine.
    assert ROCMBackend().warmup() is None


def test_module_singleton_is_rocm_backend() -> None:
    """``ROCM_BACKEND`` is a usable :class:`ROCMBackend` instance."""
    assert isinstance(ROCM_BACKEND, ROCMBackend)
    assert isinstance(ROCM_BACKEND, ROCMBackendDirect)
    assert ROCM_BACKEND.type() == BackendType.ROCM
    assert ROCM_BACKEND.vendor() == GpuVendor.AMD
    # ultralytics string works without a torch dependency.
    assert ROCM_BACKEND.to_ultralytics_string(0) == "cuda:0"


def test_satisfies_device_backend_protocol() -> None:
    """``ROCMBackend`` exposes every method of the :class:`DeviceBackend` protocol.

    Mirrors the equivalent test in ``test_cpu.py`` so a future refactor
    that accidentally drops a method fails loudly with AttributeError
    at the relevant line.
    """
    backend = ROCMBackend()
    assert callable(backend.type)
    assert callable(backend.is_available)
    assert callable(backend.device_count)
    assert callable(backend.device_name)
    assert callable(backend.total_memory_mb)
    assert callable(backend.free_memory_mb)
    assert callable(backend.vendor)
    assert callable(backend.to_ultralytics_string)
    assert callable(backend.warmup)


def test_is_available_false_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If :mod:`torch` is not importable at all, ``is_available`` returns ``False``.

    Uses :func:`pytest.MonkeyPatch` to hide the real ``torch`` from the
    import machinery during this test, simulating a host that has no
    PyTorch installed (e.g. a minimal documentation build image).
    """
    # Block the import resolution. We patch both ``sys.modules`` and
    # ``importlib.import_module`` so that any path the backend might
    # take to import ``torch`` fails.
    monkeypatch.setitem(sys.modules, "torch", None)
    # Re-import the backend to make sure no cached state matters; in
    # practice the module is already loaded but the lazy import inside
    # :meth:`is_available` is what we are testing.
    importlib.reload(importlib.import_module("img2svg.backends.rocm"))
    backend = ROCMBackend()
    assert backend.is_available() is False
