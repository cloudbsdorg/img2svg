# img2svg - tests for the BackendRegistry.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`img2svg.backends.registry.BackendRegistry`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from img2svg.backends import (
    CPU_BACKEND,
    CUDA_BACKEND,
    MPS_BACKEND,
    REGISTRY,
    ROCM_BACKEND,
    BackendRegistry,
)
from img2svg.backends.protocol import BackendType, DeviceBackend
from img2svg.errors import DeviceUnavailableError
from img2svg.models import BackendSpec


def test_available_returns_priority_ordered_filtered_list() -> None:
    """available() returns backends in priority order, filtered by is_available()."""
    avail = REGISTRY.available()
    # The result must preserve the priority order CUDA > ROCM > MPS > CPU
    # and exclude any backend that reports is_available() == False.
    expected = [b for b in REGISTRY._all if b.is_available()]
    assert avail == expected
    # On this Linux+CUDA host, ROCM and MPS are filtered out.
    assert ROCM_BACKEND not in avail
    assert MPS_BACKEND not in avail
    assert CUDA_BACKEND in avail
    assert CPU_BACKEND in avail


def test_detect_returns_cuda_on_this_system() -> None:
    """detect() returns CUDA_BACKEND on this Linux+CUDA workstation."""
    # This test is host-specific; on a macOS box it would return
    # MPS_BACKEND, on a CPU-only host it would return CPU_BACKEND.
    # The plan's hands-on verification is on this Linux+CUDA host.
    detected = REGISTRY.detect()
    assert detected is CUDA_BACKEND
    assert detected.type() == BackendType.CUDA


def test_detect_falls_back_to_cpu_when_nothing_available() -> None:
    """detect() returns CPU_BACKEND when every GPU backend is unavailable."""
    # Construct a fresh registry whose only available backend is CPU.
    fake_cpu = MagicMock(spec=DeviceBackend)
    fake_cpu.type.return_value = BackendType.CPU
    fake_cpu.is_available.return_value = True
    fake_unavail = MagicMock(spec=DeviceBackend)
    fake_unavail.type.return_value = BackendType.CUDA
    fake_unavail.is_available.return_value = False
    reg = BackendRegistry()
    # Patch the instance attributes directly so we don't have to
    # touch the module-level singleton.
    with (
        patch.object(reg, "_all", [fake_unavail, fake_cpu]),
        patch.object(reg, "_by_type", {BackendType.CUDA: fake_unavail, BackendType.CPU: fake_cpu}),
    ):
        # Only CPU is available; detect() must pick it.
        assert reg.detect() is fake_cpu


def test_resolve_cpu_returns_cpu_backend() -> None:
    """resolve(BackendSpec(requested='cpu')) returns CPU_BACKEND."""
    spec = BackendSpec(requested="cpu")
    assert REGISTRY.resolve(spec) is CPU_BACKEND


def test_resolve_auto_returns_detect_result() -> None:
    """resolve(BackendSpec(requested='auto')) delegates to detect()."""
    spec = BackendSpec(requested="auto")
    assert REGISTRY.resolve(spec) is REGISTRY.detect()
    # On this host that's CUDA_BACKEND.
    assert REGISTRY.resolve(spec) is CUDA_BACKEND


def test_resolve_bogus_raises_device_unavailable() -> None:
    """resolve() with an unknown backend name raises DeviceUnavailableError.

    Uses ``model_construct`` to bypass the ``Literal`` validator in
    :class:`BackendSpec`; this is the same path :meth:`for_device_string`
    takes for unknown legacy strings.
    """
    spec = BackendSpec.model_construct(requested="bogus")
    with pytest.raises(DeviceUnavailableError) as excinfo:
        REGISTRY.resolve(spec)
    assert excinfo.value.requested == "bogus"
    # The error message must list what IS available.
    available = excinfo.value.available
    assert "cpu" in available
    # On this host, cuda is also available.
    assert "cuda" in available


def test_resolve_mps_unavailable_on_linux() -> None:
    """MPS is not available on Linux, so resolve('mps') raises."""
    if MPS_BACKEND.is_available():
        pytest.skip("MPS is available on this host (macOS?)")
    with pytest.raises(DeviceUnavailableError) as excinfo:
        REGISTRY.resolve(BackendSpec(requested="mps"))
    assert excinfo.value.requested == "mps"


def test_resolve_unavailable_cuda_via_mock() -> None:
    """When CUDA is mocked as unavailable, resolve('cuda') raises."""
    fake_cuda = MagicMock(spec=DeviceBackend)
    fake_cuda.type.return_value = BackendType.CUDA
    fake_cuda.is_available.return_value = False
    reg = BackendRegistry()
    with (
        patch.object(reg, "_all", [fake_cuda, CPU_BACKEND]),
        patch.object(
            reg,
            "_by_type",
            {BackendType.CUDA: fake_cuda, BackendType.CPU: CPU_BACKEND},
        ),
    ):
        with pytest.raises(DeviceUnavailableError) as excinfo:
            reg.resolve(BackendSpec(requested="cuda"))
        assert excinfo.value.requested == "cuda"
        # CPU is still available.
        assert "cpu" in excinfo.value.available


def test_resolve_with_mocked_mps_available() -> None:
    """When MPS is mocked as available, detect() and resolve() pick it up correctly."""
    fake_cuda = MagicMock(spec=DeviceBackend)
    fake_cuda.type.return_value = BackendType.CUDA
    fake_cuda.is_available.return_value = True
    fake_mps = MagicMock(spec=DeviceBackend)
    fake_mps.type.return_value = BackendType.MPS
    fake_mps.is_available.return_value = True
    reg = BackendRegistry()
    # Put CUDA first (priority order: CUDA > MPS > CPU)
    with (
        patch.object(reg, "_all", [fake_cuda, fake_mps, CPU_BACKEND]),
        patch.object(
            reg,
            "_by_type",
            {
                BackendType.CUDA: fake_cuda,
                BackendType.MPS: fake_mps,
                BackendType.CPU: CPU_BACKEND,
            },
        ),
    ):
        # detect() still prefers CUDA (higher priority).
        assert reg.detect() is fake_cuda
        # resolve('mps') goes to the fake MPS backend.
        assert reg.resolve(BackendSpec(requested="mps")) is fake_mps


def test_for_device_string_cuda_indexed() -> None:
    """for_device_string('cuda:0') returns BackendSpec(requested='cuda', index=0)."""
    spec = REGISTRY.for_device_string("cuda:0")
    assert spec.requested == "cuda"
    assert spec.index == 0


def test_for_device_string_rocm_indexed() -> None:
    """for_device_string('rocm:2') returns BackendSpec(requested='rocm', index=2)."""
    spec = REGISTRY.for_device_string("rocm:2")
    assert spec.requested == "rocm"
    assert spec.index == 2


def test_for_device_string_auto() -> None:
    """for_device_string('auto') returns BackendSpec(requested='auto')."""
    spec = REGISTRY.for_device_string("auto")
    assert spec.requested == "auto"
    assert spec.index is None


def test_for_device_string_empty_is_auto() -> None:
    """for_device_string('') returns BackendSpec(requested='auto')."""
    spec = REGISTRY.for_device_string("")
    assert spec.requested == "auto"


def test_for_device_string_bogus_defers_to_resolve() -> None:
    """for_device_string('bogus') returns a BackendSpec that fails at resolve().

    The deferred-validation design lets :meth:`resolve` produce a
    :class:`DeviceUnavailableError` listing the actually-available
    backends, instead of a Pydantic ``ValidationError`` that names
    the offending literal.
    """
    spec = REGISTRY.for_device_string("bogus")
    # The spec is constructed (no ValidationError at parse time).
    assert spec.requested == "bogus"
    # And it fails at resolve() with the helpful error shape.
    with pytest.raises(DeviceUnavailableError) as excinfo:
        REGISTRY.resolve(spec)
    assert excinfo.value.requested == "bogus"
    assert "cuda" in excinfo.value.available
    assert "cpu" in excinfo.value.available


def test_for_device_string_known_backends() -> None:
    """for_device_string handles every known backend name (cpu, cuda, rocm, mps)."""
    for name, expected_type in (
        ("cpu", BackendType.CPU),
        ("cuda", BackendType.CUDA),
        ("rocm", BackendType.ROCM),
        ("mps", BackendType.MPS),
    ):
        spec = REGISTRY.for_device_string(name)
        assert spec.requested == expected_type.value
        assert spec.index is None


def test_for_device_string_normalises_case_and_whitespace() -> None:
    """for_device_string is case-insensitive and strips leading/trailing whitespace."""
    spec1 = REGISTRY.for_device_string("  CUDA  ")
    spec2 = REGISTRY.for_device_string("cuda")
    assert spec1.requested == spec2.requested == "cuda"


def test_registry_singleton_is_backend_registry() -> None:
    """REGISTRY is an instance of BackendRegistry with the expected backends."""
    assert isinstance(REGISTRY, BackendRegistry)
    # Priority order is preserved.
    assert REGISTRY._all[0] is CUDA_BACKEND
    assert REGISTRY._all[1] is ROCM_BACKEND
    assert REGISTRY._all[2] is MPS_BACKEND
    assert REGISTRY._all[3] is CPU_BACKEND
