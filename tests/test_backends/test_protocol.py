# img2svg - tests for the DeviceBackend protocol and BackendType enum.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the DeviceBackend protocol and BackendType enum.

These tests exercise the structural contract only — no torch, no real
hardware, no subprocess. Concrete backend implementations get their
own test modules in later tasks.
"""

from __future__ import annotations

import enum

import pytest

from img2svg.backends import BackendType, DeviceBackend
from img2svg.backends.protocol import BackendType as BackendTypeDirect
from img2svg.enums import GpuVendor


def test_backend_type_cuda_value_is_lowercase() -> None:
    """BackendType.CUDA.value must be the lowercase string 'cuda'.

    The CLI / config layer maps user input through ``BackendType(value)``,
    so the string form is part of the public contract.
    """
    assert BackendType.CUDA.value == "cuda"


def test_backend_type_is_str_enum() -> None:
    """BackendType must be a StrEnum (str subclass) for ergonomic round-tripping."""
    assert issubclass(BackendType, str)
    assert issubclass(BackendType, enum.Enum)
    # The exact stdlib class is `enum.StrEnum` on 3.11+; on 3.10 we use
    # the shim. Both satisfy `str + Enum`, which is what the contract requires.
    assert isinstance(BackendType("cuda"), str)
    assert BackendType("cuda") is BackendType.CUDA


def test_backend_type_all_expected_members_present() -> None:
    """Every value mandated by the spec must exist on the enum."""
    members = {bt.value for bt in BackendType}
    assert members == {"auto", "cuda", "rocm", "mps", "cpu"}


def test_backend_type_re_exported_from_package() -> None:
    """The enum re-exported by img2svg.backends must be the same class as the one in protocol.py."""
    assert BackendType is BackendTypeDirect


class MockBackend:
    """Minimal concrete implementation that satisfies the DeviceBackend protocol.

    Used to verify the protocol's structural typing — the methods
    don't have to do anything useful, they just have to exist with the
    right signatures.
    """

    def type(self) -> BackendType:
        return BackendType.CPU

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def device_name(self, i: int) -> str:
        return f"MockDevice{i}"

    def total_memory_mb(self, i: int) -> int:
        return 1024

    def free_memory_mb(self, i: int) -> int:
        return 512

    def vendor(self) -> GpuVendor:
        return GpuVendor.UNKNOWN

    def to_ultralytics_string(self, i: int) -> str:
        return "cpu"

    def warmup(self) -> None:
        return None


def test_mock_backend_satisfies_protocol() -> None:
    """A plain class with the right methods must be considered a DeviceBackend."""
    backend = MockBackend()
    assert isinstance(backend, DeviceBackend)


def test_protocol_lists_all_nine_methods() -> None:
    """The protocol must declare exactly the nine methods the spec requires.

    This is a regression guard against accidentally adding or removing
    methods during refactors. The list is sorted alphabetically for
    diff stability.
    """
    expected = {
        "device_count",
        "device_name",
        "free_memory_mb",
        "is_available",
        "to_ultralytics_string",
        "total_memory_mb",
        "type",
        "vendor",
        "warmup",
    }
    actual = set(getattr(DeviceBackend, "__annotations__", {}).keys()) | {
        name for name in vars(DeviceBackend) if not name.startswith("_")
    }
    assert expected.issubset(actual), f"missing methods: {expected - actual}"


def test_object_lacking_methods_does_not_satisfy_protocol() -> None:
    """A bare object missing required methods must not be a DeviceBackend.

    Guards against a protocol that is so loose that everything matches.
    """

    class Incomplete:
        def type(self) -> BackendType:
            return BackendType.CPU

    assert not isinstance(Incomplete(), DeviceBackend)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", BackendType.AUTO),
        ("cuda", BackendType.CUDA),
        ("rocm", BackendType.ROCM),
        ("mps", BackendType.MPS),
        ("cpu", BackendType.CPU),
    ],
)
def test_backend_type_round_trips_via_value(value: str, expected: BackendType) -> None:
    """Constructing from the string value yields the matching member (round-trip)."""
    assert BackendType(value) is expected
