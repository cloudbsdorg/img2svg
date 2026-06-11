# img2svg - tests for device detection.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for device detection."""

from __future__ import annotations

from unittest import mock

import pytest

from img2svg import device
from img2svg.errors import DeviceUnavailableError


def test_is_available_cpu() -> None:
    assert device.is_available("cpu") is True


def test_is_available_unknown_device() -> None:
    assert device.is_available("bogus") is False


def test_is_available_cuda_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """If torch reports no CUDA, is_available('cuda') must be False."""
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setattr(device, "torch", fake_torch)
    assert device.is_available("cuda") is False
    assert device.is_available("cuda:0") is False


def test_is_available_cuda_when_torch_says_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    monkeypatch.setattr(device, "torch", fake_torch)
    assert device.is_available("cuda") is True
    assert device.is_available("cuda:0") is True
    assert device.is_available("cuda:1") is True
    assert device.is_available("cuda:5") is False


def test_detect_device_auto_picks_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setattr(device, "torch", fake_torch)
    assert device.detect_device("auto") == "cuda:0"


def test_detect_device_auto_picks_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    monkeypatch.setattr(device, "torch", fake_torch)
    assert device.detect_device("auto") == "mps"


def test_detect_device_auto_picks_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setattr(device, "torch", fake_torch)
    assert device.detect_device("auto") == "cpu"


def test_detect_device_explicit_cpu() -> None:
    assert device.detect_device("cpu") == "cpu"


def test_detect_device_explicit_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setattr(device, "torch", fake_torch)
    with pytest.raises(DeviceUnavailableError) as exc_info:
        device.detect_device("cuda")
    assert "cuda" in str(exc_info.value)
    assert "auto" in exc_info.value.user_message()


def test_list_available_devices_cpu_always_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setattr(device, "torch", fake_torch)
    devs = device.list_available_devices()
    assert "cpu" in devs


def test_list_available_devices_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setattr(device, "torch", fake_torch)
    devs = device.list_available_devices()
    assert "cuda:0" in devs
    assert "cuda:1" in devs
    assert "cpu" in devs
