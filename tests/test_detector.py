# img2svg - tests for the YOLO detector wrapper.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the YOLO detector wrapper. Mocks the ultralytics.YOLO class."""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest

from img2svg import detector
from img2svg.errors import DeviceUnavailableError, ModelLoadError
from img2svg.models import BackendSpec, BoundingBox, Detection


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = mock.MagicMock()
        self.xyxy.cpu.return_value.numpy.return_value = xyxy
        self.conf = mock.MagicMock()
        self.conf.cpu.return_value.numpy.return_value = conf
        self.cls = mock.MagicMock()
        self.cls.cpu.return_value.numpy.return_value = cls


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class _FakeYOLO:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def __call__(self, image: np.ndarray, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append({"image_shape": image.shape, **kwargs})
        xyxy = np.array([[10.0, 20.0, 110.0, 220.0], [50.0, 60.0, 150.0, 260.0]])
        conf = np.array([0.95, 0.80])
        cls = np.array([0, 67])  # person, cell phone
        return [
            _FakeResult(
                _FakeBoxes(xyxy, conf, cls),
                {0: "person", 67: "cell phone"},
            )
        ]


class _FakeBackend:
    """Minimal duck-typed DeviceBackend for detector tests.

    Only the methods the detector actually invokes (``to_ultralytics_string``)
    are implemented. The protocol's other methods are not exercised by the
    detector, so we leave them absent and rely on Python's duck typing.
    """

    def __init__(self, to_ultralytics: str = "cpu") -> None:
        self._ultralytics = to_ultralytics

    def to_ultralytics_string(self, i: int) -> str:
        return self._ultralytics


def test_get_detector_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling get_detector() twice with same args returns the same instance."""
    detector._MODEL_CACHE.clear()
    fake_backend = _FakeBackend(to_ultralytics="cpu")
    monkeypatch.setattr(detector.REGISTRY, "resolve", lambda spec: fake_backend)
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    spec = BackendSpec(requested="cpu")
    d1 = detector.get_detector("yolo11n.pt", spec)
    d2 = detector.get_detector("yolo11n.pt", spec)
    assert d1 is d2


def test_get_detector_cache_key_normalizes_backend_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two BackendSpec instances with identical fields map to the same cache slot.

    This is the latent bug fix: the old code used the raw user string as
    part of the cache key, so ``"cuda:0"`` and ``"cuda:0 "`` (trailing
    space) ended up in different cache slots. The BackendSpec-derived
    key normalizes both to the same slot.
    """
    detector._MODEL_CACHE.clear()
    fake_backend = _FakeBackend(to_ultralytics="cpu")
    monkeypatch.setattr(detector.REGISTRY, "resolve", lambda spec: fake_backend)
    monkeypatch.setattr("ultralytics.YOLO", lambda name: _FakeYOLO("yolo11n.pt"))
    spec1 = BackendSpec(requested="cpu")
    spec2 = BackendSpec(requested="cpu")
    assert spec1 is not spec2  # distinct instances, identical content
    d1 = detector.get_detector("yolo11n.pt", spec1)
    d2 = detector.get_detector("yolo11n.pt", spec2)
    assert d1 is d2


def test_get_detector_different_index_yields_different_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BackendSpec(requested="cuda") and BackendSpec(requested="cuda", index=0) are distinct cache slots.

    The cache key includes ``backend.index`` explicitly, so a spec with
    an explicit device index is a different cache slot from one without.
    This is the documented semantics: ``"cuda"`` and ``"cuda:0"`` are
    not the same request and therefore not the same cached detector.
    """
    detector._MODEL_CACHE.clear()
    fake_backend = _FakeBackend(to_ultralytics="cuda:0")
    monkeypatch.setattr(detector.REGISTRY, "resolve", lambda spec: fake_backend)
    monkeypatch.setattr("ultralytics.YOLO", lambda name: _FakeYOLO("yolo11n.pt"))
    d1 = detector.get_detector("yolo11n.pt", BackendSpec(requested="cuda"))
    d2 = detector.get_detector("yolo11n.pt", BackendSpec(requested="cuda", index=0))
    assert d1 is not d2


def test_constructor_does_not_resolve_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a YOLODetector must NOT call REGISTRY.resolve (lazy)."""
    detector._MODEL_CACHE.clear()
    resolve_called = mock.MagicMock(return_value=_FakeBackend())
    monkeypatch.setattr(detector.REGISTRY, "resolve", resolve_called)
    detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cuda"))
    assert resolve_called.call_count == 0


def test_detect_returns_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector.REGISTRY, "resolve", lambda spec: _FakeBackend(to_ultralytics="cpu")
    )
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cpu"))
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = d.detect(img, conf=0.25, iou=0.7)
    assert len(detections) == 2
    assert isinstance(detections[0], Detection)
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.95
    assert detections[0].bbox == BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
    assert detections[1].class_name == "cell phone"
    assert detections[1].class_id == 67


def test_detect_passes_device_and_imgsz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device string passed to YOLO comes from backend.to_ultralytics_string(index)."""
    fake_backend = mock.MagicMock()
    fake_backend.to_ultralytics_string.return_value = "cuda:0"
    monkeypatch.setattr(detector.REGISTRY, "resolve", lambda spec: fake_backend)
    fake_yolo = _FakeYOLO("yolo11x.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11x.pt", BackendSpec(requested="cuda", index=0))
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    d.detect(img, imgsz=320)
    # to_ultralytics_string must be called with the spec's index (0).
    fake_backend.to_ultralytics_string.assert_called_with(0)
    assert fake_yolo.calls[0]["device"] == "cuda:0"
    assert fake_yolo.calls[0]["imgsz"] == 320
    assert fake_yolo.calls[0]["conf"] == 0.25
    assert fake_yolo.calls[0]["iou"] == 0.7
    assert fake_yolo.calls[0]["verbose"] is False


def test_detect_passes_indexed_device_for_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROCM backend's to_ultralytics_string returns 'cuda:N'; the detector passes that through unchanged.

    This is the integration point for the bug fix: a user passes
    ``BackendSpec(requested='rocm', index=2)`` and the YOLO call site
    ends up with ``device='cuda:2'`` (the string ultralytics actually
    understands) — not ``'rocm:2'`` (which ultralytics would reject).
    """
    fake_backend = mock.MagicMock()
    fake_backend.to_ultralytics_string.return_value = "cuda:2"
    monkeypatch.setattr(detector.REGISTRY, "resolve", lambda spec: fake_backend)
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="rocm", index=2))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    d.detect(img)
    fake_backend.to_ultralytics_string.assert_called_with(2)
    assert fake_yolo.calls[0]["device"] == "cuda:2"


def test_detect_handles_no_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """If YOLO returns no boxes, return empty list."""
    monkeypatch.setattr(
        detector.REGISTRY, "resolve", lambda spec: _FakeBackend(to_ultralytics="cpu")
    )
    fake_yolo = mock.MagicMock()
    fake_yolo.return_value = [_FakeResult(None, {0: "person"})]
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cpu"))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert d.detect(img) == []


def test_model_load_error_on_device_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If REGISTRY.resolve raises DeviceUnavailableError, ModelLoadError is raised on first detect()."""

    def _raise(spec: BackendSpec) -> _FakeBackend:
        raise DeviceUnavailableError("cuda", ["cpu"])

    monkeypatch.setattr(detector.REGISTRY, "resolve", _raise)
    monkeypatch.setattr("ultralytics.YOLO", lambda name: _FakeYOLO("yolo11n.pt"))
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cuda"))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ModelLoadError) as exc_info:
        d.detect(img)
    assert "yolo11n.pt" in str(exc_info.value)
    # The original DeviceUnavailableError is preserved as the cause.
    assert isinstance(exc_info.value.__cause__, DeviceUnavailableError)


def test_model_load_error_on_yolo_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If YOLO() raises (e.g., download failure), ModelLoadError is raised on first detect()."""
    monkeypatch.setattr(
        detector.REGISTRY, "resolve", lambda spec: _FakeBackend(to_ultralytics="cpu")
    )

    def _raise(name: str) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr("ultralytics.YOLO", _raise)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cpu"))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ModelLoadError) as exc_info:
        d.detect(img)
    assert "yolo11n.pt" in str(exc_info.value)
    assert "network down" in str(exc_info.value)


def test_subsequent_detect_calls_do_not_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first detect() call, REGISTRY.resolve and YOLO() are not called again."""
    detector._MODEL_CACHE.clear()
    resolve_mock = mock.MagicMock(return_value=_FakeBackend(to_ultralytics="cpu"))
    monkeypatch.setattr(detector.REGISTRY, "resolve", resolve_mock)
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cpu"))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    d.detect(img)
    d.detect(img)
    d.detect(img)
    assert resolve_mock.call_count == 1
    # YOLO() is the constructor; the model instance is built once and
    # called three times (one per detect()).
    assert len(fake_yolo.calls) == 3


def test_constructor_succeeds_when_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing a YOLODetector with an unavailable backend does not raise.

    This is the lazy-resolution contract: construction is purely
    storage; the failure surface is at first ``detect()`` call. This
    property matters for CLI commands like ``list-gpus`` / ``info`` and
    for serialization flows that need to construct a detector without
    immediately running inference.
    """
    detector._MODEL_CACHE.clear()

    def _raise(spec: BackendSpec) -> _FakeBackend:
        raise DeviceUnavailableError("cuda", ["cpu"])

    monkeypatch.setattr(detector.REGISTRY, "resolve", _raise)
    monkeypatch.setattr("ultralytics.YOLO", lambda name: _FakeYOLO("yolo11n.pt"))
    # Construction must NOT raise.
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cuda"))
    assert d._model is None
    assert d._resolved_backend is None


def test_detect_converts_rgba_input_to_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    """YOLO rejects 4-channel input; the detector must convert RGBA to RGB at the boundary.

    The loader intentionally preserves the alpha channel for downstream
    renderers (transparent SVGs), but YOLO's first conv layer expects
    3-channel RGB uint8. The detector normalizes via PIL's
    ``Image.fromarray(img).convert("RGB")`` before calling YOLO.

    Regression test: previously, passing an RGBA PNG (e.g., a logo with
    transparency) to the pipeline raised::

        RuntimeError: Given groups=1, weight of size [96, 3, 3, 3],
        expected input[1, 4, 640, 640] to have 3 channels, but got 4
        channels instead
    """
    monkeypatch.setattr(
        detector.REGISTRY, "resolve", lambda spec: _FakeBackend(to_ultralytics="cpu")
    )
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cpu"))

    # 4-channel RGBA input (e.g., a transparent PNG)
    rgba = np.zeros((480, 640, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = 128
    rgba[:, :, 3] = 255
    d.detect(rgba)

    # Verify YOLO received a 3-channel uint8 array (the conversion happened)
    assert fake_yolo.calls[0]["image_shape"] == (480, 640, 3)


def test_detect_converts_grayscale_input_to_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grayscale (H, W) input is also converted to 3-channel RGB.

    Some PNGs are saved as 1-channel grayscale (PIL mode "L"). YOLO
    needs 3 channels, so the detector must expand the channel axis
    before inference.
    """
    monkeypatch.setattr(
        detector.REGISTRY, "resolve", lambda spec: _FakeBackend(to_ultralytics="cpu")
    )
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", BackendSpec(requested="cpu"))

    # 2D grayscale input (no channel axis)
    gray = np.zeros((480, 640), dtype=np.uint8)
    d.detect(gray)

    # Verify YOLO received a 3-channel uint8 array
    assert fake_yolo.calls[0]["image_shape"] == (480, 640, 3)
