"""Tests for the YOLO detector wrapper. Mocks the ultralytics.YOLO class."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from img2svg import detector
from img2svg.errors import ModelLoadError
from img2svg.models import BoundingBox, Detection


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


def test_get_detector_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling get_detector() twice with same args returns the same instance."""
    detector._MODEL_CACHE.clear()
    monkeypatch.setattr("img2svg.detector.device.detect_device", lambda x: "cpu")
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d1 = detector.get_detector("yolo11n.pt", "cpu")
    d2 = detector.get_detector("yolo11n.pt", "cpu")
    assert d1 is d2


def test_detect_returns_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("img2svg.detector.device.detect_device", lambda x: "cpu")
    fake_yolo = _FakeYOLO("yolo11n.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", "cpu")
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
    monkeypatch.setattr("img2svg.detector.device.detect_device", lambda x: "cuda:0")
    fake_yolo = _FakeYOLO("yolo11x.pt")
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11x.pt", "cuda:0")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    d.detect(img, imgsz=320)
    assert fake_yolo.calls[0]["device"] == "cuda:0"
    assert fake_yolo.calls[0]["imgsz"] == 320
    assert fake_yolo.calls[0]["conf"] == 0.25
    assert fake_yolo.calls[0]["iou"] == 0.7
    assert fake_yolo.calls[0]["verbose"] is False


def test_detect_handles_no_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """If YOLO returns no boxes, return empty list."""
    monkeypatch.setattr("img2svg.detector.device.detect_device", lambda x: "cpu")
    fake_yolo = mock.MagicMock()
    fake_yolo.return_value = [_FakeResult(None, {0: "person"})]
    monkeypatch.setattr("ultralytics.YOLO", lambda name: fake_yolo)
    d = detector.YOLODetector("yolo11n.pt", "cpu")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert d.detect(img) == []


def test_model_load_error_on_device_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If device.detect_device raises, ModelLoadError is raised."""
    from img2svg.errors import DeviceUnavailableError
    def _raise(x: str) -> str:
        raise DeviceUnavailableError("cuda", ["cpu"])
    monkeypatch.setattr("img2svg.detector.device.detect_device", _raise)
    monkeypatch.setattr("ultralytics.YOLO", lambda name: _FakeYOLO("yolo11n.pt"))
    with pytest.raises(ModelLoadError) as exc_info:
        detector.YOLODetector("yolo11n.pt", "cuda")
    assert "yolo11n.pt" in str(exc_info.value)


def test_model_load_error_on_yolo_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If YOLO() raises (e.g., download failure), ModelLoadError is raised."""
    monkeypatch.setattr("img2svg.detector.device.detect_device", lambda x: "cpu")
    def _raise(name: str) -> None:
        raise RuntimeError("network down")
    monkeypatch.setattr("ultralytics.YOLO", _raise)
    with pytest.raises(ModelLoadError) as exc_info:
        detector.YOLODetector("yolo11n.pt", "cpu")
    assert "yolo11n.pt" in str(exc_info.value)
    assert "network down" in str(exc_info.value)
