# img2svg - YOLO object detection wrapper for img2svg.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""YOLO object detection wrapper for img2svg.

Wraps `ultralytics.YOLO` with a clean dataclass interface. Singleton factory
to avoid re-loading the (heavy) YOLO model on every detection call.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from img2svg import device, paths
from img2svg.errors import ModelLoadError
from img2svg.models import BoundingBox, Detection

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, YOLODetector] = {}


def get_detector(model_name: str = "yolo11x.pt", device_str: str = "auto") -> YOLODetector:
    """Get or create a cached YOLODetector instance."""
    key = f"{model_name}::{device_str}"
    if key not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = YOLODetector(model_name=model_name, device_str=device_str)
    return _MODEL_CACHE[key]


def _ensure_model_downloaded(model_name: str) -> Path:
    """Make sure the model is in the XDG cache dir. ultralytics auto-downloads on first YOLO() call.

    We do NOT proactively download — we let ultralytics handle it on YOLO()
    instantiation. This function just returns the expected cache path.
    """
    return paths.model_cache_path(model_name)


class YOLODetector:
    """YOLO-based object detector with device dispatch.

    The underlying `ultralytics.YOLO` is heavy to load (model weights, CUDA
    initialization). Use `get_detector()` to get a cached singleton rather
    than instantiating directly in hot paths.
    """

    def __init__(self, model_name: str = "yolo11x.pt", device_str: str = "auto") -> None:
        self.model_name = model_name
        self._requested_device = device_str
        try:
            self.device = device.detect_device(device_str)
        except Exception as e:
            raise ModelLoadError(model_name, original=e) from e
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            self._model = YOLO(model_name)
        except Exception as e:
            raise ModelLoadError(model_name, original=e) from e
        # Trigger download/cache
        _ensure_model_downloaded(model_name)

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
    ) -> list[Detection]:
        """Run YOLO inference. Returns a list of Detection objects."""
        results = self._model(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        for (x1, y1, x2, y2), confidence, cls_id in zip(xyxy, confs, clss):
            class_name = names.get(int(cls_id), str(cls_id))
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=str(class_name),
                    confidence=float(confidence),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                )
            )
        return detections
