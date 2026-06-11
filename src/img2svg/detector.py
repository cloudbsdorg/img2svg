# img2svg - YOLO object detection wrapper for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""YOLO object detection wrapper for img2svg.

Wraps `ultralytics.YOLO` with a clean dataclass interface. Singleton factory
to avoid re-loading the (heavy) YOLO model on every detection call.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from img2svg import paths
from img2svg.backends.protocol import DeviceBackend
from img2svg.backends.registry import REGISTRY
from img2svg.errors import DeviceUnavailableError, ModelLoadError
from img2svg.models import BackendSpec, BoundingBox, Detection

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ultralytics import YOLO  # type: ignore[attr-defined]

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, YOLODetector] = {}


def get_detector(
    model_name: str = "yolo11x.pt", backend: BackendSpec | None = None
) -> YOLODetector:
    """Get or create a cached YOLODetector instance.

    The cache key is a stable repr of ``(model_name, backend.requested,
    backend.index)`` so two ``BackendSpec`` instances that describe the
    same logical backend collide on the same cache slot. ``backend=None``
    is normalized to ``BackendSpec()`` (i.e. ``requested="auto"``).

    The earlier implementation used the raw user string as part of the
    key, which meant ``get_detector("yolo11x.pt", "cuda:0")`` and
    ``get_detector("yolo11x.pt", "cuda:0 ")`` (trailing space) ended up
    as two different cached detectors. The BackendSpec-derived key
    normalizes both to the same slot.
    """
    if backend is None:
        backend = BackendSpec()
    key = f"{model_name}::{backend.requested}::{backend.index}"
    if key not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = YOLODetector(model_name=model_name, backend=backend)
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

    The backend is resolved lazily on the first :meth:`detect` call so
    that constructing a detector never fails on a host that lacks the
    requested accelerator. This is important for the CLI's
    ``list-gpus`` / ``info`` paths and for serialization: a user on
    CPU-only hardware can still construct a detector with
    ``backend=BackendSpec(requested="cuda")`` without an immediate
    :class:`ModelLoadError`. The error surfaces on first inference.
    """

    def __init__(
        self, model_name: str = "yolo11x.pt", backend: BackendSpec | None = None
    ) -> None:
        """Store the backend spec; the actual backend is resolved on first .detect() call (lazy)."""
        if backend is None:
            backend = BackendSpec()
        self._model_name = model_name
        self._backend_spec = backend
        self._model: YOLO | None = None
        self._resolved_backend: DeviceBackend | None = None
        # Populated by _ensure_loaded() before the first detect() call.
        # Empty string is a safe sentinel: ultralytics would reject it,
        # but we never call YOLO() without first going through
        # _ensure_loaded().
        self.device: str = ""

    def _ensure_loaded(self) -> None:
        """Resolve the backend and load the YOLO model on first call.

        Subsequent calls are no-ops because :attr:`_model` is set.
        Exceptions from :meth:`REGISTRY.resolve` (e.g. the requested
        backend is not available on this host) are wrapped in
        :class:`ModelLoadError` to preserve the historical failure
        surface — callers (and the test suite) only know to catch
        :class:`ModelLoadError` for detector construction failures.
        """
        if self._model is not None:
            return
        try:
            self._resolved_backend = REGISTRY.resolve(self._backend_spec)
        except DeviceUnavailableError as e:
            raise ModelLoadError(self._model_name, original=e) from e
        # ``backend.index`` is None for plain ``requested="cuda"`` /
        # ``"rocm"`` / ``"auto"``; default to device 0 in that case.
        # ``to_ultralytics_string`` is responsible for raising
        # ``IndexError`` on an out-of-range index; the registry does
        # not pre-validate the index.
        self.device = self._resolved_backend.to_ultralytics_string(
            self._backend_spec.index or 0
        )
        _ensure_model_downloaded(self._model_name)
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._model = YOLO(self._model_name)
        except Exception as e:
            raise ModelLoadError(self._model_name, original=e) from e

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
    ) -> list[Detection]:
        """Run YOLO inference. Returns a list of Detection objects."""
        self._ensure_loaded()
        assert self._model is not None  # invariant: _ensure_loaded sets this
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
