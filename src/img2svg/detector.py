# img2svg - YOLO object detection and segmentation wrappers for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""YOLO object detection and segmentation wrappers for img2svg.

Wraps `ultralytics.YOLO` with clean dataclass interfaces. Singleton
factories to avoid re-loading the (heavy) YOLO models on every call.

Two parallel wrappers live in this module:

* :class:`YOLODetector` — the detection-only model (``yolo11x.pt``).
* :class:`YOLOSegmentor` — the instance segmentation model
  (``yolo11s-seg.pt``) which returns boxes, masks, and polygons.

The two model families are NOT interchangeable, so each has its own
cache slot (:data:`_MODEL_CACHE` and :data:`_SEGMENTOR_CACHE`).
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from lxml import etree
from PIL import Image

from img2svg import paths
from img2svg.backends.protocol import BackendType, DeviceBackend
from img2svg.backends.registry import REGISTRY
from img2svg.errors import DeviceUnavailableError, ModelLoadError
from img2svg.logging import get_logger
from img2svg.models import BackendSpec, BoundingBox, Detection
from img2svg.svg_builder import SVG_NS
from img2svg.vectorizer import VtracerVectorizer

_log = get_logger("img2svg.detector")

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ultralytics import YOLO  # type: ignore[attr-defined]

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, YOLODetector] = {}

# The segmentor uses a separate cache because its model weights
# (``yolo11s-seg``) are distinct from the detector's (``yolo11x``).
# Caching both in the same dict would let a 4 GB GPU try to keep two
# large models in VRAM at once, which is a known OOM scenario.
_SEGMENTOR_LOCK = threading.Lock()
_SEGMENTOR_CACHE: dict[str, YOLOSegmentor] = {}


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


def get_segmentor(
    model_name: str = "yolo11s-seg.pt", backend: BackendSpec | None = None
) -> YOLOSegmentor:
    """Get or create a cached YOLOSegmentor instance.

    Parallel to :func:`get_detector` but with a SEPARATE cache dict
    (:data:`_SEGMENTOR_CACHE`) so detection and segmentation models
    don't compete for the same cache slot. Caching both into a single
    dict would let a 4 GB GPU hold two large models in VRAM at once,
    which OOMs in practice.

    The cache key is the same shape as :func:`get_detector`'s —
    ``f"{model_name}::{backend.requested}::{backend.index}"`` — so the
    normalization rules (trailing whitespace, ``None`` → ``"auto"``)
    behave identically.
    """
    if backend is None:
        backend = BackendSpec()
    key = f"{model_name}::{backend.requested}::{backend.index}"
    if key not in _SEGMENTOR_CACHE:
        with _SEGMENTOR_LOCK:
            if key not in _SEGMENTOR_CACHE:
                _SEGMENTOR_CACHE[key] = YOLOSegmentor(model_name=model_name, backend=backend)
    return _SEGMENTOR_CACHE[key]


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

    def __init__(self, model_name: str = "yolo11x.pt", backend: BackendSpec | None = None) -> None:
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
        self.device = self._resolved_backend.to_ultralytics_string(self._backend_spec.index or 0)
        _ensure_model_downloaded(self._model_name)
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._model = YOLO(self._model_name)
        except Exception as e:
            raise ModelLoadError(self._model_name, original=e) from e

    def detect(
        self,
        image: np.ndarray[Any, np.dtype[np.uint8]],
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
    ) -> list[Detection]:
        """Run YOLO inference. Returns a list of Detection objects."""
        self._ensure_loaded()
        assert self._model is not None  # invariant: _ensure_loaded sets this
        # YOLO's first conv layer expects 3-channel RGB uint8 input. The
        # loader intentionally preserves the alpha channel (RGBA) so
        # downstream renderers can produce transparent SVGs, but YOLO
        # cannot process the alpha channel — normalize at the model
        # boundary. PIL's ``convert("RGB")`` is idempotent on already-RGB
        # input and correctly handles 1ch (grayscale), 2ch (LA), 4ch
        # (RGBA), 16-bit, and float modes. PIL is a transitive dep of
        # ultralytics, so importing it here is free.
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            from PIL import Image  # local import: keep import-time deps lean

            image = np.asarray(Image.fromarray(image).convert("RGB"))
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
        for (x1, y1, x2, y2), confidence, cls_id in zip(xyxy, confs, clss, strict=False):
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


@dataclass
class SegmentationResult:
    """Result of a single YOLO segmentation inference call.

    Parallel to the plain :class:`list[Detection]` return shape of
    :meth:`YOLODetector.detect` but bundles the extra artifacts the
    segmentation model produces: per-instance binary masks and
    pixel-coordinate polygons. One entry per detection, in matching
    list order, so ``boxes[i]``, ``masks[i]``, and ``polygons[i]``
    describe the same object.

    The dataclass is deliberately NOT a Pydantic model: this is an
    internal data container passed between :class:`YOLOSegmentor` and
    the per-region vtracer renderer, and we don't need validation /
    serialization on the hot path. Pydantic's :class:`Detection` is
    reused for the ``boxes`` field so the sidecar's existing
    serialization keeps working.
    """

    boxes: list[Detection]
    """One :class:`Detection` per instance, in the order YOLO returns
    them. Same shape the plain :class:`YOLODetector` produces."""

    masks: list[np.ndarray[Any, np.dtype[np.uint8]]]
    """One ``(H, W)`` ``uint8`` binary mask per instance. ``0`` is
    background, ``1`` is foreground. ``H`` and ``W`` match
    :attr:`orig_shape` because :meth:`YOLOSegmentor.predict` always
    sets ``retina_masks=True``."""

    polygons: list[np.ndarray[Any, np.dtype[np.float32]]]
    """One ``(P, 2)`` ``float32`` pixel polygon per instance, sourced
    from :attr:`ultralytics.engine.results.Masks.xy`. Empty masks
    yield a ``(0, 2)`` array rather than ``None`` so callers can
    iterate uniformly without None checks."""

    orig_shape: tuple[int, int]
    """``(H, W)`` of the original input image, captured at predict
    time so downstream code can sanity-check mask dimensions after
    the intermediate YOLO tensor is garbage-collected."""


class YOLOSegmentor:
    """YOLO-based instance segmentor with device dispatch.

    Sister class to :class:`YOLODetector` for the segmentation model
    variant (``yolo11s-seg.pt``, ``yolo11m-seg.pt``, ...). The model
    weights are different and NOT interchangeable with the detection
    model — loading a seg weights file into a detection model
    produces garbage at runtime, and vice versa. The segmentor has
    its own cache slot in :func:`get_segmentor` precisely to keep the
    two model families from colliding.

    The backend is resolved lazily on the first :meth:`predict` call,
    mirroring :class:`YOLODetector`. Constructing a segmentor never
    fails on a host that lacks the requested accelerator; the
    :class:`ModelLoadError` surfaces on first inference.
    """

    def __init__(
        self, model_name: str = "yolo11s-seg.pt", backend: BackendSpec | None = None
    ) -> None:
        """Store the backend spec; the actual backend is resolved on first .predict() call (lazy)."""
        if backend is None:
            backend = BackendSpec()
        self._model_name = model_name
        self._backend_spec = backend
        self._model: YOLO | None = None
        self._resolved_backend: DeviceBackend | None = None
        # Populated by _ensure_loaded() before the first predict() call.
        # Empty string is a safe sentinel: ultralytics would reject it,
        # but we never call YOLO() without first going through
        # _ensure_loaded().
        self.device: str = ""

    def _ensure_loaded(self) -> None:
        """Resolve the backend and load the YOLO segmentation model on first call.

        Mirrors :meth:`YOLODetector._ensure_loaded`. Subsequent calls
        are no-ops because :attr:`_model` is set. Exceptions from
        :meth:`REGISTRY.resolve` (e.g. the requested backend is not
        available on this host) are wrapped in :class:`ModelLoadError`
        to preserve the historical failure surface — callers only know
        to catch :class:`ModelLoadError` for detector/segmentor
        construction failures.
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
        self.device = self._resolved_backend.to_ultralytics_string(self._backend_spec.index or 0)
        _ensure_model_downloaded(self._model_name)
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._model = YOLO(self._model_name)
        except Exception as e:
            raise ModelLoadError(self._model_name, original=e) from e

    def _is_cuda(self) -> bool:
        """Return True when the resolved backend is NVIDIA CUDA.

        Used to decide whether to pass ``half=True`` (FP16) to the
        YOLO model. FP16 is enabled for CUDA only per the segmentor
        spec; ROCm users get FP32 because their backend type is
        distinct (and the conservative default avoids surprises on
        rare AMD drivers). CPU and MPS are not eligible for FP16 on
        the YOLO inference path.
        """
        backend = self._resolved_backend
        if backend is None:
            return False
        return backend.type() == BackendType.CUDA

    def predict(
        self,
        image: np.ndarray[Any, np.dtype[np.uint8]],
        conf: float = 0.25,
        iou: float = 0.6,
        imgsz: int = 1024,
    ) -> SegmentationResult:
        """Run YOLO segmentation inference. Returns boxes, masks, and polygons.

        Parameters
        ----------
        image : np.ndarray
            Input image as a ``(H, W, 3)`` ``uint8`` RGB array. RGBA /
            grayscale / non-uint8 input is normalized to 3-channel RGB
            uint8 at the model boundary (mirrors :meth:`YOLODetector.detect`).
        conf : float
            Confidence threshold. Default ``0.25``.
        iou : float
            IoU threshold for NMS. Default ``0.6`` (slightly more
            permissive than the detector's ``0.7`` to keep overlapping
            mask instances).
        imgsz : int
            Inference image size (longer edge). Default ``1024`` for
            higher mask fidelity on the segmentation model.
        """
        self._ensure_loaded()
        assert self._model is not None  # invariant: _ensure_loaded sets this
        # Capture the original image dimensions BEFORE the RGBA → RGB
        # normalization, because masks come back at the input image's
        # (H, W) when retina_masks=True. Recording the post-normalization
        # shape would also be correct (RGBA → RGB preserves H, W), but
        # the convention is to record what the caller passed in.
        orig_shape = (int(image.shape[0]), int(image.shape[1]))
        # YOLO's first conv layer expects 3-channel RGB uint8 input. The
        # loader intentionally preserves the alpha channel (RGBA) so
        # downstream renderers can produce transparent SVGs, but YOLO
        # cannot process the alpha channel — normalize at the model
        # boundary. PIL's ``convert("RGB")`` is idempotent on already-RGB
        # input and correctly handles 1ch (grayscale), 2ch (LA), 4ch
        # (RGBA), 16-bit, and float modes. PIL is a transitive dep of
        # ultralytics, so importing it here is free.
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            from PIL import Image  # local import: keep import-time deps lean

            image = np.asarray(Image.fromarray(image).convert("RGB"))
        results = self._model(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            retina_masks=True,
            half=self._is_cuda(),
            verbose=False,
        )
        # YOLO returns an empty list when nothing is detected at the
        # current conf threshold. Return a consistent empty
        # SegmentationResult (all three lists empty, orig_shape still
        # recorded) so downstream callers can iterate without
        # type-checking.
        if not results:
            return SegmentationResult(boxes=[], masks=[], polygons=[], orig_shape=orig_shape)
        result = results[0]
        if result.boxes is None or result.masks is None:
            return SegmentationResult(boxes=[], masks=[], polygons=[], orig_shape=orig_shape)
        # Build Detection list from box tensors (mirrors YOLODetector).
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        boxes: list[Detection] = []
        for (x1, y1, x2, y2), confidence, cls_id in zip(xyxy, confs, clss, strict=False):
            class_name = names.get(int(cls_id), str(cls_id))
            boxes.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=str(class_name),
                    confidence=float(confidence),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                )
            )
        # ``result.masks.data`` is a ``torch.uint8`` tensor of shape
        # ``(N, H, W)`` with ``retina_masks=True``. We move to CPU and
        # split into a list of ``(H, W)`` ``uint8`` ndarrays (one per
        # detection) so callers can consume them as numpy arrays
        # without the ``.data[0]``, ``.data[1]`` dance. The values are
        # already binary (0/1) — ultralytics applies ``.gt_(0.0)`` /
        # ``.byte()`` internally when ``retina_masks=True``, so we do
        # NOT add another sigmoid + 0.5 here.
        mask_arrays: list[np.ndarray[Any, np.dtype[np.uint8]]] = [
            m.cpu().numpy() for m in result.masks.data
        ]
        # ``result.masks.xy`` is already a list of ``(P, 2)`` numpy
        # arrays in pixel coordinates (not normalized). Cast to
        # ``float32`` explicitly for a stable downstream dtype
        # (ultralytics may return ``float64`` on some versions). Empty
        # arrays are normalized to ``(0, 2)`` so callers can iterate
        # uniformly without a ``None`` check.
        polygons: list[np.ndarray[Any, np.dtype[np.float32]]] = [
            np.asarray(p, dtype=np.float32) if p.size else np.zeros((0, 2), dtype=np.float32)
            for p in result.masks.xy
        ]
        return SegmentationResult(
            boxes=boxes, masks=mask_arrays, polygons=polygons, orig_shape=orig_shape
        )


def extract_polygons(
    masks_data: list[np.ndarray[Any, np.dtype[np.uint8]]],
) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
    """Extract the largest external contour from each binary mask as (P, 2) float32.

    Empty masks yield (0, 2) zeros. RETR_EXTERNAL drops interior holes (YOLO convention).
    """
    polygons: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for mask in masks_data:
        if mask.size == 0 or not (mask > 0).any():
            polygons.append(np.zeros((0, 2), dtype=np.float32))
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            polygons.append(np.zeros((0, 2), dtype=np.float32))
            continue
        largest = max(contours, key=cv2.contourArea)
        polygons.append(largest.reshape(-1, 2).astype(np.float32))
    return polygons


def compute_mask_area(mask: np.ndarray[Any, np.dtype[np.uint8]]) -> int:
    """Return the number of nonzero pixels in a binary mask."""
    return int((mask > 0).sum())


def get_tight_bbox(mask: np.ndarray[Any, np.dtype[np.uint8]]) -> tuple[int, int, int, int]:
    """Return the (x1, y1, x2, y2) tight bounding box of nonzero mask pixels.

    Empty masks yield (0, 0, 0, 0).
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def trace_region(
    image: np.ndarray[Any, np.dtype[np.uint8]],
    mask: np.ndarray[Any, np.dtype[np.uint8]],
    preset: str = "photo_hifi",
    vtracer_params_override: dict[str, object] | None = None,
) -> tuple[list[str], tuple[int, int]]:
    """Trace a single image region with vtracer.

    Crops `image` and `mask` to the tight bounding box of the mask's
    non-zero pixels, builds a 4-channel RGBA array (alpha = mask), runs
    vtracer with `preset`, and returns the extracted ``<path d="...">``
    strings plus the ``(x1, y1)`` offset needed to position the paths
    inside the original image.

    Returns ``([], (0, 0))`` when `mask` has no non-zero pixels.

    Parameters
    ----------
    image : np.ndarray
        H x W x 3 uint8 RGB image of the full source.
    mask : np.ndarray
        H x W uint8 binary mask; non-zero pixels are "in" the region.
    preset : str
        One of the keys in ``img2svg.vectorizer.PRESETS``.
    vtracer_params_override : dict[str, object] | None
        Optional kwargs forwarded to :class:`VtracerVectorizer` to
        override the preset's defaults (used by the pipeline to apply
        ``--max-colors`` to per-region traces).

    Returns
    -------
    (list[str], (int, int))
        List of vtracer ``<path d="...">`` strings (possibly empty) and
        the ``(x1, y1)`` top-left offset of the cropped region in
        `image`.
    """
    if not (mask > 0).any():
        return ([], (0, 0))
    # ultralytics sometimes returns masks as (H, W, 1) instead of (H, W).
    if mask.ndim == 3:
        mask = mask[..., 0]
    x1, y1, x2, y2 = get_tight_bbox(mask)
    # ``get_tight_bbox`` returns inclusive max indices; add 1 for
    # half-open numpy slicing so the rightmost/bottom mask pixel is
    # included in the crop.
    crop_img = image[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    # vtracer needs 3-channel RGB; coerce before dstack. Source images
    # may be RGBA (drop alpha) or 1-channel grayscale (expand to RGB).
    if crop_img.ndim == 2:
        crop_img = np.repeat(crop_img[..., None], 3, axis=-1)
    elif crop_img.shape[-1] == 1:
        crop_img = np.repeat(crop_img, 3, axis=-1)
    elif crop_img.shape[-1] == 4:
        crop_img = crop_img[..., :3]
    rgba = np.dstack([crop_img, crop_mask])

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_png = td_path / "input.png"
        output_svg = td_path / "output.svg"
        Image.fromarray(rgba).save(str(input_png))
        VtracerVectorizer(preset=preset, params_override=vtracer_params_override).vectorize(  # type: ignore[arg-type]
            input_png, output_svg
        )
        tree = etree.parse(str(output_svg))
        root = tree.getroot()
        paths = [el.get("d", "") for el in root.findall(f"{{{SVG_NS}}}path")]
    return (paths, (x1, y1))
