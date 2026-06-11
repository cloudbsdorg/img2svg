# img2svg - Pydantic v2 data models for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Pydantic v2 data models for img2svg."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from img2svg.enums import DeviceStrategy, GpuVendor, ImageType, Mode


class BoundingBox(BaseModel):
    """Rectangle in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """A single object detection result from YOLO."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class GeometricAnalysis(BaseModel):
    """Geometric (non-ML) analysis of an image."""

    dominant_colors: list[tuple[int, int, int]] = Field(default_factory=list)
    edge_density: float = Field(ge=0.0, le=1.0, default=0.0)
    contour_count: int = Field(ge=0, default=0)
    has_alpha: bool = False


class GPUInfo(BaseModel):
    """Information about a single GPU device."""

    index: int
    vendor: GpuVendor
    name: str
    vram_total_mb: int = Field(ge=0, default=0)
    vram_free_mb: int = Field(ge=0, default=0)
    compute_capability: str | None = None
    utilization_pct: float | None = Field(default=None, ge=0.0, le=100.0)


class BackendSpec(BaseModel):
    """Validated specification of a compute backend for image processing.

    Parses user-supplied device strings like ``"cuda:0"`` into a structured
    representation that downstream code can dispatch on. The model is frozen
    because it represents a parsed request, not mutable state.

    The ``requested`` field accepts the literal backends ``"auto"``,
    ``"cuda"``, ``"rocm"``, ``"mps"``, and ``"cpu"``. In addition, a string
    of the form ``"cuda:N"`` or ``"rocm:N"`` is parsed into
    ``(requested="cuda"|"rocm", index=N)`` by the field validator. The
    ``Literal`` type does not natively support variable-index forms, so the
    parser is implemented in code rather than as a typing construct.
    """

    model_config = ConfigDict(frozen=True)

    requested: Literal["auto", "cuda", "rocm", "mps", "cpu"] = "auto"
    index: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_indexed_requested(cls, data: object) -> object:
        """Split ``"cuda:0"`` / ``"rocm:0"`` into (base, index).

        A ``model_validator(mode="before")`` is used rather than a
        ``field_validator`` because the ``Literal`` type annotation on
        ``requested`` rejects ``"cuda:N"`` strings before any field-level
        validator can run. Rewriting the input dict here lets us extract
        the index and then let the per-field validation see the cleaned
        form.

        Plain backends (``"auto"``, ``"cuda"``, ``"rocm"``, ``"mps"``,
        ``"cpu"``) and non-string inputs pass through unchanged. Strings
        with ``":"`` are rejected unless the prefix is ``"cuda"`` or
        ``"rocm"`` (the only backends that support multi-device indexing
        in this codebase).
        """
        if not isinstance(data, dict):
            return data
        req = data.get("requested")
        if not isinstance(req, str) or ":" not in req:
            return data
        base, _, idx = req.partition(":")
        if base not in ("cuda", "rocm"):
            raise ValueError(
                f"Backend {base!r} does not support ':N' indexing "
                f"(got {req!r}); valid indexed forms are 'cuda[:N]' or "
                f"'rocm[:N]'"
            )
        try:
            parsed_idx = int(idx)
        except ValueError as exc:
            raise ValueError(
                f"Invalid index {idx!r} in {req!r}: must be an integer"
            ) from exc
        if parsed_idx < 0:
            raise ValueError(
                f"Backend index must be non-negative (got {parsed_idx})"
            )
        return {**data, "requested": base, "index": parsed_idx}


class ConversionOptions(BaseModel):
    """User-facing options for a conversion run."""

    model_config = ConfigDict(use_enum_values=False)

    mode: Mode = Mode.AUTO
    model: str = "yolo11x.pt"
    device: str = "auto"
    """Deprecated device string. Use ``backend`` instead.

    Setting this field is a transient shim that populates ``backend`` and
    emits a :class:`DeprecationWarning`. Will be removed in 0.3.0.
    """
    backend: BackendSpec = BackendSpec()
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    gpu_strategy: DeviceStrategy = DeviceStrategy.POWER
    no_clobber: bool = False
    force_overwrite: bool = False
    palette_size: int = Field(default=8, ge=2, le=64)

    @model_validator(mode="after")
    def _forward_device_to_backend(self) -> ConversionOptions:
        """Forward the deprecated ``device`` field into ``backend``.

        Only fires when the user explicitly passed ``device`` to the
        constructor (i.e. it is in ``model_fields_set``). The default
        value ``"auto"`` is the same that ``BackendSpec()`` would produce
        on its own, so no warning is needed for plain construction.

        If the user provided ``backend`` explicitly, the explicit value
        wins and the deprecation warning still fires (so they know to
        migrate), but the deprecated ``device`` string does not clobber
        their choice.
        """
        if "device" not in self.model_fields_set:
            return self
        warnings.warn(
            "ConversionOptions.device is deprecated; use "
            "ConversionOptions.backend=BackendSpec(requested=...) instead. "
            "The ``device`` field will be removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "backend" not in self.model_fields_set:
            self.backend = BackendSpec.model_validate({"requested": self.device})
        return self


class Sidecar(BaseModel):
    """JSON metadata written next to every SVG output."""

    version: str
    input_path: Path
    input_hash: str
    output_path: Path
    output_size: int = Field(ge=0)
    mode_used: Mode
    mode_reasoning: str
    model: str
    device: str
    """Deprecated device string. Superseded by ``backend_requested`` and
    ``backend_resolved`` for auditability. Kept for backward compatibility
    with existing sidecar consumers.
    """
    backend_requested: str = ""
    """Original backend requested by the user (e.g. ``"auto"``,
    ``"cuda:0"``). Empty string when the pipeline did not record a value."""
    backend_resolved: str = ""
    """Concrete backend the pipeline dispatched to (e.g. ``"cuda:0"``,
    ``"mps"``). Empty string when the pipeline did not record a value."""
    image_type: ImageType
    detections: list[Detection] = Field(default_factory=list)
    geometric: GeometricAnalysis | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversionResult(BaseModel):
    """Result of converting a single image."""

    svg_path: Path
    sidecar_path: Path
    sidecar: Sidecar
    detections: list[Detection] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
