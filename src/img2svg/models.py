"""Pydantic v2 data models for img2svg."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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


class ConversionOptions(BaseModel):
    """User-facing options for a conversion run."""

    model_config = ConfigDict(use_enum_values=False)

    mode: Mode = Mode.AUTO
    model: str = "yolo11x.pt"
    device: str = "auto"
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    gpu_strategy: DeviceStrategy = DeviceStrategy.POWER
    no_clobber: bool = False
    force_overwrite: bool = False
    palette_size: int = Field(default=8, ge=2, le=64)


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
