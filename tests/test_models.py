# img2svg - tests for enums and Pydantic models.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for enums and Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from img2svg.enums import DeviceStrategy, GpuVendor, ImageType, Mode
from img2svg.models import (
    BoundingBox,
    ConversionOptions,
    ConversionResult,
    Detection,
    GeometricAnalysis,
    GPUInfo,
    Sidecar,
)


def test_mode_enum_values() -> None:
    assert Mode.AUTO == "auto"
    assert Mode.LABELS == "labels"
    assert Mode.VISUAL == "visual"
    assert Mode.ANNOTATED == "annotated"
    assert Mode.TRACE == "trace"


def test_image_type_enum_values() -> None:
    assert ImageType.LOGO == "logo"
    assert ImageType.PHOTO == "photo"
    assert ImageType.UNKNOWN == "unknown"


def test_gpu_vendor_enum_values() -> None:
    assert GpuVendor.NVIDIA == "nvidia"
    assert GpuVendor.AMD == "amd"
    assert GpuVendor.APPLE == "apple"


def test_bounding_box_properties() -> None:
    bb = BoundingBox(x1=10, y1=20, x2=110, y2=220)
    assert bb.width == 100
    assert bb.height == 200
    assert bb.area == 20000


def test_detection_confidence_bounds() -> None:
    Detection(
        class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)
    )
    with pytest.raises(ValidationError):
        Detection(
            class_id=0, class_name="x", confidence=1.5, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1)
        )
    with pytest.raises(ValidationError):
        Detection(
            class_id=0, class_name="x", confidence=-0.1, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1)
        )


def test_geometric_analysis_defaults() -> None:
    ga = GeometricAnalysis()
    assert ga.dominant_colors == []
    assert ga.edge_density == 0.0
    assert ga.contour_count == 0
    assert ga.has_alpha is False


def test_gpu_info_fields() -> None:
    g = GPUInfo(
        index=0, vendor=GpuVendor.NVIDIA, name="RTX 4090", vram_total_mb=24576, vram_free_mb=20000
    )
    assert g.vram_total_mb == 24576
    assert g.vram_free_mb == 20000


def test_conversion_options_defaults() -> None:
    o = ConversionOptions()
    assert o.mode == Mode.AUTO
    assert o.model == "yolo11x.pt"
    assert o.device == "auto"
    assert o.conf == 0.25
    assert o.iou == 0.7
    assert o.gpu_strategy == DeviceStrategy.POWER
    assert o.palette_size == 8


def test_conversion_options_conf_bounds() -> None:
    with pytest.raises(ValidationError):
        ConversionOptions(conf=2.0)
    with pytest.raises(ValidationError):
        ConversionOptions(conf=-0.1)


def test_sidecar_round_trip_json() -> None:
    s = Sidecar(
        version="0.1.0",
        input_path=Path("/tmp/in.png"),
        input_hash="abc123",
        output_path=Path("/tmp/out.svg"),
        output_size=1024,
        mode_used=Mode.LABELS,
        mode_reasoning="auto: photo -> labels",
        model="yolo11x.pt",
        device="cpu",
        image_type=ImageType.PHOTO,
        detections=[],
        geometric=GeometricAnalysis(has_alpha=True),
        timings={"load": 0.1, "detect": 0.5},
        timestamp="2026-06-09T00:00:00+00:00",
    )
    json_str = s.model_dump_json()
    parsed = json.loads(json_str)
    s2 = Sidecar.model_validate_json(json_str)
    assert s2 == s
    assert parsed["mode_used"] == "labels"
    assert parsed["image_type"] == "photo"


def test_conversion_result_basic() -> None:
    sidecar = Sidecar(
        version="0.1.0",
        input_path=Path("/tmp/in.png"),
        input_hash="h",
        output_path=Path("/tmp/out.svg"),
        output_size=100,
        mode_used=Mode.VISUAL,
        mode_reasoning="r",
        model="yolo11x.pt",
        device="cpu",
        image_type=ImageType.LOGO,
    )
    r = ConversionResult(
        svg_path=Path("/tmp/out.svg"),
        sidecar_path=Path("/tmp/out.json"),
        sidecar=sidecar,
    )
    assert r.detections == []
    assert r.errors == []
