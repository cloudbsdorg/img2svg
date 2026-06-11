# img2svg - tests for enums and Pydantic models.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for enums and Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from img2svg.enums import DeviceStrategy, GpuVendor, ImageType, Mode
from img2svg.models import (
    BackendSpec,
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
    assert o.preprocess == []
    assert o.denoise == ""
    assert o.sharpen == ""
    assert o.max_colors == 0
    assert o.quality == 90
    assert o.no_preprocess is False
    assert o.seg_model == "yolo11s-seg"
    assert o.no_seg is False


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


def test_backend_spec_default() -> None:
    b = BackendSpec()
    assert b.requested == "auto"
    assert b.index is None


def test_backend_spec_cuda_indexed_parses() -> None:
    assert BackendSpec.model_validate({"requested": "cuda:0"}) == BackendSpec(
        requested="cuda", index=0
    )
    assert BackendSpec.model_validate({"requested": "cuda:3"}) == BackendSpec(
        requested="cuda", index=3
    )


def test_backend_spec_rocm_indexed_parses() -> None:
    assert BackendSpec.model_validate({"requested": "rocm:0"}) == BackendSpec(
        requested="rocm", index=0
    )
    assert BackendSpec.model_validate({"requested": "rocm:2"}) == BackendSpec(
        requested="rocm", index=2
    )


def test_backend_spec_plain_backends_have_no_index() -> None:
    assert BackendSpec(requested="cpu") == BackendSpec(requested="cpu", index=None)
    assert BackendSpec(requested="mps") == BackendSpec(requested="mps", index=None)
    assert BackendSpec(requested="cuda") == BackendSpec(requested="cuda", index=None)
    assert BackendSpec(requested="rocm") == BackendSpec(requested="rocm", index=None)
    assert BackendSpec(requested="auto") == BackendSpec(requested="auto", index=None)


def test_backend_spec_rejects_bogus() -> None:
    with pytest.raises(ValidationError):
        BackendSpec.model_validate({"requested": "bogus"})


def test_backend_spec_rejects_unsupported_indexed_backends() -> None:
    with pytest.raises(ValidationError):
        BackendSpec.model_validate({"requested": "mps:0"})
    with pytest.raises(ValidationError):
        BackendSpec.model_validate({"requested": "cpu:0"})
    with pytest.raises(ValidationError):
        BackendSpec.model_validate({"requested": "auto:0"})


def test_backend_spec_rejects_non_integer_index() -> None:
    with pytest.raises(ValidationError):
        BackendSpec.model_validate({"requested": "cuda:abc"})


def test_backend_spec_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        BackendSpec.model_validate({"requested": "cuda:-1"})


def test_backend_spec_is_frozen() -> None:
    b = BackendSpec(requested="cpu")
    with pytest.raises(ValidationError):
        b.requested = "cuda"


def test_conversion_options_default_backend() -> None:
    o = ConversionOptions()
    assert o.backend == BackendSpec(requested="auto")
    assert o.backend.index is None


def test_conversion_options_explicit_backend() -> None:
    o = ConversionOptions(backend=BackendSpec(requested="cuda", index=1))
    assert o.backend.requested == "cuda"
    assert o.backend.index == 1


def test_conversion_options_device_deprecation_warning_and_populates_backend() -> None:
    with pytest.warns(DeprecationWarning):
        o = ConversionOptions(device="cuda:0")
    assert o.backend.requested == "cuda"
    assert o.backend.index == 0
    assert o.device == "cuda:0"


def test_conversion_options_device_default_emits_no_warning() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        o = ConversionOptions()
    assert o.backend == BackendSpec(requested="auto")
    assert o.device == "auto"


def test_conversion_options_explicit_backend_wins_over_device() -> None:
    with pytest.warns(DeprecationWarning):
        o = ConversionOptions(device="cpu", backend=BackendSpec(requested="cuda", index=2))
    assert o.backend.requested == "cuda"
    assert o.backend.index == 2
    assert o.device == "cpu"


def test_sidecar_new_backend_fields_construct() -> None:
    s = Sidecar(
        version="0.1.0",
        input_path=Path("/tmp/in.png"),
        input_hash="h",
        output_path=Path("/tmp/out.svg"),
        output_size=100,
        mode_used=Mode.LABELS,
        mode_reasoning="r",
        model="yolo11x.pt",
        device="cpu",
        image_type=ImageType.PHOTO,
        backend_requested="auto",
        backend_resolved="cuda:0",
    )
    assert s.backend_requested == "auto"
    assert s.backend_resolved == "cuda:0"
    assert s.device == "cpu"


def test_sidecar_new_backend_fields_default_to_empty() -> None:
    s = Sidecar(
        version="0.1.0",
        input_path=Path("/tmp/in.png"),
        input_hash="h",
        output_path=Path("/tmp/out.svg"),
        output_size=100,
        mode_used=Mode.LABELS,
        mode_reasoning="r",
        model="yolo11x.pt",
        device="cpu",
        image_type=ImageType.PHOTO,
    )
    assert s.backend_requested == ""
    assert s.backend_resolved == ""


def test_sidecar_loads_legacy_json_without_new_fields() -> None:
    legacy_json = (
        '{"version":"0.1.0","input_path":"/tmp/in.png","input_hash":"h",'
        '"output_path":"/tmp/out.svg","output_size":100,'
        '"mode_used":"labels","mode_reasoning":"r","model":"yolo11x.pt",'
        '"device":"cpu","image_type":"photo"}'
    )
    s = Sidecar.model_validate_json(legacy_json)
    assert s.device == "cpu"
    assert s.backend_requested == ""
    assert s.backend_resolved == ""
