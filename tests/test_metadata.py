# img2svg - tests for the metadata sidecar writer.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the metadata sidecar writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img2svg.metadata import compute_file_hash, read_sidecar, write_sidecar
from img2svg.models import BoundingBox, RegionInfo, Sidecar


def _make_sidecar(tmp_path: Path) -> Sidecar:
    return Sidecar(
        version="0.1.0",
        input_path=tmp_path / "in.png",
        input_hash="deadbeef" * 8,
        output_path=tmp_path / "out.svg",
        output_size=12345,
        mode_used="visual",
        mode_reasoning="auto: photo -> visual",
        model="yolo11x.pt",
        device="auto",
        image_type="photo",
        detections=[],
        geometric=None,
        timings={"total": 1.23},
    )


def test_write_produces_valid_json(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    out = tmp_path / "out.json"
    write_sidecar(sidecar, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["input_hash"] == "deadbeef" * 8
    assert data["output_size"] == 12345
    assert data["mode_used"] == "visual"
    assert data["image_type"] == "photo"


def test_round_trip_preserves_data(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    out = tmp_path / "round.json"
    write_sidecar(sidecar, out)
    loaded = read_sidecar(out)
    assert loaded.input_hash == sidecar.input_hash
    assert loaded.output_size == sidecar.output_size
    assert loaded.mode_used == sidecar.mode_used
    assert loaded.image_type == sidecar.image_type
    assert loaded.timings == sidecar.timings


def test_atomic_write_leaves_no_tmp_files(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    out = tmp_path / "atomic.json"
    write_sidecar(sidecar, out)
    siblings = [p for p in tmp_path.iterdir() if p.name != out.name]
    assert siblings == [], f"unexpected leftover files: {siblings}"


def test_creates_parent_directory(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    nested = tmp_path / "deep" / "nested" / "side.json"
    write_sidecar(sidecar, nested)
    assert nested.exists()


def test_compute_file_hash_known_value(tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert compute_file_hash(f) == expected


def test_compute_file_hash_changes_with_content(tmp_path: Path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"alpha")
    f2.write_bytes(b"beta")
    assert compute_file_hash(f1) != compute_file_hash(f2)


def test_read_sidecar_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(Exception):
        read_sidecar(bad)


def test_sidecar_preprocessing_field_round_trip(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    sidecar.preprocessing = ["bilateral", "unsharp"]
    out = tmp_path / "pp.json"
    write_sidecar(sidecar, out)
    loaded = read_sidecar(out)
    assert loaded.preprocessing == ["bilateral", "unsharp"]


def test_sidecar_regions_field_round_trip(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    sidecar.regions = [
        RegionInfo(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0),
            area_pixels=20000,
            polygon=[(10.0, 20.0), (110.0, 20.0), (110.0, 220.0), (10.0, 220.0)],
            mask_path=None,
        )
    ]
    out = tmp_path / "regions.json"
    write_sidecar(sidecar, out)
    loaded = read_sidecar(out)
    assert len(loaded.regions) == 1
    assert loaded.regions[0].class_name == "person"
    assert loaded.regions[0].area_pixels == 20000
    assert loaded.regions[0].polygon == [
        (10.0, 20.0),
        (110.0, 20.0),
        (110.0, 220.0),
        (10.0, 220.0),
    ]


def test_sidecar_model_variant_field_round_trip(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    sidecar.model_variant = "yolo11s-seg"
    out = tmp_path / "variant.json"
    write_sidecar(sidecar, out)
    loaded = read_sidecar(out)
    assert loaded.model_variant == "yolo11s-seg"


def test_sidecar_new_fields_default_to_empty(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    assert sidecar.preprocessing == []
    assert sidecar.regions == []
    assert sidecar.model_variant == ""


def test_sidecar_loads_legacy_json_without_new_fields(tmp_path: Path) -> None:
    legacy = (
        '{"version":"0.1.0","input_path":"/tmp/in.png","input_hash":"h",'
        '"output_path":"/tmp/out.svg","output_size":100,'
        '"mode_used":"visual","mode_reasoning":"r","model":"yolo11x.pt",'
        '"device":"auto","image_type":"photo","detections":[]}'
    )
    bad = tmp_path / "legacy.json"
    bad.write_text(legacy, encoding="utf-8")
    loaded = read_sidecar(bad)
    assert loaded.preprocessing == []
    assert loaded.regions == []
    assert loaded.model_variant == ""
