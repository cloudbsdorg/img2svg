# img2svg - JSON sidecar writer for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""JSON sidecar writer for img2svg.

Writes a `.json` metadata file alongside every produced SVG. The sidecar
captures input/output paths, hashes, mode used, model, device, image type,
detections, geometric analysis, and timings for full reproducibility.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from img2svg.models import Sidecar


def write_sidecar(sidecar: Sidecar, path: Path) -> None:
    """Serialize `sidecar` to JSON at `path` atomically.

    Writes to a temporary file in the same directory, then renames atomically
    to avoid half-written files if the process is killed mid-write.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = sidecar.model_dump_json(indent=2, exclude_none=False)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_sidecar(path: Path) -> Sidecar:
    """Read a sidecar JSON file and return a validated `Sidecar` instance."""
    return Sidecar.model_validate_json(Path(path).read_text(encoding="utf-8"))


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at `path`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
