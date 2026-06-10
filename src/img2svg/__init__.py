# img2svg - convert raster images to clean, optimized SVG.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""img2svg — convert raster images to clean, optimized SVG.

Public entry points (T19):
- :func:`convert` — convert a single image to SVG.
- :func:`convert_batch` — convert a list of images or a glob pattern.
- :class:`ConversionOptions` — user-facing options (mode, model, device, ...).
- :class:`ConversionResult` — the result of a single conversion (paths, sidecar).
- :class:`Sidecar` — the JSON metadata written next to every SVG output.
- :class:`Mode` — output rendering mode (AUTO / LABELS / VISUAL / ANNOTATED / TRACE).
- :class:`ImageType` — heuristic image type classification (LOGO / PHOTO / ...).
- :class:`DeviceStrategy` — strategy for picking the best GPU.
"""

from __future__ import annotations

from img2svg.api import convert, convert_batch
from img2svg.enums import DeviceStrategy, ImageType, Mode
from img2svg.models import ConversionOptions, ConversionResult, Sidecar

__all__ = [
    "ConversionOptions",
    "ConversionResult",
    "DeviceStrategy",
    "ImageType",
    "Mode",
    "Sidecar",
    "convert",
    "convert_batch",
]
