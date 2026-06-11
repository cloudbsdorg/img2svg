# img2svg - image loading with format validation and normalization.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Image loading with format validation and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from img2svg.errors import CorruptImageError, UnsupportedFormatError

SUPPORTED_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG", "BMP", "WEBP", "TIFF", "GIF"})


@dataclass
class LoadedImage:
    """A loaded and normalized image."""

    pil_image: Image.Image
    format: str
    original_mode: str
    has_alpha: bool
    width: int
    height: int

    @property
    def np_array(self) -> np.ndarray:
        """Image as numpy array (H, W, C) uint8."""
        return np.asarray(self.pil_image)


def load_image(path: str | Path) -> LoadedImage:
    """Load an image from `path`, validate format, normalize.

    Normalizations:
      - Animated GIF: take first frame.
      - 16-bit PNG: convert to 8-bit.
      - CMYK JPEG: convert to RGB.
      - RGBA images: preserved (alpha kept, not flattened).

    Raises:
      UnsupportedFormatError: if the file's format is not in SUPPORTED_FORMATS.
      CorruptImageError: if Pillow cannot decode the file.
    """
    p = Path(path)
    try:
        with Image.open(p) as raw:
            fmt = (raw.format or "").upper()
            if fmt not in SUPPORTED_FORMATS:
                raise UnsupportedFormatError(fmt or "unknown")

            # Take first frame of animated formats (GIF, animated PNG/WEBP).
            frame = ImageSequence.Iterator(raw)
            first = next(frame)
            first.load()

            original_mode = first.mode
            has_alpha = "A" in original_mode or first.mode == "PA"

            # Normalize bit depth to 8.
            if first.mode in ("I", "F"):
                first = first.convert("RGB")
            if first.mode == "CMYK":
                first = first.convert("RGB")
            # 16-bit RGB(A) → 8-bit.
            if first.mode.startswith("I;16") or (hasattr(first, "bits") and first.bits != 8):
                first = first.convert("RGB")

            width, height = first.size
            return LoadedImage(
                pil_image=first,
                format=fmt,
                original_mode=original_mode,
                has_alpha=has_alpha,
                width=width,
                height=height,
            )
    except UnsupportedFormatError:
        raise
    except (Image.UnidentifiedImageError, OSError, ValueError) as e:
        raise CorruptImageError(str(p), original=e) from e
