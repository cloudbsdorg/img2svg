# img2svg - composable OpenCV preprocessing filters for vtracer input.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Composable OpenCV preprocessing filters for vtracer input.

Seven filter functions transform a uint8 image of shape ``(H, W, 3)`` or
``(H, W, 4)`` and return the same dtype/shape:

  * :func:`denoise_bilateral` / :func:`denoise_nlmeans` / :func:`denoise_median`
  * :func:`sharpen_unsharp` / :func:`posterize`
  * :func:`detect_edges_canny` / :func:`apply_clahe_yuv`

Chain with :class:`PreprocessingPipeline` or pick a named
:data:`PREPROCESSING_PRESETS` entry (``light``, ``medium``, ``heavy``, ``edge``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

# Kwargs is loose-typed: filters accept int, float, and tuple values.
Step = tuple[str, dict[str, Any]]


def _check_uint8(image: np.ndarray) -> None:
    """Raise TypeError if image dtype is not uint8."""
    if image.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 image, got {image.dtype}. "
            "Convert with `image.astype(np.uint8)` before applying filters."
        )


def _split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split off alpha if RGBA. The alpha slice keeps a trailing axis so
    :func:`np.dstack` can reattach it. Returns ``(rgb, None)`` for 3-channel.
    """
    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., :3], image[..., 3:4]
    return image, None


def _merge_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Reattach alpha channel if one was split off, else return rgb unchanged."""
    if alpha is not None:
        return np.dstack([rgb, alpha])
    return rgb


def denoise_bilateral(image: np.ndarray, d: int = 5, sigma: int = 50) -> np.ndarray:
    """Edge-preserving bilateral denoise (default for photos).

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        d: Diameter of the pixel neighborhood.
        sigma: Filter sigma in color and coordinate space.

    Returns:
        uint8 array of the same shape as input.
    """
    _check_uint8(image)
    rgb, alpha = _split_alpha(image)
    denoised = cv2.bilateralFilter(rgb, d, sigma, sigma)
    return _merge_alpha(denoised, alpha)


def denoise_nlmeans(image: np.ndarray, h: int = 6) -> np.ndarray:
    """Non-Local Means denoise (heavier statistical, slower than bilateral).

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        h: Filter strength for the luminance channel.

    Returns:
        uint8 array of the same shape as input.
    """
    _check_uint8(image)
    rgb, alpha = _split_alpha(image)
    denoised = cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 7, 21)
    return _merge_alpha(denoised, alpha)


def denoise_median(image: np.ndarray, k: int = 3) -> np.ndarray:
    """Median filter (cheap baseline; used in the edge preset).

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        k: Kernel size (must be odd; OpenCV requires it).

    Returns:
        uint8 array of the same shape as input.

    Raises:
        ValueError: if ``k`` is even.
    """
    _check_uint8(image)
    if k % 2 == 0:
        raise ValueError(f"k must be odd, got {k}")
    rgb, alpha = _split_alpha(image)
    denoised = cv2.medianBlur(rgb, k)
    return _merge_alpha(denoised, alpha)


def sharpen_unsharp(image: np.ndarray, sigma: float = 2.0, amount: float = 0.5) -> np.ndarray:
    """Unsharp mask via the deforum pattern.

    Computes ``sharpened = (1 + amount) * img - amount * gaussian_blur(img)``
    using :func:`cv2.addWeighted` (``alpha = 1 + amount``, ``beta = -amount``,
    ``gamma = 0``).

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        sigma: Gaussian sigma for the blur (larger = wider sharpening scale).
        amount: Sharpening amount. ``0`` = identity, ``1`` = strong.

    Returns:
        uint8 array of the same shape as input.
    """
    _check_uint8(image)
    rgb, alpha = _split_alpha(image)
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
    sharpened = cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0)
    return _merge_alpha(sharpened, alpha)


def posterize(image: np.ndarray, bits: int = 4) -> np.ndarray:
    """Reduce color levels per channel by zeroing the lowest bits.

    Uses :func:`np.bitwise_and` with a left-shift mask (no PIL). The mask
    is ``np.uint8((255 << (8 - bits)) & 0xFF)``; the bitwise AND with
    ``0xFF`` clamps the Python int to 8 bits before the numpy uint8 cast
    (numpy >= 2.0 raises ``OverflowError`` for out-of-range ints).

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        bits: Number of high bits to keep per channel, in ``[1, 8]``.

    Returns:
        uint8 array of the same shape as input.

    Raises:
        ValueError: if ``bits`` is not in ``[1, 8]``.
    """
    _check_uint8(image)
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    rgb, alpha = _split_alpha(image)
    mask = np.uint8((255 << (8 - bits)) & 0xFF)
    out = np.bitwise_and(rgb, mask)
    return _merge_alpha(out, alpha)


def detect_edges_canny(image: np.ndarray, low: int = 80, high: int = 180) -> np.ndarray:
    """Canny edge detection. Output is 3-channel uint8 for vtracer.

    Converts to gray, blurs, runs Canny, then re-expands the 1-channel
    edges to 3-channel so the output shape matches the input contract.
    Output values are restricted to ``{0, 255}``.

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        low: Canny low threshold (hysteresis).
        high: Canny high threshold (hysteresis).

    Returns:
        uint8 array of the same shape as input, values in ``{0, 255}``.
    """
    _check_uint8(image)
    rgb, alpha = _split_alpha(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return _merge_alpha(edges_3ch, alpha)


def apply_clahe_yuv(
    image: np.ndarray, clip: float = 2.0, tile: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """CLAHE on the YUV L-channel only (preserves color).

    Converts to YUV, runs CLAHE on luminance, converts back. Avoids the
    color shifts that come from CLAHE on raw RGB.

    Args:
        image: uint8 array of shape ``(H, W, 3)`` or ``(H, W, 4)``.
        clip: CLAHE clip limit.
        tile: CLAHE tile grid size, ``(width, height)`` in cells.

    Returns:
        uint8 array of the same shape as input.
    """
    _check_uint8(image)
    rgb, alpha = _split_alpha(image)
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    yuv[..., 0] = clahe.apply(yuv[..., 0])
    out = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return _merge_alpha(out, alpha)


# Filter name -> callable dispatch table used by PreprocessingPipeline.
_FILTERS: dict[str, Callable[..., np.ndarray]] = {
    "apply_clahe_yuv": apply_clahe_yuv,
    "denoise_bilateral": denoise_bilateral,
    "denoise_median": denoise_median,
    "denoise_nlmeans": denoise_nlmeans,
    "detect_edges_canny": detect_edges_canny,
    "posterize": posterize,
    "sharpen_unsharp": sharpen_unsharp,
}


class PreprocessingPipeline:
    """Chain of composable filter steps.

    Each step is a ``(filter_name, kwargs)`` tuple; filter names must be
    keys in the internal registry. Steps run in order during :meth:`apply`.

    Example::

        pipeline = PreprocessingPipeline([
            ("denoise_bilateral", {"d": 5, "sigma": 50}),
            ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
        ])
        result = pipeline.apply(image)
        assert pipeline.steps_applied() == [
            "denoise_bilateral", "sharpen_unsharp",
        ]
    """

    def __init__(self, steps: list[Step]) -> None:
        self._steps: list[Step] = list(steps)
        self._applied: list[str] = []

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Run all steps in order, returning the transformed image.

        Records each filter name so :meth:`steps_applied` can report what
        ran. Output dtype and shape match the input.

        Raises:
            ValueError: if a step references an unknown filter name.
        """
        self._applied = []
        result = image
        for name, kwargs in self._steps:
            if name not in _FILTERS:
                raise ValueError(
                    f"Unknown preprocessing filter '{name}'. Available: {sorted(_FILTERS)}"
                )
            result = _FILTERS[name](result, **kwargs)
            self._applied.append(name)
        return result

    def steps_applied(self) -> list[str]:
        """Return filter names that ran during the last :meth:`apply` (defensive copy)."""
        return list(self._applied)


# Photo chains are progressive (denoise -> sharpen -> posterize -> edges);
# ``edge`` is the line-art chain (median + Canny).
PREPROCESSING_PRESETS: dict[str, list[Step]] = {
    "light": [
        ("denoise_bilateral", {"d": 5, "sigma": 50}),
        ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
    ],
    "medium": [
        ("denoise_bilateral", {"d": 5, "sigma": 50}),
        ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
        ("posterize", {"bits": 4}),
    ],
    "heavy": [
        ("denoise_bilateral", {"d": 5, "sigma": 50}),
        ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
        ("posterize", {"bits": 4}),
        ("detect_edges_canny", {"low": 80, "high": 180}),
    ],
    "edge": [
        ("denoise_median", {"k": 3}),
        ("detect_edges_canny", {"low": 80, "high": 180}),
    ],
}


__all__ = [
    "PREPROCESSING_PRESETS",
    "PreprocessingPipeline",
    "apply_clahe_yuv",
    "denoise_bilateral",
    "denoise_median",
    "denoise_nlmeans",
    "detect_edges_canny",
    "posterize",
    "sharpen_unsharp",
]
