# img2svg - tests for the preprocessing filter pipeline.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the preprocessing filter pipeline (`img2svg.preprocessing`).

Covers all 7 filter functions, the :class:`PreprocessingPipeline` chain,
the :data:`PREPROCESSING_PRESETS` dict, and the alpha-channel / dtype
contracts. The tests use synthetic numpy arrays (no fixtures needed) so
they run in milliseconds and have no dependency on vtracer.
"""

from __future__ import annotations

import numpy as np
import pytest

from img2svg.preprocessing import (
    PREPROCESSING_PRESETS,
    PreprocessingPipeline,
    apply_clahe_yuv,
    denoise_bilateral,
    denoise_median,
    denoise_nlmeans,
    detect_edges_canny,
    posterize,
    sharpen_unsharp,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _rgb(height: int = 20, width: int = 20, seed: int = 0) -> np.ndarray:
    """Build a deterministic uint8 RGB image with a value range."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _rgba(height: int = 20, width: int = 20) -> np.ndarray:
    """Build a uint8 RGBA image with a recognizable alpha gradient."""
    rgb = _rgb(height, width, seed=1)
    alpha = np.linspace(0, 255, height * width, dtype=np.uint8).reshape(height, width, 1)
    return np.dstack([rgb, alpha])


# ----------------------------------------------------------------------
# dtype / shape contracts
# ----------------------------------------------------------------------


class TestDtypeShapeContracts:
    """All filters preserve dtype (uint8) and shape (H, W, 3 or 4)."""

    @pytest.mark.parametrize(
        "fn",
        [
            denoise_bilateral,
            denoise_nlmeans,
            denoise_median,
            sharpen_unsharp,
            posterize,
            detect_edges_canny,
            apply_clahe_yuv,
        ],
    )
    def test_filters_preserve_uint8_rgb(self, fn) -> None:
        """Each filter returns the same dtype and (H, W, 3) shape on RGB input."""
        img = _rgb(24, 24, seed=2)
        out = fn(img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    @pytest.mark.parametrize(
        "fn",
        [
            denoise_bilateral,
            denoise_nlmeans,
            denoise_median,
            sharpen_unsharp,
            posterize,
            detect_edges_canny,
            apply_clahe_yuv,
        ],
    )
    def test_filters_preserve_uint8_rgba(self, fn) -> None:
        """Each filter returns the same dtype and (H, W, 4) shape on RGBA input."""
        img = _rgba(16, 16)
        out = fn(img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    @pytest.mark.parametrize(
        "fn",
        [
            denoise_bilateral,
            denoise_nlmeans,
            denoise_median,
            sharpen_unsharp,
            posterize,
            apply_clahe_yuv,
        ],
    )
    def test_filters_preserve_alpha_channel(self, fn) -> None:
        """For RGBA input, the alpha slice is preserved exactly across every filter."""
        img = _rgba(12, 12)
        expected_alpha = img[..., 3:4].copy()
        out = fn(img)
        np.testing.assert_array_equal(out[..., 3:4], expected_alpha)

    def test_canny_preserves_alpha_channel(self) -> None:
        """Canny must also preserve alpha even though its output is binary (0/255)."""
        img = _rgba(12, 12)
        expected_alpha = img[..., 3:4].copy()
        out = detect_edges_canny(img)
        np.testing.assert_array_equal(out[..., 3:4], expected_alpha)


# ----------------------------------------------------------------------
# TypeError on non-uint8 input
# ----------------------------------------------------------------------


class TestTypeError:
    """Every filter must reject non-uint8 input with TypeError."""

    @pytest.mark.parametrize(
        "fn",
        [
            denoise_bilateral,
            denoise_nlmeans,
            denoise_median,
            sharpen_unsharp,
            posterize,
            detect_edges_canny,
            apply_clahe_yuv,
        ],
    )
    def test_uint16_input_raises(self, fn) -> None:
        img = _rgb(8, 8).astype(np.uint16) * 2
        with pytest.raises(TypeError, match="uint8"):
            fn(img)

    @pytest.mark.parametrize(
        "fn",
        [
            denoise_bilateral,
            denoise_nlmeans,
            denoise_median,
            sharpen_unsharp,
            posterize,
            detect_edges_canny,
            apply_clahe_yuv,
        ],
    )
    def test_float_input_raises(self, fn) -> None:
        img = _rgb(8, 8).astype(np.float32) / 255.0
        with pytest.raises(TypeError, match="uint8"):
            fn(img)


# ----------------------------------------------------------------------
# Per-filter behavior smoke tests
# ----------------------------------------------------------------------


class TestFilterBehavior:
    """Sanity checks on each filter's specific behavior."""

    def test_posterize_reduces_levels(self) -> None:
        """``bits=4`` collapses each channel to ≤16 unique levels."""
        img = _rgb(16, 16, seed=3)
        out = posterize(img, bits=4)
        # Each channel must be in {0, 16, 32, ..., 240} = 16 distinct levels.
        for ch in range(3):
            levels = np.unique(out[..., ch])
            assert levels.max() <= 240
            assert len(levels) <= 16
        # On an actual 0..255 gradient, the lower bits should be 0.
        assert (out % 16 == 0).all()

    def test_posterize_bits_out_of_range_raises(self) -> None:
        img = _rgb(8, 8)
        with pytest.raises(ValueError, match=r"bits must be in \[1, 8\]"):
            posterize(img, bits=0)
        with pytest.raises(ValueError, match=r"bits must be in \[1, 8\]"):
            posterize(img, bits=9)

    def test_posterize_full_color_passthrough(self) -> None:
        """``bits=8`` is identity (mask = 0xFF = keep all bits)."""
        img = _rgb(8, 8, seed=4)
        out = posterize(img, bits=8)
        np.testing.assert_array_equal(out, img)

    def test_canny_output_is_binary(self) -> None:
        """Canny output values are restricted to {0, 255}."""
        img = _rgb(20, 20, seed=5)
        out = detect_edges_canny(img, low=1, high=10)
        unique = np.unique(out[..., 0])
        assert set(unique.tolist()).issubset({0, 255})

    def test_median_rejects_even_k(self) -> None:
        """``denoise_median`` requires odd kernel size (OpenCV contract)."""
        img = _rgb(8, 8)
        with pytest.raises(ValueError, match="k must be odd"):
            denoise_median(img, k=4)

    def test_bilateral_reduces_noise_in_flat_region(self) -> None:
        """Bilateral reduces stddev on a flat gray + Gaussian-noise patch."""
        rng = np.random.default_rng(7)
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        noisy = np.clip(flat.astype(np.int16) + rng.normal(0, 20, flat.shape), 0, 255).astype(
            np.uint8
        )
        denoised = denoise_bilateral(noisy, d=5, sigma=50)
        assert denoised.std() < noisy.std()


# ----------------------------------------------------------------------
# PreprocessingPipeline
# ----------------------------------------------------------------------


class TestPreprocessingPipeline:
    """The :class:`PreprocessingPipeline` chain class."""

    def test_init_stores_steps(self) -> None:
        steps = [
            ("denoise_bilateral", {"d": 5, "sigma": 50}),
            ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
        ]
        p = PreprocessingPipeline(steps)
        assert list(p._steps) == steps

    def test_apply_returns_same_shape_and_dtype(self) -> None:
        img = _rgb(20, 20, seed=10)
        p = PreprocessingPipeline([("denoise_bilateral", {"d": 5, "sigma": 50})])
        out = p.apply(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_steps_applied_records_filter_order(self) -> None:
        img = _rgb(16, 16, seed=11)
        p = PreprocessingPipeline(
            [
                ("denoise_bilateral", {"d": 5, "sigma": 50}),
                ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
                ("posterize", {"bits": 4}),
            ]
        )
        p.apply(img)
        assert p.steps_applied() == [
            "denoise_bilateral",
            "sharpen_unsharp",
            "posterize",
        ]

    def test_steps_applied_is_empty_before_apply(self) -> None:
        p = PreprocessingPipeline([("denoise_bilateral", {"d": 5, "sigma": 50})])
        assert p.steps_applied() == []

    def test_steps_applied_resets_between_runs(self) -> None:
        img = _rgb(12, 12)
        p = PreprocessingPipeline(
            [
                ("denoise_bilateral", {"d": 5, "sigma": 50}),
                ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
            ]
        )
        p.apply(img)
        assert len(p.steps_applied()) == 2
        # Re-run with a single-step pipeline (defensive copy inside apply).
        p2 = PreprocessingPipeline([("posterize", {"bits": 4})])
        p2.apply(img)
        assert p2.steps_applied() == ["posterize"]

    def test_apply_unknown_filter_raises(self) -> None:
        img = _rgb(12, 12)
        p = PreprocessingPipeline([("not_a_real_filter", {})])
        with pytest.raises(ValueError, match="Unknown preprocessing filter"):
            p.apply(img)

    def test_empty_pipeline_is_identity(self) -> None:
        """Zero steps → return the image unchanged."""
        img = _rgb(12, 12, seed=12)
        p = PreprocessingPipeline([])
        out = p.apply(img)
        np.testing.assert_array_equal(out, img)
        assert p.steps_applied() == []


# ----------------------------------------------------------------------
# PREPROCESSING_PRESETS
# ----------------------------------------------------------------------


class TestPreprocessingPresets:
    """The :data:`PREPROCESSING_PRESETS` dict."""

    def test_preset_keys(self) -> None:
        assert set(PREPROCESSING_PRESETS.keys()) == {"light", "medium", "heavy", "edge"}

    def test_all_presets_are_pipelines(self) -> None:
        """Every preset value is a list of (name, kwargs) tuples that PreprocessingPipeline accepts."""
        for key, steps in PREPROCESSING_PRESETS.items():
            p = PreprocessingPipeline(steps)
            img = _rgb(12, 12, seed=hash(key) & 0xFFFF)
            out = p.apply(img)
            assert out.shape == img.shape
            assert out.dtype == np.uint8
            assert len(p.steps_applied()) == len(steps)

    def test_preset_step_names_match_keys(self) -> None:
        """Every (name, kwargs) tuple in every preset uses a real filter name."""
        for key, steps in PREPROCESSING_PRESETS.items():
            for name, _ in steps:
                # Each name must be callable from the module.
                fn = globals()[name]
                assert callable(fn), f"Preset {key!r} references unknown filter {name!r}"

    def test_presets_are_progressively_stronger(self) -> None:
        """``light`` ⊊ ``medium`` ⊊ ``heavy`` (edge is a separate branch)."""
        light_names = [n for n, _ in PREPROCESSING_PRESETS["light"]]
        medium_names = [n for n, _ in PREPROCESSING_PRESETS["medium"]]
        heavy_names = [n for n, _ in PREPROCESSING_PRESETS["heavy"]]
        # Each is a strict superset of the previous (in order).
        assert medium_names[: len(light_names)] == light_names
        assert heavy_names[: len(medium_names)] == medium_names

    def test_edge_preset_uses_line_art_filters(self) -> None:
        """The ``edge`` preset uses median + Canny, the line-art chain."""
        names = [n for n, _ in PREPROCESSING_PRESETS["edge"]]
        assert "denoise_median" in names
        assert "detect_edges_canny" in names
