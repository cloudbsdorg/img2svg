"""Tests for the image loader."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from img2svg.errors import CorruptImageError, UnsupportedFormatError
from img2svg.loader import SUPPORTED_FORMATS, LoadedImage, load_image


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.png"
    Image.new("RGB", (10, 10), (255, 0, 0)).save(p)
    return p


@pytest.fixture
def jpeg_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.jpg"
    Image.new("RGB", (10, 10), (0, 255, 0)).save(p, format="JPEG")
    return p


@pytest.fixture
def bmp_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.bmp"
    Image.new("RGB", (10, 10), (0, 0, 255)).save(p, format="BMP")
    return p


@pytest.fixture
def webp_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.webp"
    Image.new("RGB", (10, 10), (128, 128, 128)).save(p, format="WEBP")
    return p


@pytest.fixture
def tiff_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.tiff"
    Image.new("RGB", (10, 10), (64, 64, 64)).save(p, format="TIFF")
    return p


@pytest.fixture
def gif_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.gif"
    Image.new("RGB", (10, 10), (32, 32, 32)).save(p, format="GIF")
    return p


@pytest.fixture
def rgba_path(tmp_path: Path) -> Path:
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(p)
    return p


@pytest.fixture
def cmyk_path(tmp_path: Path) -> Path:
    p = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (10, 10), (0, 255, 0, 0)).save(p, format="JPEG")
    return p


def test_supported_formats_constant() -> None:
    assert "PNG" in SUPPORTED_FORMATS
    assert "JPEG" in SUPPORTED_FORMATS
    assert "BMP" in SUPPORTED_FORMATS
    assert "WEBP" in SUPPORTED_FORMATS
    assert "TIFF" in SUPPORTED_FORMATS
    assert "GIF" in SUPPORTED_FORMATS


def test_load_png(png_path: Path) -> None:
    loaded = load_image(png_path)
    assert isinstance(loaded, LoadedImage)
    assert loaded.format == "PNG"
    assert loaded.width == 10
    assert loaded.height == 10
    assert loaded.has_alpha is False


def test_load_jpeg(jpeg_path: Path) -> None:
    loaded = load_image(jpeg_path)
    assert loaded.format == "JPEG"
    assert loaded.has_alpha is False


def test_load_bmp(bmp_path: Path) -> None:
    loaded = load_image(bmp_path)
    assert loaded.format == "BMP"


def test_load_webp(webp_path: Path) -> None:
    loaded = load_image(webp_path)
    assert loaded.format == "WEBP"


def test_load_tiff(tiff_path: Path) -> None:
    loaded = load_image(tiff_path)
    assert loaded.format == "TIFF"


def test_load_gif(gif_path: Path) -> None:
    loaded = load_image(gif_path)
    assert loaded.format == "GIF"


def test_load_rgba_preserves_alpha(rgba_path: Path) -> None:
    loaded = load_image(rgba_path)
    assert loaded.has_alpha is True
    assert loaded.original_mode == "RGBA"


def test_load_cmyk_converts_to_rgb(cmyk_path: Path) -> None:
    loaded = load_image(cmyk_path)
    # Original was CMYK, but should be converted to RGB internally.
    assert loaded.original_mode == "CMYK"
    assert loaded.pil_image.mode == "RGB"


def test_load_ico_raises_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "test.ico"
    # Create a minimal valid ICO file (16x16, 1 frame).
    img = Image.new("RGB", (16, 16), (255, 255, 255))
    img.save(p, format="ICO")
    with pytest.raises(UnsupportedFormatError) as exc_info:
        load_image(p)
    assert "ICO" in str(exc_info.value)
    assert "PNG" in exc_info.value.user_message()  # mentions supported formats


def test_load_corrupt_file_raises_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "bad.png"
    p.write_bytes(b"not actually a PNG file")
    with pytest.raises(CorruptImageError) as exc_info:
        load_image(p)
    assert "bad.png" in exc_info.value.user_message()


def test_load_accepts_string_path(png_path: Path) -> None:
    loaded = load_image(str(png_path))
    assert loaded.format == "PNG"


def test_np_array_property(png_path: Path) -> None:
    import numpy as np
    loaded = load_image(png_path)
    arr = loaded.np_array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10, 10, 3)
