#!/usr/bin/env python3
"""Generate synthetic test images for img2svg tests.

Idempotent: re-running produces the same files. Run from the project root:

    uv run python scripts/gen_test_images.py

Output:
    tests/fixtures/logo.png        — circle + rectangle (geometric, few colors)
    tests/fixtures/photo.jpg       — noisy gradient (many colors)
    tests/fixtures/diagram.png     — black lines on white (line art)
    tests/fixtures/line_art.png    — strokes only
    tests/fixtures/transparent.png — RGBA with semi-transparent corners
    tests/fixtures/screenshot.png  — solid blocks of color (UI mockup)
    tests/fixtures/corrupt.bin     — garbage bytes for error tests
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Fixed seed for reproducibility.
_SEED = 42


def _ensure_dir() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def _gen_logo() -> Path:
    """Logo: circle + rectangle on a white background. Few colors, sharp edges."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((30, 30, 170, 170), fill=(220, 50, 50), outline=(0, 0, 0), width=2)
    d.rectangle((60, 60, 140, 140), fill=(50, 50, 200), outline=(0, 0, 0), width=2)
    p = FIXTURES_DIR / "logo.png"
    img.save(p)
    return p


def _gen_photo() -> Path:
    """Photo-like: noisy gradient, many colors, no clear edges."""
    rng = np.random.default_rng(_SEED)
    h, w = 256, 256
    grad = np.linspace(0, 255, h, dtype=np.uint8).reshape(h, 1, 1)
    grad = np.broadcast_to(grad, (h, w, 1))
    noise = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    blended = (0.4 * grad + 0.6 * noise).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(blended, mode="RGB")
    p = FIXTURES_DIR / "photo.jpg"
    img.save(p, format="JPEG", quality=85)
    return p


def _gen_diagram() -> Path:
    """Diagram: black lines on white. Line art, low color count."""
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.line((20, 20, 280, 20), fill=(0, 0, 0), width=2)
    d.line((20, 20, 20, 180), fill=(0, 0, 0), width=2)
    d.rectangle((40, 40, 130, 100), outline=(0, 0, 0), width=2)
    d.rectangle((150, 40, 260, 100), outline=(0, 0, 0), width=2)
    d.line((130, 70, 150, 70), fill=(0, 0, 0), width=2)
    d.line((20, 180, 280, 180), fill=(0, 0, 0), width=2)
    d.text((30, 110), "Node A", fill=(0, 0, 0))
    d.text((150, 110), "Node B", fill=(0, 0, 0))
    p = FIXTURES_DIR / "diagram.png"
    img.save(p)
    return p


def _gen_line_art() -> Path:
    """Line art: only thin strokes, lots of whitespace."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(20, 200, 15):
        d.line((10, y, 190, y), fill=(0, 0, 0), width=1)
    p = FIXTURES_DIR / "line_art.png"
    img.save(p)
    return p


def _gen_transparent() -> Path:
    """RGBA with semi-transparent corners and opaque center."""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 80, 80), fill=(255, 0, 0, 255))
    # Fade alpha toward corners
    arr = np.array(img)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    alpha = np.clip(255 * (1 - dist / max_dist), 0, 255).astype(np.uint8)
    arr[..., 3] = alpha
    p = FIXTURES_DIR / "transparent.png"
    Image.fromarray(arr, mode="RGBA").save(p)
    return p


def _gen_screenshot() -> Path:
    """Screenshot mock: solid color blocks (UI mockup style)."""
    img = Image.new("RGB", (400, 300), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 400, 40), fill=(50, 100, 200))  # header
    d.rectangle((0, 40, 100, 300), fill=(220, 220, 220))  # sidebar
    d.rectangle((120, 60, 380, 120), fill=(255, 255, 255), outline=(200, 200, 200))  # card
    d.rectangle((120, 140, 240, 220), fill=(180, 220, 180))  # button
    d.rectangle((260, 140, 380, 220), fill=(220, 180, 180))  # button
    p = FIXTURES_DIR / "screenshot.png"
    img.save(p)
    return p


def _gen_corrupt() -> Path:
    """Garbage bytes that no image loader should accept."""
    random.seed(_SEED)
    p = FIXTURES_DIR / "corrupt.bin"
    p.write_bytes(bytes(random.randint(0, 255) for _ in range(256)))
    return p


def main() -> int:
    _ensure_dir()
    paths = [
        _gen_logo(),
        _gen_photo(),
        _gen_diagram(),
        _gen_line_art(),
        _gen_transparent(),
        _gen_screenshot(),
        _gen_corrupt(),
    ]
    total = sum(p.stat().st_size for p in paths)
    print(f"Generated {len(paths)} fixture files ({total} bytes total) in {FIXTURES_DIR}")
    for p in paths:
        print(f"  {p.name}: {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
