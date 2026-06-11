# img2svg - typed exception hierarchy for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed exception hierarchy for img2svg.

All exceptions inherit from `Img2SvgError` so callers can catch every
img2svg-specific error with a single except clause. Each exception exposes
both `user_message()` (safe for end-user display) and `dev_message()` (full
diagnostic info for logs / devs).
"""

from __future__ import annotations


class Img2SvgError(Exception):
    """Base class for all img2svg-specific errors."""

    def user_message(self) -> str:
        return str(self)

    def dev_message(self) -> str:
        return f"{type(self).__name__}: {self}"


class UnsupportedFormatError(Img2SvgError):
    """Raised when an input file's format is not in the supported set."""

    def __init__(self, format_name: str, supported: tuple[str, ...] | None = None) -> None:
        self.format_name = format_name
        self.supported = supported or (
            "PNG",
            "JPEG",
            "BMP",
            "WEBP",
            "TIFF",
            "GIF",
        )
        super().__init__(
            f"unsupported image format: {format_name!r} (supported: {', '.join(self.supported)})"
        )

    def user_message(self) -> str:
        return (
            f"Format {self.format_name!r} is not supported. "
            f"Supported formats: {', '.join(self.supported)}."
        )


class CorruptImageError(Img2SvgError):
    """Raised when Pillow (or another loader) cannot decode the image bytes."""

    def __init__(self, path: str, original: Exception | None = None) -> None:
        self.path = path
        self.original = original
        msg = f"failed to decode image: {path!r}"
        if original is not None:
            msg += f" ({type(original).__name__}: {original})"
        super().__init__(msg)

    def user_message(self) -> str:
        return f"The file {self.path!r} is corrupt or not a valid image."


class ModelLoadError(Img2SvgError):
    """Raised when the YOLO model cannot be loaded or downloaded."""

    def __init__(self, model_name: str, original: Exception | None = None) -> None:
        self.model_name = model_name
        self.original = original
        msg = f"failed to load model: {model_name!r}"
        if original is not None:
            msg += f" ({type(original).__name__}: {original})"
        super().__init__(msg)

    def user_message(self) -> str:
        return (
            f"Could not load the {self.model_name!r} model. "
            "Check your internet connection and try again."
        )


class DeviceUnavailableError(Img2SvgError):
    """Raised when an explicit --device X is requested but X is not available."""

    def __init__(self, requested: str, available: list[str]) -> None:
        self.requested = requested
        self.available = available
        super().__init__(
            f"requested device {requested!r} not available; "
            f"available: {', '.join(available) or 'none'}"
        )

    def user_message(self) -> str:
        avail = ", ".join(self.available) or "none"
        return (
            f"Device {self.requested!r} is not available on this system. "
            f"Available devices: {avail}. "
            f"Use --device auto to autodetect, or --device cpu to always use CPU."
        )


class OutputPathCollisionError(Img2SvgError):
    """Raised when the target output path exists and --no-clobber is set."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"output path already exists: {path!r} (use --force to overwrite)")

    def user_message(self) -> str:
        return (
            f"Output {self.path!r} already exists. "
            "Use --force to overwrite or --no-clobber to skip."
        )


class VectorizationError(Img2SvgError):
    """Raised when vtracer (or another vectorizer) fails to produce output."""

    def __init__(self, input_path: str, original: Exception | None = None) -> None:
        self.input_path = input_path
        self.original = original
        msg = f"vectorization failed for {input_path!r}"
        if original is not None:
            msg += f" ({type(original).__name__}: {original})"
        super().__init__(msg)


class ConfigError(Img2SvgError):
    """Raised when configuration cannot be loaded or is invalid."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"invalid config at {path!r}: {reason}")
