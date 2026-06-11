# img2svg - renderer base class and shared types for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Renderer base class and shared types for img2svg.

A `Renderer` mutates an `SVGDocument` in place by adding groups, rects, paths,
or text. Each renderer corresponds to a `Mode` (LABELS, VISUAL, ANNOTATED, etc.)
and is responsible for producing the SVG content for that mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from img2svg.loader import LoadedImage
from img2svg.models import Detection, GeometricAnalysis
from img2svg.svg_builder import SVGDocument


class Renderer(ABC):
    """Abstract base class for SVG renderers.

    Subclasses must implement `render()` which mutates `self.svg` in place by
    adding groups/rects/text. The renderer does NOT write to disk; that is
    handled by the pipeline via `SVGDocument.write()`.
    """

    def __init__(
        self,
        svg: SVGDocument,
        image: LoadedImage,
        detections: list[Detection],
        geometric: GeometricAnalysis | None,
    ) -> None:
        """Store render context. Subclasses should call `super().__init__(...)`.

        Args:
            svg: The SVG document to mutate. Will be modified in place.
            image: The loaded source image (provides width/height).
            detections: List of YOLO detections to render. May be empty.
            geometric: Optional geometric analysis (palette, edge density, etc.).
        """
        self.svg = svg
        self.image = image
        self.detections = detections
        self.geometric = geometric

    @abstractmethod
    def render(self) -> None:
        """Mutate `self.svg` in place. Must be implemented by subclasses."""
        raise NotImplementedError
