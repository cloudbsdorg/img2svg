# img2svg - pipeline orchestrator for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Pipeline orchestrator for img2svg.

Composes the full conversion flow:

    load_image → analyze_global → classify → select_mode → detect
    → SVGDocument → renderer.render → write SVG + sidecar JSON

This is the heart of the library. The high-level public API in
`img2svg.api` is a thin wrapper around `Pipeline`.

`RENDERER_REGISTRY` is a `dict[Mode, type[Renderer]]` used by the
pipeline to look up the renderer class for a resolved mode. `Mode.AUTO`
is deliberately absent — the pipeline calls `select_mode()` first to
resolve AUTO to a concrete mode, then looks up the renderer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as _PILImage

from img2svg.classifier import classify
from img2svg.detector import (
    compute_mask_area,
    get_detector,
    get_segmentor,
    get_tight_bbox,
)
from img2svg.enums import Mode
from img2svg.errors import OutputPathCollisionError
from img2svg.loader import load_image
from img2svg.logging import get_logger
from img2svg.metadata import compute_file_hash, write_sidecar
from img2svg.models import (
    BackendSpec,
    BoundingBox,
    ConversionOptions,
    ConversionResult,
    RegionInfo,
    Sidecar,
)
from img2svg.patterns import analyze_global
from img2svg.preprocessing import PREPROCESSING_PRESETS, PreprocessingPipeline
from img2svg.presets import select_mode
from img2svg.renderers.annotated import AnnotatedRenderer
from img2svg.renderers.base import Renderer
from img2svg.renderers.detailed import DetailedRenderer
from img2svg.renderers.edge import EdgeRenderer
from img2svg.renderers.labels import LabelsRenderer
from img2svg.renderers.poster import PosterRenderer
from img2svg.renderers.segmented import SegmentedRenderer
from img2svg.renderers.trace import TraceRenderer
from img2svg.renderers.visual import VisualRenderer
from img2svg.renderers.watercolor import WatercolorRenderer
from img2svg.svg_builder import SVGDocument

if TYPE_CHECKING:
    from img2svg.detector import SegmentationResult
    from img2svg.models import Detection, GeometricAnalysis, LoadedImage  # noqa: F401

_logger = get_logger("img2svg.pipeline")

# Sidecar's `version` field. Kept in sync with `pyproject.toml` `version`.
# Inlined as a constant so the pipeline has no pyproject.toml parsing dep.
_VERSION: str = "0.1.0"

# Map a resolved `Mode` to the corresponding renderer class. `Mode.AUTO` is
# intentionally absent — the pipeline must resolve AUTO via `select_mode()`
# before looking up a renderer. `Mode.SEGMENTED` is also absent — its
# `SegmentedRenderer` lands in T16.
RENDERER_REGISTRY: dict[Mode, type[Renderer]] = {
    Mode.LABELS: LabelsRenderer,
    Mode.VISUAL: VisualRenderer,
    Mode.ANNOTATED: AnnotatedRenderer,
    Mode.TRACE: TraceRenderer,
    Mode.POSTER: PosterRenderer,
    Mode.DETAILED: DetailedRenderer,
    Mode.EDGE: EdgeRenderer,
    Mode.WATERCOLOR: WatercolorRenderer,
    Mode.SEGMENTED: SegmentedRenderer,
}


def _format_backend_requested(spec: BackendSpec) -> str:
    """Render a `BackendSpec` as the ultralytics-style device string.

    Produces ``"cuda:0"`` for indexed CUDA/ROCm backends and the bare
    backend name (``"cpu"``, ``"mps"``, ``"auto"``) otherwise. This is
    the canonical "what the user asked for" form, suitable for both the
    detector cache key and the ``sidecar.backend_requested`` field.
    """
    if spec.index is not None:
        return f"{spec.requested}:{spec.index}"
    return spec.requested


# Map a user-facing short alias (e.g. ``"bilateral"`` from ``--preprocess``)
# to the full preprocessing function name plus its default kwargs. The short
# aliases match what the CLI lists in ``--preprocess``'s help text. Including
# the unaliased full names (e.g. ``"denoise_bilateral"``) keeps the mapping
# robust to programmatic callers that already use the full names.
_PREPROCESS_ALIASES: dict[str, tuple[str, dict[str, object]]] = {
    "bilateral": ("denoise_bilateral", {"d": 5, "sigma": 50}),
    "denoise_bilateral": ("denoise_bilateral", {"d": 5, "sigma": 50}),
    "nlmeans": ("denoise_nlmeans", {"h": 6}),
    "denoise_nlmeans": ("denoise_nlmeans", {"h": 6}),
    "median": ("denoise_median", {"k": 3}),
    "denoise_median": ("denoise_median", {"k": 3}),
    "unsharp": ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
    "sharpen_unsharp": ("sharpen_unsharp", {"sigma": 2.0, "amount": 0.5}),
    "posterize": ("posterize", {"bits": 4}),
    "canny": ("detect_edges_canny", {"low": 80, "high": 180}),
    "detect_edges_canny": ("detect_edges_canny", {"low": 80, "high": 180}),
    "clahe": ("apply_clahe_yuv", {"clip": 2.0, "tile": (8, 8)}),
    "apply_clahe_yuv": ("apply_clahe_yuv", {"clip": 2.0, "tile": (8, 8)}),
}
_PREPROCESS_SHORT_NAMES: tuple[str, ...] = (
    "bilateral",
    "nlmeans",
    "median",
    "unsharp",
    "posterize",
    "canny",
    "clahe",
)

# Modes whose auto-preprocessing default is the ``light`` photo preset.
# Preprocessing on line-art / logos / diagrams tends to soften edges that
# vtracer depends on, so we only enable it for the explicitly photo-aware
# renderers.
_PHOTO_MODES: frozenset[Mode] = frozenset({Mode.DETAILED, Mode.WATERCOLOR, Mode.SEGMENTED})


def _format_preprocessing_label(name: str, kwargs: dict[str, object]) -> str:
    """Render a filter step as ``"alias(k=v, k=v)"`` for the sidecar.

    Uses the short alias when one is registered so the sidecar output
    matches the user's CLI form (``"bilateral(d=5, sigma=50)"``) rather
    than the verbose function name (``"denoise_bilateral(d=5, sigma=50)"``).
    """
    short = name
    for alias, (full, _) in _PREPROCESS_ALIASES.items():
        if full == name and alias != full:
            short = alias
            break
    if not kwargs:
        return short
    parts = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{short}({parts})"


def _resolve_preprocessing_steps(
    options: ConversionOptions, mode_used: Mode
) -> list[tuple[str, dict[str, object]]]:
    """Pick the list of ``(filter_name, kwargs)`` steps to apply.

    Decision tree (from T17 spec):
      1. ``options.no_preprocess`` → skip (overrides everything).
      2. ``options.preprocess`` non-empty → resolve each alias to its
         full function name and default kwargs.
      3. ``mode_used`` in the photo-mode set → use the ``light`` preset.
      4. Otherwise → skip (logo / diagram / etc. don't need it).
    """
    if options.no_preprocess:
        return []
    if options.preprocess:
        steps: list[tuple[str, dict[str, object]]] = []
        for alias in options.preprocess:
            if alias not in _PREPROCESS_ALIASES:
                raise ValueError(
                    f"Unknown preprocessing filter '{alias}'. "
                    f"Available: {list(_PREPROCESS_SHORT_NAMES)}"
                )
            full_name, default_kwargs = _PREPROCESS_ALIASES[alias]
            steps.append((full_name, dict(default_kwargs)))
        return steps
    if mode_used in _PHOTO_MODES:
        return list(PREPROCESSING_PRESETS["light"])
    return []


class Pipeline:
    """Orchestrate a single image → SVG conversion.

    The pipeline owns no heavy resources — the YOLO detector is loaded on
    first `.run()` and cached via the module-level `get_detector()`
    singleton. Constructing a `Pipeline` is cheap; pass the same instance
    to multiple `.run()` calls in a batch to amortize setup.

    Args:
        options: User-facing conversion options (mode, model, backend, ...).
    """

    def __init__(self, options: ConversionOptions) -> None:
        """Store options. The YOLO detector is loaded lazily on first `.run()`."""
        self.options = options
        self._segmentation_result: SegmentationResult | None = None

    def run(self, input_path: Path, output_path: Path) -> ConversionResult:
        """Run the full conversion pipeline on a single image.

        Steps (matches plan T19):
          1. Load image (raises `UnsupportedFormatError` / `CorruptImageError`).
          2. Compute global geometric analysis.
          3. Classify the image (LOGO / PHOTO / DIAGRAM / ...).
          4. Resolve the mode (user override or AUTO).
          5. Run YOLO detection.
          6. (Per-ROI analysis — skipped in this critical-path version.)
          7. Build the empty `SVGDocument`.
          8. Construct the renderer for the resolved mode and render in place.
          9. Write the SVG to disk.
         10. Build the `Sidecar` with all metadata.
         11. Write the sidecar JSON next to the SVG.
         12. Return the `ConversionResult`.

        Args:
            input_path: Source image path.
            output_path: Destination `.svg` path.

        Returns:
            `ConversionResult` with paths, sidecar, and detections.

        Raises:
            img2svg.errors.UnsupportedFormatError: Input format is not in
                the supported set.
            img2svg.errors.CorruptImageError: Pillow could not decode the input.
            img2svg.errors.ModelLoadError: YOLO could not be loaded.
            img2svg.errors.OutputPathCollisionError: `options.no_clobber` is
                `True` and `output_path` already exists.
        """
        options = self.options
        timings: dict[str, float] = {}
        t_total_start = time.perf_counter()

        # 1. Load
        t0 = time.perf_counter()
        loaded = load_image(input_path)
        timings["load"] = time.perf_counter() - t0

        # 1a. No-clobber guard: fail fast before running YOLO when the output
        #     path already exists and the user has asked not to overwrite.
        if options.no_clobber and output_path.exists():
            raise OutputPathCollisionError(str(output_path))

        # 1b. Preprocessing (T17). Determine the effective mode first so the
        #     mode-driven default (photo modes get the ``light`` preset) can
        #     be applied. When ``options.mode`` is set explicitly, trust it
        #     and skip the classifier round-trip; otherwise do a quick
        #     ``classify`` so AUTO resolves correctly.
        t0 = time.perf_counter()
        if options.mode == Mode.AUTO:
            _quick_image_type, _ = classify(loaded.np_array)
            _quick_mode_used, _ = select_mode(_quick_image_type, options.mode)
        else:
            _quick_mode_used = options.mode
        preprocessing_steps = _resolve_preprocessing_steps(options, _quick_mode_used)
        sidecar_preprocessing: list[str] = []
        if preprocessing_steps:
            _preprocessor = PreprocessingPipeline(preprocessing_steps)
            _preprocessed_array = _preprocessor.apply(loaded.np_array)
            loaded.pil_image = _PILImage.fromarray(_preprocessed_array)
            sidecar_preprocessing = [
                _format_preprocessing_label(name, kwargs)
                for name, kwargs in preprocessing_steps
            ]
        timings["preprocess"] = time.perf_counter() - t0

        # 2. Global geometric analysis
        t0 = time.perf_counter()
        analysis_global = analyze_global(loaded.np_array)
        timings["analyze"] = time.perf_counter() - t0

        # 3. Classify
        t0 = time.perf_counter()
        image_type, _classify_reasoning = classify(loaded.np_array, analysis_global)
        timings["classify"] = time.perf_counter() - t0

        # 4. Resolve mode (user override or AUTO)
        t0 = time.perf_counter()
        mode_used, mode_reasoning = select_mode(image_type, options.mode)
        timings["select_mode"] = time.perf_counter() - t0

        # 5. YOLO detection
        t0 = time.perf_counter()
        detector = get_detector(model_name=options.model, backend=options.backend)
        detections: list[Detection] = detector.detect(
            loaded.np_array, conf=options.conf, iou=options.iou
        )
        timings["detect"] = time.perf_counter() - t0
        backend_requested = _format_backend_requested(options.backend)
        backend_resolved = detector.device

        # 6. (Per-ROI analysis skipped for the critical path.)

        # 6b. YOLO segmentation (SEGMENTED mode only).
        t0 = time.perf_counter()
        segmentation_result: SegmentationResult | None = None
        region_infos: list[RegionInfo] = []
        model_variant = ""
        if mode_used == Mode.SEGMENTED and not options.no_seg:
            segmentor = get_segmentor(
                model_name=options.seg_model, backend=options.backend
            )
            segmentation_result = segmentor.predict(
                loaded.np_array, conf=options.conf, iou=options.iou
            )
            model_variant = options.seg_model
            for i, det in enumerate(segmentation_result.boxes):
                mask = segmentation_result.masks[i]
                poly = segmentation_result.polygons[i]
                mask_x1, mask_y1, mask_x2, mask_y2 = get_tight_bbox(mask)
                region_infos.append(
                    RegionInfo(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=BoundingBox(
                            x1=float(mask_x1),
                            y1=float(mask_y1),
                            x2=float(mask_x2),
                            y2=float(mask_y2),
                        ),
                        area_pixels=compute_mask_area(mask),
                        polygon=[(float(x), float(y)) for x, y in poly],
                        mask_path=None,
                    )
                )
        self._segmentation_result = segmentation_result
        timings["segment"] = time.perf_counter() - t0

        # 7. Build the empty SVG document.
        svg = SVGDocument(loaded.width, loaded.height, title=input_path.name)

        # 8. Look up the renderer and render in place.
        t0 = time.perf_counter()
        renderer_cls = RENDERER_REGISTRY[mode_used]
        renderer = renderer_cls(svg, loaded, detections, analysis_global)
        renderer.render()
        timings["render"] = time.perf_counter() - t0

        # 9. Write the SVG to disk.
        t0 = time.perf_counter()
        svg.write(output_path)
        timings["write"] = time.perf_counter() - t0

        # 10. Record total BEFORE building the sidecar so timings["total"]
        #     is part of what gets serialized.
        timings["total"] = time.perf_counter() - t_total_start

        # 11. Build the sidecar.
        input_hash = compute_file_hash(input_path)
        output_size = output_path.stat().st_size
        sidecar = Sidecar(
            version=_VERSION,
            input_path=input_path,
            input_hash=input_hash,
            output_path=output_path,
            output_size=output_size,
            mode_used=mode_used,
            mode_reasoning=mode_reasoning,
            model=options.model,
            device=backend_resolved,
            backend_requested=backend_requested,
            backend_resolved=backend_resolved,
            image_type=image_type,
            detections=detections,
            geometric=analysis_global,
            timings=timings,
            preprocessing=sidecar_preprocessing,
            regions=region_infos,
            model_variant=model_variant,
        )

        # 12. Write the sidecar JSON next to the SVG.
        sidecar_path = output_path.with_suffix(".json")
        write_sidecar(sidecar, sidecar_path)

        _logger.debug(
            "pipeline.run(%s) -> %s (mode=%s, detections=%d, total=%.3fs)",
            input_path.name,
            output_path.name,
            mode_used,
            len(detections),
            timings["total"],
        )

        return ConversionResult(
            svg_path=output_path,
            sidecar_path=sidecar_path,
            sidecar=sidecar,
            detections=detections,
        )
