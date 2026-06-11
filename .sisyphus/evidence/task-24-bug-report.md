# T24 BUG Report — SegmentedRenderer never receives SegmentationResult

**Date:** 2026-06-11
**Severity:** HIGH (blocks T24 acceptance criterion: multi-layer SEGMENTED output)
**Status:** OPEN — production code not modified per T24 "verification only" constraint

## Symptom

`uv run img2svg convert tests/testimg/Designer\ \(1\).jpeg --mode segmented --seg-model yolo11s-seg`
produces a single `<g id="vtracer-output">` group instead of the expected
multi-layer `<g id="background">` + `<g id="obj_...">` structure.

## Evidence

- `task-24-seg-photo.svg` (8,036,185 bytes) — only `<g id="vtracer-output">` at top level
- `task-24-seg-photo.json` — sidecar has `regions: 1` (person detected, area_pixels=238358, polygon length=1253)
- `task-24-verification-report.txt` — full verification output showing the failure

## Root cause

`src/img2svg/pipeline.py` line 305-338 runs YOLO segmentation and stores the
result on `self._segmentation_result` (line 338). But when constructing the
renderer at line 362:

```python
renderer = renderer_cls(svg, loaded, detections, analysis_global)
```

The pipeline NEVER injects the segmentation result into the renderer. There is
no `isinstance(renderer, SegmentedRenderer)` check, no
`renderer.set_segmentation(segmentation_result)` call.

`SegmentedRenderer.__init__` (renderers/segmented.py:138) initializes
`self._segmentation_result = None`. With no injection, the renderer's
`_has_any_region(result)` (line 45-55) returns False (None check), and the
renderer falls back to `_render_with_vtracer(self, "default")` (line 174) —
which produces the same `<g id="vtracer-output">` as `VisualRenderer`.

## Why this passes T18's fallback check

T18's fallback (pipeline.py:348-359) checks
`not (segmentation_result and segmentation_result.masks)`. When the YOLO
segmentor finds 1 region, `masks` is a non-empty list, so the condition is
False and the fallback is skipped. But the renderer created at line 362
(after the fallback check) is `SegmentedRenderer`, and IT has no idea about
the segmentation result because the pipeline never told it.

So the pipeline does its job (segment + sidecar + check fallback), but the
renderer falls back internally because of the missing `set_segmentation` call.

## Fix (NOT applied per T24 spec)

Add to `src/img2svg/pipeline.py` after line 362:

```python
if isinstance(renderer, SegmentedRenderer) and segmentation_result is not None:
    renderer.set_segmentation(segmentation_result)
```

Plus import `SegmentedRenderer` in the TYPE_CHECKING block (or as a runtime
import at the top of the file).

## Affected evidence

- `.sisyphus/evidence/task-24-seg-photo.svg` — broken (no multi-layer)
- All T24 acceptance criteria related to multi-layer SEGMENTED output FAIL
- The 9 non-SEGMENTED modes pass on rtlogo-1.png
- The SEGMENTED-on-logo case still produces vtracer-output via T18's
  pipeline-level fallback (correct per spec)
- The SEGMENTED-on-photo case is the one that should produce multi-layer but doesn't

## T15/T16/T10 integration gap

Per the T16 notepad (line 753-755):
> T16 (SegmentedRenderer) is the consumer of `self._segmentation_result`.
> When it lands, it will read the cached result and emit multi-layer SVG.

T16 was implemented with `set_segmentation` setter, but the pipeline (T15)
never wires the result through. T10 follow-up (the "wire SegmentedRenderer"
entry, line 915-933) only added the registry entry, not the set_segmentation
call. This is an integration gap that T24 surfaced.

The T16 unit tests in `tests/test_pipeline.py` likely use the stub pattern
from T15 (line 743-749 of the notepad) which sets `RENDERER_REGISTRY` to a
stub renderer that doesn't need the result. So the integration gap is hidden
from the unit tests.
