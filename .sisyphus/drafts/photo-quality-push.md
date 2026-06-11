# Draft: photo-quality-push

> **Working memory for the photo-quality push plan.** Continuously updated during interview phase. Will be deleted after the plan is generated.

## Original Request
"now, back in plan mode, can we make it work harder for real world photos? SVG is meant for more simple images, but hey.. i want to give it a shot"

## User Decisions (confirmed)

| Decision | Choice | Notes |
|---|---|---|
| Plan scope | **All of the above (XL)** | Pre-processing + new modes + YOLO segmentation + CLI flags. ~15-25 tasks. |
| Defaults | **Aggressive** | All photo modes (auto/trace/visual/annotated) get default preprocessing unless `--no-preprocess`. SEGMENTED auto-selects yolo11x-seg for PHOTO images. Maximum quality, larger files, slower. |
| Segmented output format | **Multi-layer editable SVG** | Each detected object becomes its own `<g id="obj_person_0">` group. Background traced separately. Easy to edit in Inkscape/Illustrator/Figma. |
| **Auto-mode behavior** | **NEW** | "Annotated mode should not be included in auto when rendering, annotated must be explicitly requested, as well as label" — LABELS and ANNOTATED are EXPLICIT only. Auto mode picks from {VISUAL, TRACE, POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED} only. |

## Key Constraints (verified during interview)
- BSD 3-Clause, **`Copyright (c) 2026, REVYTECH, Inc.`**
- Existing 5 modes: AUTO, LABELS, VISUAL, ANNOTATED, TRACE
- Existing 5 vtracer presets: default, bw, logo, poster, photo (in `vectorizer.py:PRESETS`)
- Existing RENDERER_REGISTRY: `Mode.LABELS → LabelsRenderer`, `Mode.VISUAL → VisualRenderer`, `Mode.ANNOTATED → AnnotatedRenderer`, `Mode.TRACE → TraceRenderer`
- Existing MODE_TO_PRESET: only 4 mapped (no `bw` or `poster` mapping)
- Existing IMAGE_TYPE_TO_MODE: `{LOGO: LABELS, PHOTO: ANNOTATED, DIAGRAM: LABELS, SCREENSHOT: VISUAL, LINE_ART: LABELS, UNKNOWN: ANNOTATED}` — **MUST CHANGE** per user constraint
- Tests mock `img2svg.pipeline.get_detector` and `img2svg.renderers.visual.VtracerVectorizer`
- No test_renderers.py exists — renderers tested indirectly via `test_pipeline.py`
- Public API: `convert()`, `convert_batch()` (in `api.py`)
- Loader intentionally preserves alpha; normalization happens at the model boundary (in `detector.py:detect()`)
- YOLO model cache key: `f"{model_name}::{backend.requested}::{backend.index}"`
- Tests use `tests/conftest.py` fixtures: `logo_path`, `photo_path`, `diagram_path`, `line_art_path`, `transparent_path`, `screenshot_path`, `corrupt_path`
- Working tree clean. HEAD = current. 474 fast tests pass (1 pre-existing i18n failure, unrelated). 90% coverage.

## Research Findings (3 librarian reports)

### vtracer (librarian bg_5ddfd33b, 5m 0s)
- **3 Python API functions**: `convert_image_to_svg_py` (file→file), `convert_raw_image_to_svg` (bytes→string), `convert_pixels_to_svg` (pixel array→string)
- **11 kwargs** with defaults and ranges; **vtracer does NOT validate ranges in Python** (only CLI does) — must validate in our wrapper
- **Presets are NOT exposed in Python API** — must manually replicate kwargs
- **`path_precision` default is 2** (not 8 as .pyi says — documentation bug)
- **CLI "pixel" mode = Python "none" mode** (naming asymmetry)
- vtracer `photo` preset differs from existing `photo` in `vectorizer.py`:
  - vtracer official: `color_precision=8, filter_speckle=10, layer_difference=48, corner_threshold=180`
  - existing: `color_precision=6, filter_speckle=2, layer_difference=8, corner_threshold=60`
- License: MIT (vtracer)
- No 4K size limit but 4K+ can produce 100MB+ SVG output
- 5 OSS production patterns studied: sserada/image-to-svg, vtracer_autotune, vectalab, etjones/vtracer_py
- `MAX_SVG_SIZE` guard recommended (50MB cap)
- Multi-pass strategies: coarse background + detailed foreground, or vtracer twice with different params

### YOLO11 Instance Segmentation (librarian bg_1fc486ff, 4m 57s)
- **5 model variants** (n/s/m/l/x) — recommend `yolo11s-seg` default, `yolo11m-seg` for high quality
- `result.masks.data` = `(N, H, W) uint8` binary tensor
- `result.masks.xy` = list of `(P, 2)` pixel polygons
- `result.masks.xyn` = normalized polygons (size-stable for sidecar)
- `retina_masks=True` is **CRITICAL** for getting masks at original image size (not imgsz)
- `masks.gt_(0.0).byte()` (no sigmoid, threshold 0.0)
- `cv2.RETR_EXTERNAL` returns only outer contours (no holes)
- 4K feasibility: yolo11s-seg at imgsz=1280, retina_masks=True, FP16 — feasible on 6-10 GB GPU
- License: AGPL-3.0 with Enterprise alternative
- **License risk: img2svg is BSD-3-Clause, but YOLO/ultralytics is AGPL-3.0. Already covered by existing NOTICE. New YOLO-seg usage is same risk profile.**

### OpenCV Pre-processing (librarian bg_c07a9001, 3m 54s)
- **`bilateralFilter` is available in headless build** (in `imgproc` module, source code confirmed)
- **Pipeline order**: denoise → contrast → sharpen → posterize → vtracer
- **Default for photos**: `bilateral(d=5, σ=50)` → `unsharp(σ=2, amount=0.5)` (~0.5s on 1080p)
- **Edge mode**: `medianBlur(3)` → `Canny(80, 180)` → vtracer with `colormode='binary'`
- **Poster mode**: `bilateral(σ=60)` → `unsharp(amount=0.3)` → `posterize(bits=3)` → vtracer `color_precision=3`
- **Use `cv2.addWeighted` for sharpening** (deforum pattern, production-quality)
- **Skip CLAHE by default** (risk of amplifying noise)
- **Use numpy bit-shift for posterize** (not PIL)
- **vtracer's `color_precision` subsumes explicit k-means** (use vtracer's parameter unless palette-specific)
- Performance: bilateral is the bottleneck (~2.3s on 1080p). 4K = 4-5s.
- 4 OSS examples: sohail000/img2vector (uses `preprocessing_level` none/light/medium/heavy), nicewang/bitmap2svg, lichgu/img2svg, yuism23/PictureVectorization

## Architecture Decisions

### New Modes (final list — 5 new modes)
| Mode | Preset | Pre-processing | YOLO | Notes |
|---|---|---|---|---|
| `POSTER` | `poster` (existing) | bilateral(σ=60) → unsharp(0.3) → posterize(3 bits) | detect only | Stylized, limited colors |
| `DETAILED` | new `photo_hifi` | bilateral(σ=50) → unsharp(0.5) | detect only | Aggressive: pre-process + high-fidelity trace |
| `EDGE` | new `bw_edge` | medianBlur(3) → Canny(80,180) | detect only | Line-art look, transparent fill |
| `WATERCOLOR` | new `watercolor` | bilateral(σ=80) → CLAHE → unsharp(0.2) | detect only | Soft organic curves |
| `SEGMENTED` | varies per region | bilateral only (whole image) | **seg required** (yolo11s-seg) | Per-region vtracer, multi-layer editable SVG |

### New vtracer Presets (in `vectorizer.py`)
```python
PRESETS = {
    "default":  { ... existing ... },
    "bw":       { ... existing ... },
    "logo":     { ... existing ... },
    "poster":   { ... existing ... },
    "photo":    { ... existing ... },
    # NEW
    "photo_hifi": {  # DETAILED mode
        "colormode": "color", "hierarchical": "stacked", "mode": "spline",
        "filter_speckle": 4, "color_precision": 8, "layer_difference": 24,
        "corner_threshold": 60, "length_threshold": 3.5,
        "max_iterations": 20, "splice_threshold": 30, "path_precision": 4,
    },
    "bw_edge": {  # EDGE mode
        "colormode": "binary", "hierarchical": "stacked", "mode": "polygon",
        "filter_speckle": 8, "color_precision": 6, "layer_difference": 16,
        "corner_threshold": 120, "length_threshold": 5.0,
        "max_iterations": 5, "splice_threshold": 60, "path_precision": 2,
    },
    "watercolor": {  # WATERCOLOR mode
        "colormode": "color", "hierarchical": "stacked", "mode": "spline",
        "filter_speckle": 14, "color_precision": 7, "layer_difference": 32,
        "corner_threshold": 20, "length_threshold": 5.0,
        "max_iterations": 15, "splice_threshold": 20, "path_precision": 3,
    },
}
```

### New IMAGE_TYPE_TO_MODE (per user constraint)
```python
# Auto mode: NEVER picks LABELS or ANNOTATED — those are explicit only.
IMAGE_TYPE_TO_MODE: dict[ImageType, Mode] = {
    ImageType.LOGO:       Mode.VISUAL,    # Clean default; no detection overlay
    ImageType.PHOTO:      Mode.DETAILED,  # Aggressive: pre-process + high-fidelity
    ImageType.DIAGRAM:    Mode.VISUAL,
    ImageType.SCREENSHOT: Mode.VISUAL,
    ImageType.LINE_ART:   Mode.VISUAL,    # Could be EDGE, but VISUAL is safer default
    ImageType.UNKNOWN:    Mode.VISUAL,
}
```

### New MODE_TO_PRESET
```python
MODE_TO_PRESET: dict[Mode, Preset] = {
    Mode.LABELS:    "logo",
    Mode.VISUAL:    "default",
    Mode.ANNOTATED: "default",
    Mode.TRACE:     "photo",
    # NEW
    Mode.POSTER:    "poster",
    Mode.DETAILED:  "photo_hifi",
    Mode.EDGE:      "bw_edge",
    Mode.WATERCOLOR: "watercolor",
    Mode.SEGMENTED: "default",  # Per-region uses photo_hifi; whole image uses default
}
```

### Sidecar additions
- `preprocessing: list[str]` — list of preprocessing steps applied (e.g., `["denoise_bilateral", "unsharp", "posterize"]`)
- `regions: list[RegionInfo]` — for SEGMENTED mode, per-region metadata (class, bbox, polygon, area_pixels, vtracer params used)
- `model_variant: str` — e.g., `yolo11s-seg` (extends existing `model` field, or new field)

### New CLI flags
- `--preprocess` (`none|light|medium|heavy|all`) — pre-processing level. Default: `light` for new modes. Default: `none` for existing modes (backward compat).
- `--denoise` (int 0-10) — bilateral sigma. Default: 50 if preprocess on, else 0.
- `--sharpen` (int 0-10) — unsharp amount. Default: 0.5 if preprocess on, else 0.
- `--max-colors` (int 2-64) — caps color count via PIL `quantize(colors=N)` before vtracer. Default: 0 (no cap).
- `--seg-model` (`yolo11n-seg|yolo11s-seg|yolo11m-seg`) — segmentation model for SEGMENTED mode. Default: `yolo11s-seg`.
- `--no-preprocess` — opt-out of pre-processing. Implicit when `--preprocess none`.
- `--quality` (`draft|standard|premium`) — maps to filter_speckle and length_threshold inversely. Default: `standard`.
- `--no-seg` — opt-out of YOLO segmentation (use bbox detection only).

### Module structure (new + modified)
- **NEW** `src/img2svg/preprocessing.py` — composable filter functions
- **MOD** `src/img2svg/vectorizer.py` — add 3 new presets
- **MOD** `src/img2svg/enums.py` — add 5 new Mode values
- **MOD** `src/img2svg/presets.py` — new IMAGE_TYPE_TO_MODE, MODE_TO_PRESET
- **MOD** `src/img2svg/models.py` — Sidecar: add `preprocessing`, `regions` fields
- **MOD** `src/img2svg/cli.py` — add new flags, update help text
- **MOD** `src/img2svg/pipeline.py` — wire preprocessing + segmentation
- **MOD** `src/img2svg/detector.py` — add seg model support, mask extraction
- **NEW** `src/img2svg/renderers/poster.py` — PosterRenderer
- **NEW** `src/img2svg/renderers/detailed.py` — DetailedRenderer
- **NEW** `src/img2svg/renderers/edge.py` — EdgeRenderer
- **NEW** `src/img2svg/renderers/watercolor.py` — WatercolorRenderer
- **NEW** `src/img2svg/renderers/segmented.py` — SegmentedRenderer (multi-layer)
- **NEW** `tests/test_preprocessing.py` — preprocessing tests
- **NEW** `tests/test_segmentation.py` — segmentation tests
- **MOD** `tests/test_presets.py` — new modes covered
- **MOD** `tests/test_pipeline.py` — new renderers in registry
- **MOD** `tests/test_cli.py` — new flags accepted
- **MOD** `docs/photo-modes.md` — new documentation
- **MOD** `README.md` — update mode table
- **MOD** `man/img2svg.1` — update man page

## Open Questions (resolved)
- ~~License: AGPL-3.0 risk?~~ — Already covered by existing NOTICE for YOLO. YOLO-seg has same risk profile. No new license issue.
- ~~Defaults: aggressive vs balanced vs conservative?~~ — User picked Aggressive.
- ~~Output format for SEGMENTED?~~ — User picked Multi-layer editable SVG.
- ~~ANNOTATED/LABELS in auto-mode?~~ — User said NO, explicit only.
- ~~Existing bw/poster presets — wire to modes?~~ — Yes, `poster` → POSTER mode, `bw_edge` (new) → EDGE mode.

## Open Questions (remaining)
- None blocking. Plan is ready to generate.

## Plan Outline (25 tasks, 6 waves + final verification)

**Wave 1 — Foundation (5 tasks, parallel)**
- T1: Create `preprocessing.py` with composable OpenCV filters
- T2: Extend `vectorizer.py` with new vtracer presets (photo_hifi, bw_edge, watercolor)
- T3: Extend `enums.py` Mode with 5 new values (POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
- T4: Extend `models.py` Sidecar with `preprocessing`, `regions` fields
- T5: Add CLI flags (`--preprocess`, `--denoise`, `--sharpen`, `--max-colors`, `--quality`, `--no-preprocess`)

**Wave 2 — New Renderers (5 tasks, parallel)**
- T6: PosterRenderer
- T7: DetailedRenderer
- T8: EdgeRenderer
- T9: WatercolorRenderer
- T10: Wire new renderers to RENDERER_REGISTRY, MODE_TO_PRESET, IMAGE_TYPE_TO_MODE

**Wave 3 — YOLO Segmentation (5 tasks, parallel-ish)**
- T11: Add seg model support to `detector.py` (yolo11s-seg, retina_masks=True)
- T12: Add mask extraction (masks.xy → polygons)
- T13: Add per-region vtracer tracing (crop, trace, embed with offset)
- T14: Add CLI flags (`--seg-model`, `--no-seg`)
- T15: Update Sidecar with `regions` field

**Wave 4 — Segmented Renderer (2 tasks, parallel)**
- T16: SegmentedRenderer (multi-layer editable SVG, each region as `<g id="obj_...">`)
- T17: Pipeline integration (wire segmentation step into pipeline.run)

**Wave 5 — Pipeline + Preprocessing Integration (3 tasks, parallel)**
- T18: Wire preprocessing into pipeline.run (post-load, pre-vtracer)
- T19: Add timing entries for preprocessing, segmentation
- T20: Add max output size guard (MAX_SVG_SIZE = 50MB)

**Wave 6 — Tests + Documentation (5 tasks, parallel)**
- T21: Tests for new renderers
- T22: Tests for preprocessing
- T23: Tests for segmentation
- T24: End-to-end test on real photos (user's rtlogo + sample photo)
- T25: Update README + new docs/photo-modes.md + man page

**Wave FINAL — Verification (4 tasks, parallel)**
- F1: Plan compliance audit
- F2: Code quality review
- F3: Real manual QA (Playwright + tmux + curl)
- F4: Scope fidelity check

## Critical Context for Executor
- Working tree is clean; user expects commit+push without confirmation
- All CLI changes must update help text and the `_mode_callback` validator
- All new renderers must follow `_render_with_vtracer` pattern from `visual.py`
- All new tests follow existing pattern: mock `img2svg.pipeline.get_detector`, mock `img2svg.renderers.visual.VtracerVectorizer`
- Use `conftest.py` fixtures: `logo_path`, `photo_path`
- Sidecar backward compat: new fields are optional with defaults
- Existing tests must continue to pass (474 fast tests)
- Test on user's `/home/mlapointe/Documents/rtlogo-1.png` for end-to-end verification
- 90% coverage must be maintained

## Decision Log
- "All of the above (XL)" chosen for plan scope
- Aggressive defaults chosen
- Multi-layer editable SVG chosen for SEGMENTED output
- LABELS and ANNOTATED are explicit-only in auto-mode (per user)
- Default PHOTO → DETAILED in IMAGE_TYPE_TO_MODE (aggressive)
- Default SEGMENTED model: yolo11s-seg (CPU-feasible + fast on GPU)
- Default segmentation pre-processing: bilateral only (whole image)
- Pre-processing order: denoise → sharpen → posterize → vtracer
- Sharpening: `cv2.addWeighted` (deforum pattern)
- Skip CLAHE by default (opt-in via `--clahe` flag, deferred to v2)
- Use vtracer's `color_precision` instead of explicit k-means
- Use numpy bit-shift for posterize (not PIL)
- 5 new modes total (POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
- 3 new vtracer presets (photo_hifi, bw_edge, watercolor) — wire existing `bw` and `poster` to modes
- Multi-layer SVG: each region as `<g id="obj_class_id">` with class name attribute
- Background: traced separately with simple params
- License: AGPL-3.0 risk already covered by existing NOTICE. YOLO-seg has same risk profile.
- Test strategy: TDD with mocks + real photos at the end
- Quality metrics: deferred to v2 (too complex for v1)
- Tile-based parallel tracing: deferred to v2
- Style transfer: not in scope
- Depth estimation: not in scope

## What Comes Next
- Generate `.sisyphus/plans/photo-quality-push.md`
- Use Write (skeleton) + Edits (tasks in batches) per incremental write protocol
- Run Metis gap analysis before final plan
- Ask user for go-ahead to start work after plan is generated
