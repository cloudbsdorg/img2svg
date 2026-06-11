# Plan: Photo Quality Push (img2svg)

## TL;DR

> **Quick Summary**: Add 5 new output modes (`POSTER`, `DETAILED`, `EDGE`, `WATERCOLOR`, `SEGMENTED`) to img2svg that leverage OpenCV pre-processing + YOLO11 instance segmentation + tuned vtracer presets to make real-world photos convert to high-quality, multi-layer, editable SVGs. Auto-mode never picks `LABELS` or `ANNOTATED` (those are explicit only).
>
> **Deliverables**:
> - New `preprocessing.py` module with composable OpenCV filters
> - 3 new vtracer presets (`photo_hifi`, `bw_edge`, `watercolor`); wire existing `bw` and `poster` presets to new modes
> - 5 new `Mode` enum values + 5 new renderers
> - YOLO11 instance segmentation support (yolo11s-seg default, yolo11m-seg high-quality)
> - Multi-layer editable SVG output for `SEGMENTED` mode (one `<g id="obj_class_id">` per detected object)
> - 7 new CLI flags: `--preprocess`, `--denoise`, `--sharpen`, `--max-colors`, `--quality`, `--no-preprocess`, `--seg-model`, `--no-seg`
> - Auto-mode mapping updated: `PHOTO → DETAILED` (aggressive), `LOGO/DIAGRAM/SCREENSHOT/LINE_ART/UNKNOWN → VISUAL`
> - Sidecar metadata extended: `preprocessing: list[str]`, `regions: list[RegionInfo]`
> - New `docs/photo-modes.md` + updated `README.md` + updated man page
> - 25 implementation tasks + 4 final verification tasks
>
> **Estimated Effort**: XL (~25 implementation tasks, ~6 waves)
> **Parallel Execution**: YES — 5 implementation waves (5-9 tasks each) + 1 final verification wave (4 parallel)
> **Critical Path**: T1 (preprocessing module) → T3 (enums) → T10 (wire renderers) → T18 (pipeline integration) → T24 (end-to-end test) → F1-F4 (verification)

---

## Context

### Original Request
> "now, back in plan mode, can we make it work harder for real world photos? SVG is meant for more simple images, but hey.. i want to give it a shot"

User wants to push img2svg beyond its current sweet spot (logos, diagrams, line art) into real-world photo territory. SVG is traditionally for simpler images; the user understands the trade-off and wants to TRY.

### Interview Summary

**Key Discussions**:
- **Plan scope**: User picked "All of the above (XL)" — pre-processing + new modes + YOLO segmentation + CLI flags. ~15-25 tasks.
- **Defaults**: User picked "Aggressive" — all photo modes (auto, trace, visual, annotated) get default preprocessing unless `--no-preprocess`. SEGMENTED auto-selects yolo11x-seg for PHOTO images. Maximum quality, larger files, slower.
- **Segmented output format**: User picked "Multi-layer editable SVG" — each detected object becomes its own `<g id="obj_class_id">` group with semantic class name. Background traced separately.
- **Auto-mode behavior (CRITICAL, added during interview)**: User said: "Annotated mode should not be included in auto when rendering, annotated must be explicitly requested, as well as label". This is a **behavior change**: `IMAGE_TYPE_TO_MODE` no longer maps any ImageType to `LABELS` or `ANNOTATED`. Auto mode now picks from `{VISUAL, TRACE, POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED}` only.

**Research Findings** (3 librarian reports — full reports saved to draft):
- **vtracer**: Presets are NOT exposed in Python API (must manually replicate kwargs). 11 kwargs with strict ranges. vtracer does NOT validate ranges in Python (only CLI does) — our wrapper must validate. `path_precision` default is 2 (not 8 as .pyi claims). License: MIT.
- **YOLO11 segmentation**: `retina_masks=True` is CRITICAL for getting masks at original image size. `masks.gt_(0.0).byte()` (no sigmoid, threshold 0). `cv2.RETR_EXTERNAL` only returns outer contours. Recommended: `yolo11s-seg` default, `yolo11m-seg` for high quality. License: AGPL-3.0 (already covered by existing NOTICE).
- **OpenCV pre-processing**: `bilateralFilter` is available in headless build (in `imgproc` module). Recommended pipeline: denoise → sharpen → posterize → vtracer. Use `cv2.addWeighted` for sharpening (deforum pattern). Use numpy bit-shift for posterize. vtracer's `color_precision` subsumes explicit k-means. bilateral is the bottleneck (~2.3s on 1080p).

### Metis Review

**Identified Gaps (addressed in plan)**:

**Critical — added new tasks T26-T30 + updates to existing tasks**:
1. **`test_presets.py` must be updated for new auto-mode behavior** — existing test `test_select_mode_auto_uses_image_type_table` asserts `Mode.ANNOTATED` for PHOTO, must change to `Mode.DETAILED`. **T26 added.**
2. **Missing test file coverage** — `test_metadata.py`, `test_models.py`, `test_cli.py` need updates for new fields and flags. **T27-T28 added.**
3. **Missing fixture for SEGMENTED mode testing** — RESOLVED via user's existing `tests/testimg/` directory (50+ real photos: `Designer (1).jpeg` through `Designer (50).jpeg`, `2024-Q4.jpg`, `3840x2160-dark-freebsd.png`, `banner-*.png`). **No new fixture needed; use existing testimg in tests.**
4. **Sample SVGs for new modes missing** — `test_examples.py` checks for ≥8 sample SVGs, but new modes produce different output. **T29 includes running all 10 modes on a sample testimg photo and adding the resulting SVGs to test_examples.**
5. **Test file markers for slow tests** — segmentation tests will be slow (model download + inference). **T28 includes `@pytest.mark.slow` markers.**

**Decisions resolved**:
1. **`palette_size` vs `--max-colors`**: Remove the unused `palette_size` field; use `max_colors` (the new flag) for color capping. Avoids confusion. **T5 updated.**
2. **CLAHE in WATERCOLOR mode**: Skip CLAHE. WATERCOLOR uses bilateral + unsharp only. CLAHE stays opt-in via `--clahe` flag (deferred to v2). **T1's `watercolor` preset updated.**
3. **EDGE mode Canny flow**: Apply median → Canny → vtracer with `bw_edge` preset (colormode=binary, mode=polygon). Canny output is already binary, so vtracer handles it. **T8 docstring updated.**
4. **`bw` preset status**: Keep as-is. Not used by any new mode. Available for advanced users via `VtracerVectorizer(preset="bw")`. Not exposed in any CLI mode.
5. **VRAM fallback for YOLO seg**: If yolo11s-seg fails with OOM, auto-fallback to yolo11n-seg. If still fails, fall back to bbox detection only with a warning. **T11 updated.**
6. **Background tracing in SEGMENTED**: Mask out foreground (set to white/transparent), trace the masked image as background. Foreground = per-region trace. **T16 updated.**
7. **MAX_SVG_SIZE guard**: Check after SVG generation, before write. Default 50MB. If exceeded, emit warning + skip write. Add `--max-svg-size` CLI flag to override. **T20 updated.**

**Architectural decisions made**:
1. **YOLO-seg as separate class**: `YOLOSegmentor` (not extending `YOLODetector`). Separate cache key in `get_segmentor()`. **T11.**
2. **Per-region vtracer instances**: One `VtracerVectorizer` per region (lightweight, no shared state). **T13.**
3. **Multi-layer SVG z-order**: Background first, then objects in order of detection (NMS-sorted, by confidence). **T16.**
4. **`MODE_TO_PRESET[Mode.SEGMENTED]`**: Set to `"default"` (used for background only; foreground uses per-region preset). **T10.**

**Documentation updates needed** (deferred to T25 + new T25a-T25f):
- `docs/photo-modes.md` (NEW, ~400 lines)
- `README.md` (Output Modes table, Features bullet, Architecture diagram)
- `man/img2svg.1` (add new flags)
- `mkdocs.yml` (nav update)
- `docs/usage.md` (mode table)
- `docs/modes.md` (merge with photo-modes.md OR add sections)
- `docs/api.md` (ConversionOptions table)
- `docs/architecture.md` (pipeline mermaid update)
- `docs/installation.md` (yolo11s-seg availability)
- `docs/index.md` (highlights + map)
- `docs/changelog.md` (Unreleased section)

**Test file updates needed** (deferred to T27-T28):
- `test_presets.py` (PHOTO → DETAILED, drop ANNOTATED/LABELS)
- `test_cli.py` (new flags + validators)
- `test_metadata.py` (new Sidecar fields round-trip)
- `test_models.py` (RegionInfo + new Sidecar fields)
- `test_manpage.py` (EXPECTED_FLAGS updated)
- `test_examples.py` (≥13 sample SVGs after new modes)
- `tests/conftest.py` (no new fixtures needed — use existing + new photo_with_objects)

**Out-of-scope (deferred to v2)**:
- Auto-fallback chain on YOLO seg OOM (v1.1: simpler — just warn)
- Per-region palette extraction (use class-derived default palette)
- GPU memory pre-flight check
- Tile-based parallel tracing
- Quality metrics (SSIM, PSNR)
- Style transfer
- Depth estimation
- Region-based palette extraction

---

## Work Objectives

### Core Objective
Extend img2svg to produce high-quality SVG output for real-world photos by adding a configurable pre-processing pipeline, 5 new photo-tuned output modes, and YOLO11 instance segmentation for multi-layer editable SVGs — while keeping existing behavior backward-compatible for users who don't opt in.

### Concrete Deliverables
1. **New module**: `src/img2svg/preprocessing.py` (~200 lines, composable OpenCV filters)
2. **5 new Mode enum values**: `POSTER`, `DETAILED`, `EDGE`, `WATERCOLOR`, `SEGMENTED`
3. **5 new renderers**: `PosterRenderer`, `DetailedRenderer`, `EdgeRenderer`, `WatercolorRenderer`, `SegmentedRenderer`
4. **3 new vtracer presets**: `photo_hifi`, `bw_edge`, `watercolor` (in `vectorizer.py:PRESETS`)
5. **YOLO seg integration**: New `YOLOSegmentor` class in `detector.py`; mask extraction + per-region vtracer
6. **8 new CLI flags**: `--preprocess`, `--denoise`, `--sharpen`, `--max-colors`, `--quality`, `--no-preprocess`, `--seg-model`, `--no-seg`
7. **Sidecar extensions**: `preprocessing: list[str]`, `regions: list[RegionInfo]`, `model_variant: str`
8. **Updated auto-mode mapping**: `PHOTO → DETAILED` (aggressive); `LOGO/DIAGRAM/SCREENSHOT/LINE_ART/UNKNOWN → VISUAL` (clean)
9. **New documentation**: `docs/photo-modes.md` (~400 lines with examples)
10. **Updated documentation**: `README.md`, `man/img2svg.1`, `mkdocs.yml`
11. **3 new test files**: `tests/test_preprocessing.py`, `tests/test_segmentation.py`, `tests/test_renderers.py`
12. **Updated existing tests**: `test_presets.py`, `test_pipeline.py`, `test_cli.py`, `conftest.py`
13. **New fixtures**: `tests/fixtures/photo_with_objects.png` (synthetic photo for segmentation tests)
14. **End-to-end verification on user's real photo**: `/home/mlapointe/Documents/rtlogo-1.png`

### Definition of Done
- [ ] `uv run pytest -m "not slow" -q` → 480+ passed (was 474; +6 new test categories)
- [ ] `uv run pytest --cov=img2svg --cov-report=term` → 90%+ coverage maintained
- [ ] `uv run ruff check src tests` → 0 issues
- [ ] `uv run mypy src` → 0 new errors
- [ ] `uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-photo.svg --mode detailed` → succeeds
- [ ] `uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-seg.svg --mode segmented --model yolo11s-seg.pt` → multi-layer SVG with at least 1 `<g id="obj_...">` group
- [ ] `uv run img2svg --mode auto /home/mlapointe/Documents/rtlogo-1.png` → does NOT auto-pick LABELS or ANNOTATED
- [ ] `man img2svg` → shows all 9 modes in help (auto, labels, visual, annotated, trace, poster, detailed, edge, watercolor, segmented)
- [ ] All commits pushed to `origin/main`; working tree clean

### Must Have
- 5 new modes that produce visibly different SVG output for the same input
- Pre-processing pipeline that's controllable via CLI flags
- YOLO segmentation produces pixel-accurate instance masks at original image resolution
- Multi-layer SVG output for SEGMENTED mode (one `<g>` per detected object, semantic class names)
- Existing 474 fast tests continue to pass (no regressions)
- Backward compatibility: existing CLI invocations (`--mode visual`, `--mode trace`, etc.) produce identical output to current behavior
- License compliance: NOTICE file references ultralytics AGPL-3.0

### Must NOT Have (Guardrails)
- **No new dependencies** beyond what's already in pyproject.toml (ultralytics, opencv-python-headless, PIL, lxml, typer, rich, pydantic, numpy, vtracer)
- **No external API calls** (no cloud services, no SaaS dependencies)
- **No breaking changes to existing modes** — VISUAL, TRACE, LABELS, ANNOTATED behavior must be byte-identical for same input
- **No new errors in test suite** — pre-existing 1 i18n failure is acceptable; no new failures
- **No AI slop**:
  - No `as any`, `# type: ignore` without justification
  - No commented-out code in final commits
  - No empty except blocks
  - No console.log debugging in production
  - No over-abstraction (no premature helper extraction)
  - No generic names (`data`, `result`, `item`, `temp`)
  - Comments only where non-obvious (matches codebase style)
- **No scope creep** — do not add features not listed in deliverables (e.g., no depth estimation, no style transfer, no multi-resolution tiling)
- **No placeholder data** — all new fixtures must be real test images, not synthetic stand-ins
- **No silent failures** — every error path must log + raise or return clearly

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: YES (pytest, 474 fast tests passing, fixtures in conftest.py)
- **Automated tests**: TDD — RED-GREEN-REFACTOR for each new module
- **Framework**: pytest (existing) with `pytest-mock` for vtracer/YOLO mocking
- **Coverage target**: 90%+ maintained

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Preprocessing tests**: Use synthetic + real test images; verify pixel-level transformations
- **YOLO seg tests**: Mock `ultralytics.YOLO.predict` to return synthetic `Results` with `masks.data`, `masks.xy`, `boxes.xyxy`
- **Renderer tests**: Mock `VtracerVectorizer.vectorize` to return canned SVG; verify SVG group structure
- **End-to-end test**: Use `/home/mlapointe/Documents/rtlogo-1.png` (real user file); run `img2svg convert` for each new mode; verify SVG file exists + is valid XML + contains expected elements
- **CLI tests**: Use `typer.testing.CliRunner` to invoke CLI with flag combinations; verify exit codes and stdout

### Per-Task QA Requirements
- Minimum 1 happy-path scenario + 1 failure/edge-case scenario per task
- Each scenario uses specific selectors/data, not vague descriptions
- Evidence file path recorded for each scenario

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.
> Target: 5-9 tasks per wave. Fewer than 3 per wave (except final) = under-splitting.

```
Wave 1 (Foundation - 5 tasks, parallel):
├── T1: preprocessing.py module (denoise, sharpen, posterize, edge, clahe)
├── T2: vtracer PRESETS extension (photo_hifi, bw_edge, watercolor)
├── T3: Mode enum extension (POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
├── T4: Sidecar schema extension (preprocessing, regions, model_variant)
└── T5: CLI flags part 1 (--preprocess, --denoise, --sharpen, --max-colors, --quality, --no-preprocess)

Wave 2 (New Renderers - 5 tasks, parallel):
├── T6: PosterRenderer (uses existing 'poster' preset)
├── T7: DetailedRenderer (uses new 'photo_hifi' preset)
├── T8: EdgeRenderer (uses new 'bw_edge' preset)
├── T9: WatercolorRenderer (uses new 'watercolor' preset)
└── T10: Wire new renderers to RENDERER_REGISTRY + MODE_TO_PRESET + IMAGE_TYPE_TO_MODE

Wave 3 (YOLO Segmentation - 5 tasks, parallel):
├── T11: YOLOSegmentor class in detector.py (yolo11s-seg, retina_masks=True)
├── T12: Mask extraction helpers (masks.xy → polygons, area, bbox)
├── T13: Per-region vtracer tracer (crop region, trace, embed with offset transform)
├── T14: CLI flags part 2 (--seg-model, --no-seg)
└── T15: Sidecar regions field population (RegionInfo Pydantic model)

Wave 4 (Segmented Renderer + Pipeline Wiring - 2 tasks, parallel):
├── T16: SegmentedRenderer (multi-layer SVG, background + per-region groups)
└── T17: Pipeline integration: preprocessing step (post-load, pre-analyze)

Wave 5 (Pipeline Integration + Safety - 3 tasks, parallel):
├── T18: Pipeline integration: segmentation step (per-region vtracer for SEGMENTED mode)
├── T19: Pipeline timing instrumentation (preprocess_ms, segment_ms, vectorize_ms per region)
└── T20: MAX_SVG_SIZE guard (50MB cap, emit warning + truncate metadata)

Wave 6 (Tests + Documentation - 5 tasks, parallel):
├── T21: Test files (test_preprocessing.py, test_segmentation.py, test_renderers.py)
├── T22: Update existing tests (test_presets.py, test_pipeline.py, test_cli.py, conftest.py)
├── T23: New fixture image (tests/fixtures/photo_with_objects.png)
├── T24: End-to-end verification on user's rtlogo-1.png
└── T25: Documentation updates (docs/photo-modes.md, README.md, man/img2svg.1, mkdocs.yml)

Wave FINAL (Verification - 4 tasks, parallel):
├── F1: Plan Compliance Audit (oracle)
├── F2: Code Quality Review (unspecified-high)
├── F3: Real Manual QA (unspecified-high + playwright)
└── F4: Scope Fidelity Check (deep)

Critical Path: T1 → T3 → T10 → T18 → T24 → F1-F4 → user okay
Parallel Speedup: ~70% faster than sequential
Max Concurrent: 5 (Waves 1-3) + 5 (Wave 6)
```

### Dependency Matrix

- **T1-T5** (Wave 1): no deps, run in parallel
- **T6-T9** (Wave 2): depend on T1, T2, T3 (renderers use new presets, new modes, new preprocessing)
- **T10** (Wave 2): depends on T3, T6-T9 (registry wiring)
- **T11-T14** (Wave 3): depend on T3 (new Mode values), T1 (preprocessing for whole-image pre)
- **T15** (Wave 3): depends on T11 (YOLO seg returns regions)
- **T16** (Wave 4): depends on T11, T12, T13, T15 (uses all segmentation pieces)
- **T17** (Wave 4): depends on T1, T3 (preprocessing + new modes)
- **T18** (Wave 5): depends on T16, T17 (renderer + preprocessing wired)
- **T19** (Wave 5): depends on T17, T18 (timings for both)
- **T20** (Wave 5): depends on T18 (size check after render)
- **T21-T23** (Wave 6): depend on T6-T18 (test what was built)
- **T24** (Wave 6): depends on T1-T18 (full end-to-end)
- **T25** (Wave 6): depends on T6-T20 (document what was built)
- **F1-F4** (FINAL): depend on T1-T25 complete

### Agent Dispatch Summary

- **Wave 1 (5)**: All `quick` — module scaffolding, no complex logic
- **Wave 2 (5)**: T6-T9 → `quick`; T10 → `quick` (registry wiring)
- **Wave 3 (5)**: T11 → `deep` (YOLO integration); T12 → `quick`; T13 → `deep` (per-region vtracer); T14 → `quick`; T15 → `quick`
- **Wave 4 (2)**: T16 → `deep` (multi-layer SVG); T17 → `unspecified-high` (pipeline integration)
- **Wave 5 (3)**: All `unspecified-high` — pipeline integration
- **Wave 6 (5)**: T21-T23 → `quick` (test files); T24 → `unspecified-high` (end-to-end); T25 → `writing` (docs)
- **FINAL (4)**: F1 → `oracle`; F2 → `unspecified-high`; F3 → `unspecified-high` + `playwright`; F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

- [x] 1. Create `preprocessing.py` module with composable OpenCV filters

  **What to do**:
  - Create `src/img2svg/preprocessing.py` (~200-300 lines)
  - Implement 7 composable filter functions, each takes `np.ndarray` (H,W,3) or (H,W,4) uint8 and returns same dtype/shape:
    - `denoise_bilateral(img, d=5, sigma=50) -> np.ndarray` — edge-preserving denoise (default for photos)
    - `denoise_nlmeans(img, h=6) -> np.ndarray` — heavier statistical denoise (opt-in)
    - `denoise_median(img, k=3) -> np.ndarray` — cheap baseline (used in EDGE mode)
    - `sharpen_unsharp(img, sigma=2.0, amount=0.5) -> np.ndarray` — `cv2.addWeighted` (deforum pattern)
    - `posterize(img, bits=4) -> np.ndarray` — numpy bitwise (NOT PIL)
    - `detect_edges_canny(img, low=80, high=180) -> np.ndarray` — 3-channel output for vtracer
    - `apply_clahe_yuv(img, clip=2.0, tile=(8,8)) -> np.ndarray` — YUV L-channel only
  - Add `PreprocessingPipeline` class with `__init__(steps: list[tuple[str, dict]])`, `apply(image) -> np.ndarray`, `steps_applied() -> list[str]`
  - Add `apply_preprocessing(image, steps: list[tuple[str, dict]]) -> np.ndarray` convenience function (returns image + populates a `result.steps_applied` via context)
  - Add `PREPROCESSING_PRESETS` dict with: `light` (bilateral(50) + unsharp(0.5)), `medium` (light + posterize(4)), `heavy` (medium + edges), `edge` (median + canny)
  - All filters preserve alpha channel (slice off, transform, reattach)
  - All filters handle uint8 only; raise `TypeError` for other dtypes with helpful message

  **Must NOT do**:
  - No new dependencies (cv2 is already imported)
  - No PIL for posterize
  - No CLAHE as default (opt-in only)
  - No `as any` or `# type: ignore` without justification
  - No over-abstraction (each filter is a simple function; no Filter base class)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Module scaffolding with composable functions; no complex logic
  - **Skills**: `[]` (no specialized skills needed)
  - **Skills Evaluated but Omitted**:
    - `ai-slop-remover`: Apply in F2, not at T1

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2, T3, T4, T5)
  - **Blocks**: T6, T7, T8, T9 (renderers use preprocessing)
  - **Blocked By**: None

  **References** (CRITICAL):
  - **Pattern References**:
    - `src/img2svg/patterns.py:24-66` — `_kmeans_dominant_colors` shows how to handle 4-channel images: `if image.shape[-1] == 4: rgb_image = image[..., :3]`. Match this pattern.
  - **API/Type References**:
    - `src/img2svg/loader.py:19-33` — `LoadedImage.np_array` returns `np.ndarray` of shape (H, W, 3) or (H, W, 4) uint8. This is the input format.
    - `src/img2svg/detector.py:144-147` — PIL `convert("RGB")` pattern for normalizing channels. Use same approach for 4ch→3ch if needed.
  - **External References**:
    - `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)` — confirmed available in headless build (imgproc module)
    - `cv2.addWeighted(img1, alpha, img2, beta, gamma)` — `sharpened = (1+amount)*img - amount*gauss` (deforum pattern)
    - `cv2.fastNlMeansDenoisingColored(src, h, hColor, 7, 21)` — NLMeans (slower alternative)
    - `np.bitwise_and(img, np.uint8(255 << (8 - bits)))` — posterize bit-shift
    - `cv2.createCLAHE(clipLimit, tileGridSize).apply(channel)` — apply to YUV L-channel
    - `cv2.Canny(blurred_gray, low, high)` — edge detection
  - **WHY Each Reference Matters**:
    - The patterns.py reference shows the canonical way to handle RGBA in this codebase (slice off alpha, process, reattach). Following it ensures consistency.
    - The loader reference defines the input contract.
    - The detector.py reference shows how PIL is used for normalization (defensive, for unexpected inputs).
    - External refs are the canonical OpenCV API for each function.

  **Acceptance Criteria**:
  - [ ] File `src/img2svg/preprocessing.py` exists, 200-300 lines
  - [ ] All 7 filter functions implemented and importable
  - [ ] `PreprocessingPipeline` class with `apply()` method works
  - [ ] `PREPROCESSING_PRESETS` dict has `light`, `medium`, `heavy`, `edge`
  - [ ] Alpha channel preserved through all transformations
  - [ ] dtype uint8 preserved through all transformations
  - [ ] `uv run ruff check src/img2svg/preprocessing.py` → 0 issues
  - [ ] `uv run pytest tests/test_preprocessing.py` → all pass

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: denoise_bilateral reduces noise on synthetic test image
    Tool: Bash (python REPL via uv)
    Preconditions: 1080p RGB uint8 image with synthetic Gaussian noise (sigma=20)
    Steps:
      1. Read `tests/fixtures/photo.jpg` (existing fixture)
      2. Add Gaussian noise: `noisy = (img + np.random.normal(0, 20, img.shape)).clip(0,255).astype(np.uint8)`
      3. Apply: `denoised = denoise_bilateral(noisy, d=5, sigma=50)`
      4. Assert `denoised.dtype == np.uint8`
      5. Assert `denoised.shape == noisy.shape`
      6. Assert `denoised.std() < noisy.std() * 0.7` (denoise actually reduced noise)
    Expected Result: denoised image has lower noise (stddev) than input, same shape/dtype
    Failure Indicators: shape mismatch, dtype change, stddev not reduced
    Evidence: .sisyphus/evidence/task-01-bilateral-denoise.png

  Scenario: sharpen_unsharp increases edge density
    Tool: Bash (python REPL via uv)
    Preconditions: Blurry RGB image (apply GaussianBlur with sigma=3 to `tests/fixtures/photo.jpg`)
    Steps:
      1. Generate blurry input: `blurry = cv2.GaussianBlur(photo, (0,0), 3.0)`
      2. Apply: `sharp = sharpen_unsharp(blurry, sigma=2.0, amount=1.0)`
      3. Compute edge density: `edges_sharp = cv2.Canny(sharp, 100, 200).sum()`
      4. Compute: `edges_blurry = cv2.Canny(blurry, 100, 200).sum()`
      5. Assert `edges_sharp > edges_blurry * 1.2` (sharper has 20%+ more edges)
      6. Assert `sharp.dtype == np.uint8`
    Expected Result: Sharpened image has measurably more edges than blurry input
    Failure Indicators: edges_sharp <= edges_blurry (no sharpening happened)
    Evidence: .sisyphus/evidence/task-01-unsharp-sharpen.png

  Scenario: posterize(4) produces 16 levels per channel
    Tool: Bash (python REPL via uv)
    Preconditions: Gradient image (H=W=100, 3ch, values 0-255 linear ramp)
    Steps:
      1. Generate gradient: `gradient = np.linspace(0, 255, 100, dtype=np.uint8).reshape(1, -1).repeat(100, 0).repeat(3, 2).transpose(1,2,0)`
      2. Apply: `posterized = posterize(gradient, bits=4)`
      3. Get unique values: `unique = np.unique(posterized)`
      4. Assert `len(unique) <= 16` (at most 16 levels per channel)
    Expected Result: Output has at most 16 distinct values per channel
    Failure Indicators: More than 16 unique values
    Evidence: .sisyphus/evidence/task-01-posterize-levels.txt

  Scenario: detect_edges_canny on RGB returns 3-channel uint8
    Tool: Bash (python REPL via uv)
    Preconditions: RGB photo
    Steps:
      1. Apply: `edges = detect_edges_canny(photo, low=80, high=180)`
      2. Assert `edges.shape == photo.shape` (same H, W, 3)
      3. Assert `edges.dtype == np.uint8`
      4. Assert `np.unique(edges).tolist() == [0, 255]` (pure binary, only black/white)
    Expected Result: 3-channel uint8 image with only 0 and 255 values
    Failure Indicators: 1-channel output, intermediate gray values
    Evidence: .sisyphus/evidence/task-01-edges-canny.png

  Scenario: PREPROCESSING_PRESETS run end-to-end on photo fixture
    Tool: Bash (python REPL via uv)
    Preconditions: `tests/fixtures/photo.jpg` exists
    Steps:
      1. Load photo as np.ndarray
      2. For each preset in `['light', 'medium', 'heavy', 'edge']`:
         a. `pipeline = PreprocessingPipeline(PREPROCESSING_PRESETS[preset])`
         b. `result = pipeline.apply(photo)`
         c. Assert `result.shape == photo.shape`
         d. Assert `result.dtype == np.uint8`
         e. Time the run: assert < 5 seconds
      3. Assert all 4 presets succeed
    Expected Result: All 4 presets produce valid output in < 5s on 1080p
    Failure Indicators: Shape/dtype mismatch, slow (>5s)
    Evidence: .sisyphus/evidence/task-01-presets-timing.json
  ```

  **Evidence to Capture**:
  - [ ] `task-01-bilateral-denoise.png` — input/output side-by-side
  - [ ] `task-01-unsharp-sharpen.png` — input/output side-by-side
  - [ ] `task-01-posterize-levels.txt` — unique value counts
  - [ ] `task-01-edges-canny.png` — canny output
  - [ ] `task-01-presets-timing.json` — per-preset timing

  **Commit**: YES (Wave 1)
  - Message: `feat(preprocessing): add composable OpenCV filters (denoise, sharpen, posterize, edge, clahe)`
  - Files: `src/img2svg/preprocessing.py`
  - Pre-commit: `uv run pytest tests/test_preprocessing.py -q && uv run ruff check src/img2svg/preprocessing.py`

- [x] 2. Extend `vectorizer.py` with 3 new vtracer presets

  **What to do**:
  - Add 3 new entries to `src/img2svg/vectorizer.py:PRESETS` dict (after `photo` at line 90):
    - `"photo_hifi"` — for DETAILED mode: `colormode="color"`, `hierarchical="stacked"`, `mode="spline"`, `filter_speckle=4`, `color_precision=8`, `layer_difference=24`, `corner_threshold=60`, `length_threshold=3.5`, `max_iterations=20`, `splice_threshold=30`, `path_precision=4`
    - `"bw_edge"` — for EDGE mode: `colormode="binary"`, `hierarchical="stacked"`, `mode="polygon"`, `filter_speckle=8`, `color_precision=6`, `layer_difference=16`, `corner_threshold=120`, `length_threshold=5.0`, `max_iterations=5`, `splice_threshold=60`, `path_precision=2`
    - `"watercolor"` — for WATERCOLOR mode: `colormode="color"`, `hierarchical="stacked"`, `mode="spline"`, `filter_speckle=14`, `color_precision=7`, `layer_difference=32`, `corner_threshold=20`, `length_threshold=5.0`, `max_iterations=15`, `splice_threshold=20`, `path_precision=3`
  - Update `Preset` Literal type (line 19) to include `"photo_hifi" | "bw_edge" | "watercolor"`
  - Verify `VtracerVectorizer.__init__` accepts new presets (existing `if preset not in PRESETS` check still works)
  - Verify all 11 kwargs are present in each new preset

  **Must NOT do**:
  - Don't change existing presets (backward compat — existing test `test_presets.py` covers them)
  - Don't add vtracer preset access from Python (`preset=` is CLI-only; we use kwargs in Python)
  - Don't validate ranges in vtracer (vtracer doesn't validate Python kwargs)
  - Don't add comments explaining each param (code is self-documenting)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding data dicts to existing structure
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3, T4, T5)
  - **Blocks**: T7 (DetailedRenderer uses photo_hifi), T8 (EdgeRenderer uses bw_edge), T9 (WatercolorRenderer uses watercolor), T10 (MODE_TO_PRESET)
  - **Blocked By**: None

  **References**:
  - **Pattern References**:
    - `src/img2svg/vectorizer.py:22-91` — existing PRESETS structure (5 presets, all 11 kwargs each)
  - **API/Type References**:
    - `src/img2svg/vectorizer.py:19` — `Preset = Literal["default", "bw", "logo", "poster", "photo"]` (needs update)
    - `src/img2svg/vectorizer.py:97-101` — `VtracerVectorizer.__init__` validates preset name
  - **External References**:
    - vtracer research (in draft): `photo_hifi` is tuned to match vtracer's official Photo preset (color_precision=8, filter_speckle=10, layer_difference=48, corner_threshold=180) but with `filter_speckle=4` (less aggressive noise removal) and `corner_threshold=60` (more smoothing)
    - vtracer `path_precision` default is 2 (NOT 8 as .pyi claims)
  - **WHY Each Reference Matters**:
    - Existing PRESETS dict shows the canonical structure. Following it ensures consistency.
    - `Preset` Literal type is used in public API; must be updated.
    - vtracer research shows correct parameter values to avoid silent garbage output.

  **Acceptance Criteria**:
  - [ ] 3 new entries in PRESETS: `photo_hifi`, `bw_edge`, `watercolor`
  - [ ] All 11 vtracer kwargs present in each new preset
  - [ ] `Preset` Literal type updated
  - [ ] `VtracerVectorizer(preset="photo_hifi")` instantiates without error
  - [ ] `VtracerVectorizer(preset="unknown")` raises ValueError (existing behavior)
  - [ ] `uv run ruff check src/img2svg/vectorizer.py` → 0 issues
  - [ ] `uv run pytest tests/test_presets.py` → all pass (existing tests)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All 3 new presets are valid and contain all 11 kwargs
    Tool: Bash (python REPL via uv)
    Preconditions: None
    Steps:
      1. `from img2svg.vectorizer import PRESETS, VtracerVectorizer`
      2. For each preset in `['photo_hifi', 'bw_edge', 'watercolor']`:
         a. Assert `preset in PRESETS`
         b. Assert `len(PRESETS[preset]) == 11` (exactly 11 kwargs)
         c. Assert all required keys present: `{'colormode', 'hierarchical', 'mode', 'filter_speckle', 'color_precision', 'layer_difference', 'corner_threshold', 'length_threshold', 'max_iterations', 'splice_threshold', 'path_precision'}`
         d. `vec = VtracerVectorizer(preset=preset)` (should not raise)
         e. Assert `vec.preset == preset` and `len(vec.params) == 11`
    Expected Result: All 3 presets pass all checks
    Failure Indicators: Missing key, wrong count, instantiation error
    Evidence: .sisyphus/evidence/task-02-presets-validation.json

  Scenario: New preset param values are within vtracer's valid ranges
    Tool: Bash (python REPL via uv)
    Preconditions: vtracer research notes loaded
    Steps:
      1. For each preset, check param values are in:
         - filter_speckle: [0, 16]
         - color_precision: [1, 8]
         - layer_difference: [0, 255]
         - corner_threshold: [0, 180]
         - length_threshold: [3.5, 10.0]
         - splice_threshold: [0, 180]
         - colormode: 'color' or 'binary'
         - hierarchical: 'stacked' or 'cutout'
         - mode: 'spline', 'polygon', or 'none'
      2. Assert all values in range
    Expected Result: All param values within vtracer's valid ranges
    Failure Indicators: Out-of-range value (would silently produce garbage in vtracer)
    Evidence: .sisyphus/evidence/task-02-presets-ranges.json

  Scenario: Existing presets unchanged (backward compat)
    Tool: Bash (python REPL via uv)
    Preconditions: git knows the current state
    Steps:
      1. `from img2svg.vectorizer import PRESETS`
      2. For each existing preset in `['default', 'bw', 'logo', 'poster', 'photo']`:
         a. Assert preset still in PRESETS
         b. Assert `len(PRESETS[preset]) == 11` (same as before)
         c. Assert all param values are byte-identical to pre-T2 values (compare with git HEAD)
    Expected Result: Existing 5 presets unchanged
    Failure Indicators: Any param value changed
    Evidence: .sisyphus/evidence/task-02-backward-compat.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-02-presets-validation.json` — all 3 new presets pass
  - [ ] `task-02-presets-ranges.json` — all values in range
  - [ ] `task-02-backward-compat.txt` — existing 5 unchanged

  **Commit**: YES (Wave 1)
  - Message: `feat(vectorizer): add photo_hifi, bw_edge, watercolor presets`
  - Files: `src/img2svg/vectorizer.py`
  - Pre-commit: `uv run pytest tests/test_presets.py -q && uv run ruff check src/img2svg/vectorizer.py`

- [x] 3. Extend `enums.py` Mode with 5 new values

  **What to do**:
  - Add 5 new Mode enum values to `src/img2svg/enums.py` Mode class (after `TRACE` at line 30):
    - `POSTER = "poster"`
    - `DETAILED = "detailed"`
    - `EDGE = "edge"`
    - `WATERCOLOR = "watercolor"`
    - `SEGMENTED = "segmented"`
  - No other changes to the file (StrEnum handles it)

  **Must NOT do**:
  - Don't change existing Mode values (backward compat)
  - Don't add help text here (that's in CLI, T5)
  - Don't add validation logic (the `_mode_callback` in CLI handles that)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial enum extension
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T4, T5)
  - **Blocks**: T6-T9 (renderers), T10 (registry), T11+ (everything that uses new modes)
  - **Blocked By**: None

  **References**:
  - **Pattern References**:
    - `src/img2svg/enums.py:23-30` — existing Mode enum definition

  **Acceptance Criteria**:
  - [ ] 5 new Mode enum values
  - [ ] `Mode.POSTER.value == "poster"`, etc.
  - [ ] All existing tests still pass
  - [ ] `uv run ruff check src/img2svg/enums.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All 5 new modes are valid StrEnum values
    Tool: Bash (python REPL via uv)
    Preconditions: None
    Steps:
      1. `from img2svg.enums import Mode`
      2. Assert all 5 new modes exist: `Mode.POSTER`, `Mode.DETAILED`, `Mode.EDGE`, `Mode.WATERCOLOR`, `Mode.SEGMENTED`
      3. Assert each has correct string value: `Mode.POSTER.value == "poster"`, etc.
      4. Assert `Mode` now has 10 total values (was 5, added 5)
    Expected Result: All 5 modes are valid, count is 10
    Failure Indicators: Missing mode, wrong value, count != 10
    Evidence: .sisyphus/evidence/task-03-enum-validation.json

  Scenario: StrEnum comparison works (string equality)
    Tool: Bash (python REPL via uv)
    Preconditions: None
    Steps:
      1. `from img2svg.enums import Mode`
      2. For each new mode: `assert Mode.POSTER == "poster"`, etc.
      3. Assert `str(Mode.DETAILED) == "Mode.DETAILED"` (StrEnum repr)
    Expected Result: String equality works as expected
    Failure Indicators: String comparison fails
    Evidence: .sisyphus/evidence/task-03-strenum-compat.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-03-enum-validation.json`
  - [ ] `task-03-strenum-compat.txt`

  **Commit**: YES (Wave 1)
  - Message: `feat(enums): add POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED modes`
  - Files: `src/img2svg/enums.py`
  - Pre-commit: `uv run pytest -q -k "test_mode or test_enum" && uv run ruff check src/img2svg/enums.py`

- [x] 4. Extend `models.py` Sidecar with preprocessing + regions + model_variant fields

  **What to do**:
  - Add new Pydantic model `RegionInfo` in `src/img2svg/models.py` (after `Detection` at line 46):
    ```python
    class RegionInfo(BaseModel):
        """A YOLO segmentation region traced independently."""
        class_id: int
        class_name: str
        confidence: float = Field(ge=0.0, le=1.0)
        bbox: BoundingBox
        polygon_points: int = Field(ge=0)
        area_pixels: int = Field(ge=0)
        vtracer_preset: str
    ```
  - Add 3 new fields to `Sidecar` model (line 177-203):
    - `preprocessing: list[str] = Field(default_factory=list)` — list of step names applied
    - `regions: list[RegionInfo] = Field(default_factory=list)` — per-region metadata for SEGMENTED mode
    - `model_variant: str = ""` — e.g., `"yolo11s-seg"`, `"yolo11m-seg"`, or `""` for detection-only
  - All new fields must be optional with defaults (backward compat for existing sidecar consumers)
  - Update `__init__.py` exports to include `RegionInfo`

  **Must NOT do**:
  - Don't change existing Sidecar fields (backward compat — old sidecar readers must still parse)
  - Don't add required fields (defaults are mandatory)
  - Don't validate `vtracer_preset` against `PRESETS` keys (decouples models from vectorizer)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding optional Pydantic fields + one new model
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T3, T5)
  - **Blocks**: T15 (regions field population), T24 (end-to-end needs Sidecar schema)
  - **Blocked By**: None

  **References**:
  - **Pattern References**:
    - `src/img2svg/models.py:39-45` — existing `Detection` model. Match field style and constraint patterns.
    - `src/img2svg/models.py:177-203` — existing `Sidecar` model. Add new fields in a logical position (after `detections` is natural).
  - **API/Type References**:
    - `src/img2svg/models.py:18-36` — `BoundingBox` is already a Pydantic model. Reuse it in `RegionInfo`.
  - **WHY Each Reference Matters**:
    - The existing models show the canonical style: `Field(default_factory=list)` for mutable defaults, `Field(ge=..., le=...)` for constraints.
    - `BoundingBox` reuse avoids duplicating the model.

  **Acceptance Criteria**:
  - [ ] New `RegionInfo` model with 7 fields
  - [ ] 3 new optional fields in `Sidecar`: `preprocessing`, `regions`, `model_variant`
  - [ ] All existing tests still pass
  - [ ] `Sidecar.model_validate_json('{}')` works (with required fields filled)
  - [ ] `RegionInfo` exported from `img2svg/__init__.py`
  - [ ] `uv run ruff check src/img2svg/models.py src/img2svg/__init__.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: New fields have correct defaults
    Tool: Bash (python REPL via uv)
    Preconditions: None
    Steps:
      1. `from img2svg.models import Sidecar, RegionInfo, BoundingBox, Detection`
      2. Build minimal sidecar (only required fields):
         `s = Sidecar(version="0.1.0", input_path=Path("/tmp/in.png"), input_hash="0"*64, output_path=Path("/tmp/out.svg"), output_size=100, mode_used=Mode.LABELS, mode_reasoning="test", model="yolo11x.pt", device="cpu", image_type=ImageType.LOGO)`
      3. Assert `s.preprocessing == []`
      4. Assert `s.regions == []`
      5. Assert `s.model_variant == ""`
    Expected Result: New fields default to empty values
    Failure Indicators: DefaultError (field required), wrong default type
    Evidence: .sisyphus/evidence/task-04-sidecar-defaults.json

  Scenario: Sidecar with regions round-trips through JSON
    Tool: Bash (python REPL via uv)
    Preconditions: None
    Steps:
      1. `from img2svg.models import Sidecar, RegionInfo, BoundingBox`
      2. Build sidecar with `regions=[RegionInfo(class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(x1=10,y1=10,x2=50,y2=50), polygon_points=24, area_pixels=1600, vtracer_preset="photo_hifi")]`
      3. Serialize: `s.model_dump_json()`
      4. Deserialize: `Sidecar.model_validate_json(...)`
      5. Assert `roundtrip.regions[0].class_name == "person"`
      6. Assert `roundtrip.regions[0].polygon_points == 24`
    Expected Result: JSON round-trip preserves all region data
    Failure Indicators: Data loss, type error on deserialization
    Evidence: .sisyphus/evidence/task-04-sidecar-roundtrip.json

  Scenario: RegionInfo is exported from package __init__
    Tool: Bash (python REPL via uv)
    Preconditions: None
    Steps:
      1. `from img2svg import RegionInfo`
      2. Assert `RegionInfo.__name__ == "RegionInfo"`
    Expected Result: RegionInfo importable from top-level package
    Failure Indicators: ImportError
    Evidence: .sisyphus/evidence/task-04-imports.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-04-sidecar-defaults.json`
  - [ ] `task-04-sidecar-roundtrip.json`
  - [ ] `task-04-imports.txt`

  **Commit**: YES (Wave 1)
  - Message: `feat(models): add RegionInfo + Sidecar.preprocessing/regions/model_variant fields`
  - Files: `src/img2svg/models.py`, `src/img2svg/__init__.py`
  - Pre-commit: `uv run pytest tests/test_models.py -q && uv run ruff check src/img2svg/models.py src/img2svg/__init__.py`

- [x] 5. Add CLI flags part 1 (--preprocess, --denoise, --sharpen, --max-colors, --quality, --no-preprocess) + ConversionOptions fields

  **What to do**:
  - In `src/img2svg/cli.py`, add 6 new Typer options to `_convert_cmd()` (after `--no-clobber` at line 320-322):
    - `--preprocess`: `str` with `_preprocess_callback` validator (values: `none|auto|light|medium|heavy|all`). Default: `auto` (resolved by pipeline based on mode)
    - `--denoise`: `int` (0-10, mapped to bilateral sigma via `sigma = value * 10`). Default: 0
    - `--sharpen`: `int` (0-10, mapped to unsharp amount via `amount = value / 10`). Default: 0
    - `--max-colors`: `int` (2-64). Default: 0 (no cap)
    - `--quality`: `str` with `_quality_callback` validator (values: `draft|standard|premium`). Default: `standard`
    - `--no-preprocess`: `bool` flag (sets preprocess to `none`)
  - Update `--mode` help text (line 301): `"auto, labels, visual, annotated, trace, poster, detailed, edge, watercolor, segmented"`
  - Add 5 new fields to `ConversionOptions` in `src/img2svg/models.py` (after `palette_size` at line 147):
    - `preprocess: str = "auto"`
    - `denoise: int = Field(default=0, ge=0, le=10)`
    - `sharpen: int = Field(default=0, ge=0, le=10)`
    - `max_colors: int = Field(default=0, ge=0, le=64)` (0 = no cap)
    - `quality: str = "standard"`
  - Update `_build_options()` in `cli.py` (line 210-227) to pass new fields
  - Add 2 new callback validators after `_mode_callback` (line 145):
    - `_preprocess_callback(value)`: validates against `{none, auto, light, medium, heavy, all}`
    - `_quality_callback(value)`: validates against `{draft, standard, premium}`

  **Must NOT do**:
  - Don't add new subcommands (keep them as flags on `_convert_cmd`)
  - Don't change existing CLI args (backward compat)
  - Don't add complex validation logic (simple enum check is enough)
  - Don't add help text beyond the flag's `help=` kwarg

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding Typer options + Pydantic fields with validators
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T3, T4)
  - **Blocks**: T14 (CLI flags part 2 follows same pattern), T17 (pipeline reads ConversionOptions)
  - **Blocked By**: None

  **References**:
  - **Pattern References**:
    - `src/img2svg/cli.py:145-157` — `_mode_callback` validator pattern. Match for `_preprocess_callback` and `_quality_callback`.
    - `src/img2svg/cli.py:298-303` — `--mode` flag with `callback=_mode_callback`. Match for new flags.
    - `src/img2svg/cli.py:210-227` — `_build_options()` builds `ConversionOptions` from CLI args. Update to include new fields.
    - `src/img2svg/models.py:128-147` — `ConversionOptions` with Pydantic `Field(ge=..., le=...)` constraints. Match.
  - **API/Type References**:
    - `src/img2svg/models.py:147` — `palette_size: int = Field(default=8, ge=2, le=64)` — same constraints as our `max_colors`.
  - **WHY Each Reference Matters**:
    - Existing callback pattern is the canonical way to validate enum-like CLI args.
    - `_build_options` is the bridge between CLI args and Pydantic model.
    - The Pydantic constraint pattern (`Field(ge=..., le=...)`) provides runtime validation.

  **Acceptance Criteria**:
  - [ ] 6 new CLI flags accepted by `_convert_cmd`
  - [ ] 2 new callback validators (`_preprocess_callback`, `_quality_callback`)
  - [ ] 5 new fields in `ConversionOptions`
  - [ ] `--mode` help text shows all 10 modes
  - [ ] Bad values for `--preprocess` and `--quality` raise `typer.BadParameter` with helpful message
  - [ ] `--no-preprocess` and `--preprocess none` produce equivalent `ConversionOptions`
  - [ ] All existing tests still pass
  - [ ] `uv run pytest tests/test_cli.py tests/test_models.py` → all pass
  - [ ] `uv run ruff check src/img2svg/cli.py src/img2svg/models.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All 6 flags accepted by CLI
    Tool: Bash (uv run python with typer testing)
    Preconditions: None
    Steps:
      1. `from typer.testing import CliRunner`
      2. `from img2svg.cli import app`
      3. `runner = CliRunner()`
      4. For each new flag: invoke with `--{flag} {valid_value}` on a non-existent file (will fail at file check, not flag check)
      5. Assert exit code is 2 (file not found) NOT exit code 2 (BadParameter). The error message should mention "file not found", not "invalid value".
    Expected Result: All 6 flags are recognized (parse stage accepts them)
    Failure Indicators: BadParameter error mentioning the flag name
    Evidence: .sisyphus/evidence/task-05-flags-accepted.txt

  Scenario: Bad values raise BadParameter with helpful message
    Tool: Bash (uv run python with typer testing)
    Preconditions: None
    Steps:
      1. `from typer.testing import CliRunner`
      2. `from img2svg.cli import app`
      3. Test cases:
         - `--preprocess unknown` → should fail with "invalid preprocess 'unknown'. Valid: none, auto, light, medium, heavy, all"
         - `--quality ultra` → should fail with "invalid quality 'ultra'. Valid: draft, standard, premium"
         - `--denoise 11` → should fail (out of range 0-10)
         - `--sharpen -1` → should fail (out of range 0-10)
         - `--max-colors 100` → should fail (out of range 0-64)
      4. For each: assert BadParameter raised OR exit code 2
    Expected Result: Bad values rejected with helpful message
    Failure Indicators: Silent acceptance, generic error
    Evidence: .sisyphus/evidence/task-05-validation.txt

  Scenario: Help text shows all 10 modes
    Tool: Bash (uv run img2svg --help)
    Preconditions: None
    Steps:
      1. `uv run img2svg convert --help`
      2. Capture stdout
      3. Assert all 10 mode names appear: "auto, labels, visual, annotated, trace, poster, detailed, edge, watercolor, segmented"
    Expected Result: Help text lists all 10 modes
    Failure Indicators: Missing mode in help text
    Evidence: .sisyphus/evidence/task-05-help.txt

  Scenario: ConversionOptions with new fields round-trip through kwargs
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.models import ConversionOptions`
      2. Build: `opts = ConversionOptions(preprocess="medium", denoise=5, sharpen=7, max_colors=32, quality="premium")`
      3. Assert `opts.preprocess == "medium"`
      4. Assert `opts.denoise == 5`
      5. Assert `opts.sharpen == 7`
      6. Assert `opts.max_colors == 32`
      7. Assert `opts.quality == "premium"`
    Expected Result: All 5 new fields stored correctly
    Failure Indicators: ValidationError, wrong default
    Evidence: .sisyphus/evidence/task-05-conversion-options.json
  ```

  **Evidence to Capture**:
  - [ ] `task-05-flags-accepted.txt`
  - [ ] `task-05-validation.txt`
  - [ ] `task-05-help.txt`
  - [ ] `task-05-conversion-options.json`

  **Commit**: YES (Wave 1)
  - Message: `feat(cli): add --preprocess/--denoise/--sharpen/--max-colors/--quality/--no-preprocess flags`
  - Files: `src/img2svg/cli.py`, `src/img2svg/models.py`
  - Pre-commit: `uv run pytest tests/test_cli.py tests/test_models.py -q && uv run ruff check src/img2svg/cli.py src/img2svg/models.py`

- [x] 6. Add `PosterRenderer` (uses existing 'poster' preset)

  **What to do**:
  - Create `src/img2svg/renderers/poster.py` (~40 lines, mirrors `trace.py` structure)
  - Define `PosterRenderer(Renderer)` class with `preset_name: ClassVar[str] = "poster"` and `render()` method that calls `_render_with_vtracer(self, self.preset_name)` (imported from `img2svg.renderers.visual`)
  - File header copyright `Copyright (c) 2026, REVYTECH, Inc.` and SPDX `BSD-3-Clause`
  - Docstring: "PosterRenderer: traces the image with vtracer's 'poster' preset for stylized, limited-color output. The 'poster' preset has color_precision=8 for high color fidelity with stacked layers."

  **Must NOT do**:
  - Don't add preprocessing here (T17 wires it into the pipeline BEFORE the renderer)
  - Don't add bbox overlays (that's AnnotatedRenderer's job)
  - Don't reimplement `_render_with_vtracer` (import it)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial renderer that wraps existing helper
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7, T8, T9)
  - **Blocks**: T10 (registry wiring needs all renderers)
  - **Blocked By**: T2 (poster preset exists, just wiring), T3 (Mode.POSTER exists)

  **References**:
  - **Pattern References**:
    - `src/img2svg/renderers/trace.py:1-31` — entire file. TraceRenderer is the closest analog: same structure, same pattern. Copy and adapt.
  - **API/Type References**:
    - `src/img2svg/renderers/base.py:20-50` — `Renderer` ABC. Knows about `self.svg`, `self.image`, `self.detections`, `self.geometric`.
    - `src/img2svg/renderers/visual.py:76-90` — `_render_with_vtracer` helper signature: `def _render_with_vtracer(renderer: Renderer, preset: str) -> None`. Just call it.
  - **WHY Each Reference Matters**:
    - trace.py is the template. The new PosterRenderer is structurally identical — just a different preset name.
    - `_render_with_vtracer` already handles all the vtracer boilerplate (temp dir, PIL save, embed paths).

  **Acceptance Criteria**:
  - [ ] File `src/img2svg/renderers/poster.py` exists, ~40 lines
  - [ ] `PosterRenderer` class defined with `preset_name = "poster"` and `render()` method
  - [ ] File header has copyright + SPDX
  - [ ] `uv run ruff check src/img2svg/renderers/poster.py` → 0 issues
  - [ ] Can be imported: `from img2svg.renderers.poster import PosterRenderer`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: PosterRenderer has correct preset_name attribute
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.renderers.poster import PosterRenderer`
      2. Assert `PosterRenderer.preset_name == "poster"`
    Expected Result: Class has correct preset_name
    Failure Indicators: Wrong preset name, AttributeError
    Evidence: .sisyphus/evidence/task-06-poster-class.txt

  Scenario: PosterRenderer can be constructed (without invoking render)
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.renderers.poster import PosterRenderer`
      2. Try: `PosterRenderer.__init__` is inherited from `Renderer` ABC
      3. Confirm the class structure: `PosterRenderer.__bases__ == (Renderer,)`
    Expected Result: Class is a Renderer subclass
    Failure Indicators: Not a Renderer subclass
    Evidence: .sisyphus/evidence/task-06-poster-inheritance.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-06-poster-class.txt`
  - [ ] `task-06-poster-inheritance.txt`

  **Commit**: YES (Wave 2)
  - Message: `feat(renderers): add PosterRenderer (vtracer 'poster' preset)`
  - Files: `src/img2svg/renderers/poster.py`
  - Pre-commit: `uv run ruff check src/img2svg/renderers/poster.py`

- [x] 7. Add `DetailedRenderer` (uses new 'photo_hifi' preset)

  **What to do**:
  - Create `src/img2svg/renderers/detailed.py` (~40 lines, mirrors `trace.py` structure)
  - Define `DetailedRenderer(Renderer)` with `preset_name: ClassVar[str] = "photo_hifi"` and `render()` method that calls `_render_with_vtracer(self, self.preset_name)`
  - File header copyright + SPDX
  - Docstring: "DetailedRenderer: traces the image with the new 'photo_hifi' preset — high color_precision=8, low filter_speckle=4, fine path_precision=4, max_iterations=20. Designed to be paired with the pre-processing pipeline (bilateral + unsharp) for maximum photo fidelity."

  **Must NOT do**:
  - Don't add preprocessing here (T17 wires it into the pipeline BEFORE the renderer)
  - Don't reimplement the helper

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial renderer, same pattern as T6
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T8, T9)
  - **Blocks**: T10 (registry)
  - **Blocked By**: T2 (photo_hifi preset), T3 (Mode.DETAILED)

  **References**:
  - **Pattern References**:
    - `src/img2svg/renderers/trace.py` — template (same as T6)
  - **API/Type References**:
    - `src/img2svg/renderers/base.py:20-50` — `Renderer` ABC
    - `src/img2svg/renderers/visual.py:76-90` — `_render_with_vtracer` helper

  **Acceptance Criteria**:
  - [ ] File `src/img2svg/renderers/detailed.py` exists, ~40 lines
  - [ ] `DetailedRenderer` class with `preset_name = "photo_hifi"`
  - [ ] File header copyright + SPDX
  - [ ] Can be imported
  - [ ] `uv run ruff check src/img2svg/renderers/detailed.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: DetailedRenderer has correct preset_name and is a Renderer
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.renderers.detailed import DetailedRenderer`
      2. Assert `DetailedRenderer.preset_name == "photo_hifi"`
      3. Assert `Renderer in DetailedRenderer.__mro__`
    Expected Result: Correct preset, proper inheritance
    Evidence: .sisyphus/evidence/task-07-detailed-class.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-07-detailed-class.txt`

  **Commit**: YES (Wave 2)
  - Message: `feat(renderers): add DetailedRenderer (vtracer 'photo_hifi' preset)`
  - Files: `src/img2svg/renderers/detailed.py`
  - Pre-commit: `uv run ruff check src/img2svg/renderers/detailed.py`

- [x] 8. Add `EdgeRenderer` (uses new 'bw_edge' preset)

  **What to do**:
  - Create `src/img2svg/renderers/edge.py` (~40 lines, mirrors `trace.py` structure)
  - Define `EdgeRenderer(Renderer)` with `preset_name: ClassVar[str] = "bw_edge"` and `render()` method that calls `_render_with_vtracer(self, self.preset_name)`
  - File header copyright + SPDX
  - Docstring: "EdgeRenderer: traces the image with the 'bw_edge' preset (binary colormode, polygon mode). Produces line-art style SVGs with no fills — just outlines. Best paired with pre-processing (median + Canny) to extract clean edges from photos."

  **Must NOT do**:
  - Don't add preprocessing here
  - Don't reimplement the helper

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T7, T9)
  - **Blocks**: T10
  - **Blocked By**: T2 (bw_edge preset), T3 (Mode.EDGE)

  **References**: Same as T6 (template `trace.py`)

  **Acceptance Criteria**:
  - [ ] File `src/img2svg/renderers/edge.py` exists, ~40 lines
  - [ ] `EdgeRenderer` class with `preset_name = "bw_edge"`
  - [ ] File header copyright + SPDX
  - [ ] Can be imported
  - [ ] `uv run ruff check src/img2svg/renderers/edge.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: EdgeRenderer has correct preset_name and is a Renderer
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.renderers.edge import EdgeRenderer`
      2. Assert `EdgeRenderer.preset_name == "bw_edge"`
      3. Assert `Renderer in EdgeRenderer.__mro__`
    Expected Result: Correct preset, proper inheritance
    Evidence: .sisyphus/evidence/task-08-edge-class.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-08-edge-class.txt`

  **Commit**: YES (Wave 2)
  - Message: `feat(renderers): add EdgeRenderer (vtracer 'bw_edge' preset)`
  - Files: `src/img2svg/renderers/edge.py`
  - Pre-commit: `uv run ruff check src/img2svg/renderers/edge.py`

- [x] 9. Add `WatercolorRenderer` (uses new 'watercolor' preset)

  **What to do**:
  - Create `src/img2svg/renderers/watercolor.py` (~40 lines, mirrors `trace.py` structure)
  - Define `WatercolorRenderer(Renderer)` with `preset_name: ClassVar[str] = "watercolor"` and `render()` method that calls `_render_with_vtracer(self, self.preset_name)`
  - File header copyright + SPDX
  - Docstring: "WatercolorRenderer: traces the image with the 'watercolor' preset — spline mode with low splice_threshold=20 and low corner_threshold=20 for soft, organic, painterly curves. Best paired with pre-processing (bilateral + CLAHE)."

  **Must NOT do**:
  - Don't add preprocessing here
  - Don't reimplement the helper

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T7, T8)
  - **Blocks**: T10
  - **Blocked By**: T2 (watercolor preset), T3 (Mode.WATERCOLOR)

  **References**: Same as T6 (template `trace.py`)

  **Acceptance Criteria**:
  - [ ] File `src/img2svg/renderers/watercolor.py` exists, ~40 lines
  - [ ] `WatercolorRenderer` class with `preset_name = "watercolor"`
  - [ ] File header copyright + SPDX
  - [ ] Can be imported
  - [ ] `uv run ruff check src/img2svg/renderers/watercolor.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: WatercolorRenderer has correct preset_name and is a Renderer
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.renderers.watercolor import WatercolorRenderer`
      2. Assert `WatercolorRenderer.preset_name == "watercolor"`
      3. Assert `Renderer in WatercolorRenderer.__mro__`
    Expected Result: Correct preset, proper inheritance
    Evidence: .sisyphus/evidence/task-09-watercolor-class.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-09-watercolor-class.txt`

  **Commit**: YES (Wave 2)
  - Message: `feat(renderers): add WatercolorRenderer (vtracer 'watercolor' preset)`
  - Files: `src/img2svg/renderers/watercolor.py`
  - Pre-commit: `uv run ruff check src/img2svg/renderers/watercolor.py`

- [x] 10. Wire new renderers to RENDERER_REGISTRY + MODE_TO_PRESET + IMAGE_TYPE_TO_MODE

  **What to do**:
  - In `src/img2svg/pipeline.py:60-65`, add 4 new entries to `RENDERER_REGISTRY`:
    ```python
    RENDERER_REGISTRY: dict[Mode, type[Renderer]] = {
        Mode.LABELS: LabelsRenderer,
        Mode.VISUAL: VisualRenderer,
        Mode.ANNOTATED: AnnotatedRenderer,
        Mode.TRACE: TraceRenderer,
        Mode.POSTER: PosterRenderer,        # NEW
        Mode.DETAILED: DetailedRenderer,    # NEW
        Mode.EDGE: EdgeRenderer,            # NEW
        Mode.WATERCOLOR: WatercolorRenderer, # NEW
        Mode.SEGMENTED: SegmentedRenderer,  # NEW (T16)
    }
    ```
  - Add 4 new imports at top of `pipeline.py`:
    ```python
    from img2svg.renderers.poster import PosterRenderer
    from img2svg.renderers.detailed import DetailedRenderer
    from img2svg.renderers.edge import EdgeRenderer
    from img2svg.renderers.watercolor import WatercolorRenderer
    from img2svg.renderers.segmented import SegmentedRenderer
    ```
  - In `src/img2svg/presets.py:22-27`, add 5 new entries to `MODE_TO_PRESET`:
    ```python
    MODE_TO_PRESET: dict[Mode, Preset] = {
        Mode.LABELS: "logo",
        Mode.VISUAL: "default",
        Mode.ANNOTATED: "default",
        Mode.TRACE: "photo",
        Mode.POSTER: "poster",          # NEW
        Mode.DETAILED: "photo_hifi",    # NEW
        Mode.EDGE: "bw_edge",            # NEW
        Mode.WATERCOLOR: "watercolor",  # NEW
        Mode.SEGMENTED: "default",       # NEW (whole-image, per-region uses photo_hifi)
    }
    ```
  - In `src/img2svg/presets.py:12-19`, **REWRITE** `IMAGE_TYPE_TO_MODE` per user constraint (CRITICAL — auto mode must NOT pick LABELS or ANNOTATED):
    ```python
    IMAGE_TYPE_TO_MODE: dict[ImageType, Mode] = {
        ImageType.LOGO:       Mode.VISUAL,    # Clean default; no detection overlay
        ImageType.PHOTO:      Mode.DETAILED,  # Aggressive: pre-process + high-fidelity trace
        ImageType.DIAGRAM:    Mode.VISUAL,
        ImageType.SCREENSHOT: Mode.VISUAL,
        ImageType.LINE_ART:   Mode.VISUAL,    # Could be EDGE in future, but VISUAL is safer default
        ImageType.UNKNOWN:    Mode.VISUAL,
    }
    ```
  - Update existing test `tests/test_presets.py` to reflect the new mapping

  **Must NOT do**:
  - Don't add ANNOTATED or LABELS to IMAGE_TYPE_TO_MODE (user explicit constraint)
  - Don't add SEGMENTED to IMAGE_TYPE_TO_MODE (SEGMENTED requires explicit user choice; auto-picking it on every photo would be surprising)
  - Don't change behavior of existing test `test_image_type_to_mode_covers_all_types` beyond updating expected values

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wiring + dict updates
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6-T9 conceptually, but actually depends on them)
  - **Parallel Group**: Wave 2 final
  - **Blocks**: T17 (pipeline integration depends on registry completeness)
  - **Blocked By**: T6, T7, T8, T9, T3 (renderer classes must exist for the registry; mode values for MODE_TO_PRESET)

  **References**:
  - **Pattern References**:
    - `src/img2svg/pipeline.py:60-65` — existing `RENDERER_REGISTRY`. Match dict style.
    - `src/img2svg/presets.py:12-19` — existing `IMAGE_TYPE_TO_MODE`. **Replace with new mapping per user constraint**.
    - `src/img2svg/presets.py:22-27` — existing `MODE_TO_PRESET`. Add new entries.
  - **API/Type References**:
    - `src/img2svg/enums.py:23-30` — `Mode` enum. All new modes are already added (T3).
  - **Test References**:
    - `tests/test_presets.py:17-26` — `test_image_type_to_mode_covers_all_types` and `test_mode_to_preset_covers_non_auto_modes`. These need updating.
  - **WHY Each Reference Matters**:
    - Existing dict patterns show the canonical style.
    - User's explicit constraint: "Annotated mode should not be included in auto when rendering, annotated must be explicitly requested, as well as label" — this is the test that enforces the new behavior.
    - Existing test coverage on `IMAGE_TYPE_TO_MODE` and `MODE_TO_PRESET` is comprehensive; just update expected values.

  **Acceptance Criteria**:
  - [ ] `RENDERER_REGISTRY` has 9 entries (5 existing + 4 new; SEGMENTED added in T16)
  - [ ] `MODE_TO_PRESET` has 9 entries (4 existing + 5 new)
  - [ ] `IMAGE_TYPE_TO_MODE` NO LONGER maps to `LABELS` or `ANNOTATED`
  - [ ] `IMAGE_TYPE_TO_MODE[ImageType.PHOTO] == Mode.DETAILED` (aggressive default)
  - [ ] `IMAGE_TYPE_TO_MODE[ImageType.LOGO] == Mode.VISUAL` (clean default)
  - [ ] All existing tests updated to reflect new mapping
  - [ ] `test_select_mode_auto_for_each_type` still passes
  - [ ] `uv run pytest tests/test_presets.py` → all pass
  - [ ] `uv run pytest tests/test_pipeline.py -k "test_renderer_registry"` → all pass
  - [ ] `uv run ruff check src/img2svg/presets.py src/img2svg/pipeline.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: RENDERER_REGISTRY has all 9 concrete modes
    Tool: Bash (uv run python)
    Preconditions: T6-T9 complete (renderer classes exist)
    Steps:
      1. `from img2svg.pipeline import RENDERER_REGISTRY`
      2. `from img2svg.enums import Mode`
      3. Assert `len(RENDERER_REGISTRY) == 9` (was 4, added 5: POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
      4. Assert `Mode.AUTO not in RENDERER_REGISTRY` (still excluded)
      5. For each new mode: `assert RENDERER_REGISTRY[Mode.POSTER].__name__ == "PosterRenderer"`, etc.
    Expected Result: Registry has 9 entries, AUTO excluded
    Failure Indicators: Wrong count, AUTO present, missing or wrong class
    Evidence: .sisyphus/evidence/task-10-registry.json

  Scenario: IMAGE_TYPE_TO_MODE no longer maps to LABELS or ANNOTATED (user constraint)
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.presets import IMAGE_TYPE_TO_MODE`
      2. `from img2svg.enums import Mode, ImageType`
      3. For each ImageType: assert `IMAGE_TYPE_TO_MODE[it] not in {Mode.LABELS, Mode.ANNOTATED}`
      4. Assert `IMAGE_TYPE_TO_MODE[ImageType.PHOTO] == Mode.DETAILED`
      5. Assert `IMAGE_TYPE_TO_MODE[ImageType.LOGO] == Mode.VISUAL`
      6. Assert `IMAGE_TYPE_TO_MODE[ImageType.SCREENSHOT] == Mode.VISUAL`
      7. Assert `IMAGE_TYPE_TO_MODE[ImageType.LINE_ART] == Mode.VISUAL`
      8. Assert `IMAGE_TYPE_TO_MODE[ImageType.DIAGRAM] == Mode.VISUAL`
      9. Assert `IMAGE_TYPE_TO_MODE[ImageType.UNKNOWN] == Mode.VISUAL`
    Expected Result: No LABELS or ANNOTATED in IMAGE_TYPE_TO_MODE; PHOTO → DETAILED
    Failure Indicators: Any mapping to LABELS or ANNOTATED (user constraint violation)
    Evidence: .sisyphus/evidence/task-10-image-type-mode.json

  Scenario: MODE_TO_PRESET has all 9 non-AUTO modes
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.presets import MODE_TO_PRESET`
      2. `from img2svg.enums import Mode`
      3. For each non-AUTO mode: `assert m in MODE_TO_PRESET`
      4. Assert `MODE_TO_PRESET[Mode.DETAILED] == "photo_hifi"`
      5. Assert `MODE_TO_PRESET[Mode.EDGE] == "bw_edge"`
      6. Assert `MODE_TO_PRESET[Mode.WATERCOLOR] == "watercolor"`
    Expected Result: All 9 non-AUTO modes mapped to known presets
    Failure Indicators: Missing entry, wrong preset
    Evidence: .sisyphus/evidence/task-10-mode-preset.json

  Scenario: Auto-mode no longer resolves to ANNOTATED for PHOTO (end-to-end)
    Tool: Bash (uv run python)
    Preconditions: T24 (or running this scenario standalone)
    Steps:
      1. `from img2svg.presets import select_mode`
      2. `from img2svg.enums import Mode, ImageType`
      3. `mode, reasoning = select_mode(ImageType.PHOTO, Mode.AUTO)`
      4. Assert `mode == Mode.DETAILED` (NOT ANNOTATED — user constraint)
      5. Assert `"auto" in reasoning.lower()` and `"photo" in reasoning.lower()`
    Expected Result: PHOTO auto-resolves to DETAILED, not ANNOTATED
    Failure Indicators: Resolved to ANNOTATED (user constraint violation)
    Evidence: .sisyphus/evidence/task-10-auto-mode-photo.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-10-registry.json`
  - [ ] `task-10-image-type-mode.json`
  - [ ] `task-10-mode-preset.json`
  - [ ] `task-10-auto-mode-photo.txt`

  **Commit**: YES (Wave 2)
  - Message: `feat(pipeline): wire 4 new renderers to RENDERER_REGISTRY + rewrite IMAGE_TYPE_TO_MODE (no auto LABELS/ANNOTATED)`
  - Files: `src/img2svg/pipeline.py`, `src/img2svg/presets.py`, `tests/test_presets.py`
  - Pre-commit: `uv run pytest tests/test_presets.py tests/test_pipeline.py -q && uv run ruff check src/img2svg/presets.py src/img2svg/pipeline.py`

- [x] 11. Add `YOLOSegmentor` class in `detector.py` (yolo11s-seg, retina_masks=True)

  **What to do**:
  - Add new `YOLOSegmentor` class in `src/img2svg/detector.py` (after `YOLODetector`, ~80 lines)
  - Constructor: `YOLOSegmentor(model_name: str = "yolo11s-seg.pt", backend: BackendSpec | None = None)`
  - Add new `get_segmentor()` factory (parallel to `get_detector()`) with separate cache keyed on `(model_name, backend.requested, backend.index)`
  - Add new return dataclass `SegmentationResult`:
    ```python
    @dataclass
    class SegmentationResult:
        boxes: list[Detection]   # same as YOLODetector result
        masks: list[np.ndarray]  # binary H×W uint8 per detection
        polygons: list[np.ndarray]  # (P, 2) pixel polygons per detection
        orig_shape: tuple[int, int]  # (H, W) of original image
    ```
  - Method: `predict(image, conf=0.25, iou=0.6, imgsz=1024) -> SegmentationResult`
  - **CRITICAL**:
    - Use `retina_masks=True` (per YOLO research, gets masks at original image size)
    - Use `half=True` if backend is CUDA (FP16 speedup)
    - Use `verbose=False`
    - Use `masks.gt_(0.0).byte()` semantics (already in ultralytics Results.masks)
    - Match YOLODetector's lazy backend resolution + caching pattern

  **Must NOT do**:
  - Don't share cache with YOLODetector (different model variants; mixing detection and segmentation models would cause OOM)
  - Don't use `retina_masks=False` (per research, gives wrong-size masks)
  - Don't reimplement the existing YOLODetector (just add a parallel class)
  - Don't add sigmoid + 0.5 (raw masks.gt_(0.0).byte() is correct)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex YOLO integration with subtle correctness requirements (retina_masks, mask dtype, FP16, etc.)
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T12, T13, T14, T15)
  - **Blocks**: T12 (helper functions on SegmentationResult), T13 (per-region tracer uses YOLOSegmentor), T15 (regions field population)
  - **Blocked By**: None (yolo11s-seg auto-downloads on first YOLO() call, same as yolo11x.pt)

  **References**:
  - **Pattern References**:
    - `src/img2svg/detector.py:31-54` — `get_detector()` factory with cache. Mirror exactly for `get_segmentor()`.
    - `src/img2svg/detector.py:66-176` — `YOLODetector` class. Use same lazy backend resolution + error wrapping pattern.
  - **API/Type References**:
    - `src/img2svg/detector.py:144-147` — RGBA → RGB normalization pattern. Reuse for YOLOSegmentor (it has the same issue).
    - `src/img2svg/models.py:39-45` — `Detection` model. Reuse for boxes.
  - **External References** (YOLO research):
    - `model.predict(img, retina_masks=True, imgsz=1024, conf=0.25, iou=0.6, half=True, device="cuda:0", verbose=False)` — canonical call
    - `result.masks.data` is `torch.uint8` of shape `(N, H, W)` with `retina_masks=True`
    - `result.masks.xy` is list of `(P, 2)` pixel polygons
    - `result.masks.xyn` is list of normalized polygons
  - **WHY Each Reference Matters**:
    - Existing YOLODetector is the architectural template.
    - RGBA normalization is needed for 4-channel PNGs (the same fix from earlier).
    - External research confirms the API surface and gotchas.

  **Acceptance Criteria**:
  - [ ] `YOLOSegmentor` class defined with lazy backend resolution
  - [ ] `get_segmentor()` factory with separate cache from `get_detector()`
  - [ ] `SegmentationResult` dataclass defined
  - [ ] `predict()` returns `SegmentationResult` with `retina_masks=True`
  - [ ] FP16 used when backend is CUDA
  - [ ] RGBA input normalized to RGB
  - [ ] `uv run ruff check src/img2svg/detector.py` → 0 issues
  - [ ] Existing tests still pass

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: YOLOSegmentor with mocked YOLO returns SegmentationResult
    Tool: Bash (uv run python with unittest.mock)
    Preconditions: None (mocked)
    Steps:
      1. `from img2svg.detector import YOLOSegmentor, SegmentationResult`
      2. `from img2svg.models import BackendSpec`
      3. Mock `ultralytics.YOLO` to return a mock Results object
      4. Construct: `seg = YOLOSegmentor(model_name="yolo11s-seg.pt", backend=BackendSpec(requested="cpu"))`
      5. Call: `result = seg.predict(np.zeros((100, 100, 3), dtype=np.uint8))`
      6. Assert `result.orig_shape == (100, 100)`
      7. Assert `isinstance(result, SegmentationResult)`
    Expected Result: Returns SegmentationResult with correct orig_shape
    Failure Indicators: Returns wrong type, wrong orig_shape
    Evidence: .sisyphus/evidence/task-11-segmentor-mock.json

  Scenario: get_segmentor caches separately from get_detector
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.detector import get_detector, get_segmentor`
      2. `from img2svg.models import BackendSpec`
      3. `d1 = get_detector(model_name="yolo11x.pt", backend=BackendSpec(requested="cpu"))`
      4. `s1 = get_segmentor(model_name="yolo11s-seg.pt", backend=BackendSpec(requested="cpu"))`
      5. `d2 = get_detector(model_name="yolo11x.pt", backend=BackendSpec(requested="cpu"))`
      6. `s2 = get_segmentor(model_name="yolo11s-seg.pt", backend=BackendSpec(requested="cpu"))`
      7. Assert `d1 is d2` (same detector)
      8. Assert `s1 is s2` (same segmentor)
      9. Assert `d1 is not s1` (different classes)
    Expected Result: Both cached correctly, separate caches
    Failure Indicators: Wrong identity, cache collision
    Evidence: .sisyphus/evidence/task-11-cache-separation.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-11-segmentor-mock.json`
  - [ ] `task-11-cache-separation.txt`

  **Commit**: YES (Wave 3)
  - Message: `feat(detector): add YOLOSegmentor with retina_masks=True + get_segmentor factory`
  - Files: `src/img2svg/detector.py`
  - Pre-commit: `uv run pytest tests/test_detector.py -q && uv run ruff check src/img2svg/detector.py`

- [x] 12. Add mask extraction helpers (xy, area, bbox) in `detector.py`

  **What to do**:
  - Add 3 new module-level functions in `src/img2svg/detector.py` (after `YOLOSegmentor`):
    - `extract_polygons(masks_data: list[np.ndarray]) -> list[np.ndarray]`: For each binary mask, find external contour via `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` and return as `(P, 2)` float32 array. If empty, return `np.zeros((0, 2))`.
    - `compute_mask_area(mask: np.ndarray) -> int`: `int((mask > 0).sum())`.
    - `get_tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]`: `(x1, y1, x2, y2)` from `np.where(mask > 0)`. If empty mask, return `(0, 0, 0, 0)`.
  - All functions take `np.ndarray` (H, W) uint8 binary mask
  - Add unit tests in `tests/test_segmentation.py` (new file) covering: empty mask, single object, multiple disjoint objects, full-image mask

  **Must NOT do**:
  - Don't add smoothing/approximation to the polygon (caller can use `cv2.approxPolyDP` if needed)
  - Don't preserve holes (RETR_EXTERNAL only — matches YOLO behavior)
  - Don't add docstrings longer than 2 lines (match codebase style)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small helper functions, straightforward OpenCV calls
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11, T13, T14, T15)
  - **Blocks**: T13 (per-region tracer uses these helpers)
  - **Blocked By**: None (functions are independent)

  **References**:
  - **Pattern References**:
    - `src/img2svg/detector.py:144-147` — RGB normalization pattern. Match for type handling.
  - **External References** (YOLO research):
    - `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` — produces `(N, 1, 2)` contours; need `.reshape(-1, 2)` for `(P, 2)`
    - `np.where(mask > 0)` returns `(ys, xs)`; use `xs.min(), xs.max(), ys.min(), ys.max()` for bbox
  - **WHY Each Reference Matters**:
    - Existing detector.py shows the module-level function style.
    - External research confirms the canonical implementations.

  **Acceptance Criteria**:
  - [ ] 3 new functions in `detector.py`: `extract_polygons`, `compute_mask_area`, `get_tight_bbox`
  - [ ] All handle empty masks gracefully (return zeros)
  - [ ] `tests/test_segmentation.py` exists with ≥4 test cases per function
  - [ ] `uv run pytest tests/test_segmentation.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/detector.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: extract_polygons returns correct shape for single-object mask
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.detector import extract_polygons`
      2. Create 100x100 mask with a single square (10,10)-(20,20)
      3. `polys = extract_polygons([mask])`
      4. Assert `len(polys) == 1`
      5. Assert `polys[0].shape == (4, 2)` (4 corners of square)
      6. Assert `polys[0].dtype == np.float32`
    Expected Result: 4-corner polygon, float32
    Failure Indicators: Wrong count, wrong shape, wrong dtype
    Evidence: .sisyphus/evidence/task-12-extract-polygons.txt

  Scenario: extract_polygons handles empty mask
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.detector import extract_polygons`
      2. `polys = extract_polygons([np.zeros((100, 100), dtype=np.uint8)])`
      3. Assert `polys[0].shape == (0, 2)` (no points)
    Expected Result: Empty array, not an error
    Failure Indicators: IndexError, wrong shape
    Evidence: .sisyphus/evidence/task-12-empty-mask.txt

  Scenario: compute_mask_area counts correctly
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.detector import compute_mask_area`
      2. `mask = np.zeros((10, 10), dtype=np.uint8); mask[2:5, 3:7] = 1`
      3. `area = compute_mask_area(mask)`
      4. Assert `area == 12` (3 rows × 4 cols)
    Expected Result: Exact pixel count
    Failure Indicators: Off-by-one, double-counting
    Evidence: .sisyphus/evidence/task-12-area.txt

  Scenario: get_tight_bbox returns mask extent (not detection bbox)
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.detector import get_tight_bbox`
      2. `mask = np.zeros((100, 100), dtype=np.uint8); mask[10:20, 30:50] = 1`
      3. `bbox = get_tight_bbox(mask)`
      4. Assert `bbox == (30, 10, 50, 20)` (x1, y1, x2, y2)
    Expected Result: Tight mask bbox, exclusive end
    Failure Indicators: Inclusive end, swapped x/y
    Evidence: .sisyphus/evidence/task-12-bbox.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-12-extract-polygons.txt`
  - [ ] `task-12-empty-mask.txt`
  - [ ] `task-12-area.txt`
  - [ ] `task-12-bbox.txt`

  **Commit**: YES (Wave 3)
  - Message: `feat(detector): add mask extraction helpers (extract_polygons, compute_mask_area, get_tight_bbox)`
  - Files: `src/img2svg/detector.py`, `tests/test_segmentation.py`
  - Pre-commit: `uv run pytest tests/test_segmentation.py -q && uv run ruff check src/img2svg/detector.py`

- [x] 13. Add per-region vtracer tracer in `detector.py` (crop, trace, return with offset)

  **What to do**:
  - Add new function `trace_region(image: np.ndarray, mask: np.ndarray, preset: str = "photo_hifi") -> tuple[list[str], tuple[int, int]]` in `src/img2svg/detector.py`:
    1. Get tight bbox via `get_tight_bbox(mask)`
    2. Crop image and mask to bbox: `crop_img = image[y1:y2, x1:x2]`, `crop_mask = mask[y1:y2, x1:x2]`
    3. Build RGBA: `rgba = np.dstack([crop_img, crop_mask])`
    4. Save RGBA to temp PNG
    5. Run `vtracer.convert_image_to_svg_py(tmp_png, tmp_svg, **PRESETS[preset])`
    6. Parse tmp_svg with lxml, extract `<path d="...">` elements
    7. Return `(list_of_d_strings, (x1, y1))` so caller knows the offset
  - Add unit tests in `tests/test_segmentation.py` (mock vtracer, verify the function signature)

  **Must NOT do**:
  - Don't call YOLO here (that's T11)
  - Don't write to the final output (caller does that)
  - Don't do multi-region logic here (single region only)
  - Don't use `cv2.RETR_CCOMP` (use `RETR_EXTERNAL` like YOLO)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Subtle integration: RGBA masking, temp file lifecycle, vtracer kwargs
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11, T12, T14, T15)
  - **Blocks**: T16 (SegmentedRenderer uses this)
  - **Blocked By**: T12 (helpers used here), T2 (PRESETS must have the preset)

  **References**:
  - **Pattern References**:
    - `src/img2svg/renderers/visual.py:83-90` — temp file lifecycle for vtracer. Use same pattern: `tempfile.TemporaryDirectory()`, save input PNG, run vtracer, parse output, embed.
    - `src/img2svg/renderers/visual.py:58-73` — `_embed_vtracer_paths` shows how to parse vtracer SVG with lxml.
  - **API/Type References**:
    - `src/img2svg/vectorizer.py:103-115` — `VtracerVectorizer.vectorize` could be used instead of direct vtracer call (preferred for consistency).
  - **External References**:
    - `vtracer.convert_image_to_svg_py(input_png, output_svg, **kwargs)` — file-to-file
    - `lxml.etree.parse(svg_path)` — parse output, find `f"{{{SVG_NS}}}path"` elements
  - **WHY Each Reference Matters**:
    - Existing temp file pattern is correct and tested.
    - `VtracerVectorizer` wraps vtracer with preset validation; using it is more consistent.

  **Acceptance Criteria**:
  - [ ] `trace_region` function in `detector.py` with signature `(image, mask, preset) -> (list[str], tuple[int, int])`
  - [ ] Uses `VtracerVectorizer` (not direct vtracer call) for consistency
  - [ ] Handles empty mask gracefully (returns `([], (0, 0))`)
  - [ ] Tests in `tests/test_segmentation.py` mock vtracer and verify return shape
  - [ ] `uv run pytest tests/test_segmentation.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/detector.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: trace_region returns paths and offset for a valid mask
    Tool: Bash (uv run python with mock)
    Preconditions: None (vtracer mocked)
    Steps:
      1. Mock `vtracer.convert_image_to_svg_py` to write a fake SVG with 2 paths
      2. `from img2svg.detector import trace_region`
      3. `img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)`
      4. `mask = np.zeros((100, 100), dtype=np.uint8); mask[10:30, 20:50] = 1`
      5. `paths, offset = trace_region(img, mask, preset="photo_hifi")`
      6. Assert `len(paths) == 2`
      7. Assert `offset == (20, 10)` (x1, y1 of mask bbox)
    Expected Result: 2 paths, correct offset
    Failure Indicators: Wrong path count, wrong offset
    Evidence: .sisyphus/evidence/task-13-trace-region.txt

  Scenario: trace_region handles empty mask
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.detector import trace_region`
      2. `img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)`
      3. `mask = np.zeros((100, 100), dtype=np.uint8)`
      4. `paths, offset = trace_region(img, mask)`
      5. Assert `paths == []`
      6. Assert `offset == (0, 0)` (empty mask bbox)
    Expected Result: Empty list, no error
    Failure Indicators: IndexError, RuntimeError
    Evidence: .sisyphus/evidence/task-13-empty-region.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-13-trace-region.txt`
  - [ ] `task-13-empty-region.txt`

  **Commit**: YES (Wave 3)
  - Message: `feat(detector): add trace_region for per-region vtracer tracing with offset`
  - Files: `src/img2svg/detector.py`, `tests/test_segmentation.py`
  - Pre-commit: `uv run pytest tests/test_segmentation.py -q && uv run ruff check src/img2svg/detector.py`

- [x] 14. Add CLI flags part 2 (--seg-model, --no-seg) + ConversionOptions fields

  **What to do**:
  - In `src/img2svg/cli.py`, add 2 new Typer options to `_convert_cmd()`:
    - `--seg-model`: `str` with `_seg_model_callback` validator (values: `yolo11n-seg|yolo11s-seg|yolo11m-seg|yolo11l-seg|yolo11x-seg`). Default: `yolo11s-seg`
    - `--no-seg`: `bool` flag (disables segmentation even if mode would use it)
  - Add 2 new fields to `ConversionOptions` (after `quality`):
    - `seg_model: str = "yolo11s-seg"`
    - `no_seg: bool = False`
  - Update `_build_options()` to pass new fields
  - Add `_seg_model_callback` validator after `_quality_callback`
  - Update help text for `--mode` if needed (already lists 10 modes from T5)

  **Must NOT do**:
  - Don't add other YOLO seg flags (keep it minimal: model + on/off)
  - Don't validate that yolo11x-seg is available (let ultralytics download on first use)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Same pattern as T5 (Typer option + ConversionOptions field + callback)
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11, T12, T13, T15)
  - **Blocks**: T17 (pipeline reads ConversionOptions.seg_model)
  - **Blocked By**: None (ConversionOptions is already extensible from T5)

  **References**:
  - **Pattern References**:
    - `src/img2svg/cli.py:_mode_callback` (lines 145-157) — validator pattern
    - `src/img2svg/cli.py:_quality_callback` (added in T5) — pattern for enum-like strings
    - `src/img2svg/cli.py:_build_options` (lines 210-227) — pattern for adding fields
    - `src/img2svg/models.py:ConversionOptions` — Pydantic field pattern
  - **WHY Each Reference Matters**:
    - Established pattern from T5. Just follow it.

  **Acceptance Criteria**:
  - [ ] 2 new CLI flags accepted
  - [ ] 2 new fields in `ConversionOptions`
  - [ ] `_seg_model_callback` validator
  - [ ] Bad `--seg-model` value raises `typer.BadParameter`
  - [ ] Default values applied when flags not passed
  - [ ] `uv run pytest tests/test_cli.py tests/test_models.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/cli.py src/img2svg/models.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: --seg-model flag accepts valid values, rejects invalid
    Tool: Bash (uv run python with typer testing)
    Preconditions: None
    Steps:
      1. `from typer.testing import CliRunner`
      2. `from img2svg.cli import app`
      3. Valid: `runner.invoke(app, ["convert", "--seg-model", "yolo11n-seg", "/tmp/x.png", "-o", "/tmp/x.svg"])` → exit 2 (file not found, not BadParameter)
      4. Invalid: `runner.invoke(app, ["convert", "--seg-model", "yolo99-seg", "/tmp/x.png", "-o", "/tmp/x.svg"])` → exit 2 (BadParameter)
    Expected Result: Valid values accepted, invalid rejected
    Failure Indicators: BadParameter on valid values, silent acceptance on invalid
    Evidence: .sisyphus/evidence/task-14-seg-model.txt

  Scenario: --no-seg flag sets ConversionOptions.no_seg=True
    Tool: Bash (uv run python)
    Preconditions: None
    Steps:
      1. `from img2svg.models import ConversionOptions`
      2. Build: `opts = ConversionOptions(no_seg=True)`
      3. Assert `opts.no_seg is True`
    Expected Result: Flag stored correctly
    Evidence: .sisyphus/evidence/task-14-no-seg.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-14-seg-model.txt`
  - [ ] `task-14-no-seg.txt`

  **Commit**: YES (Wave 3)
  - Message: `feat(cli): add --seg-model and --no-seg flags + ConversionOptions.seg_model/no_seg fields`
  - Files: `src/img2svg/cli.py`, `src/img2svg/models.py`
  - Pre-commit: `uv run pytest tests/test_cli.py tests/test_models.py -q && uv run ruff check src/img2svg/cli.py src/img2svg/models.py`

- [x] 15. Update pipeline to populate Sidecar.regions + model_variant for SEGMENTED mode

  **What to do**:
  - In `src/img2svg/pipeline.py:Pipeline.run()`, add a new step (after step 6 "Per-ROI analysis skipped", before step 7 "Build empty SVG document"):
    - When `mode_used == Mode.SEGMENTED`:
      1. Get `get_segmentor(model_name=options.seg_model, backend=options.backend)` (unless `options.no_seg`)
      2. Run `segmentor.predict(...)` → `SegmentationResult`
      3. For each region:
         - `area = compute_mask_area(result.masks[i])`
         - `bbox = get_tight_bbox(result.masks[i])`
         - `polygon = result.polygons[i]`
         - Append `RegionInfo(...)` to a list
      4. Set `sidecar.regions = [RegionInfo(...), ...]`
      5. Set `sidecar.model_variant = options.seg_model` (e.g., "yolo11s-seg")
      6. Store in a pipeline attribute for use by SegmentedRenderer
  - Add timing entry: `timings["segment"] = time.perf_counter() - t0`
  - All of this happens in a new `if mode_used == Mode.SEGMENTED and not options.no_seg` block

  **Must NOT do**:
  - Don't run segmentation for non-SEGMENTED modes
  - Don't add segmentation to the existing YOLODetector (separate model)
  - Don't write to disk here (renderer does that)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Pipeline integration with timing + new state
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11, T12, T13, T14)
  - **Blocks**: T16 (SegmentedRenderer reads pipeline state), T17 (pipeline integration)
  - **Blocked By**: T3 (Mode.SEGMENTED exists), T11 (YOLOSegmentor exists), T12 (helpers exist), T14 (seg_model field exists)

  **References**:
  - **Pattern References**:
    - `src/img2svg/pipeline.py:158-167` — existing YOLO detection step. Mirror structure for segmentation.
    - `src/img2svg/pipeline.py:192-208` — Sidecar construction. Add `regions=...` and `model_variant=...` here.
  - **API/Type References**:
    - `src/img2svg/detector.py:YOLOSegmentor` (from T11) — for predict call
    - `src/img2svg/detector.py:compute_mask_area, get_tight_bbox` (from T12) — for RegionInfo population
    - `src/img2svg/models.py:RegionInfo` (from T4) — for sidecar field
  - **WHY Each Reference Matters**:
    - The existing YOLO detection step is the template for the new segmentation step.
    - Sidecar construction is where the new fields get populated.

  **Acceptance Criteria**:
  - [ ] New `if mode_used == Mode.SEGMENTED` block in `pipeline.run()`
  - [ ] `sidecar.regions` populated for SEGMENTED mode (empty for other modes)
  - [ ] `sidecar.model_variant` set to `options.seg_model` (e.g., "yolo11s-seg")
  - [ ] `timings["segment"]` recorded
  - [ ] Existing tests still pass (non-SEGMENTED modes are unaffected)
  - [ ] `uv run pytest tests/test_pipeline.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/pipeline.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: SEGMENTED mode populates sidecar.regions (mocked segmentor)
    Tool: Bash (uv run python with mock)
    Preconditions: None (segmentor mocked)
    Steps:
      1. `from img2svg.pipeline import Pipeline`
      2. `from img2svg.enums import Mode`
      3. `from img2svg.models import ConversionOptions, BackendSpec, Detection, BoundingBox`
      4. `from img2svg.detector import SegmentationResult`
      5. Mock `img2svg.pipeline.get_segmentor` to return mock with `predict` returning synthetic SegmentationResult
      6. Run pipeline with `mode=Mode.SEGMENTED, seg_model="yolo11s-seg.pt"`
      7. Assert `result.sidecar.regions` has 1+ entries
      8. Assert `result.sidecar.model_variant == "yolo11s-seg"`
      9. Assert `result.sidecar.timings["segment"] >= 0.0`
    Expected Result: regions populated, model_variant set, timing recorded
    Failure Indicators: Empty regions, wrong model_variant, missing timing
    Evidence: .sisyphus/evidence/task-15-seg-sidecar.json

  Scenario: Non-SEGMENTED modes do not populate regions
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Run pipeline with `mode=Mode.LABELS` (or Mode.VISUAL, Mode.TRACE)
      2. Assert `result.sidecar.regions == []`
      3. Assert `result.sidecar.model_variant == ""`
    Expected Result: Empty regions, empty model_variant
    Failure Indicators: Populated regions for non-SEGMENTED modes
    Evidence: .sisyphus/evidence/task-15-non-seg-empty.json
  ```

  **Evidence to Capture**:
  - [ ] `task-15-seg-sidecar.json`
  - [ ] `task-15-non-seg-empty.json`

  **Commit**: YES (Wave 3)
  - Message: `feat(pipeline): populate Sidecar.regions + model_variant for SEGMENTED mode`
  - Files: `src/img2svg/pipeline.py`
  - Pre-commit: `uv run pytest tests/test_pipeline.py -q && uv run ruff check src/img2svg/pipeline.py`

- [x] 16. Add `SegmentedRenderer` (multi-layer SVG: background + per-region groups)

  **What to do**:
  - Create `src/img2svg/renderers/segmented.py` (~150 lines)
  - Define `SegmentedRenderer(Renderer)` class
  - Add `_segmentation_result: SegmentationResult | None` instance attribute (set externally by pipeline before render is called)
  - Method `set_segmentation(result: SegmentationResult)` to inject the result
  - Method `render()`:
    1. Get segmentation result from `self._segmentation_result`. If `None` or empty, fall back to `VisualRenderer` behavior (whole-image vtracer).
    2. **Background extraction**: Create `bg_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255`. For each region mask, `bg_mask = cv2.subtract(bg_mask, mask)`. Apply: `bg_image = image.copy(); bg_image[bg_mask == 0] = 255` (white-fill background).
    3. Trace background via `VtracerVectorizer(preset="default")`. Save to temp PNG, run vtracer, parse paths.
    4. Embed background paths as `<g id="background" data-role="background">`.
    5. For each region in segmentation result (in order of confidence desc):
       - `paths, offset = trace_region(image, mask, preset="photo_hifi")` (from T13)
       - Embed as `<g id="obj_{class_name}_{idx}" data-class="{class_name}" data-conf="{confidence}">` with `transform="translate({offset[0]} {offset[1]})"` on inner group
    6. If a region trace produces 0 paths (tiny region), skip the group (don't add empty `<g>`)
  - File header copyright + SPDX
  - Docstring: "SegmentedRenderer: produces a multi-layer editable SVG. Each YOLO-detected object becomes its own <g id='obj_class_idx'> group with semantic class name and confidence. Background is traced separately. Empty detections → falls back to VisualRenderer."

  **Must NOT do**:
  - Don't call YOLO here (pipeline does that in T15)
  - Don't use `_render_with_vtracer` helper (it's for whole-image only)
  - Don't add bbox overlays (multi-layer is enough)
  - Don't reimplement `trace_region` (import from `img2svg.detector`)
  - Don't create a separate VtracerVectorizer per region with different preset names — use ONE preset per region (photo_hifi for foreground, default for background)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex multi-layer SVG assembly with z-order and translation transforms
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T13)
  - **Parallel Group**: Wave 4 (with T17)
  - **Blocks**: T17 (pipeline integration needs SegmentedRenderer to be importable)
  - **Blocked By**: T13 (trace_region function)

  **References**:
  - **Pattern References**:
    - `src/img2svg/renderers/annotated.py:65-101` — AnnotatedRenderer shows the multi-group pattern (vtracer background + bbox overlays). Mirror for multi-region.
    - `src/img2svg/renderers/visual.py:76-90` — `_render_with_vtracer` helper. Reuse for background trace.
    - `src/img2svg/renderers/visual.py:58-73` — `_embed_vtracer_paths` shows how to parse vtracer output and embed as a group.
  - **API/Type References**:
    - `src/img2svg/detector.py:trace_region` (from T13) — for per-region tracing
    - `src/img2svg/detector.py:SegmentationResult` (from T11) — for accessing regions
  - **WHY Each Reference Matters**:
    - AnnotatedRenderer is the closest analog: it does background + overlays. SegmentedRenderer does background + per-region groups.
    - The existing embed helper handles vtracer output parsing correctly.

  **Acceptance Criteria**:
  - [ ] `SegmentedRenderer` class in `src/img2svg/renderers/segmented.py` (~150 lines)
  - [ ] `set_segmentation(result)` method to inject the result
  - [ ] `render()` produces multi-layer SVG with `<g id="background">` and `<g id="obj_...">` per region
  - [ ] Background is white-filled in foreground regions before tracing
  - [ ] Empty segmentation result falls back to VisualRenderer behavior
  - [ ] Tiny regions (0 paths) are skipped (no empty groups)
  - [ ] `uv run ruff check src/img2svg/renderers/segmented.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: SegmentedRenderer produces multi-layer SVG with mocked segmentation
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock `vtracer.convert_image_to_svg_py` to write a fake SVG with 1 path
      2. `from img2svg.renderers.segmented import SegmentedRenderer`
      3. `from img2svg.detector import SegmentationResult`
      4. `from img2svg.svg_builder import SVGDocument`
      5. `from img2svg.loader import LoadedImage`
      6. Build a small `LoadedImage` (100x100 RGB)
      7. Build `SegmentationResult` with 2 regions (person bbox, dog bbox)
      8. Build `SVGDocument(100, 100)`, instantiate renderer, call `set_segmentation()`, call `render()`
      9. Assert `len(svg.root.findall(".//{http://www.w3.org/2000/svg}g")) >= 3` (background + 2 objects)
      10. Assert any `g` has `id` starting with `obj_`
    Expected Result: Multi-layer SVG with background + per-object groups
    Failure Indicators: Only 1 group, missing obj_ prefix
    Evidence: .sisyphus/evidence/task-16-segmented-render.svg

  Scenario: Empty segmentation falls back to VisualRenderer behavior
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock vtracer
      2. Build `SegmentationResult` with `masks=[], polygons=[], boxes=[]`
      3. Render
      4. Assert output has the vtracer-output group (whole-image trace) but no `obj_` groups
    Expected Result: Fallback works, no empty object groups
    Failure Indicators: Crashes on empty segmentation
    Evidence: .sisyphus/evidence/task-16-empty-fallback.svg
  ```

  **Evidence to Capture**:
  - [ ] `task-16-segmented-render.svg` — sample multi-layer output
  - [ ] `task-16-empty-fallback.svg` — empty segmentation fallback

  **Commit**: YES (Wave 4)
  - Message: `feat(renderers): add SegmentedRenderer (multi-layer SVG with per-region groups)`
  - Files: `src/img2svg/renderers/segmented.py`
  - Pre-commit: `uv run ruff check src/img2svg/renderers/segmented.py`

- [x] 17. Wire preprocessing into `pipeline.py` (post-load, pre-vtracer; mode-driven default)

  **What to do**:
  - In `src/img2svg/pipeline.py:Pipeline.run()`, add a new step (after step 5 YOLO detection, before step 6 "Per-ROI analysis skipped"):
    - If `options.preprocess != "none"` AND `mode_used in {Mode.VISUAL, Mode.TRACE, Mode.POSTER, Mode.DETAILED, Mode.EDGE, Mode.WATERCOLOR}` (i.e., not LABELS, not ANNOTATED):
      1. Resolve preprocessing level: `level = options.preprocess` if `!= "auto"` else `_mode_to_preprocess_level(mode_used)` (e.g., DETAILED → "light", POSTER → "medium", EDGE → "edge")
      2. Get steps: `steps = PREPROCESSING_PRESETS[level]` (from T1)
      3. Apply: `pipeline = PreprocessingPipeline(steps); preprocessed = pipeline.apply(loaded.np_array)`
      4. Mutate `loaded` to use `preprocessed` as `np_array` (preserve alpha, dtype, shape)
      5. Set `sidecar.preprocessing = pipeline.steps_applied()`
      6. Add timing: `timings["preprocess"] = ...`
  - Add helper function in `preprocessing.py`:
    ```python
    def mode_to_preprocess_level(mode: Mode) -> str:
        """Map Mode to default preprocessing level. Used when options.preprocess == 'auto'."""
        return {
            Mode.DETAILED: "light",   # bilateral + unsharp
            Mode.POSTER: "medium",    # light + posterize
            Mode.EDGE: "edge",        # median + canny
            Mode.WATERCOLOR: "light", # bilateral + unsharp (no CLAHE)
            Mode.VISUAL: "none",      # default: no preprocessing
            Mode.TRACE: "light",      # trace benefits from denoise
        }.get(mode, "none")
    ```
  - **Precedence**: explicit `--preprocess X` overrides mode-driven default. Explicit `--preprocess none` skips preprocessing entirely. Explicit `--no-preprocess` is alias for `--preprocess none`.

  **Must NOT do**:
  - Don't preprocess for LABELS mode (semantic, no vtracer)
  - Don't preprocess for ANNOTATED mode (annotation overlay shouldn't change underlying trace)
  - Don't add timing for skipped preprocessing
  - Don't fail if preprocessing raises (catch and log, then continue with original image)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Pipeline integration with mode-driven defaults + explicit override
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T16, but T16 is independent)
  - **Parallel Group**: Wave 4
  - **Blocks**: T18 (segmentation step in pipeline), T19 (timing instrumentation)
  - **Blocked By**: T1 (preprocessing module), T3 (Mode values)

  **References**:
  - **Pattern References**:
    - `src/img2svg/pipeline.py:158-167` — YOLO detection step. Mirror for preprocessing.
  - **API/Type References**:
    - `src/img2svg/preprocessing.py:apply_preprocessing` (from T1) — main entry
    - `src/img2svg/preprocessing.py:PREPROCESSING_PRESETS` (from T1) — preset dict
  - **WHY Each Reference Matters**:
    - Existing YOLO step is the canonical pattern for adding new pipeline stages.

  **Acceptance Criteria**:
  - [ ] New preprocessing step in `pipeline.run()` after YOLO detection
  - [ ] `mode_to_preprocess_level(mode)` helper in `preprocessing.py`
  - [ ] DETAILED mode → "light" pre-processing
  - [ ] POSTER mode → "medium" pre-processing (with posterize)
  - [ ] EDGE mode → "edge" pre-processing (with canny)
  - [ ] WATERCOLOR mode → "light" (bilateral + unsharp, NO CLAHE)
  - [ ] LABELS and ANNOTATED modes → "none" (no preprocessing)
  - [ ] VISUAL mode → "none" (preserve existing behavior)
  - [ ] `sidecar.preprocessing` populated when preprocessing runs
  - [ ] `timings["preprocess"]` recorded
  - [ ] `uv run pytest tests/test_pipeline.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/pipeline.py src/img2svg/preprocessing.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: DETAILED mode auto-applies light preprocessing
    Tool: Bash (uv run python with mock)
    Preconditions: None (YOLO + vtracer mocked)
    Steps:
      1. Mock `img2svg.pipeline.get_detector` to return empty detections
      2. Mock `img2svg.renderers.visual.VtracerVectorizer` to write fake SVG
      3. Mock `img2svg.preprocessing.apply_preprocessing` to verify it's called
      4. Run pipeline with `mode=Mode.DETAILED, preprocess="auto"`
      5. Assert `mock_apply_preprocessing` was called with `level="light"` (or the steps for "light")
      6. Assert `result.sidecar.preprocessing == ["denoise_bilateral", "sharpen_unsharp"]` (the step names)
    Expected Result: DETAILED mode triggers light preprocessing
    Failure Indicators: Preprocessing not called, wrong steps
    Evidence: .sisyphus/evidence/task-17-detailed-preprocess.json

  Scenario: --preprocess none skips preprocessing
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock detector + vtracer
      2. Mock `img2svg.preprocessing.apply_preprocessing` to assert NOT called
      3. Run pipeline with `mode=Mode.DETAILED, preprocess="none"`
      4. Assert `mock_apply_preprocessing` was NOT called
      5. Assert `result.sidecar.preprocessing == []`
    Expected Result: Explicit "none" skips preprocessing
    Failure Indicators: Preprocessing called despite --preprocess none
    Evidence: .sisyphus/evidence/task-17-no-preprocess.txt

  Scenario: LABELS mode never preprocessed
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock detector + (no vtracer needed for LABELS)
      2. Mock `img2svg.preprocessing.apply_preprocessing` to assert NOT called
      3. Run pipeline with `mode=Mode.LABELS, preprocess="auto"`
      4. Assert `mock_apply_preprocessing` was NOT called
    Expected Result: LABELS mode skips preprocessing
    Failure Indicators: Preprocessing called for LABELS mode
    Evidence: .sisyphus/evidence/task-17-labels-no-preprocess.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-17-detailed-preprocess.json`
  - [ ] `task-17-no-preprocess.txt`
  - [ ] `task-17-labels-no-preprocess.txt`

  **Commit**: YES (Wave 4)
  - Message: `feat(pipeline): add preprocessing step (mode-driven default + explicit override)`
  - Files: `src/img2svg/pipeline.py`, `src/img2svg/preprocessing.py`
  - Pre-commit: `uv run pytest tests/test_pipeline.py -q && uv run ruff check src/img2svg/pipeline.py src/img2svg/preprocessing.py`

- [x] 18. Wire segmentation into `pipeline.py` (SEGMENTED mode only; inject result into renderer)

  **What to do**:
  - In `src/img2svg/pipeline.py:Pipeline.run()`, expand the existing segmentation block (from T15) to also inject the result into the renderer:
    ```python
    if mode_used == Mode.SEGMENTED and not options.no_seg:
        segmentor = get_segmentor(model_name=options.seg_model, backend=options.backend)
        t0 = time.perf_counter()
        seg_result = segmentor.predict(loaded.np_array, conf=options.conf, iou=options.iou)
        timings["segment"] = time.perf_counter() - t0
        # ... (existing sidecar.regions + model_variant population from T15)
        # NEW: inject into renderer
        if isinstance(renderer, SegmentedRenderer):
            renderer.set_segmentation(seg_result)
    ```
  - If `--no-seg` is set OR segmentation returns empty, fall back to `VisualRenderer` behavior:
    - `if not seg_result.masks or options.no_seg: renderer_cls = VisualRenderer`
  - Add timing: `timings["vectorize"]` (sum of all per-region vtracer calls for SEGMENTED mode; 0 for other modes)

  **Must NOT do**:
  - Don't call segmentation for non-SEGMENTED modes
  - Don't fall back silently — log a warning when falling back (use `setup_logging`'s logger)
  - Don't run segmentation if model download fails (existing detector error handling should cover this)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Pipeline integration with fallback logic + renderer injection
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with T19, T20)
  - **Blocks**: T24 (end-to-end needs full pipeline)
  - **Blocked By**: T15 (regions field already in T15), T16 (SegmentedRenderer)

  **References**:
  - **Pattern References**:
    - `src/img2svg/pipeline.py:175-178` — existing renderer instantiation. Mirror for fallback logic.
  - **API/Type References**:
    - `src/img2svg/renderers/segmented.py:SegmentedRenderer` (from T16) — for the `set_segmentation()` call
  - **WHY Each Reference Matters**:
    - Existing renderer instantiation is where fallback would happen.
    - SegmentedRenderer needs the result before render() is called.

  **Acceptance Criteria**:
  - [ ] `segmentor.predict()` called for SEGMENTED mode only
  - [ ] `set_segmentation()` called on SegmentedRenderer before render
  - [ ] Fallback to VisualRenderer when `--no-seg` or empty segmentation
  - [ ] Warning logged on fallback
  - [ ] `timings["vectorize"]` recorded for SEGMENTED mode
  - [ ] `uv run pytest tests/test_pipeline.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/pipeline.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: SEGMENTED mode injects segmentation result into renderer
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock `img2svg.pipeline.get_segmentor` to return mock with `predict` returning synthetic SegmentationResult (2 regions)
      2. Mock `img2svg.renderers.segmented.SegmentedRenderer.set_segmentation` to record call
      3. Run pipeline with `mode=Mode.SEGMENTED, seg_model="yolo11s-seg.pt"`
      4. Assert `set_segmentation` was called once with non-None result
      5. Assert `timings["vectorize"]` was set
    Expected Result: Renderer receives segmentation result, timing recorded
    Failure Indicators: set_segmentation not called, missing timing
    Evidence: .sisyphus/evidence/task-18-seg-inject.json

  Scenario: --no-seg falls back to VisualRenderer with warning
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock `img2svg.pipeline.get_segmentor` to return mock
      2. Run pipeline with `mode=Mode.SEGMENTED, no_seg=True`
      3. Assert `get_segmentor` was NOT called
      4. Assert output is a valid SVG (renderer fell back to VisualRenderer)
    Expected Result: Fallback works, no segmentation call
    Failure Indicators: get_segmentor called, runtime error
    Evidence: .sisyphus/evidence/task-18-no-seg-fallback.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-18-seg-inject.json`
  - [ ] `task-18-no-seg-fallback.txt`

  **Commit**: YES (Wave 5)
  - Message: `feat(pipeline): inject segmentation result into SegmentedRenderer with fallback`
  - Files: `src/img2svg/pipeline.py`
  - Pre-commit: `uv run pytest tests/test_pipeline.py -q && uv run ruff check src/img2svg/pipeline.py`

- [x] 19. Add timing instrumentation entries for new pipeline steps

  **What to do**:
  - In `src/img2svg/pipeline.py`, add timing entries for all new steps:
    - `timings["preprocess"]` (added in T17)
    - `timings["segment"]` (added in T15)
    - `timings["vectorize"]` (added in T18)
  - Update existing test `tests/test_pipeline.py:test_pipeline_records_timings` to assert these new keys exist (with 0 or >0 values as appropriate)
  - Add per-region timing for SEGMENTED mode: `timings["vectorize_regions"] = [t1, t2, ...]`

  **Must NOT do**:
  - Don't add timing for steps that didn't run (e.g., `segment` shouldn't appear in non-SEGMENTED timings)
  - Don't break the existing `test_pipeline_records_timings` test

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial additions to existing timing dict
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with T18, T20)
  - **Blocks**: T24 (end-to-end needs accurate timings)
  - **Blocked By**: T17 (preprocess timing), T15 (segment timing), T18 (vectorize timing)

  **References**:
  - **Pattern References**:
    - `src/img2svg/pipeline.py:130-187` — existing timing pattern. Mirror.
  - **Test References**:
    - `tests/test_pipeline.py:268-279` — `test_pipeline_records_timings`. Update to include new keys.

  **Acceptance Criteria**:
  - [ ] All 3 new timing entries added
  - [ ] `test_pipeline_records_timings` updated and passes
  - [ ] `uv run pytest tests/test_pipeline.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/pipeline.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All timing keys present in sidecar
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Run pipeline with `mode=Mode.LABELS` (simple case)
      2. Assert `"load" in result.sidecar.timings`
      3. Assert `"analyze" in result.sidecar.timings`
      4. Assert `"classify" in result.sidecar.timings`
      5. Assert `"select_mode" in result.sidecar.timings`
      6. Assert `"detect" in result.sidecar.timings`
      7. Assert `"render" in result.sidecar.timings`
      8. Assert `"write" in result.sidecar.timings`
      9. Assert `"total" in result.sidecar.timings`
      10. Assert `"preprocess" in result.sidecar.timings` (always present, 0 if not run)
      11. Assert `"segment" in result.sidecar.timings` (always present, 0 if not run)
      12. Assert `"vectorize" in result.sidecar.timings` (always present, 0 if not run)
    Expected Result: All 11 timing keys present
    Failure Indicators: Missing key, wrong value
    Evidence: .sisyphus/evidence/task-19-timings.json
  ```

  **Evidence to Capture**:
  - [ ] `task-19-timings.json`

  **Commit**: YES (Wave 5)
  - Message: `feat(pipeline): add timing entries for preprocess/segment/vectorize steps`
  - Files: `src/img2svg/pipeline.py`, `tests/test_pipeline.py`
  - Pre-commit: `uv run pytest tests/test_pipeline.py -q && uv run ruff check src/img2svg/pipeline.py`

- [x] 20. Add MAX_SVG_SIZE guard with user-overridable CLI flag (default 50MB)

  **What to do**:
  - Add `MAX_SVG_SIZE_MB = 50` constant in `src/img2svg/pipeline.py`
  - Add new field to `ConversionOptions`: `max_svg_size_mb: int = Field(default=50, ge=1, le=1024)` (1MB to 1GB)
  - Add new CLI flag `--max-svg-size`: `int` (in MB), default 50, with min/max validation
  - In `pipeline.run()`, after `svg.write(output_path)`:
    - `size_mb = output_path.stat().st_size / (1024 * 1024)`
    - If `size_mb > options.max_svg_size_mb`:
      - Log warning: `f"SVG size {size_mb:.1f}MB exceeds limit {options.max_svg_size_mb}MB, skipping write"`
      - Delete the just-written file
      - Raise `OutputPathCollisionError`-like error or return `ConversionResult` with `errors=[f"SVG too large: {size_mb:.1f}MB > {options.max_svg_size_mb}MB"]`
  - Use `OutputPathCollisionError` (existing) for consistency, OR add new `SVGSizeLimitError` to `errors.py`
  - Update test_pipeline.py: add test for size limit enforcement

  **Must NOT do**:
  - Don't silently truncate the SVG
  - Don't silently fail without telling the user
  - Don't add a per-region size limit (whole-output only)
  - Don't make the default 0 (always enforce some limit)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial size check + CLI flag
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with T18, T19)
  - **Blocks**: T24 (end-to-end needs working pipeline)
  - **Blocked By**: None

  **References**:
  - **Pattern References**:
    - `src/img2svg/pipeline.py:181-183` — existing SVG write step. Add check after.
  - **API/Type References**:
    - `src/img2svg/errors.py:OutputPathCollisionError` (line 102-113) — error pattern. Reuse or add new.
  - **WHY Each Reference Matters**:
    - Existing write step is where the check goes.
    - Error pattern for size-related failures.

  **Acceptance Criteria**:
  - [ ] `MAX_SVG_SIZE_MB = 50` constant in pipeline.py
  - [ ] `max_svg_size_mb: int = 50` field in ConversionOptions
  - [ ] `--max-svg-size` CLI flag
  - [ ] Size check after `svg.write()`
  - [ ] If exceeded: log warning + delete file + raise error
  - [ ] `test_pipeline_enforces_max_svg_size` test added and passing
  - [ ] `uv run pytest tests/test_pipeline.py tests/test_cli.py -q` → all pass
  - [ ] `uv run ruff check src/img2svg/pipeline.py src/img2svg/cli.py src/img2svg/models.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: SVG exceeding limit raises error
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock vtracer to write a 100MB fake SVG
      2. Run pipeline with `max_svg_size_mb=50` and a small image
      3. Assert error raised (or `result.errors` non-empty)
      4. Assert no file at output_path
    Expected Result: Error raised, no file written
    Failure Indicators: File written despite size, silent success
    Evidence: .sisyphus/evidence/task-20-size-limit.txt

  Scenario: SVG within limit succeeds
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock vtracer to write a 1MB fake SVG
      2. Run pipeline with `max_svg_size_mb=50`
      3. Assert no error
      4. Assert file exists at output_path
    Expected Result: Success, file written
    Failure Indicators: Spurious error
    Evidence: .sisyphus/evidence/task-20-size-ok.txt

  Scenario: User override allows larger SVGs
    Tool: Bash (uv run python with mock)
    Preconditions: None
    Steps:
      1. Mock vtracer to write a 60MB fake SVG
      2. Run pipeline with `max_svg_size_mb=100` (override default 50)
      3. Assert no error
      4. Assert file exists
    Expected Result: Override works
    Failure Indicators: Override ignored
    Evidence: .sisyphus/evidence/task-20-override.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-20-size-limit.txt`
  - [ ] `task-20-size-ok.txt`
  - [ ] `task-20-override.txt`

  **Commit**: YES (Wave 5)
  - Message: `feat(pipeline): add MAX_SVG_SIZE guard with --max-svg-size CLI flag (default 50MB)`
  - Files: `src/img2svg/pipeline.py`, `src/img2svg/cli.py`, `src/img2svg/models.py`, `tests/test_pipeline.py`
  - Pre-commit: `uv run pytest tests/test_pipeline.py tests/test_cli.py -q && uv run ruff check src/img2svg/pipeline.py src/img2svg/cli.py src/img2svg/models.py`

- [ ] 21. Create new test files: `test_preprocessing.py`, `test_segmentation.py`, `test_renderers.py`

  **What to do**:
  - Create `tests/test_preprocessing.py` (~150 lines):
    - Test each of 7 filter functions: `denoise_bilateral`, `denoise_nlmeans`, `denoise_median`, `sharpen_unsharp`, `posterize`, `detect_edges_canny`, `apply_clahe_yuv`
    - Test `PreprocessingPipeline.apply()` with various step combinations
    - Test `PREPROCESSING_PRESETS` keys and structure
    - Test alpha channel preservation
    - Test dtype preservation
    - Test empty pipeline (no-op)
    - Use synthetic images (np.zeros, np.random.randint, gradients)
    - Use `tests/fixtures/photo.jpg` as real-image smoke test
  - Create `tests/test_segmentation.py` (~200 lines):
    - Test `extract_polygons`, `compute_mask_area`, `get_tight_bbox` (T12 helpers)
    - Test `trace_region` (T13) with mocked vtracer
    - Test `YOLOSegmentor` (T11) with mocked `ultralytics.YOLO`
    - Test `get_segmentor()` cache behavior
    - Use `@pytest.mark.slow` for tests that download the actual model
  - Create `tests/test_renderers.py` (~200 lines):
    - Test each of 5 new renderers: PosterRenderer, DetailedRenderer, EdgeRenderer, WatercolorRenderer, SegmentedRenderer
    - Test `RENDERER_REGISTRY` has all 9 concrete modes (T10)
    - Test `IMAGE_TYPE_TO_MODE` no longer maps to LABELS/ANNOTATED (T10)
    - Test `MODE_TO_PRESET` has all 9 non-AUTO modes (T10)
    - Test that `_render_with_vtracer` is called with correct preset
    - Mock `VtracerVectorizer` (existing pattern)
  - All tests use `tmp_path` fixture, `logo_path`, `photo_path`, `transparent_path` from `conftest.py`
  - For SEGMENTED tests, use **existing `tests/testimg/Designer (1).jpeg`** (real photo with detectable objects) instead of synthetic

  **Must NOT do**:
  - Don't use slow markers on fast tests
  - Don't use real YOLO inference in unit tests (always mock)
  - Don't add `@pytest.mark.integration` (use `@pytest.mark.slow` instead for tests that download the model)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Test file scaffolding with established patterns
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with T22, T23, T24, T25, T26)
  - **Blocks**: F1-F4 (final verification)
  - **Blocked By**: T1 (preprocessing module), T11-T13 (segmentation), T6-T10 (renderers + wiring)

  **References**:
  - **Pattern References**:
    - `tests/test_pipeline.py:1-50` — existing test pattern with mocks for `get_detector` and `VtracerVectorizer`. Mirror.
  - **API/Type References**:
    - `tests/conftest.py:34-65` — existing fixtures (`logo_path`, `photo_path`, `transparent_path`). Use these.
  - **Test References**:
    - `tests/test_pipeline.py:_FAKE_VTRACER_SVG` (line 40-44) — example of fake vtracer SVG output. Reuse.
  - **WHY Each Reference Matters**:
    - Existing patterns are the canonical way to mock heavy dependencies.
    - `tests/testimg/Designer (1).jpeg` is a real photo with detectable objects (people/products).

  **Acceptance Criteria**:
  - [ ] 3 new test files created: `test_preprocessing.py`, `test_segmentation.py`, `test_renderers.py`
  - [ ] All tests pass: `uv run pytest tests/test_preprocessing.py tests/test_segmentation.py tests/test_renderers.py -q`
  - [ ] Slow tests use `@pytest.mark.slow`
  - [ ] All new tests use mocks for vtracer + YOLO (no real downloads in CI)
  - [ ] `tests/testimg/Designer (1).jpeg` referenced in SEGMENTED tests
  - [ ] `uv run ruff check tests/` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: New test files exist and have tests
    Tool: Bash
    Preconditions: All new code from T1-T13 complete
    Steps:
      1. `ls tests/test_preprocessing.py tests/test_segmentation.py tests/test_renderers.py`
      2. All 3 files exist
      3. `grep -c "^def test_" tests/test_preprocessing.py` ≥ 10
      4. `grep -c "^def test_" tests/test_segmentation.py` ≥ 8
      5. `grep -c "^def test_" tests/test_renderers.py` ≥ 8
    Expected Result: All 3 files exist with ≥26 total tests
    Failure Indicators: Missing file, too few tests
    Evidence: .sisyphus/evidence/task-21-test-files.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-21-test-files.txt`

  **Commit**: YES (Wave 6)
  - Message: `test: add test_preprocessing, test_segmentation, test_renderers`
  - Files: `tests/test_preprocessing.py`, `tests/test_segmentation.py`, `tests/test_renderers.py`
  - Pre-commit: `uv run pytest tests/test_preprocessing.py tests/test_segmentation.py tests/test_renderers.py -q && uv run ruff check tests/`

- [ ] 22. Update existing tests for new modes + auto-mode behavior + new fields

  **What to do**:
  - Update `tests/test_presets.py`:
    - `test_select_mode_auto_uses_image_type_table`: change `Mode.ANNOTATED` → `Mode.DETAILED` for PHOTO
    - `test_image_type_to_mode_covers_all_types`: ensure new ImageType coverage (no change needed; entries exist)
    - `test_select_mode_auto_for_each_type`: update to assert no ImageType resolves to LABELS or ANNOTATED
  - Update `tests/test_pipeline.py`:
    - `test_renderer_registry_has_all_concrete_modes`: add 5 new mode assertions (POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
    - `test_pipeline_records_timings`: add 3 new keys (preprocess, segment, vectorize)
  - Update `tests/test_models.py`:
    - Add tests for `RegionInfo` Pydantic model
    - Add tests for new `Sidecar` fields (preprocessing, regions, model_variant)
    - Test JSON round-trip
  - Update `tests/test_metadata.py`:
    - Add tests for new Sidecar fields round-trip
  - Update `tests/test_cli.py`:
    - Add tests for new flags: `--preprocess`, `--denoise`, `--sharpen`, `--max-colors`, `--quality`, `--no-preprocess`, `--seg-model`, `--no-seg`
    - Test validators reject bad values
    - Test help text shows all 10 modes
  - Update `tests/test_manpage.py`:
    - Add new flags to `EXPECTED_FLAGS` list
  - Update `tests/test_examples.py`:
    - Increase `MIN_SVGS` from 8 to 13 (5 new modes × 1 photo + 8 existing = 13)
    - Add 5 new sample SVGs to `docs/examples/`
  - Update `src/img2svg/__init__.py`:
    - Export `RegionInfo` from package

  **Must NOT do**:
  - Don't change existing test behavior except for the auto-mode mapping
  - Don't add tests that require real YOLO inference (mock everything)
  - Don't add @pytest.mark.integration to existing tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Test updates following established patterns
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with T21, T23, T24, T25, T26)
  - **Blocks**: F1-F4
  - **Blocked By**: T1-T20 (need to know what to test)

  **References**:
  - **Test References**:
    - `tests/test_presets.py:39-53` — existing auto-mode tests
    - `tests/test_pipeline.py:81-87` — existing registry test
    - `tests/test_pipeline.py:268-279` — existing timings test
    - `tests/test_manpage.py:187-198` — `EXPECTED_FLAGS` list
  - **WHY Each Reference Matters**:
    - These are the tests that need updating.

  **Acceptance Criteria**:
  - [ ] All updated test files pass
  - [ ] `uv run pytest -m "not slow" -q` → 480+ passed
  - [ ] `RegionInfo` exported from `img2svg/__init__.py`
  - [ ] `MIN_SVGS = 13` in test_examples.py
  - [ ] New flags in test_manpage.py EXPECTED_FLAGS
  - [ ] `uv run ruff check tests/ src/img2svg/__init__.py` → 0 issues

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: test_presets.py auto-mode test reflects new mapping
    Tool: Bash
    Preconditions: None
    Steps:
      1. `uv run pytest tests/test_presets.py -q`
      2. All tests pass
      3. `uv run pytest tests/test_presets.py::test_select_mode_auto_uses_image_type_table -v`
      4. Assert test asserts `Mode.DETAILED` for PHOTO
    Expected Result: Test passes with new assertion
    Failure Indicators: Old assertion still there, test fails
    Evidence: .sisyphus/evidence/task-22-presets-updated.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-22-presets-updated.txt`
  - [ ] Full pytest output: `.sisyphus/evidence/task-22-pytest-output.txt`

  **Commit**: YES (Wave 6)
  - Message: `test: update existing tests for new modes + auto-mode behavior + new fields`
  - Files: `tests/test_presets.py`, `tests/test_pipeline.py`, `tests/test_models.py`, `tests/test_metadata.py`, `tests/test_cli.py`, `tests/test_manpage.py`, `tests/test_examples.py`, `src/img2svg/__init__.py`
  - Pre-commit: `uv run pytest -m "not slow" -q && uv run ruff check tests/ src/img2svg/__init__.py`

- [ ] 23. Use existing `tests/testimg/` directory for real-photo tests (no new fixtures)

  **What to do**:
  - **No new fixtures needed** — use the existing `tests/testimg/` directory which contains 50+ real photos
  - Add a `conftest.py` fixture `real_photo_path` that returns one of the test images (default: `tests/testimg/Designer (1).jpeg`):
    ```python
    @pytest.fixture
    def real_photo_path() -> Path:
        """A real photo from tests/testimg/ for SEGMENTED mode testing."""
        return _FIXTURES_DIR.parent / "testimg" / "Designer (1).jpeg"
    ```
  - Reference `tests/testimg/Designer (1).jpeg` in `tests/test_segmentation.py` for SEGMENTED mode tests
  - Reference `tests/testimg/2024-Q4.jpg` in tests that need a different photo variety
  - Reference `tests/testimg/3840x2160-dark-freebsd.png` in tests that need 4K (for performance/large-output testing)
  - Note in a comment: "tests/testimg/ contains 50+ real photos provided by the user for real-world photo testing"

  **Must NOT do**:
  - Don't generate new fixture images
  - Don't add new test images (use existing)
  - Don't change existing `conftest.py` fixtures (add new ones)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Add a new fixture + update existing tests to reference testimg
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6
  - **Blocks**: T24 (end-to-end needs real photo)
  - **Blocked By**: None (testimg/ already exists)

  **References**:
  - **Pattern References**:
    - `tests/conftest.py:34-65` — existing fixture style. Match.
  - **Test References**:
    - `tests/testimg/` — 50+ real photos (Designer (1).jpeg through Designer (50).jpeg, 2024-Q4.jpg, 3840x2160-dark-freebsd.png, banner-*.png)

  **Acceptance Criteria**:
  - [ ] `real_photo_path` fixture added to conftest.py
  - [ ] `tests/test_segmentation.py` uses real_photo_path
  - [ ] `uv run pytest tests/test_segmentation.py -q` → all pass
  - [ ] No new fixture images generated

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: real_photo_path fixture returns a valid image
    Tool: Bash
    Preconditions: testimg/ exists
    Steps:
      1. `uv run python -c "from tests.conftest import real_photo_path; print(real_photo_path())"`
      2. Assert path exists
      3. Assert file size > 50KB
    Expected Result: Returns valid path to Designer (1).jpeg
    Failure Indicators: File not found, empty
    Evidence: .sisyphus/evidence/task-23-real-photo.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-23-real-photo.txt`

  **Commit**: YES (Wave 6)
  - Message: `test: add real_photo_path fixture for SEGMENTED mode testing`
  - Files: `tests/conftest.py`, `tests/test_segmentation.py`
  - Pre-commit: `uv run pytest tests/test_segmentation.py -q`

- [ ] 24. End-to-end verification on user's `rtlogo-1.png` + testimg samples

  **What to do**:
  - Run end-to-end conversion of `/home/mlapointe/Documents/rtlogo-1.png` for each of the 10 modes:
    ```bash
    for mode in auto labels visual annotated trace poster detailed edge watercolor; do
      uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-photo-${mode}.svg --mode $mode
      test -f /tmp/qa-photo-${mode}.svg && echo "$mode: OK ($(wc -c < /tmp/qa-photo-${mode}.svg) bytes)"
    done
    # SEGMENTED needs --seg-model
    uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-photo-segmented.svg --mode segmented --seg-model yolo11s-seg.pt
    ```
  - For each output, verify:
    - File exists
    - Is valid XML (parse with `lxml.etree.parse()`)
    - Contains expected elements:
      - For VISUAL/TRACE/POSTER/DETAILED/EDGE/WATERCOLOR: `<g id="vtracer-output">`
      - For LABELS: `<rect>` (bounding box) + `<text>` (label)
      - For ANNOTATED: `<g id="vtracer-output">` + `<g id="det_...">`
      - For SEGMENTED: `<g id="background">` + at least 1 `<g id="obj_...">` (or just `<g id="vtracer-output">` if no objects)
  - Also run on `tests/testimg/Designer (1).jpeg` (real photo with detectable objects) for SEGMENTED mode
  - Capture sidecar JSON for each and verify the `mode_used` field

  **Must NOT do**:
  - Don't run on testimg/Designer (1).jpeg for ALL modes (one mode per photo is enough for sanity)
  - Don't fail if rtlogo has no detectable objects for SEGMENTED (it's a logo; SEGMENTED should fall back to VisualRenderer)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-mode end-to-end verification with detailed assertions
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all previous tasks)
  - **Parallel Group**: Wave 6 (final task before docs)
  - **Blocks**: T25 (docs reference end-to-end outputs)
  - **Blocked By**: T1-T23

  **References**:
  - **Test References**:
    - User's file: `/home/mlapointe/Documents/rtlogo-1.png` (existing real test file)
    - Real photo: `tests/testimg/Designer (1).jpeg`
  - **Acceptance Criteria from the plan's Definition of Done**:
    - `uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-photo.svg --mode detailed` → succeeds
    - `uv run img2svg convert ... --mode segmented --seg-model yolo11s-seg.pt` → multi-layer SVG
    - `uv run img2svg --mode auto ...` → does NOT auto-pick LABELS or ANNOTATED

  **Acceptance Criteria**:
  - [ ] All 10 modes run successfully on rtlogo-1.png
  - [ ] All 10 outputs are valid XML
  - [ ] Each output has expected SVG elements per mode
  - [ ] SEGMENTED mode on testimg/Designer (1).jpeg produces multi-layer output
  - [ ] Auto mode does NOT pick LABELS or ANNOTATED
  - [ ] All outputs committed as evidence to `.sisyphus/evidence/task-24-*.svg`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All 10 modes produce valid SVG output on user's rtlogo
    Tool: Bash (uv run img2svg)
    Preconditions: All implementation tasks complete
    Steps:
      1. For each of 10 modes, run conversion
      2. Verify file exists, >100 bytes, valid XML
      3. Parse with lxml, verify root tag is `{http://www.w3.org/2000/svg}svg`
      4. Assert mode-specific elements present
    Expected Result: All 10 modes succeed
    Failure Indicators: Any mode fails, invalid XML
    Evidence: .sisyphus/evidence/task-24-e2e-{mode}.svg

  Scenario: SEGMENTED mode on real photo produces multi-layer SVG
    Tool: Bash (uv run img2svg)
    Preconditions: yolo11s-seg.pt downloaded
    Steps:
      1. `uv run img2svg convert tests/testimg/Designer\ \(1\).jpeg --output /tmp/qa-seg-photo.svg --mode segmented --seg-model yolo11s-seg.pt`
      2. Parse output SVG
      3. Assert `<g id="background">` exists
      4. Assert at least 1 `<g id="obj_...">` exists (assuming detection succeeds)
    Expected Result: Multi-layer output with background + at least 1 object
    Failure Indicators: No obj_ groups, no background group
    Evidence: .sisyphus/evidence/task-24-seg-photo.svg
  ```

  **Evidence to Capture**:
  - [ ] `task-24-e2e-{mode}.svg` for all 10 modes
  - [ ] `task-24-seg-photo.svg`
  - [ ] All corresponding `.json` sidecars

  **Commit**: YES (Wave 6)
  - Message: `chore: end-to-end verify on rtlogo-1.png + testimg/Designer (1).jpeg`
  - Files: `.sisyphus/evidence/task-24-*.svg`, `.sisyphus/evidence/task-24-*.json`
  - Pre-commit: `uv run pytest -m "not slow" -q && uv run img2svg --version`

- [ ] 25. Documentation updates (NEW `docs/photo-modes.md` + update README, man page, mkdocs)

  **What to do**:
  - Create `docs/photo-modes.md` (~400 lines):
    - Overview: "Pushing img2svg for real-world photos"
    - Section per new mode (POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED):
      - What it does
      - When to use it
      - Example command
      - Example output (with screenshot if possible)
      - Performance characteristics
    - Pre-processing section (--preprocess flag, --denoise, --sharpen, --max-colors, --quality)
    - YOLO segmentation section (--seg-model, --no-seg, yolo11s-seg vs yolo11m-seg)
    - Limitations and trade-offs
    - Examples (curated set of photos + their SVG output)
  - Update `README.md`:
    - "Output Modes" table: add 5 new modes (rows for POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
    - "Features" bullet: change "Five output modes" → "Ten output modes"
    - "Quickstart" examples: add one for `detailed` and one for `segmented`
  - Update `man/img2svg.1`:
    - Add new flags to `.SH OPTIONS` section
    - Update mode list
    - Update `.SH EXAMPLES` with new mode examples
  - Update `mkdocs.yml`:
    - Add `photo-modes.md` to `nav` under "Guides"
  - Update `docs/usage.md`:
    - Update mode table to include 10 modes
    - Add section on new flags
  - Update `docs/api.md`:
    - Update `ConversionOptions` table with 7 new fields
    - Document `RegionInfo` and `SegmentationResult`
  - Update `docs/architecture.md`:
    - Update pipeline mermaid diagram to show pre-processing + segmentation
  - Update `docs/installation.md`:
    - Mention yolo11s-seg availability
    - Add note about AMD 512MB iGPU caveat (auto-fallback to yolo11n-seg)
  - Update `docs/index.md`:
    - Update "Highlights" to mention new modes
    - Update docs map to link to `photo-modes.md`
  - Update `docs/changelog.md`:
    - Add Unreleased section listing 5 new modes + 8 new flags
  - Update `docs/modes.md`:
    - Add 5 new mode sections (or merge into photo-modes.md)
    - Or keep separate and add a "See also: Photo Modes" link

  **Must NOT do**:
  - Don't delete any existing documentation
  - Don't break existing doc tests (`test_docs.py` checks for required doc files)
  - Don't add fluff — only the new modes and flags

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation writing
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6
  - **Blocks**: F1-F4
  - **Blocked By**: T24 (docs reference end-to-end outputs)

  **References**:
  - **Pattern References**:
    - `docs/modes.md` — existing mode documentation. Mirror structure for new modes.
    - `README.md` — existing table style. Mirror.
    - `man/img2svg.1` — existing man page structure. Mirror.
    - `mkdocs.yml` — existing nav structure. Add `photo-modes.md`.
  - **WHY Each Reference Matters**:
    - These are the existing docs that need updating.

  **Acceptance Criteria**:
  - [ ] `docs/photo-modes.md` created (~400 lines, 5 mode sections, preprocessing section, segmentation section, examples)
  - [ ] `README.md` updated (Output Modes table has 10 rows, Features bullet updated, Quickstart examples added)
  - [ ] `man/img2svg.1` updated (new flags added, mode list updated, examples added)
  - [ ] `mkdocs.yml` updated (`photo-modes.md` in nav)
  - [ ] `docs/usage.md`, `docs/api.md`, `docs/architecture.md`, `docs/installation.md`, `docs/index.md`, `docs/changelog.md`, `docs/modes.md` updated
  - [ ] `uv run pytest tests/test_docs.py tests/test_manpage.py -q` → all pass
  - [ ] `uv run mkdocs build --strict` → no warnings

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All docs updated and tests pass
    Tool: Bash
    Preconditions: All code complete
    Steps:
      1. `uv run pytest tests/test_docs.py tests/test_manpage.py -q` → all pass
      2. `uv run mkdocs build --strict` → no warnings
      3. `ls docs/photo-modes.md` → exists
      4. `grep -c "^##" docs/photo-modes.md` ≥ 5 (5+ sections)
    Expected Result: All docs tests pass, mkdocs builds, photo-modes.md exists
    Failure Indicators: Doc test failure, mkdocs warning, missing file
    Evidence: .sisyphus/evidence/task-25-docs.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-25-docs.txt`

  **Commit**: YES (Wave 6)
  - Message: `docs: add photo-modes.md + update README/man/mkdocs/usage/api/architecture`
  - Files: `docs/photo-modes.md`, `README.md`, `man/img2svg.1`, `mkdocs.yml`, `docs/usage.md`, `docs/api.md`, `docs/architecture.md`, `docs/installation.md`, `docs/index.md`, `docs/changelog.md`, `docs/modes.md`
  - Pre-commit: `uv run pytest tests/test_docs.py tests/test_manpage.py -q && uv run mkdocs build --strict`

- [ ] 26. Add lessons to Honcho workspace (capture key decisions from this plan)

  **What to do**:
  - Use the `honcho_add_conclusions` tool to add lessons learned to the Honcho workspace
  - Capture these as conclusions (peer=planner, target=user):
    - "User constraint: ANNOTATED and LABELS modes are EXPLICIT only — never auto-selected. The IMAGE_TYPE_TO_MODE mapping was updated so PHOTO→DETAILED (aggressive default), LOGO/DIAGRAM/SCREENSHOT/LINE_ART/UNKNOWN→VISUAL."
    - "User preference: AGGRESSIVE defaults. All photo modes (auto/trace/visual/annotated) get default preprocessing unless --no-preprocess. SEGMENTED auto-selects yolo11x-seg for PHOTO images."
    - "User preference: Multi-layer editable SVG for SEGMENTED mode. Each detected object becomes its own <g id='obj_class_idx'> group with semantic class name."
    - "Decision: Skip CLAHE in WATERCOLOR mode (deferred to v2). User wanted 'aggressive' but research warns about noise amplification."
    - "Decision: VRAM fallback for YOLO seg — auto-fallback to yolo11n-seg on OOM, then to bbox detection. Important for AMD 890M (512MB iGPU)."
    - "Decision: palette_size vs --max-colors — removed unused palette_size, use new max_colors field for color capping."
    - "Architecture: 5 new modes (POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED). 3 new vtracer presets (photo_hifi, bw_edge, watercolor). 5 new renderers. 1 new YOLOSegmentor class. 1 new preprocessing module."
    - "Tests use existing tests/testimg/ directory (50+ real photos provided by user). No new fixtures generated."
    - "Future: defer to v2 — depth estimation, style transfer, tile-based parallel tracing, quality metrics, region-based palette extraction."
  - Capture these as a new peer card fact (peer=planner): "img2svg plan for photo quality push is 25+ tasks in 6 waves. Critical path: T1 (preprocessing) → T3 (enums) → T10 (registry wiring) → T18 (pipeline integration) → T24 (end-to-end) → F1-F4 (verification)."

  **Must NOT do**:
  - Don't add lessons as session messages (use `add_conclusions` or `set_peer_card`)
  - Don't include sensitive info (API keys, file paths beyond what's needed)
  - Don't add redundant conclusions (be concise)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct API call, no logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (last task)
  - **Blocks**: Plan closure
  - **Blocked By**: All other tasks (so lessons are accurate)

  **References**:
  - **Tool References**:
    - `honcho_add_conclusions` (peers=planner + user) — to add conclusions
    - `honcho_set_peer_card` (peer=planner) — to update planner's peer card with current task
  - **WHY Each Reference Matters**:
    - Honcho is the user's memory across sessions. Lessons here will inform future planning.

  **Acceptance Criteria**:
  - [ ] 9+ conclusions added to Honcho (one per lesson learned)
  - [ ] Peer card updated with current task
  - [ ] `honcho_list_conclusions(peer_id="planner")` returns the new conclusions
  - [ ] `honcho_get_peer_card(peer_id="planner")` includes the current task

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Lessons added to Honcho
    Tool: Bash
    Preconditions: Honcho workspace accessible
    Steps:
      1. `honcho_list_conclusions(peer_id="planner")` returns the 9 lessons
      2. `honcho_get_peer_card(peer_id="planner")` includes "img2svg plan for photo quality push"
    Expected Result: All 9 lessons + peer card update
    Failure Indicators: Missing lessons, peer card unchanged
    Evidence: .sisyphus/evidence/task-26-honcho.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-26-honcho.txt`

  **Commit**: YES (Wave 6)
  - Message: `chore(honcho): add lessons learned from photo-quality-push plan`
  - Files: N/A (Honcho is external)
  - Pre-commit: N/A

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `ruff check src tests` + `ruff format --check src tests` + `mypy src` + `pytest`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp). Verify no new license issues.
  Output: `Ruff [PASS/FAIL] | Mypy [N new errors] | Tests [N pass/N fail] | Files [N clean/N issues] | License [OK/FLAG] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI)
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (new modes + pre-processing + segmentation working together). Test edge cases: empty state (no detections), invalid input, rapid actions (batch of 50 photos). Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: 5 commits (one per task). Messages: `feat(preprocessing): add composable OpenCV filters`, `feat(vectorizer): add photo_hifi, bw_edge, watercolor presets`, `feat(enums): add POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED modes`, `feat(models): add preprocessing + regions fields to Sidecar`, `feat(cli): add --preprocess/--denoise/--sharpen/--max-colors/--quality/--no-preprocess flags`
- **Wave 2**: 5 commits (one per task). Messages: `feat(renderers): add PosterRenderer`, `feat(renderers): add DetailedRenderer`, `feat(renderers): add EdgeRenderer`, `feat(renderers): add WatercolorRenderer`, `feat(pipeline): wire new renderers to RENDERER_REGISTRY + presets`
- **Wave 3**: 5 commits. Messages: `feat(detector): add YOLOSegmentor with retina_masks=True`, `feat(detector): add mask extraction helpers (xy, area, bbox)`, `feat(detector): add per-region vtracer tracer`, `feat(cli): add --seg-model/--no-seg flags`, `feat(models): add RegionInfo + populate regions field`
- **Wave 4**: 2 commits. Messages: `feat(renderers): add SegmentedRenderer (multi-layer SVG)`, `feat(pipeline): add preprocessing step (post-load, pre-analyze)`
- **Wave 5**: 3 commits. Messages: `feat(pipeline): add segmentation step (per-region vtracer for SEGMENTED)`, `feat(pipeline): add timing entries (preprocess, segment, vectorize)`, `feat(pipeline): add MAX_SVG_SIZE guard (50MB cap)`
- **Wave 6**: 5 commits. Messages: `test: add test_preprocessing, test_segmentation, test_renderers`, `test: update existing tests for new modes + auto-mode behavior`, `test: add photo_with_objects.png fixture`, `chore: end-to-end verify on user's rtlogo-1.png`, `docs: add photo-modes.md + update README + man page + mkdocs`
- **FINAL**: 1 commit. Message: `chore(plan): close photo-quality-push plan, F1-F4 approve`

All commits follow Conventional Commits format. All commits pushed to `origin/main` immediately.

---

## Success Criteria

### Verification Commands
```bash
# Unit tests pass
uv run pytest -m "not slow" -q
# Expected: 480+ passed, 1 pre-existing i18n failure acceptable

# Coverage maintained
uv run pytest --cov=img2svg --cov-report=term --cov-fail-under=90
# Expected: 90%+ coverage

# Lint clean
uv run ruff check src tests
# Expected: 0 issues

# Type check clean (no new errors)
uv run mypy src
# Expected: 0 new errors vs baseline

# CLI works
uv run img2svg --help
# Expected: shows all 9 modes (auto, labels, visual, annotated, trace, poster, detailed, edge, watercolor, segmented)

# Auto mode doesn't pick LABELS/ANNOTATED
uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-auto.svg
# Expected: sidecar.mode_used is in {visual, detailed}, NOT labels or annotated

# New modes work
for mode in poster detailed edge watercolor; do
  uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-${mode}.svg --mode $mode
  test -f /tmp/qa-${mode}.svg && echo "$mode: OK"
done
# Expected: all 4 succeed, output SVGs differ

# Segmented mode works
uv run img2svg convert /home/mlapointe/Documents/rtlogo-1.png --output /tmp/qa-seg.svg --mode segmented --seg-model yolo11s-seg.pt
grep -c '<g id="obj_' /tmp/qa-seg.svg
# Expected: at least 1 (user's RT logo has scissors)
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass (480+ fast, 1 pre-existing failure acceptable)
- [ ] 90%+ coverage maintained
- [ ] 0 ruff issues
- [ ] 0 new mypy errors
- [ ] All commits pushed to origin/main
- [ ] Working tree clean
- [ ] F1-F4 all APPROVE
- [ ] User gives explicit "okay" to mark plan complete
