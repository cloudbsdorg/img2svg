# F4 Scope Fidelity Check — Final Verdict

**Plan:** `.sisyphus/plans/photo-quality-push.md`
**Baseline:** `79c0f38` ("Adding plans for photorealistic, as far as we can right now")
**HEAD:** `a6fa9fd` on `main` (28 commits)
**Verifier:** F4 deep reviewer
**Date:** 2026-06-11

---

## VERDICT

```
Tasks [25/26 compliant] | Contamination [CLEAN/0 issues] | Unaccounted [CLEAN/0 files] | VERDICT: REJECT
```

**REJECT** — Task T26 (Honcho lessons) is not implemented. 25 of 26 tasks (95.8%) are 1:1 spec-compliant, no cross-task contamination, no unaccounted changes, and no forbidden AI-slop patterns. The single gap is T26, which requires external Honcho workspace state and was never executed.

---

## Coverage Matrix: Task → Files Changed

| Task | Commit(s) | Files Modified | Compliance |
|------|-----------|----------------|------------|
| T1 preprocessing.py | `9a6889a` | `src/img2svg/preprocessing.py` (+ notepad learnings) | ✅ PASS |
| T2 vectorizer PRESETS | `7379efe` | `src/img2svg/vectorizer.py` | ✅ PASS |
| T3 enums Mode | `0431aa7` | `src/img2svg/enums.py`, `src/img2svg/presets.py` | ✅ PASS (documented expansion) |
| T4 Sidecar + RegionInfo | `2805cd2` | `src/img2svg/models.py`, `src/img2svg/__init__.py`, `tests/test_models.py` | ✅ PASS |
| T5 CLI flags part 1 | `6e46294` | `src/img2svg/cli.py` | ✅ PASS |
| T6 PosterRenderer | `7da8464` | `src/img2svg/renderers/poster.py` | ✅ PASS |
| T7 DetailedRenderer | `7da8464` | `src/img2svg/renderers/detailed.py` | ✅ PASS |
| T8 EdgeRenderer | `7da8464` | `src/img2svg/renderers/edge.py` | ✅ PASS |
| T9 WatercolorRenderer | `7da8464` | `src/img2svg/renderers/watercolor.py` | ✅ PASS |
| T10 Wiring | `fcd351a` | `src/img2svg/pipeline.py`, `src/img2svg/presets.py`, `tests/test_presets.py`, `tests/test_pipeline.py` | ✅ PASS (documented test_pipeline.py expansion) |
| T11 YOLOSegmentor | `cb8f7f1` | `src/img2svg/detector.py`, `tests/test_segmentation.py` | ✅ PASS |
| T12 mask helpers | `cb8f7f1` | `src/img2svg/detector.py` | ✅ PASS (combined with T11/T13) |
| T13 trace_region | `cb8f7f1` | `src/img2svg/detector.py` | ✅ PASS (combined with T11/T12) |
| T14 CLI flags part 2 | `d4fd3c3` | `src/img2svg/cli.py`, `src/img2svg/models.py`, `tests/test_models.py` | ✅ PASS |
| T15 Sidecar regions | `756273c` | `src/img2svg/pipeline.py` | ✅ PASS |
| T16 SegmentedRenderer | `6ee3656` | `src/img2svg/pipeline.py`, `src/img2svg/renderers/segmented.py` | ✅ PASS |
| T17 preprocessing step | `09b5517` | `src/img2svg/pipeline.py` | ✅ PASS (combined with T18/T19/T20) |
| T18 SEGMENTED fallback + inject | `09b5517` | `src/img2svg/pipeline.py` | ✅ PASS (combined) |
| T19 timing instrumentation | `09b5517` | `src/img2svg/pipeline.py`, `tests/test_pipeline.py` | ✅ PASS (combined) |
| T20 MAX_SVG_SIZE guard | `09b5517` | `src/img2svg/pipeline.py`, `src/img2svg/cli.py`, `src/img2svg/models.py`, `src/img2svg/errors.py`, `tests/test_pipeline.py` | ✅ PASS (combined) |
| T21 new test files | `b5a8124` (Wave 6 commit) | `tests/test_preprocessing.py`, `tests/test_renderers/test_new_renderers.py`, `tests/test_segmentation.py` | ✅ PASS (63 tests across 3 files) |
| T22 update existing tests | `b5a8124` + `a6fa9fd` | `tests/test_cli.py`, `tests/test_metadata.py`, `tests/test_manpage.py`, `tests/test_examples.py`, `tests/test_vectorizer.py`, `tests/test_docs.py` | ✅ PASS |
| T23 real_photo_path fixture | `b5a8124` | `tests/conftest.py` | ✅ PASS |
| T24 E2E verification | `b5a8124` (+ `b5a8124` bug fix) | `.sisyphus/evidence/task-24-*` (12 modes, 11 e2e svgs, seg-photo-FIXED) | ✅ PASS (bug fixed in same commit) |
| T25 Documentation | `b5a8124` | `docs/photo-modes.md` (470 lines), `README.md`, `man/img2svg.1`, `mkdocs.yml`, `docs/*`, `examples/sample_outputs/{logo_detailed,logo_edge,logo_poster,logo_segmented,logo_watercolor}.svg` | ✅ PASS |
| T26 Honcho lessons | **NONE** | **NONE** (Honcho workspace is empty for `planner` peer) | ❌ **FAIL** |

---

## Evidence Per Task

### T1 — preprocessing.py (PASS)
- File: `src/img2svg/preprocessing.py` (300 lines, BSD-3-Clause header)
- Verified 7 filter functions: `denoise_bilateral`, `denoise_nlmeans`, `denoise_median`, `sharpen_unsharp`, `posterize`, `detect_edges_canny`, `apply_clahe_yuv`
- `PreprocessingPipeline` class with `apply()` + `steps_applied()`
- `PREPROCESSING_PRESETS` dict with `light`, `medium`, `heavy`, `edge`
- Alpha channel preserved, dtype preserved, `ruff check` 0 issues
- Commit `9a6889a`

### T2 — vectorizer PRESETS (PASS)
- File: `src/img2svg/vectorizer.py`
- `Preset` Literal updated: `Literal["default", "bw", "logo", "poster", "photo", "photo_hifi", "bw_edge", "watercolor"]`
- 3 new entries in PRESETS: `photo_hifi`, `bw_edge`, `watercolor` (each with all 11 vtracer kwargs)
- Commit `7379efe` (single file, 42 insertions)

### T3 — enums Mode (PASS, documented expansion)
- File: `src/img2svg/enums.py`
- 5 new Mode enum values: `POSTER`, `DETAILED`, `EDGE`, `WATERCOLOR`, `SEGMENTED` (plus `TRACE` rearrangement)
- `src/img2svg/presets.py` also updated to add 5 MODE_TO_PRESET entries (documented in notepad as necessary for test compatibility)
- Commit `0431aa7`

### T4 — Sidecar + RegionInfo (PASS)
- File: `src/img2svg/models.py`
- New `RegionInfo` Pydantic model with 7 fields: `class_id`, `class_name`, `confidence`, `bbox`, `area_pixels`, `polygon`, `mask_path`
- 3 new Sidecar fields: `preprocessing: list[str]`, `regions: list[RegionInfo]`, `model_variant: str`
- Exported from `src/img2svg/__init__.py` (import + `__all__` list)
- Commit `2805cd2`

### T5 — CLI flags part 1 (PASS)
- File: `src/img2svg/cli.py`
- 6 new flags: `--preprocess`, `--denoise`, `--sharpen`, `--max-colors`, `--quality`, `--no-preprocess`
- Help text updated to list all 10 modes
- Commit `6e46294`

### T6-T9 — New Renderers (PASS, combined)
- `src/img2svg/renderers/poster.py` (26 lines, `preset_name = "poster"`)
- `src/img2svg/renderers/detailed.py` (33 lines, `preset_name = "photo_hifi"`)
- `src/img2svg/renderers/edge.py` (35 lines, `preset_name = "bw_edge"`)
- `src/img2svg/renderers/watercolor.py` (30 lines, `preset_name = "watercolor"`)
- All 4 use `_render_with_vtracer(self, self.preset_name)` pattern
- All 4 ruff-clean
- Combined in single commit `7da8464` (T6/T7/T8/T9 packaged together — acceptable per "5-9 tasks per wave" guidance)

### T10 — Wiring (PASS, documented expansion)
- `RENDERER_REGISTRY` has 9 entries (4 existing + 5 new: POSTER, DETAILED, EDGE, WATERCOLOR, SEGMENTED)
- `MODE_TO_PRESET` has 9 entries (4 existing + 5 new)
- `IMAGE_TYPE_TO_MODE` rewritten: PHOTO → DETAILED, all others → VISUAL (NO LABELS, NO ANNOTATED — user constraint respected)
- `tests/test_presets.py` updated; `tests/test_pipeline.py` also touched (documented in notepad as necessary for test regression)
- Commit `fcd351a`

### T11-T13 — YOLOSegmentor + mask helpers + trace_region (PASS, combined)
- File: `src/img2svg/detector.py`
- `YOLOSegmentor` class (line 272) with `retina_masks=True`, `imgsz=1024`, lazy backend resolution
- `SegmentationResult` dataclass (line 232) with `boxes`, `masks`, `polygons`, `orig_shape`
- `get_segmentor()` factory with separate `_SEGMENTOR_CACHE` (NOT shared with `_MODEL_CACHE`)
- Module-level helpers: `extract_polygons`, `compute_mask_area`, `get_tight_bbox` (lines 458-488)
- `trace_region` function (line 495) with empty-mask guard, RGBA stacking, vtracer mocking-friendly
- All in single commit `cb8f7f1` (T11/T12/T13 packaged together — acceptable per wave batching)
- 22 tests in `tests/test_segmentation.py`

### T14 — CLI flags part 2 (PASS)
- File: `src/img2svg/cli.py` + `src/img2svg/models.py`
- 2 new flags: `--seg-model` (with `_seg_model_callback` validator), `--no-seg`
- 2 new ConversionOptions fields: `seg_model: str = "yolo11s-seg"`, `no_seg: bool = False`
- Commit `d4fd3c3`

### T15 — Sidecar regions (PASS)
- File: `src/img2svg/pipeline.py`
- New "6b" step in `Pipeline.run()`: YOLO segmentation for SEGMENTED mode
- `sidecar.regions` populated from `region_infos: list[RegionInfo]`
- `sidecar.model_variant = options.seg_model`
- `self._segmentation_result` stashed for renderer injection
- `timings["segment"]` always present
- Commit `756273c`

### T16 — SegmentedRenderer (PASS, with bug fix in b5a8124)
- File: `src/img2svg/renderers/segmented.py` (229 lines)
- `SegmentedRenderer(Renderer)` with `_segmentation_result` instance attribute
- `set_segmentation(result)` setter for pipeline injection
- `render()` method: 4-step flow (background extraction → background trace → per-region trace → multi-group SVG)
- Multi-layer SVG structure: `<g id="background">` + `<g id="obj_...">` per region
- Empty segmentation falls back to VisualRenderer
- Pipeline wires it into RENDERER_REGISTRY
- BUG FIX (b5a8124): `pipeline.py:362` now calls `renderer.set_segmentation(segmentation_result)` for SegmentedRenderer
- Verified: E2E test on `tests/testimg/Designer (1).jpeg` produces 1 `<g id="background">` + 1 `<g id="obj_person_0">` (per task-24-seg-photo-FIXED.svg evidence)
- Commits `6ee3656` + `b5a8124`

### T17-T20 — Pipeline integration + safety (PASS, combined)
- File: `src/img2svg/pipeline.py` (+ `cli.py`, `models.py`, `errors.py`, `test_pipeline.py`)
- T17: `_resolve_preprocessing_steps()` + `PreprocessingPipeline` in step 1b (post-load, pre-analyze)
- T18: Step 8 fallback gate: SEGMENTED + no_seg/empty → VisualRenderer; also injects `segmentation_result` into renderer via `isinstance(renderer, SegmentedRenderer)` check (the b5a8124 fix)
- T19: `timings["preprocess"]`, `timings["segment"]`, `timings["vectorize"]` all added; `test_pipeline_records_timings` updated to 11 keys
- T20: `MAX_SVG_SIZE_MB = 50` constant; `max_svg_size_mb: int = Field(default=50, ge=1, le=1024)` on ConversionOptions; `--max-svg-size` CLI flag; `SVGSizeLimitError` in `errors.py`; size check after `svg.write()` deletes the file before raising
- All 4 tasks combined in single commit `09b5517` (acceptable per wave batching)

### T21 — New test files (PASS)
- `tests/test_preprocessing.py` (24 test functions, 12060 bytes)
- `tests/test_segmentation.py` (22 test functions, 10560 bytes — including 6 trace_region tests)
- `tests/test_renderers/test_new_renderers.py` (17 test functions, 9431 bytes)
- All 3 files ruff-clean
- Note: spec said `tests/test_renderers.py` (file), but pytest cannot have both file and package with same name; agent created `tests/test_renderers/test_new_renderers.py` (file in existing package) — documented deviation, equivalent coverage

### T22 — Update existing tests (PASS)
- `tests/test_cli.py` (547 lines, up from 251 baseline — added 14+ new tests)
- `tests/test_metadata.py` (160 lines, added 4 new tests for preprocessing/regions/model_variant round-trips)
- `tests/test_manpage.py` (249 lines, added 9 new EXPECTED_FLAGS)
- `tests/test_examples.py` (97 lines, MIN_SVGS raised from 8 to 13)
- `tests/test_vectorizer.py` (updated `test_presets_dict_has_five_entries` → `test_presets_dict_has_eight_entries`)
- `tests/test_docs.py` (added photo-modes.md to REQUIRED_DOCS)

### T23 — real_photo_path fixture (PASS)
- `tests/conftest.py:69` — `real_photo_path` fixture (10 lines, docstring included)
- Returns `tests/testimg/Designer (1).jpeg` (real photo, 253KB — well above 50KB threshold)

### T24 — E2E verification (PASS)
- 10 mode E2E SVGs + JSON sidecars in `.sisyphus/evidence/task-24-e2e-*.{svg,json}` (auto/labels/visual/annotated/trace/poster/detailed/edge/watercolor/segmented)
- SEGMENTED on real photo: `task-24-seg-photo-FIXED.svg` (18583 lines) — multi-layer output verified
- Bug report: `task-24-bug-report.md` (86 lines)
- Verification report: `task-24-verification-report.txt`
- All 10 modes produce valid XML
- Auto mode does NOT pick LABELS/ANNOTATED (verified: mode_used=detailed)

### T25 — Documentation (PASS)
- `docs/photo-modes.md` (470 lines, 22KB, 13 H2 sections)
- `README.md` (Output Modes table 10 rows, Features bullet updated, Quickstart examples added)
- `man/img2svg.1` (301 lines added — new flags, mode list, examples)
- `mkdocs.yml` (root + `docs/mkdocs.yml` both updated — dual-file gotcha)
- `docs/usage.md`, `docs/api.md`, `docs/architecture.md`, `docs/installation.md`, `docs/index.md`, `docs/changelog.md`, `docs/modes.md` — all updated
- `examples/sample_outputs/`: 5 new sample SVGs (logo_detailed, logo_edge, logo_poster, logo_segmented, logo_watercolor)
- T25 follow-up (a6fa9fd): wired 4 broken CLI flags (`--denoise`, `--sharpen`, `--max-colors`, `--quality`) + ruff format + mypy cleanup

### T26 — Honcho lessons (**FAIL**)
- **Acceptance criteria from plan:**
  - 9+ conclusions added to Honcho (one per lesson learned)
  - Peer card updated with current task
- **Actual state (queried via `honcho_list_conclusions(peer_id="planner")` and `honcho_get_peer_card(peer_id="planner")`):**
  - `conclusions: []`, `total: 0`
  - `No peer card found.`
- **No evidence file:** `.sisyphus/evidence/task-26-honcho.txt` does not exist
- **Root cause:** T26 was an external-only task (no repo files modified). The notepad mentions lessons learned extensively, but the Honcho API was never called to persist them.
- **Impact:** T26 acceptance criteria unmet. The plan closure cannot proceed without these external state changes.

---

## Cross-Task Contamination Check (CLEAN)

| Check | Result |
|-------|--------|
| T1 (preprocessing.py) touched only its spec'd files | ✅ Clean |
| T2 (vectorizer.py) touched only vectorizer.py | ✅ Clean |
| T3 expanded to presets.py (documented in notepad) | ✅ Documented, not contamination |
| T4 (models.py + __init__.py) — no other files | ✅ Clean |
| T5 (cli.py) — no other files | ✅ Clean |
| T6-T9 (4 renderers) — no other files | ✅ Clean |
| T10 (pipeline + presets + test_presets) — test_pipeline.py also touched (documented in notepad) | ✅ Documented |
| T11-T13 (detector.py + test_segmentation.py) — no other files | ✅ Clean |
| T14 (cli.py + models.py + test_models.py) — no other files | ✅ Clean |
| T15 (pipeline.py) — no other files | ✅ Clean |
| T16 (pipeline.py + renderers/segmented.py) — no other files | ✅ Clean |
| T17-T20 (pipeline + cli + models + errors + test_pipeline) — all spec'd | ✅ Clean |
| T21-T25 (Wave 6) — multi-file commit, all in spec | ✅ Clean |
| T25 follow-up (a6fa9fd) — 4 broken flags + format + mypy across 16 files | ✅ Documented F2/F3 remediation |

**No cross-task contamination detected.**

---

## Unaccounted Files Check (CLEAN)

Total files changed: 165 (between `79c0f38` and `a6fa9fd`)

Categorized:
- `.sisyphus/*` (orchestrator metadata + evidence + notepad): expected
- `docs/*` (T25 docs): expected
- `README.md`, `man/img2svg.1`, `mkdocs.yml` (T25 docs): expected
- `examples/sample_outputs/*` (T22 + T25): expected
- `src/img2svg/preprocessing.py` (T1): expected
- `src/img2svg/vectorizer.py` (T2): expected
- `src/img2svg/enums.py`, `src/img2svg/presets.py` (T3, T10): expected
- `src/img2svg/models.py` (T4, T5, T14, T20, T25 follow-up): expected
- `src/img2svg/cli.py` (T5, T14, T20, T25 follow-up): expected
- `src/img2svg/renderers/{poster,detailed,edge,watercolor,segmented}.py` (T6-T9, T16): expected
- `src/img2svg/pipeline.py` (T10, T15, T16, T17-T20, T24 fix, T25 follow-up): expected
- `src/img2svg/detector.py` (T11-T13, T25 follow-up): expected
- `src/img2svg/errors.py` (T20, T25 follow-up): expected
- `src/img2svg/__init__.py` (T4): expected
- `src/img2svg/renderers/{base,visual}.py` (T25 follow-up): expected
- `tests/test_*` (T21, T22, T23, T25 follow-up): expected

**No unaccounted files detected.** All 165 files in the diff map to a documented task.

---

## Forbidden Patterns Check (CLEAN)

Searched entire diff (`79c0f38..HEAD`) for:
- `as any` in src/ or tests/ → **0 matches**
- `# type: ignore` without bracket justification → **0 matches** (all uses are `#[code]` form)
- `console.log` / `console.debug` / `console.warn` / `console.error` in src/ → **0 matches**
- Empty `except:` blocks → **0 matches**
- `TODO` / `FIXME` / `HACK` / `XXX` in src/ or tests/ → **0 matches** (only matches in notepad learnings text "no TODO/FIXME/HACK" — those are guidelines, not code)
- Generic names like `data`, `result`, `item`, `temp` as primary variables → **0 matches** (only used appropriately, e.g. `result` for a return value)

**No forbidden patterns detected.**

---

## Tests Status

- `uv run pytest -m "not slow" -q` → **615 passed, 1 failed**
- 1 failure: `tests/test_i18n.py::test_ngettext_returns_singular_in_c_locale` — **pre-existing** per multi-vendor-gpu plan, documented acceptable
- 0 new failures
- 0 regressions

## Ruff Status

- Pre-existing baseline (`79c0f38`): 32 errors
- Current HEAD (`a6fa9fd`): 32 errors
- **Delta: 0** — plan introduced no new ruff issues
- New files (preprocessing.py, 5 new renderers, 3 new test files) — all ruff-clean

---

## Summary

**Strengths:**
- 25/26 tasks (95.8%) fully compliant with spec
- All "Must Have" deliverables present
- All "Must NOT Have" forbidden patterns absent
- No cross-task contamination
- No unaccounted files
- All tests pass (no regressions)
- All new files ruff-clean
- Two bug fix commits (b5a8124 SegmentedRenderer injection + a6fa9fd 4 broken flags + format + mypy) demonstrate honest post-review remediation

**Gaps:**
- T26 (Honcho lessons) not implemented — 0/9 conclusions added, peer card not set
- T26 is an external-state task (no repo files); cannot be verified from git diff
- Honcho workspace query confirms 0 conclusions for `planner` peer

**Recommended Action:**
Complete T26 by calling:
- `honcho_add_conclusions(peer_id="planner", target_peer_id="user", conclusions=[...9 lessons...])`
- `honcho_set_peer_card(peer_id="planner", peer_card=["img2svg plan for photo quality push: 25 tasks in 6 waves..."])`
- Save evidence to `.sisyphus/evidence/task-26-honcho.txt`

After T26 is complete, re-run F4 to flip the verdict to APPROVE.
