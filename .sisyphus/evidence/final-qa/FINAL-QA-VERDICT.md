# F3 Final QA Report — photo-quality-push

**Date**: 2026-06-11
**Reviewer**: F3 (Real Manual QA)
**Working tree**: clean, on `main`, at commit `b5a8124`
**Pytest**: 609 passed, 1 pre-existing i18n failure (acceptable per spec), 8 deselected (slow)
**Ruff on T1-T26 files**: 0 new issues (5 pre-existing in cli.py — Typer pattern)

---

## Executive Summary

**Scenarios**: [21/24 pass] | **Integration**: [1/1] | **Edge Cases**: [9/10 tested] | **VERDICT: REJECT**

The plan delivers 21 out of 24 scenarios correctly. The most critical scenario — SEGMENTED multi-layer output on real photos — works correctly thanks to the b5a8124 fix. However, 4 of the 9 explicitly-listed new CLI flags are non-functional: they are accepted by the CLI and stored on `ConversionOptions` but never read by the pipeline. This violates the user-specified acceptance criterion "9 new CLI flags work end-to-end".

---

## Scenario-by-Scenario Results

### T-1: All 10 modes on rtlogo-1.png — **PASS** (10/10)
| Mode | File size | XML valid | Expected structure | Sidecar mode_used |
|------|-----------|-----------|-------------------|-------------------|
| auto | 19740 | OK | vtracer-output | detailed |
| labels | 1324 | OK | bg+det_scissors, 3 rect, 2 text | labels |
| visual | 19442 | OK | vtracer-output, 47 paths | visual |
| annotated | 20261 | OK | vtracer+det_scissors, 2 rect, 2 text, 47 paths | annotated |
| trace | 27126 | OK | vtracer-output, 92 paths | trace |
| poster | 19442 | OK | vtracer-output, 47 paths | poster |
| detailed | 19740 | OK | vtracer-output, 30 paths | detailed |
| edge | 965 | OK | vtracer-output, 1 path (binary) | edge |
| watercolor | 9841 | OK | vtracer-output, 9 paths | watercolor |
| segmented | 16453 | OK | vtracer-output (fallback, no detections) | segmented |

All 10 modes produce valid XML and structurally correct output. Each output is visually distinct (different path counts, different IDs, different sizes).
Evidence: `01-rtlogo-modes.log`, `02-rtlogo-sidecars.log`, `03-rtlogo-structures.log`, `qa-rtlogo-*.svg/.json`

### T-2: SEGMENTED on Designer (1).jpeg — **PASS** (CRITICAL TEST)
- 1 region detected: `person`, confidence 0.645, area 238358px
- SVG contains `<g id="background">` + `<g id="obj_person_0">` ✓
- Sidecar `model_variant: "yolo11s-seg"` ✓
- Preprocessing applied: `bilateral + unsharp` (mode-driven default) ✓
- File size: 9.5MB (within 50MB cap) ✓

**The b5a8124 fix works end-to-end on real photos.** Multi-layer SVG with background + per-region groups confirmed.

Evidence: `04-seg-photo.log`, `qa-seg-photo.svg`, `qa-seg-photo.json`

### T-3: Auto mode on Designer (1).jpeg — **PASS**
- `mode_used: visual` (not labels/annotated) ✓
- Tested on 7 more photos: NEVER picks labels/annotated
  - Designer (2-5, 10): photo → detailed
  - Designer (25, 50): screenshot → visual
- The `auto` mode constraint is satisfied across all image types tested.

Evidence: `05-auto-photo.log`, `qa-auto-photo.svg/.json`

### T-4: 9 new CLI flags end-to-end — **PARTIAL FAIL** (5/9 working)

| Flag | Status | Notes |
|------|--------|-------|
| `--preprocess` | ✅ WORKS | Repeated values land in sidecar as `[bilateral, unsharp]` |
| `--denoise` | ❌ **BROKEN** | Accepted by CLI, never read by pipeline |
| `--sharpen` | ❌ **BROKEN** | Accepted by CLI, never read by pipeline |
| `--max-colors` | ❌ **BROKEN** | Accepted by CLI, never read; max-colors=4 vs =64 produce IDENTICAL output |
| `--quality` | ❌ **BROKEN** | Accepted by CLI, never read; help text lies ("stored in sidecar" but isn't) |
| `--no-preprocess` | ✅ WORKS | Properly overrides `--preprocess` |
| `--seg-model` | ✅ WORKS | `yolo11n-seg` → sidecar `model_variant: yolo11n-seg` |
| `--no-seg` | ✅ WORKS | Falls back to VisualRenderer (vtracer-output, no obj_ groups) |
| `--max-svg-size` | ✅ WORKS | Cap triggered, file NOT created, exit 1 |

**CRITICAL FINDING**: 4 of 9 flags are non-functional. The `ConversionOptions` model has fields for `denoise`, `sharpen`, `max_colors`, `quality`, but no code in `src/img2svg/` ever reads them. Searched the entire src/ tree:
```bash
grep -rn "options\.denoise\|options\.sharpen\|options\.max_colors\|options\.quality" src/img2svg/ --include="*.py"
# Result: 0 matches (outside models.py and cli.py which only DEFINE them)
```

This is a real bug, not a configuration issue. The flags do nothing.

Evidence: `06-flags.log`, `07-flag-investigations.log`, `16-flag-search.log`, `qa-flag-*.svg/.json`

### T-5: MAX_SVG_SIZE cap (--max-svg-size 1) — **PASS**
- Photo: 8.1MB detailed SVG; with `--max-svg-size 1`: WARNING emitted, file deleted, exit code 1
- With `--max-svg-size 2` (still under photo): same behavior — exit 1
- The pipeline raises `SVGSizeLimitError` (verified in code), the CLI catches it via `except Img2SvgError` and exits 1
- File is NEVER created (verified: `ls` shows no file)

Evidence: `08-size-cap-investigation.log`

### T-6: --no-seg fallback — **PASS**
- `img2svg convert ... --mode segmented --no-seg` on real photo
- Output: `vtracer-output` group only, NO `obj_` groups ✓
- Sidecar: `mode_used: segmented`, `regions: []` ✓
- Warning logged: "SEGMENTED mode requested but no detections or --no-seg, falling back to VisualRenderer"

Evidence: `06-flags.log`, `qa-flag-noseg.svg/.json`

### T-7: Batch of 50 photos (--mode auto) — **PASS**
- 50/50 succeeded
- 0 failures
- Elapsed: 350s (~7s/photo average)
- All SVGs valid XML

Evidence: `10-batch-test.log`

### T-8: Invalid input — **PASS**
- Nonexistent file: exit 2, "file not found: /tmp/..." ✓
- Corrupt file: exit 2, "failed to decode image: ... UnidentifiedImageError" ✓

Evidence: `09-invalid-input.log`

### T-9: Edge cases — **PASS** (8/9)
- ✅ Empty state (SEGMENTED on rtlogo, no detections) → fallback to vtracer-output
- ✅ PREPROCESS + SEGMENTED combination: `median + canny` applied + `obj_clock_0` detected
- ✅ All 7 preprocessing filters work individually when used via `--preprocess`
- ✅ All 4 preprocessing presets exist with correct steps
- ✅ Invalid flag values rejected with exit 2:
  - `--max-svg-size 0` → exit 2 (range check)
  - `--max-svg-size 9999` → exit 2 (range check)
  - `--seg-model foo` → exit 2 with "Valid models: ..." list
  - `--mode bogus` → exit 2 with "Valid modes: ..." list
- ⚠️ `--denoise bogus` is NOT validated (free-form string). Exit 2 only because file doesn't exist. (Acceptable since the pipeline ignores it anyway.)
- ✅ Multi-photo SEGMENTED: 2/5 photos have detections; multi-layer works on different object counts (1, 2, 4 regions)

Evidence: `11-edge-cases.log`, `14-multi-photo-seg.log`

### T-10: Cross-task integration (photo + preprocessing + segmentation) — **PASS**
Command:
```
img2svg convert tests/testimg/Designer\ \(1\).jpeg --output /tmp/qa-integration.svg \
  --mode segmented --preprocess bilateral --preprocess unsharp --preprocess posterize \
  --seg-model yolo11s-seg
```
Result:
- ✅ SVG: `<g id="background">` + `<g id="obj_person_0">` (multi-layer)
- ✅ Sidecar: `preprocessing: [bilateral, unsharp, posterize]`
- ✅ Sidecar: `model_variant: yolo11s-seg`
- ✅ Sidecar: `regions: 1` (1 person detected)
- ✅ Sidecar timings: all 11 keys present (load, preprocess, analyze, classify, select_mode, detect, segment, render, vectorize, write, total)

All 11 timing keys are recorded correctly.

Evidence: `12-integration.log`, `qa-integration.svg/.json`

### T-11: SEGMENTED on multiple photos — **PASS**
| Photo | Detections | Multi-layer | Notes |
|-------|-----------|-------------|-------|
| Designer (1).jpeg | 1 (person) | ✅ bg + obj_person_0 | The critical test |
| Designer (2).jpeg | 1 (cat) | ✅ bg + obj_cat_0 | |
| Designer (5).jpeg | 0 | vtracer-output | Fallback (no objects) |
| Designer (10).jpeg | 0 | vtracer-output | Fallback (no objects) |
| Designer (15).jpeg | 2 (scissors) | ✅ bg + obj_scissors_0 + obj_scissors_1 | Multi-object works |
| Designer (20).jpeg | 4 (spoon, bottle, cake, cat) | ✅ bg + 4 obj_ groups | Multi-class works |

The fix from b5a8124 is confirmed to work across multiple photos with varying object counts.

Evidence: `14-multi-photo-seg.log`, `qa-seg-multi-*.svg/.json`

---

## Definition of Done Status

| DoD Item | Status |
|----------|--------|
| All 10 modes run successfully on rtlogo-1.png | ✅ PASS |
| All 10 outputs are valid XML | ✅ PASS |
| SEGMENTED on Designer (1).jpeg produces MULTI-LAYER output | ✅ PASS (b5a8124 fix verified) |
| Auto mode does NOT pick LABELS or ANNOTATED | ✅ PASS |
| 9 new CLI flags work end-to-end | ❌ **FAIL (5/9)** |
| MAX_SVG_SIZE guard works | ✅ PASS |
| Cross-task integration (photo + preprocess + seg) | ✅ PASS |
| Edge cases tested | ✅ PASS |
| 609/610 fast tests pass | ✅ PASS (1 pre-existing i18n failure) |
| Working tree clean, pushed to origin/main | ✅ PASS |

---

## Bugs Found

### BUG-1: 4 of 9 new CLI flags are non-functional [CRITICAL]

**Affected flags**: `--denoise`, `--sharpen`, `--max-colors`, `--quality`

**Symptom**: Flags are accepted by the CLI help and store their values on `ConversionOptions`, but the pipeline never reads them. The SVG output is identical whether the flag is set or not.

**Evidence**:
- `qa-flag-max-colors-0.svg` and `qa-flag-max-colors-16.svg` are byte-identical (8,525,685 bytes each)
- `--denoise nlmeans` produces the same preprocessing list as default
- `--quality 50` does not appear anywhere in the sidecar JSON (and the help text misleadingly says it should be)

**Root cause**: 
- The fields exist on `ConversionOptions` (T5 deliverable)
- The CLI flags are defined (T5 deliverable)
- But the pipeline's `_resolve_preprocessing_steps()` in `src/img2svg/pipeline.py` only checks `options.preprocess` and `options.no_preprocess`
- No code reads `options.denoise`, `options.sharpen`, `options.max_colors`, or `options.quality`
- The `--quality` help text "JPEG quality hint stored in the sidecar (1-100)" is a lie — nothing is stored

**Where the wiring should be**:
- `--denoise`/`--sharpen`/`--max-colors`: in `_resolve_preprocessing_steps` in `pipeline.py` and in vtracer's `color_precision`/`filter_speckle` mapping
- `--quality`: in `Sidecar` model + sidecar write in `pipeline.py` (T4's schema, T15's sidecar construction)

**Workaround**: Use `--preprocess bilateral --preprocess unsharp` instead of `--denoise`/`--sharpen` (the underlying filters work, just not via the convenience flags).

---

## Files Saved as Evidence

```
.sisyphus/evidence/final-qa/
├── 01-rtlogo-modes.log         # 10 modes on rtlogo-1.png run log
├── 02-rtlogo-sidecars.log      # 10 modes sidecar analysis
├── 03-rtlogo-structures.log    # SVG structure per mode
├── 04-seg-photo.log            # CRITICAL: SEGMENTED on photo
├── 05-auto-photo.log           # auto-mode constraint test
├── 06-flags.log                # 9 flag tests
├── 07-flag-investigations.log  # bug investigation
├── 08-size-cap-investigation.log  # max-svg-size test
├── 09-invalid-input.log        # invalid input tests
├── 10-batch-test.log           # 50-photo batch
├── 11-edge-cases.log           # edge cases
├── 12-integration.log          # cross-task integration
├── 13-pytest.log               # pytest output
├── 14-multi-photo-seg.log      # SEGMENTED on multiple photos
├── 15-ruff.log                 # ruff full output
├── 15-ruff-summary.log         # ruff summary
├── 16-flag-search.log          # proof that 4 flags are unread
├── qa-rtlogo-*.svg/.json (10 modes)
├── qa-seg-photo.svg/.json
├── qa-auto-photo.svg/.json
├── qa-flag-*.svg/.json (9 flag tests)
├── qa-edge-*.svg/.json
├── qa-integration.svg/.json
├── qa-seg-multi-*.svg/.json
```

---

## VERDICT: REJECT

**Reason**: 4 of 9 new CLI flags explicitly required by the user's task spec are non-functional. The plan's deliverable for T5 added CLI flags without wiring them into the pipeline. The `ConversionOptions` model has the fields, the CLI parses them, but nothing reads them.

**What passes** (most of the work is solid):
- 10/10 modes work on rtlogo-1.png
- SEGMENTED multi-layer output works (b5a8124 fix verified)
- Auto mode never picks LABELS/ANNOTATED
- MAX_SVG_SIZE guard works
- --no-seg fallback works
- --preprocess, --no-preprocess, --seg-model, --max-svg-size all work
- Cross-task integration works
- 609/610 fast tests pass
- Working tree clean, all commits pushed

**What fails**:
- `--denoise`, `--sharpen`, `--max-colors`, `--quality` are non-functional
- These are explicitly required by the user spec: "9 new CLI flags work end-to-end"

**Required to flip to APPROVE**:
1. Wire `--denoise` to call the corresponding preprocessing filter (bilateral/nlmeans/median)
2. Wire `--sharpen` to call the unsharp filter
3. Wire `--max-colors` to set vtracer's `color_precision` (or use k-means to cap)
4. Wire `--quality` to be stored in the sidecar (and fix the misleading help text)
5. Add unit tests that exercise each flag end-to-end
6. Verify the SVG output differs when the flag is set vs not set

After these 6 items are completed and re-verified, the plan will meet the user-specified "9 new CLI flags work end-to-end" criterion.
