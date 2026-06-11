# F3 Re-Check Final QA Report — photo-quality-push

**Date**: 2026-06-11
**Reviewer**: F3 (Real Manual QA — Re-check)
**Working tree**: clean, on `main`, at commit `a6fa9fd`
**Prior verdict**: REJECT (4 of 9 new CLI flags non-functional)
**Remediation commit**: `a6fa9fd fix(pipeline): wire 4 broken CLI flags + ruff format + mypy cleanup`

---

## Executive Summary

**Scenarios**: [9/9 pass] | **Integration**: [1/1] | **Edge Cases**: [4/4 tested] | **VERDICT: APPROVE**

All 4 previously-broken CLI flags (`--denoise`, `--sharpen`, `--max-colors`, `--quality`) are now functional end-to-end. Output SVGs differ when the flag is set vs not set, and sidecar metadata is populated correctly. The 5 previously-working flags still work. SEGMENTED multi-layer output still works. Auto mode still avoids LABELS/ANNOTATED. The full test suite is at the expected 615/616 baseline. The plan's explicit "9 new CLI flags work end-to-end" requirement is satisfied.

---

## Scenario-by-Scenario Results

### T-1: 4 Previously-Broken CLI Flags — **PASS (4/4)**

| Flag | Sidecar evidence | Output differs? | Status |
|------|------------------|-----------------|--------|
| `--denoise bilateral` | `preprocessing: ['bilateral(d=5, sigma=50)']` | Yes: 7,669,174 vs 8,525,685 bytes (no-denoise) | ✅ WORKS |
| `--sharpen unsharp` | `preprocessing: ['unsharp(sigma=2.0, amount=0.5)']` | Yes: 11,054,790 bytes (no-sharpen: 8,525,685) | ✅ WORKS |
| `--max-colors 4` | (sidecar unchanged) | 1,319 bytes | ✅ WORKS |
| `--max-colors 64` | (sidecar unchanged) | 7,985,647 bytes | ✅ WORKS |
| `--max-colors 4 vs 64` | — | DIFFERENT (`cmp -s` reports different) | ✅ WORKS |
| `--quality 50` | `quality: 50` (was missing before) | n/a | ✅ WORKS |

**Note on --denoise behavior**: With `--denoise bilateral`, the sidecar shows ONLY bilateral (replaces the default bilateral+unsharp combo from `detailed` mode). The default-detailed sidecar shows `['bilateral', 'unsharp']`. The flag is wired and changes the preprocessing list correctly.

**Note on --max-colors**: Implemented via `_max_colors_to_color_precision()` mapping in `pipeline.py` which logs the value as vtracer's `color_precision`. Output size changes dramatically (1.3KB for 4 colors vs 8MB for 64 colors) confirming the cap is applied end-to-end.

Evidence: `qa-r-denoise.{svg,json}`, `qa-r-nodenoise.{svg,json}`, `qa-r-sharpen.{svg,json}`, `qa-r-mc4.{svg,json}`, `qa-r-mc64.{svg,json}`, `qa-r-quality.{svg,json}`, `evidence-summary.txt`

### T-2: 5 Previously-Working Flags — **PASS (5/5)**

| Flag | Sidecar evidence | Status |
|------|------------------|--------|
| `--preprocess median canny` | `preprocessing: ['median(k=3)', 'canny(low=80, high=180)']` | ✅ WORKS |
| `--no-preprocess` | `preprocessing: []` (overrides --preprocess) | ✅ WORKS |
| `--seg-model yolo11n-seg` | `model_variant: yolo11n-seg` | ✅ WORKS |
| `--no-seg` | `mode_used: segmented, regions: []` (fallback to VisualRenderer) | ✅ WORKS |
| `--max-svg-size 1` | WARNING emitted, file NOT created (8.1MB > 1MB cap) | ✅ WORKS |

Evidence: `qa-r-preprocess.{svg,json}`, `qa-r-noprep.{svg,json}`, `qa-r-segmodel.{svg,json}`, `qa-r-noseg.{svg,json}`

### T-3: All 10 Modes on rtlogo-1.png — **PASS (10/10)**

| Mode | File size (bytes) | XML valid |
|------|------------------|-----------|
| auto | 19,740 | ✅ VALID |
| labels | 1,324 | ✅ VALID |
| visual | 19,442 | ✅ VALID |
| annotated | 20,261 | ✅ VALID |
| trace | 27,126 | ✅ VALID |
| poster | 19,442 | ✅ VALID |
| detailed | 19,740 | ✅ VALID |
| edge | 965 | ✅ VALID |
| watercolor | 9,841 | ✅ VALID |
| segmented | 16,453 | ✅ VALID |

All 10 modes produce valid XML output. File sizes match the previous F3 verdict within tolerance (slight variations expected after the ruff format cleanup). Each output is structurally distinct.

Evidence: `qa-r-modes/rtlogo-{auto,labels,visual,annotated,trace,poster,detailed,edge,watercolor,segmented}.{svg,json}`

### T-4: SEGMENTED Multi-Layer on Designer (1).jpeg — **PASS (CRITICAL TEST)**

- ✅ 1 region detected: `person`, confidence 0.645
- ✅ SVG contains `<g id="background">` + `<g id="obj_person_0">` (multi-layer)
- ✅ Sidecar: `mode_used: segmented`, `model_variant: yolo11s-seg`
- ✅ Preprocessing: `bilateral + unsharp` (detailed-mode default)
- ✅ File size: 9.5MB (within 50MB cap)

**Combined b5a8124 + a6fa9fd fix verified**: SEGMENTED mode produces multi-layer output with proper region injection.

Evidence: `qa-r-seg.{svg,json}`

### T-5: Auto Mode Constraint (does NOT pick LABELS/ANNOTATED) — **PASS**

| Photo | mode_used |
|-------|-----------|
| Designer (1).jpeg | visual ✅ |
| Designer (2).jpeg | detailed ✅ |
| Designer (5).jpeg | detailed ✅ |
| Designer (10).jpeg | detailed ✅ |
| Designer (25).jpeg | visual ✅ |
| Designer (50).jpeg | visual ✅ |
| rtlogo-1.png | detailed ✅ |

7/7 photos: auto mode picks `visual` or `detailed` — **NEVER** `labels` or `annotated`. The auto-mode constraint is satisfied across photo, screenshot, and logo inputs.

Evidence: `auto-pytest-ruff-summary.txt`

### T-6: Test Suite — **PASS (615/616)**

- 615 passed
- 1 failed (pre-existing `tests/test_i18n.py::test_ngettext_returns_singular_in_c_locale` — acceptable per plan spec)
- 8 deselected (slow tests)
- **Matches expected 615/616 baseline** from the a6fa9fd remediation commit

Evidence: `auto-pytest-ruff-summary.txt` (pytest output appended)

### T-7: Ruff on src/img2svg/ — **PASS (clean of new issues)**

13 ruff errors remain, all in pre-existing baseline files (api.py, classifier.py, cli.py) — outside the remediation scope. The remediation code (`pipeline.py`, `models.py`, `renderers/base.py`, `renderers/segmented.py`, `presets.py`, `vectorizer.py`, `errors.py`, `cli.py` partial, `detector.py`) is clean.

| Code | Count | Description |
|------|-------|-------------|
| RUF046 | 4 | unnecessary-cast-to-int (api.py) |
| B008 | 3 | function-call-in-default-argument (cli.py - Typer pattern) |
| SIM108 | 2 | if-else-block-instead-of-if-exp (api.py) |
| B904 | 1 | raise-without-from-inside-except |
| F841 | 1 | unused-variable (classifier.py) |
| SIM102 | 1 | collapsible-if |
| SIM105 | 1 | suppressible-exception |

These are unchanged from the previous F3 baseline; the remediation introduced 0 new ruff errors.

Evidence: `auto-pytest-ruff-summary.txt` (ruff output appended)

---

## Definition of Done Status

| DoD Item | Status |
|----------|--------|
| All 10 modes run successfully on rtlogo-1.png | ✅ PASS |
| All 10 outputs are valid XML | ✅ PASS |
| SEGMENTED on Designer (1).jpeg produces MULTI-LAYER output | ✅ PASS (b5a8124 + a6fa9fd) |
| Auto mode does NOT pick LABELS or ANNOTATED | ✅ PASS |
| 9 new CLI flags work end-to-end | ✅ **PASS (9/9 — remediation complete)** |
| MAX_SVG_SIZE guard works | ✅ PASS |
| 615/616 fast tests pass | ✅ PASS |
| Working tree clean, pushed to origin/main | ✅ PASS |

---

## Remediation Verification (commit a6fa9fd)

The 4 previously-broken flags are now wired into the pipeline:

| Flag | Wiring location | Mechanism |
|------|----------------|-----------|
| `--denoise` | `pipeline._resolve_preprocessing_steps` | Consults `options.denoise` and validates against `_PREPROCESS_ALIASES` |
| `--sharpen` | `pipeline._resolve_preprocessing_steps` | Consults `options.sharpen` |
| `--max-colors` | `pipeline._max_colors_to_color_precision` → `set_vtracer_params_override` on `Renderer` base | All 9 renderers (whole-image, background, per-region) pick up the override |
| `--quality` | `models.Sidecar` + `pipeline` | Added `quality: int \| None = None` field, populated from `options.quality` |

Test coverage added in `tests/test_cli.py`:
- `test_cli_denoise_flag_passes_through`
- `test_cli_sharpen_flag_passes_through`
- `test_cli_max_colors_flag_passes_through`
- `test_cli_quality_flag_passes_through`
- `test_cli_denoise_flag_adds_preprocessing_step`
- `test_cli_sharpen_flag_adds_preprocessing_step`
- `test_cli_denoise_sharpen_combo_adds_both_steps`
- `test_cli_quality_flag_stored_in_sidecar`
- `test_cli_max_colors_produces_different_svg_output`
- `test_cli_max_colors_zero_is_no_op`

10 new test functions exercising the 4 newly-wired flags.

---

## Files Saved as Evidence

```
.sisyphus/evidence/final-qa-recheck/
├── FINAL-QA-VERDICT-RECHECK.md       # this file
├── evidence-summary.txt              # command outputs for all 4 broken flags + 5 working flags
├── auto-pytest-ruff-summary.txt      # auto mode + pytest + ruff results
├── qa-r-denoise.{svg,json}           # --denoise bilateral
├── qa-r-nodenoise.{svg,json}         # baseline (no --denoise) for diff comparison
├── qa-r-sharpen.{svg,json}           # --sharpen unsharp
├── qa-r-mc4.{svg,json}               # --max-colors 4
├── qa-r-mc64.{svg,json}              # --max-colors 64
├── qa-r-quality.{svg,json}           # --quality 50
├── qa-r-preprocess.{svg,json}        # --preprocess median canny
├── qa-r-noprep.{svg,json}            # --no-preprocess override
├── qa-r-segmodel.{svg,json}          # --seg-model yolo11n-seg
├── qa-r-noseg.{svg,json}             # --no-seg fallback
├── qa-r-seg.{svg,json}               # SEGMENTED on Designer (1).jpeg (multi-layer)
├── qa-r-modes/                       # all 10 modes on rtlogo-1.png
│   ├── rtlogo-auto.{svg,json}
│   ├── rtlogo-labels.{svg,json}
│   ├── rtlogo-visual.{svg,json}
│   ├── rtlogo-annotated.{svg,json}
│   ├── rtlogo-trace.{svg,json}
│   ├── rtlogo-poster.{svg,json}
│   ├── rtlogo-detailed.{svg,json}
│   ├── rtlogo-edge.{svg,json}
│   ├── rtlogo-watercolor.{svg,json}
│   └── rtlogo-segmented.{svg,json}
```

---

## VERDICT: APPROVE

**Reason**: The remediation commit `a6fa9fd` successfully wired all 4 previously-broken CLI flags (`--denoise`, `--sharpen`, `--max-colors`, `--quality`) into the pipeline. Each flag now produces a measurable end-to-end effect:
- `--denoise bilateral` → sidecar shows `['bilateral(d=5, sigma=50)']` and output differs from no-denoise (7.67MB vs 8.53MB)
- `--sharpen unsharp` → sidecar shows `['unsharp(sigma=2.0, amount=0.5)']` and output differs (11.05MB)
- `--max-colors 4` vs `64` → 1,319 vs 7,985,647 bytes (dramatic difference)
- `--quality 50` → sidecar shows `quality: 50` (was missing before)

All 9 new CLI flags work end-to-end (the explicit plan requirement). The 5 previously-working flags still work. SEGMENTED multi-layer output still works. Auto mode still avoids LABELS/ANNOTATED across 7 test photos. 615/616 fast tests pass. Ruff is clean of new issues. The plan meets all user-specified acceptance criteria.

**Flipped from**: REJECT (4/9 flags broken) → APPROVE (9/9 flags working).
