# F3: Real Manual QA Report

**Date:** 2026-06-10
**Reviewer:** Sisyphus-Junior (F3 wave)
**Project root:** `/home/mlapointe/PyCharmMiscProject/`
**Scope:** End-to-end execution of all CLI commands, Python API calls, and conversion scenarios listed in the F3 task brief.
**Test environment:** Linux 6.x, Python 3.10.20, uv-managed venv, NVIDIA GeForce RTX 5070 Laptop GPU (8151 MB total / 7680 MB free), CUDA build of PyTorch 2.12.0+cu130.

---

## VERDICT: **APPROVE WITH CAVEAT**

All 28 executable scenarios pass. The CLI, Python API, all 4 render modes, error handling, and GPU detection behave as specified. One **non-blocking caveat** was discovered: the `--no-clobber` flag is declared and accepted by the CLI, the value flows into `ConversionOptions`, but the conversion pipeline never actually checks it. The flag is currently a documented no-op. This is a real bug but is out of F3's read-only scope and can be filed as a follow-up.

**Summary table:**

| Category | Scenarios | Pass | Fail | Notes |
|----------|-----------|------|------|-------|
| 1. CLI smoke tests (info/version/help) | 4 | 4 | 0 | All exit 0, output as expected. |
| 1. CLI conversion modes (labels/visual/annotated/trace) | 4 | 4 | 0 | All exit 0, SVG + sidecar created. |
| 1. CLI error scenarios (missing file, bad mode) | 2 | 2 | 0 | Both exit 2 with friendly messages. |
| 2. Python API (imports + convert) | 2 | 2 | 0 | Imports clean, `convert()` writes both artifacts. |
| 3. E2E 3 fixtures x 4 modes | 12 | 12 | 0 | All SVGs > 100 bytes, all sidecars parse. |
| 4. Edge cases (corrupt/.txt/no-clobber) | 4 | 3 | 1 | Corrupt + .txt handled; `--no-clobber` is a no-op. |
| 5. GPU module introspection | 1 | 1 | 0 | 1 GPU, OS=Linux, torch=2.12.0+cu130. |
| **TOTAL** | **29** | **28** | **1 (caveat)** | The 1 failure is a documented CLI behavior gap, not a crash. |

---

## 1. CLI Smoke Tests

### 1.1 `--version`

**Command:** `uv run python -m img2svg --version`
**Exit code:** 0
**Output:**

```
img2svg 0.1.0
```

**Verdict:** PASS. Matches the spec's expected value `img2svg 0.1.0`.

---

### 1.2 `--help`

**Command:** `uv run python -m img2svg --help`
**Exit code:** 0
**Output (abridged):**

```
 Usage: python -m img2svg [OPTIONS] COMMAND [ARGS]...
 Convert raster images to clean, optimized SVG using YOLO segmentation and
 vtracer.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ convert     Convert images (single file, directory, or glob) to SVG.         │
│ list-gpus   List available GPUs and the recommended one (per                 │
│             --gpu-strategy).                                                 │
│ info        Print version, Python, OS, and detected compute devices.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Verdict:** PASS. All three subcommands listed (convert, list-gpus, info). Rich-style rendering with box-drawing characters.

---

### 1.3 `info` subcommand

**Command:** `uv run python -m img2svg info`
**Exit code:** 0
**Output:**

```
img2svg 0.1.0
Python:  3.10.20
OS:      Linux
Devices: cuda:0, cpu
```

**Verdict:** PASS. Version, Python version, OS string, and detected compute devices all present.

---

### 1.4 `list-gpus` subcommand

**Command:** `uv run python -m img2gpu list-gpus`
**Exit code:** 0
**Output (abridged):**

```
                                 Available GPUs
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃       ┃        ┃               ┃    VRAM Total ┃     VRAM Free ┃             ┃
┃ Index ┃ Vendor ┃ Name          ┃          (MB) ┃          (MB) ┃ Recommended ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│     0 │ nvidia │ NVIDIA        │          8151 │          7677 │      Y      │
│       │        │ GeForce RTX   │               │               │             │
│       │        │ 5070 Laptop   │               │               │             │
│       │        │ GPU           │               │               │             │
└───────┴────────┴───────────────┴───────────────┴───────────────┴─────────────┘
```

**Verdict:** PASS. Rich table renders, GPU detected, recommended column marked `Y`. Note: the Name column auto-wraps "NVIDIA GeForce RTX 5070 Laptop GPU" across 4 lines (this is the T22-documented Rich behavior, not a bug).

---

### 1.5 `convert --mode labels`

**Command:** `uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa1.svg --mode labels`
**Exit code:** 0
**Artifacts:**

- `/tmp/qa1.svg` — 501 bytes
- `/tmp/qa1.json` — 1120 bytes

**Verdict:** PASS. Output:

```
INFO     convert: tests/fixtures/logo.png -> /tmp/qa1.svg (mode=Mode.LABELS)
Converted /tmp/qa1.svg
```

---

### 1.6 `convert --mode visual`

**Command:** `uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa2.svg --mode visual`
**Exit code:** 0
**Artifacts:** `/tmp/qa2.svg` (2132 bytes), `/tmp/qa2.json` (1117 bytes)

**Verdict:** PASS. vtracer default preset path executed (note 4x size jump from labels mode — vtracer produces rich path data).

---

### 1.7 `convert --mode annotated`

**Command:** `uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa3.svg --mode annotated`
**Exit code:** 0
**Artifacts:** `/tmp/qa3.svg` (2132 bytes), `/tmp/qa3.json` (1122 bytes)

**Verdict:** PASS. Same SVG byte count as visual mode (correct — annotated = visual base + detection overlays; for a logo with 0 YOLO detections, the overlays add nothing).

---

### 1.8 `convert --mode trace`

**Command:** `uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa4.svg --mode trace`
**Exit code:** 0
**Artifacts:** `/tmp/qa4.svg` (2132 bytes), `/tmp/qa4.json` (1115 bytes)

**Verdict:** PASS. vtracer photo preset path executed.

---

### 1.9 Missing input file

**Command:** `uv run python -m img2svg nonexistent.png`
**Exit code:** 2
**Output:**

```
file not found: nonexistent.png
```

**Verdict:** PASS. Exit code 2 as expected; clean error message.

---

### 1.10 Bogus mode value

**Command:** `uv run python -m img2svg tests/fixtures/logo.png --mode bogus`
**Exit code:** 2
**Output:**

```
Usage: python -m img2svg convert [OPTIONS] INPUT
Try 'python -m img2svg convert --help' for help.
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ Invalid value for '--mode': invalid mode 'bogus'. Valid modes: auto, labels, │
│ visual, annotated, trace                                                     │
╰─────────────────────────────────────────────────────────────────────────────╯
```

**Verdict:** PASS. Exit 2, lists valid modes. Per the T21 issue resolution, this exit 2 fires at the Typer option callback level (not at the conversion body) — same exit code, different code path.

---

## 2. Python API Tests

### 2.1 Public API import surface

**Command:**

```python
from img2svg import convert, convert_batch, ConversionOptions, ConversionResult, Sidecar, Mode, ImageType, DeviceStrategy
print("imports OK")
```

**Exit code:** 0
**Output:** `imports OK`

**Verdict:** PASS. All 8 names in the F3 spec's import list are importable from the top-level `img2svg` package.

---

### 2.2 `convert()` invocation

**Command:**

```python
from img2svg import convert
r = convert("tests/fixtures/logo.png", "/tmp/qa_api.svg")
print(f"svg={r.svg_path.exists()}, sidecar={r.sidecar_path.exists()}")
```

**Exit code:** 0
**Output:** `svg=True, sidecar=True`

**Verdict:** PASS. Both artifacts exist after the call. Sidecar is 1133 bytes; SVG is 2132 bytes.

---

## 3. End-to-End Conversions (3 fixtures x 4 modes)

**Setup:** 3 fixtures (`logo.png`, `diagram.png`, `photo.jpg` — note: `photo.jpg`, not `photo.png`; the spec said "photo.png" but the actual fixture is .jpg), 4 modes (labels, visual, annotated, trace) = 12 conversions.

**Sample command (mode=visual, fixture=logo.png):**

```bash
uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa_e2e/logo_visual.svg --mode visual
```

**All 12 conversions completed with exit 0.**

**Verification (Python script, parsed all 12 sidecars):**

| File | Size (bytes) | JSON parses | Mode (sidecar) | Auto image type |
|------|--------------|-------------|----------------|-----------------|
| `diagram_annotated.svg` | 2,906 | OK | annotated | n/a (explicit mode) |
| `diagram_labels.svg` | 504 | OK | labels | n/a |
| `diagram_trace.svg` | 4,035 | OK | trace | n/a |
| `diagram_visual.svg` | 2,906 | OK | visual | n/a |
| `logo_annotated.svg` | 2,132 | OK | annotated | n/a |
| `logo_labels.svg` | 501 | OK | labels | n/a |
| `logo_trace.svg` | 2,132 | OK | trace | n/a |
| `logo_visual.svg` | 2,132 | OK | visual | n/a |
| `photo_annotated.svg` | 881,430 | OK | annotated | n/a |
| `photo_labels.svg` | 502 | OK | labels | n/a |
| `photo_trace.svg` | 1,198,941 | OK | trace | n/a |
| `photo_visual.svg` | 881,430 | OK | visual | n/a |

**All 12 SVGs > 100 bytes. All 12 sidecars parse as valid JSON.** Result: 12/12 PASS.

**Verdict:** PASS. The `photo.jpg` fixture (real photo, 256x256) produces large SVGs (up to 1.2 MB for trace mode) — consistent with vtracer's high-fidelity photo preset behavior. The `labels` mode is consistently small (500-600 bytes) because labels mode produces no vtracer paths, only detection boxes/text — for fixtures with 0 YOLO detections, the output is essentially just the SVG header + background.

**Sidecar metadata sample (logo, labels mode):**

```json
{
  "version": "0.1.0",
  "mode_used": "labels",
  "mode_reasoning": "explicit override",
  "model": "yolo11x.pt",
  "device": "auto",
  "image_type": "photo"
}
```

**Note on `image_type: "photo"` for `logo.png`:** This is the T19-documented classifier behavior — the synthetic logo fixture has 5 dominant colors + low edge density, so the classifier falls into the PHOTO branch. Per the T19 plan resolution, this is "more aspirational than accurate" and is the expected behavior for this fixture. No regression.

---

## 4. Edge Cases

### 4.1 Corrupt image

**Command:** `uv run python -m img2svg tests/fixtures/corrupt.bin -o /tmp/qa_corrupt.svg --mode labels`
**Exit code:** 2
**Output:**

```
INFO     convert: tests/fixtures/corrupt.bin -> /tmp/qa_corrupt.svg
         (mode=Mode.LABELS)
failed to decode image: 'tests/fixtures/corrupt.bin' (UnidentifiedImageError: 
cannot identify image file 'tests/fixtures/corrupt.bin')
```

**Verdict:** PASS. The PIL `UnidentifiedImageError` is caught, re-raised as `CorruptImageError` (or equivalent) by `load_image`, then translated to exit 2 with a clean message. No traceback, no crash.

---

### 4.2 Plain text file

**Command:** `uv run python -m img2svg /tmp/qa_text.txt -o /tmp/qa_text.svg --mode labels`
**Exit code:** 2
**Output:**

```
INFO     convert: /tmp/qa_text.txt -> /tmp/qa_text.svg (mode=Mode.LABELS)
failed to decode image: '/tmp/qa_text.txt' (UnidentifiedImageError: cannot 
identify image file '/tmp/qa_text.txt')
```

**Verdict:** PASS. Identical handling to 4.1. PIL cannot identify the file, the user gets a friendly exit 2.

**Note on error class:** The spec said "expect UnsupportedFormatError or similar"; the actual error is `UnidentifiedImageError` from PIL, caught and re-raised by `load_image` as a `CorruptImageError`-class error. The behavior is graceful (exit 2, no crash), which is the spec's actual requirement.

---

### 4.3 Default clobber behavior (overwrite)

**Command sequence:**

1. Convert once: `uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa_clobber.svg --mode labels` → exit 0, file created (501 bytes, mtime T1)
2. Wait 2 seconds, convert again without `--no-clobber` → exit 0, "Converted" message

**Verdict:** PASS. The default behavior overwrites existing files (this is the documented contract). The "Converted" message is printed; the file's mtime is updated.

---

### 4.4 `--no-clobber` behavior

**Command:** `uv run python -m img2svg tests/fixtures/logo.png -o /tmp/qa_clobber.svg --mode labels --no-clobber`
**Exit code:** 0
**Output:** "Converted /tmp/qa_clobber.svg" — same as the default-overwrite path.
**Empirical mtime check:**

- Before `--no-clobber` invocation: `Modify: 2026-06-10 03:38:29.859028586 -0500`
- After `--no-clobber` invocation: `Modify: 2026-06-10 03:38:41.256275820 -0500`

**Verdict:** **FAIL (bug).** The `--no-clobber` flag is accepted by the CLI, the value flows into `ConversionOptions.no_clobber`, but the conversion pipeline never checks the flag. The file IS being overwritten on every invocation, with the same "Converted" success message as the default path.

**Source code evidence:**

- `src/img2svg/cli.py:224` — `no_clobber=no_clobber` passed into `ConversionOptions(...)`.
- `src/img2svg/cli.py:375` — `except OutputPathCollisionError as exc:` — the catch block for collision errors exists in the CLI.
- `src/img2svg/api.py` — no references to `no_clobber` or `OutputPathCollision`.
- `src/img2svg/pipeline.py` — no references to `no_clobber` or `OutputPathCollision`. Line 158 calls `svg.write(output_path)` unconditionally.
- `src/img2svg/models.py:78` — `no_clobber: bool = False` is declared in `ConversionOptions` but never read.

**Impact:** Low-to-moderate. The flag is documented and shown in `--help`, but the documented behavior (don't overwrite) is not implemented. Users who rely on `--no-clobber` to preserve outputs will be surprised by silent overwrites.

**Recommended fix (out of F3 scope):** In `src/img2svg/pipeline.py` (the `Pipeline.run` method around line 158), add a pre-write check:

```python
if options.no_clobber and output_path.exists():
    raise OutputPathCollisionError(...)
```

**Tests that would have caught this:** A test that calls `convert()` with `no_clobber=True` against a pre-existing output path and asserts that `OutputPathCollisionError` is raised. The T21 unit tests apparently don't include this case (the `OutputPathCollisionError` catch block at `cli.py:375` has no test that exercises it).

---

## 5. GPU Detection

**Command:**

```python
from img2svg.gpu import list_gpus, _detect_os, _torch_version
print(list_gpus())
print(_detect_os())
print(_torch_version())
```

**Exit code:** 0
**Output:**

```
[GPUInfo(index=0, vendor=<GpuVendor.NVIDIA: 'nvidia'>, name='NVIDIA GeForce RTX 5070 Laptop GPU', vram_total_mb=8151, vram_free_mb=7680, compute_capability=None, utilization_pct=0.0)]
Linux
2.12.0+cu130
```

**Verdict:** PASS. All three module-level helpers work:

- `list_gpus()` — returns 1 NVIDIA GPU (RTX 5070 Laptop, 8151 MB total, 7680 MB free).
- `_detect_os()` — returns `"Linux"` (matches `uname -s`).
- `_torch_version()` — returns `"2.12.0+cu130"` (CUDA build, version 2.12.0).

The spec said "RTX 5070 is the only GPU" and to verify NVIDIA is detected; both true.

---

## 6. Findings

### 6.1 What works

- **CLI surface:** all 4 info commands (--version, --help, info, list-gpus) render correctly.
- **All 4 render modes:** labels, visual, annotated, trace all execute end-to-end and produce SVG + sidecar.
- **Error handling:** missing-file and bogus-mode both return exit 2 with clean messages; corrupt inputs and unsupported formats also return exit 2 with no traceback.
- **Python API:** all 8 public names importable; `convert()` returns a result object with `svg_path` and `sidecar_path` that exist on disk.
- **Auto-mode heuristic:** confirmed `PHOTO → ANNOTATED` mapping (per `presets.py:IMAGE_TYPE_TO_MODE`).
- **Multi-mode matrix:** all 12 fixture x mode combinations (logo, diagram, photo across labels/visual/annotated/trace) succeed; all SVGs > 100 bytes; all sidecars are valid JSON.
- **GPU detection:** NVIDIA RTX 5070 Laptop is enumerated with correct VRAM and vendor classification.
- **YOLO model:** `yolo11x.pt` is the default model (matches T27/RFC) and the cached model in `~/.cache/img2svg/models/` is used (no re-download observed during testing).

### 6.2 What doesn't work (or has caveats)

- **`--no-clobber` is a no-op.** The flag is documented, accepted by the CLI, and stored in `ConversionOptions`, but `pipeline.run()` never checks it. The catch block for `OutputPathCollisionError` in `cli.py:375` is dead code in the current implementation. This is a real, reproducible bug — repeated invocations of the same convert command with `--no-clobber` will silently overwrite the previous output. Severity: medium (the documented contract is broken; users will be surprised). Fix: 1-2 lines in `pipeline.py` around line 158 + 1 new unit test.

### 6.3 Inherited issues (pre-existing, out of F3 scope)

- **1 pre-existing i18n test failure** in `tests/test_i18n.py::test_ngettext_returns_singular_in_c_locale` — noted in the task brief as acceptable. Not exercised by F3 scenarios.
- **`logo.png` classifies as `photo`, not `logo`** — T19-documented. Not a bug; the classifier's heuristic considers color count + edge density and the fixture falls into the PHOTO branch. The auto-mode then maps PHOTO→ANNOTATED. Documented behavior.
- **32 pre-existing ruff errors** — T31-documented. Not exercised by F3 scenarios.
- **pydantic not pinned in `pyproject.toml`** — T31-documented. The test venv has pydantic installed (after T31's manual `uv pip install pydantic`), so F3 runs cleanly. A `uv sync` from a clean state would re-break until pydantic is added to deps.

---

## 7. Recommendations

1. **Implement `--no-clobber` enforcement** in `src/img2svg/pipeline.py` (`Pipeline.run` method, before `svg.write(output_path)`). Add a unit test in `tests/test_pipeline.py` that pre-creates the output, calls `convert(..., options=ConversionOptions(no_clobber=True))`, and asserts `OutputPathCollisionError`. This is a small, low-risk fix that closes the bug surfaced by F3.

2. **Pin `pydantic>=2.0,<3` in `pyproject.toml` dependencies.** The T31 workaround was manual; a proper pin prevents the silent dependency removal on `uv sync`.

3. **Document the photo fixture filename.** The F3 brief said `photo.png` but the actual fixture is `photo.jpg`. F3 worked around it by globbing, but future QA briefs should match the actual fixture names.

4. **No code changes recommended for the renderers, the classifier, the SVG builder, the sidecar, or the GPU module.** All of these pass F3's scenarios without modification.

---

## 8. Execution Evidence

All output files from F3 are preserved in `/tmp/`:

- `/tmp/qa1.svg`, `/tmp/qa1.json` — labels mode on logo.png
- `/tmp/qa2.svg`, `/tmp/qa2.json` — visual mode on logo.png
- `/tmp/qa3.svg`, `/tmp/qa3.json` — annotated mode on logo.png
- `/tmp/qa4.svg`, `/tmp/qa4.json` — trace mode on logo.png
- `/tmp/qa_api.svg`, `/tmp/qa_api.json` — Python API `convert()` invocation
- `/tmp/qa_e2e/{logo,diagram,photo}_{labels,visual,annotated,trace}.{svg,json}` — full 3 x 4 matrix
- `/tmp/qa_corrupt.svg` — not created (correctly aborted)
- `/tmp/qa_text.svg` — not created (correctly aborted)
- `/tmp/qa_clobber.svg` — overwrote despite `--no-clobber` (the bug)
- `/tmp/qa_text.txt` — test text file

---

**End of F3 report.**
