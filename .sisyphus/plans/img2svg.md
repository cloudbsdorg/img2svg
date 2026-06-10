# img2svg — Image-to-SVG Converter with Object & Pattern Detection

## TL;DR

> **Quick Summary**: A Python CLI tool and importable library that converts raster images (PNG, JPEG, BMP, WebP, TIFF, GIF, etc.) into SVG, using YOLO ML object detection + OpenCV geometric pattern detection, with 4 configurable output modes (labels, visual, annotated, trace) and automatic GPU selection across NVIDIA CUDA, AMD ROCm, Apple MPS, and CPU.
>
> **Deliverables**:
> - Python package `img2svg` importable as a library: `from img2svg import convert, convert_batch, ConversionOptions, ConversionResult`
> - CLI binary `img2svg` (thin Typer wrapper around the library)
> - GPU autodetect + recommendation (lists devices, recommends by raw power or availability)
> - 4 output modes: `labels` (semantic `<g>` groups), `visual` (vtracer default), `annotated` (visual + bounding boxes), `trace` (vtracer photo preset)
> - JSON metadata sidecar per output
> - Rich terminal output (progress bars, tables, colors)
> - Full documentation (README + `docs/` + man page)
> - Jenkinsfile + local `scripts/ci.sh` (no GitHub integrations available — runs on "this system")
> - Test suite (pytest, TDD, 80%+ coverage, AI-parseable output)
> - BSD 3-Clause license, XDG-compliant config, i18n-ready
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves, 5–8 tasks per wave
> **Critical Path**: T1 (scaffolding) → T5 (device/GPU) → T8 (detector) → T12 (auto-mode) → T16 (CLI integration) → F1–F4 (verification)

---

## Context

### Original Request
A program that takes in an image (PNG, JPEG, BMP, etc.), performs object and pattern detection, and converts the image to SVG format.

### Interview Summary

**Key Decisions**:
- **Library-first Python API** — CLI is a thin wrapper; another app can import and call
- **Image types**: ALL (logos, photos, diagrams, mixed) — adaptive pipeline
- **Detection**: Hybrid (ML via YOLO + geometric via OpenCV)
- **Output modes**: All 4 (labels, visual, annotated, trace) — auto-detect picks best, `--mode` overrides
- **Platform**: Python 3.10+, Linux (required), FreeBSD (best-effort), macOS (supported)
- **YOLO model default**: `yolo11x.pt` (user chose for accuracy); `--model` flag for `yolo11n/s/m/l/x`
- **Vectorization**: `vtracer` (MIT, Rust/PyO3, prebuilt wheels, multi-color + binary)
- **Geometric CV**: OpenCV (K-means in LAB, Canny, findContours, Hough)
- **CLI**: Typer, terminal output: Rich, SVG: lxml
- **GPU**: Auto-detect priority = CUDA (covers NVIDIA + AMD ROCm) → DirectML (Windows AMD) → MPS (Apple) → CPU
- **GPU recommendation**: `--list-gpus` enumerates; `--gpu-strategy power|availability` picks
- **Multi-vendor GPU support**: PyTorch's `cuda.is_available()` returns True for both NVIDIA CUDA and AMD ROCm — single code path
- **Batch**: multiple files or directory
- **Logging**: Rich terminal + JSON sidecar; `-q`/`-v` flags
- **Tests**: TDD, pytest, 80%+ coverage, 100% critical paths, AI-parseable output (JSON/TAP/JUnit XML)
- **License**: BSD 3-Clause, Copyright (c) 2026, CloudBSD
- **Git author**: `Mark LaPointe <mark@cloudbsd.org>` (per CloudBSD guidelines)
- **Config**: XDG Base Directory (`$XDG_CONFIG_HOME/img2svg/`, `$XDG_DATA_HOME/img2svg/`, `$XDG_CACHE_HOME/img2svg/`)
- **i18n**: gettext for user-facing strings
- **Docs**: README + `docs/` (Mermaid diagrams) + man page (`man/img2svg.1`)
- **CI**: Jenkinsfile + local `scripts/ci.sh` (no GitHub integrations; runs on Linux + macOS × Python 3.10/3.11/3.12)
- **Git remote**: `git@github.com:cloudbsdorg/img2svg.git` (exists, has commits)

**Research Findings**:
- **vtracer** is the only Python vectorizer that ships prebuilt wheels for all platforms, has a permissive license, handles full-color + binary, and is fast enough for 10+ megapixel inputs. 6.2k stars, MIT, last release Mar 23, 2026. Used in production by Alibaba.
- **ultralytics** (YOLO11x) is 56MB, ~480ms CPU, ~11ms T4 GPU. AGPL-3.0 license — must be addressed before shipping.
- **OpenCV** for geometric: K-means (LAB), Canny, findContours, Hough
- **supervision** (40k+ stars, MIT) recommended as glue for detector-agnostic API
- **PyTorch** with ROCm: `torch.cuda.is_available()` returns True for both NVIDIA and AMD — single path covers both

### Metis Review
**Identified Gaps (addressed)**:
- **AGPL-3.0 inheritance**: Flagged as risk. Default: assume AGPL acceptable (BSD-3-Clause + AGPL deps is permitted). User can opt for ONNX export path if needed.
- **"Photorealistic trace" undefined**: Renamed to `trace` mode, spec'd as vtracer photo preset with multi-color config.
- **Geometric CV role**: Defined — provides per-ROI color analysis (K-means in LAB) and contour data; not used for primary detection. Outputs feed into per-detection sub-shapes and sidecar metadata.
- **YOLO model size**: Confirmed 56MB for yolo11x. Model cache: `$XDG_CACHE_HOME/img2svg/models/`. Add `--model` flag.
- **`--device` failure behavior**: `auto` = warn-and-fall-back; explicit `--device X` = hard-fail with clear error.
- **JSON sidecar schema**: Defined as Pydantic model with `version`, `input_path`, `input_hash`, `output_path`, `output_size`, `mode_used`, `mode_reasoning`, `model`, `device`, `image_type`, `detections[]`, `geometric`, `timings`, `timestamp`.
- **Image type classifier**: Heuristic on (alpha channel, color count, edge density, dominant saturation). Outputs: `logo | photo | diagram | screenshot | line_art | unknown`.
- **Auto-mode mapping**: `logo → labels|visual`, `photo → trace|annotated`, `diagram → labels`, `screenshot → labels|visual`, `line_art → labels|wireframe`. `auto` prints reasoning.
- **Exit codes**: 0=success, 1=partial fail, 2=invalid args, 3=dep/model fail.
- **Batch semantics**: Continue-on-individual-fail with summary.
- **Output path collision**: Default overwrite with warning; `--no-clobber` to skip.
- **CI**: Minimal — install + lint (`ruff`) + fast tests on matrix.
- **Test output AI-parseable**: pytest plugins: `pytest-json-report` (JSON), `pytest-tap` (TAP), `pytest-junitxml` (JUnit XML).
- **Heavy deps**: Add `[ml]` extra for torch/ultralytics, `[vtracer-only]` for non-ML users.
- **Pinned versions**: All critical deps in `pyproject.toml` with `>=X,<Y` ranges; `uv.lock` committed.

---

## Work Objectives

### Core Objective
Build `img2svg` — a Python library + CLI that converts raster images to SVG using ML object detection (YOLO) + geometric pattern detection (OpenCV), with 4 configurable output modes, auto GPU selection across NVIDIA/AMD/Apple/CPU, BSD 3-Clause license, and full documentation (README, man page, docs/, Mermaid diagrams).

### Concrete Deliverables
- [ ] `pyproject.toml` with BSD 3-Clause license, pinned deps, `[ml]` extra
- [ ] `src/img2svg/` package with public API: `convert`, `convert_batch`, `ConversionOptions`, `ConversionResult`, enums
- [ ] `src/img2svg/device.py` — GPU autodetect (CUDA covers NVIDIA + AMD ROCm) + CPU/MPS fallback
- [ ] `src/img2svg/gpu.py` — GPU enumeration + recommendation (power | availability)
- [ ] `src/img2svg/loader.py` — image I/O via Pillow (PNG, JPEG, BMP, WebP, TIFF, GIF)
- [ ] `src/img2svg/classifier.py` — image type heuristic (logo/photo/diagram/screenshot/line_art/unknown)
- [ ] `src/img2svg/detector.py` — ultralytics YOLO wrapper, COCO 80 classes
- [ ] `src/img2svg/patterns.py` — OpenCV geometric analysis (K-means LAB, Canny, findContours)
- [ ] `src/img2svg/vectorizer.py` — vtracer wrapper with preset configs
- [ ] `src/img2svg/renderers/` — one Renderer per mode (Labels, Visual, Annotated, Trace)
- [ ] `src/img2svg/svg_builder.py` — SVG assembly via lxml
- [ ] `src/img2svg/metadata.py` — JSON sidecar (Pydantic schema)
- [ ] `src/img2svg/pipeline.py` — orchestrates load → classify → detect → analyze → render → write
- [ ] `src/img2svg/cli.py` — Typer CLI (thin wrapper around API)
- [ ] `src/img2svg/paths.py` — XDG Base Directory config/cache/data paths
- [ ] `src/img2svg/i18n.py` — gettext setup for user-facing strings
- [ ] `src/img2svg/errors.py` — typed exceptions (UnsupportedFormat, CorruptImage, ModelLoad, etc.)
- [ ] `src/img2svg/logging.py` — Rich console + structured logging
- [ ] `tests/` — pytest suite, 80%+ coverage, AI-parseable output
- [ ] `docs/` — installation, usage, api, modes, gpu, configuration, troubleshooting, development, architecture, changelog
- [ ] `man/img2svg.1` — man page (NAME/SYNOPSIS/DESCRIPTION/OPTIONS/EXAMPLES/EXIT STATUS/FILES/SEE ALSO/AUTHOR/BUGS)
- [ ] `README.md` — overview, install, quickstart, Mermaid architecture diagram
- [ ] `LICENSE` — BSD 3-Clause
- [ ] `Jenkinsfile` — declarative Jenkins pipeline (Setup, Lint, Test, Build Docs, Package)
- [ ] `scripts/ci.sh` — local CI script that runs on "this system" (no GitHub integrations available)
- [ ] `examples/` — basic_usage.md, sample_inputs/, sample_outputs/
- [ ] Sample test images (synthetic, CC0, < 1MB total)

### Definition of Done
- [ ] `uv sync` installs cleanly on Linux + macOS
- [ ] `img2svg --help` shows all flags with examples
- [ ] `img2svg foo.png -o foo.svg` works on Linux + macOS
- [ ] `img2svg *.png` batch processes with progress bar
- [ ] All 4 modes (`labels`, `visual`, `annotated`, `trace`) produce valid SVG (lxml-parseable)
- [ ] `img2svg --list-gpus` lists available devices with details + recommendation
- [ ] `--device auto` falls back gracefully to CPU when GPU unavailable
- [ ] `--device cuda` fails hard with clear error if CUDA unavailable
- [ ] Multi-vendor GPU (AMD + NVIDIA) both detected; `CUDA_VISIBLE_DEVICES` controls
- [ ] JSON sidecar written per output; schema validates with Pydantic
- [ ] FreeBSD install works (best-effort; may require source builds)
- [ ] `pytest` exits 0 with 80%+ coverage; output in JSON/TAP/JUnit
- [ ] `man img2svg` displays the man page
- [ ] `mkdocs build` (or equivalent) produces HTML docs
- [ ] CI green on Linux + macOS matrix
- [ ] BSD 3-Clause LICENSE file present
- [ ] First commit sets git author to `Mark LaPointe <mark@cloudbsd.org>`

### Must Have
- Library + CLI both functional
- All 4 output modes
- GPU autodetect + recommendation
- Multi-vendor GPU (NVIDIA + AMD) support
- Linux + macOS + FreeBSD (best-effort) platforms
- BSD 3-Clause license
- TDD with pytest, 80%+ coverage
- JSON sidecar
- Rich terminal output
- Man page
- Full documentation (README + docs/)
- XDG-compliant paths
- i18n via gettext

### Must NOT Have (Guardrails)
- MUST NOT: ship YOLO model weights in repo
- MUST NOT: ship copyrighted test images
- MUST NOT: auto-push to git (user action only)
- MUST NOT: modify input files (read-only)
- MUST NOT: silently overwrite output (default: warn + overwrite; `--no-clobber` to skip)
- MUST NOT: ship Docker/rest API/web UI in v1
- MUST NOT: silently fall back from explicit `--device` choices
- MUST NOT: include custom YOLO training
- MUST NOT: include telemetry / phone-home / auto-update
- MUST NOT: include image editing or SVG post-processing
- MUST NOT: trust container/VM OS detection (use `uname -s`)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — all verification is agent-executed. No "user manually tests."

### Test Decision
- **Infrastructure exists**: NO (greenfield)
- **Automated tests**: YES (TDD)
- **Framework**: pytest + pytest-cov + pytest-json-report + pytest-tap + pytest-junitxml
- **Approach**: TDD — write test first, then implementation, then refactor
- **Coverage target**: ≥80% overall, 100% on critical paths (device detection, mode dispatch, file I/O, error mapping)

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/API tests**: pytest with concrete assertions
- **CLI tests**: `typer.testing.CliRunner` with specific input/output and exit code assertions
- **SVG validity**: `lxml.etree.fromstring()` + assert root tag, viewBox
- **JSON sidecar**: Pydantic `model_validate(json.loads(...))`
- **GPU detection**: mock `torch.cuda.is_available()`, `torch.backends.mps.is_available()` for unit tests; real check on integration
- **Multi-vendor GPU**: integration test on dgx system (or CI matrix) if available
- **Error paths**: concrete input (corrupt bytes, missing file) + assert specific error message + exit code

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.
> Target: 5–8 tasks per wave.

```
Wave 1 (Start Immediately — foundation, scaffolding, TDD setup):
├── T1: Project scaffolding + git init + LICENSE + pyproject.toml
├── T2: Path/XDG module + config file loading
├── T3: Typed exceptions + i18n/gettext scaffold
├── T4: Pydantic data models (Detection, GeometricAnalysis, ConversionOptions, ConversionResult, Sidecar)
├── T5: Device detection + GPU enumeration/recommendation module
├── T6: Image loader (Pillow, format validation, alpha handling)
└── T7: Synthetic test image generator + conftest.py fixtures

Wave 2 (After Wave 1 — core detection/analysis modules, MAX PARALLEL):
├── T8: YOLO detector wrapper (ultralytics, model cache, device dispatch)
├── T9: OpenCV geometric patterns (K-means LAB, Canny, findContours, Hough)
├── T10: Image type classifier (heuristic on color/edge stats)
├── T11: vtracer vectorizer wrapper (preset configs)
├── T12: vtracer preset → mode mapping (auto-mode selection)
├── T13: SVG builder (lxml, namespace handling, accessibility)
├── T14: JSON sidecar writer (Pydantic serialization)
└── T15: Rich logging + progress bars

Wave 3 (After Wave 2 — renderers + pipeline + CLI, MAX PARALLEL):
├── T16: Renderer base class + LabelsRenderer
├── T17: VisualRenderer + TraceRenderer (vtracer-based)
├── T18: AnnotatedRenderer (visual + bounding boxes)
├── T19: Pipeline orchestrator (load → classify → detect → analyze → render → write)
├── T20: Batch processing (multi-file, directory, glob, error continuation)
├── T21: Typer CLI (subcommands, --list-gpus, --gpu-strategy, --mode, --device, --model)
├── T22: GPU recommendation CLI command + Rich table
└── T23: Man page (man/img2svg.1)

Wave 4 (After Wave 3 — docs, CI, integration tests, packaging):
├── T24: README.md (overview, install, quickstart, Mermaid arch diagram)
├── T25: docs/ (installation, usage, api, modes, gpu, configuration, troubleshooting, development, architecture, changelog)
├── T26: Jenkinsfile + local `scripts/ci.sh` (matrix: Linux + macOS × 3.10/3.11/3.12)
├── T27: Integration test suite (real YOLO on small model, real vtracer, batch, GPU)
├── T28: PyPI packaging metadata (readme rendering, classifiers, URLs)
├── T29: FreeBSD compatibility notes + smoke test
├── T30: Example gallery (sample inputs → outputs) + before/after comparisons
└── T31: Final cleanup (deps audit, .gitignore, AGPL notice, BSD headers)

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T4 → T8 → T19 → T21 → F1–F4
Parallel Speedup: ~70% faster than sequential
Max Concurrent: 8 (Wave 2)
```

### Dependency Matrix

```
T1 (scaffolding)  →  T2, T3, T4, T5, T6, T7  →  Wave 2
T4 (Pydantic models)  →  T8, T9, T14
T5 (device)  →  T8, T22
T6 (loader)  →  T10, T19
T7 (fixtures)  →  T8, T9, T10, T11, T27
T8 (YOLO)  →  T19
T9 (patterns)  →  T16, T18
T10 (classifier)  →  T12, T19
T11 (vtracer)  →  T17, T19
T12 (auto-mode)  →  T19
T13 (SVG)  →  T16, T17, T18
T14 (sidecar)  →  T19
T15 (logging)  →  T19, T21, T22
T16-T18 (renderers)  →  T19
T19 (pipeline)  →  T20, T21
T20 (batch)  →  T21
T21 (CLI)  →  T22, T23
T22 (gpu-recommend)  →  T21
T23 (man page)  →  T28
T24-T25 (docs)  →  T28
T26 (CI)  →  independent
T27 (integration tests)  →  Wave 4
T28 (PyPI)  →  T29
T29 (FreeBSD)  →  independent
T30 (examples)  →  independent
T31 (cleanup)  →  Final
```

### Agent Dispatch Summary

- **Wave 1**: 7 tasks — T1 → `quick`, T2 → `quick`, T3 → `quick`, T4 → `unspecified-high`, T5 → `deep`, T6 → `quick`, T7 → `quick`
- **Wave 2**: 8 tasks — T8 → `deep`, T9 → `deep`, T10 → `unspecified-high`, T11 → `unspecified-high`, T12 → `unspecified-high`, T13 → `unspecified-high`, T14 → `quick`, T15 → `quick`
- **Wave 3**: 8 tasks — T16 → `unspecified-high`, T17 → `unspecified-high`, T18 → `deep`, T19 → `ultrabrain`, T20 → `unspecified-high`, T21 → `deep`, T22 → `quick`, T23 → `writing`
- **Wave 4**: 8 tasks — T24 → `writing`, T25 → `writing`, T26 → `quick`, T27 → `unspecified-high`, T28 → `quick`, T29 → `unspecified-high`, T30 → `writing`, T31 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

### Wave 1 — Foundation (7 tasks, parallel)

- [x] 1. **Project scaffolding + git init + LICENSE + pyproject.toml**

  **What to do**:
  - Run `git init` in `/home/mlapointe/PyCharmMiscProject`
  - `git remote add origin git@github.com:cloudbsdorg/img2svg.git` (remote exists with commits — verify with `git ls-remote`)
  - `git fetch origin` and decide on merge strategy (likely `git reset --hard origin/main` if local has no project code)
  - Set local git author: `git config user.name "Mark LaPointe"` and `git config user.email "mark@cloudbsd.org"`
  - Create `LICENSE` (BSD 3-Clause, Copyright (c) 2026, CloudBSD)
  - Create `pyproject.toml` with:
    - Project name `img2svg`, version `0.1.0`
    - Python `>=3.10,<3.13`
    - BSD-3-Clause license classifier
- Dependencies: `ultralytics>=8.3,<9` (transitively pulls `torch`), `opencv-python>=4.10,<5`, `numpy>=1.26,<3`, `Pillow>=10.0,<12`, `vtracer>=0.6.15,<1`, `typer>=0.12,<1`, `rich>=13.0,<14`, `lxml>=5.0,<6`, `supervision>=0.28,<1`
- Explicitly pin: `torch>=2.0,<3` (used directly by T5 device detection; transitive via ultralytics but pin for clarity)
- Optional extras: `[ml]` (none — torch/ultralytics are core), `[dev]` (pytest, pytest-cov, pytest-mock, pytest-json-report, pytest-tap, pytest-junitxml, ruff, mypy, mkdocs, mkdocstrings)
    - Console script: `img2svg = img2svg.cli:app`
    - Package data: include `man/img2svg.1`, `locale/`
  - Create `.gitignore` (Python, venv, `.sisyphus/`, `__pycache__`, `.pytest_cache`, `*.egg-info`, `.coverage`, `htmlcov/`, `dist/`, `build/`)
  - Create `.python-version` (`3.10`)
  - Create empty `src/img2svg/__init__.py`, `tests/__init__.py`
  - Run `uv sync` and verify install works
  - Run `git add . && git commit -m "chore: scaffold img2svg project with BSD 3-Clause license"` and push (with user confirmation)

  **Must NOT do**:
  - Do not include YOLO model weights in repo
  - Do not include copyrighted test images
  - Do not include CI/CD publishing config in v1
  - Do not modify input files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Boilerplate scaffolding with well-defined outputs
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed for initial scaffold; can be added when handling actual commit operations

  **Parallelization**:
  - **Can Run In Parallel**: YES (this is T1, the first task; no dependencies)
  - **Parallel Group**: Wave 1
  - **Blocks**: All other tasks
  - **Blocked By**: None

  **References**:
  - **External**: BSD 3-Clause template: https://opensource.org/licenses/BSD-3-Clause
  - **External**: PEP 621 pyproject.toml spec: https://peps.python.org/pep-0621/
  - **Pattern**: Standard Python project layout (src/ layout)

  **Acceptance Criteria**:
  - [ ] `LICENSE` file present, contains "BSD 3-Clause" and "Copyright (c) 2026, CloudBSD"
  - [ ] `pyproject.toml` parses: `uv run python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"` exits 0
  - [ ] `uv sync` exits 0
  - [ ] `git remote -v` shows `origin git@github.com:cloudbsdorg/img2svg.git`
  - [ ] `git config user.name` returns "Mark LaPointe"
  - [ ] `git config user.email` returns "mark@cloudbsd.org"
  - [ ] `.gitignore` excludes `.sisyphus/`, `__pycache__/`, `.venv/`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`, `.coverage`, `htmlcov/`
  - [ ] Initial commit on `main` branch (or rebased onto remote `main`)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: License file has correct BSD 3-Clause text
    Tool: Bash
    Preconditions: T1 complete
    Steps:
      1. Read LICENSE file
      2. Assert "BSD 3-Clause" or "BSD-3-Clause" in content
      3. Assert "Copyright (c) 2026, CloudBSD" in content
      4. Assert presence of 3 standard clauses (no-endorsement, source form, binary form)
    Expected Result: All assertions pass
    Evidence: .sisyphus/evidence/task-1-license.txt

  Scenario: pyproject.toml is valid TOML
    Tool: Bash (uv run python)
    Preconditions: T1 complete
    Steps:
      1. uv run python -c "import tomllib; print(tomllib.loads(open('pyproject.toml').read()))"
      2. Assert exit code 0
      3. Assert output contains 'name', 'img2svg', 'version', '0.1.0'
    Expected Result: Parses successfully, shows img2svg metadata
    Evidence: .sisyphus/evidence/task-1-pyproject-valid.txt

  Scenario: uv sync installs cleanly
    Tool: Bash
    Preconditions: T1 complete
    Steps:
      1. rm -rf .venv uv.lock
      2. uv sync
      3. Assert exit code 0
      4. Assert .venv/bin/img2svg or .venv/bin/python exists
    Expected Result: Clean install, no errors
    Evidence: .sisyphus/evidence/task-1-uv-sync.txt
  ```

  **Commit**: YES
  - Message: `chore: scaffold img2svg project with BSD 3-Clause license`
  - Files: `LICENSE`, `pyproject.toml`, `.gitignore`, `.python-version`, `src/img2svg/__init__.py`, `tests/__init__.py`

---

- [x] 2. **XDG path module + config file loading**

  **What to do**:
  - Create `src/img2svg/paths.py` with functions:
    - `config_dir() -> Path` — `$XDG_CONFIG_HOME/img2svg/` (default `~/.config/img2svg/`)
    - `data_dir() -> Path` — `$XDG_DATA_HOME/img2svg/` (default `~/.local/share/img2svg/`)
    - `cache_dir() -> Path` — `$XDG_CACHE_HOME/img2svg/` (default `~/.cache/img2svg/`)
    - `system_config_dir() -> Path` — `/usr/local/etc/cloudbsd/img2svg/` (FreeBSD convention)
    - `model_cache_path() -> Path` — `$XDG_CACHE_HOME/img2svg/models/yolo11x.pt` etc.
    - `load_config() -> dict` — read `config.toml` from XDG config dir, return parsed dict
    - `save_config(config: dict) -> None` — write to XDG config dir
  - Create `src/img2svg/paths_test.py` (or `tests/test_paths.py`):
    - Test default paths when env vars unset
    - Test override when `XDG_CONFIG_HOME` set
    - Test creation of missing directories
    - Test `load_config` with valid + invalid TOML
    - Use `tmp_path` fixture to avoid touching real filesystem

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T19 (pipeline uses XDG paths for model cache)
  - **Blocked By**: T1

  **References**:
  - **External**: XDG Base Directory Specification: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
  - **External**: Python `os.path.expanduser`, `pathlib.Path.home()`

  **Acceptance Criteria**:
  - [ ] `from img2svg.paths import config_dir, cache_dir` works
  - [ ] `paths.config_dir()` returns `~/.config/img2svg/` when `XDG_CONFIG_HOME` unset
  - [ ] `paths.config_dir()` returns `$XDG_CONFIG_HOME/img2svg/` when env var set
  - [ ] `paths.system_config_dir()` returns `/usr/local/etc/cloudbsd/img2svg/`
  - [ ] `paths.cache_dir()` creates directory if missing (with `mkdir -p` semantics)

  **QA Scenarios**:
  ```
  Scenario: XDG paths respect environment variables
    Tool: Bash (uv run python -m pytest)
    Preconditions: T1, T2 complete
    Steps:
      1. XDG_CONFIG_HOME=/tmp/test_cfg uv run python -c "from img2svg.paths import config_dir; print(config_dir())"
      2. Assert output equals "/tmp/test_cfg/img2svg"
      3. XDG_CACHE_HOME=/tmp/test_cache uv run python -c "from img2svg.paths import cache_dir; print(cache_dir())"
      4. Assert output equals "/tmp/test_cache/img2svg"
    Expected Result: XDG env vars respected
    Evidence: .sisyphus/evidence/task-2-xdg-env.txt

  Scenario: Default paths use ~/.config/img2svg when XDG unset
    Tool: Bash
    Preconditions: T2 complete
    Steps:
      1. unset XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME
      2. uv run python -c "from img2svg.paths import config_dir, cache_dir, data_dir; print(config_dir(), cache_dir(), data_dir())"
      3. Assert output contains ".config/img2svg", ".cache/img2svg", ".local/share/img2svg"
    Expected Result: Defaults to ~/.config, ~/.cache, ~/.local/share
    Evidence: .sisyphus/evidence/task-2-xdg-defaults.txt

  Scenario: pytest test_paths.py passes
    Tool: Bash (pytest)
    Preconditions: T2 complete
    Steps:
      1. uv run pytest tests/test_paths.py -v
      2. Assert exit code 0
      3. Assert at least 5 tests collected
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-2-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(foundation): add XDG Base Directory paths and config loading`
  - Files: `src/img2svg/paths.py`, `tests/test_paths.py`

---

- [x] 3. **Typed exceptions + i18n/gettext scaffold**

  **What to do**:
  - Create `src/img2svg/errors.py` with exception hierarchy:
    - `Img2SvgError(Exception)` — base
    - `UnsupportedFormatError(Img2SvgError)` — file extension not in supported list
    - `CorruptImageError(Img2SvgError)` — Pillow can't decode
    - `ModelLoadError(Img2SvgError)` — YOLO model download/load fails
    - `DeviceUnavailableError(Img2SvgError)` — explicit `--device X` requested but not available
    - `OutputPathCollisionError(Img2SvgError)` — target exists and `--no-clobber` set
    - `VectorizationError(Img2SvgError)` — vtracer failed
    - Each has `.user_message()` and `.dev_message()` for layered output
  - Create `src/img2svg/i18n.py`:
    - `_()` function wrapping `gettext.gettext()`
    - `init_locale(lang: str | None = None)` — sets up gettext with `locale/img2svg.pot` template
    - `ngettext()` for plurals
    - Create `src/img2svg/locale/img2svg.pot` with initial English strings (msgid + msgstr)
  - Create tests:
    - `tests/test_errors.py` — assert each exception type is catchable as base
    - `tests/test_i18n.py` — assert `_(...)` returns the input string when locale is `C` (default)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: All other modules that raise exceptions or use i18n
  - **Blocked By**: T1

  **References**:
  - **External**: Python `gettext` docs: https://docs.python.org/3/library/gettext.html
  - **External**: GNU gettext: https://www.gnu.org/software/gettext/

  **Acceptance Criteria**:
  - [ ] `from img2svg.errors import UnsupportedFormatError` works
  - [ ] Each error class has `user_message()` and `dev_message()` methods
  - [ ] `from img2svg.i18n import _` works and returns the string in default locale

  **QA Scenarios**:
  ```
  Scenario: All exception classes import and are subclasses of Img2SvgError
    Tool: Bash (pytest)
    Preconditions: T1, T3 complete
    Steps:
      1. uv run pytest tests/test_errors.py -v
      2. Assert exit code 0
      3. Assert at least 7 tests collected (one per exception type)
    Expected Result: All exceptions work
    Evidence: .sisyphus/evidence/task-3-errors.txt

  Scenario: gettext returns input when no locale
    Tool: Bash (uv run python)
    Preconditions: T3 complete
    Steps:
      1. uv run python -c "from img2svg.i18n import _; print(_('Hello, world!'))"
      2. Assert output equals "Hello, world!"
    Expected Result: Identity in default locale
    Evidence: .sisyphus/evidence/task-3-i18n.txt
  ```

  **Commit**: YES
  - Message: `feat(foundation): add typed exceptions and gettext i18n scaffold`
  - Files: `src/img2svg/errors.py`, `src/img2svg/i18n.py`, `src/img2svg/locale/img2svg.pot`, `tests/test_errors.py`, `tests/test_i18n.py`

---

- [x] 4. **Pydantic data models (enums + core types)**

  **What to do**:
  - Create `src/img2svg/enums.py` with `StrEnum` types:
    - `class Mode(StrEnum)`: `AUTO = "auto"`, `LABELS = "labels"`, `VISUAL = "visual"`, `ANNOTATED = "annotated"`, `TRACE = "trace"`
    - `class ImageType(StrEnum)`: `LOGO`, `PHOTO`, `DIAGRAM`, `SCREENSHOT`, `LINE_ART`, `UNKNOWN`
    - `class DeviceStrategy(StrEnum)`: `AUTO = "auto"`, `POWER = "power"`, `AVAILABILITY = "availability"`
    - `class GpuVendor(StrEnum)`: `NVIDIA`, `AMD`, `APPLE`, `INTEL`, `UNKNOWN`
  - Create `src/img2svg/models.py` with Pydantic v2 BaseModel subclasses:
    - `class BoundingBox(BaseModel)`: `x1: float, y1: float, x2: float, y2: float`
    - `class Detection(BaseModel)`: `class_id: int, class_name: str, confidence: float, bbox: BoundingBox`
    - `class GeometricAnalysis(BaseModel)`: `dominant_colors: list[tuple[int,int,int]]`, `edge_density: float`, `contour_count: int`, `has_alpha: bool`
    - `class GPUInfo(BaseModel)`: `index: int, vendor: GpuVendor, name: str, vram_total_mb: int, vram_free_mb: int, compute_capability: str | None, utilization_pct: float | None`
    - `class ConversionOptions(BaseModel)`: `mode: Mode = Mode.AUTO`, `model: str = "yolo11x.pt"`, `device: str = "auto"`, `conf: float = 0.25`, `iou: float = 0.7`, `gpu_strategy: DeviceStrategy = DeviceStrategy.POWER`, `no_clobber: bool = False`, `force_overwrite: bool = False`, `palette_size: int = 8`
    - `class Sidecar(BaseModel)`: `version: str`, `input_path: Path`, `input_hash: str`, `output_path: Path`, `output_size: int`, `mode_used: Mode`, `mode_reasoning: str`, `model: str`, `device: str`, `image_type: ImageType`, `detections: list[Detection]`, `geometric: GeometricAnalysis | None`, `timings: dict[str, float]`, `timestamp: str`
    - `class ConversionResult(BaseModel)`: `svg_path: Path`, `sidecar_path: Path`, `sidecar: Sidecar`, `detections: list[Detection]`, `errors: list[str] = []`
  - Create `tests/test_models.py`:
    - Each model serializes + deserializes correctly (JSON round-trip)
    - Invalid types raise ValidationError
    - Defaults are applied
    - `Sidecar.model_validate_json(...)` works

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Pydantic models are foundational; correct serialization matters
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T8, T9, T14 (data flow), T19 (pipeline)
  - **Blocked By**: T1

  **References**:
  - **External**: Pydantic v2 docs: https://docs.pydantic.dev/2.10/
  - **External**: Python `StrEnum`: https://docs.python.org/3/library/enum.html#strenum

  **Acceptance Criteria**:
  - [ ] All enums importable
  - [ ] All Pydantic models importable
  - [ ] `Sidecar.model_validate_json(...)` works for a valid JSON
  - [ ] Invalid input raises `pydantic.ValidationError`

  **QA Scenarios**:
  ```
  Scenario: Pydantic models round-trip through JSON
    Tool: Bash (pytest)
    Preconditions: T1, T4 complete
    Steps:
      1. uv run pytest tests/test_models.py -v
      2. Assert exit code 0
      3. Assert at least 10 tests collected
    Expected Result: All models serialize/deserialize correctly
    Evidence: .sisyphus/evidence/task-4-models.txt

  Scenario: Sidecar schema validates a sample JSON
    Tool: Bash (uv run python)
    Preconditions: T4 complete
    Steps:
      1. uv run python -c "from img2svg.models import Sidecar, ConversionResult; from img2svg.enums import Mode, ImageType; r = ConversionResult(svg_path='/tmp/x.svg', sidecar_path='/tmp/x.json', sidecar=Sidecar(version='0.1.0', input_path='/tmp/x.png', input_hash='abc', output_path='/tmp/x.svg', output_size=1024, mode_used=Mode.LABELS, mode_reasoning='test', model='yolo11x.pt', device='cpu', image_type=ImageType.PHOTO, detections=[], geometric=None, timings={}, timestamp='2026-06-09T00:00:00'), detections=[]); print(r.model_dump_json())"
      2. Assert exit code 0
      3. Assert output is valid JSON
    Expected Result: Pydantic serialization works
    Evidence: .sisyphus/evidence/task-4-sidecar-serialize.txt
  ```

  **Commit**: YES
  - Message: `feat(foundation): add enums and Pydantic data models`
  - Files: `src/img2svg/enums.py`, `src/img2svg/models.py`, `tests/test_models.py`

---

- [x] 5. **Device detection + GPU enumeration/recommendation module**

  **What to do**:
  - Create `src/img2svg/device.py`:
    - `detect_device(requested: str = "auto") -> str`:
      - `auto`: try `torch.cuda.is_available()` (covers NVIDIA CUDA + AMD ROCm) → `torch.backends.mps.is_available()` (Apple) → `"cpu"`
      - `cpu`: always returns `"cpu"`
      - `cuda`, `cuda:N`: validate `torch.cuda.device_count() > N`, else raise `DeviceUnavailableError`
      - `mps`: validate `torch.backends.mps.is_available()`, else raise
      - `rocm`: alias for `cuda` (since PyTorch ROCm uses CUDA API)
    - `is_available(device: str) -> bool`: check without raising
  - Create `src/img2svg/gpu.py`:
    - `list_gpus() -> list[GPUInfo]`:
      - NVIDIA: parse `nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv` if available
      - AMD: parse `rocm-smi --showidname --showmeminfo` if available
      - Apple: enumerate via `torch.backends.mps` + system_profiler
      - Fallback: just `torch.cuda.device_count()` + `torch.cuda.get_device_properties(i)`
    - `recommend_gpu(gpus: list[GPUInfo], strategy: DeviceStrategy) -> GPUInfo | None`:
      - `power`: rank by VRAM total (descending)
      - `availability`: rank by VRAM free (descending)
    - `print_gpu_recommendation(strategy: DeviceStrategy = DeviceStrategy.POWER) -> None`: print Rich table
  - Create tests with mocked `torch`:
    - `test_device.py`: mock `torch.cuda.is_available()` returning True/False; assert correct device string
    - `test_device.py`: explicit `--device cuda` on no-CUDA system raises `DeviceUnavailableError`
    - `test_gpu.py`: mock `list_gpus()` returning 2 fake GPUs; assert `recommend_gpu(strategy=POWER)` picks higher-VRAM one
    - `test_gpu.py`: assert `recommend_gpu(strategy=AVAILABILITY)` picks higher-free-VRAM one

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Multi-vendor GPU detection is subtle; needs careful mock testing
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T8, T22
  - **Blocked By**: T1, T4

  **References**:
  - **External**: PyTorch CUDA semantics: https://pytorch.org/docs/stable/notes/cuda.html
  - **External**: PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html
  - **External**: AMD ROCm PyTorch: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html
  - **External**: `nvidia-smi` query fields: https://nvidia.custhelp.com/app/answers/detail/a_id/3751

  **Acceptance Criteria**:
  - [ ] `detect_device("auto")` returns "cuda" on NVIDIA systems, "mps" on Apple Silicon, "cpu" on CPU-only
  - [ ] `detect_device("cuda")` raises `DeviceUnavailableError` on CPU-only systems
  - [ ] `list_gpus()` returns at least 1 GPU on dgx (multi-vendor) systems
  - [ ] `recommend_gpu(strategy=POWER)` picks highest VRAM
  - [ ] `recommend_gpu(strategy=AVAILABILITY)` picks highest free VRAM

  **QA Scenarios**:
  ```
  Scenario: detect_device('auto') on CPU-only system returns 'cpu'
    Tool: Bash (uv run python with mocked torch)
    Preconditions: T1, T4, T5 complete
    Steps:
      1. uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False); from img2svg.device import detect_device; print('detected:', detect_device('auto'))"
      2. If CUDA False and MPS False: assert "detected: cpu"
      3. If CUDA True: assert "detected: cuda"
    Expected Result: Correct device string for current host
    Evidence: .sisyphus/evidence/task-5-detect-auto.txt

  Scenario: detect_device('cuda') on CPU-only system raises DeviceUnavailableError
    Tool: Bash (uv run python)
    Preconditions: T1, T4, T5 complete
    Steps:
      1. uv run python -c "from img2svg.device import detect_device; from img2svg.errors import DeviceUnavailableError
try:
    detect_device('cuda')
    print('NO ERROR (unexpected)')
except DeviceUnavailableError as e:
    print('OK:', e.user_message())
except Exception as e:
    print('WRONG ERROR:', type(e).__name__, e)"
      2. If CUDA False: assert "OK:" in output (or "WRONG ERROR" if torch not installed)
      3. If CUDA True: skip this scenario
    Expected Result: Raises DeviceUnavailableError on CPU-only systems
    Evidence: .sisyphus/evidence/task-5-detect-explicit.txt

  Scenario: pytest test_device.py + test_gpu.py pass with 80%+ coverage
    Tool: Bash (pytest)
    Preconditions: T1, T4, T5 complete
    Steps:
      1. uv run pytest tests/test_device.py tests/test_gpu.py -v --cov=img2svg.device --cov=img2svg.gpu
      2. Assert exit code 0
      3. Assert coverage >= 80%
    Expected Result: All tests pass with good coverage
    Evidence: .sisyphus/evidence/task-5-pytest-cov.txt
  ```

  **Commit**: YES
  - Message: `feat(foundation): add device detection and GPU enumeration/recommendation`
  - Files: `src/img2svg/device.py`, `src/img2svg/gpu.py`, `tests/test_device.py`, `tests/test_gpu.py`

---

- [x] 6. **Image loader (Pillow, format validation, alpha handling)**

  **What to do**:
  - Create `src/img2svg/loader.py`:
    - `SUPPORTED_FORMATS: frozenset[str] = {"PNG", "JPEG", "BMP", "WEBP", "TIFF", "GIF"}`
    - `load_image(path: Path) -> LoadedImage`:
      - Open with `PIL.Image.open(...)`
      - Verify format in `SUPPORTED_FORMATS`, else raise `UnsupportedFormatError`
      - Convert RGBA → preserve alpha (don't flatten)
      - For animated GIF: take first frame
      - For 16-bit PNG: convert to 8-bit
      - For CMYK: convert to RGB
      - Return `LoadedImage(pil_image, format, original_mode, has_alpha, width, height)`
    - `LoadedImage` dataclass: `pil_image: PIL.Image.Image`, `format: str`, `original_mode: str`, `has_alpha: bool`, `width: int`, `height: int`
  - Create `tests/test_loader.py`:
    - Load valid PNG, JPEG, BMP, WebP, TIFF, GIF → assert success
    - Load .ico → assert `UnsupportedFormatError`
    - Load corrupt bytes → assert `CorruptImageError`
    - Load RGBA → assert `has_alpha=True`
    - Load CMYK JPEG → assert converted to RGB
    - Load 16-bit PNG → assert converted to 8-bit
    - Load animated GIF → assert first frame only

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T10 (classifier), T19 (pipeline)
  - **Blocked By**: T1

  **References**:
  - **External**: Pillow formats: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
  - **External**: Pillow modes: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

  **Acceptance Criteria**:
  - [ ] `load_image(Path("test.png"))` returns `LoadedImage` for valid PNG
  - [ ] `load_image(Path("test.ico"))` raises `UnsupportedFormatError`
  - [ ] `load_image(Path("corrupt.png"))` raises `CorruptImageError`
  - [ ] RGBA image: `loaded.has_alpha == True`
  - [ ] CMYK image: converted to RGB before return

  **QA Scenarios**:
  ```
  Scenario: Load supported formats successfully
    Tool: Bash (uv run python -m pytest)
    Preconditions: T1, T6 complete
    Steps:
      1. uv run pytest tests/test_loader.py -v
      2. Assert exit code 0
      3. Assert at least 8 tests collected (one per format + error cases)
    Expected Result: All format loading works
    Evidence: .sisyphus/evidence/task-6-pytest.txt

  Scenario: Unsupported format raises clear error
    Tool: Bash (uv run python)
    Preconditions: T1, T6 complete
    Steps:
      1. echo "fake" > /tmp/fake.ico
      2. uv run python -c "from img2svg.loader import load_image; from img2svg.errors import UnsupportedFormatError
try:
    load_image('/tmp/fake.ico')
except UnsupportedFormatError as e:
    print('OK:', e.user_message())"
      3. Assert output contains "OK:" and ".ico"
    Expected Result: Clear error message naming the unsupported format
    Evidence: .sisyphus/evidence/task-6-unsupported.txt
  ```

  **Commit**: YES
  - Message: `feat(foundation): add image loader with format validation`
  - Files: `src/img2svg/loader.py`, `tests/test_loader.py`

---

- [x] 7. **Synthetic test image generator + conftest.py fixtures**

  **What to do**:
  - Create `scripts/gen_test_images.py`:
    - `generate_solid_color(path, size=(100, 100), color=(255, 0, 0))` — solid red square
    - `generate_gradient(path, size=(100, 100), start=(0,0,0), end=(255,255,255))` — diagonal gradient
    - `generate_logo(path, size=(200, 200))` — circle + rectangle (geometric, few colors, sharp edges)
    - `generate_diagram(path, size=(300, 300))` — black lines on white (line art)
    - `generate_photo_like(path, size=(256, 256))` — random noise with color regions (high color count)
    - `generate_transparent(path, size=(100, 100))` — RGBA with 50% transparent corners
    - `generate_corrupt_jpeg(path)` — write garbage bytes that Pillow rejects
  - Create `tests/conftest.py` with fixtures:
    - `logo_path` — fixture that returns `tests/fixtures/logo.png`, generating it if missing
    - `photo_path` — fixture for `tests/fixtures/photo.jpg`
    - `diagram_path` — fixture for `tests/fixtures/diagram.png`
    - `line_art_path` — fixture for `tests/fixtures/line_art.png`
    - `transparent_path` — fixture for `tests/fixtures/transparent.png`
    - `corrupt_path` — fixture for `tests/fixtures/corrupt.bin`
    - All fixtures use `tmp_path` to avoid filesystem pollution; commit pre-generated versions to `tests/fixtures/`
  - Run the generator script to commit actual test images:
    ```bash
    uv run python scripts/gen_test_images.py
    ```
  - Verify all fixtures are < 100KB each (total < 500KB)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T8, T9, T10, T11 (all need test fixtures), T27 (integration)
  - **Blocked By**: T1

  **References**:
  - **External**: pytest fixtures: https://docs.pytest.org/en/stable/explanation/fixtures.html
  - **External**: numpy random: https://numpy.org/doc/stable/reference/random/index.html

  **Acceptance Criteria**:
  - [ ] `scripts/gen_test_images.py` runs without error
  - [ ] `tests/fixtures/logo.png` exists and is a valid PNG
  - [ ] `tests/fixtures/photo.jpg` exists and is a valid JPEG
  - [ ] All 6 fixture files exist
  - [ ] Total fixture size < 1MB

  **QA Scenarios**:
  ```
  Scenario: gen_test_images.py produces all fixture files
    Tool: Bash (uv run python)
    Preconditions: T1, T7 complete
    Steps:
      1. rm -rf tests/fixtures/*
      2. uv run python scripts/gen_test_images.py
      3. Assert exit code 0
      4. ls tests/fixtures/ | wc -l: assert >= 6
      5. du -sh tests/fixtures/: assert < 1MB
    Expected Result: All fixtures generated, total size reasonable
    Evidence: .sisyphus/evidence/task-7-gen-fixtures.txt

  Scenario: pytest discovers conftest.py fixtures
    Tool: Bash (pytest)
    Preconditions: T7 complete
    Steps:
      1. uv run pytest --collect-only tests/
      2. Assert exit code 0
      3. Assert "logo_path" or "photo_path" in collected fixture names
    Expected Result: Fixtures discovered
    Evidence: .sisyphus/evidence/task-7-pytest-collect.txt
  ```

  **Commit**: YES
  - Message: `test(foundation): add synthetic test image generator and conftest fixtures`
  - Files: `scripts/gen_test_images.py`, `tests/conftest.py`, `tests/fixtures/*.png`, `tests/fixtures/*.jpg`, `tests/fixtures/corrupt.bin`

---

### Wave 2 — Core detection/analysis (8 tasks, parallel)

- [x] 8. **YOLO detector wrapper (ultralytics, model cache, device dispatch)**

  **What to do**:
  - Create `src/img2svg/detector.py`:
    - `class YOLODetector`:
      - `__init__(self, model_name: str = "yolo11x.pt", device: str = "auto", cache_dir: Path | None = None)`:
        - Resolves model path: `cache_dir` or `$XDG_CACHE_HOME/img2svg/models/`
        - Downloads model via `YOLO("yolo11x.pt")` (ultralytics auto-downloads)
        - Validates `device` via `device.detect_device()`
      - `detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> list[Detection]`:
        - Run YOLO inference
        - Extract `result.boxes.xyxy`, `.conf`, `.cls`
        - Map to `Detection` Pydantic models
        - Filter by `conf` threshold
    - Singleton factory: `get_detector(model: str, device: str) -> YOLODetector` (model is heavy, cache the instance)
  - Create `tests/test_detector.py`:
    - Mock `YOLO` class: assert `detect()` returns expected `list[Detection]`
    - Assert `Detection` has correct class_name from COCO names
    - Assert model path uses XDG cache when `cache_dir` not specified
    - Use `pytest-mock` to patch `ultralytics.YOLO`

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Heavy ML dependency; need careful mocking
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T19
  - **Blocked By**: T1, T4, T5, T7

  **References**:
  - **External**: ultralytics API: https://docs.ultralytics.com/usage/python/
  - **External**: YOLO result.boxes: https://docs.ultralytics.com/reference/engine/results/

  **Acceptance Criteria**:
  - [ ] `YOLODetector(model="yolo11n.pt", device="cpu")` instantiates without error
  - [ ] `detector.detect(image)` returns `list[Detection]` with correct class names
  - [ ] Model file is cached at `$XDG_CACHE_HOME/img2svg/models/yolo11n.pt` after first run

  **QA Scenarios**:
  ```
  Scenario: YOLO detector returns Detection objects
    Tool: Bash (pytest with mock)
    Preconditions: T1, T4, T5, T7, T8 complete
    Steps:
      1. uv run pytest tests/test_detector.py -v
      2. Assert exit code 0
      3. Assert at least 4 tests collected
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-8-pytest.txt

  Scenario: YOLO detects a "person" in a synthetic test image (integration, real model)
    Tool: Bash (uv run python with small model)
    Preconditions: T7, T8 complete; yolo11n.pt downloaded
    Steps:
      1. uv run python -c "from img2svg.detector import YOLODetector; from PIL import Image; import numpy as np; d = YOLODetector('yolo11n.pt', 'cpu'); img = np.array(Image.open('tests/fixtures/logo.png').convert('RGB')); print(d.detect(img, conf=0.1))"
      2. Assert exit code 0
      3. Assert output is a list (may be empty for the logo fixture, that's fine)
    Expected Result: Detector runs without error on real model
    Evidence: .sisyphus/evidence/task-8-real-inference.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add YOLO detector wrapper with model cache and device dispatch`
  - Files: `src/img2svg/detector.py`, `tests/test_detector.py`

---

- [x] 9. **OpenCV geometric patterns (K-means LAB, Canny, findContours, Hough)**

  **What to do**:
  - Create `src/img2svg/patterns.py`:
    - `analyze_global(image: np.ndarray) -> GeometricAnalysis`:
      - `has_alpha`: check image shape for 4 channels
      - `dominant_colors`: K-means in LAB with K=5, return RGB centroids
      - `edge_density`: Canny edges / total pixels
      - `contour_count`: `cv2.findContours` on thresholded image
    - `analyze_roi(image: np.ndarray, bbox: BoundingBox) -> GeometricAnalysis`:
      - Extract ROI, run same analysis on sub-image
      - Return per-ROI analysis
    - Use `opencv-python`; convert BGR↔RGB at boundaries
  - Create `tests/test_patterns.py`:
    - Test K-means returns 5 colors
    - Test edge_density in [0, 1]
    - Test has_alpha correctly detected
    - Test ROI analysis on a sub-region

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T16 (LabelsRenderer uses contour data), T18 (AnnotatedRenderer), T19
  - **Blocked By**: T1, T4, T7

  **References**:
  - **External**: OpenCV K-means: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
  - **External**: OpenCV Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
  - **External**: OpenCV findContours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

  **Acceptance Criteria**:
  - [ ] `analyze_global(image)` returns `GeometricAnalysis` with valid fields
  - [ ] `dominant_colors` has exactly 5 entries
  - [ ] `edge_density` in [0.0, 1.0]
  - [ ] `has_alpha` matches actual image channels

  **QA Scenarios**:
  ```
  Scenario: Geometric analysis on synthetic fixtures
    Tool: Bash (pytest)
    Preconditions: T1, T4, T7, T9 complete
    Steps:
      1. uv run pytest tests/test_patterns.py -v
      2. Assert exit code 0
      3. Assert at least 6 tests collected
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-9-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add OpenCV geometric pattern analysis`
  - Files: `src/img2svg/patterns.py`, `tests/test_patterns.py`

---

- [x] 10. **Image type classifier (heuristic on color/edge stats)**

  **What to do**:
  - Create `src/img2svg/classifier.py`:
    - `classify(image: np.ndarray, analysis: GeometricAnalysis) -> tuple[ImageType, str]`:
      - Returns `(image_type, reasoning_string)` for sidecar
      - Heuristic rules:
        - `has_alpha and len(dominant_colors) <= 3 and edge_density > 0.3` → `LOGO`
        - `len(dominant_colors) >= 6 and edge_density < 0.1` → `PHOTO`
        - `edge_density > 0.2 and not has_alpha` → `DIAGRAM`
        - `len(dominant_colors) <= 3 and edge_density > 0.4` → `LINE_ART`
        - else → `UNKNOWN`
  - Create `tests/test_classifier.py`:
    - Synthetic fixtures → expected types
    - Logo → `LOGO`
    - Photo (noise) → `PHOTO`
    - Diagram (lines) → `DIAGRAM`
    - Reasoning string is non-empty

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T12, T19
  - **Blocked By**: T1, T4, T6, T7, T9

  **Acceptance Criteria**:
  - [ ] `classify(logo_image, analysis)` returns `(ImageType.LOGO, "...")`
  - [ ] `classify(photo_image, analysis)` returns `(ImageType.PHOTO, "...")`
  - [ ] Reasoning string is at least 10 characters

  **QA Scenarios**:
  ```
  Scenario: Classifier identifies image types from fixtures
    Tool: Bash (pytest)
    Preconditions: T1, T4, T6, T7, T9, T10 complete
    Steps:
      1. uv run pytest tests/test_classifier.py -v
      2. Assert exit code 0
      3. Assert logo→LOGO, photo→PHOTO, diagram→DIAGRAM tests pass
    Expected Result: All image types classified correctly
    Evidence: .sisyphus/evidence/task-10-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add image type classifier with heuristic rules`
  - Files: `src/img2svg/classifier.py`, `tests/test_classifier.py`

---

- [x] 11. **vtracer vectorizer wrapper (preset configs)**

  **What to do**:
  - Create `src/img2svg/vectorizer.py`:
    - `class VtracerVectorizer`:
      - `__init__(self, preset: str = "default")`:
        - `default` = vtracer defaults (multi-color, spline, stacked)
        - `bw` = colormode=binary
        - `photo` = high color precision, smooth (for trace mode)
        - `logo` = sharp edges, high precision
        - `poster` = flat color regions
      - `vectorize(self, input_path: Path, output_path: Path) -> None`:
        - Call `vtracer.convert_image_to_svg_py(input_path, output_path, **params)`
        - Raise `VectorizationError` on failure
    - `PRESETS: dict[str, dict] = {...}` — preset name → vtracer params
  - Create `tests/test_vectorizer.py`:
    - Test each preset is a valid vtracer params dict
    - Mock `vtracer.convert_image_to_svg_py` and assert called with correct args
    - Test `VectorizationError` raised on mock failure

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T17, T19
  - **Blocked By**: T1, T4, T7

  **References**:
  - **External**: vtracer Python API: https://github.com/visioncortex/vtracer/blob/master/cmdapp/vtracer/vtracer.pyi

  **Acceptance Criteria**:
  - [ ] `VtracerVectorizer(preset="photo").vectorize(input, output)` runs without error
  - [ ] All 4 presets (default, bw, photo, logo) defined in `PRESETS`
  - [ ] Mock failure raises `VectorizationError`

  **QA Scenarios**:
  ```
  Scenario: vtracer vectorizer wrapper with mocked vtracer
    Tool: Bash (pytest)
    Preconditions: T1, T4, T7, T11 complete
    Steps:
      1. uv run pytest tests/test_vectorizer.py -v
      2. Assert exit code 0
      3. Assert at least 5 tests
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-11-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add vtracer vectorizer wrapper with preset configs`
  - Files: `src/img2svg/vectorizer.py`, `tests/test_vectorizer.py`

---

- [x] 12. **Image type → mode mapping (auto-mode selection)**

  **What to do**:
  - Create `src/img2svg/presets.py`:
    - `IMAGE_TYPE_TO_MODE: dict[ImageType, Mode] = {...}`:
      - `LOGO` → `Mode.LABELS`
      - `PHOTO` → `Mode.ANNOTATED`
      - `DIAGRAM` → `Mode.LABELS`
      - `SCREENSHOT` → `Mode.VISUAL`
      - `LINE_ART` → `Mode.LABELS`
      - `UNKNOWN` → `Mode.ANNOTATED`
    - `MODE_TO_PRESET: dict[Mode, str] = {...}`:
      - `LABELS` → `"logo"`
      - `VISUAL` → `"default"`
      - `ANNOTATED` → `"default"` (visual + overlays)
      - `TRACE` → `"photo"`
    - `select_mode(image_type: ImageType, requested: Mode = Mode.AUTO) -> tuple[Mode, str]`:
      - If `requested != Mode.AUTO`, return `(requested, "explicit override")`
      - Else: return `(IMAGE_TYPE_TO_MODE[image_type], f"auto: {image_type} → {mode}")`
  - Create `tests/test_presets.py`:
    - All `ImageType` values map to a `Mode`
    - All `Mode` values map to a preset (except AUTO)
    - Explicit override always wins

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T19
  - **Blocked By**: T1, T4, T10

  **Acceptance Criteria**:
  - [ ] `select_mode(ImageType.PHOTO, Mode.AUTO)` returns `(Mode.ANNOTATED, "...")`
  - [ ] `select_mode(ImageType.PHOTO, Mode.TRACE)` returns `(Mode.TRACE, "explicit override")`

  **QA Scenarios**:
  ```
  Scenario: Auto-mode selection and override
    Tool: Bash (pytest)
    Preconditions: T1, T4, T10, T12 complete
    Steps:
      1. uv run pytest tests/test_presets.py -v
      2. Assert exit code 0
      3. Assert explicit override test passes
    Expected Result: Auto-mode works, override works
    Evidence: .sisyphus/evidence/task-12-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add auto-mode selection with image type mapping`
  - Files: `src/img2svg/presets.py`, `tests/test_presets.py`

---

- [x] 13. **SVG builder (lxml, namespace handling, accessibility)**

  **What to do**:
  - Create `src/img2svg/svg_builder.py`:
    - `SVG_NS = "http://www.w3.org/2000/svg"`
    - `class SVGDocument`:
      - `__init__(self, width: int, height: int, title: str | None = None, desc: str | None = None)`
      - `add_group(self, id: str, class_: str | None = None, **attrs) -> SVGGroup`
      - `add_rect(self, x, y, w, h, fill, stroke, ...) -> etree.Element`
      - `add_path(self, d: str, fill: str | None, ...) -> etree.Element`
      - `add_text(self, x, y, text: str, font_size: int, fill: str, ...) -> etree.Element`
      - `to_string(self, pretty: bool = True) -> str` — serialize to SVG XML
      - `write(self, path: Path) -> None`
      - Auto-include `<title>` and `<desc>` for accessibility (WCAG)
    - `class SVGGroup`: helper for `<g>` with attributes
  - Create `tests/test_svg_builder.py`:
    - Build minimal SVG, parse with lxml, assert root is `<svg>` with correct `viewBox`
    - Add group, assert it's nested correctly
    - Include title, assert it's the first child

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T16, T17, T18
  - **Blocked By**: T1, T4

  **References**:
  - **External**: lxml.etree: https://lxml.de/tutorial.html
  - **External**: SVG accessibility: https://www.w3.org/TR/SVG-access/

  **Acceptance Criteria**:
  - [ ] `SVGDocument(100, 100, title="foo")` creates document with title
  - [ ] `doc.to_string()` returns valid XML parseable by lxml
  - [ ] Output has `xmlns="http://www.w3.org/2000/svg"`

  **QA Scenarios**:
  ```
  Scenario: SVG document serializes to valid XML
    Tool: Bash (pytest)
    Preconditions: T1, T4, T13 complete
    Steps:
      1. uv run pytest tests/test_svg_builder.py -v
      2. Assert exit code 0
      3. Assert at least 5 tests
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-13-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add SVG builder with lxml and accessibility`
  - Files: `src/img2svg/svg_builder.py`, `tests/test_svg_builder.py`

---

- [x] 14. **JSON sidecar writer (Pydantic serialization)**

  **What to do**:
  - Create `src/img2svg/metadata.py`:
    - `write_sidecar(sidecar: Sidecar, path: Path) -> None`:
      - Serialize via `sidecar.model_dump_json(indent=2, exclude_none=False)`
      - Write atomically (write to `path.tmp`, then rename) to avoid half-written files
    - `read_sidecar(path: Path) -> Sidecar`:
      - `Sidecar.model_validate_json(path.read_text())`
    - `compute_file_hash(path: Path) -> str` — SHA-256 of file contents
  - Create `tests/test_metadata.py`:
    - Round-trip: write → read → assert equal
    - Atomic write: kill mid-write, assert tmp file cleaned up
    - Hash matches known SHA-256 of fixture

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T19
  - **Blocked By**: T1, T4

  **Acceptance Criteria**:
  - [ ] `write_sidecar(sidecar, path)` produces valid JSON
  - [ ] `read_sidecar(path)` returns the same `Sidecar` instance
  - [ ] Atomic write: no `.tmp` files left after success

  **QA Scenarios**:
  ```
  Scenario: Sidecar round-trip
    Tool: Bash (pytest)
    Preconditions: T1, T4, T14 complete
    Steps:
      1. uv run pytest tests/test_metadata.py -v
      2. Assert exit code 0
      3. Assert at least 4 tests
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-14-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add JSON sidecar writer with atomic writes`
  - Files: `src/img2svg/metadata.py`, `tests/test_metadata.py`

---

- [x] 15. **Rich logging + progress bars)**

  **What to do**:
  - Create `src/img2svg/logging.py`:
    - `setup_logging(verbose: bool = False, quiet: bool = False) -> None`:
      - Default: INFO level, Rich console handler
      - `-v`: DEBUG level
      - `-q`: WARNING level only
    - `get_logger(name: str) -> logging.Logger`:
      - Returns logger with Rich handler
    - `ProgressSpinner` context manager for indeterminate operations (model download)
    - `ProgressBar` for batch operations (1 progress bar, N items)
  - Create `tests/test_logging.py`:
    - `setup_logging(verbose=True)` sets DEBUG level
    - `setup_logging(quiet=True)` sets WARNING level
    - Logger output goes to stderr (not stdout) so piping works

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T19, T21, T22
  - **Blocked By**: T1

  **References**:
  - **External**: Rich logging: https://rich.readthedocs.io/en/stable/logging.html
  - **External**: Rich progress: https://rich.readthedocs.io/en/stable/progress.html

  **Acceptance Criteria**:
  - [ ] `setup_logging(verbose=True)` → logger.level == DEBUG
  - [ ] `setup_logging(quiet=True)` → logger.level == WARNING
  - [ ] Default setup → logger.level == INFO

  **QA Scenarios**:
  ```
  Scenario: Logging level configuration
    Tool: Bash (pytest)
    Preconditions: T1, T15 complete
    Steps:
      1. uv run pytest tests/test_logging.py -v
      2. Assert exit code 0
      3. Assert at least 3 tests (one per verbosity level)
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-15-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add Rich logging with verbosity flags`
  - Files: `src/img2svg/logging.py`, `tests/test_logging.py`

---

### Wave 3 — Renderers + pipeline + CLI (8 tasks, parallel)

- [x] 16. **Renderer base class + LabelsRenderer**

  **What to do**:
  - Create `src/img2svg/renderers/base.py`:
    - `class Renderer(ABC)`:
      - `__init__(self, svg: SVGDocument, image: LoadedImage, detections: list[Detection], geometric: GeometricAnalysis | None)`
      - `@abstractmethod render(self) -> None`
  - Create `src/img2svg/renderers/labels.py`:
    - `class LabelsRenderer(Renderer)`:
      - `render()`: for each `Detection`, add a `<g id="det_class_idx" data-class="..." data-conf="...">` with:
        - `<rect>` for bounding box (stroke="black", fill="none")
        - `<text>` with class name and confidence above the box
        - White outline on text for readability
      - Add background `<rect>` for the full image
      - NO vtracer output (this mode is pure semantic)
  - Create `tests/test_renderers/test_labels.py`:
    - Empty detections list → SVG with just background
    - 1 detection → 1 `<g>` group with rect + text
    - 3 detections of same class → 3 separate groups
    - Group IDs follow pattern `det_{class_name}_{idx}`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T19
  - **Blocked By**: T1, T4, T13

  **Acceptance Criteria**:
  - [ ] `LabelsRenderer(svg, image, [det1, det2], geom).render()` produces SVG with 2 detection groups
  - [ ] Each group has `data-class` and `data-conf` attributes
  - [ ] Empty detections still produces valid SVG (just background)

  **QA Scenarios**:
  ```
  Scenario: LabelsRenderer produces semantic SVG
    Tool: Bash (pytest)
    Preconditions: T1, T4, T13, T16 complete
    Steps:
      1. uv run pytest tests/test_renderers/test_labels.py -v
      2. Assert exit code 0
      3. Assert at least 4 tests
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-16-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(renderers): add Renderer base class and LabelsRenderer`
  - Files: `src/img2svg/renderers/__init__.py`, `src/img2svg/renderers/base.py`, `src/img2svg/renderers/labels.py`, `tests/test_renderers/test_labels.py`

---

- [x] 17. **VisualRenderer + TraceRenderer (vtracer-based)**

  **What to do**:
  - Create `src/img2svg/renderers/visual.py`:
    - `class VisualRenderer(Renderer)`:
      - `render()`: call `VtracerVectorizer(preset="default").vectorize(image.path, temp_path)`
      - Read the vtracer output, parse with lxml, extract the `<g>` elements
      - Insert them as a single background `<g id="vtracer-output">` in our SVG
  - Create `src/img2svg/renderers/trace.py`:
    - `class TraceRenderer(Renderer)`:
      - `render()`: same as VisualRenderer but with `preset="photo"`
  - Create `tests/test_renderers/test_visual.py` + `test_trace.py`:
    - Mock vtracer; assert correct preset is used
    - Assert SVG contains `<g id="vtracer-output">`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T19
  - **Blocked By**: T1, T4, T11, T13

  **Acceptance Criteria**:
  - [ ] `VisualRenderer(...).render()` produces SVG with vtracer paths inside
  - [ ] `TraceRenderer(...).render()` uses photo preset
  - [ ] vtracer preset is correctly selected per mode

  **QA Scenarios**:
  ```
  Scenario: Visual/Trace renderers use correct vtracer preset
    Tool: Bash (pytest)
    Preconditions: T1, T4, T11, T13, T17 complete
    Steps:
      1. uv run pytest tests/test_renderers/test_visual.py tests/test_renderers/test_trace.py -v
      2. Assert exit code 0
      3. Assert at least 4 tests
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-17-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(renderers): add VisualRenderer and TraceRenderer with vtracer presets`
  - Files: `src/img2svg/renderers/visual.py`, `src/img2svg/renderers/trace.py`, `tests/test_renderers/test_visual.py`, `tests/test_renderers/test_trace.py`

---

- [x] 18. **AnnotatedRenderer (visual + bounding boxes)**

  **What to do**:
  - Create `src/img2svg/renderers/annotated.py`:
    - `class AnnotatedRenderer(Renderer)`:
      - `render()`: combines VisualRenderer + LabelsRenderer output
      - Background: vtracer output (default preset)
      - Foreground: detection bounding boxes + labels with white text outline
  - Create `tests/test_renderers/test_annotated.py`:
    - Assert SVG contains BOTH vtracer group AND detection groups
    - Assert bounding boxes are drawn on top of (after) the vtracer paths

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Composes two renderers; needs careful z-order
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T19
  - **Blocked By**: T1, T4, T13, T16, T17

  **Acceptance Criteria**:
  - [ ] `AnnotatedRenderer(...).render()` produces SVG with both vtracer paths and detection overlays
  - [ ] Detection overlays appear after vtracer paths in the SVG (z-order)

  **QA Scenarios**:
  ```
  Scenario: AnnotatedRenderer combines visual + labels
    Tool: Bash (pytest)
    Preconditions: T1, T4, T13, T16, T17, T18 complete
    Steps:
      1. uv run pytest tests/test_renderers/test_annotated.py -v
      2. Assert exit code 0
      3. Assert at least 3 tests
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-18-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(renderers): add AnnotatedRenderer combining visual + labels`
  - Files: `src/img2svg/renderers/annotated.py`, `tests/test_renderers/test_annotated.py`

---

- [x] 19. **Pipeline orchestrator (load → classify → detect → analyze → render → write)**

  **What to do**:
  - Create `src/img2svg/pipeline.py`:
    - `class Pipeline`:
      - `__init__(self, options: ConversionOptions)`:
        - Stores options
        - Lazy-loads YOLODetector (singleton) on first `.run()`
      - `def run(self, input_path: Path, output_path: Path) -> ConversionResult`:
        1. `loaded = load_image(input_path)` (raises on unsupported/corrupt)
        2. `analysis_global = analyze_global(loaded.np_array)`
        3. `image_type, reasoning = classify(loaded.np_array, analysis_global)`
        4. `mode_used, mode_reasoning = select_mode(image_type, options.mode)`
        5. `detections = detector.detect(loaded.np_array, conf=options.conf, iou=options.iou)`
        6. `analysis_roi = [analyze_roi(loaded.np_array, d.bbox) for d in detections]`
        7. `svg = SVGDocument(loaded.width, loaded.height, title=input_path.name)`
        8. `renderer = RENDERER_REGISTRY[mode_used](svg, loaded, detections, analysis_global); renderer.render()`
        9. `svg.write(output_path)`
        10. Build `Sidecar` with all metadata
        11. `sidecar_path = output_path.with_suffix(".json"); write_sidecar(sidecar, sidecar_path)`
        12. Return `ConversionResult(svg_path=output_path, sidecar_path=sidecar_path, sidecar=sidecar, detections=detections)`
  - Create `src/img2svg/api.py`:
    - `convert(input_path: str | Path, output_path: str | Path, *, options: ConversionOptions | None = None, **kwargs) -> ConversionResult`:
      - High-level public API
      - Builds `ConversionOptions` from kwargs if not provided
      - Calls `Pipeline(options).run(input_path, output_path)`
    - `convert_batch(inputs: list[str | Path] | str, output_dir: str | Path | None = None, *, options: ConversionOptions | None = None, **kwargs) -> list[ConversionResult]`:
      - If `inputs` is a string, glob-expand it
      - If `output_dir` not provided, use `inputs[0].parent`
      - Process each input, return list of results
  - `src/img2svg/__init__.py`:
    - Re-export `convert`, `convert_batch`, `ConversionOptions`, `ConversionResult`, `Sidecar`, `Mode`, `ImageType`, `DeviceStrategy`
  - Create `tests/test_pipeline.py`:
    - Full pipeline run on a fixture → assert SVG + JSON created
    - `convert()` public API works
    - `convert_batch()` processes multiple files
  - Create `tests/test_api.py`:
    - `from img2svg import convert` works
    - `convert(fixture_png, /tmp/out.svg)` returns `ConversionResult`
    - `convert_batch([fixture1, fixture2])` returns list

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Orchestrates all modules; the heart of the library
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T2, T3, T4, T5, T6, T8-T18)
  - **Parallel Group**: Wave 3 (sequential after T8-T18)
  - **Blocks**: T20, T21
  - **Blocked By**: T2, T3, T4, T5, T6, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18

  **Acceptance Criteria**:
  - [ ] `convert(input, output)` returns a valid `ConversionResult`
  - [ ] `output` file exists and is valid SVG
  - [ ] `output.with_suffix(".json")` exists and validates as `Sidecar`
  - [ ] `from img2svg import convert, convert_batch, ConversionOptions` works

  **QA Scenarios**:
  ```
  Scenario: Full pipeline produces SVG + sidecar from fixture
    Tool: Bash (pytest + uv run python)
    Preconditions: T1-T18 complete
    Steps:
      1. uv run pytest tests/test_pipeline.py tests/test_api.py -v
      2. Assert exit code 0
      3. Assert at least 6 tests
      4. End-to-end: uv run python -c "from img2svg import convert; r = convert('tests/fixtures/logo.png', '/tmp/out.svg'); assert r.svg_path.exists(); assert r.sidecar_path.exists()"
    Expected Result: Pipeline works end-to-end
    Evidence: .sisyphus/evidence/task-19-pipeline.txt
  ```

  **Commit**: YES
  - Message: `feat(pipeline): add pipeline orchestrator and high-level API`
  - Files: `src/img2svg/pipeline.py`, `src/img2svg/api.py`, `src/img2svg/__init__.py`, `tests/test_pipeline.py`, `tests/test_api.py`

---

- [x] 20. **Batch processing (multi-file, directory, glob, error continuation)**

  **What to do**:
  - Enhance `src/img2svg/api.py` `convert_batch()`:
    - Accept `inputs` as: list of paths, single path (single file), or string (glob pattern, or directory)
    - If `inputs` is a directory: walk it (recursive or not based on flag), filter to supported formats, sort
    - If `inputs` is a glob: use `pathlib.Path.glob()`
    - If `inputs` is a single file: process it
    - Use `rich.progress.Progress` to show per-file progress
    - On individual file failure: log error, continue to next file, return error in `ConversionResult.errors`
    - Return `list[ConversionResult]`
  - Create `tests/test_batch.py`:
    - Batch of 3 valid files → 3 results, all successful
    - 1 corrupt + 2 valid → 2 successful, 1 in errors
    - Directory input → all supported files processed
    - Glob input → matching files processed
    - Progress bar shown (mock Rich and assert `Progress` is called)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (after T19)
  - **Parallel Group**: Wave 3
  - **Blocks**: T21
  - **Blocked By**: T19

  **Acceptance Criteria**:
  - [ ] `convert_batch("./fixtures/*.png")` processes all matching files
  - [ ] `convert_batch("./fixtures/")` processes all supported files in directory
  - [ ] Corrupt file doesn't kill batch; other files still processed
  - [ ] Final summary printed: "X succeeded, Y failed"

  **QA Scenarios**:
  ```
  Scenario: Batch processes multiple files and handles errors
    Tool: Bash (pytest)
    Preconditions: T19, T20 complete
    Steps:
      1. uv run pytest tests/test_batch.py -v
      2. Assert exit code 0
      3. Assert at least 5 tests including error continuation
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-20-pytest.txt
  ```

  **Commit**: YES
  - Message: `feat(batch): add batch processing with error continuation and progress`
  - Files: `src/img2svg/api.py` (modified), `tests/test_batch.py`

---

- [x] 21. **Typer CLI (subcommands, --list-gpus, --gpu-strategy, --mode, --device, --model)**

  **What to do**:
  - Create `src/img2svg/cli.py`:
    - `app = typer.Typer(help="Convert raster images to SVG with object and pattern detection.")`
    - Main command `convert`:
      - `input: Path = typer.Argument(..., help="Input file, glob, or directory")`
      - `-o, --output: Path | None = typer.Option(None, help="Output file (or directory for batch)")`
      - `--mode: Mode = typer.Option(Mode.AUTO, help="Output mode: auto, labels, visual, annotated, trace")`
      - `--model: str = typer.Option("yolo11x.pt", help="YOLO model name")`
      - `--device: str = typer.Option("auto", help="Device: auto, cpu, cuda, cuda:N, mps, rocm")`
      - `--gpu-strategy: DeviceStrategy = typer.Option(DeviceStrategy.POWER, help="GPU recommendation strategy")`
      - `--conf: float = typer.Option(0.25, help="YOLO confidence threshold")`
      - `--no-clobber: bool = typer.Option(False, help="Don't overwrite existing output files")`
      - `-q, --quiet: bool = typer.Option(False, help="Suppress non-essential output")`
      - `-v, --verbose: bool = typer.Option(False, help="Enable debug output")`
      - `--version: bool = typer.Option(False, help="Print version and exit")`
    - Subcommand `list-gpus`:
      - `--strategy: DeviceStrategy = typer.Option(DeviceStrategy.POWER)`
      - Calls `print_gpu_recommendation(strategy)`
      - Exits 0
    - Subcommand `info`:
      - Print img2svg version, Python version, OS (`uname -s`), detected devices
  - Exit code mapping (from errors.py):
    - Success → 0
    - Partial fail (some files failed) → 1
    - Invalid args / unsupported format → 2
    - Dependency / model load failure → 3
  - `src/img2svg/__main__.py`: `from img2svg.cli import app; app()`
  - Create `tests/test_cli.py`:
    - `typer.testing.CliRunner` invokes `app`
    - `img2svg --help` exits 0, lists all flags
    - `img2svg --version` prints version, exits 0
    - `img2svg list-gpus` exits 0, prints table
    - `img2svg fixtures/logo.png -o /tmp/out.svg` exits 0
    - `img2svg nonexistent.png` exits 2 (invalid args)
    - `img2svg fixtures/logo.png -o /tmp/out.svg --mode bogus` exits 2 (invalid mode)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T19, T20)
  - **Parallel Group**: Wave 3 (sequential after T19, T20)
  - **Blocks**: T28
  - **Blocked By**: T19, T20

  **References**:
  - **External**: Typer docs: https://typer.tiangolo.com/
  - **External**: typer.testing.CliRunner: https://typer.tiangolo.com/tutorial/testing/

  **Acceptance Criteria**:
  - [ ] `img2svg --help` exits 0 and shows all flags
  - [ ] `img2svg list-gpus` exits 0 and shows GPU table
  - [ ] `img2svg foo.png -o out.svg` exits 0 on success
  - [ ] `img2svg foo.png --mode bogus` exits 2 with clear error
  - [ ] Exit code 0 on success, 2 on invalid args, 1 on partial fail

  **QA Scenarios**:
  ```
  Scenario: CLI smoke tests
    Tool: Bash (pytest)
    Preconditions: T1-T20, T21 complete
    Steps:
      1. uv run pytest tests/test_cli.py -v
      2. Assert exit code 0
      3. Assert at least 8 tests covering all subcommands and exit codes
    Expected Result: All CLI tests pass
    Evidence: .sisyphus/evidence/task-21-pytest.txt

  Scenario: img2svg --help shows all flags
    Tool: Bash (uv run)
    Preconditions: T21 complete
    Steps:
      1. uv run img2svg --help
      2. Assert exit code 0
      3. Assert output contains "--mode", "--device", "--model", "--list-gpus", "--gpu-strategy"
    Expected Result: All flags documented in --help
    Evidence: .sisyphus/evidence/task-21-help.txt
  ```

  **Commit**: YES
  - Message: `feat(cli): add Typer CLI with subcommands and exit code mapping`
  - Files: `src/img2svg/cli.py`, `src/img2svg/__main__.py`, `tests/test_cli.py`

---

- [x] 22. **GPU recommendation CLI command + Rich table**

  **What to do**:
  - Enhance `src/img2svg/gpu.py` with `print_gpu_recommendation(strategy)`:
    - Use `rich.table.Table` to show: index, vendor, name, VRAM total, VRAM free, util%, recommended
    - Highlight the recommended row (bold + accent color)
    - Print OS info (from `uname -s`)
    - Print PyTorch version
  - Wire into `list-gpus` subcommand of CLI
  - Create `tests/test_gpu_recommend.py`:
    - Mock `list_gpus()` to return 2 fake GPUs (NVIDIA + AMD)
    - Call `print_gpu_recommendation(POWER)` → assert both GPUs shown in table
    - Call `print_gpu_recommendation(AVAILABILITY)` → assert recommended GPU differs when free VRAM differs

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T21, which provides the `list-gpus` subcommand)
  - **Parallel Group**: Wave 3 (sequential after T21)
  - **Blocks**: None
  - **Blocked By**: T1, T5, T21

  **Acceptance Criteria**:
  - [ ] `img2svg list-gpus` shows Rich table with all detected GPUs
  - [ ] Recommended GPU is highlighted
  - [ ] `--gpu-strategy power` ranks by VRAM total
  - [ ] `--gpu-strategy availability` ranks by VRAM free

  **QA Scenarios**:
  ```
  Scenario: list-gpus command shows GPU table
    Tool: Bash (pytest + uv run)
    Preconditions: T1, T5, T22 complete
    Steps:
      1. uv run pytest tests/test_gpu_recommend.py -v
      2. Assert exit code 0
      3. uv run img2svg list-gpus (capture stdout)
      4. Assert output contains at least one GPU name (if system has GPU) or "No GPU detected" message
    Expected Result: Table renders correctly
    Evidence: .sisyphus/evidence/task-22-list-gpus.txt
  ```

  **Commit**: YES
  - Message: `feat(gpu): add GPU recommendation command with Rich table`
  - Files: `src/img2svg/gpu.py` (modified), `tests/test_gpu_recommend.py`

---

- [x] 23. **Man page (man/img2svg.1)**

  **What to do**:
  - Create `man/img2svg.1` in standard roff/groff man page format
  - Sections:
    - `.TH IMG2SVG 1 "2026-06-09" "0.1.0" "User Commands"` (title header)
    - `.SH NAME` — `img2svg \- convert raster images to SVG with object and pattern detection`
    - `.SH SYNOPSIS` — `img2svg [OPTIONS] INPUT [-o OUTPUT]`
    - `.SH DESCRIPTION` — overview, supported formats, output modes
    - `.SH OPTIONS` — every flag with description
    - `.SH EXAMPLES` — 5+ examples (basic, batch, GPU selection, mode selection)
    - `.SH OUTPUT MODES` — labels, visual, annotated, trace
    - `.SH GPU SUPPORT` — autodetect, multi-vendor (NVIDIA + AMD), override
    - `.SH EXIT STATUS` — 0/1/2/3
    - `.SH FILES` — XDG paths: `~/.config/img2svg/`, `~/.local/share/img2svg/`, `~/.cache/img2svg/models/`
    - `.SH ENVIRONMENT` — `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `CUDA_VISIBLE_DEVICES`
    - `.SH SEE ALSO` — references to vtracer, ultralytics
    - `.SH AUTHOR` — `Mark LaPointe <mark@cloudbsd.org>`
    - `.SH BUGS` — bug report URL (GitHub issues)
  - Add `man/img2svg.1` to `pyproject.toml` `package-data` so it installs with the package
  - Create `scripts/install_manpage.sh` (optional convenience):
    - `cp man/img2svg.1 /usr/local/share/man/man1/` (FreeBSD) or `/usr/local/share/man/man1/` (Linux)
    - `mandb`
  - Create `tests/test_manpage.py`:
    - File exists
    - Contains all required sections (NAME, SYNOPSIS, DESCRIPTION, OPTIONS, EXAMPLES, EXIT STATUS)
    - `man --warnings -l man/img2svg.1` (if `man` available) renders without warnings

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T28 (packaging)
  - **Blocked By**: T1, T21

  **References**:
  - **External**: man page format: https://man7.org/linux/man-pages/man7/man-pages.7.html
  - **External**: groff man macros: https://man7.org/linux/man-pages/man7/groff_man.7.html

  **Acceptance Criteria**:
  - [ ] `man/img2svg.1` exists
  - [ ] Contains all required sections
  - [ ] `man -l man/img2svg.1` renders (test only on systems with `man`)

  **QA Scenarios**:
  ```
  Scenario: Man page contains all required sections
    Tool: Bash (pytest)
    Preconditions: T1, T21, T23 complete
    Steps:
      1. uv run pytest tests/test_manpage.py -v
      2. Assert exit code 0
      3. Assert man file contains "SH NAME", "SH SYNOPSIS", "SH DESCRIPTION", "SH OPTIONS", "SH EXAMPLES", "SH EXIT STATUS", "SH AUTHOR"
    Expected Result: Man page valid
    Evidence: .sisyphus/evidence/task-23-man.txt
  ```

  **Commit**: YES
  - Message: `docs: add man page (man/img2svg.1)`
  - Files: `man/img2svg.1`, `scripts/install_manpage.sh`, `tests/test_manpage.py`, `pyproject.toml` (updated package-data)

---

### Wave 4 — Docs, CI, integration, packaging (8 tasks, parallel)

- [x] 24. **README.md (overview, install, quickstart, Mermaid arch diagram)**

  **What to do**:
  - Create `README.md` with:
    - Title + tagline
    - Badges (optional): license (BSD-3-Clause), Python versions, CI status (placeholder)
    - Overview: what `img2svg` does (1 paragraph)
    - Features list (bullets)
    - Quickstart: 3 commands (install, run, view output)
    - Installation section: `pip install img2svg`, `uv add img2svg`, source install
    - Usage examples: 5+ CLI examples with code blocks
    - Python API example: `from img2svg import convert`
    - Mermaid architecture diagram (per CloudBSD guidelines):
      ```mermaid
      flowchart LR
        A[Image Input] --> B[Loader]
        B --> C[Classifier]
        C --> D[YOLO Detector]
        C --> E[Geometric Analysis]
        D --> F[Pipeline]
        E --> F
        F --> G[Renderer]
        G --> H[SVG Output]
        F --> I[Sidecar JSON]
      ```
    - GPU support section: autodetect, multi-vendor, `--list-gpus`
    - Output modes section
    - Configuration (XDG paths)
    - Platform support matrix
    - Development section: `uv sync --all-extras`, `pytest`, `ruff`
    - License: BSD 3-Clause
    - Author: Mark LaPointe <mark@cloudbsd.org>
  - Create `tests/test_readme.py`:
    - File exists
    - Contains key sections (install, usage, license, author)
    - Contains Mermaid code block

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T28
  - **Blocked By**: T1, T19, T21, T23

  **Acceptance Criteria**:
  - [ ] `README.md` exists
  - [ ] Contains Mermaid diagram
  - [ ] Contains license info
  - [ ] Contains author info

  **QA Scenarios**:
  ```
  Scenario: README has all required sections
    Tool: Bash (pytest + grep)
    Preconditions: T1, T19, T21, T23, T24 complete
    Steps:
      1. uv run pytest tests/test_readme.py -v
      2. Assert exit code 0
      3. grep -E "^# |^## " README.md (assert "Installation", "Usage", "License", "Author" present)
    Expected Result: README valid
    Evidence: .sisyphus/evidence/task-24-readme.txt
  ```

  **Commit**: YES
  - Message: `docs: add README with Mermaid architecture diagram`
  - Files: `README.md`, `tests/test_readme.py`

---

- [x] 25. **docs/ — installation, usage, api, modes, gpu, configuration, troubleshooting, development, architecture, changelog**

  **What to do**:
  - Create `docs/`:
    - `index.md` — landing page, links to all other docs
    - `installation.md` — install from PyPI, source, FreeBSD notes, macOS notes
    - `usage.md` — basic + advanced CLI usage, batch, GPU selection
    - `api.md` — Python API reference (auto-generate from docstrings using mkdocstrings)
    - `modes.md` — output modes reference with examples
    - `gpu.md` — GPU setup for NVIDIA/AMD/Apple Silicon
    - `configuration.md` — XDG config, `img2svg.toml` (if implemented)
    - `troubleshooting.md` — common errors and solutions
    - `development.md` — dev setup, TDD workflow, contributing
    - `architecture.md` — Mermaid diagrams of pipeline, renderers, GPU dispatch
    - `changelog.md` — version history
  - Create `docs/mkdocs.yml`:
    - Site name: img2svg
    - Theme: material (or readthedocs)
    - nav: list of all docs
    - plugins: mkdocstrings (for api.md auto-generation)
  - Run `uv run mkdocs build --strict` to verify it builds without warnings
  - Create `tests/test_docs.py`:
    - All required docs files exist
    - `mkdocs build --strict` exits 0
    - `index.md` links to all other docs

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T28
  - **Blocked By**: T1, T19, T21

  **References**:
  - **External**: MkDocs: https://www.mkdocs.org/
  - **External**: mkdocstrings: https://mkdocstrings.github.io/

  **Acceptance Criteria**:
  - [ ] All 11 docs files exist
  - [ ] `mkdocs build --strict` exits 0
  - [ ] Architecture docs include Mermaid diagrams

  **QA Scenarios**:
  ```
  Scenario: mkdocs build succeeds
    Tool: Bash
    Preconditions: T1, T19, T21, T25 complete
    Steps:
      1. uv run mkdocs build --strict
      2. Assert exit code 0
      3. Assert site/index.html exists
    Expected Result: Docs build cleanly
    Evidence: .sisyphus/evidence/task-25-mkdocs.txt
  ```

  **Commit**: YES
  - Message: `docs: add full documentation in docs/ with mkdocs`
  - Files: `docs/*.md`, `docs/mkdocs.yml`, `tests/test_docs.py`

---

- [x] 26. **Jenkinsfile + local CI script (replacing GitHub Actions)**

  **What to do**:
  - **No GitHub Actions** — user has no GitHub integrations. Use Jenkinsfile + local CI script.
  - Create `Jenkinsfile` (declarative pipeline):
    ```groovy
    pipeline {
      agent any
      options {
        timeout(time: 30, unit: 'MINUTES')
      }
      stages {
        stage('Setup') {
          steps {
            sh 'uv sync --all-extras'
          }
        }
        stage('Lint') {
          parallel {
            stage('ruff check') { steps { sh 'uv run ruff check' } }
            stage('ruff format') { steps { sh 'uv run ruff format --check' } }
            stage('mypy') { steps { sh 'uv run mypy src/' } }
          }
        }
        stage('Test') {
          steps {
            sh 'uv run pytest --cov=img2svg --cov-report=term-missing --cov-fail-under=80 --junitxml=build/junit.xml --json-report --json-report-file=build/report.json'
          }
        }
        stage('Build Docs') {
          steps {
            sh 'uv run mkdocs build --strict'
          }
        }
        stage('Package') {
          when { branch 'main' }
          steps {
            sh 'uv build'
          }
        }
      }
      post {
        always {
          junit 'build/junit.xml'
          archiveArtifacts artifacts: 'build/,dist/,site/,htmlcov/', allowEmptyArchive: true
        }
      }
    }
    ```
  - Create `scripts/ci.sh` — local CI script that mimics Jenkins pipeline (for "this system"):
    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    echo "==> img2svg local CI"
    echo "==> OS: $(uname -s)"
    echo "==> Python: $(python --version 2>&1 || true)"
    echo "==> uv: $(uv --version 2>&1 || true)"

    cd "$(dirname "$0")/.."

    echo "==> uv sync"
    uv sync --all-extras

    echo "==> Lint"
    uv run ruff check
    uv run ruff format --check
    uv run mypy src/

    echo "==> Tests"
    mkdir -p build
    uv run pytest \
      --cov=img2svg \
      --cov-report=term-missing \
      --cov-fail-under=80 \
      --junitxml=build/junit.xml \
      --json-report \
      --json-report-file=build/report.json \
      "$@"

    echo "==> Docs"
    uv run mkdocs build --strict

    echo "==> Done. See build/ for reports."
    ```
  - Make `scripts/ci.sh` executable
  - Create `tests/test_ci.py`:
    - `Jenkinsfile` exists, contains key stages (Setup, Lint, Test, Build Docs)
    - `scripts/ci.sh` exists, is executable
    - `scripts/ci.sh` syntax is valid bash (`bash -n scripts/ci.sh`)
    - Does NOT require actually running the script (CI test is structural only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T1, T19, T21, T25

  **References**:
  - **External**: Jenkinsfile syntax: https://www.jenkins.io/doc/book/pipeline/syntax/
  - **External**: pytest JSON report: https://github.com/numirias/pytest-json-report

  **Acceptance Criteria**:
  - [ ] `Jenkinsfile` exists with all required stages
  - [ ] `scripts/ci.sh` exists, is executable
  - [ ] `bash -n scripts/ci.sh` passes
  - [ ] NO `.github/workflows/` directory created

  **QA Scenarios**:
  ```
  Scenario: CI files exist and are valid
    Tool: Bash (pytest + bash -n)
    Preconditions: T1, T19, T21, T25, T26 complete
    Steps:
      1. uv run pytest tests/test_ci.py -v
      2. Assert exit code 0
      3. bash -n scripts/ci.sh; assert exit code 0
      4. ls -la .github/ 2>&1; assert "No such file" (no GitHub Actions)
    Expected Result: CI files valid, no GitHub Actions created
    Evidence: .sisyphus/evidence/task-26-ci.txt
  ```

  **Commit**: YES
  - Message: `ci: add Jenkinsfile and local CI script (no GitHub Actions)`
  - Files: `Jenkinsfile`, `scripts/ci.sh`, `tests/test_ci.py`

---

- [x] 27. **Integration tests (real YOLO on small model, real vtracer, batch, GPU)**

  **What to do**:
  - Create `tests/integration/` (or add to existing tests with `@pytest.mark.integration` and `@pytest.mark.slow`):
    - `test_integration.py::test_real_yolo_detects_person` — use a synthetic image with a person-like shape, assert detection
    - `test_integration.py::test_real_vtracer_produces_svg` — feed a logo, assert SVG output is non-empty
    - `test_integration.py::test_full_pipeline_end_to_end` — `convert(fixture, /tmp/out.svg)`, assert SVG + sidecar
    - `test_integration.py::test_batch_with_errors` — 1 corrupt + 2 valid, assert 2 succeed
    - `test_integration.py::test_all_4_modes` — run all 4 modes, assert each produces valid SVG
  - Mark slow tests with `@pytest.mark.slow` so default `pytest` skips them
  - Default `pytest` should run only fast unit tests; `pytest -m "not slow"` or `pytest --run-integration` for full suite
  - Update `pyproject.toml`:
    - `[tool.pytest.ini_options]` add `markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]`
  - Add to `Jenkinsfile` and `scripts/ci.sh`:
    - `uv run pytest -m "not slow"` for fast feedback
    - `uv run pytest` (full suite) for nightly
  - Run integration tests locally to verify they work (with timeout)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (T27 modifies `Jenkinsfile` and `scripts/ci.sh` created in T26)
  - **Parallel Group**: Wave 4 (sequential after T26)
  - **Blocks**: F1-F4 (final verification)
  - **Blocked By**: T1-T26

  **Acceptance Criteria**:
  - [ ] `pytest -m "not slow"` passes (fast unit tests)
  - [ ] `pytest` (full suite, with slow) passes when run with sufficient time
  - [ ] `pytest --cov=img2svg --cov-fail-under=80` passes

  **QA Scenarios**:
  ```
  Scenario: Integration test suite passes
    Tool: Bash (pytest)
    Preconditions: T1-T25 complete
    Steps:
      1. uv run pytest -m "not slow" --cov=img2svg --cov-report=term-missing
      2. Assert exit code 0
      3. Assert coverage >= 80%
    Expected Result: Fast tests pass with good coverage
    Evidence: .sisyphus/evidence/task-27-integration.txt
  ```

  **Commit**: YES
  - Message: `test(integration): add real-model integration test suite`
  - Files: `tests/integration/`, `pyproject.toml` (pytest markers)

---

- [x] 28. **PyPI packaging metadata (readme rendering, classifiers, URLs)**

  **What to do**:
  - Update `pyproject.toml`:
    - `[project]` add:
      - `description = "Convert raster images to SVG with object and pattern detection"`
      - `readme = "README.md"`
      - `keywords = ["svg", "image", "vector", "yolo", "opencv", "object-detection", "vtracer"]`
      - `authors = [{name = "Mark LaPointe", email = "mark@cloudbsd.org"}]`
      - `license = {text = "BSD-3-Clause"}`
      - `classifiers`:
        - `"Development Status :: 4 - Beta"`
        - `"Intended Audience :: Developers"`
        - `"License :: OSI Approved :: BSD License"`
        - `"Operating System :: POSIX :: Linux"`
        - `"Operating System :: MacOS :: MacOS X"`
        - `"Operating System :: POSIX :: BSD :: FreeBSD"`
        - `"Programming Language :: Python :: 3.10"`
        - `"Programming Language :: Python :: 3.11"`
        - `"Programming Language :: Python :: 3.12"`
        - `"Topic :: Multimedia :: Graphics :: Graphics Conversion"`
        - `"Topic :: Scientific/Engineering :: Artificial Intelligence"`
      - `[project.urls]`:
        - `Homepage = "https://github.com/cloudbsdorg/img2svg"`
        - `Repository = "https://github.com/cloudbsdorg/img2svg.git"`
        - `Issues = "https://github.com/cloudbsdorg/img2svg/issues"`
        - `Documentation = "https://cloudbsdorg.github.io/img2svg/"`
    - `[project.scripts]`:
      - `img2svg = "img2svg.cli:app"`
    - `[tool.hatch.build.targets.wheel]` include `man/img2svg.1`, `locale/`
  - Test: `uv build` produces `dist/img2svg-0.1.0-py3-none-any.whl` and `.tar.gz`
  - Inspect wheel: `unzip -l dist/img2svg-0.1.0-py3-none-any.whl | grep -E "img2svg.1|locale"`
  - Test install from wheel: `uv pip install dist/img2svg-0.1.0-py3-none-any.whl --force-reinstall`
  - Verify: `which img2svg` and `img2svg --version` work after install
  - Add `dist/`, `build/`, `*.egg-info/` to `.gitignore` (already done in T1)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T1, T19, T21, T23, T24, T25

  **Acceptance Criteria**:
  - [ ] `uv build` exits 0
  - [ ] Wheel contains `man/img2svg.1` and `locale/`
  - [ ] `img2svg` command works after installing wheel

  **QA Scenarios**:
  ```
  Scenario: uv build produces installable wheel
    Tool: Bash
    Preconditions: T1, T19, T21, T23, T24, T25, T28 complete
    Steps:
      1. rm -rf dist/ build/ *.egg-info
      2. uv build
      3. Assert exit code 0
      4. ls dist/ (assert both .whl and .tar.gz present)
      5. uv pip install dist/img2svg-0.1.0-py3-none-any.whl --force-reinstall
      6. img2svg --version; assert exit code 0
    Expected Result: Build + install work
    Evidence: .sisyphus/evidence/task-28-build.txt
  ```

  **Commit**: YES
  - Message: `build: add PyPI packaging metadata`
  - Files: `pyproject.toml` (updated)

---

- [x] 29. **FreeBSD compatibility notes + smoke test**

  **What to do**:
  - Create `docs/platforms/freebsd.md`:
    - Required ports/packages: `python311`, `uv` (via `pkg install py311-uv` or `cargo install uv`)
    - vtracer install: from FreeBSD ports (`graphics/vtracer`) or pip (may need `pkg install rust` for source build)
    - PyTorch + ultralytics on FreeBSD: limited; CPU-only realistic
    - Known limitations: no CUDA, no ROCm; vtracer needs source build
    - Test instructions: `uv sync --all-extras` then `uv run img2svg foo.png -o foo.svg`
  - Create `scripts/check_freebsd.sh`:
    - Detect FreeBSD via `uname -s`
    - Print install instructions
    - Run smoke test: `uv run img2svg tests/fixtures/logo.png -o /tmp/out.svg`
  - Create `tests/test_platform.py`:
    - Detect OS via `uname -s`, not `platform.system()`
    - Test: `os_name == subprocess.check_output(["uname", "-s"]).text.strip()`
    - Test: `PlatformInfo(supports_gpu=...)` correctly identifies FreeBSD as CPU-only
  - Document macOS support in `docs/platforms/macos.md`:
    - `brew install python@3.11 uv`
    - `uv sync --all-extras`
    - MPS works out of the box on Apple Silicon

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T1, T19, T25

  **Acceptance Criteria**:
  - [ ] `docs/platforms/freebsd.md` exists
  - [ ] `docs/platforms/macos.md` exists
  - [ ] `scripts/check_freebsd.sh` works
  - [ ] `uname -s` is used (not `platform.system()`)

  **QA Scenarios**:
  ```
  Scenario: Platform detection uses uname
    Tool: Bash (pytest)
    Preconditions: T1, T19, T25, T29 complete
    Steps:
      1. uv run pytest tests/test_platform.py -v
      2. Assert exit code 0
      3. grep -r "platform.system" src/; assert no matches (use uname instead)
    Expected Result: Platform detection correct
    Evidence: .sisyphus/evidence/task-29-platform.txt
  ```

  **Commit**: YES
  - Message: `docs(platforms): add FreeBSD and macOS compatibility notes`
  - Files: `docs/platforms/freebsd.md`, `docs/platforms/macos.md`, `scripts/check_freebsd.sh`, `tests/test_platform.py`

---

- [x] 30. **Example gallery (sample inputs → outputs) + before/after comparisons**

  **What to do**:
  - Create `examples/`:
    - `basic_usage.md` — 5 common usage patterns with code
    - `sample_inputs/README.md` — list of sample images
    - `sample_outputs/` — pre-generated SVG outputs (4 modes for 2-3 sample images)
    - `before_after.md` — visual comparison: input image vs SVG output (use Markdown image links)
    - `python_api.md` — Python API usage examples (3-5 patterns)
    - `gpu_recommendation.md` — example output of `img2svg list-gpus`
  - Generate sample outputs:
    ```bash
    uv run img2svg tests/fixtures/logo.png -o examples/sample_outputs/logo_labels.svg --mode labels
    uv run img2svg tests/fixtures/logo.png -o examples/sample_outputs/logo_visual.svg --mode visual
    # ... etc for all 4 modes x 2 images
    ```
  - Verify all outputs are valid SVG (parse with lxml)

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T19, T21

  **Acceptance Criteria**:
  - [ ] `examples/basic_usage.md` exists with 5+ examples
  - [ ] `examples/sample_outputs/` contains valid SVGs for 4 modes × 2 images
  - [ ] `examples/before_after.md` shows input/output comparisons

  **QA Scenarios**:
  ```
  Scenario: Example gallery is complete
    Tool: Bash
    Preconditions: T19, T21, T30 complete
    Steps:
      1. ls examples/sample_outputs/ (assert >= 8 SVGs)
      2. for f in examples/sample_outputs/*.svg; do uv run python -c "import lxml.etree; lxml.etree.parse('$f')"; done; assert all exit 0
    Expected Result: All examples valid
    Evidence: .sisyphus/evidence/task-30-examples.txt
  ```

  **Commit**: YES
  - Message: `docs(examples): add example gallery with sample inputs and outputs`
  - Files: `examples/`

---

- [x] 31. **Final cleanup (deps audit, .gitignore, AGPL notice, BSD headers)**

  **What to do**:
  - Run `uv tree` and verify no unused dependencies
  - Run `uv run pip-audit` (if available) or `uv run safety check` for security audit
  - Verify all source files have BSD 3-Clause header:
    ```python
    # img2svg - Convert raster images to SVG with object and pattern detection.
    # Copyright (c) 2026, CloudBSD
    # SPDX-License-Identifier: BSD-3-Clause
    ```
  - Add `NOTICE` file mentioning AGPL-3.0 dependency (ultralytics) is used under its license
  - Verify `.gitignore` is complete:
    ```
    .sisyphus/
    __pycache__/
    *.pyc
    .venv/
    *.egg-info/
    dist/
    build/
    .pytest_cache/
    .coverage
    htmlcov/
    .mypy_cache/
    .ruff_cache/
    ```
  - Verify no secrets in repo: `grep -rE "(api[_-]?key|secret|password|token)" --include="*.py" --include="*.yml" --include="*.toml" --include="*.md" src/ docs/`
  - Run final test pass: `uv run pytest --cov=img2svg --cov-fail-under=80`
  - Run final lint: `uv run ruff check` and `uv run ruff format --check`
  - Final commit: `chore: final cleanup and pre-release audit`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: F1-F4
  - **Blocked By**: T1-T30

  **Acceptance Criteria**:
  - [ ] All source files have BSD 3-Clause header
  - [ ] `NOTICE` file present with AGPL notice
  - [ ] `uv run pytest` passes with 80%+ coverage
  - [ ] `uv run ruff check` passes
  - [ ] No secrets in repo
  - [ ] `.gitignore` is complete

  **QA Scenarios**:
  ```
  Scenario: Final pre-release audit
    Tool: Bash
    Preconditions: T1-T30 complete
    Steps:
      1. uv run pytest --cov=img2svg --cov-fail-under=80
      2. uv run ruff check
      3. uv run ruff format --check
      4. grep -rE "Copyright.*CloudBSD" src/ --include="*.py" | wc -l; assert >= 10
      5. test -f NOTICE; assert exit 0
    Expected Result: All clean
    Evidence: .sisyphus/evidence/task-31-cleanup.txt
  ```

  **Commit**: YES
  - Message: `chore: final cleanup and pre-release audit`
  - Files: `NOTICE`, source headers added, `.gitignore` finalization

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.

### Consolidated Final Wave Result (2026-06-10)

| Reviewer | Verdict | Evidence |
|----------|---------|----------|
| F1 Plan Compliance | **APPROVE** | `.sisyphus/evidence/f1-plan-compliance.md` |
| F2 Code Quality | **APPROVE** | `.sisyphus/evidence/f2-code-quality.md` |
| F3 Real Manual QA | **APPROVE WITH CAVEAT** | `.sisyphus/evidence/f3-manual-qa.md` |
| F4 Scope Fidelity | **APPROVE WITH ADVISORIES** | `.sisyphus/evidence/f4-scope-fidelity.md` |

**F3 caveat**: `--no-clobber` flag is a documented no-op (stored in `ConversionOptions` but `pipeline.run()` never checks it). Reproducible bug. Fix recommendation: 1-2 lines in `pipeline.py` + 1 unit test.

**F4 advisories**: (1) `pydantic>=2.2,<3` should be added to `pyproject.toml` dependencies (currently transitive via ultralytics). (2) `tomli>=2.0,<3 ; python_version < "3.11"` should be declared for Py3.10 fallback. (3) Unused `supervision` dep can be removed. (4) `.idea/` should be added to `.gitignore` for portability. None block release.

- [x] F1. **Plan Compliance Audit** — `oracle` ✅ APPROVE
  Report: `.sisyphus/evidence/f1-plan-compliance.md` — All must-haves verified, all must-nots absent, 31/31 tasks.

- [x] F2. **Code Quality Review** — `unspecified-high` ✅ APPROVE
  Report: `.sisyphus/evidence/f2-code-quality.md` — Format PASS, Tests 319/320, Coverage 86.18%, AI slop 0; 32 ruff + 40 mypy + 1 i18n pre-existing (T31-documented).

- [x] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI output reviewed) ✅ APPROVE WITH CAVEAT
  Report: `.sisyphus/evidence/f3-manual-qa.md` — 28/29 scenarios pass; one bug found: `--no-clobber` is a no-op (flag stored in `ConversionOptions` but `pipeline.run()` never checks it). Out of F3 scope — recommended fix in report §6.2.

- [x] F4. **Scope Fidelity Check** — `deep` ✅ APPROVE WITH ADVISORIES
  Report: `.sisyphus/evidence/f4-scope-fidelity.md` — 31/31 tasks compliant, no contamination, 11/11 must-nots respected. Advisories: 2 missing dep declarations (`pydantic`, conditional `tomli`), 1 unused dep (`supervision`). All non-blocking.

---

## Commit Strategy

- **T1**: `chore: scaffold img2svg project with BSD 3-Clause license` — `pyproject.toml`, `LICENSE`, `src/`, `tests/`, `.gitignore`
- **T2–T7**: `feat(foundation): add XDG paths, errors, models, device, loader, fixtures` — per module
- **T8–T15**: `feat(core): add detection, patterns, classification, vectorization, SVG, sidecar, logging` — per module
- **T16–T23**: `feat(render+pipeline): add renderers, pipeline, batch, CLI, GPU command, man page` — per module
- **T24–T31**: `docs+ci: add README, docs/, CI, integration tests, packaging, FreeBSD notes, examples, cleanup` — per module
- **Final**: `release: img2svg v0.1.0` — tagged, on `main`

Git author: `Mark LaPointe <mark@cloudbsd.org>` (per CloudBSD guidelines, set via `git config user.name` and `user.email` on first commit).

---

## Success Criteria

### Verification Commands
```bash
# Install
uv sync --all-extras

# Run all tests with coverage
uv run pytest --cov=img2svg --cov-report=term-missing --cov-fail-under=80

# AI-parseable test output
uv run pytest --json-report --json-report-file=.sisyphus/evidence/pytest-report.json
uv run pytest --tap         # TAP output
uv run pytest --junitxml=.sisyphus/evidence/junit.xml

# Lint + types
uv run ruff check
uv run ruff format --check
uv run mypy src/

# CLI smoke
uv run img2svg --help
uv run img2svg --version
uv run img2svg --list-gpus
uv run img2svg tests/fixtures/logo.png -o /tmp/out.svg
uv run img2svg tests/fixtures/*.png --output-dir /tmp/

# Man page
man man/img2svg.1

# Build docs
cd docs && mkdocs build --strict
```

### Final Checklist
- [ ] All "Must Have" present and verified
- [ ] All "Must NOT Have" absent (search confirms)
- [ ] All tests pass with ≥80% coverage
- [ ] Lint and types pass
- [ ] CI green on Linux + macOS matrix
- [ ] Man page renders correctly
- [ ] Docs build without warnings
- [ ] License is BSD 3-Clause
- [ ] Git author is Mark LaPointe <mark@cloudbsd.org>
- [ ] JSON sidecar validates with Pydantic
- [ ] All 4 modes produce valid SVG
- [ ] GPU autodetect + recommendation works
- [ ] Multi-vendor GPU detected (when applicable)
- [ ] FreeBSD compatibility documented
- [ ] macOS install + run works
- [ ] Linux install + run works
- [ ] All 4 review tasks (F1–F4) APPROVE
- [ ] User gives explicit okay

---

## Platform Support Matrix

| Platform | Status | Notes |
|---|---|---|
| **Linux x86_64** | ✅ Full | All features; CUDA + ROCm both supported via PyTorch |
| **Linux aarch64** | ✅ Full | CUDA + ROCm both supported (DGX, Jetson) |
| **macOS arm64 (Apple Silicon)** | ✅ Full | MPS backend; vtracer wheel available |
| **macOS x86_64 (Intel)** | ⚠️ Best-effort | vtracer wheel available; PyTorch CPU only (no MPS) |
| **FreeBSD amd64** | ⚠️ Best-effort | vtracer needs source build; ultralytics limited; CPU-only realistic |
| **FreeBSD arm64** | ⚠️ Best-effort | Same as amd64; less tested |
| **Windows x86_64** | ❌ Out of scope (v1) | Not in CloudBSD's target list; can be added if requested |

### OS Detection
Always use `uname -s` (per CloudBSD guidelines) — never trust `platform.system()` from inside a container/VM.

```python
import platform, subprocess
# WRONG: relies on Python's view, which may be wrong inside a container
# os_name = platform.system()
# RIGHT: query the kernel directly
os_name = subprocess.check_output(["uname", "-s"], text=True).strip()
# Returns: "Linux", "Darwin", "FreeBSD", etc.
```

---

## File Layout

```
img2svg/
├── pyproject.toml
├── uv.lock                       # committed
├── LICENSE                       # BSD 3-Clause
├── README.md
├── .gitignore
├── .python-version               # 3.10
├── src/
│   └── img2svg/
│       ├── __init__.py           # public API exports
│       ├── __main__.py           # python -m img2svg
│       ├── api.py                # convert, convert_batch, high-level API
│       ├── pipeline.py           # orchestrator
│       ├── cli.py                # Typer CLI
│       ├── device.py             # GPU autodetect
│       ├── gpu.py                # GPU enumeration + recommendation
│       ├── loader.py             # image I/O
│       ├── classifier.py         # image type heuristic
│       ├── detector.py           # YOLO wrapper
│       ├── patterns.py           # OpenCV geometric
│       ├── vectorizer.py         # vtracer wrapper
│       ├── svg_builder.py        # SVG assembly
│       ├── metadata.py           # JSON sidecar
│       ├── paths.py              # XDG paths
│       ├── i18n.py               # gettext setup
│       ├── errors.py             # typed exceptions
│       ├── logging.py            # Rich logging
│       ├── enums.py              # Mode, ImageType, Device
│       ├── models.py             # Pydantic models
│       ├── presets.py            # vtracer configs + mode presets
│       ├── renderers/
│       │   ├── __init__.py
│       │   ├── base.py           # Renderer ABC
│       │   ├── labels.py         # LabelsRenderer
│       │   ├── visual.py         # VisualRenderer
│       │   ├── annotated.py      # AnnotatedRenderer
│       │   └── trace.py          # TraceRenderer
│       └── locale/
│           └── img2svg.pot       # gettext template
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── logo.png              # synthetic
│   │   ├── photo.jpg             # synthetic
│   │   ├── diagram.png           # synthetic
│   │   ├── line_art.png          # synthetic
│   │   ├── transparent.png       # synthetic
│   │   └── corrupt.bin           # for error tests
│   ├── test_api.py
│   ├── test_pipeline.py
│   ├── test_device.py
│   ├── test_gpu.py
│   ├── test_loader.py
│   ├── test_classifier.py
│   ├── test_detector.py
│   ├── test_patterns.py
│   ├── test_vectorizer.py
│   ├── test_svg_builder.py
│   ├── test_metadata.py
│   ├── test_paths.py
│   ├── test_i18n.py
│   ├── test_errors.py
│   ├── test_logging.py
│   ├── test_renderers/
│   │   ├── test_labels.py
│   │   ├── test_visual.py
│   │   ├── test_annotated.py
│   │   └── test_trace.py
│   ├── test_cli.py
│   ├── test_batch.py
│   ├── test_gpu_recommend.py
│   └── test_platform.py
├── man/
│   └── img2svg.1
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   ├── api.md
│   ├── modes.md
│   ├── gpu.md
│   ├── configuration.md
│   ├── troubleshooting.md
│   ├── development.md
│   ├── architecture.md
│   ├── changelog.md
│   └── mkdocs.yml
├── examples/
│   ├── basic_usage.md
│   ├── sample_inputs/
│   │   └── README.md
│   ├── sample_outputs/
│   │   ├── labels.svg
│   │   ├── visual.svg
│   │   ├── annotated.svg
│   │   └── trace.svg
│   └── before_after.md
├── Jenkinsfile                    # Jenkins pipeline (declarative)
├── scripts/
│   ├── gen_test_images.py        # synthetic test image generator
│   ├── check_platform.sh         # uname-based platform check
│   ├── check_freebsd.sh          # FreeBSD install + smoke test
│   ├── install_manpage.sh        # install man page
│   ├── update_locale.sh          # gettext regeneration
│   └── ci.sh                     # local CI (replaces GitHub Actions)
└── .sisyphus/
    ├── drafts/                   # working memory (deleted after plan)
    ├── plans/
    │   └── img2svg.md            # this file
    └── evidence/                 # QA evidence
        └── task-{N}-{slug}.{ext}
```
