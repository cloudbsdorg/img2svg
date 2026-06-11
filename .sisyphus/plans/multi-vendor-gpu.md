# Multi-Vendor GPU Support (DeviceBackend abstraction)

## TL;DR

> **Quick Summary**: Introduce a `DeviceBackend` Protocol with 4 first-class implementations (CUDA, ROCm, MPS, CPU) so the codebase abstracts over PyTorch's per-vendor wheel matrix. A smart `scripts/install_backend.sh` detects hardware via `lspci` and installs the right PyTorch wheel. Jenkins gets per-vendor stages.
>
> **Deliverables**:
> - `src/img2svg/backends/` package: `protocol.py`, `registry.py`, `cuda.py`, `rocm.py`, `mps.py`, `cpu.py`, `__init__.py`
> - `src/img2svg/models.py`: `BackendSpec` Pydantic model replacing unconstrained `device: str`
> - `scripts/install_backend.sh`: hardware-detect → wheel install → verify
> - `pyproject.toml`: `[nvidia]`, `[amd]`, `[apple]`, `[cpu]` extras
> - `Jenkinsfile`: 4 parallel backend stages (matrix project)
> - Updated `device.py`, `detector.py`, `pipeline.py`, `gpu.py`, `cli.py`
> - Updated `docs/installation.md`, `README.md`
>
> **Estimated Effort**: Short (~12-15 implementation tasks + 4 final verifications)
> **Parallel Execution**: YES (3 waves: foundation → vendors → wiring)
> **Critical Path**: T1 protocol → T3 BackendSpec → T4-T7 backends → T9 registry → T10-T13 wiring

---

## Context

### Original Request
"come up with a plan to get both amd/nvidia/etc to work"

### Interview Summary
**User decisions (2026-06-10):**
- **Pluggable backend architecture** — Build `DeviceBackend` Protocol, all 4 vendors first-class
- **Smart install script** — `scripts/install_backend.sh` detects GPU via lspci, installs right wheel
- **Scope**: NVIDIA + AMD + Apple MPS + CPU (4 backends, no Intel XPU/ONNX Runtime in v1)
- **CI**: Backend-specific Jenkins stages (matrix project pattern)

### Research Findings (full report: `.sisyphus/research/multi-vendor-gpu-2026.md`)

| Finding | Implication |
|---------|-------------|
| **512 MB iGPU cannot run YOLO11x** (114 MB model + activations) | AMD iGPU is "detection only"; NVIDIA discrete is the actual inference device |
| **PyTorch wheel matrix is per-vendor** (no single wheel covers NVIDIA + AMD + Apple) | Need a smart install script that picks the right index-url |
| **ultralytics 1-line ONNX export + 1-line ONNX inference** | Future option to migrate to ONNX Runtime; not v1 scope |
| **gfx1150 (Radeon 890M) needs ROCm 7.0.2+** | `repo.radeon.com` wheels or Docker; PyTorch.org stable still ships ROCm 6.3 |
| **`HSA_OVERRIDE_GFX_VERSION` does NOT work for gfx1150** | ISA gap too wide; only the proper ROCm wheel works |
| **ZLUDA / SCALe are not production-ready for PyTorch** in 2026 | Skip these paths; track for future |
| **Apple MPS works with ultralytics ≥ 8.4.x** | Pin ultralytics version in `pyproject.toml` |
| **Jenkins matrix-project plugin is the standard pattern** | Per-vendor stages with Docker + device passthrough |

### Audit Findings (full report: `.sisyphus/research/audit-gpu-touchpoints.md`)

| Layer | torch.cuda touch | Abstraction priority |
|-------|------------------|----------------------|
| `device.py` (82 lines) | 5 calls (queries) | **MUST** — single source of truth |
| `detector.py` (104 lines) | 0 direct (uses ultralytics) | **MUST** — `"rocm"` string breaks at YOLO call site (L81) |
| `models.py` (L74) | 0 | **MUST** — `device: str` is unconstrained |
| `pipeline.py` (L146, L185) | 0 | **SHOULD** — cache key + sidecar use raw user string, not canonical |
| `gpu.py` (L458) | 4 (enumeration) | **SHOULD** — `_torch_fallback` hardcodes `vendor=NVIDIA` even for ROCm |
| `vectorizer.py`, `patterns.py`, `loader.py`, `renderers/*` | 0 | **None** — already backend-agnostic (vtracer=CPU, cv2=CPU, PIL=CPU) |
| `tests/test_device.py` | 10 mock points | **MUST** — will need updating |

**Surface area is small**: 9 torch.cuda calls in 2 files; all are queries. The detector is the only consumer that needs device-string translation (L81). vtracer, OpenCV, and Pillow are already CPU-only and unaffected.

### Metis Review (to be done before commit)
- [ ] **Scope creep risk**: Resist adding ONNX Runtime migration to v1; defer to v2
- [ ] **Hardware assumption**: Plan must work for systems WITHOUT a GPU (CPU-only CI agent)
- [ ] **Test isolation**: Each backend's integration test must not require that backend to be installed
- [ ] **Docstring debt**: All Protocol methods need full docstrings (current code style)

---

## Work Objectives

### Core Objective
Replace the string-based device abstraction in img2svg with a `DeviceBackend` Protocol that supports 4 first-class compute backends (CUDA, ROCm, MPS, CPU) with a smart install script and per-vendor CI.

### Concrete Deliverables
- [ ] `src/img2svg/backends/__init__.py` — public API
- [ ] `src/img2svg/backends/protocol.py` — `DeviceBackend` Protocol + `BackendType` enum
- [ ] `src/img2svg/backends/registry.py` — `BackendRegistry` with auto-detection
- [ ] `src/img2svg/backends/cuda.py` — `CUDABackend` (uses `torch.cuda`)
- [ ] `src/img2svg/backends/rocm.py` — `ROCMBackend` (uses `torch.cuda` API but marks vendor=AMD; checks `torch.version.hip`)
- [ ] `src/img2svg/backends/mps.py` — `MPSBackend` (uses `torch.backends.mps`)
- [ ] `src/img2svg/backends/cpu.py` — `CPUBackend` (always available)
- [ ] `src/img2svg/models.py` — replace `device: str` with `backend: BackendSpec` (Pydantic-validated)
- [ ] `scripts/install_backend.sh` — detect GPU, install matching PyTorch wheel, verify
- [ ] `pyproject.toml` — `[nvidia]`, `[amd]`, `[apple]`, `[cpu]` extras + `ultralytics>=8.4.0` pin
- [ ] `Jenkinsfile` — 4 parallel backend stages (matrix project)
- [ ] Updated `src/img2svg/device.py` — delegate to `BackendRegistry`
- [ ] Updated `src/img2svg/detector.py` — use `BackendSpec`; translate "rocm" → "cuda:0" before passing to ultralytics
- [ ] Updated `src/img2svg/pipeline.py` — use `BackendSpec` for cache key + sidecar
- [ ] Updated `src/img2svg/gpu.py` — `_torch_fallback` uses `torch.version.hip` to label ROCm correctly
- [ ] Updated `src/img2svg/cli.py` — `--device` help text + `info` shows active backend
- [ ] `docs/installation.md` — split by vendor with wheel install commands
- [ ] Updated `README.md` — quickstart with "what GPU do you have?" decision tree
- [ ] `tests/test_backends.py` — Protocol conformance + registry tests
- [ ] `tests/integration/test_backend_matrix.py` — per-vendor smoke tests (env-gated)

### Definition of Done
- [ ] `uv run pytest -m "not slow"` passes (current 345 + new backend tests, 0 regressions)
- [ ] `uv run python -m img2svg list-gpus` shows NVIDIA RTX 5070 + AMD Radeon 890M (already done)
- [ ] `uv run python -m img2svg info` shows active backend type (NEW: `Backend: CUDA (torch 2.12.0+cu130)`)
- [ ] `scripts/install_backend.sh` correctly identifies vendor and prints right install command
- [ ] `Jenkinsfile` matrix builds succeed on CPU + NVIDIA agents (manual verification on AMD/Apple)
- [ ] `uv run ruff check src tests` shows no new errors
- [ ] `uv run mypy src/` shows no new errors

### Must Have
- `DeviceBackend` Protocol with methods: `is_available()`, `device_count()`, `device_name(i)`, `total_memory_mb(i)`, `free_memory_mb(i)`, `vendor()`, `to_ultralytics_string(i)`, `warmup()`
- All 4 backend implementations
- Smart install script that doesn't lie (prints what it would install, requires user confirmation if destructive)
- BackendSpec Pydantic model with `Literal["auto", "cuda", "cuda:N", "rocm", "mps", "cpu"]` constraint
- Per-vendor Jenkins stages (can be no-op agents if hardware unavailable)

### Must NOT Have (Guardrails)
- **No ONNX Runtime migration** (defer to v2 — out of scope for v1)
- **No ZLUDA / SCALe / HSA_OVERRIDE_GFX_VERSION hacks** (not production-ready per research)
- **No Intel XPU / DirectML** (out of scope per user decision)
- **No silent wheel downgrades** — install script must warn if current torch doesn't match GPU vendor
- **No breaking changes to the public API** (`from img2svg import convert, convert_batch, ...` still works)
- **No new top-level deps without explicit user opt-in** — backend extras are opt-in

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - All verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (345 fast tests, 8 slow deselected)
- **Automated tests**: Tests-after (new tests added per task; existing 345 stay green)
- **Framework**: pytest (existing)
- **If TDD**: Each task follows RED (failing test) → GREEN (impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.

- **Backend detection**: `uv run python -c "from img2svg.backends import BackendRegistry; print(BackendRegistry.detect())"`
- **Per-vendor paths**: `uv run python -m img2svg list-gpus` (NVIDIA + AMD both visible)
- **Install script**: `bash scripts/install_backend.sh --dry-run` (verify detection without installing)
- **CLI**: `uv run python -m img2svg info` (shows active backend)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation + abstractions):
├── T1: DeviceBackend Protocol + BackendType enum
├── T2: BackendSpec Pydantic model
├── T3: CPUBackend (always available, lowest risk)
└── T4: Refactor `device.py` to use new abstraction

Wave 2 (After Wave 1 - vendor backends, MAX PARALLEL):
├── T5: CUDABackend (uses torch.cuda)
├── T6: ROCMBackend (uses torch.cuda API + torch.version.hip)
├── T7: MPSBackend (uses torch.backends.mps)
└── T8: BackendRegistry with auto-detection

Wave 3 (After Wave 2 - wire into existing code):
├── T9: Refactor `detector.py` to use BackendSpec
├── T10: Refactor `pipeline.py` (cache key + sidecar)
├── T11: Refactor `gpu.py` (_torch_fallback vendor detection)
└── T12: Refactor `cli.py` (--device help + info output)

Wave 4 (After Wave 3 - install + docs + CI):
├── T13: Smart install script + pyproject extras
├── T14: Documentation updates (installation.md, README)
└── T15: Jenkinsfile matrix stages

Wave FINAL (After ALL implementation tasks - 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)

Critical Path: T1 → T2 → T3 → T5/T6/T7 → T8 → T9 → T10 → T11 → T12 → T13 → T14 → T15 → F1-F4
Parallel Speedup: ~50% faster than sequential (4 backends in parallel)
```

### Dependency Matrix (abbreviated)

- **T1-T4**: No deps, run in parallel
- **T5-T7**: Depend on T1 (Protocol)
- **T8**: Depends on T5, T6, T7
- **T9-T12**: Depend on T8 (registry)
- **T13-T15**: Depend on T12 (wiring complete)

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.

- [x] 1. **DeviceBackend Protocol + BackendType enum** ✅ (commit pending in Wave 1)

  **Status**: Done. `src/img2svg/backends/protocol.py` created (160 lines, BSD-3-Clause). `BackendType` StrEnum with 5 values. `DeviceBackend` `@runtime_checkable` Protocol with 9 methods. Python 3.10 `StrEnum` shim (matches existing `enums.py` pattern). 12 tests pass in `tests/test_backends/test_protocol.py`. mypy clean.

  **What to do**:
  - Create `src/img2svg/backends/__init__.py` and `src/img2svg/backends/protocol.py`
  - Define `BackendType(StrEnum)` with values: `AUTO`, `CUDA`, `ROCM`, `MPS`, `CPU`
  - Define `DeviceBackend(Protocol)` with methods:
    - `type() -> BackendType` — return the backend's type
    - `is_available() -> bool` — is this backend usable right now?
    - `device_count() -> int` — number of devices (0 if unavailable)
    - `device_name(i: int) -> str` — human-readable name
    - `total_memory_mb(i: int) -> int` — total VRAM/RAM in MB
    - `free_memory_mb(i: int) -> int` — free VRAM/RAM in MB
    - `vendor() -> GpuVendor` — return GpuVendor (NVIDIA, AMD, APPLE, UNKNOWN)
    - `to_ultralytics_string(i: int) -> str` — translate to ultralytics device arg (e.g. "cuda:0", "mps", "cpu")
    - `warmup() -> None` — optional pre-flight (e.g. CUDA lazy init); default no-op
  - BSD 3-Clause header. Full docstrings on Protocol + methods.
  - Tests: `tests/test_backends/test_protocol.py` — `CpuBackend` and `MockBackend` satisfy the Protocol (use `@runtime_checkable` or explicit `isinstance` test).

  **Must NOT do**: Don't import torch in this file (Protocol is type-only).

  **Recommended Agent Profile**:
  - **Category**: `quick` — single file, well-specified interface
  - **Skills**: none (no special domain)

  **Parallelization**: Wave 1, parallel with T2/T3/T4.

  **References**:
  - `src/img2svg/enums.py` — `GpuVendor` enum (NVIDIA, AMD, APPLE, INTEL, UNKNOWN) exists
  - `src/img2svg/gpu.py:1-60` — existing pattern for backend-like code (BSD header, type hints, docstrings)
  - `src/img2svg/device.py` — current device-string API; will be replaced

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from img2svg.backends.protocol import DeviceBackend, BackendType; print(BackendType.CUDA)"` exits 0
  - [ ] Protocol has all 9 methods documented
  - [ ] Tests for protocol conformance pass

  **QA Scenarios**:
  ```
  Scenario: Protocol import
    Tool: Bash (uv run python)
    Preconditions: clean working tree
    Steps:
      1. Run: `uv run python -c "from img2svg.backends import DeviceBackend, BackendType; print(BackendType.CUDA.value)"`
      2. Assert: output is `cuda`
    Expected Result: exit 0, stdout contains `cuda`
    Evidence: .sisyphus/evidence/task-T1-protocol-import.txt

  Scenario: Protocol conformance
    Tool: Bash (uv run pytest)
    Preconditions: tests written
    Steps:
      1. Run: `uv run pytest tests/test_backends/test_protocol.py -v`
      2. Assert: all tests pass
    Expected Result: at least 3 tests pass (Protocol import, type-checkable, mock backend satisfies)
    Evidence: .sisyphus/evidence/task-T1-protocol-tests.txt
  ```

  **Commit**: YES
  - Message: `feat(backends): add DeviceBackend Protocol and BackendType enum`
  - Files: `src/img2svg/backends/__init__.py`, `src/img2svg/backends/protocol.py`, `tests/test_backends/__init__.py`, `tests/test_backends/test_protocol.py`
  - Pre-commit: `uv run pytest tests/test_backends/test_protocol.py -q`

- [x] 2. **BackendSpec Pydantic model** ✅ (commit pending in Wave 1)

  **Status**: Done. `BackendSpec` Pydantic v2 model in `src/img2svg/models.py` with `Literal["auto","cuda","cuda:N","rocm","mps","cpu"]` validator that parses `"cuda:0"` → `(requested="cuda", index=0)`. `ConversionOptions.backend` field added with deprecation shim on legacy `device` field. `Sidecar.backend_requested` and `backend_resolved` fields added. 11 new tests in `tests/test_models.py` (28 total, all pass). Pre-commit clean.

  **What to do**:
  - In `src/img2svg/models.py`, add a new `BackendSpec` Pydantic model:
    ```python
    class BackendSpec(BaseModel):
        model_config = ConfigDict(frozen=True)
        requested: Literal["auto", "cuda", "cuda:N", "rocm", "mps", "cpu"] = "auto"
        # Index for cuda/rocm; None = first available
        index: int | None = None
    ```
  - Add a validator that parses `"cuda:0"` → `BackendSpec(requested="cuda", index=0)`, `"cpu"` → `BackendSpec(requested="cpu")`, etc.
  - Update `ConversionOptions.device: str` → `ConversionOptions.backend: BackendSpec = BackendSpec()`. Keep a deprecated `device: str | None = None` shim that, if set, populates `backend` (warn once).
  - Update `Sidecar`: add `backend_requested: str` and `backend_resolved: str` fields (record both for auditability). Keep `device: str` for backward compat with a deprecation warning.
  - Update `tests/test_models.py` to add tests for the new `BackendSpec` model.
  - Run `uv run pytest tests/test_models.py -q` — should pass with new tests + existing 345.

  **Must NOT do**: Don't remove the `device` field (deprecate, don't break).

  **Recommended Agent Profile**:
  - **Category**: `quick` — model + tests
  - **Skills**: none

  **Parallelization**: Wave 1, parallel with T1/T3/T4.

  **References**:
  - `src/img2svg/models.py:67-80` — current `ConversionOptions` (the field to update)
  - `src/img2svg/models.py:55-64` — `GPUInfo` (Pydantic v2 pattern reference)
  - `tests/test_models.py` — existing test patterns

  **Acceptance Criteria**:
  - [ ] `BackendSpec(requested="cuda:0")` parses to `BackendSpec(requested="cuda", index=0)`
  - [ ] `ConversionOptions()` defaults to `BackendSpec(requested="auto")`
  - [ ] Setting `ConversionOptions(device="cuda:0")` populates `backend` and emits a DeprecationWarning
  - [ ] `Sidecar.backend_resolved` is populated when the pipeline runs
  - [ ] All existing tests still pass

  **QA Scenarios**:
  ```
  Scenario: BackendSpec parsing
    Tool: Bash (uv run python)
    Preconditions: clean
    Steps:
      1. Run: `uv run python -c "from img2svg.models import BackendSpec; b = BackendSpec(requested='cuda:0'); print(b.requested, b.index)"`
      2. Assert: output is `cuda 0`
    Expected Result: exit 0
    Evidence: .sisyphus/evidence/task-T2-backendspec-parse.txt

  Scenario: ConversionOptions accepts new backend
    Tool: Bash (uv run pytest)
    Preconditions: tests written
    Steps:
      1. Run: `uv run pytest tests/test_models.py -v -k BackendSpec`
    Expected Result: 5+ new tests pass
    Evidence: .sisyphus/evidence/task-T2-models-tests.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add BackendSpec Pydantic model with parsing`
  - Files: `src/img2svg/models.py`, `tests/test_models.py`
  - Pre-commit: `uv run pytest tests/test_models.py -q`

- [ ] 3. **CPUBackend implementation**

  **What to do**:
  - Create `src/img2svg/backends/cpu.py` with `class CPUBackend:` (no Protocol, just implementation).
  - Methods: `type() -> BackendType.CPU`, `is_available() -> True`, `device_count() -> 1`, `device_name(0) -> "CPU"`, `total_memory_mb(0)` (use `psutil` or `os.sysconf` for RAM), `free_memory_mb(0)` (same), `vendor() -> GpuVendor.UNKNOWN` (or a new `GpuVendor.CPU`), `to_ultralytics_string(0) -> "cpu"`, `warmup() -> None`.
  - Add a singleton `CPU_BACKEND = CPUBackend()`.
  - Tests: `tests/test_backends/test_cpu.py` — verify all methods, especially the memory queries.

  **Must NOT do**: Don't add `psutil` as a dependency; use stdlib `os.sysconf` or `resource.getrusage` for RAM.

  **Recommended Agent Profile**:
  - **Category**: `quick` — single backend impl + tests
  - **Skills**: none

  **Parallelization**: Wave 1, parallel with T1/T2/T4.

  **References**:
  - `src/img2svg/gpu.py:439-470` — `_torch_fallback` pattern (BSD header, type hints, docstrings)
  - `src/img2svg/backends/protocol.py` (T1) — Protocol to implement

  **Acceptance Criteria**:
  - [ ] `CPUBackend().is_available() == True` always
  - [ ] `CPUBackend().to_ultralytics_string(0) == "cpu"`
  - [ ] `CPUBackend().total_memory_mb(0) > 0` on any real system
  - [ ] Tests pass

  **QA Scenarios**:
  ```
  Scenario: CPUBackend always available
    Tool: Bash (uv run python)
    Preconditions: clean
    Steps:
      1. Run: `uv run python -c "from img2svg.backends.cpu import CPU_BACKEND; print(CPU_BACKEND.is_available(), CPU_BACKEND.total_memory_mb(0))"`
      2. Assert: first value is `True`, second is `> 0`
    Expected Result: `True <number>`
    Evidence: .sisyphus/evidence/task-T3-cpu-backend.txt
  ```

  **Commit**: YES (combined with T1 in same commit if Wave 1 lands together; else separate)
  - Message: `feat(backends): add CPUBackend (always available)`
  - Files: `src/img2svg/backends/cpu.py`, `tests/test_backends/test_cpu.py`
  - Pre-commit: `uv run pytest tests/test_backends/test_cpu.py -q`

- [ ] 4. **Refactor `device.py` to use BackendRegistry stub**

  **What to do**:
  - Refactor `src/img2svg/device.py` so the public functions delegate to a (placeholder) registry.
  - Add a minimal `BackendRegistry` class in `src/img2svg/backends/registry.py` that currently returns only `CPUBackend` (T5-T7 will add the rest).
  - Keep the existing public API: `is_available(device: str) -> bool`, `list_available_devices() -> list[str]`, `detect_device(requested: str) -> str`. They must remain backward-compatible.
  - Internally, `detect_device` parses the string into a `BackendSpec`, consults the registry, and returns the ultralytics-compatible string.
  - Add deprecation warnings if `"rocm"` is passed (handled gracefully but warned).

  **Must NOT do**: Don't change the public API. Don't remove the string-accepting functions.

  **Recommended Agent Profile**:
  - **Category**: `quick` — refactor
  - **Skills**: none

  **Parallelization**: Wave 1, parallel with T1/T2/T3.

  **References**:
  - `src/img2svg/device.py:1-82` — full file to refactor
  - `tests/test_device.py` — 9 tests to keep passing

  **Acceptance Criteria**:
  - [ ] All 9 existing `tests/test_device.py` tests still pass
  - [ ] Public API unchanged
  - [ ] Internally uses `BackendSpec` and `BackendRegistry`

  **QA Scenarios**:
  ```
  Scenario: device.py backward compat
    Tool: Bash (uv run pytest)
    Preconditions: refactor done
    Steps:
      1. Run: `uv run pytest tests/test_device.py -v`
      2. Assert: 9 tests pass
    Expected Result: 9 passed
    Evidence: .sisyphus/evidence/task-T4-device-tests.txt

  Scenario: detect_device returns canonical string
    Tool: Bash (uv run python)
    Preconditions: refactor done
    Steps:
      1. Run: `uv run python -c "from img2svg.device import detect_device; print(detect_device('cpu'))"`
      2. Assert: output is `cpu`
    Expected Result: `cpu`
    Evidence: .sisyphus/evidence/task-T4-detect-cpu.txt
  ```

  **Commit**: YES
  - Message: `refactor(device): delegate to BackendRegistry stub (CPU only)`
  - Files: `src/img2svg/device.py`, `src/img2svg/backends/registry.py`, `src/img2svg/backends/__init__.py`
  - Pre-commit: `uv run pytest tests/test_device.py -q`

- [ ] 5. **CUDABackend (NVIDIA)**

  **What to do**:
  - Create `src/img2svg/backends/cuda.py` with `class CUDABackend:`.
  - Methods:
    - `is_available() -> bool` — `torch.cuda.is_available()`
    - `device_count() -> int` — `torch.cuda.device_count()`
    - `device_name(i) -> str` — `torch.cuda.get_device_name(i)`
    - `total_memory_mb(i) -> int` — `torch.cuda.get_device_properties(i).total_memory // (1024*1024)`
    - `free_memory_mb(i) -> int` — `torch.cuda.mem_get_info(i)[0] // (1024*1024)`
    - `vendor() -> GpuVendor.NVIDIA`
    - `to_ultralytics_string(i) -> f"cuda:{i}"`
    - `warmup() -> None` — `torch.cuda.init()` (lazy init)
  - Lazy-import `torch` inside methods (so the backend can be imported on CPU-only systems).
  - Tests: `tests/test_backends/test_cuda.py` with mocked `torch.cuda`.

  **Must NOT do**: Don't fail at import time if torch is missing.

  **Recommended Agent Profile**:
  - **Category**: `quick` — single backend impl + tests
  - **Skills**: none

  **Parallelization**: Wave 2, parallel with T6/T7.

  **References**:
  - `src/img2svg/gpu.py:439-470` — `_torch_fallback` for the torch.cuda pattern
  - `src/img2svg/device.py:30-50` — current `is_available` for `cuda`/`cuda:N`

  **Acceptance Criteria**:
  - [ ] `CUDABackend()` is importable on CPU-only systems (no torch.cuda calls at import)
  - [ ] `CUDABackend().is_available()` returns `True` on this system
  - [ ] `CUDABackend().to_ultralytics_string(0) == "cuda:0"`
  - [ ] `CUDABackend().vendor() == GpuVendor.NVIDIA`
  - [ ] Tests pass

  **QA Scenarios**:
  ```
  Scenario: CUDABackend detects RTX 5070
    Tool: Bash (uv run python)
    Preconditions: this system has NVIDIA
    Steps:
      1. Run: `uv run python -c "from img2svg.backends.cuda import CUDABackend; b = CUDABackend(); print(b.is_available(), b.device_name(0), b.total_memory_mb(0))"`
      2. Assert: `True`, name contains "RTX 5070", total ~ 8151
    Expected Result: `True NVIDIA GeForce RTX 5070 Laptop GPU 8151`
    Evidence: .sisyphus/evidence/task-T5-cuda-backend.txt
  ```

  **Commit**: YES
  - Message: `feat(backends): add CUDABackend (NVIDIA)`
  - Files: `src/img2svg/backends/cuda.py`, `tests/test_backends/test_cuda.py`
  - Pre-commit: `uv run pytest tests/test_backends/test_cuda.py -q`

- [ ] 6. **ROCMBackend (AMD via ROCm)**

  **What to do**:
  - Create `src/img2svg/backends/rocm.py` with `class ROCMBackend:`.
  - Same shape as `CUDABackend` (uses `torch.cuda` API since ROCm exposes it), but:
    - `vendor() -> GpuVendor.AMD`
    - `is_available()` checks `torch.cuda.is_available() AND torch.version.hip is not None`
    - `to_ultralytics_string(i) -> f"cuda:{i}"` (ROCm hides behind CUDA API)
    - `device_name(i)` — prefix with "[ROCm] " so list-gpus is honest
  - The "is this ROCm vs CUDA?" detection MUST use `torch.version.hip`, not lspci — lspci sees hardware; `torch.version.hip` sees the actual installed PyTorch build.
  - Tests: `tests/test_backends/test_rocm.py` with mocked `torch.cuda` + `torch.version.hip`.

  **Must NOT do**: Don't try to use lspci here (it's not Python's job; the install script uses lspci).

  **Recommended Agent Profile**:
  - **Category**: `quick` — single backend impl + tests
  - **Skills**: none

  **Parallelization**: Wave 2, parallel with T5/T7.

  **References**:
  - `src/img2svg/gpu.py:458` — known issue: `_torch_fallback` hardcodes `GpuVendor.NVIDIA` even for ROCm
  - `src/img2svg/backends/cuda.py` (T5) — pattern to follow

  **Acceptance Criteria**:
  - [ ] `ROCMBackend().is_available()` returns `False` on this system (PyTorch is `+cu130`, not `+rocm`)
  - [ ] With a mocked `torch.version.hip = "6.2.41134"`, `is_available()` returns `True` if `torch.cuda.is_available()` is also True
  - [ ] `ROCMBackend().vendor() == GpuVendor.AMD`
  - [ ] Tests pass

  **QA Scenarios**:
  ```
  Scenario: ROCMBackend is_available on this system
    Tool: Bash (uv run python)
    Preconditions: this system has PyTorch CUDA build (not ROCm)
    Steps:
      1. Run: `uv run python -c "from img2svg.backends.rocm import ROCMBackend; b = ROCMBackend(); print(b.is_available(), torch.version.hip)"` — wait, need to import torch first
      2. Assert: `False` (PyTorch is CUDA, not ROCm)
    Expected Result: `False None`
    Evidence: .sisyphus/evidence/task-T6-rocm-backend.txt
  ```

  **Commit**: YES
  - Message: `feat(backends): add ROCMBackend (AMD via ROCm)`
  - Files: `src/img2svg/backends/rocm.py`, `tests/test_backends/test_rocm.py`
  - Pre-commit: `uv run pytest tests/test_backends/test_rocm.py -q`

- [ ] 7. **MPSBackend (Apple Silicon)**

  **What to do**:
  - Create `src/img2svg/backends/mps.py` with `class MPSBackend:`.
  - Methods:
    - `is_available() -> bool` — `torch.backends.mps.is_available()` (only if `hasattr(torch.backends, "mps")` else False)
    - `device_count() -> int` — 1 if available, else 0
    - `device_name(0) -> str` — `platform.processor()` or "Apple Silicon"
    - `total_memory_mb(0) -> int` — system RAM via `psutil` or `os.sysconf` (MPS shares system RAM, like an iGPU)
    - `free_memory_mb(0) -> int` — same source
    - `vendor() -> GpuVendor.APPLE`
    - `to_ultralytics_string(0) -> "mps"`
    - `warmup() -> None`
  - Lazy-import torch.
  - Tests: `tests/test_backends/test_mps.py` with mocked `torch.backends.mps`.

  **Must NOT do**: Don't fail on non-Mac systems (return `False` for `is_available`).

  **Recommended Agent Profile**:
  - **Category**: `quick` — single backend impl + tests
  - **Skills**: none

  **Parallelization**: Wave 2, parallel with T5/T6.

  **References**:
  - `src/img2svg/device.py:67-76` — current `auto` resolution for MPS (already there)
  - `src/img2svg/backends/cuda.py` (T5) — pattern to follow

  **Acceptance Criteria**:
  - [ ] `MPSBackend().is_available()` returns `False` on this Linux system
  - [ ] With a mocked `torch.backends.mps.is_available() -> True`, `is_available()` returns `True`
  - [ ] `MPSBackend().vendor() == GpuVendor.APPLE`
  - [ ] `MPSBackend().to_ultralytics_string(0) == "mps"`
  - [ ] Tests pass

  **QA Scenarios**:
  ```
  Scenario: MPSBackend is_available on Linux
    Tool: Bash (uv run python)
    Preconditions: Linux system
    Steps:
      1. Run: `uv run python -c "from img2svg.backends.mps import MPSBackend; print(MPSBackend().is_available())"`
      2. Assert: `False`
    Expected Result: `False`
    Evidence: .sisyphus/evidence/task-T7-mps-backend.txt
  ```

  **Commit**: YES
  - Message: `feat(backends): add MPSBackend (Apple Silicon)`
  - Files: `src/img2svg/backends/mps.py`, `tests/test_backends/test_mps.py`
  - Pre-commit: `uv run pytest tests/test_backends/test_mps.py -q`

- [ ] 8. **BackendRegistry with auto-detection**

  **Note**: T4 subagent did NOT create the registry stub (deferred to T8). T8 will create the registry from scratch AND wire all 4 backends.

  **What to do**:
  - In `src/img2svg/backends/registry.py`, implement `class BackendRegistry:` (or extend the stub from T4).
  - Methods:
    - `available() -> list[DeviceBackend]` — return backends in priority order: CUDA, ROCM, MPS, CPU
    - `detect() -> DeviceBackend` — return the first available backend
    - `resolve(spec: BackendSpec) -> DeviceBackend` — pick the backend matching `spec.requested`; raise `DeviceUnavailableError` if not available
    - `for_device_string(s: str) -> BackendSpec` — parse legacy `"cuda:0"`, `"cpu"`, `"mps"`, `"rocm"`, `"auto"` strings
  - Singletons: `CUDA_BACKEND`, `ROCM_BACKEND`, `MPS_BACKEND`, `CPU_BACKEND` (already created in T3, T5, T6, T7).
  - Tests: `tests/test_backends/test_registry.py` — mock each backend, verify detection order, error handling.

  **Must NOT do**: Don't add new dependencies. Use stdlib.

  **Recommended Agent Profile**:
  - **Category**: `quick` — registry + tests
  - **Skills**: none

  **Parallelization**: Wave 2, depends on T5/T6/T7.

  **References**:
  - `src/img2svg/gpu.py:147-169` — `list_gpus()` and `recommend_gpu()` pattern (priority order)
  - `src/img2svg/errors.py:82` — `DeviceUnavailableError` exists

  **Acceptance Criteria**:
  - [ ] `BackendRegistry.detect()` returns `CUDA_BACKEND` on this system (CUDA available)
  - [ ] `BackendRegistry.resolve(BackendSpec(requested="cpu"))` returns `CPU_BACKEND`
  - [ ] `BackendRegistry.resolve(BackendSpec(requested="nonexistent"))` raises `DeviceUnavailableError`
  - [ ] `BackendRegistry.for_device_string("cuda:0")` returns `BackendSpec(requested="cuda", index=0)`
  - [ ] Tests pass

  **QA Scenarios**:
  ```
  Scenario: BackendRegistry detects CUDA on this system
    Tool: Bash (uv run python)
    Preconditions: this system has NVIDIA
    Steps:
      1. Run: `uv run python -c "from img2svg.backends import BackendRegistry; b = BackendRegistry.detect(); print(b.type(), b.vendor())"`
      2. Assert: type is `cuda`, vendor is `nvidia`
    Expected Result: `cuda nvidia` (or `BackendType.CUDA GpuVendor.NVIDIA` enum repr)
    Evidence: .sisyphus/evidence/task-T8-registry-detect.txt
  ```

  **Commit**: YES
  - Message: `feat(backends): add BackendRegistry with auto-detection`
  - Files: `src/img2svg/backends/registry.py`, `tests/test_backends/test_registry.py`
  - Pre-commit: `uv run pytest tests/test_backends/test_registry.py -q`

- [ ] 9. **Refactor `detector.py` to use BackendSpec**

  **What to do**:
  - In `src/img2svg/detector.py`:
    - `YOLODetector.__init__(self, model_name: str, backend: BackendSpec)` — accept `BackendSpec` instead of `device_str: str`
    - Internally: `self._backend = BackendRegistry.resolve(backend)` (raises `DeviceUnavailableError` on failure, wrap as `ModelLoadError`)
    - `self.device = self._backend.to_ultralytics_string(0)` (or the requested index)
    - Cache key uses `backend.requested` + `index` (not raw string)
    - The `"rocm"` translation problem is now solved by `ROCMBackend.to_ultralytics_string(0) -> "cuda:0"`.
  - Update `tests/test_detector.py` to use `BackendSpec`.

  **Must NOT do**: Don't change `YOLODetector.detect()` signature.

  **Recommended Agent Profile**:
  - **Category**: `quick` — refactor + test updates
  - **Skills**: none

  **Parallelization**: Wave 3, depends on T8.

  **References**:
  - `src/img2svg/detector.py:1-104` — full file
  - `src/img2svg/detector.py:81` — the YOLO call site (where `"rocm"` would break)

  **Acceptance Criteria**:
  - [ ] `YOLODetector("yolo11x.pt", BackendSpec(requested="cpu"))` works
  - [ ] `YOLODetector("yolo11x.pt", BackendSpec(requested="nonexistent"))` raises `ModelLoadError`
  - [ ] `tests/test_detector.py` passes
  - [ ] `src.img2svg.detector` still passes the right string to ultralytics (not `"rocm"`)

  **QA Scenarios**:
  ```
  Scenario: detector uses BackendSpec
    Tool: Bash (uv run pytest)
    Preconditions: refactor done
    Steps:
      1. Run: `uv run pytest tests/test_detector.py -v`
      2. Assert: all tests pass
    Expected Result: all passed
    Evidence: .sisyphus/evidence/task-T9-detector-tests.txt
  ```

  **Commit**: YES
  - Message: `refactor(detector): use BackendSpec instead of device_str`
  - Files: `src/img2svg/detector.py`, `tests/test_detector.py`
  - Pre-commit: `uv run pytest tests/test_detector.py -q`

- [ ] 10. **Refactor `pipeline.py` (cache key + sidecar)**

  **What to do**:
  - In `src/img2svg/pipeline.py`:
    - Change `Pipeline.__init__` to accept `backend: BackendSpec` (or read from `options.backend`)
    - Detector cache key: use `backend.requested` + `index` (not raw user string)
    - Sidecar: record `backend_requested`, `backend_resolved` fields (deprecate `device` field with warning)
  - Update `tests/test_pipeline.py` to use `BackendSpec` in fixtures.

  **Must NOT do**: Don't change `Pipeline.run()` signature.

  **Recommended Agent Profile**:
  - **Category**: `quick` — refactor + test updates
  - **Skills**: none

  **Parallelization**: Wave 3, depends on T8.

  **References**:
  - `src/img2svg/pipeline.py:140-195` — pipeline flow
  - `src/img2svg/detector.py:27` — current cache key uses raw user string (latent bug)

  **Acceptance Criteria**:
  - [ ] `Pipeline(ConversionOptions(backend=BackendSpec(requested="cpu"))).run(...)` works
  - [ ] `Sidecar.backend_resolved` is populated correctly
  - [ ] `tests/test_pipeline.py` passes (including the 3 from T35)

  **QA Scenarios**:
  ```
  Scenario: pipeline uses BackendSpec
    Tool: Bash (uv run pytest)
    Preconditions: refactor done
    Steps:
      1. Run: `uv run pytest tests/test_pipeline.py -v`
      2. Assert: all tests pass
    Expected Result: all passed
    Evidence: .sisyphus/evidence/task-T10-pipeline-tests.txt
  ```

  **Commit**: YES
  - Message: `refactor(pipeline): use BackendSpec for cache key + sidecar`
  - Files: `src/img2svg/pipeline.py`, `tests/test_pipeline.py`
  - Pre-commit: `uv run pytest tests/test_pipeline.py -q`

- [ ] 11. **Refactor `gpu.py` (`_torch_fallback` vendor detection)**

  **What to do**:
  - In `src/img2svg/gpu.py`:
    - `_torch_fallback()` line 458: replace hardcoded `vendor = GpuVendor.NVIDIA` with `vendor = GpuVendor.AMD if torch.version.hip else GpuVendor.NVIDIA`.
    - Also use `device_name` to detect RDNA iGPUs by name prefix ("Radeon") as a tiebreaker (for systems where `torch.version.hip` is None but the GPU is AMD).
  - Tests: `tests/test_gpu.py` — add a test for ROCm-style torch fallback.

  **Must NOT do**: Don't refactor the rest of `gpu.py` (T33-T35 are done).

  **Recommended Agent Profile**:
  - **Category**: `quick` — single bug fix + test
  - **Skills**: none

  **Parallelization**: Wave 3, depends on T8.

  **References**:
  - `src/img2svg/gpu.py:458` — the line with the hardcoded vendor
  - `src/img2svg/enums.py` — `GpuVendor` enum

  **Acceptance Criteria**:
  - [ ] `uv run python -m img2svg list-gpus` still shows NVIDIA RTX 5070 + AMD Radeon 890M
  - [ ] With `torch.version.hip` mocked to a string, the fallback labels as AMD
  - [ ] Tests pass

  **QA Scenarios**:
  ```
  Scenario: torch_fallback uses correct vendor
    Tool: Bash (uv run python)
    Preconditions: this system has both
    Steps:
      1. Run: `uv run python -m img2svg list-gpus`
      2. Assert: both NVIDIA and AMD entries present
    Expected Result: 2 rows in table, NVIDIA + AMD
    Evidence: .sisyphus/evidence/task-T11-gpu-fallback.txt
  ```

  **Commit**: YES
  - Message: `fix(gpu): _torch_fallback uses torch.version.hip for vendor`
  - Files: `src/img2svg/gpu.py`, `tests/test_gpu.py`
  - Pre-commit: `uv run pytest tests/test_gpu.py -q`

- [ ] 12. **Refactor `cli.py` (--device help + info output)**

  **What to do**:
  - In `src/img2svg/cli.py`:
    - Update `--device` help text: "auto, cpu, cuda, cuda:N, mps, rocm (requires ROCm PyTorch)"
    - The `info` subcommand should show: "Backend: CUDA (torch 2.12.0+cu130)" — pulls from `BackendRegistry.detect()` and `torch.__version__`
    - Validate `--device` against the registry (deprecation warning for "rocm" if CUDA build is loaded)
  - Tests: `tests/test_cli.py` — add assertions on the new info output.

  **Must NOT do**: Don't break the existing CLI surface.

  **Recommended Agent Profile**:
  - **Category**: `quick` — refactor + test updates
  - **Skills**: none

  **Parallelization**: Wave 3, depends on T8.

  **References**:
  - `src/img2svg/cli.py:283` — `--device` typer option
  - `src/img2svg/cli.py:438` — `info` subcommand

  **Acceptance Criteria**:
  - [ ] `uv run python -m img2svg info` shows "Backend: CUDA" on this system
  - [ ] `uv run python -m img2svg --help` shows the updated --device help text
  - [ ] `tests/test_cli.py` passes

  **QA Scenarios**:
  ```
  Scenario: info shows backend
    Tool: Bash (uv run python)
    Preconditions: refactor done
    Steps:
      1. Run: `uv run python -m img2svg info`
      2. Assert: output contains "Backend:" and the active backend type
    Expected Result: includes backend info
    Evidence: .sisyphus/evidence/task-T12-cli-info.txt
  ```

  **Commit**: YES
  - Message: `feat(cli): info shows active backend type`
  - Files: `src/img2svg/cli.py`, `tests/test_cli.py`
  - Pre-commit: `uv run pytest tests/test_cli.py -q`

- [ ] 13. **Smart install script + pyproject extras**

  **What to do**:
  - Create `scripts/install_backend.sh` (executable):
    1. Detect GPU via `lspci` (already proven to work for AMD + NVIDIA)
    2. If NVIDIA found: print `pip install img2svg[nvidia]` command
    3. If AMD found: warn that 512 MB iGPU cannot run YOLO11x, then print `pip install img2svg[amd]` command
    4. If Apple: print `pip install img2svg[apple]`
    5. If no GPU: print `pip install img2svg[cpu]`
    6. Run `--dry-run` by default (don't actually install); require `--apply` to install
    7. Verify after install: `uv run python -c "from img2svg.backends import BackendRegistry; print(BackendRegistry.detect())"`
  - In `pyproject.toml`:
    ```toml
    [project.optional-dependencies]
    nvidia = ["torch>=2.0,<3"]
    amd = ["torch>=2.0,<3"]  # Will be replaced by repo.radeon.com index
    apple = ["torch>=2.0,<3"]
    cpu = ["torch>=2.0,<3"]
    ```
    And add a `[tool.uv] extra-index-url` per vendor (or document in the script).
  - Pin `ultralytics>=8.4.0` (for MPS coordinate fix).

  **Must NOT do**: Don't add the `index-url` magic in `pyproject.toml` (uv doesn't honor it for extras); document it in the script output.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — script + pyproject + verification
  - **Skills**: none

  **Parallelization**: Wave 4, depends on T12.

  **References**:
  - `src/img2svg/gpu.py:_parse_lspci()` — existing lspci parser (Python). Shell script can re-implement or call `uv run python -c "from img2svg.gpu import list_gpus; print(list_gpus())"`.
  - `pyproject.toml:38` — current torch pin
  - `pyproject.toml:148-150` — current `dev` extras pattern

  **Acceptance Criteria**:
  - [ ] `bash scripts/install_backend.sh --dry-run` prints the right install command for this system
  - [ ] `bash scripts/install_backend.sh --dry-run` exits 0
  - [ ] `pip install img2svg[nvidia]` works (existing test environment)
  - [ ] `pyproject.toml` validates as TOML

  **QA Scenarios**:
  ```
  Scenario: install script detects vendor
    Tool: Bash
    Preconditions: this system has NVIDIA + AMD
    Steps:
      1. Run: `bash scripts/install_backend.sh --dry-run`
      2. Assert: output mentions NVIDIA and AMD, prints install command
    Expected Result: "Detected: NVIDIA + AMD. Recommended: pip install img2svg[nvidia] (AMD iGPU too small for YOLO11x)"
    Evidence: .sisyphus/evidence/task-T13-install-dry-run.txt
  ```

  **Commit**: YES
  - Message: `feat(install): add install_backend.sh + pyproject extras for nvidia/amd/apple/cpu`
  - Files: `scripts/install_backend.sh`, `pyproject.toml`
  - Pre-commit: `bash scripts/install_backend.sh --dry-run && uv run pytest -q`

- [ ] 14. **Documentation updates**

  **What to do**:
  - Update `docs/installation.md`:
    - Replace single install section with 4 sections: NVIDIA, AMD, Apple, CPU
    - Each section: "what you have" → "what to install" → "how to verify"
    - AMD section: warn about 512 MB iGPU not being usable for YOLO11x
  - Update `README.md`:
    - Add "What GPU do you have?" decision tree in the install section
    - Link to `scripts/install_backend.sh`
    - Add a small "Backend architecture" section explaining the 4 backends
  - Update `docs/api.md`:
    - Document `BackendSpec` Pydantic model
    - Document the new `backend=` parameter on `ConversionOptions`
  - Add `docs/backends.md`:
    - Explain the DeviceBackend Protocol
    - Show how to add a new backend (for future contributors)

  **Must NOT do**: Don't change unrelated docs sections.

  **Recommended Agent Profile**:
  - **Category**: `writing` — docs
  - **Skills**: none

  **Parallelization**: Wave 4, parallel with T13/T15.

  **References**:
  - `docs/installation.md` — current file
  - `README.md` — current file
  - `docs/api.md` — current file

  **Acceptance Criteria**:
  - [ ] `docs/installation.md` has 4 vendor sections
  - [ ] `README.md` has GPU decision tree
  - [ ] `docs/backends.md` exists with Protocol explanation
  - [ ] `docs/api.md` documents `BackendSpec`
  - [ ] `mkdocs build` succeeds (if applicable)

  **QA Scenarios**:
  ```
  Scenario: docs build
    Tool: Bash (uv run mkdocs)
    Preconditions: docs updated
    Steps:
      1. Run: `uv run mkdocs build --strict 2>&1 | tail -10`
      2. Assert: no broken links, no warnings
    Expected Result: build succeeds
    Evidence: .sisyphus/evidence/task-T14-docs-build.txt
  ```

  **Commit**: YES
  - Message: `docs: split installation.md by vendor, add backends.md, update README`
  - Files: `docs/installation.md`, `docs/backends.md`, `docs/api.md`, `README.md`
  - Pre-commit: `uv run mkdocs build --strict`

- [ ] 15. **Jenkinsfile matrix stages**

  **What to do**:
  - Update `Jenkinsfile` to add a `matrix` block for backend testing:
    - Axis: `BACKEND` ∈ {`cpu`, `nvidia`, `amd`, `apple`}
    - Skip rules: `apple` is no-op on Linux agents; `amd` is no-op on agents without AMD hardware
    - Each stage: install matching `img2svg[$BACKEND]` extra, run `pytest -m "not slow"`
    - CPU stage: always runs
    - NVIDIA stage: requires label `nvidia-gpu`
    - AMD stage: requires label `amd-gpu`
    - Apple stage: requires label `macos`
  - Add a `verify-backend.sh` script that each stage calls:
    - `uv run python -c "from img2svg.backends import BackendRegistry; print(BackendRegistry.detect())"`
    - If the expected backend isn't detected, fail the stage

  **Must NOT do**: Don't add new Jenkins plugins (use existing `matrix-project`).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — Jenkinsfile + script
  - **Skills**: none

  **Parallelization**: Wave 4, parallel with T13/T14.

  **References**:
  - `Jenkinsfile:1-79` — current file
  - `scripts/ci.sh:1-74` — current CI script

  **Acceptance Criteria**:
  - [ ] `Jenkinsfile` parses (Jenkins Linter or `jenkins-cli` if available)
  - [ ] Matrix stages defined for cpu, nvidia, amd, apple
  - [ ] CPU stage works on this Linux system (run locally to verify the script)
  - [ ] `verify-backend.sh` correctly detects the expected backend

  **QA Scenarios**:
  ```
  Scenario: jenkinsfile parses
    Tool: Bash (jenkins-cli or local lint)
    Preconditions: Jenkinsfile updated
    Steps:
      1. Run: `bash scripts/ci.sh 2>&1 | tail -20` (existing CI script as smoke test)
      2. Assert: exit 0
    Expected Result: ci.sh runs cleanly
    Evidence: .sisyphus/evidence/task-T15-jenkinsfile.txt
  ```

  **Commit**: YES
  - Message: `ci: add Jenkinsfile matrix stages for backend testing`
  - Files: `Jenkinsfile`, `scripts/ci.sh`, `scripts/verify_backend.sh`
  - Pre-commit: `bash scripts/ci.sh`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE.

- [ ] F1. **Plan Compliance Audit** — `oracle`
- [ ] F2. **Code Quality Review** — `unspecified-high`
- [ ] F3. **Real Manual QA** — `unspecified-high`
- [ ] F4. **Scope Fidelity Check** — `deep`

---

## Commit Strategy

- **Wave 1**: `feat(backends): add DeviceBackend Protocol + BackendSpec + CPUBackend` (T1-T4)
- **Wave 2**: `feat(backends): add CUDA, ROCm, MPS backends + auto-detect registry` (T5-T8)
- **Wave 3**: `refactor: migrate device.py, detector.py, pipeline.py, gpu.py, cli.py to BackendSpec` (T9-T12)
- **Wave 4**: `feat(install): smart install_backend.sh + pyproject extras + Jenkins matrix + docs` (T13-T15)
- **Final**: `release: multi-vendor GPU support v0.2.0` — tagged, on `main`

---

## Success Criteria

### Verification Commands
```bash
uv run pytest -m "not slow" -q                          # Expected: 345+ passed
uv run python -m img2svg list-gpus                     # Expected: shows NVIDIA + AMD
uv run python -m img2svg info                          # Expected: shows Backend type
bash scripts/install_backend.sh --dry-run              # Expected: detects vendor, prints plan
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Hands-on: `img2svg list-gpus` shows both NVIDIA and AMD GPUs
- [ ] Hands-on: `img2svg info` shows active backend + torch build
- [ ] Install script correctly identifies vendor on this system (should detect NVIDIA + AMD)
