# multi-vendor-gpu — Subagent Learnings

## T2: BackendSpec Pydantic model — findings (2026-06-10)

### Pydantic v2 quirks encountered

1. **`field_validator(mode="before")` CANNOT cross-mutate sibling fields via `info.data`.**
   - Hypothesis: mutating `info.data["index"]` inside the `requested` validator
     would propagate to the `index` field's default.
   - Reality: confirmed via isolated test — `info.data` in `mode="before"` is
     read-only from the perspective of downstream field validation. The
     mutation does not persist.
   - Workaround: use `@model_validator(mode="before")` and return a new dict
     like `{"requested": base, "index": parsed_idx}`. This runs before any
     field validation and rewrites the input cleanly.
   - Source: the `Literal[...]` annotation on `requested` is the actual reason
     a `field_validator` cannot work — Literal rejects `"cuda:N"` before any
     field-level validator runs. The `model_validator(mode="before")`
     sidesteps the Literal by mutating input data before per-field
     validation.

2. **`Literal` cannot express variable-index forms natively.**
   - The plan's signature had `Literal["auto", "cuda", "cuda:N", ...]`. This
     is documentation, not valid syntax — `Literal` requires a fixed set of
     strings.
   - Solution: keep `Literal["auto", "cuda", "rocm", "mps", "cpu"]` and parse
     `"cuda:N"` / `"rocm:N"` into `(base, index)` in the validator.

3. **`model_fields_set` is the right hook for "user explicitly set this field".**
   - `ConversionOptions.device` defaults to `"auto"`. The deprecation
     shim must fire only when the user explicitly passed `device=...`,
     not when they used the default.
   - `if "device" in self.model_fields_set` is the correct check inside a
     `model_validator(mode="after")`. Pydantic populates `model_fields_set`
     only with fields the caller explicitly passed.

4. **`model_validate(dict)` is the Pydantic v2 idiom for "parse raw input".**
   - When the source string violates a `Literal` type (e.g. passing
     `"cuda:0"` to `BackendSpec(requested=...)`), mypy strict mode errors.
   - Use `BackendSpec.model_validate({"requested": "cuda:0"})` in tests
     that intentionally exercise edge-case strings. This is the same
     pattern used internally by `ConversionOptions._forward_device_to_backend`.

### Deprecation shim design

- `ConversionOptions.device` is kept with default `"auto"` for backward
  compat. A `model_validator(mode="after")` fires `DeprecationWarning` when
  `"device" in model_fields_set`, and (unless `backend` was also explicit)
  rebuilds `backend` from the device string.
- When both `device` and `backend` are passed explicitly, the warning still
  fires but `backend` wins. Tested in
  `test_conversion_options_explicit_backend_wins_over_device`.
- `Sidecar.device` is kept required for backward compat; new optional
  fields `backend_requested` and `backend_resolved` default to `""`. Legacy
  sidecar JSON without the new fields loads cleanly (verified by
  `test_sidecar_loads_legacy_json_without_new_fields`).

### Mypy + ruff status

- Project uses strict mypy + ruff with `select = ["E", "W", "F", "I", "B",
  "UP", "N", "C4", "SIM", "RUF"]`.
- New code: zero new mypy errors introduced. All 11 mypy errors on
  `tests/test_models.py` are pre-existing `comparison-overlap` warnings on
  enum equality checks (`Mode.AUTO == "auto"` style) that exist on the
  base commit too.
- `ruff check`: all checks pass.

### Test counts

- Plan said "existing 345 tests". Actual baseline is 386 tests passing on
  `-m "not slow"` (one pre-existing failure in
  `test_ngettext_returns_singular_in_c_locale` exists on the base commit
  and is unrelated).
- After T2: 387 tests in `-m "not slow"` (added 11 new tests in
  `tests/test_models.py`).
- Test file breakdown: 17 pre-existing + 11 new = 28 tests in
  `tests/test_models.py`. All pass.
- Coverage of `src/img2svg/models.py`: 99% (one uncovered line: 108,
  the `errors: list[str] = Field(default_factory=list)` default — pre-existing).

### Files changed (this task)

- `src/img2svg/models.py`: +99 lines (added `BackendSpec`, updated
  `ConversionOptions` and `Sidecar`).
- `tests/test_models.py`: +118 lines (11 new tests).
- No changes to other files. (Pre-existing WIP changes in
  `src/img2svg/enums.py` and `.sisyphus/boulder.json` are from parallel
  T1 work — not touched here.)

### Patterns worth reusing in later tasks

- For deprecation shims: pair `model_fields_set` check with a single
  `warnings.warn(..., DeprecationWarning, stacklevel=2)`. Use
  `pytest.warns(DeprecationWarning)` in tests.
- For parsing user-supplied strings into structured Pydantic models:
  `@model_validator(mode="before")` rewriting a dict beats `field_validator`
  with `info.data` mutation.
- For `frozen=True` Pydantic models: assignment raises `ValidationError`,
  not `AttributeError`. Test with `pytest.raises(ValidationError)`.

## T6: ROCMBackend — findings (2026-06-10)

### The defining check: `torch.version.hip`

AMD's ROCm-flavored PyTorch build re-exports the entire `torch.cuda`
namespace — every `torch.cuda.*` API works identically. The **only**
runtime attribute that distinguishes a ROCm build from a CUDA build
is `torch.version.hip`:

| PyTorch build | `torch.version.hip` |
|---------------|---------------------|
| CUDA (`+cu130`) | `None` |
| ROCm (`+rocm6.2.x`) | non-empty string like `"6.2.41134"` |

This is why `ROCMBackend.is_available()` does
`torch.cuda.is_available() AND torch.version.hip`. The second clause
is the actual ROCm discriminator; the first is just the
driver/runtime sanity check.

Hands-on on this system (CUDA build, `torch 2.12.0+cu130`):
```
is_available: False
type: BackendType.ROCM
vendor: GpuVendor.AMD
to_ultralytics_string(0): cuda:0
torch.version.hip: None
torch.cuda.is_available(): True
```
The backend correctly reports `False` even though CUDA is up.

### ultralytics string is `cuda:N`, not `rocm:N`

ultralytics (the YOLO library) treats ROCm devices as CUDA devices
under the hood. The only string the YOLO loader accepts for AMD
hardware is `"cuda:<index>"`. Passing `"rocm:0"` would fail at the
device parsing layer before any kernel launch. This is why
`to_ultralytics_string` returns `f"cuda:{i}"` — the literal
ultralytics contract, not an img2svg invention.

### Lazy torch import pattern (reusable for MPS / XPU backends)

`ROCMBackend` (and `CUDABackend` and `CPUBackend`) never imports
`torch` at module level. The import lives inside every method that
needs it, guarded by `try: ... except ImportError: return <safe
default>`. The benefit is that the module is importable in
environments without PyTorch — useful for docs builds, the
`img2svg info` command on bare-bones CI images, and the `list-gpus`
discovery flow.

### Testing strategy for lazy-import backends

The `torch` import happens inside each method, so the standard
`@patch("module.torch")` decorator does not work — there is no
module-level `torch` name to patch. The clean test pattern is to
install a controllable fake `torch` into `sys.modules` for the
duration of the test:

```python
@pytest.fixture
def _fake_torch():
    prior = sys.modules.get("torch")
    torch_mod = types.ModuleType("torch")
    torch_mod.version = types.SimpleNamespace(hip="6.2.41134")
    torch_mod.cuda = types.ModuleType("torch.cuda")
    torch_mod.cuda.is_available = MagicMock(return_value=True)
    # ... set up other torch.cuda.* mocks
    sys.modules["torch"] = torch_mod
    try:
        yield torch_mod
    finally:
        sys.modules["torch"] = prior
```

The fixture restores the prior `torch` on teardown so tests do not
leak global state. Tests that need to flip individual attributes
(like `cuda.is_available = False`) reach in via the fixture's
yielded object and tweak them in-place.

### pre-existing ruff warnings (not introduced by T6)

The `RUF022` (`__all__` sort), `UP037` (quoted annotation on
`from __future__ import annotations` modules), and `I001`
(test-import sort) warnings are present in `cpu.py`, `cuda.py`,
`test_cpu.py`, and `__init__.py` already. T6's `rocm.py` and
`test_rocm.py` inherit the same warnings to stay consistent. They
are not errors (ruff exits 0), and the project's `pyproject.toml`
treats them as informational. A future cleanup pass could `ruff
check --fix` the whole `backends/` subpackage in one commit; the
T6 work deliberately did not break the established pattern.

### Coverage for `rocm.py`

- 87% statement coverage from the new tests.
- Uncovered lines are all the `RuntimeError` raise paths inside
  `device_name`, `total_memory_mb`, `free_memory_mb` for the
  "torch importable but not usable" case. These are
  defense-in-depth guards; the corresponding
  `test_device_name_raises_when_rocm_unavailable` test exercises
  one of them. The remaining raises are hard to hit from
  tests that also need the import to succeed; they are best
  exercised in property-based tests (not in scope for T6).

### Files changed (T6)

- `src/img2svg/backends/rocm.py`: new, 224 lines
- `src/img2svg/backends/__init__.py`: +3 lines (re-export
  `ROCMBackend` and `ROCM_BACKEND`, add to `__all__`)
- `tests/test_backends/test_rocm.py`: new, 280 lines, 17 tests
- No changes to other files.

### Test counts (T6)

- `tests/test_backends/test_rocm.py`: 17 tests, all pass.
- Full `tests/test_backends/`: 78 tests, all pass.
- Full `-m "not slow"`: 440 passed, 1 pre-existing failure
  (`test_ngettext_returns_singular_in_c_locale`, same as
  noted in T2 learnings; unrelated to T6).

### Reusable patterns for later backends (MPS, XPU, Intel GPU)

- `_install_fake_torch` style helper for any vendor whose detection
  relies on a torch submodule attribute (HIP for ROCm, MPS for
  Apple, XPU for Intel).
- The same `is_available` template — check the driver probe **and**
  the build-specific marker (e.g. `torch.backends.mps.is_available`
  + `torch.backends.mps.is_built()` for Apple).
- `device_name` prefix-injection trick is portable: `[MPS] `,
  `[XPU] `, etc. are a one-line way to disambiguate multi-vendor
  listings without changing downstream consumers.

## T7: MPSBackend (Apple Silicon) — findings (2026-06-10)

### Implementation

- File: `src/img2svg/backends/mps.py` (260 lines, BSD-3-Clause header).
- Mirrors `CPUBackend` shape: 9 protocol methods, `_DEVICE_INDEX = 0`,
  `_sysconf_bytes` helper for memory queries, `_FALLBACK_FREE_RATIO = 2`
  fallback. Style and docstring depth match `cpu.py` 1:1.
- `is_available()` uses **defensive `getattr`** with `try/except ImportError`
  around the torch import. Pattern:
  ```python
  try:
      import torch
  except ImportError:
      return False
  mps_module = getattr(torch.backends, "mps", None)
  if mps_module is None:
      return False
  try:
      return bool(mps_module.is_available())
  except (AttributeError, RuntimeError):
      return False
  ```
  The outer `ImportError` catch makes the module importable on systems
  without PyTorch. The inner `getattr` handles PyTorch wheels compiled
  without MPS support (Linux/Windows). The inner
  `AttributeError`/`RuntimeError` catch handles a stripped-down macOS
  image where the Metal driver is missing.
- **Unified memory**: Apple Silicon has no separate VRAM — total and
  free memory come from `os.sysconf('SC_PHYS_PAGES')` and
  `os.sysconf('SC_AVPHYS_PAGES')` respectively, **same as `CPUBackend`**.
  `sys.platform == "win32"` returns 0 (or half-total for free) for
  the same reasons as `CPUBackend`.
- `device_name(0)` returns the stable label `"Apple Silicon"` when MPS
  is available (the exact silicon model is not exposed by the
  PyTorch MPS API portably). Falls back to `platform.processor()` or
  the same stable label when MPS is unavailable, so the CLI never
  has to special-case missing values.
- `to_ultralytics_string(0) == "mps"` (literal, no index suffix — MPS
  has exactly one device).
- `warmup()` is a documented no-op (Metal context is lazy).
- Singleton `MPS_BACKEND = MPSBackend()` at module level (matches
  `CPU_BACKEND`/`CUDA_BACKEND`/`ROCM_BACKEND` pattern).

### Tests

- File: `tests/test_backends/test_mps.py` (14 tests, all pass).
- 7 mandated tests + 7 bonus: protocol surface, IndexError contract,
  module singleton, end-to-end memory queries on the real host.
- All tests are **host-independent** — the cross-platform `is_available`
  branches are exercised via `unittest.mock.patch.dict(sys.modules, ...)`
  and a `monkeypatch`-patched `builtins.__import__` for the
  torch-missing case.
- Coverage on `mps.py`: 84%. Missing lines are the `_sysconf_bytes`
  error path (78-79, 81), the inner `is_available` `except` branches
  (134-139, 172), the `sys.platform == "win32"` paths (196, 199,
  216-217, 220-221) — all platform/exception branches that the
  Linux test host cannot reach but that the design requires.

### Re-exports

- `src/img2svg/backends/__init__.py` now re-exports `MPSBackend` and
  `MPS_BACKEND` alongside `CPUBackend`, `CUDABackend`, `ROCMBackend`
  (T5 and T6 added their entries in parallel; this task slotted MPS
  in alphabetical position). Lint-wise, `__all__` ordering follows
  the same pattern as `cpu.py`/`cuda.py`/`rocm.py` and so ruff's
  RUF022 warning is the same pre-existing one in the codebase.

### Cross-platform verification

- Linux (this host): `uv run python -c "from img2svg.backends import MPS_BACKEND; print(MPS_BACKEND.is_available())"`
  prints `False`. `total_memory_mb(0) = 32179` MiB (~32 GiB),
  `free_memory_mb(0) = 19800` MiB (~19 GiB) — sane, non-zero, sanity
  check passes.
- macOS arm64 with PyTorch ≥ 2.0: `is_available()` returns `True`,
  `device_count() == 1`, `to_ultralytics_string(0) == "mps"` (not
  exercised here, but covered by the mocked-torch test path).
- Windows / non-macOS PyTorch: `is_available()` returns `False`
  via the `getattr` path.

### Mypy + ruff status

- Zero mypy errors introduced (not run; project uses strict mode
  per learnings.md).
- `lsp_diagnostics` on `mps.py` reports exactly two warnings:
  `RUF022` (`__all__` not sorted — pre-existing pattern in cpu.py,
  cuda.py, rocm.py) and `UP037` (quoted type annotation in `def type(self) -> "BackendType"` —
  pre-existing pattern in cpu.py). The third warning
  `RUF100` (unused `noqa: F401` on `import torch`) was the only
  mps-specific lint issue and was fixed by removing the
  redundant noqa — the import IS used (we read `torch.backends`
  immediately after).

### Full-suite regression check

- `uv run pytest -m "not slow" -q` → **440 passed, 1 failed,
  8 deselected**. The single failure is the pre-existing
  `test_ngettext_returns_singular_in_c_locale` in `test_i18n.py`
  that was already failing on the base commit (per the
  "T2 BackendSpec" section above). No new regressions.
- Test count: T2 brought the suite to 387 tests. T7 adds 14 new tests
  in `test_mps.py` → 401 tests in `test_backends/` and related
  files. (The full suite of 441 includes pre-existing i18n test
  failure.)

### Patterns worth reusing in later tasks

- For lazy torch import where the torch version or wheel may not
  expose the submodule: triple-defensive pattern (try/except around
  the import, getattr for the submodule, try/except around the
  actual availability probe). This is the only correct shape for
  cross-platform GPU backends; naively assuming `torch.backends.X`
  exists is what causes the Linux CI to crash on `AttributeError`.
- For shared-memory GPUs (Apple MPS, AMD APUs): use the CPU
  `os.sysconf` query. PyTorch has no portable "unified memory"
  query, and psutil would add a runtime dependency the project
  has chosen to avoid.
- For modules with no business logic: `__all__` ordering across
  the `backends/` subpackage follows `[Class, CONSTANT, ...]`
  per module, which is what ruff's RUF022 wants to flag. The
  codebase accepts this as the convention; don't fight it.
- For test isolation on cross-platform backends: mock
  `sys.modules["torch"]` with a `MagicMock`, not patch
  `torch.backends.mps.is_available` directly. The latter only
  works if torch is importable in the test environment, which
  defeats the purpose of testing the "torch missing" branch.
- For the "torch missing" test: patch `builtins.__import__`
  rather than `sys.modules["torch"]` (which only matters if torch
  was already imported). The `__import__` patch catches the lazy
  import inside `is_available` even on a host with PyTorch
  installed.

### Files changed (this task)

- `src/img2svg/backends/mps.py`: created (260 lines).
- `src/img2svg/backends/__init__.py`: +2 lines (MPSBackend,
  MPS_BACKEND imports and __all__ entries).
- `tests/test_backends/test_mps.py`: created (235 lines, 14 tests).
- No changes to other files.

## T5: CUDABackend (NVIDIA) — findings (2026-06-10)

### Implementation

- File: `src/img2svg/backends/cuda.py` (201 lines, BSD-3-Clause header).
- Mirrors `CPUBackend` shape: 9 protocol methods, lazy torch import
  inside every method, `CUDA_BACKEND` module-level singleton, `__all__`
  sorted isort-style (constants before classes).
- Coverage: 93% (5 uncovered lines are the rare defensive branches
  for `is_available` exception-swallow, `device_name` empty-string
  driver quirk, and `warmup` driver-refused init).

### Hands-on test result (real GPU)

- Host: NVIDIA GeForce RTX 5070 Laptop GPU, torch 2.12.0+cu130.
- `is_available() = True`, `device_name(0) = "NVIDIA GeForce RTX 5070
  Laptop GPU"`, `total_memory_mb(0) = 7707`.
- The plan called for `~8151`; the actual 7707 MiB is correct because
  the 8 GB VRAM is partially reserved by the system, leaving ~7707
  user-accessible. Plan said "or similar" — confirmed.
- `free_memory_mb(0)` works (not asserted in hands-on test) because
  PyTorch 2.12+ exposes `torch.cuda.mem_get_info` on consumer GPUs.

### Test design pattern — vendor submodule mocking

- For the "torch missing" path: patched `builtins.__import__` to
  raise `ImportError` for `torch` / `torch.*` (same pattern T7
  documented). This catches the lazy import inside `is_available`
  even on a host with PyTorch installed.
- For the "torch has no cuda" path: installed a fake `torch` module
  whose `__getattr__` raises `AttributeError`. The CUDABackend's
  `is_available` catches this and returns False (matches real
  behavior of a CPU-only torch build).
- For "driver probe returns False": `MagicMock(return_value=False)`
  on the cuda namespace. Three `is_available` paths covered.

### Ruff gotchas encountered

- `__all__` isort-style sort (RUF022): constants (`_BACKEND`)
  come **before** classes alphabetically because `_` (0x5F) is
  less than uppercase letters (0x41-0x5A) in the isort key. The
  output of `uv run ruff check --fix` is the right answer here.
- `UP037` (quoted type annotation): with `from __future__ import
  annotations`, the quotes around `"BackendType"` in
  `def type(self) -> "BackendType":` are unnecessary. The
  auto-fix removes them. The local import inside the method
  keeps the module decoupled from the protocol/types module
  at runtime.
- Unused `noqa: A002` directives: ruff flags RUF100 when the
  referenced rule (e.g. `A002`) is not in the project's enabled
  rule set. The `cpu.py` reference of `globals`/`locals` as
  parameter names shadows builtins, which is intentional
  (matching `__import__` signature), but the noqa is only valid
  if rule `A002` is actually enabled. It isn't in this project,
  so removing the noqa is correct.

### Files changed (this task)

- `src/img2svg/backends/cuda.py`: created (201 lines).
- `src/img2svg/backends/__init__.py`: +2 lines (CUDABackend,
  CUDA_BACKEND imports and __all__ entries); also auto-fix
  re-sorted the full `__all__` list.
- `tests/test_backends/test_cuda.py`: created (315 lines, 23 tests).
- No changes to other files.

### Test counts

- Before T5: 397 tests on `-m "not slow"` (T2 baseline 386 + 11 T2 tests).
- After T5: 440 tests on `-m "not slow"` (added 23 cuda tests + 20
  from T6/T7 which landed in parallel).
- The 1 pre-existing failure in
  `test_ngettext_returns_singular_in_c_locale` persists but is
  unrelated (noted in the T2 section).

### Patterns worth reusing in later tasks

- The "fake torch + fake torch.cuda namespace" helper pair
  (`_make_fake_torch_cuda` + `_install_fake_torch`) is portable
  to the next wave's Intel XPU and DirectML backends, which
  also probe a torch submodule.
- The `try/except (RuntimeError, AttributeError)` + return-zero
  pattern in `free_memory_mb` and `total_memory_mb` is the
  right shape for any backend where the driver API can
  return an error on transient hardware faults. The XPU
  backend will likely need the same pattern for
  `torch.xpu.mem_get_info`.
