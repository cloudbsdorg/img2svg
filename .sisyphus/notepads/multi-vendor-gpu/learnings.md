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
