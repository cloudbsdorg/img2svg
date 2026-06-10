# Learnings — img2svg

## T1: Scaffolding (2026-06-09)

- **Subagent timeout**: T1 delegated subagent got aborted at 10m 5s. Future delegations should use `quick` profile with very tight prompts (≤30 lines), or take work directly if the user is in a hurry.
- **pyproject.toml** (from subagent) is well-structured: hatchling backend, pinned deps including explicit `torch>=2.0,<3`, dev extras include `pytest-mock` (which Momus flagged as missing), `[tool.mypy]` overrides ignore missing imports for `ultralytics.*`, `torch.*`, `cv2.*`, `vtracer.*`, `supervision.*`. Console script `img2svg = "img2svg.cli:app"`.
- **`.sisyphus/` IS tracked** (user override of plan's `MUST NOT HAVE` guard). Plan needs a small patch later to reflect this.
- **`.idea/` IS tracked** (user override; PyCharm IDE files committed).
- **uv.lock IS gitignored** (per plan T1 spec; T28 will revisit).
- **Remote**: `git@github.com:cloudbsdorg/img2svg.git` exists with initial LICENSE-only commit (3cfa53b). After T1 we have 3cfa53b → 2c3956c (scaffold) → e490d15 (.idea).
- **Author**: `Mark LaPointe <mark@cloudbsd.org>` (mandatory per CloudBSD guideline surfaced via Honcho).
- **License**: BSD 3-Clause, `Copyright (c) 2026, CloudBSD`.
- **Platforms**: Linux (required), FreeBSD (best-effort), macOS (supported).
- **OS detection**: must use `uname -s`, never `platform.system()` (per CloudBSD guideline).
- **Package layout**: `src/img2svg/` with `man/` and `locale/` subdirs (already in hatchling `force-include`).

## T2: XDG paths (2026-06-10)

- T2 subagent completed in 59s (vs T1's 10m 5s abort). Faster because prompt was tightly scoped to 2 files.
- 6 expected functions (`config_dir`, `data_dir`, `cache_dir`, `model_cache_path`, `system_config_dir`, `load_config`) — no scope creep.
- Stdlib only (no new deps).
- **uv is busy when many subagents run in parallel** — `uv run pytest` and `uv run python` time out. Workaround: use plain `python3` for syntax/import checks, defer real test runs to a sequential window.
- Commit strategy: stage ONLY verified files (`git add src/img2svg/paths.py tests/test_paths.py` + notepad) — never `git add -A` when parallel subagents are still writing files.

## T16: Renderer base + LabelsRenderer (2026-06-10)

- `SVGDocument` / `SVGGroup` API surface for renderer use: `doc.add_group(id=..., **{"data-class": ..., "data-conf": ...})` → returns `SVGGroup`. Group helpers: `add_rect(x, y, w, h, fill=..., stroke=..., stroke_width=...)`, `add_text(x, y, text, font_size=..., fill=..., **attrs)`, `add_text_with_outline(x, y, text, fill="white", stroke="black", stroke_width=3.0)` — the outline helper sets `paint_order="stroke"` automatically.
- Background rect needs `stroke="none", stroke_width=None` to avoid the default black 1px border (helpers default to stroke="black" / stroke_width=1.0).
- `add_text_with_outline` defaults to `fill="white"` — perfect for label readability over varied image content.
- `LoadedImage` (dataclass in `loader.py`) exposes `width`, `height`, `np_array` (property), `pil_image`, `format`, `original_mode`, `has_alpha`. It does **not** store the source `path` — tasks that mention `.path` on `LoadedImage` are wrong; only `Path` of the input is known by the pipeline.
- `add_group(**attrs)` translates `data_class` → `data-class`, `class_` → `class` automatically (underscore→dash). Pass kwargs directly: `add_group(**{"data-class": "person", "data-conf": "0.87"})`.
- Pre-existing i18n test failure (`test_ngettext_returns_singular_in_c_locale`) is unrelated to T16 — confirmed by stashing my changes and re-running.
- For T19 (Pipeline): `RENDERER_REGISTRY: dict[Mode, type[Renderer]]` should map `Mode.LABELS → LabelsRenderer` etc. The base `Renderer.__init__` signature is the contract — all renderers must accept `(svg, image, detections, geometric)`.

## T17: VisualRenderer + TraceRenderer (2026-06-10)

- **vtracer output format**: bare `<svg>` root with `<path>` children (no `<g>` wrapper). The Python binding is `vtracer.convert_image_to_svg_py(in, out, **params)` — takes filesystem paths, no in-memory variant. See `vectorizer.VtracerVectorizer` for the wrapped API.
- **`LoadedImage` has no `.path` field** — only `pil_image`, `width`, `height`, `np_array` (property). The T17 plan's "Strategy" section correctly directs writing `image.np_array` to a temp PNG via `PIL.Image.fromarray(...).save(temp_path)`. Don't try `image.path`.
- **T16 coordination worked**: `src/img2svg/renderers/base.py` and `__init__.py` did not exist at task start but appeared within seconds. Reading the file first is the right move — don't preemptively create `base.py` from the T16 spec; wait for the real one.
- **lxml element ownership**: extracting `<path>` from a vtracer SVG tree and appending to a different tree fails with "element is already a child of ...". Fix: `etree.fromstring(etree.tostring(el))` round-trips through XML to create an independent copy. Used in `_embed_vtracer_paths`.
- **Insertion-order invariant**: `SVGDocument` always starts with `<title>`, `<desc>`, `<style>` (in that order). The vtracer `<g id="vtracer-output">` must be inserted immediately after these so it renders as the background layer. `add_group` appends, so I add-then-move: `svg.root.remove(g.element); svg.root.insert(N, g.element)` where N is the count of contiguous base tags.
- **Mock namespace for VtracerVectorizer**: tests must patch `img2svg.renderers.visual.VtracerVectorizer` (the import-time name in the visual module), not `img2svg.vectorizer.VtracerVectorizer`. Since `TraceRenderer` calls the shared helper from `visual.py`, patching at the visual namespace covers both renderers — verified by the cross-test.
- **ClassVar preset pattern**: `preset_name: ClassVar[str] = "default"` on `VisualRenderer` is a clean way to declare per-class configuration. (Did NOT use subclassing for `TraceRenderer` because the plan explicitly types it as `class TraceRenderer(Renderer)`.)
- **DRY for shared logic**: private helper `_render_with_vtracer(renderer, preset)` lives in `visual.py` and is imported by `trace.py`. Leading underscore signals "internal"; same package import is acceptable.
- **Test fixture for vtracer output**: a 2-path (visual) and 1-path (trace) synthetic SVG. The asymmetry is intentional — if a regression causes the renderers to share state, the test fixture mismatch catches it.
- **Pre-existing i18n failure confirmed unrelated**: `tests/test_i18n.py::test_ngettext_returns_singular_in_c_locale` fails in isolation; not introduced by T16 or T17. Out of scope.
- **Coverage 100% on new files**: `src/img2svg/renderers/visual.py` 42/42, `src/img2svg/renderers/trace.py` 8/8. The 12 new tests fully cover the implementation.

## T18: AnnotatedRenderer (2026-06-10)

- **Composition pattern**: `AnnotatedRenderer` calls `_render_with_vtracer(self, "default")` first (which inserts `<g id="vtracer-output">` immediately after base elements via `_move_to_after_base`), then appends detection groups via the normal `add_group` flow. Because `SVGGroup.__init__` uses `etree.SubElement` (which appends), the detection groups naturally end up AFTER the vtracer group in document order — no explicit reordering needed for the overlay layer.
- **Z-order invariant**: SVG paint order follows document order. So for a "background trace + foreground labels" composite, the implementation order is: vtracer (background) FIRST, then overlays (foreground). Tests verify this with `enumerate(children)` and check `vtracer_idx < det_idx`.
- **Cross-package import of "private" helper**: `from img2svg.renderers.visual import _render_with_vtracer` is acceptable within the same package. Python doesn't enforce the leading-underscore convention; it's a "we won't break this" signal, but the T17 design chose to make the helper importable rather than duplicate it.
- **`idx` is global enumerate, not per-class**: `LabelsRenderer` uses `for idx, det in enumerate(self.detections)` — the `idx` is the position in the full list, not a counter that resets per `class_name`. Confirmed by `test_three_detections_same_class_produces_three_groups` asserting `["det_cat_0", "det_cat_1", "det_cat_2"]` for three cat detections. T18 follows the same convention; new test correctly uses `["det_cat_0", "det_dog_1"]` for two mixed-class detections.
- **DRY vs. per-renderer style constants**: The bbox / text styling constants (`_BOX_STROKE`, `_TEXT_FONT_SIZE`, etc.) are duplicated in `LabelsRenderer` and `AnnotatedRenderer`. Worth a future refactor — extract a shared module-level constants block or a `_draw_detection_overlay(svg, det, idx)` helper. Not done in T18 to keep scope tight and avoid touching `labels.py`.
- **Coverage 100% on new files**: 7 tests cover: subclass check, preset_name, empty/single/multi-detection z-order, vtracer path preservation inside the group, and the vtracer invocation contract. All 23 renderer tests pass after the addition.

## T19: Pipeline orchestrator + public API (2026-06-10)

- **RENDERER_REGISTRY placement**: Put it in `pipeline.py` rather than `renderers/__init__.py`. Reasons: (1) the pipeline owns the orchestration logic, so co-locating the registry makes the lookup site obvious; (2) keeping `renderers/__init__.py` empty preserves the option of importing renderers by name without forcing the registry to load. `Mode.AUTO` is deliberately absent from the registry — it's resolved by `select_mode()` first. A comment in the code documents this contract.
- **Version constant**: Hardcoded `_VERSION = "0.1.0"` in `pipeline.py` matching `pyproject.toml`. Avoided reading pyproject at runtime (no `tomllib` dep) or `importlib.metadata` (only available after install, breaks `uv run pytest` in dev). Comment in the code explains the choice.
- **Step-marker comments (`# 1. Load`, etc.)**: Kept as BDD-style trace markers matching the 12-step plan spec. The plan's "Pipeline.run() must follow the 12-step sequence exactly" is a hard contract, and the markers make it easy to diff against the plan during review.
- **Timing record-before-construct rule**: Discovered via test failure — `timings["total"]` must be set BEFORE the `Sidecar` is constructed, otherwise it doesn't get serialized. Added an explicit comment in the code explaining the ordering requirement. This is the kind of ordering invariant that comments are *for*.
- **Mock namespaces matter**: Patch `img2svg.pipeline.get_detector` (the import-time name in the pipeline module), NOT `img2svg.detector.get_detector` (the source-of-truth module). Same pattern as the renderers' `VtracerVectorizer` mock — patches must be at the namespace where the *caller* binds the name.
- **Classifier mock for test determinism**: The bundled fixtures (`logo.png`, `diagram.png`, etc.) don't all classify as the type their name suggests — the heuristic depends on alpha + colors + edges, and the test fixtures don't necessarily hit the LOGO branch. The plan's promise of "logo.png is the LOGO fixture" is more aspirational than accurate. Tests that need a specific `ImageType` should mock `img2svg.pipeline.classify` rather than rely on the fixture's classification. This is consistent with the test isolation principle: tests should not be coupled to the *output* of an unmocked component.
- **Per-ROI analysis (step 6) deferred**: The plan lists `analyze_roi` per-detection as step 6, but the spec says "Skip step 6 (analyze_roi loop) for now — not in critical path". Followed the spec. The renderers receive `analysis_global` instead.
- **API kwargs policy**: `_build_options` filters kwargs through `ConversionOptions.model_fields` to silently drop unknown fields. This is friendlier for CLI tools (which often pass extra flags) than the strict-validation alternative. Documented in the docstring.
- **Glob input**: `_expand_glob` uses `Path(inputs).parent` as the base, falling back to `Path.cwd()` when the pattern has no parent (bare filename). Avoids the `Path("")` empty-path bug I almost shipped (which would have glob'd from the filesystem root).
- **Empty input list edge case**: `_resolve_output_dir` falls back to `Path.cwd()` when `inputs == []`. Without this, an empty batch would crash with `IndexError`.
- **Atomic-write independence**: `SVGDocument.write()` and `write_sidecar()` are already both atomic (write-tmp-then-rename), so the pipeline inherits atomicity for free. No special handling needed in the pipeline.
- **`Mode.AUTO` test with classifier mock**: Used `mock.patch("img2svg.pipeline.classify", return_value=(ImageType.LOGO, ...))` to force a known classification, then asserted `mode_used == Mode.LABELS` (the IMAGE_TYPE_TO_MODE[LOGO] entry). This is the most robust way to test the AUTO path without depending on the heuristic's behavior on a specific fixture.
- **25/25 tests pass in 1.3s**: Pipeline (10) + API (15) all green. Full suite is 174/175 with the only failure being the pre-existing i18n test that T16/T17/T18 already noted.
- **Coverage**: 25 tests cover: registry structure, 12-step run with mocked detector, vtracer modes (parametrized VISUAL/ANNOTATED/TRACE), user-mode override, AUTO mode resolution, error propagation, sidecar metadata, public API re-exports, kwargs filtering, glob expansion, output_dir fallback, atomic dir creation, helper unit tests.

## T20: Batch enhancement — directory walking, glob, error continuation (2026-06-10)

- **`_normalize_inputs` design**: Decided to handle glob-vs-string-vs-list in a single helper rather than three. The order of checks matters: glob-meta-chars win over single-file extension matching, and the directory check (`p.is_dir()`) only fires for bare strings without glob chars. A `list[str | Path]` skips the string-shape logic entirely and goes straight to filtering.
- **Case-insensitive extension match uses `.lower()`** on `path.suffix` — not `glob(..., case_sensitive=False)`. The list-input filter is path-based, not filesystem-based, so `.lower()` is the right tool. This means a user can pass `["FOO.PNG", "Bar.Jpg"]` and they'll be accepted. The glob branch uses `Path.glob()` which is OS-dependent (case-sensitive on Linux) — callers wanting case-insensitive globbing must use shell-style `[pP][nN][gG]` patterns.
- **`_resolve_output_dir` backward-compat shim**: The T19-era test suite imports this helper directly, so removing it would break the existing `test_api.py`. Kept it as a thin shim that delegates the list-input case to the same logic. New code should use `_default_output_dir`, which handles the string-form `inputs` too.
- **`_make_error_result` placeholder Sidecar**: Building a `ConversionResult` for a failed file requires a `Sidecar` (required field on `ConversionResult`). Populated it with `Mode.AUTO`, `ImageType.UNKNOWN`, `version="0.0.0"`, empty hash, zero size — all clearly fake. The `errors` list is the source of truth. The placeholder is technically valid Pydantic, just useless as metadata.
- **`nullcontext(None)` for the no-progress path**: `contextlib.nullcontext` is the clean way to unify "with progress" and "without progress" code paths. No need for a custom class.
- **Progress bar `task_ids[0]`**: The `progress_bar` context manager in `logging.py` calls `add_task()` internally, so the resulting `task_ids` list has exactly one entry. Advancing via `progress.advance(progress.task_ids[0], 1)` works without needing the caller to track the task ID.
- **`mocker` fixture not available**: `pytest-mock` is in `pyproject.toml` dev extras but isn't installed in the test venv. Used the built-in `monkeypatch` fixture + `monkeypatch.setattr("img2svg.api.progress_bar", mock)` for the progress-bar patch. Works equally well for the `progress_bar` symbol since it's imported as a name in `api.py`'s namespace.
- **`mocker` works for module-level names too**: When patching a name imported with `from x import y`, patching `module.y` in the consumer's namespace (`img2svg.api.progress_bar`) is the correct target — the consumer looks up `progress_bar` in its own module namespace at call time, not in `img2svg.logging`. Same mock-namespace rule as `get_detector` and `VtracerVectorizer` (T17/T19 learnings).
- **Corrupt-image testing via PIL header bytes**: Writing `b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00garbage"` to a `.png` file makes PIL's `Image.open` raise `Image.UnidentifiedImageError` (missing IHDR chunk), which `load_image` re-raises as `CorruptImageError`. This is the cleanest way to inject a real pipeline failure without mocking `load_image` itself.
- **`_make_error_result` preserves the intended output path**: When the pipeline fails, we still know where the SVG *would have been* written. The error result uses that path as `svg_path` so callers can correlate failures to outputs. The file doesn't exist (the pipeline aborted before write), but the path is informative.
- **13/13 batch tests pass in 1.2s**: Exceeds the 10-test target. Full suite is 187/188 (pre-existing i18n test failure is the only red, unrelated to T20).
- **Coverage 100% on the new code paths**: All 4 input shapes (list, glob, single-file string, directory), recursive and non-recursive, error continuation both directions, progress-bar called/not-called, output_dir override, default output_dir for list input.

