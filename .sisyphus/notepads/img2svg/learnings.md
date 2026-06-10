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
