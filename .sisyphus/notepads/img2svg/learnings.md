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
