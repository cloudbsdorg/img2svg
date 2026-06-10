# Issues — img2svg

## T1: Subagent timeout

- **Issue**: T1 delegated to `quick` profile subagent ran 10m 5s then was "Aborted" (likely a system-level timeout for monitored background tasks).
- **Impact**: Subagent had created most files (LICENSE, pyproject.toml, .gitignore, .python-version, src/img2svg/__init__.py, tests/__init__.py, locale/.gitkeep, man/.gitkeep, README.md) but had not committed or pushed.
- **Resolution**: Atlas (me) took over directly: updated `.gitignore` (removed `.sisyphus/`, added `.idea/`), set git author, committed with detailed message, pushed to `origin/main`. Then second commit to add `.idea/`.
- **Mitigation for future tasks**: When delegating, use very tight prompts (≤30 lines), focus on 1-2 files per task, OR take over directly for simple work. The user is in a hurry; delegation overhead may exceed the benefit for trivial tasks.

## pyproject.toml — first-commit

- The subagent's pyproject.toml is **good** — better than the plan's spec, actually: it has `pytest-mock` in dev extras (Momus flagged this as missing), has explicit `torch>=2.0,<3` pin, has `[tool.mypy] ignore_missing_imports` overrides for the heavy ML deps. Keep it as-is.

## T1 deviation from plan

- Plan said to NOT commit `.sisyphus/`. User override: commit it.
- Plan said to NOT commit `.idea/`. User override: commit it.
- These overrides are now in `decisions.md` for future reference.
