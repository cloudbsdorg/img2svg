# Development

This page covers setting up a development environment, the test-driven development workflow, and the contribution conventions. If you are looking to file a bug or request a feature, see the [issue tracker](https://github.com/cloudbsdorg/img2svg/issues) instead.

## Setting up

Clone the repository and install the dev extras:

```bash
git clone https://github.com/cloudbsdorg/img2svg.git
cd img2svg
pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-mock`, and the test fixtures. The `uv` package manager works the same way:

```bash
git clone https://github.com/cloudbsdorg/img2svg.git
cd img2svg
uv sync --extra dev
```

The repo uses Python 3.10+. Most development happens on 3.11 or 3.12; the CI matrix tests 3.10, 3.11, and 3.12 on Linux, and 3.11 on macOS.

## Running the tests

The full suite:

```bash
uv run pytest
```

A specific test file:

```bash
uv run pytest tests/test_api.py
```

A specific test by name:

```bash
uv run pytest tests/test_api.py::test_convert_returns_conversion_result
```

With coverage:

```bash
uv run pytest --cov=img2svg
```

The suite is fast — under 5 seconds on a modern laptop, including the tests that load small YOLO weights. The slow tests (YOLO loading, real image conversion) are gated behind markers; only a few use the real model, the rest use mocks.

## TDD workflow

The project follows a strict test-first workflow:

1. **Write the test first.** A failing test should describe the change you're about to make. The test name is the spec:

   ```python
   def test_convert_batch_with_nonexistent_dir_returns_empty_results():
       results = convert_batch("does/not/exist/", output_dir=tmp_path)
       assert results == []
   ```

2. **Run the test and confirm it fails.** `uv run pytest tests/test_foo.py::test_name` should error. If it passes, either the test is wrong or the feature already exists.

3. **Add the minimal code to make it pass.** Don't refactor surrounding code or add features the test doesn't need. The smallest change that makes the test green is the right change.

4. **Re-run the test and confirm it passes.** Then run the full suite to make sure nothing else broke:

   ```bash
   uv run pytest
   ```

5. **Refactor.** With the test green, you can safely clean up. The test guards against regressions.

6. **Commit.** Atlas (the orchestrator) handles all commits. A typical commit message is:

   ```
   Add convert_batch empty-input handling

   - New test: test_convert_batch_with_nonexistent_dir_returns_empty_results
   - Resolved input list falls back to Path.cwd() instead of crashing
   - Sidecar placeholder for failed files uses ImageType.UNKNOWN
   ```

## Project structure

```
img2svg/
├── src/img2svg/         # Library code
│   ├── api.py           # Public API: convert(), convert_batch()
│   ├── cli.py           # Typer CLI entry point
│   ├── classifier.py    # Image type classification
│   ├── detector.py      # YOLO detection
│   ├── device.py        # Compute device enumeration
│   ├── enums.py         # Mode, ImageType, DeviceStrategy, GpuVendor
│   ├── errors.py        # Public exception hierarchy
│   ├── gpu.py           # GPU detection + recommendation
│   ├── loader.py        # Image loading
│   ├── logging.py       # Logging + progress bar
│   ├── man/             # roff man page (force-included in the wheel)
│   ├── locale/          # gettext message catalogs
│   ├── metadata.py      # File hash, sidecar writing
│   ├── models.py        # Pydantic v2 data models
│   ├── paths.py         # XDG path resolution
│   ├── patterns.py      # Geometric (non-ML) image analysis
│   ├── pipeline.py      # Orchestrator: load → classify → detect → render
│   ├── presets.py       # select_mode() for Mode.AUTO
│   ├── renderers/       # Renderer classes (one per Mode)
│   ├── svg_builder.py   # SVGDocument and SVGGroup
│   └── vectorizer.py    # vtracer wrapper
├── tests/               # pytest test suite
├── docs/                # MkDocs documentation
├── man/                 # Source for the man page
└── pyproject.toml       # Build config + dependencies
```

## Code conventions

- **Python**: 3.10+ syntax, `from __future__ import annotations` at the top of every module, StrEnum-via-shim for cross-compat.
- **Type hints**: Mypy-clean. The `pyproject.toml` ignores missing imports for `ultralytics.*`, `torch.*`, `cv2.*`, `vtracer.*`, `supervision.*` (heavy ML deps without published stubs).
- **Pydantic v2**: All data models use `BaseModel` with `model_config = ConfigDict(use_enum_values=False)` so enum fields stay as enum instances on the model.
- **Docstrings**: Module-level + class-level + function-level. NumPy-style sections for `Args`, `Returns`, `Raises`.
- **Logging**: Use `img2svg.logging.get_logger(__name__)`. Never call `print()` from library code.
- **Errors**: Raise a subclass of `img2svg.errors.Img2SvgError`. The CLI maps them to exit codes.
- **Atomic writes**: Use `tempfile.NamedTemporaryFile` + `os.replace` for any file write that must not leave partial files on disk.

## Adding a renderer

The pipeline maps `Mode` to a `Renderer` class via the `RENDERER_REGISTRY` dict in `img2svg.pipeline`. To add a new mode:

1. Add the enum value in `src/img2svg/enums.py`.
2. Subclass `img2svg.renderers.base.Renderer` in `src/img2svg/renderers/<name>.py`.
3. Register the class in `RENDERER_REGISTRY`.
4. Add tests in `tests/test_renderers/test_<name>.py`.
5. Update the CLI's mode callback to accept the new value.
6. Update the man page's "OUTPUT MODES" section.
7. Update `docs/modes.md`.

## Submitting changes

1. Fork the repository on GitHub.
2. Create a feature branch.
3. Make your change with a test.
4. Run the full test suite (`uv run pytest`) and confirm it passes.
5. Open a pull request. The PR description should reference any open issue it resolves and describe the change at a high level.
6. Atlas (the orchestrator) will review the PR and either merge it or request changes.

## Release process

Versions follow [SemVer](https://semver.org/). Releases are tagged in git and uploaded to PyPI by Atlas.

The version string lives in three places that must stay in sync:

- `pyproject.toml` → `version = "..."`
- `src/img2svg/pipeline.py` → `_VERSION = "..."`
- `src/img2svg/cli.py` → `__version__ = "..."`

When bumping, edit all three and add a `## X.Y.Z` entry to `docs/changelog.md`.

## See also

- [Architecture](architecture.md) — internal design and the 12-step pipeline.
- [Python API](api.md) — public surface.
- [Changelog](changelog.md) — version history.
