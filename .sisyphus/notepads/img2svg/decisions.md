# Decisions — img2svg

## T1 decisions (user overrides of plan)

- **`.sisyphus/` committed to git** — User explicitly said "commit the @.sisyphus dir". The plan's `MUST NOT HAVE` guard is overridden. Future delegations should NOT re-add `.sisyphus/` to `.gitignore`.
- **`.idea/` (PyCharm IDE) committed to git** — User said "i want the .idea folder up there too". Override plan's typical convention.
- **Push without confirmation** — User said "commit and push" — explicit override of plan's `MUST NOT: auto-push to git; the git remote add and git push are explicit user actions` guard. User pre-authorizes push for T1 commit.

## Honcho-stored user facts (2026-06-10)

- mlapointe operates across many workstations (m5, framework, claw, dgx).
- mlapointe wants to push everything to git that doesn't contain secrets.
- mlapointe's standard workflow: init → commit all state → push → repeat on every workstation.

## Architectural decisions (per plan)

- **Library-first Python API**, CLI is thin wrapper.
- **Vectorizer**: vtracer (MIT, PyO3, prebuilt wheels, multi-color + binary modes).
- **ML detector**: ultralytics YOLO11x (user's explicit accuracy choice; AGPL-3.0 dependency, flagged).
- **Geometric CV**: OpenCV (K-means in LAB, Canny, findContours, Hough).
- **GPU autodetect priority**: CUDA (covers NVIDIA + AMD ROCm) → DirectML (Windows AMD) → MPS (Apple) → CPU.
- **GPU recommendation**: `--gpu-strategy power|availability` (rank by VRAM total or VRAM free).
- **4 output modes**: `labels`, `visual`, `annotated`, `trace` (renamed from "photorealistic trace").
- **Output path collision**: default overwrite with warning; `--no-clobber` to skip.
- **Exit codes**: 0=success, 1=partial fail, 2=invalid args, 3=dep/model fail.
- **Sidecar**: Pydantic-validated JSON next to every SVG output.
- **i18n**: gettext for user-facing strings; English default; UTF-8.
- **Config**: XDG Base Directory (`$XDG_CONFIG_HOME/img2svg/`, etc.) + FreeBSD `/usr/local/etc/cloudbsd/img2svg/`.
- **Testing**: TDD, pytest, 80%+ coverage, 100% critical paths, AI-parseable output (JSON via `pytest-json-report`, TAP via `pytest-tap`, JUnit via `pytest --junitxml`).
- **CI**: Jenkinsfile + local `scripts/ci.sh` (no GitHub Actions — user has no GitHub integrations).
