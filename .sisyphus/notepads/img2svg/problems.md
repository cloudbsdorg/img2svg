# Problems — img2svg

*No unresolved blockers yet.*

## Open questions for future sessions

- Should T28 (PyPI packaging) include `uv.lock` in the wheel? Plan says to revisit; my current `.gitignore` excludes it.
- Should the plan itself be patched to reflect `.sisyphus/` and `.idea/` being committed? (low priority, doesn't block work)

## Items to revisit

- YOLO model size on first run (131MB for `yolo11x.pt`) — document in README and docs/installation.md.
- ultralytics AGPL-3.0 license — needs NOTICE file (T31) and README mention.
- FreeBSD install verification — needs a FreeBSD host to actually test (not currently in user's "this system").
