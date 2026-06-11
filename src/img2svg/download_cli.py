# img2svg - console-script entry point for pre-downloading YOLO models.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Console-script entry point for pre-downloading YOLO model weights.

Exposed as ``img2svg-download-models`` on the user's PATH after
``pip install img2svg``. Thin wrapper over :mod:`img2svg.model_download`
that handles the CLI argument parsing and human-friendly output.

The Makefile (``make install-models``) calls the project-local
``scripts/download_models.py`` instead, which has the same behaviour
but works before the package is installed (e.g. during a fresh
``make install``).

Exit codes
----------
0  success (all requested models are in the cache)
1  a download failed
2  invalid CLI arguments
"""

from __future__ import annotations

import argparse
import sys

from img2svg.errors import ModelLoadError
from img2svg.model_download import (
    ALL_MODELS,
    DEFAULT_MODELS,
    cache_dir_size_bytes,
    list_cached_models,
    pre_download_models,
)


def _humanize_bytes(n: int) -> str:
    """Format a byte count with a single-letter suffix (KB/MB/GB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _parse_models(value: str) -> list[str]:
    """Parse a comma-separated model list from the CLI."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img2svg-download-models",
        description=(
            "Pre-download YOLO model weights into the XDG cache directory "
            "($XDG_CACHE_HOME/img2svg/models/, default ~/.cache/img2svg/models/)."
        ),
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=None,
        help=(
            "Comma-separated list of model filenames to download "
            "(e.g. 'yolo11x.pt,yolo11s-seg.pt'). Defaults to the "
            "recommended set: yolo11x.pt + yolo11s-seg.pt (~135 MB total)."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 10 YOLO11 model variants (~550 MB total).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model is already in the cache.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the models currently in the cache and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``img2svg-download-models`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        cached = list_cached_models()
        if not cached:
            print("No models in the XDG cache yet.")
            return 0
        print(f"Cached models ({_humanize_bytes(cache_dir_size_bytes())} total):")
        for p in cached:
            print(f"  {p.name}  ({_humanize_bytes(p.stat().st_size)})")
        return 0

    if args.all:
        models = list(ALL_MODELS)
    elif args.models is not None:
        models = args.models
    else:
        models = list(DEFAULT_MODELS)

    print(f"Downloading {len(models)} model(s) to the XDG cache...")
    try:
        results = pre_download_models(models, force=args.force)
    except ModelLoadError as e:
        print(f"ERROR: {e.user_message()}", file=sys.stderr)
        if e.original is not None:
            print(f"  cause: {e.original}", file=sys.stderr)
        return 1

    for path in results:
        size = path.stat().st_size if path.exists() else 0
        marker = "(re-downloaded)" if args.force else "(cached)"
        print(f"  ok  {path.name}  {_humanize_bytes(size)} {marker}")
    print(
        f"Done. {len(results)} model(s) in cache "
        f"({_humanize_bytes(cache_dir_size_bytes())} total)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
