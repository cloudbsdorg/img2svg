# img2svg - Rich-based logging setup for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Rich-based logging setup for img2svg.

Provides:
- setup_logging(verbose, quiet) to configure root logger
- get_logger(name) for module-level loggers
- Progress context managers for indeterminate (model download) and
  determinate (batch) operations.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

_VERBOSITY_LEVELS: dict[str, int] = {
    "quiet": logging.WARNING,
    "default": logging.INFO,
    "verbose": logging.DEBUG,
}


def setup_logging(verbosity: str = "default") -> None:
    """Configure root logger with Rich handler.

    verbosity: one of 'quiet' (WARNING), 'default' (INFO), 'verbose' (DEBUG).
    """
    level = _VERBOSITY_LEVELS.get(verbosity, logging.INFO)
    root = logging.getLogger()
    # Clear existing handlers (in case setup_logging is called twice)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = RichHandler(
        console=Console(stderr=True),
        show_path=False,
        show_time=False,
        rich_tracebacks=True,
    )
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)
    # Silence noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Call setup_logging() first to configure output."""
    return logging.getLogger(name)


@contextmanager
def progress_spinner(description: str) -> Iterator[None]:
    """Indeterminate progress: spinner with elapsed time, for model downloads etc."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=Console(stderr=True),
    ) as progress:
        progress.add_task(description, total=None)
        yield


@contextmanager
def progress_bar(total: int, description: str = "Processing") -> Iterator[Progress]:
    """Determinate progress bar for batch operations.

    Yields the Progress object so callers can call `.add_task()` to track substeps
    or `.update()` to advance.
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    ) as progress:
        progress.add_task(description, total=total)
        yield progress
