# img2svg - tests for the logging module.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the logging module."""

from __future__ import annotations

import logging

from img2svg import logging as imlogging


def test_setup_logging_default_is_info() -> None:
    imlogging.setup_logging("default")
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_verbose_is_debug() -> None:
    imlogging.setup_logging("verbose")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_quiet_is_warning() -> None:
    imlogging.setup_logging("quiet")
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_unknown_verbosity_falls_back_to_info() -> None:
    imlogging.setup_logging("nonsense")
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_silences_ultralytics() -> None:
    imlogging.setup_logging("verbose")
    assert logging.getLogger("ultralytics").level == logging.WARNING


def test_get_logger_returns_named_logger() -> None:
    imlogging.setup_logging("default")
    log = imlogging.get_logger("img2svg.test")
    assert log.name == "img2svg.test"
    assert isinstance(log, logging.Logger)


def test_setup_logging_idempotent() -> None:
    """Calling setup_logging twice should not duplicate handlers."""
    imlogging.setup_logging("default")
    n1 = len(logging.getLogger().handlers)
    imlogging.setup_logging("default")
    n2 = len(logging.getLogger().handlers)
    assert n1 == n2


def test_progress_spinner_yields() -> None:
    with imlogging.progress_spinner("loading"):
        pass  # just check the context manager works


def test_progress_bar_yields_with_total() -> None:
    with imlogging.progress_bar(total=10, description="test") as p:
        # advance the task
        task_id = p.task_ids[0]
        p.update(task_id, advance=5)
        # task is still there
        assert task_id in p.task_ids
