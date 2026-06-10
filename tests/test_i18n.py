# img2svg - tests for the i18n/gettext scaffold.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the i18n/gettext scaffold."""

from __future__ import annotations

import pytest

from img2svg import i18n


def test_gettext_returns_input_in_c_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    i18n.init_locale("C")
    assert i18n._("Hello, world!") == "Hello, world!"


def test_ngettext_returns_singular_in_c_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    i18n.init_locale("C")
    assert i18n.ngettext("one file", "many files", 1) == "one file"
    assert i18n.ngettext("one file", "many files", 5) == "one file"  # NullTranslations
