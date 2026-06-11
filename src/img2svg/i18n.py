# img2svg - gettext-based internationalization scaffold for img2svg.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""gettext-based internationalization scaffold for img2svg.

In the default (C) locale this module is effectively a no-op: `_(s)` returns
`s` unchanged. A future release can ship a `locale/<lang>/LC_MESSAGES/img2svg.mo`
compiled catalog to enable real translation.
"""

from __future__ import annotations

import gettext
import os
from pathlib import Path

_DOMAIN = "img2svg"
_LOCALE_DIR = Path(__file__).parent / "locale"

_translation: gettext.GNUTranslations | gettext.NullTranslations = gettext.NullTranslations()
_initialized = False


def init_locale(language: str | None = None) -> None:
    """Initialize gettext for the given language (or $LANG / $LC_ALL)."""
    global _translation, _initialized
    lang = language or os.environ.get("LC_ALL") or os.environ.get("LANG") or "C"
    try:
        _translation = gettext.translation(
            _DOMAIN,
            localedir=str(_LOCALE_DIR),
            languages=[lang],
            fallback=True,
        )
    except (OSError, FileNotFoundError):
        _translation = gettext.NullTranslations()
    _initialized = True


def _(message: str) -> str:
    """Translate a message string. Returns the input unchanged if no catalog loaded."""
    if not _initialized:
        init_locale()
    return _translation.gettext(message)


def ngettext(singular: str, plural: str, n: int) -> str:
    """Plural-aware translation."""
    if not _initialized:
        init_locale()
    return _translation.ngettext(singular, plural, n)
