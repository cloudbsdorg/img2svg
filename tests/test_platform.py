"""Tests for platform compatibility docs and the FreeBSD smoke-test script.

Covers the T29 deliverables:

* ``docs/platforms/freebsd.md`` and ``docs/platforms/macos.md`` exist.
* ``scripts/check_freebsd.sh`` exists, is executable, parses with
  ``bash -n``, and uses ``uname -s`` for OS detection.
* The img2svg source tree never calls ``platform.system(...)`` — the
  CloudBSD guideline is to detect OS with ``uname -s`` (or
  ``os.uname().sysname``) instead.
* The OS reported by the shell's ``uname -s`` matches what Python's
  ``os.uname().sysname`` reports, so cross-language smoke tests are
  consistent.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLATFORMS_DIR = PROJECT_ROOT / "docs" / "platforms"
FREEBSD_MD = PLATFORMS_DIR / "freebsd.md"
MACOS_MD = PLATFORMS_DIR / "macos.md"
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_freebsd.sh"
SRC_DIR = PROJECT_ROOT / "src" / "img2svg"


def _shell_uname_s() -> str:
    """Return ``uname -s`` output, stripped and decoded."""
    return subprocess.check_output(["uname", "-s"]).decode().strip()


def test_uname_s_matches_python_os_uname() -> None:
    """``uname -s`` in the shell and ``os.uname().sysname`` in Python must agree.

    This guards against a future change where the shell script's
    detection logic diverges from the Python module's detection logic.
    The two should always report the same OS on the same host.
    """
    assert _shell_uname_s() == os.uname().sysname


def test_freebsd_md_exists() -> None:
    """``docs/platforms/freebsd.md`` must exist on disk."""
    assert FREEBSD_MD.is_file(), f"missing {FREEBSD_MD}"


def test_macos_md_exists() -> None:
    """``docs/platforms/macos.md`` must exist on disk."""
    assert MACOS_MD.is_file(), f"missing {MACOS_MD}"


def test_check_freebsd_script_exists_and_is_executable() -> None:
    """``scripts/check_freebsd.sh`` must exist and be executable."""
    assert CHECK_SCRIPT.is_file(), f"missing {CHECK_SCRIPT}"
    # Owner-execute bit set is the contract for ``chmod 755``.
    mode = CHECK_SCRIPT.stat().st_mode
    assert mode & 0o100, f"{CHECK_SCRIPT} is not executable (mode={oct(mode)})"


def test_check_freebsd_script_has_valid_bash_syntax() -> None:
    """``bash -n`` on the script must exit 0 (no syntax errors)."""
    result = subprocess.run(
        ["bash", "-n", str(CHECK_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n failed: rc={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_check_freebsd_script_uses_uname_s() -> None:
    """The script must call ``uname -s`` for OS detection.

    CloudBSD guideline: never use ``platform.system()`` (Python) or
    ``$OSTYPE`` (bash-only). ``uname -s`` is portable across Linux,
    macOS, FreeBSD, and every other Unix.
    """
    raw_text = CHECK_SCRIPT.read_text(encoding="utf-8")
    # Strip bash comments so the negative assertion does not trip on
    # a comment that *names* the forbidden pattern for documentation
    # purposes. Lines starting with ``#`` (after optional leading
    # whitespace) are comments; the rest is executable code.
    code_lines = [
        line for line in raw_text.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code_text = "\n".join(code_lines)
    assert "uname -s" in raw_text, (
        f"{CHECK_SCRIPT} must detect the OS with `uname -s`; "
        "do not use `platform.system()` or `$OSTYPE`."
    )
    # Sanity: the forbidden alternative should not appear in code.
    assert "platform.system" not in code_text, (
        f"{CHECK_SCRIPT} must not call `platform.system()`; "
        "use `uname -s` instead."
    )


def test_src_does_not_use_platform_system() -> None:
    """The img2svg source tree must not call ``platform.system(...)``.

    The CloudBSD OS-detection rule is: use ``os.uname().sysname``
    (Unix) or ``sys.platform`` (Windows fallback). ``platform.system``
    is forbidden because it can be slow and its return value is
    locale-dependent on some systems.
    """
    pattern = re.compile(r"platform\.system\(")
    offenders: list[Path] = []
    for py_file in SRC_DIR.rglob("*.py"):
        if pattern.search(py_file.read_text(encoding="utf-8")):
            offenders.append(py_file)
    assert not offenders, (
        "These files call platform.system(...), which is forbidden. "
        "Use os.uname().sysname or sys.platform instead:\n"
        + "\n".join(str(p) for p in offenders)
    )
