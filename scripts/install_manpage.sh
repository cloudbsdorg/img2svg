#!/usr/bin/env bash
# install_manpage.sh - install the img2svg(1) man page into the
# system man hierarchy and refresh the man page index.
#
# Usage:  sudo ./scripts/install_manpage.sh
#
# Behavior:
#   * Detects the host OS via `uname -s` (never platform.system()).
#   * Installs man/img2svg.1 to /usr/local/share/man/man1/img2svg.1
#     on both Linux and FreeBSD (the only two supported OS targets).
#   * Uses `install -D -m 644` when available (GNU coreutils, FreeBSD
#     install); falls back to `mkdir -p` + `cp` otherwise.
#   * Refreshes the man page index via `mandb`.
#
# Exit codes:
#   0  success
#   1  usage error (script must be run as root)
#   2  source man page missing
#   3  install failed
#   4  mandb failed
#
# NOTE: This script is a developer convenience, not a packaging
# requirement. Wheel installs place the man page under the package
# data directory; users copy it into /usr/local if they want a
# system-wide entry.
set -e

# ---- sanity checks --------------------------------------------------------

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: install_manpage.sh must be run as root (try sudo)" >&2
    exit 1
fi

# Resolve the script's own directory so the script can be invoked from
# anywhere. BASH_SOURCE works for bash; fall back to $0 for sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${PROJECT_ROOT}/man/img2svg.1"

if [ ! -f "${SRC}" ]; then
    echo "ERROR: source man page not found at ${SRC}" >&2
    exit 2
fi

# ---- OS detection ---------------------------------------------------------
# CloudBSD guideline: detect OS with `uname -s`, never with
# `platform.system()` (Python) or $OSTYPE (bash-only).
OS="$(uname -s)"
case "${OS}" in
    Linux)
        # /usr/local/share/man/man1 is the FHS-compliant location used
        # by /usr/local installs on Linux (and matches FreeBSD).
        MANDIR="/usr/local/share/man/man1"
        ;;
    FreeBSD)
        MANDIR="/usr/local/share/man/man1"
        ;;
    *)
        echo "ERROR: unsupported OS: ${OS} (expected Linux or FreeBSD)" >&2
        exit 1
        ;;
esac

DEST="${MANDIR}/img2svg.1"

# ---- install --------------------------------------------------------------
# Prefer `install -D -m 644` (creates intermediate directories, sets mode).
# Fall back to mkdir -p + cp on systems without `install -D`.
echo "Installing ${SRC} -> ${DEST} (mode 0644)"

if command -v install >/dev/null 2>&1 && install -D -m 644 /dev/null /tmp/__img2svg_install_probe__ 2>/dev/null; then
    # `install -D` exists and works. Clean up the probe file, then
    # do the real install. We pass the source as the last arg so -D
    # creates any missing parent directories for the destination.
    rm -f /tmp/__img2svg_install_probe__
    install -D -m 644 -t "${MANDIR}" "${SRC}" || {
        echo "ERROR: install failed" >&2
        exit 3
    }
else
    # Fallback: portable mkdir + cp path. Works on every Unix.
    mkdir -p "${MANDIR}" || {
        echo "ERROR: could not create ${MANDIR}" >&2
        exit 3
    }
    cp -f "${SRC}" "${DEST}" || {
        echo "ERROR: cp failed" >&2
        exit 3
    }
    chmod 0644 "${DEST}" || {
        echo "ERROR: chmod failed" >&2
        exit 3
    }
fi

# ---- mandb refresh --------------------------------------------------------
# `mandb` rebuilds the man page index used by `man -k` and `apropos(1)`.
# Failure here is non-fatal for the install (the page is on disk), but
# we surface it so users notice. mandb lives at /usr/bin/mandb on Linux
# and /usr/sbin/mandb on FreeBSD; we look it up via PATH.
if command -v mandb >/dev/null 2>&1; then
    echo "Refreshing man page index (mandb)"
    if ! mandb -q 2>/dev/null; then
        # Some systems refuse -q when run as non-root, or when the
        # manpath config blocks writes. Try without -q as a fallback.
        if ! mandb 2>/dev/null; then
            echo "WARNING: mandb refresh failed; the man page is installed but may not appear in apropos(1) output." >&2
            exit 4
        fi
    fi
else
    echo "WARNING: mandb not found; skipping index refresh."
fi

echo "Done. Try:  man img2svg"
exit 0
