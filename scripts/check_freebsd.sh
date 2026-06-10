#!/bin/sh
# check_freebsd.sh - FreeBSD smoke test for img2svg.
#
# Usage:  ./scripts/check_freebsd.sh
#
# Behavior:
# * Detects the host OS via `uname -s` (never Python's
#   `platform.system()` or `$OSTYPE`).
#   * If the host is NOT FreeBSD, prints a one-line note and exits 0.
#     The exit-0 path is deliberate: this script is safe to run from
#     any CI box, including Linux and macOS runners. The check is
#     informational; failure to be FreeBSD is not an error.
#   * If the host IS FreeBSD, prints install hints, then runs the
#     end-to-end CLI smoke test (convert a fixture to SVG in /tmp).
#
# Exit codes:
#   0  success (or not-FreeBSD, which is a no-op)
#   1  the smoke-test conversion failed
#
# This script is intended for the FreeBSD Jenkins node. It does not
# modify the system; it only reads from the source tree and writes
# the output SVG to /tmp.
set -eu

# ---- OS detection ---------------------------------------------------------
# CloudBSD guideline: detect OS with `uname -s`, never with
# `platform.system()` (Python) or $OSTYPE (bash-only). POSIX sh is
# universally available on FreeBSD, so we use it here for maximum
# portability across the BSD family.
OS="$(uname -s)"

if [ "${OS}" != "FreeBSD" ]; then
    echo "Not FreeBSD (uname -s reports ${OS}); skipping FreeBSD smoke test."
    exit 0
fi

# ---- FreeBSD-specific install hints ---------------------------------------
cat <<'EOF'
=== FreeBSD smoke test ===
Install prerequisites (one-time setup):
    sudo pkg install python311 py311-uv
    sudo pkg install rust
    uv sync --all-extras

Running end-to-end conversion test...
EOF

# ---- Smoke test ------------------------------------------------------------
# Use a known fixture (logo.png) and write the output to /tmp so the
# script never pollutes the source tree. The script aborts on a
# non-zero exit from `uv run` (set -e above), so a failure here
# produces exit code 1.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE="${PROJECT_ROOT}/tests/fixtures/logo.png"
OUTPUT="/tmp/img2svg_freebsd_smoke.svg"

if [ ! -f "${FIXTURE}" ]; then
    echo "ERROR: fixture not found at ${FIXTURE}" >&2
    exit 1
fi

uv run img2svg "${FIXTURE}" -o "${OUTPUT}"
echo "Smoke test passed: ${OUTPUT}"
exit 0
