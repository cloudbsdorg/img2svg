#!/usr/bin/env bash
# install_backend.sh - smart install of img2svg GPU/CPU backend extras.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage:  ./scripts/install_backend.sh [--dry-run] [--apply]
#         ./scripts/install_backend.sh --help
#
# Behavior:
#   * Detects the host GPU via `lspci` (PCI vendor IDs 10de=NVIDIA,
#     1002=AMD) and `uname -s` for Apple Silicon (Darwin). CloudBSD
#     guideline: detect OS with `uname -s`, never `platform.system()`.
#   * Defaults to --dry-run: prints the recommended pip command without
#     installing anything.
#   * --apply: actually runs the install, then runs a verification step
#     that imports the backend registry and prints the detected type.
#   * On Linux, if no `lspci` is on PATH, falls back to "cpu".
#   * On Darwin, always recommends the [apple] extra regardless of
#     lspci output (lspci is rare on macOS).
#   * On Linux, prefers NVIDIA over AMD when both are present. The
#     512 MB iGPU typical of Strix Halo / RDNA 3.5 cannot run YOLO11x
#     in FP16; the warning makes this visible to the user.
#
# Detection priority (Linux):
#   1. NVIDIA display-class device -> img2svg[nvidia]
#   2. AMD    display-class device -> img2svg[amd]   (with 512MB warning)
#   3. nothing                      -> img2svg[cpu]
#
# Detection priority (macOS / Darwin):
#   always -> img2svg[apple]
#
# Exit codes:
#   0  success (dry-run OR apply, including the verify step)
#   1  install failed (--apply only) or unexpected detection result
#   2  invalid CLI arguments
#
# Note on AMD ROCm wheel index: PyTorch ROCm wheels are NOT on PyPI;
# they live on https://download.pytorch.org/whl/rocmX.Y. To install
# the [amd] extra against the ROCm index, set PIP_INDEX_URL (pip) or
# UV_INDEX_URL (uv) before running. Example:
#     PIP_INDEX_URL=https://download.pytorch.org/whl/rocm6.2 \
#         pip install img2svg[amd]
# We intentionally do NOT hardcode an index-url in pyproject.toml —
# uv does not honor per-extra index URLs in pyproject.toml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

APPLY=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            ;;
        --apply)
            APPLY=1
            ;;
        --help|-h)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

if [ -t 1 ] && command -v tput >/dev/null 2>&1 \
        && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    C_BOLD="$(tput bold)"
    C_RED="$(tput setaf 1)"
    C_GREEN="$(tput setaf 2)"
    C_YELLOW="$(tput setaf 3)"
    C_BLUE="$(tput setaf 4)"
    C_RESET="$(tput sgr0)"
else
    C_BOLD="" C_RED="" C_GREEN="" C_YELLOW="" C_BLUE="" C_RESET=""
fi

say()    { printf "%s\n" "$*"; }
header() { printf "\n%s%s%s\n" "$C_BOLD" "$*" "$C_RESET"; }
warn()   { printf "%sWARN:%s %s\n" "$C_YELLOW" "$C_RESET" "$*" >&2; }

OS="$(uname -s)"

# Display class codes: 0300 VGA, 0302 3D, 0380 Display controller.
# Required to skip audio/USB siblings that share the vendor ID.
_DISPLAY_CLASS_RE='\[\s*(0300|0302|0380)\s*\]'

detect_backend() {
    if [ "${OS}" = "Darwin" ]; then
        printf "apple"
        return 0
    fi
    if ! command -v lspci >/dev/null 2>&1; then
        printf "cpu"
        return 0
    fi
    local has_nvidia=0 has_amd=0
    if lspci -nn -d 10de: 2>/dev/null | grep -qE "${_DISPLAY_CLASS_RE}"; then
        has_nvidia=1
    fi
    if lspci -nn -d 1002: 2>/dev/null | grep -qE "${_DISPLAY_CLASS_RE}"; then
        has_amd=1
    fi
    if [ "${has_nvidia}" -eq 1 ]; then
        printf "nvidia"
    elif [ "${has_amd}" -eq 1 ]; then
        printf "amd"
    else
        printf "cpu"
    fi
    return 0
}

header "img2svg backend install"
say "${C_BOLD}Host:${C_RESET}  ${OS} ($(uname -m))"

nvidia_lines=""
amd_lines=""
if [ "${OS}" != "Darwin" ] && command -v lspci >/dev/null 2>&1; then
    nvidia_lines="$(lspci -nn -d 10de: 2>/dev/null \
        | grep -E "${_DISPLAY_CLASS_RE}" || true)"
    amd_lines="$(lspci -nn -d 1002: 2>/dev/null \
        | grep -E "${_DISPLAY_CLASS_RE}" || true)"
fi
if [ -n "${nvidia_lines}" ]; then
    say "${C_BOLD}NVIDIA:${C_RESET}"
    while IFS= read -r line; do
        say "  ${line}"
    done <<<"${nvidia_lines}"
fi
if [ -n "${amd_lines}" ]; then
    say "${C_BOLD}AMD:${C_RESET}"
    while IFS= read -r line; do
        say "  ${line}"
    done <<<"${amd_lines}"
fi

backend="$(detect_backend)"

case "${backend}" in
    apple)
        say ""
        say "${C_BOLD}Detected:${C_RESET} Apple Silicon (macOS)"
        recommend="img2svg[apple]"
        ;;
    nvidia)
        if [ -n "${amd_lines}" ]; then
            say ""
            say "${C_BOLD}Detected:${C_RESET} NVIDIA + AMD"
        else
            say ""
            say "${C_BOLD}Detected:${C_RESET} NVIDIA discrete GPU"
        fi
        recommend="img2svg[nvidia]"
        ;;
    amd)
        say ""
        warn "AMD iGPU only (no NVIDIA detected)."
        warn "On Strix Halo / RDNA 3.5 (Radeon 880M / 890M), the iGPU exposes"
        warn "only 512 MB of dedicated VRAM. YOLO11x (the default model)"
        warn "needs ~700-900 MB FP16, so it will fall back to CPU on this"
        warn "device. On hybrid systems, the NVIDIA discrete GPU is the"
        warn "recommended primary."
        say ""
        say "${C_BOLD}Detected:${C_RESET} AMD iGPU only"
        recommend="img2svg[amd]"
        ;;
    cpu)
        say ""
        say "${C_BOLD}Detected:${C_RESET} no GPU"
        recommend="img2svg[cpu]"
        ;;
    *)
        echo "ERROR: unexpected detection result: ${backend}" >&2
        exit 1
        ;;
esac

if [ "${backend}" = "nvidia" ] && [ -n "${amd_lines}" ]; then
    warn "AMD iGPU also present (see above). The 512 MB iGPU cannot run"
    warn "YOLO11x and will be ignored; install img2svg[amd] only if you"
    warn "plan to use a smaller model on a discrete AMD GPU."
fi

cmd="pip install ${recommend}"

if [ "${backend}" = "amd" ]; then
    say ""
    say "${C_BOLD}Note:${C_RESET} AMD ROCm PyTorch wheels live on a separate index."
    say "  For the [amd] extra, set PIP_INDEX_URL before installing:"
    say "    PIP_INDEX_URL=https://download.pytorch.org/whl/rocm6.2 \\"
    say "        ${cmd}"
fi

if [ "${APPLY}" -eq 0 ]; then
    say ""
    say "${C_BOLD}Dry-run:${C_RESET} would run:"
    say "    ${cmd}"
    say ""
    say "Pass --apply to install."
    exit 0
fi

header "Installing"
say "  ${cmd}"
if ! ${cmd}; then
    echo "ERROR: install command failed: ${cmd}" >&2
    echo "Hint: the [amd] extra requires a ROCm PyTorch wheel index." >&2
    exit 1
fi

header "Verifying"
(
    cd "${PROJECT_ROOT}"
    uv run python -c "from img2svg.backends import REGISTRY; print(REGISTRY.detect().type().value)"
)
say "Done."
exit 0
