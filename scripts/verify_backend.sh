#!/usr/bin/env bash
# verify_backend.sh - assert that the detected backend matches the expected one.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Invoked by the Jenkins "Backend Matrix" stage (one call per matrix cell)
# to confirm each cell is running on the hardware the test expects. The
# script fails the build (exit 1) if the detected backend does not match
# the expected axis value. This is the "right hardware?" gate that turns
# a generic Linux agent into a per-vendor build.
#
# Usage:  scripts/verify_backend.sh <expected>
#
# Arguments:
#   expected   one of: cpu, nvidia, amd, apple
#
# Exit codes:
#   0  detected backend matches expected
#   1  detected backend does not match expected
#   2  invalid expected argument
#   3  backend detection command failed
#
# Detection rules (mapping public axis names to registry BackendType values):
#   nvidia  -> expects "cuda"  (BackendType.CUDA)
#   apple   -> expects "mps"   (BackendType.MPS)
#   cpu     -> expects "cpu"   (BackendType.CPU, strictly)
#   amd     -> expects "cuda" or "rocm"
#             ROCm PyTorch wheels expose the CUDA API, so on an AMD/ROCm
#             host the registry's priority chain reports "cuda" first.
#             Both are accepted because the install extra is what
#             determines which PyTorch build is active, not the registry
#             output.
#
# The script NEVER calls `platform.system()`; hardware identification is
# delegated entirely to the img2svg backend registry.
set -euo pipefail

# Anchor to project root so the script can be invoked from anywhere.
cd "$(dirname "$0")/.."

EXPECTED="${1:-}"

if [[ -z "${EXPECTED}" ]]; then
    echo "ERROR: missing expected backend argument" >&2
    echo "  usage: $0 <expected>" >&2
    echo "  expected: one of cpu, nvidia, amd, apple" >&2
    exit 2
fi

case "${EXPECTED}" in
    cpu|nvidia|amd|apple) ;;
    *)
        echo "ERROR: invalid expected backend '${EXPECTED}'" >&2
        echo "  expected: one of cpu, nvidia, amd, apple" >&2
        exit 2
        ;;
esac

# Run the detection command. Disable -e around it so a non-zero exit
# from uv or python produces a captured error message instead of
# aborting the script before we can format a useful diagnostic.
set +e
DETECTED=$(uv run python -c "from img2svg.backends import REGISTRY; print(REGISTRY.detect().type().value)" 2>&1)
DETECT_RC=$?
set -e

if [[ ${DETECT_RC} -ne 0 ]]; then
    echo "ERROR: backend detection command failed (exit ${DETECT_RC}):" >&2
    echo "${DETECTED}" >&2
    exit 3
fi

# Map our public axis names to the BackendType values the registry emits.
# nvidia and apple are the public Jenkins axis names; the registry's
# enum values are cuda and mps respectively.
case "${EXPECTED}" in
    nvidia) EXPECTED_TYPE="cuda" ;;
    apple)  EXPECTED_TYPE="mps"  ;;
    *)      EXPECTED_TYPE="${EXPECTED}" ;;
esac

# Acceptance logic.
#
# Branch 1: amd accepts cuda (ROCm PyTorch exposes the CUDA API).
if [[ "${EXPECTED_TYPE}" == "amd" && "${DETECTED}" == "cuda" ]]; then
    echo "Detected:  ${DETECTED}"
    echo "Expected:  ${EXPECTED} (accepting cuda as amd; ROCm PyTorch exposes CUDA API)"
    echo "[OK] backend verified: ${EXPECTED}"
    exit 0
fi

# Branch 2: cpu is strict. CPU is the registry's universal fallback, so
# detecting any GPU means the agent is misconfigured for the cpu cell.
if [[ "${EXPECTED_TYPE}" == "cpu" && "${DETECTED}" != "cpu" ]]; then
    echo "Detected:  ${DETECTED}"
    echo "Expected:  cpu"
    echo "[FAIL] expected CPU-only detection, but found ${DETECTED}"
    echo "        (CPU is the registry's fallback, not a priority; the"
    echo "        cpu cell requires a host with no GPU visible to torch)"
    exit 1
fi

# Branch 3: general match.
if [[ "${DETECTED}" == "${EXPECTED_TYPE}" ]]; then
    echo "Detected:  ${DETECTED}"
    echo "Expected:  ${EXPECTED}"
    echo "[OK] backend verified: ${EXPECTED}"
    exit 0
fi

# Branch 4: mismatch on the generic path.
echo "Detected:  ${DETECTED}"
echo "Expected:  ${EXPECTED}"
echo "[FAIL] expected ${EXPECTED} backend, but detected ${DETECTED}"
exit 1
