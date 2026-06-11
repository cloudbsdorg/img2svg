# img2svg - Central registry of compute backends.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Central registry of compute backends.

The :class:`BackendRegistry` is the single source of truth for "which
backend do we dispatch to?". It holds a priority-ordered list of all
concrete :class:`DeviceBackend` implementations, exposes an auto-detect
chain (:meth:`available` and :meth:`detect`), resolves a parsed
:class:`img2svg.models.BackendSpec` to a concrete backend
(:meth:`resolve`), and parses legacy device strings like ``"cuda:0"``
into a :class:`BackendSpec` (:meth:`for_device_string`).

The auto-detect priority is fixed and deliberate:

    CUDA > ROCM > MPS > CPU

CUDA is first because it is the most common and (for YOLO segmentation
at the resolutions vtracer cares about) the fastest path. CPU is last
because it is the always-available fallback. ROCm and MPS are
middle-of-the-chain: real, useful, but less common than NVIDIA.

Every method that consults the registry is side-effect free — the
registry only calls ``is_available()`` (which is itself a cheap probe)
and never invokes any vendor's runtime initialization. That keeps
``BackendRegistry.detect()`` cheap to call from the CLI's
``list-gpus`` command and from the conversion pipeline's startup
sequence alike.
"""

from __future__ import annotations

from img2svg.backends.cpu import CPU_BACKEND
from img2svg.backends.cuda import CUDA_BACKEND
from img2svg.backends.mps import MPS_BACKEND
from img2svg.backends.protocol import BackendType, DeviceBackend
from img2svg.backends.rocm import ROCM_BACKEND
from img2svg.errors import DeviceUnavailableError
from img2svg.models import BackendSpec

# Priority order for the auto-detect chain. First available wins.
# Stored as a module-level list (not a tuple) so subclasses or future
# extensions can override ``_ALL`` before instantiation.
_ALL: list[DeviceBackend] = [CUDA_BACKEND, ROCM_BACKEND, MPS_BACKEND, CPU_BACKEND]


class BackendRegistry:
    """Central registry of compute backends.

    Maintains a priority-ordered list of all known backends and exposes
    auto-detection, :class:`BackendSpec` resolution, and legacy-string
    parsing helpers. The class is intentionally stateless after
    construction — :meth:`available` re-queries each backend every call
    so that hot-plug scenarios (a USB GPU attached after startup) are
    reflected immediately.

    The constructor takes no arguments. Tests that need to exercise
    the registry against a different backend matrix should monkeypatch
    the module-level :data:`_ALL` list or patch the
    ``CUDA_BACKEND``/``ROCM_BACKEND``/``MPS_BACKEND``/``CPU_BACKEND``
    module attributes before instantiating.
    """

    def __init__(self) -> None:
        # Defensive copy so callers cannot mutate the shared class-level
        # list by reaching into an instance. The lookup map is built
        # eagerly so :meth:`resolve` does not have to scan on every call.
        self._all: list[DeviceBackend] = list(_ALL)
        self._by_type: dict[BackendType, DeviceBackend] = {b.type(): b for b in self._all}

    def available(self) -> list[DeviceBackend]:
        """Return the available backends in priority order.

        Filters :attr:`_all` to those whose :meth:`DeviceBackend.is_available`
        returns ``True``, preserving the priority order. On a typical
        Linux + CUDA workstation the result is ``[CUDA_BACKEND,
        CPU_BACKEND]`` — ROCm and MPS are filtered out because the
        host has no AMD GPU and no Apple Silicon.
        """
        return [b for b in self._all if b.is_available()]

    def detect(self) -> DeviceBackend:
        """Return the first available backend, falling back to CPU.

        The "auto-detect" path: returns the first entry of
        :meth:`available`, which is the highest-priority backend that
        reports itself as usable. If — implausibly — no backend
        reports available, returns :data:`CPU_BACKEND` as a hard
        universal fallback. (This branch is unreachable in practice
        because :class:`CPUBackend` always reports ``is_available() ==
        True``; it is a defence-in-depth guard so a future
        configuration that disabled the CPU backend could not crash
        the conversion pipeline.)
        """
        available = self.available()
        if available:
            return available[0]
        return CPU_BACKEND

    def resolve(self, spec: BackendSpec) -> DeviceBackend:
        """Resolve a :class:`BackendSpec` to a concrete backend.

        Behaviour:

        * ``spec.requested == "auto"`` → :meth:`detect` (auto-detect).
        * ``spec.requested`` is a valid :class:`BackendType` (e.g.
          ``"cuda"``, ``"mps"``) and the matching backend is
          :meth:`DeviceBackend.is_available` → that backend.
        * ``spec.requested`` is a valid :class:`BackendType` but the
          matching backend is *not* available (e.g. ``"mps"`` on a
          Linux box) → :class:`DeviceUnavailableError`.
        * ``spec.requested`` is not a valid :class:`BackendType` (e.g.
          ``"bogus"``, ``"directml"`` constructed via
          :func:`pydantic.BaseModel.model_construct`) →
          :class:`DeviceUnavailableError` listing the actually-available
          backends.

        The device index (``spec.index``) is intentionally *not*
        validated here. The CUDA/ROCm backend's
        :meth:`to_ultralytics_string` will raise :class:`IndexError`
        on an out-of-range index; that is the right place for that
        check because the registry cannot know how many devices a
        backend has without making a (potentially expensive) probe
        on every call.
        """
        if spec.requested == "auto":
            return self.detect()
        try:
            backend_type = BackendType(spec.requested)
        except ValueError:
            # ``spec.requested`` is not a valid BackendType member. This
            # happens when a spec is constructed via model_construct
            # (deliberate in for_device_string's deferred-validation
            # path) or when a future user passes an unsupported vendor
            # name. Raise the same exception shape as the
            # is_available==False case so callers can handle both with
            # one except clause.
            raise DeviceUnavailableError(
                requested=spec.requested,
                available=[b.type().value for b in self.available()],
            ) from None
        backend = self._by_type.get(backend_type)
        if backend is None or not backend.is_available():
            # Either the BackendType is a valid enum member we don't
            # support (currently impossible because the Literal in
            # BackendSpec gates the spec, but kept as a guard against
            # future enum additions) or the backend is unusable on
            # this host. Same error shape in both cases.
            raise DeviceUnavailableError(
                requested=spec.requested,
                available=[b.type().value for b in self.available()],
            )
        return backend

    def for_device_string(self, s: str) -> BackendSpec:
        """Parse a legacy device string into a :class:`BackendSpec`.

        Accepts the canonical forms the rest of the codebase has used
        since the 0.1.x series:

        * ``""`` or ``"auto"`` → :class:`BackendSpec` with
          ``requested="auto"``.
        * ``"cuda"``, ``"rocm"``, ``"mps"``, ``"cpu"`` → :class:`BackendSpec`
          with the matching ``requested`` value.
        * ``"cuda:0"``, ``"cuda:1"``, ``"rocm:0"`` … → :class:`BackendSpec`
          with the index parsed by the :class:`BackendSpec` validator.
        * ``"directml"``, ``"xpu"``, ``"bogus"`` and other unknown
          names → a :class:`BackendSpec` that *fails* at
          :meth:`resolve` rather than at parse time. This is a
          deliberate design choice: it lets the user see the full list
          of *actually available* backends in the
          :class:`DeviceUnavailableError` message instead of a
          Pydantic ``ValidationError`` that names a literal and stops
          there.

        The input is case-insensitive and leading/trailing whitespace
        is stripped, so ``"  CUDA  "`` and ``"cuda"`` produce the
        same spec.
        """
        s = (s or "").strip().lower()
        if not s:
            return BackendSpec(requested="auto")
        # Known plain backends. The string is in the Literal set so
        # model_validate succeeds without the index-parser path.
        if s in ("auto", "cuda", "rocm", "mps", "cpu"):
            return BackendSpec.model_validate({"requested": s})
        # Indexed form ("cuda:0", "rocm:1"). Defer to the BackendSpec
        # validator's _parse_indexed_requested for parsing.
        if ":" in s:
            base = s.split(":", 1)[0]
            if base in ("cuda", "rocm"):
                return BackendSpec.model_validate({"requested": s})
        # Unknown backend. Construct without Pydantic validation so the
        # error surfaces at resolve() with the helpful "available:
        # [...list...]" suffix instead of a bare Literal complaint.
        return BackendSpec.model_construct(requested=s)


# Module-level singleton. Other modules (device.py refactor in T9,
# detector.py, pipeline.py, cli.py) import this constant rather than
# constructing their own BackendRegistry. The instance is cheap to
# keep around: construction just copies a 4-element list and builds a
# 4-entry dict, and every method is read-only.
REGISTRY = BackendRegistry()

__all__ = ["REGISTRY", "BackendRegistry"]
