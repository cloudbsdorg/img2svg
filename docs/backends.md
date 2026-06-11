# Compute backends

`img2svg` ships with a vendor-neutral compute backend layer. The pipeline never talks to a vendor's PyTorch API directly; it talks to the `DeviceBackend` protocol and lets a `BackendRegistry` decide which concrete implementation gets dispatched. This page is for contributors who want to understand the design or add a new vendor.

## The `DeviceBackend` protocol

The protocol lives in `img2svg.backends.protocol`. Every concrete backend (CPU, CUDA, ROCm, MPS, and any future vendor) implements nine methods. The protocol is decorated with `@runtime_checkable`, so `isinstance(obj, DeviceBackend)` works in tests and in the registry.

```python
from img2svg.backends import DeviceBackend, BackendType

def use(b: DeviceBackend) -> None:
    if not b.is_available():
        return
    for i in range(b.device_count()):
        name = b.device_name(i)
        free_mb = b.free_memory_mb(i)
        # ...
    b.warmup()
```

### The nine methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `type()` | `BackendType` | The concrete backend identity (`CUDA`, `ROCM`, `MPS`, `CPU`, ...). |
| `is_available()` | `bool` | Whether the backend is usable on this host. The registry calls this to filter the auto-detect chain. |
| `device_count()` | `int` | Number of devices visible to this backend. Returns `0` on hosts with no devices. |
| `device_name(i)` | `str` | Human-readable marketing name (e.g. `"NVIDIA GeForce RTX 5070"`). |
| `total_memory_mb(i)` | `int` | Total VRAM/RAM in MiB. Returns `0` when unknown. |
| `free_memory_mb(i)` | `int` | Free VRAM/RAM in MiB. Returns `0` when unknown. |
| `vendor()` | `GpuVendor` | Hardware vendor (`NVIDIA`, `AMD`, `APPLE`, `INTEL`, `CPU`, `UNKNOWN`). |
| `to_ultralytics_string(i)` | `str` | The string the YOLO loader expects: `"cuda:N"`, `"mps"`, or `"cpu"`. |
| `warmup()` | `None` | Optional eager initialization. Default is a no-op. |

The `i` parameter is **zero-indexed in backend-local space**. `CudaBackend(0)` is the first CUDA device, `CudaBackend(1)` is the second, regardless of how the OS numbers PCI slots. This keeps the index portable across mixed-vendor hosts.

`is_available()` may be called many times (every call to `BackendRegistry.available()` re-queries). It must be cheap — typically a few attribute probes on the runtime. It must **not** trigger expensive initialization. The `warmup()` hook exists for backends that need to pay that cost eagerly on first use.

## The `BackendType` enum

`BackendType` is a `StrEnum` (with a Python 3.10 shim) holding the canonical selector keys. Values are lowercase for ergonomic round-tripping with shell tooling.

```python
from img2svg.backends import BackendType

BackendType.CUDA   # "cuda"
BackendType.ROCM   # "rocm"
BackendType.MPS    # "mps"
BackendType.CPU    # "cpu"
BackendType.AUTO   # "auto" — meta-selector resolved by the registry
```

The `Literal` type on `BackendSpec.requested` matches the same five values. Adding a new backend means: (1) add a value here, (2) update the `Literal` in `BackendSpec`, (3) implement the protocol, (4) register the singleton in the registry.

## The auto-detect chain

`BackendRegistry.detect()` walks the priority-ordered list of backends and returns the first one whose `is_available()` returns `True`. The priority order is fixed and deliberate:

```
CUDA > ROCM > MPS > CPU
```

CUDA is first because it is the most common and (for YOLO segmentation at the resolutions vtracer cares about) the fastest path. CPU is last because it is the always-available fallback. ROCm and MPS are middle-of-the-chain: real, useful, but less common than NVIDIA.

```python
from img2svg.backends import REGISTRY

backend = REGISTRY.detect()       # First available in priority order
print(backend.type())              # BackendType.CUDA on a typical workstation
print(backend.vendor())            # GpuVendor.NVIDIA
print(backend.to_ultralytics_string(0))  # "cuda:0"
```

The registry is a stateless facade: `available()` re-queries each backend on every call so hot-plug scenarios (a USB GPU attached after startup) are reflected immediately. The module-level `REGISTRY` singleton is shared across the codebase; the constructor is cheap (a 4-element list copy and a 4-entry dict), so tests can instantiate fresh registries as needed.

### Resolving a `BackendSpec`

`resolve(spec)` dispatches on `spec.requested`:

- `"auto"` → `detect()`.
- A valid `BackendType` whose backend is available → that backend.
- A valid `BackendType` whose backend is **not** available (e.g. `"mps"` on Linux) → `DeviceUnavailableError`.
- An unknown token (e.g. `"directml"`) → `DeviceUnavailableError` listing the actually-available backends.

```python
from img2svg.backends import REGISTRY
from img2svg.models import BackendSpec
from img2svg.errors import DeviceUnavailableError

try:
    backend = REGISTRY.resolve(BackendSpec(requested="mps"))
except DeviceUnavailableError as exc:
    print(f"{exc.requested} not available; try one of: {exc.available}")
```

The device index (`spec.index`) is **not** validated by the registry — out-of-range indices surface as `IndexError` from the backend's `to_ultralytics_string`, which is the right place for that check.

## Adding a new backend

This section walks through adding a hypothetical Intel XPU backend. The same pattern works for any vendor with a PyTorch `+<suffix>` wheel and a runtime probe.

### 1. Implement the protocol

Create `src/img2svg/backends/xpu.py`. The contract is the nine methods plus the `_DEVICE_INDEX` (or analogous) class constant.

```python
# src/img2svg/backends/xpu.py
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Intel XPU compute backend."""

from __future__ import annotations

from img2svg.enums import GpuVendor


class XPUBackend:
    """Intel XPU compute backend implementation."""

    _DEVICE_INDEX = 0

    def type(self) -> "BackendType":
        from img2svg.backends.protocol import BackendType
        return BackendType.XPU  # new enum value added in step 2

    def is_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
        try:
            return bool(xpu.is_available())
        except (AttributeError, RuntimeError):
            return False

    def device_count(self) -> int:
        if not self.is_available():
            return 0
        try:
            import torch
            return int(torch.xpu.device_count())
        except Exception:
            return 0

    # ... device_name, total_memory_mb, free_memory_mb, vendor, ...
    def vendor(self) -> GpuVendor:
        return GpuVendor.INTEL

    def to_ultralytics_string(self, i: int) -> str:
        if i < 0 or i >= self.device_count():
            raise IndexError(...)
        return f"xpu:{i}"  # depends on ultralytics support

    def warmup(self) -> None:
        return None


XPU_BACKEND = XPUBackend()
__all__ = ["XPUBackend", "XPU_BACKEND"]
```

Two important conventions to follow:

- **Lazy `import torch`.** Do not import `torch` at module level. The module must be importable on systems without PyTorch (docs builds, lint runners, minimal CI).
- **Defensive `is_available`.** Triple-defensive: try/except around the import, `getattr` for the submodule, try/except around the actual probe. Some PyTorch wheels do not expose `torch.xpu` at all.

### 2. Add the enum value

Open `src/img2svg/backends/protocol.py` and add the new variant to `BackendType`:

```python
class BackendType(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"
    XPU = "xpu"  # new
```

Also update the `Literal` in `src/img2svg/models.py`:

```python
class BackendSpec(BaseModel):
    requested: Literal["auto", "cuda", "rocm", "mps", "cpu", "xpu"] = "auto"
```

### 3. Register the singleton

Open `src/img2svg/backends/registry.py` and slot the new backend into the priority list. Pick a position that reflects how common and fast the new vendor is relative to the existing chain:

```python
_ALL: list[DeviceBackend] = [
    CUDA_BACKEND,
    ROCM_BACKEND,
    XPU_BACKEND,   # new
    MPS_BACKEND,
    CPU_BACKEND,
]
```

### 4. Re-export

Open `src/img2svg/backends/__init__.py` and add the new symbols:

```python
from img2svg.backends.xpu import XPU_BACKEND, XPUBackend
```

Then add them to `__all__` in the established isort-style order (constants before classes, both alphabetical).

### 5. Tests

The new backend needs a test file at `tests/test_backends/test_xpu.py`. The patterns from the existing `test_cuda.py`, `test_rocm.py`, and `test_mps.py` are reusable:

- Use a `MagicMock(spec=DeviceBackend)` for tests that don't exercise real hardware.
- For the "torch missing" path, patch `builtins.__import__` rather than `sys.modules["torch"]` (which only matters if torch was already imported in the test env).
- For the "torch has no `xpu` submodule" path, install a fake `torch` whose `__getattr__` raises `AttributeError`.
- Always test the `IndexError` contract for out-of-range device indices.

### 6. Documentation

Add a section to `docs/installation.md` following the existing per-vendor pattern (What you have → What to install → How to verify), and update the README's "Backend architecture" table to include the new vendor.

## Auto-detect priority in practice

The chain matters most on mixed-vendor hosts. A workstation with an NVIDIA RTX 5070 and an AMD Radeon 890M iGPU will resolve to CUDA by default — the higher-priority backend. To force the AMD path, pass `--device rocm` (or `BackendSpec(requested="rocm")` from the Python API).

iGPUs are detected but often lack the VRAM to run the default `yolo11x.pt` model. The 512 MB Radeon 890M iGPU in this project's reference host can run `yolo11n.pt` and `yolo11s.pt` but not `yolo11x.pt`. See [Installation — AMD (ROCm)](installation.md#amd-rocm) for the full caveat.

## See also

- [Installation](installation.md) — per-vendor install guide.
- [Python API](api.md#backendspec) — the `BackendSpec` model and the new `backend=` parameter on `ConversionOptions`.
- `src/img2svg/backends/protocol.py` — the protocol source of truth.
- `src/img2svg/backends/registry.py` — the auto-detect chain and the `REGISTRY` singleton.
