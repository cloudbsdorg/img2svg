# Installation

`img2svg` ships as a pure-Python package with native dependencies pulled in at install time (PyTorch, Ultralytics, OpenCV, vtracer). The recommended path is `pip` from PyPI; for development work or unreleased changes, install from source.

## Requirements

- Python 3.10 or newer (3.11+ recommended for `StrEnum` and `tomllib`).
- A C compiler (only required for some wheels; pre-built wheels cover Linux x86_64, macOS arm64/x86_64, and FreeBSD amd64 where available).
- ~1.5 GB of disk for PyTorch and Ultralytics wheels.
- Optional: a CUDA-capable NVIDIA GPU with recent drivers, an AMD GPU with ROCm, or an Apple Silicon Mac for GPU acceleration.

## What GPU do you have?

Pick the section that matches your hardware. The four first-class backends are:

- **NVIDIA (CUDA)** — any CUDA-capable GeForce, Quadro, RTX, or Tesla card with a recent proprietary driver.
- **AMD (ROCm)** — Radeon RX 6000/7000 series discrete GPUs, or Radeon Pro. iGPUs (Radeon 890M and similar) are detected but lack the VRAM to run YOLO11x; see the AMD section.
- **Apple Silicon (MPS)** — M1, M2, M3, M4 and later. Intel Macs are supported on CPU only.
- **CPU** — always available. Use this when you have no accelerator, are running on a shared/CI host, or only need occasional conversions.

If you are unsure, the bundled script picks the right backend for you:

```bash
./scripts/install_backend.sh
```

It probes the host, prints the recommended PyTorch wheel index URL, and exits with the right `pip install` command pre-filled. See [Compute backends](backends.md) for the full design.

## NVIDIA (CUDA)

### What you have

A workstation or laptop with an NVIDIA discrete GPU. Confirm with:

```bash
nvidia-smi
```

The header line should report the driver version and a `CUDA Version: 12.x` row. That value is the maximum CUDA version your driver supports; install a PyTorch wheel at or below it.

### What to install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install img2svg
```

The `cu126` index is the most common wheel; substitute `cu118`, `cu121`, `cu124`, `cu128` for older drivers. `uv` works the same way:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv add img2svg
```

### How to verify

```bash
img2svg info
```

You should see a `Backend: CUDA (torch 2.x+cu126, NVIDIA <your card>)` line, and `img2svg list-gpus` should list at least one `NVIDIA` row.

## AMD (ROCm)

### What you have

A Linux box with an AMD discrete GPU (Radeon RX 6000/7000 series, or Radeon Pro), or a Ryzen/Threadripper APU whose discrete iGPU is exposed via the amdgpu driver. List visible AMD devices with:

```bash
lspci | grep -i amd
rocm-smi
```

### What to install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install img2svg
```

ROCm PyTorch wheels are **not** on PyPI; they live under `https://download.pytorch.org/whl/rocm6.2` (or `rocm6.3`, `rocm6.4` for newer ROCm releases). Linux x86_64 only.

> **iGPU caveat (512 MB).** APU iGPUs such as the Radeon 890M typically expose only 512 MB of addressable VRAM, which is not enough to load YOLO11x (the default detection model) or the YOLO11x-seg segmentation model. The pipeline will fail with an out-of-memory error on those devices. For detection, use a smaller model — `yolo11n.pt` (~5 MB weights, runs comfortably in 1-2 GB of VRAM) or `yolo11s.pt`:
>
> ```bash
> img2svg photo.png -o photo.svg --model yolo11n.pt
> ```
>
> For segmentation (`--mode segmented`), use `--seg-model yolo11n-seg` or `--seg-model yolo11s-seg`:
>
> ```bash
> img2svg photo.jpg -o photo.svg --mode segmented --seg-model yolo11n-seg
> ```
>
> Or fall back to CPU:
>
> ```bash
> img2svg photo.png -o photo.svg --device cpu --model yolo11n.pt
> ```
>
> YOLO11x is only realistic on a discrete AMD card with at least 6 GB of VRAM. The segmentor additionally auto-falls-back to a smaller model on out-of-memory errors, so a 512 MB iGPU can still run `segmented` mode (it'll fall back to `yolo11n-seg`, then to bbox detection only).

### How to verify

```bash
img2svg info
img2svg list-gpus --strategy power
```

You should see at least one `AMD` row when a discrete GPU is present, or an `APPLE`-prefixed iGPU label on AMD APUs. `img2svg info` reports `Backend: ROCM` when the resolved backend is ROCm.

## macOS / Apple Silicon (MPS)

### What you have

A Mac with an M-series chip (M1, M2, M3, M4, and their Pro/Max/Ultra variants). Confirm with:

```bash
uname -m
```

The output should be `arm64`.

### What to install

```bash
pip3 install img2svg
```

PyTorch's stock macOS arm64 wheel includes MPS support out of the box; no extra CUDA toolkit or driver is needed. No special index URL.

### How to verify

```bash
img2svg info
```

You should see `Backend: MPS (torch 2.x, Apple Silicon)` and a single `APPLE` device in `img2svg list-gpus`. If you see `Backend: CPU` on Apple Silicon, your PyTorch build is missing MPS support — reinstall with the standard macOS arm64 wheel.

## CPU (no GPU)

### What you have

A shared Linux/macOS/Windows box with no discrete GPU, an Intel Mac without an eGPU, a CI runner, or any environment where GPU access is unavailable or unnecessary. CPU mode works everywhere Python runs.

### What to install

```bash
pip install img2svg
```

The default PyPI `torch` wheel is the CPU build; no extra index URL is required. CPU-only inference is significantly slower than GPU for YOLO segmentation, especially on `yolo11x.pt`. Use `yolo11n.pt` for CI smoke tests.

### How to verify

```bash
img2svg info
```

You should see `Backend: CPU (torch 2.x+cpu, CPU only)` and a single `cpu` device. To force CPU explicitly, pass `--device cpu` on every command or set `backend = BackendSpec(requested="cpu")` in code.

## Install from source

Clone the repository and install in editable mode with the dev extras:

```bash
git clone https://github.com/cloudbsdorg/img2svg.git
cd img2svg
pip install -e ".[dev]"
```

Editable mode is preferred during development so `pytest` picks up local edits without a reinstall. The `[dev]` extra pulls in `pytest`, `pytest-mock`, and the type stubs used by the test suite.

## FreeBSD notes

FreeBSD is supported on a best-effort basis. The `img2svg` package is pure Python, but PyTorch wheels for FreeBSD are not always available for the latest release. On FreeBSD 14+ amd64, the upstream PyTorch publishes a CPU-only wheel that works for inference on smaller models.

A typical install sequence:

```bash
pkg install python3 py311-pip
python3 -m pip install --user img2svg
```

If the PyTorch wheel is missing for your Python version, install PyTorch from the FreeBSD ports collection (`math/py-pytorch`) before installing `img2svg`, or use the `uv` package manager which handles resolution across platforms.

The system-wide config directory is `/usr/local/etc/cloudbsd/img2svg/` (see [Configuration](configuration.md)). Run `img2svg info` after install to confirm the runtime sees your hardware.

## Verifying the install

A quick smoke test runs the pipeline against the bundled fixtures:

```bash
img2svg info
img2svg list-gpus --strategy power
```

You should see the version, OS string, and a list of detected compute devices. If `list-gpus` reports "No GPUs detected" on a system that does have one, see [Troubleshooting](troubleshooting.md#no-gpu-detected).

## YOLO model availability

The pipeline uses two YOLO model families, both downloaded on first use and cached under `$XDG_CACHE_HOME/img2svg/models/`:

- **Detection** — `yolo11x.pt` (default, ~114 MB). Configurable via `--model`. Smaller variants: `yolo11n.pt` (~5 MB), `yolo11s.pt` (~20 MB), `yolo11m.pt` (~40 MB), `yolo11l.pt` (~50 MB).
- **Segmentation** — `yolo11s-seg.pt` (default, ~20 MB). Configurable via `--seg-model`. Smaller and larger variants: `yolo11n-seg` (~5 MB), `yolo11m-seg` (~40 MB), `yolo11l-seg` (~50 MB), `yolo11x-seg` (~100 MB).

The segmentation models are used by `--mode segmented` to produce a multi-layer editable SVG with one group per detected object. The detection models power the bounding-box overlays in `--mode labels` and `--mode annotated`.

### Disk-space requirements

The two default models take about 135 MB combined. The full set of 10 YOLO11 variants (5 detectors + 5 segmentors) is roughly 550 MB. Plan accordingly when deploying to air-gapped systems or low-storage CI runners.

### Pre-downloading the weights

`make install` and `uv add img2svg` both download the model weights lazily on first use. To skip the wait and make the first `img2svg convert` instant, pre-download the weights into the XDG cache:

```bash
# Default set (yolo11x.pt + yolo11s-seg.pt, ~135 MB)
make install-models
# or
img2svg-download-models

# All 10 YOLO11 variants (~550 MB)
make install-models MODEL_SET=all
# or
img2svg-download-models --all

# Specific models only
make install-models MODELS=yolo11n.pt,yolo11s-seg.pt

# Re-download a model even if it's already in the cache
img2svg-download-models --models yolo11x.pt --force
```

The download is **idempotent**: already-cached models are skipped, so `make install-models` is safe to run on every deploy. `make install` automatically calls `make install-models` after the package is installed; pass `SKIP_MODELS=1` to defer the download:

```bash
make install SKIP_MODELS=1
make install-models  # run later
```

### Consolidating stray weights

Older img2svg versions (and raw ultralytics) put the downloaded `.pt` file in the current working directory or a `weights/` subdir next to your source. The current version always uses the XDG cache. To move any stray files you've accumulated into the XDG cache:

```bash
make cleanup-models
```

This searches `./`, `~/Pictures`, `~/Downloads`, and `./weights/` for `*.pt` files and either moves them into `~/.cache/img2svg/models/` (if no model with that name is cached) or removes the stray duplicate (if the cache already has a copy).

VRAM guidance:

- 512 MB (typical iGPU): use `yolo11n.pt` for detection, `yolo11n-seg` for segmentation. The pipeline auto-falls-back to a smaller segmentation model on OOM errors.
- 2 GB (entry-level discrete): `yolo11s.pt` and `yolo11s-seg` run comfortably.
- 6 GB+ (mid-range discrete): `yolo11x.pt` and `yolo11x-seg` both fit.

Run `img2svg info` to see the resolved backend and devices after install.
