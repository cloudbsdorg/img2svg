# GPU setup

`img2svg` accelerates the YOLO detector on any CUDA-, ROCm-, or MPS-capable device. By default, the runtime picks the best available device using the `power` strategy (largest total VRAM). This page covers per-vendor setup for NVIDIA, AMD, and Apple Silicon.

## Detecting GPUs

Run `img2svg list-gpus` to see what's available:

```bash
img2svg list-gpus --strategy power
```

The table shows index, vendor, name, total VRAM, free VRAM, and which device the recommender would pick. If the table is empty, see [Troubleshooting](troubleshooting.md#no-gpu-detected).

You can also dump a summary at any time with `img2svg info`, which prints the OS, Python version, PyTorch version, and a one-line list of devices the runtime can see.

## NVIDIA CUDA

PyTorch publishes CUDA wheels for the most common CUDA versions (11.8, 12.1, 12.4, 12.6, 12.8). Pick the wheel that matches your installed NVIDIA driver.

First, check the driver's max-supported CUDA version:

```bash
nvidia-smi
```

The header line shows `CUDA Version: 12.6` (or similar). Install a PyTorch wheel whose CUDA version is **less than or equal to** that. The standard install command is:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install `img2svg`:

```bash
pip install img2svg
```

The package's PyTorch dependency will be satisfied by the wheel you just installed; `uv` and `pip` will not downgrade it.

Confirm with:

```bash
img2svg list-gpus
```

You should see at least one `NVIDIA` row. To force a specific device at runtime, use `--device cuda:0` (or `cuda:1`, etc.).

### Common driver issues

- **`nvidia-smi: command not found`** — the NVIDIA driver is not installed. Install the proprietary NVIDIA driver from your distro's package manager (e.g. `apt install nvidia-driver-535` on Debian/Ubuntu, `dnf install akmod-nvidia` on Fedora).
- **`libcudart.so: cannot open shared object file`** — the CUDA runtime libraries are missing. Reinstall PyTorch with the matching CUDA wheel.
- **Old GPU (Maxwell or older)** — PyTorch dropped CUDA 11.8 support for sm_50 and below. Use a CUDA 12.x wheel only if your GPU is sm_60 or newer; otherwise fall back to CPU.

## AMD ROCm

ROCm support in PyTorch uses the same `torch.cuda` API as NVIDIA — the underlying runtime is the ROCm HIP SDK, but PyTorch's CUDA build is a drop-in for most code paths. Install the ROCm build of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

Then install `img2svg`:

```bash
pip install img2svg
```

`img2svg` will see the GPU as a `NVIDIA` row in `list-gpus` (the vendor label is conservative — the underlying runtime is still `torch.cuda`). To force the AMD device, use `--device cuda:0`.

### ROCm requirements

- Linux only (ROCm does not support Windows or macOS for compute workloads).
- A supported GPU: Radeon RX 6000/7000 series, or Radeon Pro. Older GCN cards are not supported in recent ROCm releases.
- The `rocm-smi` utility for diagnostics (the package's `rocm-smi` is the AMD equivalent of `nvidia-smi`).

### Troubleshooting ROCm

- **`HIP not available`** — the ROCm userspace libraries are not on `LD_LIBRARY_PATH`. On most distros, install `rocm-dev` or `rocm-libs`.
- **`no ROCm devices found`** — your kernel module is not loaded. Run `lsmod | grep amdgpu` and check `dmesg` for errors.
- **PyTorch sees the device but training fails** — check the supported-GPU list at [ROCm's compatibility page](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html).

## Apple Silicon MPS

Apple Silicon Macs (M1, M2, M3, M4) use Apple's Metal Performance Shaders (MPS) backend. PyTorch's stock macOS arm64 wheel includes MPS support out of the box:

```bash
pip install img2svg
img2svg list-gpus
```

You should see an `APPLE` row. To force the GPU:

```bash
img2svg photo.png -o photo.svg --device mps
```

MPS acceleration is generally faster than CPU on Apple Silicon, but slightly slower than CUDA on comparable hardware. It's a real win over CPU for the YOLO segmentation step.

### eGPU on Intel Macs

If you are running an Intel Mac with an external eGPU, Apple's preferred-device flag can help:

```bash
PYTORCH_MPS_PREFER_APPLE_GPU=1 img2svg photo.png -o photo.svg --device mps
```

This is experimental and may not help on all configurations.

## Multi-GPU selection

When the system has more than one GPU, the recommendation strategy picks the device. Two strategies are exposed:

| Strategy       | Picks                                | Use when                                |
|----------------|--------------------------------------|------------------------------------------|
| `power`        | Largest total VRAM                   | Best raw performance, idle box.         |
| `availability` | Largest free VRAM right now          | Shared box, want to avoid OOM.          |
| `auto`         | First device, in index order         | Reproducible runs, no surprises.        |

```bash
img2svg photo.png -o photo.svg --gpu-strategy availability
```

The chosen device is recorded in the sidecar JSON's `device` field. To override manually, pass `--device cuda:N` (NVIDIA/AMD) or `--device mps` (Apple).

## See also

- [Usage](usage.md#gpu-selection) — CLI flag reference for `--device` and `--gpu-strategy`.
- [Troubleshooting](troubleshooting.md#cuda-out-of-memory) — common GPU errors.
- [Architecture](architecture.md) — how the pipeline dispatches to the device.
