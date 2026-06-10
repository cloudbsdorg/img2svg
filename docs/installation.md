# Installation

`img2svg` ships as a pure-Python package with native dependencies pulled in at install time (PyTorch, Ultralytics, OpenCV, vtracer). The recommended path is `pip` from PyPI; for development work or unreleased changes, install from source.

## Requirements

- Python 3.10 or newer (3.11+ recommended for `StrEnum` and `tomllib`).
- A C compiler (only required for some wheels; pre-built wheels cover Linux x86_64, macOS arm64/x86_64, and FreeBSD amd64 where available).
- ~1.5 GB of disk for PyTorch and Ultralytics wheels.
- Optional: a CUDA-capable NVIDIA GPU with recent drivers, an AMD GPU with ROCm, or an Apple Silicon Mac for GPU acceleration.

## Install from PyPI

The stable release lives on PyPI. A plain `pip install` pulls in the CPU build of PyTorch by default. If you have a CUDA or ROCm system, install the matching PyTorch build first (see [GPU setup](gpu.md)) and then `img2svg` will pick it up.

```bash
pip install img2svg
```

Verify the install:

```bash
img2svg --version
img2svg info
```

The `info` subcommand prints the Python version, OS string (from `os.uname()`), and the list of compute devices the runtime can see.

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

## macOS notes

macOS is fully supported on both Intel and Apple Silicon. On Apple Silicon (M1/M2/M3/M4), PyTorch's MPS backend is automatically picked up for acceleration, so no extra CUDA toolkit is needed.

```bash
pip3 install img2svg
img2svg info
```

On Intel Macs, install falls back to CPU mode. If you have an external eGPU in a Thunderbolt enclosure, you may need to launch the Python process with the eGPU as the preferred device using `PYTORCH_MPS_PREFER_APPLE_GPU=1` (Apple's experimental flag).

## Verifying the install

A quick smoke test runs the pipeline against the bundled fixtures:

```bash
img2svg info
img2svg list-gpus --strategy power
```

You should see the version, OS string, and a list of detected compute devices. If `list-gpus` reports "No GPUs detected" on a system that does have one, see [Troubleshooting](troubleshooting.md#no-gpu-detected).
