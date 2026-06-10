# FreeBSD

img2svg runs on FreeBSD as a best-effort target. CPU mode is fully supported. GPU acceleration depends on the availability of PyTorch wheels built for FreeBSD, which the upstream project does not publish for every release. This page covers installing the prerequisites, the install path for the vtracer native library, the known limitations, and a smoke-test procedure you can run after a fresh install.

## Required packages

Install a Python 3.11 interpreter and the `uv` package manager. You can use either the FreeBSD binary package or the `cargo` route if you prefer to build `uv` from source.

```bash
# Option A: binary package
sudo pkg install python311 py311-uv

# Option B: build uv from source via cargo
sudo pkg install rust cargo
cargo install uv
```

The `py311-uv` package is maintained by the FreeBSD Python team and is the recommended path. The `cargo install` route tracks upstream `uv` releases more closely but takes a few minutes to compile.

## vtracer install

vtracer ships as a Rust crate that builds a small native library. The easiest path on FreeBSD is to install it from the Ports Collection under `graphics/vtracer`:

```bash
# From the Ports Collection
cd /usr/ports/graphics/vtracer
sudo make install clean
```

If you prefer pip and a pre-built wheel is not available for your FreeBSD version, the build will compile from source. Rust is required for that path:

```bash
# Ensure rust is installed first
sudo pkg install rust
uv pip install vtracer
```

When `uv pip install vtracer` runs on FreeBSD without a matching wheel, the build system invokes `cargo` to produce a `.so` shared object. Expect the first build to take a couple of minutes; subsequent installs are cached.

## Known limitations

- **No CUDA.** NVIDIA does not publish CUDA-enabled PyTorch wheels for FreeBSD. `torch.cuda.is_available()` returns `False` even when an NVIDIA card is present, and img2svg falls back to CPU.
- **No ROCm.** AMD's ROCm build of PyTorch is also unavailable for FreeBSD. `list-gpus` will report an empty device list unless an out-of-tree community port is installed.
- **vtracer needs source build on older releases.** Pre-14.x FreeBSD may not have a prebuilt `vtracer` wheel on PyPI. The first install will compile from source; allow extra time and ensure `rust` is on `PATH`.
- **Apple MPS is not relevant** — FreeBSD runs on amd64 and arm64 server hardware, not Apple Silicon.

## Test instructions

Run the project's test suite to confirm the install is healthy, then exercise the CLI end-to-end with a small fixture image.

```bash
# Install all dev extras and run the test suite
uv sync --all-extras
uv run pytest

# Smoke test: convert a fixture to SVG
uv run img2svg tests/fixtures/logo.png -o /tmp/logo.svg
```

The smoke test exercises the loader, classifier, YOLO detector (CPU), vtracer, and the renderer pipeline. A successful run prints `Converted /tmp/logo.svg` and writes a `logo.json` sidecar next to it.
