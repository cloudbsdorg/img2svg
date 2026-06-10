# macOS

img2svg runs natively on macOS. Apple Silicon machines get GPU acceleration through the Metal Performance Shaders (MPS) backend automatically; Intel Macs fall back to CPU. This page covers installing the prerequisites, the GPU story, and a smoke-test procedure.

## Required packages

Install Python 3.11 and the `uv` package manager via Homebrew. The `uv` formula is published to the standard Homebrew tap, so no extra taps are required.

```bash
brew install python@3.11 uv
```

If you already have a Homebrew-managed Python, `brew install uv` is enough — `uv` is a standalone binary that bundles its own Python management and does not need the formula's Python to be on `PATH` for the common case.

## GPU support

MPS works out of the box on Apple Silicon. The pipeline calls `torch.backends.mps.is_available()` and routes the model and tensors to the MPS device without any user configuration.

A few practical notes:

- **Apple Silicon (M1 / M2 / M3 / M4)** is fully supported. Both the YOLO detector and vtracer (which delegates to a Rust kernel) run on CPU by design, but downstream tensor operations in the classifier and geometric-analysis stages accelerate transparently when MPS is available.
- **Intel Macs** fall back to CPU. The install is identical; you simply won't see an MPS device in `img2svg list-gpus`.
- **No CUDA on any Mac.** Even with an eGPU enclosure, CUDA is not available on macOS. The MPS backend is the only GPU path.
- **No ROCm on macOS.** AMD's ROCm does not target macOS.

If you want to confirm MPS is wired up, run `img2svg list-gpus` — on Apple Silicon the table will include a row with vendor `Apple`.

## Test instructions

Run the project's test suite to confirm the install is healthy, then exercise the CLI end-to-end with a small fixture image.

```bash
# Install all dev extras and run the test suite
uv sync --all-extras
uv run pytest

# Smoke test: convert a fixture to SVG
uv run img2svg tests/fixtures/logo.png -o /tmp/logo.svg
```

If you have an Apple Silicon Mac, the same smoke test will route the classifier and tensor operations through MPS. The output is identical to a CPU run; only the wall-clock time differs.
