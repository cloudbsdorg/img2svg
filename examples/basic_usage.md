# Basic CLI Usage

The `img2svg` command turns raster images (PNG, JPG, BMP, WebP, TIFF, GIF) into clean, optimized SVG files. The same binary handles a single file, a directory, or a glob pattern, and writes a `.json` sidecar next to every output.

The examples below assume you have a working install (`uv tool install .` or `pip install .`). The `uv run` form works directly from a checkout.

## 1. Single file conversion

Convert one image to one SVG. `--output` is required for single-file runs.

```bash
uv run img2svg tests/fixtures/logo.png -o out/logo.svg
```

Pick the mode (`auto`, `labels`, `visual`, `annotated`, `trace`) and the file lands next to a matching `.json` sidecar carrying mode, model, device, detections, and timings.

## 2. Directory batch conversion

Point the command at a directory and every supported image inside is converted. The SVGs land alongside their source files by default.

```bash
uv run img2svg tests/fixtures -o out/batch
```

Pass `--mode labels` (or any other mode) to apply the same renderer across the whole batch. Failures are reported per file and do not abort the run unless `--no-clobber` collides with an existing output.

## 3. Glob pattern

Any input containing a glob meta-character (`*`, `?`, `[`) routes through the batch path. The output directory defaults to the parent of the first match.

```bash
uv run img2svg 'tests/fixtures/*.png' -o out/globs
```

Quote the pattern to keep your shell from expanding it locally. Mixing extensions (`*.png`, `*.jpg`) works as long as the suffixes are in the supported set.

## 4. Explicit mode override

The default is `auto`, which picks a renderer from the image classifier. Override it when you know the result you want: `labels` (bbox + class), `visual` (vtracer fill), `annotated` (visual + labels), `trace` (vtracer outline only).

```bash
uv run img2svg tests/fixtures/diagram.png -o out/diagram_annotated.svg --mode annotated
```

`auto` resolves to `labels` for logo-like inputs, `visual` for photos, and `annotated` for diagrams and screenshots. Use the explicit flag to skip the heuristic.

## 5. GPU device selection

Default device is `auto`, which picks the best available backend. Force CPU, a specific CUDA device, or the Apple silicon GPU.

```bash
uv run img2svg tests/fixtures/photo.jpg -o out/photo.svg --device cpu
uv run img2svg tests/fixtures/photo.jpg -o out/photo.svg --device cuda:0
uv run img2svg tests/fixtures/photo.jpg -o out/photo.svg --device mps
```

Pair with `--gpu-strategy` (`power` or `availability`) to influence which GPU gets picked when multiple are visible. See [`gpu_recommendation.md`](./gpu_recommendation.md) for what that command reports.

## Exit codes

The CLI uses a small, stable exit-code mapping:

| Code | Meaning |
| ---- | ------- |
| 0    | Success |
| 1    | Partial batch failure (at least one file errored) |
| 2    | Invalid args, unsupported format, file not found |
| 3    | Dependency or model load failure |

Scripts can rely on these values in `if` checks without parsing stderr.
