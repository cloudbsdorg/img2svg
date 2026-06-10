# Troubleshooting

Common problems and their fixes. If your problem is not here, please open an issue on GitHub with the output of `img2svg info` and `img2svg list-gpus`.

## "Model not found"

```
FileNotFoundError: [Errno 2] No such file or directory: 'yolo11x.pt'
```

The YOLO weights file is downloaded to the XDG cache on first use, but the download can fail for a number of reasons (firewall, no network, partial download). The fix:

1. Run `img2svg info` to confirm the cache directory resolves to a writable path.
2. Run `img2svg list-gpus` to confirm the runtime is healthy.
3. Manually download the weights to the cache path:

   ```bash
   # Find the path
   img2svg --help | grep -i model
   # Or in Python
   python -c "from img2svg.paths import model_cache_path; print(model_cache_path())"
   ```

4. Drop the `.pt` file at that path and re-run.

If the path is on a read-only filesystem (e.g. a shared cluster), set `XDG_CACHE_HOME` to a writable location before running.

## "CUDA out of memory"

```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB.
```

YOLO segmentation at full resolution on a large image can exceed the GPU's VRAM. Three options, in order of preference:

1. Use a smaller model:

   ```bash
   img2svg photo.png -o photo.svg --model yolo11n.pt
   ```

   `yolo11n` is the smallest, `yolo11s` is a step up, `yolo11x` is the largest. Smaller models have lower VRAM requirements.

2. Fall back to CPU:

   ```bash
   img2svg photo.png -o photo.svg --device cpu
   ```

   CPU inference is much slower but never OOMs.

3. Resize the input before passing it in. `img2svg` does not currently do this internally; use your image editor or `convert input.png -resize 1024x1024 input-small.png` from ImageMagick.

## "No GPU detected"

`img2svg list-gpus` shows the empty table, but you have a GPU.

The detection chain is: `nvidia-smi` → `rocm-smi` → `torch.cuda`. Each step is a fallback. If `nvidia-smi` is not on `$PATH` and PyTorch was installed without CUDA, you'll see nothing even if you have an AMD GPU.

Check each step:

```bash
# NVIDIA
nvidia-smi
# AMD
rocm-smi
# PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

- If `nvidia-smi` fails on an NVIDIA box, the proprietary driver is missing or broken. Reinstall it (e.g. `apt install --reinstall nvidia-driver-535` on Debian/Ubuntu).
- If `nvidia-smi` works but PyTorch says `cuda.is_available() == False`, the PyTorch wheel is the CPU-only build. Reinstall with the CUDA wheel:

  ```bash
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```

- If `rocm-smi` fails on an AMD box, install the ROCm userspace package (`rocm-smi`).
- If PyTorch sees an AMD GPU as `NVIDIA` in the table, that's a known labeling limitation in our detection code — the runtime still uses the GPU correctly. Force it with `--device cuda:0`.

## "UnsupportedFormatError"

```
img2svg.errors.UnsupportedFormatError: unsupported format: .heic
```

The supported set is `png`, `jpg`, `jpeg`, `bmp`, `webp`, `tiff`, `tif`, `gif` (case-insensitive). HEIC, AVIF, RAW formats, and PSD are not supported. Convert the file with another tool (e.g. ImageMagick, `ffmpeg`) before passing it in.

## "CorruptImageError"

```
img2svg.errors.CorruptImageError: cannot identify image file
```

Pillow could not decode the file. The file may be truncated, the wrong extension, or a format that Pillow does not have a plugin for. Try opening the file in another image viewer to confirm it is intact. If the file is genuinely broken, re-export it from the source.

## "OutputPathCollisionError"

```
img2svg.errors.OutputPathCollisionError: output file already exists: photo.svg
```

The output file already exists and `--no-clobber` is set. Either remove the file, or omit `--no-clobber` (or pass `--force-overwrite`) to overwrite. In a batch, this error is recorded in the result's `errors` field and processing continues for the rest of the batch.

## Slow first run

The first time you run `img2svg` on a fresh install, the YOLO weights download and the vtracer native library loads. Subsequent runs are faster because both are cached. If the first run is taking longer than 60 seconds with no progress, check your network connection or proxy settings.

## Sidecar JSON is missing

By default, the sidecar is written next to the SVG with the same name plus `.json` (e.g. `photo.svg.json`). If the directory is not writable, the sidecar is not written and the conversion still succeeds — there is no error, but downstream tools that read the sidecar will not find it. Check directory permissions.

## "command not found: img2svg"

The CLI script was not installed on `$PATH`. With `pip install --user`, scripts go to `~/.local/bin` on Linux and `~/Library/Python/X.Y/bin` on macOS — both must be on `$PATH`. With system `pip` or `uv`, scripts go to a path that is already on `$PATH` (usually `/usr/local/bin`).

Add the user bin directory to your shell rc:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

Then re-source the file or open a new shell.

## Still stuck?

Open an issue at <https://github.com/cloudbsdorg/img2svg/issues> with:

1. The output of `img2svg info` (version, OS, devices).
2. The output of `img2svg list-gpus`.
3. The full command line and the full error message.
4. If the input is not sensitive, a sample image that reproduces the problem.

We can usually diagnose from those four artifacts.
