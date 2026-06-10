# GPU Recommendation

`img2svg list-gpus` enumerates every visible compute device and marks the one that the chosen strategy would pick. Use it to confirm that CUDA, ROCm, or MPS is wired up before launching a long batch.

## Example output

The command below was run on a Linux laptop with one NVIDIA RTX 5070:

```bash
$ uv run img2svg list-gpus
```

```
                                 Available GPUs
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃       ┃        ┃               ┃    VRAM Total ┃     VRAM Free ┃             ┃
┃ Index ┃ Vendor ┃ Name          ┃          (MB) ┃          (MB) ┃ Recommended ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│     0 │ nvidia │ NVIDIA        │          8151 │          7680 │      Y      │
│       │        │ GeForce RTX   │               │               │             │
│       │        │ 5070 Laptop   │               │               │             │
│       │        │ GPU           │               │               │             │
└───────┴────────┴───────────────┴───────────────┴───────────────┴─────────────┘
```

On a box with no GPU the table is replaced with `No GPUs detected.` in yellow, and exit is still 0.

## Strategy flag

Pass `--strategy` (or `-s`) to change the recommendation rule:

```bash
uv run img2svg list-gpus --strategy power          # default
uv run img2svg list-gpus --strategy availability    # most free VRAM
uv run img2svg list-gpus --strategy auto            # vendor default
```

| Strategy       | Picks                                                |
| -------------- | ---------------------------------------------------- |
| `power`        | Highest total VRAM, lowest index on ties             |
| `availability` | Most free VRAM right now, lowest index on ties       |
| `auto`         | First device reported by the detection backend       |

`power` is the right pick for offline batch jobs. `availability` matters when other processes are sharing the box. `auto` falls back to whatever order the OS reported the devices in.

## Column meanings

- **Index**: the device index `img2svg` will pass to `--device` (e.g. `cuda:0`).
- **Vendor**: `nvidia`, `amd`, `apple`, or `intel`. The detection order is `nvidia-smi` → `rocm-smi` → `torch.cuda`. On macOS, MPS devices are reported as `apple`.
- **Name**: human-readable model string from the driver.
- **VRAM Total (MB)**: physical memory on the card. Drives the `power` strategy.
- **VRAM Free (MB)**: free right now. Drives the `availability` strategy.
- **Recommended**: `Y` marks the device the strategy picked. Every other row is blank.

The YOLO model is roughly 100 MB on disk, but inference peaks well above that for a single 640×640 batch. A 4 GB card is the practical minimum for `yolo11x.pt`; smaller models like `yolo11n.pt` run on much less.

## Wiring it into a convert run

The recommendation is informational, not automatic. To use the chosen device, copy the index into `--device`:

```bash
uv run img2svg photo.jpg -o out/photo.svg --device cuda:0
```

Pair this with `--gpu-strategy availability` for shared boxes, or with `--device cpu` to skip the GPU entirely.
