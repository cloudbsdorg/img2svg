# Configuration

`img2svg` follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) for user-level configuration, with a CloudBSD-specific system-wide fallback for FreeBSD installations. The full set of paths is exposed in `img2svg.paths`.

## Path resolution

The four standard XDG directories, in the order they're checked:

| Purpose          | Environment variable   | Default                       | img2svg subdir            |
|------------------|------------------------|-------------------------------|---------------------------|
| User config      | `XDG_CONFIG_HOME`      | `~/.config`                   | `img2svg/`                |
| User data        | `XDG_DATA_HOME`        | `~/.local/share`              | `img2svg/`                |
| User cache       | `XDG_CACHE_HOME`       | `~/.cache`                    | `img2svg/`                |
| System config    | (hardcoded)            | `/usr/local/etc/cloudbsd/img2svg/` | —                   |

The system config path is intentionally not under `/etc` — CloudBSD convention keeps application config under `/usr/local/etc` to make filesystem snapshots cleaner. The constant is exposed as `img2svg.paths.SYSTEM_CONFIG_DIR`.

The functions that resolve these paths are in `img2svg.paths`:

- `config_dir()` → `Path` — user config directory. Auto-created on first call.
- `data_dir()` → `Path` — user data directory. Auto-created on first call.
- `cache_dir()` → `Path` — user cache directory. Auto-created on first call.
- `model_cache_path(name)` → `Path` — path to a cached YOLO model inside the cache directory.
- `system_config_dir()` → `Path` — the system-wide config directory (no auto-create).
- `load_config()` → `dict` — loads and parses `<config_dir>/config.toml`. Returns `{}` if the file is missing or unparseable.

Every helper that returns a path will create the directory if it does not exist (except `system_config_dir()`). The `mkdir(parents=True, exist_ok=True)` call is in the helper, so the directories are always usable on the next call.

## The config file

A minimal TOML config file at `<config_dir>/config.toml`:

```toml
# Default mode to use when --mode is not provided on the CLI.
default_mode = "auto"

# Default YOLO model.
default_model = "yolo11x.pt"

# Default confidence threshold.
default_conf = 0.25

# Default GPU recommendation strategy.
default_gpu_strategy = "power"
```

The file is loaded by `img2svg.paths.load_config()` and returned as a dict. Currently the keys are not consumed by the runtime — they are reserved for a future release. The intent is that the CLI will consult `load_config()` for defaults before falling back to the built-in defaults.

A malformed file is treated as empty (the loader catches `tomllib.TOMLDecodeError` and `OSError` and returns `{}`). If your config is silently ignored, run `img2svg info` and check that `config_dir()` points where you think it does.

## Environment variables

| Variable          | Effect                                                  |
|-------------------|---------------------------------------------------------|
| `XDG_CONFIG_HOME` | Override the user config base directory.                |
| `XDG_DATA_HOME`   | Override the user data base directory.                  |
| `XDG_CACHE_HOME`  | Override the user cache base directory.                 |
| `HF_HOME`         | HuggingFace cache (consumed by Ultralytics for some models). |
| `PYTORCH_MPS_PREFER_APPLE_GPU` | On Intel Macs with an eGPU, hint to use the Apple GPU for MPS. |

The XDG variables are honored only when set to a non-empty value. An empty string is treated the same as unset.

## Verifying the paths

Run this in a Python REPL to see what your environment resolves to:

```python
from img2svg.paths import config_dir, data_dir, cache_dir, model_cache_path

print("config:", config_dir())
print("data:  ", data_dir())
print("cache: ", cache_dir())
print("model: ", model_cache_path("yolo11x.pt"))
```

The YOLO model is downloaded to `<cache_dir>/models/<name>.pt` on first use and reused across runs.

## See also

- [Installation](installation.md) — FreeBSD and macOS notes that touch the system config path.
- [Troubleshooting](troubleshooting.md) — "Model not found" and "Config not loaded" issues.
- [Development](development.md) — how the paths module is tested.
