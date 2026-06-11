# img2svg - pre-download YOLO model weights into the XDG cache directory.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Pre-download YOLO model weights into the XDG cache directory.

The YOLO model files are large (5 MB to ~114 MB) and ultralytics' default
download path puts them in a ``weights/`` subdirectory of the current
working directory. That pollutes the user's workspace and makes the
download invisible to anyone running from a different cwd.

This module centralises the pre-download logic so the cache always lives
in the XDG cache directory (``$XDG_CACHE_HOME/img2svg/models/``,
default ``~/.cache/img2svg/models/``) and the same code path is used
by the ``make install-models`` target, the ``img2svg download-models``
CLI, and the lazy download on first detector use.

The download itself is performed by :class:`ultralytics.YOLO`. We
override ultralytics' ``settings.weights_dir`` for the duration of the
call so the file lands in our XDG cache, not in cwd. After the call
we restore the original setting.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from img2svg import paths
from img2svg.errors import ModelLoadError

_log = logging.getLogger("img2svg.model_download")

#: All YOLO11 model variants that ship on the ultralytics release page.
#: Detection: yolo11{n,s,m,l,x}; Segmentation: yolo11{n,s,m,l,x}-seg.
ALL_MODELS: tuple[str, ...] = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
)

#: Default models for "ready to use out of the box" experience.
#: Picked to match the pipeline's ``--model`` and ``--seg-model`` defaults.
DEFAULT_MODELS: tuple[str, ...] = (
    "yolo11x.pt",  # detection default (~114 MB)
    "yolo11s-seg.pt",  # segmentation default (~20 MB)
)


def pre_download_model(model_name: str, *, force: bool = False) -> Path:
    """Pre-download a YOLO model to the XDG cache dir.

    Idempotent: if the model already exists in the cache, return the
    cached path without re-downloading. Use ``force=True`` to re-download
    (useful after an ultralytics upgrade that ships new weights).

    The download is performed by constructing a :class:`ultralytics.YOLO`
    object, which triggers the auto-download path. We override ultralytics'
    ``settings.weights_dir`` for the duration of the call so the file
    lands in the XDG cache, not in cwd's ``weights/`` subdir. The YOLO
    object is discarded immediately after construction — we only want
    the file on disk, not a loaded model in memory.

    Parameters
    ----------
    model_name : str
        Model filename, e.g. ``"yolo11x.pt"`` or ``"yolo11s-seg"``. The
        ``.pt`` suffix is optional; bare stems like ``"yolo11s-seg"``
        (which the CLI uses for ``--seg-model``) are normalised to
        ``"yolo11s-seg.pt"`` so the cache directory contains a
        consistent set of ``*.pt`` files.
       force : bool, default False
        Re-download even if the file is already cached.

    Returns
    -------
    pathlib.Path
        Absolute path to the cached model.

    Raises
    ------
    ModelLoadError
        If ultralytics is not installed, the download fails, or the
        downloaded file cannot be located on disk.
    """
    # Normalise: accept both ``"yolo11s-seg"`` and ``"yolo11s-seg.pt"``
    # by appending the suffix when missing.
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    cache_path = paths.model_cache_path(model_name)

    if cache_path.exists() and not force:
        _log.debug("Model %s already in cache at %s", model_name, cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
        from ultralytics.utils import SETTINGS
    except ImportError as e:
        raise ModelLoadError(
            model_name,
            original=RuntimeError(
                f"ultralytics is not installed; cannot download {model_name}. "
                "Install img2svg[cpu|nvidia|amd|apple] first."
            ),
        ) from e

    # Override weights_dir to point at our XDG cache for this download.
    # We restore it in the finally clause so the change is local to this
    # call even if the download raises.
    original_weights_dir = SETTINGS.get("weights_dir", "weights")
    try:
        SETTINGS["weights_dir"] = str(cache_path.parent)
        _log.info("Downloading %s to %s", model_name, cache_path)
        # Construct the YOLO object; this triggers the auto-download.
        # We immediately discard the model object (we only want the file
        # on disk) to free the 2-4 GB of GPU/RAM the model may consume.
        model = YOLO(model_name)
        del model
    except Exception as e:
        raise ModelLoadError(
            model_name,
            original=RuntimeError(
                f"Failed to download {model_name} to {cache_path}: {e}"
            ),
        ) from e
    finally:
        # Always restore the original weights_dir setting, even on error.
        SETTINGS["weights_dir"] = original_weights_dir

    if not cache_path.exists():
        # Defensive: download claimed success but file is not where we expect.
        # Search a small set of candidate locations that ultralytics may
        # have used (cwd, the SETTINGS-original weights_dir, and the
        # classic ``weights/`` subdir of cwd) and move any hit into the
        # cache. This handles three real-world cases:
        #   1. The user previously ran img2svg from a project dir and
        #      the model landed in cwd (the original bug we're fixing).
        #   2. The user has a custom SETTINGS["weights_dir"] pointing
        #      somewhere other than our cache.
        #   3. ultralytics stored the file in cwd's ``weights/`` subdir
        #      because that's its default.
        candidates = [
            Path(model_name),  # cwd-relative
            Path(str(original_weights_dir)) / model_name,
            Path("weights") / model_name,
        ]
        moved_from: Path | None = None
        for cand in candidates:
            if cand.exists() and cand.resolve() != cache_path.resolve():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _log.info("Moving existing model from %s to %s", cand, cache_path)
                shutil.move(str(cand), str(cache_path))
                moved_from = cand
                break
        if moved_from is None:
            raise ModelLoadError(
                model_name,
                original=RuntimeError(
                    f"Download of {model_name} reported success but file not "
                    f"found at {cache_path} (or at any fallback: {candidates})"
                ),
            )

    return cache_path


def pre_download_models(
    model_names: list[str] | None = None,
    *,
    force: bool = False,
) -> list[Path]:
    """Pre-download a list of YOLO models to the XDG cache dir.

    If ``model_names`` is None, downloads the default model set
    (:data:`DEFAULT_MODELS`). Stops on the first failure and propagates
    the :class:`ModelLoadError` to the caller.

    Returns the list of paths to the cached models, in input order.
    """
    if model_names is None:
        model_names = list(DEFAULT_MODELS)

    results: list[Path] = []
    for name in model_names:
        results.append(pre_download_model(name, force=force))
    return results


def list_cached_models() -> list[Path]:
    """Return paths to all ``*.pt`` files currently in the XDG cache dir.

    Sorted by filename for stable output. Returns an empty list if the
    cache directory does not exist yet.
    """
    models_dir = paths.cache_dir() / "models"
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.pt"))


def cache_dir_size_bytes() -> int:
    """Return the total on-disk size of the model cache directory in bytes.

    Returns 0 if the cache directory does not exist. Used by the CLI
    to show the user how much disk the pre-downloaded models occupy.
    """
    models_dir = paths.cache_dir() / "models"
    if not models_dir.exists():
        return 0
    return sum(f.stat().st_size for f in models_dir.glob("*.pt") if f.is_file())
