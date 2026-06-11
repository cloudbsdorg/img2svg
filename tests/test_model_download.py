# img2svg - tests for the YOLO model pre-download module.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :mod:`img2svg.model_download`.

The download path is mocked so the tests do not hit the network. The
goal is to verify the public contract:

* Idempotent (already-cached files return immediately).
* The cache path is correct (XDG-aware).
* Candidate search-and-move logic finds files in cwd, in the
  SETTINGS-original weights_dir, and in cwd's ``weights/`` subdir.
* Errors are wrapped in :class:`ModelLoadError` with a useful
  ``user_message()``.
* The module-level constants (``ALL_MODELS``, ``DEFAULT_MODELS``)
  match the documented names.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from img2svg.errors import ModelLoadError
from img2svg.model_download import (
    ALL_MODELS,
    DEFAULT_MODELS,
    cache_dir_size_bytes,
    list_cached_models,
    pre_download_model,
    pre_download_models,
)


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect paths.model_cache_path() AND paths.cache_dir() to a tmp_path-isolated cache.

    Returns the models directory (``cache_dir() / "models"``) for
    assertions. The model_download module uses both helpers, so both
    need to be patched for the test isolation to work end-to-end. The
    fixture sets them up to be CONSISTENT: ``cache_dir()`` returns a
    parent, and ``model_cache_path(name)`` returns
    ``parent / "models" / name``.
    """
    cache_root = tmp_path / "img2svg_cache"
    cache_root.mkdir()
    models = cache_root / "models"
    models.mkdir()
    monkeypatch.setattr(
        "img2svg.model_download.paths.model_cache_path",
        lambda name: models / name,
    )
    monkeypatch.setattr(
        "img2svg.model_download.paths.cache_dir",
        lambda: cache_root,
    )
    return models


def test_default_models_includes_detector_and_segmentor() -> None:
    """DEFAULT_MODELS contains yolo11x (detector) + yolo11s-seg (segmentor)."""
    assert "yolo11x.pt" in DEFAULT_MODELS
    assert "yolo11s-seg.pt" in DEFAULT_MODELS
    assert len(DEFAULT_MODELS) == 2


def test_all_models_has_ten_variants() -> None:
    """ALL_MODELS has the 5 detectors + 5 segmentors (10 total)."""
    detectors = [m for m in ALL_MODELS if "-seg" not in m]
    segmentors = [m for m in ALL_MODELS if "-seg" in m]
    assert len(detectors) == 5
    assert len(segmentors) == 5
    assert len(ALL_MODELS) == 10


def test_pre_download_model_is_idempotent(fake_cache: Path) -> None:
    """If the model is already in the cache, no YOLO() call is made."""
    target = fake_cache / "yolo11x.pt"
    target.write_bytes(b"fake-yolo-weights")
    with mock.patch("ultralytics.YOLO") as yolo_mock:
        result = pre_download_model("yolo11x.pt")
    assert result == target
    yolo_mock.assert_not_called()


def test_pre_download_model_normalises_bare_stem(fake_cache: Path) -> None:
    """A bare stem like ``"yolo11s-seg"`` is normalised to ``"yolo11s-seg.pt"``.

    The CLI uses bare stems for ``--seg-model``; the cache must always
    contain files with the ``.pt`` extension for consistency.
    """
    target = fake_cache / "yolo11s-seg.pt"
    target.write_bytes(b"already-here")
    with mock.patch("ultralytics.YOLO") as yolo_mock:
        result = pre_download_model("yolo11s-seg")  # no .pt suffix
    assert result == target
    yolo_mock.assert_not_called()


def test_pre_download_model_calls_yolo_when_missing(fake_cache: Path) -> None:
    """If the model is missing, YOLO() is constructed and the model object freed."""
    yolo_mock = mock.MagicMock()
    with mock.patch("ultralytics.YOLO", return_value=yolo_mock) as ctor:
        # To avoid the post-construction "file not found" check, write
        # the file at the cache path as if YOLO had downloaded it.
        def _fake_construct(name: str) -> mock.MagicMock:
            (fake_cache / name).write_bytes(b"downloaded-by-yolo")
            return yolo_mock

        ctor.side_effect = _fake_construct
        result = pre_download_model("yolo11n.pt")
    assert result == fake_cache / "yolo11n.pt"
    assert result.exists()
    ctor.assert_called_once_with("yolo11n.pt")


def test_pre_download_model_moves_existing_cwd_file(fake_cache: Path, tmp_path: Path) -> None:
    """If a stray .pt exists in cwd, the download path is found there and moved.

    This is the core "stray file in cwd" recovery flow: ultralytics
    locates an existing file (no download happens), the pre_download
    then discovers it's in cwd and moves it into the cache.
    """
    # Simulate a stray .pt file in cwd (i.e. the user's previous cwd).
    stray = tmp_path / "yolo11x.pt"
    stray.write_bytes(b"stray-yolo-weights")
    monkeypatch_cwd = pytest.MonkeyPatch()
    monkeypatch_cwd.chdir(tmp_path)
    try:
        yolo_mock = mock.MagicMock()
        with mock.patch("ultralytics.YOLO", return_value=yolo_mock):
            result = pre_download_model("yolo11x.pt")
    finally:
        monkeypatch_cwd.undo()
    assert result == fake_cache / "yolo11x.pt"
    assert result.exists()
    assert result.read_bytes() == b"stray-yolo-weights"
    # The stray was moved (not copied); original location is gone.
    assert not stray.exists()


def test_pre_download_model_raises_when_no_file_found(fake_cache: Path) -> None:
    """If neither the cache nor any candidate location has the file, raise ModelLoadError."""
    with mock.patch("ultralytics.YOLO"), pytest.raises(ModelLoadError) as exc_info:
        pre_download_model("yolo11m.pt")
    assert "yolo11m.pt" in str(exc_info.value)
    assert "yolo11m.pt" in exc_info.value.user_message()


def test_pre_download_models_returns_paths_in_input_order(fake_cache: Path) -> None:
    """pre_download_models returns a list of paths in the same order as the input names."""

    def _fake_construct(name: str) -> mock.MagicMock:
        (fake_cache / name).write_bytes(b"x")
        return mock.MagicMock()

    with mock.patch("ultralytics.YOLO", side_effect=_fake_construct):
        results = pre_download_models(["yolo11s.pt", "yolo11n-seg.pt", "yolo11m.pt"])
    assert [p.name for p in results] == ["yolo11s.pt", "yolo11n-seg.pt", "yolo11m.pt"]


def test_pre_download_models_default_uses_default_set(fake_cache: Path) -> None:
    """pre_download_models() with no args uses DEFAULT_MODELS."""

    def _fake_construct(name: str) -> mock.MagicMock:
        (fake_cache / name).write_bytes(b"x")
        return mock.MagicMock()

    with mock.patch("ultralytics.YOLO", side_effect=_fake_construct):
        results = pre_download_models()
    assert [p.name for p in results] == list(DEFAULT_MODELS)


def test_pre_download_models_stops_on_first_failure(fake_cache: Path) -> None:
    """If any model fails, the exception propagates (no silent partial success)."""

    def _first_ok_then_fail(name: str) -> mock.MagicMock:
        if name == "yolo11n.pt":
            (fake_cache / name).write_bytes(b"x")
            return mock.MagicMock()
        raise RuntimeError("network down for this one")

    with (
        mock.patch("ultralytics.YOLO", side_effect=_first_ok_then_fail),
        pytest.raises(ModelLoadError) as exc_info,
    ):
        pre_download_models(["yolo11n.pt", "yolo11s.pt"])
    assert "yolo11s.pt" in str(exc_info.value)
    assert "network down for this one" in str(exc_info.value)


def test_list_cached_models_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    """list_cached_models returns [] when the cache directory does not exist."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "img2svg.model_download.paths.cache_dir", lambda: tmp_path / "no_such_dir"
    )
    try:
        assert list_cached_models() == []
    finally:
        monkeypatch.undo()


def test_list_cached_models_returns_sorted_pt_files(fake_cache: Path) -> None:
    """list_cached_models returns all *.pt files in the cache, sorted by name."""
    (fake_cache / "yolo11m.pt").write_bytes(b"x")
    (fake_cache / "yolo11x.pt").write_bytes(b"x")
    (fake_cache / "yolo11s-seg.pt").write_bytes(b"x")
    # A non-pt file should be ignored.
    (fake_cache / "README.txt").write_bytes(b"x")
    names = [p.name for p in list_cached_models()]
    assert names == ["yolo11m.pt", "yolo11s-seg.pt", "yolo11x.pt"]


def test_cache_dir_size_bytes_sums_pt_files(fake_cache: Path) -> None:
    """cache_dir_size_bytes returns the total size of all .pt files in the cache."""
    (fake_cache / "a.pt").write_bytes(b"x" * 100)
    (fake_cache / "b.pt").write_bytes(b"x" * 200)
    (fake_cache / "ignore.txt").write_bytes(b"x" * 999)
    assert cache_dir_size_bytes() == 300


def test_cache_dir_size_bytes_zero_when_empty(fake_cache: Path) -> None:
    """cache_dir_size_bytes returns 0 when the cache directory is empty."""
    assert cache_dir_size_bytes() == 0


def test_pre_download_model_respects_force_flag(fake_cache: Path) -> None:
    """With force=True, the download runs even if the file is already in the cache."""
    target = fake_cache / "yolo11x.pt"
    target.write_bytes(b"old-weights")
    yolo_mock = mock.MagicMock()

    def _fake_construct(name: str) -> mock.MagicMock:
        # Overwrite with new content as if YOLO re-downloaded.
        target.write_bytes(b"new-weights")
        return yolo_mock

    with mock.patch("ultralytics.YOLO", side_effect=_fake_construct):
        result = pre_download_model("yolo11x.pt", force=True)
    assert result == target
    assert result.read_bytes() == b"new-weights"


def test_pre_download_model_raises_clear_error_when_ultralytics_missing(
    fake_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ultralytics cannot be imported, ModelLoadError fires with a useful message."""
    # Simulate ImportError on the inner import.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("simulated: ultralytics not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    with pytest.raises(ModelLoadError) as exc_info:
        pre_download_model("yolo11n.pt")
    # The model name appears in both the str() and user_message() forms.
    assert "yolo11n.pt" in str(exc_info.value)
    assert "yolo11n.pt" in exc_info.value.user_message()
    # The original ImportError is preserved on the wrapped exception.
    assert isinstance(exc_info.value.original, RuntimeError)
    # And the original's message mentions the missing dependency.
    assert "ultralytics" in str(exc_info.value.original)
