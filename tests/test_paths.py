"""Tests for XDG path module."""
from __future__ import annotations

from pathlib import Path

import pytest

from img2svg import paths


def test_config_dir_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = paths.config_dir()
    assert result == tmp_path / ".config" / "img2svg"
    assert result.exists()


def test_config_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "custom_cfg"))
    result = paths.config_dir()
    assert result == tmp_path / "custom_cfg" / "img2svg"
    assert result.exists()


def test_data_dir_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = paths.data_dir()
    assert result == tmp_path / ".local" / "share" / "img2svg"
    assert result.exists()


def test_cache_dir_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = paths.cache_dir()
    assert result == tmp_path / ".cache" / "img2svg"
    assert result.exists()


def test_cache_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "custom_cache"))
    result = paths.cache_dir()
    assert result == tmp_path / "custom_cache" / "img2svg"


def test_model_cache_path_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = paths.model_cache_path("yolo11x.pt")
    assert result == tmp_path / ".cache" / "img2svg" / "models" / "yolo11x.pt"
    assert result.parent.exists()


def test_system_config_dir_is_freebsd_path() -> None:
    assert paths.system_config_dir() == Path("/usr/local/etc/cloudbsd/img2svg")


def test_load_config_missing_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert paths.load_config() == {}


def test_load_config_valid_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path
    (cfg_dir / "img2svg").mkdir()
    cfg_file = cfg_dir / "img2svg" / "config.toml"
    cfg_file.write_text('mode = "annotated"\nconf = 0.5\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    config = paths.load_config()
    assert config == {"mode": "annotated", "conf": 0.5}


def test_load_config_invalid_toml_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "img2svg"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text("not valid toml [[[")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert paths.load_config() == {}
