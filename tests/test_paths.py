import sys

import pytest

from clustergrep import paths


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every location test starts from a machine that has stated no preference."""
    for name in ("CLUSTERGREP_DATA", "CLUSTERGREP_CACHE",
                 "XDG_DATA_HOME", "XDG_CACHE_HOME",
                 "LOCALAPPDATA", "APPDATA"):
        monkeypatch.delenv(name, raising=False)


def test_explicit_overrides_win_on_every_platform(monkeypatch, tmp_path):
    """Without these, clustergrep cannot run in a container or a read-only home."""
    monkeypatch.setenv("CLUSTERGREP_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("CLUSTERGREP_CACHE", str(tmp_path / "c"))
    for platform in ("win32", "darwin", "linux"):
        monkeypatch.setattr(sys, "platform", platform)
        assert paths.data_dir() == tmp_path / "d"
        assert paths.cache_dir() == tmp_path / "c"


def test_xdg_is_honoured_even_on_macos(monkeypatch, tmp_path):
    """Setting XDG_* is an explicit statement of preference, not a Linux tell."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert paths.data_dir() == tmp_path / "share" / "clustergrep"
    assert paths.cache_dir() == tmp_path / "cache" / "clustergrep"


def test_windows_uses_local_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert paths.data_dir() == tmp_path / "Local" / "clustergrep" / "Data"
    assert paths.cache_dir() == tmp_path / "Local" / "clustergrep" / "Cache"


def test_windows_without_localappdata_still_resolves(monkeypatch, tmp_path):
    """A missing environment variable must not produce a path under the CWD."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(paths, "_home", lambda: tmp_path)
    assert paths.data_dir() == tmp_path / "AppData" / "Local" / "clustergrep" / "Data"


def test_macos_uses_library(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(paths, "_home", lambda: tmp_path)
    assert paths.data_dir() == tmp_path / "Library" / "Application Support" / "clustergrep"
    assert paths.cache_dir() == tmp_path / "Library" / "Caches" / "clustergrep"


def test_linux_uses_xdg_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(paths, "_home", lambda: tmp_path)
    assert paths.data_dir() == tmp_path / ".local" / "share" / "clustergrep"
    assert paths.cache_dir() == tmp_path / ".cache" / "clustergrep"


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_data_and_cache_never_collide(monkeypatch, tmp_path, platform):
    """Deleting the cache must never take the downloaded corpus with it."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(paths, "_home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert paths.data_dir() != paths.cache_dir()


def test_ensure_creates_the_directory(tmp_path):
    target = tmp_path / "a" / "b"
    assert paths.ensure(target).is_dir()
    assert paths.ensure(target).is_dir()  # idempotent
