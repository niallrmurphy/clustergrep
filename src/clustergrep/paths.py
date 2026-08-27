"""Where clustergrep keeps things that are downloaded rather than shipped.

The WordNet corpus is about 10MB and carries its own licence, so it is not
bundled in the distribution: a user fetches it once with ``--install-data``.
That makes "where does it go" a question this module has to answer properly on
three platforms rather than assume a Unix home directory.

Two overrides come first on every platform, because a tool that cannot be told
where to put its files is a tool that cannot run in a container, a CI job or a
read-only home directory:

    CLUSTERGREP_DATA    downloaded corpora
    CLUSTERGREP_CACHE   derived files that can be deleted at any time

After that, the platform convention:

                data                                cache
    Windows     %LOCALAPPDATA%\\clustergrep\\Data     %LOCALAPPDATA%\\clustergrep\\Cache
    macOS       ~/Library/Application Support/...   ~/Library/Caches/clustergrep
    otherwise   ~/.local/share/clustergrep          ~/.cache/clustergrep

XDG_DATA_HOME and XDG_CACHE_HOME are honoured wherever they are set, including
on macOS, since setting them is an explicit statement of preference.

The distinction between the two directories is whether losing it costs
anything: the cache holds only things clustergrep can rebuild from something
else, so "delete the cache" is always safe advice.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "clustergrep"


def _home() -> Path:
    return Path.home()


def _windows_root() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return Path(local) if local else _home() / "AppData" / "Local"


def data_dir() -> Path:
    """Where downloaded corpora live. Not created; see ``ensure``."""
    override = os.environ.get("CLUSTERGREP_DATA")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / APP
    if sys.platform == "win32":
        return _windows_root() / APP / "Data"
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP
    return _home() / ".local" / "share" / APP


def cache_dir() -> Path:
    """Where derived files live. Safe to delete at any time."""
    override = os.environ.get("CLUSTERGREP_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / APP
    if sys.platform == "win32":
        return _windows_root() / APP / "Cache"
    if sys.platform == "darwin":
        return _home() / "Library" / "Caches" / APP
    return _home() / ".cache" / APP


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe() -> str:
    """Human-readable report of where everything is, for --paths.

    Exists so that "how do I remove what this installed" and "why is it not
    finding the corpus" are answerable without reading the source.
    """
    from .wordnet import corpus_location

    found = corpus_location()
    lines = [
        f"data     {data_dir()}",
        f"cache    {cache_dir()}",
        f"wordnet  {found or 'not installed -- run: clustergrep --install-data'}",
    ]
    return "\n".join(lines) + "\n"
