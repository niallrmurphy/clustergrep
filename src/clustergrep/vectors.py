"""Vector backend: distance as cosine distance in a word-embedding space.

WordNet knows relations that a lexicographer wrote down, which makes it precise
and explainable but blind to mere association: "escape" and "jail" have no
WordNet path worth the name, though no reader would be surprised to find them
in the same sentence. Embeddings are the opposite trade -- they capture the
association but cannot tell you why, and their distances are not a path you
can argue with, only a number. Use --explain and read the list.

Reads the usual whitespace-delimited text format, so GloVe and word2vec text
exports both work:

    escape 0.1234 -0.5678 ...
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterable

from .cluster import BackendError, Term, normalise
from .paths import cache_dir

# Without a cap a generous threshold over a 400k-word vocabulary returns tens
# of thousands of terms, which is not a cluster, it is a dictionary.
DEFAULT_TOP_K = 300


class VectorBackend:
    """Nearest neighbours by cosine distance over a word-embedding model."""

    name = "vectors"

    def __init__(self, path: str | Path, top_k: int = DEFAULT_TOP_K) -> None:
        self.path = Path(path)
        self.top_k = top_k
        self._vocab: dict[str, int] | None = None
        self._matrix = None

    def _load(self) -> None:
        if self._matrix is not None:
            return
        try:
            import numpy as np
        except ImportError as exc:
            raise BackendError(
                "the vectors backend needs numpy",
                remedy="pip install 'clustergrep[vectors]'",
            ) from exc
        if not self.path.exists():
            raise BackendError(
                f"vector model not found: {self.path}",
                remedy="point --model at a GloVe or word2vec text file",
            )

        cached = _cache_path(self.path)
        words = matrix = None
        if cached.exists() and cached.stat().st_mtime >= self.path.stat().st_mtime:
            try:
                blob = np.load(cached, allow_pickle=False)
                words = _decode_vocab(blob["vocab"])
                matrix = blob["matrix"]
            except (OSError, ValueError, KeyError):
                # A truncated or stale-format cache is a performance problem,
                # not a correctness one: fall back to the real model.
                words = matrix = None
        if matrix is None:
            words, matrix = _read_text_model(self.path, np)
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cached, vocab=_encode_vocab(words, np), matrix=matrix)

        self._vocab = {w: i for i, w in enumerate(words)}
        self._words = words
        self._matrix = matrix

    def expand(self, word: str, threshold: float) -> Iterable[Term]:
        import numpy as np

        self._load()
        key = normalise(word).replace(" ", "_")
        index = self._vocab.get(key)
        if index is None:
            raise BackendError(
                f"{word!r} is not in the vocabulary of {self.path.name}",
                remedy="try a more common spelling, or --backend wordnet",
            )

        # Rows were L2-normalised at load, so a matrix-vector product is
        # exactly the cosine similarity against every word at once.
        sims = self._matrix @ self._matrix[index]
        want = min(self.top_k + 1, sims.shape[0])
        top = np.argpartition(-sims, want - 1)[:want]
        for i in top[np.argsort(-sims[top])]:
            distance = float(min(1.0, max(0.0, 1.0 - sims[i])))
            if distance > threshold:
                break
            text = self._words[i]
            yield Term(
                0.0 if i == index else max(distance, 0.01),
                text,
                f"cos={sims[i]:.3f}",
            )


def _read_text_model(path: Path, np):
    """Parse a whitespace-delimited vector file into (words, unit matrix)."""
    opener = gzip.open if path.suffix == ".gz" else open
    words: list[str] = []
    rows: list[list[float]] = []
    width = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                # word2vec text files open with a "<rows> <dims>" header.
                continue
            if width is None:
                width = len(parts) - 1
            elif len(parts) - 1 != width:
                raise BackendError(
                    f"{path}:{lineno}: expected {width} dimensions, "
                    f"got {len(parts) - 1}"
                )
            words.append(parts[0].lower())
            rows.append([float(x) for x in parts[1:]])
    if not rows:
        raise BackendError(f"no vectors found in {path}")

    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return words, matrix / norms


def _encode_vocab(words: list[str], np):
    """Pack the vocabulary into a byte array.

    Caches are loaded with allow_pickle=False, since a cache file is just a
    file on disk and unpickling one would be an arbitrary-code-execution hole
    for the sake of a speed-up. That rules out object arrays, and a fixed-width
    unicode array would pad every word out to the longest one -- tens of
    megabytes of spaces for a full-size model. Newline-joined UTF-8 is both
    compact and pickle-free.
    """
    return np.frombuffer("\n".join(words).encode("utf-8"), dtype=np.uint8)


def _decode_vocab(buffer) -> list[str]:
    return buffer.tobytes().decode("utf-8").split("\n")


def _cache_path(model: Path) -> Path:
    """Where the parsed form of a model is kept.

    Parsing a 400k-word text model takes seconds every run; the .npz form
    loads in milliseconds. Keyed by absolute path so two models with the same
    basename do not collide, and invalidated by mtime. Lives in the cache
    directory, not the data directory, because it is always rebuildable.
    """
    digest = hashlib.sha256(str(model.resolve()).encode()).hexdigest()[:16]
    return cache_dir() / f"vectors-{model.stem}-{digest}.npz"


