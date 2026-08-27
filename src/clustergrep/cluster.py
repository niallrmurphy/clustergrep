"""Core types shared by every cluster backend.

A *cluster* is what clustergrep searches for in place of a fixed string: the
query word plus the other patterns that express roughly the same concept,
each carrying a distance in [0.0, 1.0].

The distance contract, which every backend must honour:

    0.0   the query word itself
    ~0.2  a term a reader would accept as interchangeable here
    ~0.5  clearly related, plausibly a different concept
    1.0   unrelated

Distances are a property of the backend's model of the language, not a
universal truth, so they are only comparable within one backend. This is why
``--explain`` exists: the cluster is always inspectable before you trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

# The query word itself always sits at exactly this distance, so that
# `--threshold 0` collapses clustergrep into plain grep.
EXACT = 0.0


@dataclass(frozen=True, order=True)
class Term:
    """One pattern to search for, and how far it sits from the query."""

    # Ordering is distance-first so that sorted() gives nearest-first.
    distance: float
    text: str
    via: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.distance <= 1.0:
            raise ValueError(f"distance out of range: {self.distance!r}")
        if not self.text.strip():
            raise ValueError("term text must not be empty")


@dataclass(frozen=True)
class Cluster:
    """The full expansion of one query, nearest term first."""

    query: str
    backend: str
    terms: tuple[Term, ...]
    threshold: float

    @classmethod
    def build(
        cls,
        query: str,
        backend: str,
        terms: Iterable[Term],
        threshold: float,
        max_terms: int | None = None,
    ) -> "Cluster":
        """Normalise a backend's raw output into a cluster.

        Deduplicates on pattern keeping the nearest distance, guarantees
        the query itself is present at distance 0, sorts nearest-first, and
        applies the threshold and term cap.
        """
        best: dict[str, Term] = {}
        for term in terms:
            if term.distance > threshold:
                continue
            key = normalise(term.text)
            if not key:
                continue
            current = best.get(key)
            if current is None or term.distance < current.distance:
                best[key] = Term(term.distance, key, term.via)

        # The query is always searched for, at distance 0, whatever the
        # backend said about it -- including when the backend knows nothing.
        best[normalise(query)] = Term(EXACT, normalise(query), "query")

        ordered = sorted(best.values(), key=lambda t: (t.distance, t.text))
        if max_terms is not None:
            ordered = ordered[:max_terms]
        return cls(query=query, backend=backend, terms=tuple(ordered), threshold=threshold)

    def __len__(self) -> int:
        return len(self.terms)

    def distances(self) -> dict[str, float]:
        return {t.text: t.distance for t in self.terms}


def normalise(text: str) -> str:
    """Fold a term into the canonical form used as a dict key and regex source.

    WordNet writes multi-word lemmas with underscores; vector models and
    hand-written thesauri use spaces. Both collapse to single spaces here, and
    the matcher is what decides that a space may appear as a hyphen in text.
    """
    return " ".join(text.replace("_", " ").lower().split())


@runtime_checkable
class Backend(Protocol):
    """A source of semantic neighbours.

    Backends are deliberately dumb: they answer "what is near this word, and
    how near", and know nothing about files, lines or regexes.
    """

    name: str

    def expand(self, word: str, threshold: float) -> Iterable[Term]:
        """Yield terms within ``threshold`` of ``word``. Order is irrelevant."""
        ...


class BackendError(RuntimeError):
    """A backend could not be used -- missing data, model or dependency.

    Carries a human-actionable remedy rather than just a failure.
    """

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy
