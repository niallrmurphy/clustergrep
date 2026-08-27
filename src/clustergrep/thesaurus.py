"""Thesaurus backend: clusters read from, and written to, a TSV file.

This is the backend that answers the durability problem. A WordNet or vector
cluster is only as current as its corpus, and neither knows the vocabulary of
your particular domain -- the incident tag your team coined last month, the
product name, the euphemism people actually use in tickets. Here you write the
cluster down, commit it next to the code it searches, review changes to it in
a diff, and extend it the moment a new term appears.

    # concept   term          distance   note
    escape      jailbreak     0.25
    escape      exfil         0.30       our term for it

``clustergrep --explain <word> --tsv`` emits exactly this format, so the usual
route in is to generate a cluster from WordNet, then edit it by hand and pin
it. From then on the search is entirely reproducible.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, TextIO

from .cluster import BackendError, Cluster, Term, normalise


class ThesaurusBackend:
    """Clusters defined by a TSV file rather than a lexical model."""

    name = "thesaurus"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._table: dict[str, list[Term]] | None = None

    @property
    def table(self) -> dict[str, list[Term]]:
        if self._table is None:
            if not self.path.exists():
                raise BackendError(
                    f"thesaurus file not found: {self.path}",
                    remedy=f"clustergrep --explain WORD --tsv > {self.path}",
                )
            with self.path.open(encoding="utf-8") as fh:
                self._table = parse(fh, source=str(self.path))
        return self._table

    def expand(self, word: str, threshold: float) -> Iterable[Term]:
        entries = self.table.get(normalise(word))
        if entries is None:
            raise BackendError(
                f"{word!r} has no cluster in {self.path}",
                remedy=(
                    "add rows for it, or generate a starting point with "
                    f"clustergrep --backend wordnet --explain {word} --tsv"
                ),
            )
        return entries

    def concepts(self) -> list[str]:
        return sorted(self.table)


def parse(stream: TextIO, source: str = "<stream>") -> dict[str, list[Term]]:
    """Read TSV rows into concept -> terms.

    Errors name the file and line, because a thesaurus is a file humans edit
    and a silently ignored typo is a silently missing search term.
    """
    table: dict[str, list[Term]] = {}
    for lineno, raw in enumerate(stream, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split("\t") if f.strip()]
        if len(fields) < 3:
            raise BackendError(
                f"{source}:{lineno}: expected at least "
                f"concept<TAB>term<TAB>distance, got {line!r}"
            )
        concept, term, distance, *rest = fields
        try:
            value = float(distance)
        except ValueError:
            raise BackendError(
                f"{source}:{lineno}: distance {distance!r} is not a number"
            ) from None
        if not 0.0 <= value <= 1.0:
            raise BackendError(
                f"{source}:{lineno}: distance {value} is outside 0.0-1.0"
            )
        note = rest[0] if rest else ""
        table.setdefault(normalise(concept), []).append(
            Term(value, normalise(term), note)
        )
    return table


def to_tsv(cluster: Cluster) -> str:
    """Render a cluster as a thesaurus file, ready to edit and pin."""
    out = io.StringIO()
    out.write(f"# clustergrep cluster for {cluster.query!r}\n")
    # No term count here: this file exists to be edited, so any count written
    # now is wrong as soon as someone does what the documentation tells them.
    out.write(f"# backend={cluster.backend} threshold={cluster.threshold}\n")
    out.write("# concept\tterm\tdistance\tnote\n")
    for term in cluster.terms:
        note = term.via.replace("\t", " ") if term.via else ""
        out.write(f"{cluster.query}\t{term.text}\t{term.distance:g}\t{note}\n")
    return out.getvalue()
