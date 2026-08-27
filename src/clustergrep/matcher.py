"""Turning a cluster into something that can be run against a line of text.

Matching is a plain compiled regex over an alternation of every pattern
in the cluster. That is deliberate: once the cluster has been decided, finding
it is ordinary, fast, predictable string matching with no model in the loop.
Everything uncertain about clustergrep is confined to building the cluster,
which is why --explain can show you the whole of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from .cluster import Cluster, Term

# A space in a term may appear in text as a space, an underscore or a hyphen:
# "fly the coop", "fly-the-coop" and "fly_the_coop" are the same match.
_GAP = r"[\s_-]+"

_VOWELS = "aeiou"


@dataclass(frozen=True)
class Match:
    """One occurrence of a cluster term in a line."""

    term: Term
    text: str
    start: int
    end: int

    @property
    def distance(self) -> float:
        return self.term.distance


class Matcher:
    """Searches lines for any pattern belonging to a cluster."""

    def __init__(
        self,
        cluster: Cluster,
        *,
        inflect: bool = True,
        ignore_case: bool = True,
        word_variants: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        self.cluster = cluster
        self.inflect = inflect
        self.ignore_case = ignore_case

        # Every pattern we will accept, mapped back to the term that
        # justifies it. On collision the nearest term wins, so a word that is
        # both a synonym and a distant inflection reports the flattering
        # distance rather than the punishing one.
        self._lookup: dict[str, Term] = {}
        for term in cluster.terms:
            forms = {term.text}
            if inflect:
                forms |= expand(term.text, _forms)
            if word_variants is not None:
                forms |= expand(term.text, lambda w: {_key(v) for v in word_variants(w)})
            for form in forms:
                form = _key(form)
                held = self._lookup.get(form)
                if held is None or term.distance < held.distance:
                    self._lookup[form] = term

        # Every pattern the lexicon gave us is lower case, so under
        # --case-sensitive a query typed as "Guards" would compile to /guards/
        # and match nothing. The user's own capitalisation is theirs to keep,
        # so it goes in as an extra pattern; attribution still happens through
        # the lower-cased lookup, so it resolves to the query term as normal.
        patterns = set(self._lookup)
        typed = " ".join(cluster.query.split())
        key = _key(typed)
        if typed and key in self._lookup:
            patterns.add(typed)
            if not ignore_case and typed != key:
                # grep -F "Guards" does not match "guards", so neither do we:
                # under --case-sensitive the query is spelled exactly once.
                patterns.discard(key)

        # Longest first so that "break loose" wins over "break" at the same
        # position; Python's alternation takes the first branch that matches,
        # not the longest.
        forms = sorted(patterns, key=lambda f: (-len(f), f))
        body = "|".join(_GAP.join(re.escape(w) for w in form.split()) for form in forms)
        # \b would misbehave for forms that begin or end with punctuation;
        # these lookarounds mean the same thing for words and stay correct.
        flags = re.IGNORECASE if ignore_case else 0
        self.regex = re.compile(rf"(?<!\w)(?:{body})(?!\w)", flags)

    def __len__(self) -> int:
        return len(self._lookup)

    def patterns(self) -> list[str]:
        """Every string this matcher would recognise, including inflections.

        This is what a fast prefilter needs, and what --patterns prints. The
        cluster alone is not enough: it holds "flee", while the text holds
        "fled", and a filter built from cluster terms would drop the line
        before clustergrep ever saw it.

        Note the singular ``self.regex`` is the compiled alternation built
        from these; this returns the literal strings that went into it.
        """
        return sorted(self._lookup)

    def finditer(self, line: str) -> Iterator[Match]:
        for m in self.regex.finditer(line):
            term = self._lookup.get(_key(m.group(0)))
            if term is None:  # pragma: no cover - every form is in the lookup
                continue
            yield Match(term=term, text=m.group(0), start=m.start(), end=m.end())

    def search(self, line: str) -> list[Match]:
        return list(self.finditer(line))

    def best(self, line: str) -> Match | None:
        """The nearest match on this line, which is how a line is scored."""
        return min(self.finditer(line), key=lambda m: m.distance, default=None)


def _key(text: str) -> str:
    """Canonical form used for both lookup keys and matched text."""
    return " ".join(text.replace("_", " ").replace("-", " ").lower().split())


def expand(term: str, generator: Callable[[str], Iterable[str]]) -> set[str]:
    """Apply a per-word form generator to the open positions of a term.

    For a single word that is just the word. For a phrase it is the first and
    last words, because English inflects either end depending on the shape of
    the phrase: "fly the coop" bends at the head ("flies the coop") while
    "underground railroad" bends at the tail ("underground railroads"). We
    generate both rather than work out which, so "fly the cooped" is also
    produced -- a string that costs a few bytes of regex and will never occur
    in real text. Over-generating is the cheap failure here; missing the real
    form is the expensive one.
    """
    words = term.split()
    if not words:
        return set()
    positions = {0, len(words) - 1}
    out = set()
    for i in positions:
        for form in generator(words[i]):
            if form and form != words[i]:
                out.add(" ".join(words[:i] + [form] + words[i + 1:]))
    return out


def inflected(term: str) -> set[str]:
    """Regular English inflections of a term.

    Suffix rules only, so this reaches escape/escapes/escaped/escaping and
    stop/stopped/stopping but never flee/fled or run/ran. Irregular forms are
    a lexicon rather than a rule, so they arrive from a backend's variants().
    """
    return expand(term, _forms)


def _forms(word: str) -> set[str]:
    if len(word) < 3 or not word.isalpha():
        return set()

    out = {word + "s"}
    if word.endswith("ee"):
        # flee -> flees, fleeing; the silent-e rules below would give "fleing".
        out |= {word + "ing", word + "d"}
    elif word.endswith("e"):
        out |= {word + "d", word[:-1] + "ing"}
    elif word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS:
        out -= {word + "s"}
        out |= {word[:-1] + "ies", word[:-1] + "ied", word + "ing"}
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        out -= {word + "s"}
        out |= {word + "es", word + "ed", word + "ing"}
    else:
        out |= {word + "ed", word + "ing"}
        # Consonant-vowel-consonant doubles the final letter: stop -> stopping.
        if (
            word[-1] not in _VOWELS + "wxy"
            and word[-2] in _VOWELS
            and word[-3] not in _VOWELS
        ):
            out |= {word + word[-1] + "ed", word + word[-1] + "ing"}
    return out
