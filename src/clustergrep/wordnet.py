"""WordNet backend: distance as weighted shortest path over the lexical graph.

Every distance clustergrep reports from this backend is the cost of an actual
path through WordNet, so it can always be shown to the user as a chain of
named relations. That is the whole reason this is the default backend: a
number you can argue with beats a number you have to trust.

Edge costs below are judgements, not measurements. They encode two claims:

  * narrowing is safer than broadening. "escape" -> "jailbreak" keeps you in
    the concept; "escape" -> "movement" leaves it. So HYPERNYM costs more
    than HYPONYM.
  * a different word for the same synset is nearly free, but not free, since
    distance 0 is reserved for the literal query.
"""

from __future__ import annotations

import heapq
from typing import Iterable, Iterator

from .cluster import Backend, BackendError, Term
from .paths import data_dir, ensure

# Cost of traversing one edge of the given relation type.
SYNONYM = 0.15      # another lemma of the same synset
SIMILAR = 0.20      # similar_to, chiefly adjective satellites
DERIVATION = 0.20   # derivationally related form (escape -> escapee)
VERB_GROUP = 0.20   # senses of a verb grouped as one broad meaning
HYPONYM = 0.25      # narrower concept
ALSO_SEE = 0.35     # loosely "see also"
HYPERNYM = 0.40     # broader concept
MERONYM = 0.45      # part/member/substance, in either direction
ANTONYM = 0.60      # opposite; opt-in, see include_antonyms

# Each successive dictionary sense of the query starts this much further out,
# so that the dominant sense of a polysemous word dominates its cluster.
SENSE_PENALTY = 0.05

# Synsets shallower than this are near-universal abstractions ("act",
# "entity", "state"). Descending into their hyponyms would drag in a large
# slice of the language, so expansion stops at them.
ABSTRACT_MIN_DEPTH = 4

# The one remedy the CLI knows how to carry out itself; shared so that
# recognising it never depends on matching prose.
INSTALL_REMEDY = "clustergrep --install-data"

_POS_ALIASES = {
    "n": "n", "noun": "n",
    "v": "v", "verb": "v",
    "a": "a", "adj": "a", "adjective": "a",
    "r": "r", "adv": "r", "adverb": "r",
}


def _import_nltk():
    try:
        import nltk
    except ImportError as exc:  # pragma: no cover - depends on install
        raise BackendError(
            "the wordnet backend needs nltk",
            remedy="pip install clustergrep",
        ) from exc
    return nltk


def register_data_path() -> None:
    """Make clustergrep's own data directory visible to nltk.

    Appended rather than prepended, so that a corpus the user already has --
    from a previous nltk install, a system package, or NLTK_DATA -- is found
    first and never downloaded a second time. clustergrep only supplies a
    location for people who have no WordNet at all.
    """
    nltk = _import_nltk()
    location = str(data_dir())
    if location not in nltk.data.path:
        nltk.data.path.append(location)


def corpus_location() -> str | None:
    """Where WordNet was found, or None. Never raises."""
    try:
        nltk = _import_nltk()
        register_data_path()
    except BackendError:
        return None
    # A zip-installed corpus does not answer to "corpora/wordnet", so probe
    # for both the unpacked directory and the archive.
    for probe in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            return str(nltk.data.find(probe))
        except LookupError:
            continue
    return None


def _load_wordnet():
    register_data_path()
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("test")
    except LookupError as exc:
        raise BackendError(
            "the WordNet corpus is not installed (about 10MB, downloaded once)",
            remedy=INSTALL_REMEDY,
        ) from exc
    return wn


class WordNetBackend:
    """Expand a word by weighted shortest path over WordNet relations."""

    name = "wordnet"

    def __init__(
        self,
        pos: str | None = None,
        sense: int | None = None,
        sense_penalty: float = SENSE_PENALTY,
        include_antonyms: bool = False,
    ) -> None:
        if pos is not None:
            if pos not in _POS_ALIASES:
                raise ValueError(f"unknown part of speech: {pos!r}")
            pos = _POS_ALIASES[pos]
        self.pos = pos
        self.sense = sense
        self.sense_penalty = sense_penalty
        self.include_antonyms = include_antonyms
        self._wn = None

    @property
    def wn(self):
        if self._wn is None:
            self._wn = _load_wordnet()
        return self._wn

    def senses(self, word: str) -> list:
        """The synsets this backend would start from, in WordNet's own order."""
        lemma = word.replace(" ", "_").lower()
        found = self.wn.synsets(lemma, pos=self.pos) if self.pos else self.wn.synsets(lemma)
        if self.sense is not None:
            if not 0 <= self.sense < len(found):
                raise ValueError(
                    f"--sense {self.sense} out of range: {word!r} has "
                    f"{len(found)} sense(s) in WordNet"
                )
            return [found[self.sense]]
        return found

    def expand(self, word: str, threshold: float) -> Iterable[Term]:
        roots = self.senses(word)
        if not roots:
            return []
        return self._walk(word, roots, threshold)

    def _walk(self, word: str, roots: list, threshold: float) -> Iterator[Term]:
        """Dijkstra over synsets, emitting each synset's lemmas as it settles.

        Costs only ever grow along a path, so once a synset is settled we have
        its true shortest distance to the query and can emit its lemmas
        immediately. Anything that would exceed the threshold is never pushed,
        which is what keeps a walk over a graph this size cheap.
        """
        query_key = word.replace(" ", "_").lower()

        # (cost, tiebreak, synset, path taken to reach it)
        frontier: list[tuple[float, int, object, str]] = []
        counter = 0
        for synset, rank in _sense_ranks(roots).items():
            start = 0.0 if self.sense is not None else _round(rank * self.sense_penalty)
            if _within(start, threshold):
                heapq.heappush(frontier, (start, counter, synset, synset.name()))
                counter += 1

        settled: set[str] = set()
        while frontier:
            cost, _, synset, via = heapq.heappop(frontier)
            if synset.name() in settled:
                continue
            settled.add(synset.name())

            for lemma in synset.lemmas():
                name = lemma.name()
                if name.lower() == query_key:
                    # Literally the word the user typed, wherever we found it.
                    yield Term(0.0, name, "query")
                else:
                    # Lemmas inherit their synset's cost, but no lemma is ever
                    # nearer than SYNONYM: distance 0 means "the word you
                    # typed", and nothing else, so that -t 0 is plain grep.
                    lemma_cost = max(cost, SYNONYM)
                    if _within(lemma_cost, threshold):
                        yield Term(lemma_cost, name, via)

                # Derivation and antonymy are lemma-level in WordNet, so they
                # are followed here rather than in _edges.
                for related in lemma.derivationally_related_forms():
                    d = _round(cost + DERIVATION)
                    if _within(d, threshold):
                        yield Term(d, related.name(), f"{via} -derivation-> {related.name()}")
                if self.include_antonyms:
                    for opposite in lemma.antonyms():
                        d = _round(cost + ANTONYM)
                        if _within(d, threshold):
                            yield Term(d, opposite.name(), f"{via} -antonym-> {opposite.name()}")

            for neighbour, edge_cost, relation in self._edges(synset):
                d = _round(cost + edge_cost)
                if _within(d, threshold) and neighbour.name() not in settled:
                    heapq.heappush(
                        frontier,
                        (d, counter, neighbour, f"{via} -{relation}-> {neighbour.name()}"),
                    )
                    counter += 1

    def _edges(self, synset) -> Iterator[tuple[object, float, str]]:
        """Neighbouring synsets, each with the cost of getting there."""
        # Descending from a near-universal abstraction would pull in a large
        # slice of the language, so we decline to.
        if synset.min_depth() >= ABSTRACT_MIN_DEPTH:
            for s in synset.hyponyms() + synset.instance_hyponyms():
                yield s, HYPONYM, "hyponym"
        for s in synset.hypernyms() + synset.instance_hypernyms():
            yield s, HYPERNYM, "hypernym"
        for s in synset.similar_tos():
            yield s, SIMILAR, "similar"
        for s in synset.verb_groups():
            yield s, VERB_GROUP, "verb group"
        for s in synset.also_sees():
            yield s, ALSO_SEE, "also see"
        for s in (
            synset.part_meronyms()
            + synset.member_meronyms()
            + synset.substance_meronyms()
            + synset.part_holonyms()
            + synset.member_holonyms()
            + synset.substance_holonyms()
        ):
            yield s, MERONYM, "meronym"

    def describe_senses(self, word: str) -> list[tuple[int, str, str, list[str]]]:
        """(index, synset name, gloss, lemmas) for each sense -- for --senses."""
        return [
            (i, s.name(), s.definition(), s.lemma_names())
            for i, s in enumerate(self.senses(word))
        ]


def _sense_ranks(roots: list) -> dict:
    """Rank each sense within its own part of speech, not globally.

    WordNet lists every noun sense of a word before its first verb sense, so a
    global rank would charge the verb reading of "escape" a large penalty for
    nothing more than being a verb -- burying flee, elude and bolt beneath the
    plumbing sense of an escape valve. Ranking per part of speech keeps the
    penalty meaning what it should: this is a less common reading of the word.
    Part of speech is a separate axis, and --pos is how you choose it.
    """
    seen: dict[str, int] = {}
    ranks = {}
    for synset in roots:
        pos = synset.pos()
        ranks[synset] = seen.get(pos, 0)
        seen[pos] = ranks[synset] + 1
    return ranks


# Costs are sums of two-decimal constants, so they are rounded back to a clean
# value at every step and compared with a tolerance. Without this, a path
# costing 0.15 + 0.25 lands on 0.4000000000000001 and is silently dropped by a
# --threshold of 0.4 -- a bug that would look like WordNet being incomplete.
_EPS = 1e-9


def _round(cost: float) -> float:
    return round(cost, 4)


def _within(cost: float, threshold: float) -> bool:
    return cost <= threshold + _EPS


def install_data(force: bool = False) -> tuple[bool, str]:
    """Fetch the WordNet corpus into clustergrep's data directory.

    The corpus is not shipped in the distribution: it is roughly 10MB and
    carries Princeton's own licence, so it is the user's to fetch and the
    user's to delete. Returns (ok, message).
    """
    nltk = _import_nltk()
    register_data_path()

    existing = corpus_location()
    if existing and not force:
        return True, f"WordNet is already installed at {existing}"

    target = ensure(data_dir())
    if not nltk.download("wordnet", download_dir=str(target), quiet=True):
        return False, (
            f"could not download WordNet into {target}\n"
            "check network access, or set CLUSTERGREP_DATA to a writable path"
        )
    found = corpus_location() or str(target)
    return True, f"WordNet installed at {found}"
