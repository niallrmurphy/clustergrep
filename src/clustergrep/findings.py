"""Bounded, explainable clustering of matching lines.

The search backends decide which words express a concept. This module handles
the separate question of which matching lines tell the same story. It uses only
words present in the lines plus WordNet sense provenance already attached to a
term; no second model gets to reinterpret the document after matching.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")

# A matched term alone is weak evidence that two lines discuss the same event:
# an ambiguous query word may occur in several unrelated contexts. A WordNet
# root sense is stronger evidence, while repeated context words and phrases do
# the rest. These are vector weights, not user-facing semantic distances.
_TERM_WEIGHT = 1.25
_FACET_WEIGHT = 2.5
_BIGRAM_WEIGHT = 0.75
_JOIN_SIMILARITY = 0.24
_MAX_LINE_TOKENS = 512
_MAX_CENTROID_FEATURES = 2048
_PRUNE_CENTROID_AT = 3072

_TOKEN = re.compile(r"[^\W_][\w'-]*", re.UNICODE)
_STOPWORDS = frozenset(
    "a about after again against all also am an and any are as at be because "
    "been before being between both but by can could did do does doing down "
    "during each few for from further had has have having he her here hers "
    "herself him himself his how i if in into is it its itself just me more "
    "most my myself no nor not now of off on once only or other our ours "
    "ourselves out over own same she should so some such than that the their "
    "theirs them themselves then there these they this those through to too "
    "under until up very was we were what when where which while who whom why "
    "will with would you your yours yourself yourselves".split()
)
_FEATURE_PREFIX = "\0"
_TERM_PREFIX = f"{_FEATURE_PREFIX}term:"
_FACET_PREFIX = f"{_FEATURE_PREFIX}facet:"
_BIGRAM_PREFIX = f"{_FEATURE_PREFIX}bigram:"


@dataclass(frozen=True)
class FindingGroup(Generic[T]):
    """One line cluster and the line nearest its accumulated centroid."""

    id: int
    count: int
    representative: T
    representative_distance: float
    terms: tuple[tuple[str, int], ...]
    keywords: tuple[str, ...]
    members: tuple[T, ...] = ()


@dataclass(frozen=True)
class _Finding(Generic[T]):
    item: T
    text: str
    vector: dict[str, float]
    terms: tuple[str, ...]
    distance: float
    index: int


@dataclass
class _Group(Generic[T]):
    count: int
    vector: Counter[str]
    terms: Counter[str]
    representative: _Finding[T]
    first_index: int
    members: list[T] | None


class FindingClusterer(Generic[T]):
    """Online clustering with memory bounded by the requested group count.

    A new group is opened only for a line sufficiently unlike every existing
    centroid. Once ``limit`` groups exist, every later line joins its nearest
    one. Group counts therefore cover the complete input even for a stream.
    Only one representative per group is retained unless ``retain_members``
    is requested for a separately bounded input such as an HTML report sample.
    """

    def __init__(
        self, limit: int, query: str, *, retain_members: bool = False
    ) -> None:
        if limit < 1:
            raise ValueError("finding cluster limit must be at least one")
        self.limit = limit
        self.query = query
        self.retain_members = retain_members
        self._groups: list[_Group[T]] = []
        self._seen = 0

    def add(
        self,
        item: T,
        *,
        text: str,
        terms: Iterable[str],
        matched: Iterable[str],
        facets: Iterable[str] = (),
        distance: float,
    ) -> int:
        canonical_terms = tuple(sorted(set(terms)))
        finding = _Finding(
            item=item,
            text=text,
            vector=_features(
                self.query,
                text,
                canonical_terms,
                tuple(matched),
                tuple(facets),
            ),
            terms=canonical_terms,
            distance=distance,
            index=self._seen,
        )
        self._seen += 1

        if not self._groups:
            self._groups.append(_new_group(finding, self.retain_members))
            return finding.index

        similarities = [_cosine(finding.vector, group.vector) for group in self._groups]
        nearest = max(range(len(self._groups)), key=lambda i: (similarities[i], -i))
        if len(self._groups) < self.limit and similarities[nearest] < _JOIN_SIMILARITY:
            self._groups.append(_new_group(finding, self.retain_members))
            return finding.index
        _add_to_group(self._groups[nearest], finding)
        return self._groups[nearest].first_index

    def groups(self) -> list[FindingGroup[T]]:
        if not self._groups:
            return []

        group_frequency = Counter(
            feature
            for group in self._groups
            for feature in group.vector
            if not feature.startswith(_FEATURE_PREFIX)
        )
        total_groups = len(self._groups)
        ordered = sorted(
            self._groups,
            key=lambda group: (
                -group.count,
                group.representative.distance,
                group.representative.index,
            ),
        )
        return [
            FindingGroup(
                id=group.first_index,
                count=group.count,
                representative=group.representative.item,
                representative_distance=group.representative.distance,
                terms=tuple(sorted(group.terms.items(), key=lambda row: (-row[1], row[0]))),
                keywords=_keywords(group, group_frequency, total_groups),
                members=tuple(group.members or ()),
            )
            for group in ordered
        ]


def _new_group(finding: _Finding[T], retain_members: bool) -> _Group[T]:
    return _Group(
        count=1,
        vector=Counter(finding.vector),
        terms=Counter(finding.terms),
        representative=finding,
        first_index=finding.index,
        members=[finding.item] if retain_members and finding.item is not None else (
            [] if retain_members else None
        ),
    )


def _add_to_group(group: _Group[T], finding: _Finding[T]) -> None:
    group.count += 1
    if group.members is not None and finding.item is not None:
        group.members.append(finding.item)
    group.vector.update(finding.vector)
    group.terms.update(finding.terms)
    if len(group.vector) > _PRUNE_CENTROID_AT:
        kept = sorted(
            group.vector.items(), key=lambda row: (-row[1], row[0])
        )[:_MAX_CENTROID_FEATURES]
        group.vector.clear()
        group.vector.update(dict(kept))

    current = group.representative
    current_key = _representative_key(current, group.vector)
    candidate_key = _representative_key(finding, group.vector)
    if candidate_key > current_key:
        group.representative = finding


def _representative_key(finding: _Finding, centroid: Counter[str]) -> tuple:
    return (
        round(_cosine(finding.vector, centroid), 12),
        -finding.distance,
        len(finding.terms),
        -len(finding.text),
        -finding.index,
    )


def _features(
    query: str,
    text: str,
    terms: tuple[str, ...],
    matched: tuple[str, ...],
    facets: tuple[str, ...],
) -> dict[str, float]:
    excluded = set(_tokens(" ".join((query, *terms, *matched))))
    tokens = [
        token
        for token in _tokens(text)
        if token not in excluded and token not in _STOPWORDS and len(token) > 1
    ]
    if len(tokens) > _MAX_LINE_TOKENS:
        half = _MAX_LINE_TOKENS // 2
        tokens = tokens[:half] + tokens[-half:]

    vector = {token: 1.0 for token in tokens}
    for left, right in zip(tokens, tokens[1:]):
        vector[f"{_BIGRAM_PREFIX}{left} {right}"] = _BIGRAM_WEIGHT
    for term in terms:
        vector[f"{_TERM_PREFIX}{_normalise(term)}"] = _TERM_WEIGHT
    for facet in facets:
        vector[f"{_FACET_PREFIX}{facet}"] = _FACET_WEIGHT
    return vector


def _tokens(text: str) -> list[str]:
    return [
        token.lower().removesuffix("'s")
        for token in _TOKEN.findall(text)
        if any(character.isalpha() for character in token)
    ]


def _normalise(text: str) -> str:
    return " ".join(text.replace("_", " ").lower().split())


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(feature, 0.0) for feature, value in left.items())
    if dot == 0.0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm)


def _keywords(
    group: _Group,
    group_frequency: Counter[str],
    total_groups: int,
) -> tuple[str, ...]:
    scored = []
    minimum_occurrences = 1 if group.count == 1 else 2
    for token, occurrences in group.vector.items():
        if (
            token.startswith(_FEATURE_PREFIX)
            or occurrences < minimum_occurrences
        ):
            continue
        prevalence = occurrences / group.count
        distinctiveness = math.log((total_groups + 1) / group_frequency[token])
        scored.append((prevalence * distinctiveness, token))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return tuple(token for _, token in scored[:3])
