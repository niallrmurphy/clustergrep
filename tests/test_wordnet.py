import pytest

from clustergrep.cluster import BackendError, Cluster
from clustergrep.wordnet import WordNetBackend

pytest.importorskip("nltk")


@pytest.fixture(scope="module")
def backend():
    b = WordNetBackend()
    try:
        b.senses("escape")
    except BackendError as exc:
        pytest.skip(str(exc))
    return b


def expand(backend, word, threshold, **kw):
    b = WordNetBackend(**kw) if kw else backend
    return Cluster.build(word, "wordnet", b.expand(word, threshold), threshold)


def test_threshold_zero_gives_the_word_and_nothing_else(backend):
    assert [t.text for t in expand(backend, "escape", 0.0).terms] == ["escape"]


def test_synonyms_arrive_before_narrower_terms(backend):
    d = expand(backend, "escape", 0.3).distances()
    assert d["escape"] == 0.0
    assert d["flight"] < d["jailbreak"]


def test_the_obvious_neighbours_are_present(backend):
    d = expand(backend, "escape", 0.25).distances()
    for word in ("jailbreak", "breakout", "getaway", "flee", "elude"):
        assert word in d, word


def test_raising_the_threshold_only_ever_adds_terms(backend):
    near = set(expand(backend, "escape", 0.2).distances())
    far = set(expand(backend, "escape", 0.4).distances())
    assert near < far


def test_distance_is_stable_as_the_threshold_grows(backend):
    near = expand(backend, "escape", 0.2).distances()
    far = expand(backend, "escape", 0.5).distances()
    assert all(far[w] == d for w, d in near.items())


def test_a_path_summing_exactly_to_the_threshold_is_kept(backend):
    """0.15 + 0.25 is 0.4000000000000001 in binary floating point.

    Without rounding, such terms vanish at --threshold 0.4 and the tool looks
    like it has an incomplete lexicon rather than an arithmetic bug.
    """
    d = expand(backend, "escape", 0.4).distances()
    assert any(abs(v - 0.4) < 1e-9 for v in d.values())


def test_part_of_speech_restricts_the_cluster(backend):
    verbs = expand(backend, "escape", 0.3, pos="v").distances()
    nouns = expand(backend, "escape", 0.3, pos="n").distances()
    assert "elude" in verbs and "elude" not in nouns
    assert "jailbreak" in nouns and "jailbreak" not in verbs


def test_senses_are_ranked_within_a_part_of_speech_not_across_it():
    """The verb reading of a noun-first word must not be buried.

    WordNet lists all noun senses of "escape" before the first verb sense, so
    ranking senses globally would charge "flee" a penalty for nothing more
    than being a verb.
    """
    b = WordNetBackend()
    try:
        b.senses("escape")
    except BackendError as exc:
        pytest.skip(str(exc))
    d = Cluster.build("escape", "wordnet", b.expand("escape", 0.2), 0.2).distances()
    assert "elude" in d


def test_a_rarer_sense_sits_further_out(backend):
    d = expand(backend, "escape", 0.4).distances()
    # The fluid-discharge sense of "escape" is a late noun sense, so its
    # synonyms must not rank alongside the prison-break ones.
    assert d["leakage"] > d["jailbreak"]


def test_sense_selection_pins_one_reading(backend):
    d = expand(backend, "escape", 0.3, sense=0, pos="v").distances()
    assert "get away" in d
    with pytest.raises(ValueError):
        expand(backend, "escape", 0.3, sense=99)


def test_antonyms_are_opt_in(backend):
    without = expand(backend, "increase", 0.65).distances()
    with_them = expand(backend, "increase", 0.65, include_antonyms=True).distances()
    assert "decrease" in with_them
    assert "decrease" not in without


def test_unknown_words_expand_to_themselves(backend):
    assert [t.text for t in expand(backend, "gxqzzy", 0.4).terms] == ["gxqzzy"]


def test_irregular_forms_are_offered_to_the_matcher(backend):
    assert "fled" in backend.word_variants("flee")
    assert "broke" in backend.word_variants("break")


def test_expansion_of_a_broad_word_stays_bounded(backend):
    """Descending from a near-universal abstraction would return the language."""
    assert len(expand(backend, "act", 0.5).terms) < 2000
