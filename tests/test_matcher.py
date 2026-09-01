import pytest

from clustergrep.cluster import Cluster, Term
from clustergrep.matcher import Matcher, inflected


def cluster(*terms, threshold=0.5, query="escape"):
    return Cluster.build(query, "test", [Term(d, t) for d, t in terms], threshold)


@pytest.mark.parametrize(
    "word, expected",
    [
        ("escape", {"escapes", "escaped", "escaping"}),
        ("carry", {"carries", "carried", "carrying"}),
        ("watch", {"watches", "watched", "watching"}),
        ("flee", {"flees", "fleeing"}),
        ("stop", {"stopped", "stopping"}),
        ("panic", {"panics", "panicked", "panicking"}),
        ("die", {"dies", "died", "dying"}),
        ("tie", {"ties", "tied", "tying"}),
        ("wolf", {"wolves"}),
        ("knife", {"knives"}),
        ("hero", {"heroes"}),
        ("safe", {"safer", "safest"}),
        ("fly", {"flew", "flown"}),
    ],
)
def test_inflection_covers_the_real_forms(word, expected):
    assert expected <= inflected(word)


def test_inflection_bends_a_phrase_at_either_end():
    forms = inflected("fly the coop")
    assert "flies the coop" in forms
    assert "flying the coop" in forms
    assert inflected("underground railroad") >= {"underground railroads"}


def test_short_and_non_alphabetic_words_are_not_inflected():
    assert inflected("go") == set()
    assert inflected("v2") == set()


def test_inflection_avoids_common_spurious_spellings():
    forms = inflected("open")
    assert "openned" not in forms
    assert "openning" not in forms

    forms = inflected("panic")
    assert "paniced" not in forms
    assert "panicing" not in forms
    assert "panicced" not in forms
    assert "paniccing" not in forms

    forms = inflected("tie")
    assert "tiing" not in forms


def test_matches_an_inflected_form_at_its_terms_distance():
    m = Matcher(cluster((0.15, "flee")))
    hit = m.best("two inmates fleeing the yard")
    assert hit.text == "fleeing"
    assert hit.term.text == "flee"
    assert hit.distance == 0.15


def test_a_space_in_a_term_matches_a_hyphen_or_underscore():
    m = Matcher(cluster((0.25, "fly the coop")))
    for text in ["fly the coop", "fly-the-coop", "fly_the_coop"]:
        assert m.best(text) is not None, text


def test_matching_respects_word_boundaries():
    m = Matcher(cluster((0.0, "run"), threshold=0.0, query="run"), inflect=False)
    assert m.best("they run home") is not None
    assert m.best("a runcible spoon") is None
    assert m.best("overrun") is None


def test_longer_terms_win_over_their_prefixes():
    m = Matcher(cluster((0.25, "break"), (0.15, "break loose")), inflect=False)
    hit = m.best("the detainee break loose")
    assert hit.text == "break loose"


def test_case_sensitivity_is_selectable():
    assert Matcher(cluster((0.15, "flee"))).best("FLEE now") is not None
    assert Matcher(cluster((0.15, "flee")), ignore_case=False).best("FLEE") is None


def test_best_is_the_nearest_match_not_the_first():
    m = Matcher(cluster((0.4, "leak"), (0.15, "flee")), inflect=False)
    assert m.best("leak then flee").term.text == "flee"


def test_lexicon_supplies_irregular_forms():
    m = Matcher(cluster((0.15, "flee")))
    assert m.best("they fled").term.text == "flee"


def test_threshold_zero_without_inflection_matches_only_the_word():
    m = Matcher(cluster(threshold=0.0), inflect=False)
    assert m.best("the escape route") is not None
    assert m.best("he escaped") is None


def test_a_capitalised_query_survives_case_sensitive_matching():
    """Lexicons are lower case; the spelling the user typed is not."""
    m = Matcher(cluster(query="Guards", threshold=0.0), ignore_case=False, inflect=False)
    assert m.best("Guards reported a breakout") is not None
    assert m.best("guards reported a breakout") is None
