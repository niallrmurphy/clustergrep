import pytest

from clustergrep.cluster import Cluster, Term, normalise


def test_normalise_folds_separators_and_case():
    assert normalise("Fly_The__Coop") == "fly the coop"
    assert normalise("  ESCAPE  ") == "escape"


def test_term_rejects_impossible_distance():
    with pytest.raises(ValueError):
        Term(1.5, "escape")
    with pytest.raises(ValueError):
        Term(-0.1, "escape")
    with pytest.raises(ValueError):
        Term(0.2, "   ")


def test_terms_sort_nearest_first():
    terms = sorted([Term(0.5, "b"), Term(0.1, "a"), Term(0.3, "c")])
    assert [t.text for t in terms] == ["a", "c", "b"]


def test_build_drops_terms_beyond_threshold():
    c = Cluster.build("escape", "test", [Term(0.2, "flee"), Term(0.9, "leave")], 0.3)
    assert "leave" not in c.distances()
    assert c.distances()["flee"] == 0.2


def test_build_keeps_the_nearest_of_duplicate_patterns():
    c = Cluster.build(
        "escape", "test", [Term(0.4, "flee"), Term(0.2, "FLEE"), Term(0.6, "flee")], 0.9
    )
    assert c.distances()["flee"] == 0.2
    assert len(c) == 2


def test_query_is_always_present_at_zero_even_if_backend_omits_it():
    c = Cluster.build("escape", "test", [Term(0.2, "flee")], 0.3)
    assert c.distances()["escape"] == 0.0


def test_query_at_zero_survives_a_backend_claiming_otherwise():
    c = Cluster.build("escape", "test", [Term(0.7, "escape")], 0.9)
    assert c.distances()["escape"] == 0.0


def test_threshold_zero_yields_only_the_query():
    c = Cluster.build("escape", "test", [Term(0.15, "flee")], 0.0)
    assert [t.text for t in c.terms] == ["escape"]


def test_max_terms_keeps_the_nearest():
    terms = [Term(0.1 * i, f"w{i}") for i in range(1, 8)]
    c = Cluster.build("escape", "test", terms, 1.0, max_terms=3)
    assert [t.text for t in c.terms] == ["escape", "w1", "w2"]
