import pytest

np = pytest.importorskip("numpy")

from clustergrep.cluster import BackendError, Cluster
from clustergrep.vectors import VectorBackend, _cache_path


@pytest.fixture
def model(tmp_path, monkeypatch):
    """A toy space with two clearly separated neighbourhoods."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    groups = {
        "escape": 0, "jail": 0, "breakout": 0, "flee": 0,
        "coolant": 1, "reactor": 1, "valve": 1,
    }
    rng = np.random.default_rng(7)
    lines = []
    for word, group in groups.items():
        vector = rng.normal(0, 0.2, 8)
        vector[group] += 2.0
        lines.append(word + " " + " ".join(f"{x:.5f}" for x in vector))
    path = tmp_path / "toy.vec"
    path.write_text("\n".join(lines) + "\n")
    return path


def expand(model, word, threshold):
    b = VectorBackend(model)
    return Cluster.build(word, "vectors", b.expand(word, threshold), threshold)


def test_association_that_wordnet_has_no_path_for(model):
    """The reason this backend exists: 'jail' is near 'escape' by use, not by
    any relation a lexicographer wrote down."""
    assert "jail" in expand(model, "escape", 0.5).distances()


def test_the_other_neighbourhood_stays_out(model):
    d = expand(model, "escape", 0.5).distances()
    assert "coolant" not in d and "reactor" not in d


def test_the_query_is_the_only_term_at_zero(model):
    d = expand(model, "escape", 0.5).distances()
    assert d["escape"] == 0.0
    assert all(v > 0 for w, v in d.items() if w != "escape")


def test_threshold_zero_gives_the_word_alone(model):
    assert [t.text for t in expand(model, "escape", 0.0).terms] == ["escape"]


def test_an_out_of_vocabulary_word_says_so(model):
    with pytest.raises(BackendError) as exc:
        list(VectorBackend(model).expand("gxqzzy", 0.5))
    assert "vocabulary" in str(exc.value)


def test_a_missing_model_says_so(tmp_path):
    with pytest.raises(BackendError) as exc:
        list(VectorBackend(tmp_path / "absent.vec").expand("escape", 0.5))
    assert "not found" in str(exc.value)


def test_the_parsed_model_is_cached_and_gives_the_same_answer(model):
    first = expand(model, "escape", 0.5).distances()
    assert _cache_path(model).exists()
    assert expand(model, "escape", 0.5).distances() == first


def test_a_corrupt_cache_falls_back_to_the_model(model):
    expand(model, "escape", 0.5)
    cache = _cache_path(model)
    cache.write_bytes(b"not an npz file")
    # Rewritten caches must not be trusted blindly, but a bad one is a
    # performance problem, not a correctness one.
    assert "jail" in expand(model, "escape", 0.5).distances()


def test_ragged_rows_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "bad.vec"
    path.write_text("a 1.0 2.0 3.0\nb 1.0 2.0\n")
    with pytest.raises(BackendError) as exc:
        list(VectorBackend(path).expand("a", 0.5))
    assert "dimensions" in str(exc.value)
