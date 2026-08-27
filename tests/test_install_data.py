"""The corpus is downloaded by the user, not shipped. These are the promises
that behaviour has to keep."""

import pytest

pytest.importorskip("nltk")

from clustergrep import wordnet
from clustergrep.cluster import BackendError


def test_an_existing_corpus_is_reused_rather_than_downloaded(monkeypatch, tmp_path):
    """Someone with nltk already installed must not be made to download 10MB
    a second time into a different directory."""
    monkeypatch.setenv("CLUSTERGREP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(wordnet, "corpus_location", lambda: "/somewhere/wordnet.zip")

    called = []
    monkeypatch.setattr(
        "nltk.download", lambda *a, **k: called.append(a) or True
    )
    ok, message = wordnet.install_data()
    assert ok and not called
    assert "already installed" in message
    assert "/somewhere/wordnet.zip" in message


def test_the_download_goes_to_our_data_directory(monkeypatch, tmp_path):
    target = tmp_path / "data"
    monkeypatch.setenv("CLUSTERGREP_DATA", str(target))
    monkeypatch.setattr(wordnet, "corpus_location", lambda: None)

    seen = {}
    def fake_download(name, download_dir=None, quiet=False):
        seen["name"] = name
        seen["dir"] = download_dir
        return True
    monkeypatch.setattr("nltk.download", fake_download)

    ok, _ = wordnet.install_data()
    assert ok
    assert seen["name"] == "wordnet"
    assert seen["dir"] == str(target)
    assert target.is_dir()


def test_a_failed_download_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setenv("CLUSTERGREP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(wordnet, "corpus_location", lambda: None)
    monkeypatch.setattr("nltk.download", lambda *a, **k: False)

    ok, message = wordnet.install_data()
    assert not ok
    assert "CLUSTERGREP_DATA" in message


def test_our_data_directory_is_searched_last(monkeypatch, tmp_path):
    """Appending means a corpus the user already has wins the lookup."""
    import nltk

    monkeypatch.setenv("CLUSTERGREP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(nltk.data, "path", ["/pre-existing"])
    wordnet.register_data_path()
    assert nltk.data.path == ["/pre-existing", str(tmp_path / "data")]
    wordnet.register_data_path()  # idempotent, not appended twice
    assert nltk.data.path.count(str(tmp_path / "data")) == 1


def test_a_missing_corpus_points_at_the_installer(monkeypatch):
    from nltk.corpus import wordnet as wn

    monkeypatch.setattr(wordnet, "register_data_path", lambda: None)
    monkeypatch.setattr(wn, "synsets", _raise_lookup)
    with pytest.raises(BackendError) as exc:
        wordnet._load_wordnet()
    assert exc.value.remedy == "clustergrep --install-data"


def _raise_lookup(*a, **k):
    raise LookupError("no corpus")
