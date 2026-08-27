"""A wheel cannot run code at install time, so the corpus is offered on first
use instead. The safety property that makes that acceptable: it happens only
when a person is at a terminal to agree."""

import io

import pytest

from clustergrep import cli
from clustergrep.cluster import BackendError
from clustergrep.wordnet import INSTALL_REMEDY

MISSING = BackendError("the WordNet corpus is not installed", remedy=INSTALL_REMEDY)


@pytest.fixture
def installer(monkeypatch):
    """Records whether the download was attempted."""
    calls = []

    def fake(force=False):
        calls.append(force)
        return True, "WordNet installed at /somewhere"

    monkeypatch.setattr("clustergrep.wordnet.install_data", fake)
    return calls


def answer(monkeypatch, text):
    monkeypatch.setattr("builtins.input", lambda *a: text)


def interactive(monkeypatch, yes=True):
    monkeypatch.setattr(cli, "_can_prompt", lambda: yes)


def test_nothing_is_downloaded_without_a_terminal(monkeypatch, installer):
    """The property that makes this safe: a pipeline or CI job never fetches."""
    interactive(monkeypatch, False)
    answer(monkeypatch, "y")
    err = io.StringIO()
    assert cli.offer_install(MISSING, err) is False
    assert installer == []
    assert err.getvalue() == ""


def test_a_person_at_a_terminal_is_asked(monkeypatch, installer):
    interactive(monkeypatch)
    answer(monkeypatch, "y")
    err = io.StringIO()
    assert cli.offer_install(MISSING, err) is True
    assert installer == [False]
    assert "Download it into" in err.getvalue()


def test_enter_accepts(monkeypatch, installer):
    interactive(monkeypatch)
    answer(monkeypatch, "")
    assert cli.offer_install(MISSING, io.StringIO()) is True
    assert installer == [False]


@pytest.mark.parametrize("reply", ["n", "no", "q", "later"])
def test_declining_downloads_nothing(monkeypatch, installer, reply):
    interactive(monkeypatch)
    answer(monkeypatch, reply)
    assert cli.offer_install(MISSING, io.StringIO()) is False
    assert installer == []


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
def test_an_interrupted_prompt_declines(monkeypatch, installer, interrupt):
    interactive(monkeypatch)

    def raiser(*a):
        raise interrupt()

    monkeypatch.setattr("builtins.input", raiser)
    assert cli.offer_install(MISSING, io.StringIO()) is False
    assert installer == []


def test_only_this_one_failure_is_ever_offered(monkeypatch, installer):
    """A missing thesaurus or vector model is not something we can fetch."""
    interactive(monkeypatch)
    answer(monkeypatch, "y")
    other = BackendError("thesaurus file not found", remedy="write one")
    assert cli.offer_install(other, io.StringIO()) is False
    assert installer == []


def test_a_failed_download_is_reported_as_failure(monkeypatch):
    interactive(monkeypatch)
    answer(monkeypatch, "y")
    monkeypatch.setattr(
        "clustergrep.wordnet.install_data", lambda force=False: (False, "no network")
    )
    err = io.StringIO()
    assert cli.offer_install(MISSING, err) is False
    assert "no network" in err.getvalue()


def test_the_search_proceeds_after_accepting(monkeypatch, tmp_path):
    """The whole point: answering yes leaves you with results, not an error."""
    from clustergrep.cluster import Term

    class Backend:
        name = "wordnet"

        def __init__(self):
            self.ready = False

        def expand(self, word, threshold):
            if not self.ready:
                raise MISSING
            return [Term(0.2, "breakout")]

    backend = Backend()
    interactive(monkeypatch)
    answer(monkeypatch, "y")

    def install(force=False):
        backend.ready = True
        return True, "installed"

    monkeypatch.setattr("clustergrep.wordnet.install_data", install)

    args = cli.build_parser().parse_args(["escape"])
    cluster = cli.build_cluster(args, backend, io.StringIO())
    assert "breakout" in cluster.distances()


def test_the_error_still_stands_when_declined(monkeypatch):
    class Backend:
        name = "wordnet"

        def expand(self, word, threshold):
            raise MISSING

    interactive(monkeypatch)
    answer(monkeypatch, "n")
    args = cli.build_parser().parse_args(["escape"])
    with pytest.raises(BackendError):
        cli.build_cluster(args, Backend(), io.StringIO())
