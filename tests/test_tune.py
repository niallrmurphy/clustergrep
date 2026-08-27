"""--tune removes cluster terms that are drowning the query.

The failure it exists for: searching a real 6GB corpus for "escape" returned
395,587 matches, of which "run", "break" and "miss" were 91%. Those are
legitimate WordNet neighbours -- "run" is a lemma of scat.v.01, "flee; take
to one's heels" -- and useless at scale.
"""

import pytest

from clustergrep.cli import EXIT_MATCH, main

THESAURUS = (
    "escape\tbreakout\t0.25\n"
    "escape\tjailbreak\t0.25\n"
    "escape\tflee\t0.20\n"
    "escape\trun\t0.25\n"
    "escape\tmiss\t0.15\n"
)


def corpus(escapes=40, noise=4000):
    lines = []
    for i in range(noise):
        lines.append("the meeting will run long and you cannot miss it")
    for i in range(escapes):
        lines.append("the detainee escaped during a jailbreak")
    return "\n".join(lines) + "\n"


@pytest.fixture
def files(tmp_path):
    (tmp_path / "t.tsv").write_text(THESAURUS)
    (tmp_path / "c.log").write_text(corpus())
    return tmp_path


def cg(capsys, files, *argv, stdin=None):
    if stdin is not None:
        import io, sys
        sys.stdin = io.StringIO(stdin)
    try:
        code = main(["--color", "never", "-b", "thesaurus",
                     "--thesaurus", str(files / "t.tsv"), *argv])
    finally:
        if stdin is not None:
            import sys
            sys.stdin = sys.__stdin__
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_the_noise_is_dropped_and_the_signal_kept(capsys, files):
    code, out, err = cg(capsys, files, "-t", "0.25", "escape",
                        str(files / "c.log"), "--summary", "--tune")
    assert code == EXIT_MATCH
    assert "run" not in out and "miss" not in out
    assert "jailbreak" in out
    assert "escape" in out


def test_it_says_what_it_dropped(capsys, files):
    """Deciding quietly would be the one unacceptable way for this to work."""
    _, _, err = cg(capsys, files, "-t", "0.25", "escape",
                   str(files / "c.log"), "--summary", "--tune")
    assert "--tune dropped" in err
    assert "run" in err and "miss" in err
    assert "x" in err  # the ratio is shown
    assert "re-run without --tune" in err


def test_without_tune_the_noise_dominates(capsys, files):
    """The baseline the flag exists to fix."""
    _, out, _ = cg(capsys, files, "-t", "0.25", "escape",
                   str(files / "c.log"), "--summary")
    assert "run" in out and "miss" in out


def test_it_declines_when_the_query_is_too_rare_to_measure(capsys, files, tmp_path):
    """Every ratio is measured against the query's own count, so too few of
    those makes the calculation noise rather than evidence."""
    (files / "rare.log").write_text("the meeting will run long\n" * 500
                                    + "he escaped\n")
    _, out, err = cg(capsys, files, "-t", "0.25", "escape",
                     str(files / "rare.log"), "--summary", "--tune")
    assert "too few to measure" in err
    assert "run" in out  # nothing was dropped


def test_it_declines_when_there_is_no_clear_boundary(capsys, files):
    """A corpus with no noise still has a largest gap somewhere; --tune must
    not invent a boundary to cut at."""
    (files / "even.log").write_text(
        ("the detainee escaped\na jailbreak at dawn\ntwo inmates flee\n" * 40)
    )
    _, out, err = cg(capsys, files, "-t", "0.25", "escape",
                     str(files / "even.log"), "--summary", "--tune")
    assert "dropped" not in err
    assert "jailbreak" in out and "flee" in out


def test_nothing_below_the_query_rate_is_ever_dropped(capsys, files):
    _, out, err = cg(capsys, files, "-t", "0.25", "escape",
                     str(files / "c.log"), "--summary", "--tune")
    # jailbreak fires exactly as often as escape, so it is at 1.0x and safe.
    assert "jailbreak" in out


def test_it_works_on_a_stream_that_cannot_be_rewound(capsys, files):
    """The documented pipeline is `rg ... | clustergrep`, so the sampled lines
    have to be handed back rather than re-read."""
    code, out, err = cg(capsys, files, "-t", "0.25", "escape",
                        "--summary", "--tune", stdin=corpus())
    assert code == EXIT_MATCH
    assert "--tune dropped" in err
    assert "jailbreak" in out


def test_the_stream_is_counted_in_full_not_just_the_sample(capsys, files):
    """A replayed sample must not be dropped from, or double-counted in, the
    totals."""
    _, piped, _ = cg(capsys, files, "-t", "0.25", "escape",
                     "--summary", "--tune", stdin=corpus())
    _, from_file, _ = cg(capsys, files, "-t", "0.25", "escape",
                         str(files / "c.log"), "--summary", "--tune")
    assert piped.splitlines()[0] == from_file.splitlines()[0]


def test_tune_also_prunes_what_patterns_emits(capsys, files):
    """So the prefilter for the next run drops the noise too."""
    _, wide, _ = cg(capsys, files, "-t", "0.25", "escape", "--patterns")
    assert "run" in wide.split()
