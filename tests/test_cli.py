import json
import shutil
import subprocess

import pytest

from clustergrep.cli import EXIT_ERROR, EXIT_MATCH, EXIT_NO_MATCH, main

CORPUS = """The prisoner escaped through the chute.
Guards reported a breakout on B wing.
Routine inspection, nothing to report.
Two inmates were fleeing the yard.
Coolant leakage in the reactor housing.
"""

THESAURUS = """escape\tbreakout\t0.25
escape\tflee\t0.15
escape\tleakage\t0.40
"""


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "incidents.log"
    path.write_text(CORPUS)
    return path


@pytest.fixture
def thesaurus(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text(THESAURUS)
    return path


def run(capsys, *argv):
    code = main(["--color", "never", *[str(a) for a in argv]])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def cg(capsys, thesaurus, *argv):
    return run(capsys, "-b", "thesaurus", "--thesaurus", thesaurus, *argv)


def test_matches_report_distance_and_the_term_that_fired(capsys, corpus, thesaurus):
    code, out, _ = cg(capsys, thesaurus, "-t", "0.3", "escape", corpus)
    assert code == EXIT_MATCH
    assert "1:0.00:escape:" in out
    assert "2:0.25:breakout:" in out
    assert "4:0.15:flee:" in out
    # leakage sits at 0.40, beyond the threshold asked for.
    assert "leakage" not in out


def test_the_threshold_is_the_users_control_over_recall(capsys, corpus, thesaurus):
    _, narrow, _ = cg(capsys, thesaurus, "-t", "0.2", "escape", corpus)
    _, wide, _ = cg(capsys, thesaurus, "-t", "0.4", "escape", corpus)
    assert len(narrow.splitlines()) < len(wide.splitlines())
    assert "leakage" in wide


def test_no_match_exits_one(capsys, corpus, thesaurus):
    code, out, _ = cg(capsys, thesaurus, "-t", "0", "escape", "--no-inflect", corpus)
    assert code == EXIT_NO_MATCH
    assert out == ""


@pytest.mark.skipif(shutil.which("grep") is None, reason="grep is not installed")
@pytest.mark.parametrize("word", ["Guards", "escaped", "reactor"])
def test_threshold_zero_without_inflection_agrees_with_grep(tmp_path, capsys, word):
    """The headline claim: no distance tolerated means no cleverness at all.

    Run against the default backend, since this is a promise about the tool as
    people will actually invoke it. The capitalised case matters: every term a
    lexicon supplies is lower case, so a cased query has to survive on its own.
    """
    pytest.importorskip("nltk")
    path = tmp_path / "c.log"
    path.write_text(CORPUS)
    code, out, err = run(
        capsys, "-t", "0", "--no-inflect", "-s", "--no-distance", word, str(path)
    )
    if code == EXIT_ERROR:
        pytest.skip(err)
    expected = subprocess.run(
        ["grep", "-n", "-w", "-F", word, str(path)],
        capture_output=True, text=True,
    ).stdout
    assert out == expected


def test_invert_match_reports_the_conceptual_gaps(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "-v", "escape", corpus)
    assert "Routine inspection" in out
    assert "breakout" not in out


def test_count_only_prints_a_number(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "-c", "escape", corpus)
    assert out.strip() == "3"


def test_files_with_matches_prints_the_path(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "-l", "escape", corpus)
    assert out.strip() == str(corpus)


def test_only_matching_prints_the_hit_not_the_line(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "-o", "escape", corpus)
    assert "prisoner" not in out
    assert "escaped" in out


def test_json_output_is_one_object_per_match(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "--json", "escape", corpus)
    rows = [json.loads(line) for line in out.splitlines()]
    assert {r["term"] for r in rows} == {"escape", "breakout", "flee"}
    assert rows[0]["matched"] == "escaped"


def test_sort_puts_the_nearest_matches_first(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.4", "--sort", "escape", corpus)
    distances = [float(line.split(":")[1]) for line in out.splitlines()]
    assert distances == sorted(distances)


def test_max_count_stops_early(capsys, corpus, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.4", "-m", "2", "escape", corpus)
    assert len(out.splitlines()) == 2


def test_stats_reports_which_terms_actually_fired(capsys, corpus, thesaurus):
    _, _, err = cg(capsys, thesaurus, "-t", "0.4", "--stats", "escape", corpus)
    assert "breakout" in err and "flee" in err


def test_explain_shows_the_cluster_and_searches_nothing(capsys, thesaurus):
    code, out, _ = cg(capsys, thesaurus, "-t", "0.3", "--explain", "escape")
    assert code == EXIT_MATCH
    assert "breakout" in out and "0.25" in out
    assert "prisoner" not in out


def test_explain_tsv_is_a_thesaurus_file(capsys, thesaurus):
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "--explain", "--tsv", "escape")
    rows = [l for l in out.splitlines() if not l.startswith("#")]
    assert all(len(r.split("\t")) >= 3 for r in rows)


def test_recursive_search_labels_each_file(capsys, tmp_path, thesaurus):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.log").write_text(CORPUS)
    (tmp_path / "sub" / "b.txt").write_text(CORPUS)
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "-r",
                   "--include", "*.log", "escape", tmp_path / "sub")
    assert "a.log" in out
    assert "b.txt" not in out


def test_a_bad_threshold_is_rejected(capsys, corpus):
    with pytest.raises(SystemExit) as exc:
        main(["-t", "2.0", "escape", str(corpus)])
    assert exc.value.code == EXIT_ERROR


def test_a_missing_thesaurus_reports_a_remedy(capsys, corpus, tmp_path):
    code, _, err = run(capsys, "-b", "thesaurus", "--thesaurus",
                       tmp_path / "absent.tsv", "escape", corpus)
    assert code == EXIT_ERROR
    assert "try:" in err


def test_a_missing_file_is_reported_but_not_fatal(capsys, corpus, thesaurus):
    code, out, err = cg(capsys, thesaurus, "-t", "0.3", "escape",
                        corpus, "nowhere.log")
    assert "no such file" in err
    assert code == EXIT_MATCH


def test_binary_files_are_skipped(capsys, tmp_path, thesaurus):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"escape\x00escape")
    code, out, _ = cg(capsys, thesaurus, "-t", "0.3", "escape", blob)
    assert out == ""
    assert code == EXIT_NO_MATCH


def test_wordnet_end_to_end(capsys, corpus):
    pytest.importorskip("nltk")
    code, out, err = run(capsys, "-t", "0.25", "escape", str(corpus))
    if code == EXIT_ERROR:
        pytest.skip(err)
    assert "breakout" in out
    assert "flee" in out  # via the irregular form "fled"


def test_paths_reports_where_data_lives(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CLUSTERGREP_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("CLUSTERGREP_CACHE", str(tmp_path / "c"))
    code, out, _ = run(capsys, "--paths")
    assert code == EXIT_MATCH
    assert str(tmp_path / "d") in out and str(tmp_path / "c") in out


def test_paths_needs_no_search_word(capsys):
    """--paths and --install-data must work before anything is set up."""
    assert run(capsys, "--paths")[0] == EXIT_MATCH


# grep takes its options anywhere on the line, and so must this. Before Python
# 3.12 a single option between the word and a file broke parsing; the last two
# cases here broke on every version, including the one this was developed on.
@pytest.mark.parametrize(
    "argv",
    [
        ["escape", "CORPUS"],
        ["escape", "--stats", "CORPUS"],
        ["escape", "CORPUS", "--stats"],
        ["--stats", "escape", "CORPUS"],
        ["escape", "CORPUS", "--stats", "CORPUS"],
        ["escape", "-t", "0.3", "CORPUS", "CORPUS"],
        ["escape", "CORPUS", "-t", "0.3", "--sort", "CORPUS"],
    ],
)
def test_options_may_appear_anywhere_among_the_file_names(
    capsys, corpus, thesaurus, argv
):
    filled = [str(corpus) if a == "CORPUS" else a for a in argv]
    code, out, err = cg(capsys, thesaurus, *filled)
    assert code == EXIT_MATCH, err
    assert "escaped" in out


def test_every_file_named_is_actually_searched(capsys, tmp_path, thesaurus):
    """The recovered positionals must not be quietly dropped."""
    a = tmp_path / "a.log"
    b = tmp_path / "b.log"
    a.write_text("the prisoner escaped\n")
    b.write_text("a breakout on B wing\n")
    _, out, _ = cg(capsys, thesaurus, "-t", "0.3", "escape", str(a), "--sort", str(b))
    assert "a.log" in out and "b.log" in out


def test_a_mistyped_flag_is_still_an_error_not_a_file_name(capsys, corpus, thesaurus):
    """Recovering stray positionals must not turn a typo into a file name."""
    with pytest.raises(SystemExit) as exc:
        cg(capsys, thesaurus, "-t", "0.3", "escape", "--nonsense", str(corpus))
    assert exc.value.code == EXIT_ERROR
    assert "unrecognized arguments: --nonsense" in capsys.readouterr().err


def test_an_ambiguous_abbreviation_is_rejected(capsys, corpus, thesaurus):
    with pytest.raises(SystemExit):
        cg(capsys, thesaurus, "escape", "--sen", str(corpus))
    assert "ambiguous option: --sen" in capsys.readouterr().err


def test_unambiguous_abbreviations_work_as_they_do_in_grep(capsys, corpus, thesaurus):
    """getopt_long accepts unique prefixes, so --stat means --stats."""
    code, _, err = cg(capsys, thesaurus, "-t", "0.3", "escape", "--stat", str(corpus))
    assert code == EXIT_MATCH
    assert "match(es) from" in err
