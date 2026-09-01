"""--summary and --patterns exist for corpora whose lines are too big to print:
one line of a 6GB JSONL can be pages of text, so the useful output is a
report about the search rather than the text that matched."""

import json

import pytest

from clustergrep.cli import EXIT_MATCH, main

THESAURUS = "escape\tbreakout\t0.25\nescape\tflee\t0.15\nescape\tleakage\t0.40\n"
CORPUS = """the prisoner escaped
a breakout on B wing
two inmates were fleeing
routine inspection
coolant leakage detected
another breakout overnight
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "t.tsv").write_text(THESAURUS)
    (tmp_path / "c.log").write_text(CORPUS)
    return tmp_path


def cg(capsys, files, *argv):
    code = main(["--color", "never", "-b", "thesaurus",
                 "--thesaurus", str(files / "t.tsv"), *argv])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_summary_prints_no_lines_at_all(capsys, files):
    code, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                      "--summary")
    assert code == EXIT_MATCH
    for text in ("prisoner", "B wing", "coolant"):
        assert text not in out


def test_summary_counts_every_occurrence_not_just_lines(capsys, files):
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--summary")
    assert "breakout" in out
    # "breakout" is on two separate lines.
    assert [l for l in out.splitlines() if "breakout" in l][0].split()[-1] == "2"


def test_summary_goes_to_stdout_so_a_pipe_can_reach_it(capsys, files):
    """Unlike --stats, the report here is the output, not an annotation."""
    _, out, err = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                     "--summary")
    assert "cluster term(s) fired" in out
    assert err == ""


def test_summary_is_ordered_by_distance(capsys, files):
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--summary")
    rows = [l.split() for l in out.splitlines()[1:]]
    assert [float(r[0]) for r in rows] == sorted(float(r[0]) for r in rows)


def test_summary_json_is_one_object(capsys, files):
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--summary", "--json")
    d = json.loads(out)
    assert d["query"] == "escape"
    assert d["lines_matched"] == 5
    assert {t["term"] for t in d["terms"]} == {"escape", "breakout", "flee", "leakage"}
    assert all(t["distance"] is not None for t in d["terms"])


def test_summary_reports_a_search_that_found_nothing(capsys, files):
    (files / "empty.log").write_text("nothing of interest here\n")
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "empty.log"),
                   "--summary")
    assert "0 line(s) matched" in out


def test_patterns_include_inflections_the_cluster_does_not_hold(capsys, files):
    """The whole point of --patterns: a prefilter built from cluster terms alone
    would drop the line containing "fleeing" before clustergrep ever saw it."""
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", "--patterns")
    forms = set(out.split())
    assert {"flee", "breakout", "escape"} <= forms
    assert "fleeing" in forms and "escaped" in forms


def test_patterns_searches_nothing(capsys, files):
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--patterns")
    assert "prisoner" not in out


def test_a_prefilter_built_from_patterns_loses_no_matches(capsys, files, tmp_path):
    """The property the whole recipe rests on: filtering the corpus down to
    lines containing a pattern must not change the answer."""
    corpus = files / "c.log"
    _, patterns, _ = cg(capsys, files, "-t", "0.4", "escape", "--patterns")
    wanted = set(patterns.split("\n")) - {""}

    survivors = tmp_path / "survivors.log"
    kept = [
        line for line in corpus.read_text().splitlines()
        if any(f in line.lower() for f in wanted)
    ]
    survivors.write_text("\n".join(kept) + "\n")

    _, full, _ = cg(capsys, files, "-t", "0.4", "escape", str(corpus), "--summary", "--json")
    _, filtered, _ = cg(capsys, files, "-t", "0.4", "escape", str(survivors),
                        "--summary", "--json")
    assert json.loads(full)["terms"] == json.loads(filtered)["terms"]


def test_only_matching_json_omits_the_line_text(capsys, files):
    """A JSONL record can be pages long; -o means do not echo it back."""
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--json", "-o")
    rows = [json.loads(l) for l in out.splitlines()]
    assert rows and all("text" not in r for r in rows)
    assert all(r["matched"] for r in rows)


def test_json_without_only_matching_still_carries_the_line(capsys, files):
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--json")
    rows = [json.loads(l) for l in out.splitlines()]
    assert all("text" in r for r in rows)


def test_a_pinned_thesaurus_still_matches_irregular_inflections(capsys, files):
    """Irregular inflection is a fact about English, not about the backend.

    A hand-pinned cluster holding "flee" must still find "fled", or the file
    you pinned for reproducibility silently loses recall and nothing says so.
    """
    (files / "irr.log").write_text("two inmates fled the yard\n")
    code, out, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "irr.log"))
    assert code == EXIT_MATCH
    assert "fled" in out


def test_no_inflect_switches_off_irregulars_too(capsys, files):
    """--no-inflect means the exact patterns, so both halves of morphology go."""
    (files / "irr.log").write_text("two inmates fled the yard\n")
    _, out, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "irr.log"),
                   "--no-inflect")
    assert out == ""


def test_excerpt_replaces_the_page_of_text_with_a_window(capsys, files):
    page = "x " * 400 + "the detainee escaped overnight " + "y " * 400
    (files / "big.log").write_text(page + "\n")
    _, full, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "big.log"))
    _, cut, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "big.log"),
                   "--excerpt")
    assert len(cut) < len(full) / 5
    assert "escaped" in cut


def test_excerpt_width_is_respected(capsys, files):
    page = "x " * 400 + "the detainee escaped overnight " + "y " * 400
    (files / "big.log").write_text(page + "\n")
    _, narrow, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "big.log"),
                      "--excerpt", "40")
    _, wide, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "big.log"),
                    "--excerpt", "300")
    assert len(narrow) < len(wide)


def test_excerpt_json_carries_the_window_not_the_line(capsys, files):
    page = "x " * 400 + "the detainee escaped overnight " + "y " * 400
    (files / "big.log").write_text(page + "\n")
    _, out, _ = cg(capsys, files, "-t", "0.2", "escape", str(files / "big.log"),
                   "--excerpt", "--json")
    row = json.loads(out.splitlines()[0])
    assert "text" not in row
    assert "escaped" in row["excerpt"]
    assert len(row["excerpt"]) < 200


def test_excerpt_and_only_matching_are_mutually_exclusive(capsys, files):
    with pytest.raises(SystemExit):
        cg(capsys, files, "-t", "0.2", "escape", str(files / "c.log"),
           "-o", "--excerpt")
    assert "not allowed with" in capsys.readouterr().err


def test_a_nonsense_excerpt_width_is_rejected(capsys, files):
    with pytest.raises(SystemExit):
        cg(capsys, files, "-t", "0.2", "escape", str(files / "c.log"), "--excerpt", "0")
    assert "--excerpt must be at least 1" in capsys.readouterr().err


def test_each_excerpt_is_labelled_with_its_own_distance(capsys, files):
    """A window showing a distant match must not borrow the distance of a
    nearer one that appeared elsewhere in the same page."""
    line = "the detainee escaped " + "x " * 300 + " a breakout followed"
    (files / "two.log").write_text(line + "\n")
    _, out, _ = cg(capsys, files, "-t", "0.25", "escape", str(files / "two.log"),
                   "--excerpt", "60")
    rows = out.splitlines()
    assert len(rows) == 2
    assert "0.00:escape:" in rows[0]
    assert "0.25:breakout:" in rows[1]
