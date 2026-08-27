"""--summary and --forms exist for corpora whose lines are too big to print:
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


def test_forms_include_inflections_the_cluster_does_not_hold(capsys, files):
    """The whole point of --forms: a prefilter built from cluster terms alone
    would drop the line containing "fleeing" before clustergrep ever saw it."""
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", "--forms")
    forms = set(out.split())
    assert {"flee", "breakout", "escape"} <= forms
    assert "fleeing" in forms and "escaped" in forms


def test_forms_searches_nothing(capsys, files):
    _, out, _ = cg(capsys, files, "-t", "0.4", "escape", str(files / "c.log"),
                   "--forms")
    assert "prisoner" not in out


def test_a_prefilter_built_from_forms_loses_no_matches(capsys, files, tmp_path):
    """The property the whole recipe rests on: filtering the corpus down to
    lines containing a surface form must not change the answer."""
    corpus = files / "c.log"
    _, forms, _ = cg(capsys, files, "-t", "0.4", "escape", "--forms")
    wanted = set(forms.split("\n")) - {""}

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
