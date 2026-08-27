import io

import pytest

from clustergrep.cluster import BackendError, Cluster, Term
from clustergrep.thesaurus import ThesaurusBackend, parse, to_tsv

SAMPLE = """
# a comment
escape\tjailbreak\t0.25\tfrom wordnet
escape\texfil\t0.30
fear\tdread\t0.20
"""


def test_parse_reads_concepts_terms_and_distances():
    table = parse(io.StringIO(SAMPLE))
    assert sorted(table) == ["escape", "fear"]
    assert {t.text for t in table["escape"]} == {"jailbreak", "exfil"}
    assert table["escape"][0].distance == 0.25
    assert table["escape"][0].via == "from wordnet"


def test_comments_and_blank_lines_are_ignored():
    assert parse(io.StringIO("\n\n# nothing here\n")) == {}


@pytest.mark.parametrize(
    "row, fragment",
    [
        ("escape\tjailbreak\n", "expected at least"),
        ("escape\tjailbreak\tnear\n", "not a number"),
        ("escape\tjailbreak\t7\n", "outside"),
    ],
)
def test_a_malformed_row_names_the_line_rather_than_being_skipped(row, fragment):
    """A silently dropped row is a silently missing search term."""
    with pytest.raises(BackendError) as exc:
        parse(io.StringIO(row), source="t.tsv")
    assert "t.tsv:1" in str(exc.value)
    assert fragment in str(exc.value)


def test_backend_reads_a_file(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text(SAMPLE)
    b = ThesaurusBackend(path)
    assert b.concepts() == ["escape", "fear"]
    assert {t.text for t in b.expand("escape", 1.0)} == {"jailbreak", "exfil"}


def test_an_unknown_concept_says_how_to_add_one(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text(SAMPLE)
    with pytest.raises(BackendError) as exc:
        list(ThesaurusBackend(path).expand("hope", 1.0))
    assert "--explain hope --tsv" in exc.value.remedy


def test_a_missing_file_says_how_to_make_one(tmp_path):
    with pytest.raises(BackendError) as exc:
        ThesaurusBackend(tmp_path / "absent.tsv").table
    assert "not found" in str(exc.value)


def test_tsv_round_trips_through_the_parser():
    """--explain --tsv must produce a file this backend can read back."""
    original = Cluster.build(
        "escape", "wordnet",
        [Term(0.15, "flee", "escape.v.01"), Term(0.25, "fly the coop")],
        0.4,
    )
    table = parse(io.StringIO(to_tsv(original)))
    assert {t.text: t.distance for t in table["escape"]} == {
        "escape": 0.0, "flee": 0.15, "fly the coop": 0.25,
    }
