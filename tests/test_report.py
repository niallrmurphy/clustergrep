import json

from clustergrep.cli import EXIT_MATCH, main


def run(capsys, *argv):
    code = main(["--color", "never", *[str(arg) for arg in argv]])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def corpus_and_thesaurus(tmp_path):
    corpus = tmp_path / "records.jsonl"
    corpus.write_text(
        json.dumps({"body": "Press Shift-Escape to close the panel"}) + "\n"
        + json.dumps({"body": "The prisoner escaped through a tunnel"}) + "\n"
        + json.dumps({"body": "A breakout followed overnight"}) + "\n"
    )
    thesaurus = tmp_path / "terms.tsv"
    thesaurus.write_text("escape\tbreakout\t0.25\n")
    return corpus, thesaurus


def test_html_report_embeds_findings_and_source_identity(tmp_path, capsys):
    corpus, thesaurus = corpus_and_thesaurus(tmp_path)
    report = tmp_path / "findings.html"
    code, out, err = run(
        capsys,
        "--backend", "thesaurus", "--thesaurus", thesaurus,
        "--html-report", report,
        "escape", corpus,
    )

    assert code == EXIT_MATCH
    assert out == ""
    assert "wrote" in err
    document = report.read_text()
    assert "clustergrep / findings explorer" in document
    assert "The prisoner escaped" in document
    assert str(corpus).replace("<", "\\u003c") in document
    assert '"schema":"clustergrep.report/v1"' in document
    assert '"json":{"body":"The prisoner escaped through a tunnel"}' in document
    assert 'id="prettyJson" type="checkbox" checked' in document
    assert "max-height:calc(100vh - 2rem);overflow-y:auto" in document
    assert "<mark>" in document
    assert "Contexts within terms" in document
    assert "WordNet senses that led from your query" in document


def test_saved_filter_removes_context_before_reporting(tmp_path, capsys):
    corpus, thesaurus = corpus_and_thesaurus(tmp_path)
    report = tmp_path / "findings.html"
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps({
        "version": 1,
        "input": {"json_fields": ["body"]},
        "exclude": {"context_regex": ["(?i)shift[- ]escape"]},
    }))
    code, _, _ = run(
        capsys,
        "--backend", "thesaurus", "--thesaurus", thesaurus,
        "--filter", filter_path,
        "--html-report", report,
        "escape", corpus,
    )

    assert code == EXIT_MATCH
    document = report.read_text()
    assert "Shift-Escape" not in document
    assert "The prisoner escaped" in document


def test_report_limit_bounds_embedded_records_but_not_scan_count(tmp_path, capsys):
    corpus, thesaurus = corpus_and_thesaurus(tmp_path)
    report = tmp_path / "findings.html"
    _, _, err = run(
        capsys,
        "--backend", "thesaurus", "--thesaurus", thesaurus,
        "--html-report", report, "--report-limit", "1",
        "escape", corpus,
    )
    document = report.read_text()
    assert '"lines_matched":3' in document
    assert '"records_embedded":1' in document
    assert "1 of 3 matching line(s)" in err

    payload = document.split("<script>const DATA=", 1)[1].split(";\nconst $", 1)[0]
    data = json.loads(payload)
    assert data["metadata"]["term_counts"] == {"escape": 2, "breakout": 1}
    assert sum(
        group["count"] for group in data["metadata"]["contexts"]["escape"]
    ) == 2
