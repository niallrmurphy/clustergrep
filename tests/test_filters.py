import json

import pytest

from clustergrep.cluster import BackendError, Cluster, Term
from clustergrep.filters import FilterSpec, visible_text
from clustergrep.matcher import Matcher


def write_filter(tmp_path, data):
    path = tmp_path / "filter.json"
    path.write_text(json.dumps(data))
    return FilterSpec.load(path)


def test_json_fields_extract_nested_searchable_text(tmp_path):
    spec = write_filter(tmp_path, {
        "version": 1,
        "input": {"json_fields": ["title", "payload.body"]},
    })
    line = json.dumps({
        "title": "Incident",
        "payload": {"body": "the prisoner escaped"},
        "noise": "shift-escape",
    })
    assert spec.prepare(line) == "Incident the prisoner escaped"


def test_visible_html_text_omits_tags_scripts_and_styles():
    source = "<article>Prisoner <b>escaped</b></article><script>escape()</script>"
    assert visible_text(source) == "Prisoner escaped"


def test_context_rules_apply_after_input_transforms(tmp_path):
    spec = write_filter(tmp_path, {
        "version": 1,
        "input": {"json_fields": ["body"], "html": "visible_text"},
        "exclude": {"context_regex": ["(?i)shift[- ]escape"]},
    })
    assert spec.prepare(json.dumps({"body": "Press Shift-Escape now"})) is None
    assert spec.prepare(json.dumps({"body": "<b>Prisoner escaped</b>"})) == "Prisoner escaped"


def test_excluded_semantic_terms_leave_other_matches_on_the_line(tmp_path):
    spec = write_filter(tmp_path, {
        "version": 1,
        "exclude": {"terms": ["breakout"]},
    })
    cluster = Cluster.build(
        "escape",
        "test",
        [Term(0.0, "escape"), Term(0.25, "breakout")],
        0.3,
    )
    matches = Matcher(cluster).search("an escape followed the breakout")
    kept = spec.keep_matches(matches)
    assert [match.term.text for match in kept] == ["escape"]


def test_invalid_filter_regex_names_the_field(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"exclude": {"context_regex": ["("]}}))
    with pytest.raises(BackendError, match="exclude.context_regex"):
        FilterSpec.load(path)
