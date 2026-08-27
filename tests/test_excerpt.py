"""One record of a large JSONL can be pages of text, so the match has to be
shown with a little context rather than the whole line."""

import pytest

from clustergrep.excerpt import ELLIPSIS, Window, excerpts


def find(line, word):
    i = line.index(word)
    return (i, i + len(word))


def shown(window: Window):
    """What the window's spans actually point at, after the shift."""
    return [window.text[a:b] for a, b in window.spans]


def test_the_span_still_points_at_the_match_after_shifting():
    """The whole thing is worthless if the highlight lands on the wrong text."""
    line = "x" * 300 + " the detainee escaped through the corridor " + "y" * 300
    (w,) = excerpts(line, [find(line, "escaped")], 80)
    assert shown(w) == ["escaped"]


def test_context_appears_on_both_sides():
    line = "a" * 200 + " the detainee escaped overnight " + "b" * 200
    (w,) = excerpts(line, [find(line, "escaped")], 100)
    before, after = w.text.split("escaped")
    assert before.strip(ELLIPSIS).strip()
    assert after.strip(ELLIPSIS).strip()


def test_roughly_the_requested_width():
    line = "z " * 500
    line = line[:400] + "escaped " + line[400:]
    (w,) = excerpts(line, [find(line, "escaped")], 100)
    assert 80 <= len(w.text) <= 130


def test_a_short_line_is_returned_whole_and_unmarked():
    line = "the detainee escaped"
    (w,) = excerpts(line, [find(line, "escaped")], 100)
    assert w.text == line
    assert ELLIPSIS not in w.text


def test_ellipsis_marks_only_the_side_that_was_cut():
    line = "escaped " + "y" * 400
    (w,) = excerpts(line, find(line, "escaped") and [find(line, "escaped")], 60)
    assert not w.text.startswith(ELLIPSIS)
    assert w.text.endswith(ELLIPSIS)


def test_a_match_at_the_end_still_gets_a_full_window():
    """Clamping naively would return a half-empty window at the line's edge."""
    line = "y" * 400 + " escaped"
    (w,) = excerpts(line, [find(line, "escaped")], 100)
    assert len(w.text) >= 90


def test_overlapping_windows_merge_into_one():
    """Five mentions in two sentences should read as one excerpt, not five."""
    line = "x" * 200 + " a jailbreak and a breakout and an escape here " + "y" * 200
    windows = excerpts(
        line,
        [find(line, "jailbreak"), find(line, "breakout"), find(line, "escape")],
        100,
    )
    assert len(windows) == 1
    assert sorted(shown(windows[0])) == ["breakout", "escape", "jailbreak"]


def test_distant_matches_stay_separate():
    line = "jailbreak" + "x" * 600 + "breakout"
    windows = excerpts(line, [find(line, "jailbreak"), find(line, "breakout")], 60)
    assert len(windows) == 2
    assert shown(windows[0]) == ["jailbreak"]
    assert shown(windows[1]) == ["breakout"]


def test_a_match_longer_than_the_window_is_still_shown_whole():
    """Truncating the thing you searched for would be a strange way to show it."""
    line = "x" * 100 + " underground railroad " + "y" * 100
    (w,) = excerpts(line, [find(line, "underground railroad")], 10)
    assert "underground railroad" in w.text


def test_cuts_land_on_whitespace_not_mid_word():
    line = "alpha bravo charlie delta escaped echo foxtrot golf hotel india"
    (w,) = excerpts(line, [find(line, "escaped")], 30)
    body = w.text.strip(ELLIPSIS)
    words = line.split()
    # Every token in the excerpt is a whole word from the line, so neither
    # cut fell inside one.
    assert body.split() == [t for t in body.split() if t in words]
    assert body.split()[0] in words and body.split()[-1] in words


def test_no_matches_gives_no_windows():
    assert excerpts("nothing here", [], 100) == []
