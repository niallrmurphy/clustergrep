"""Progress is time-triggered, not size-triggered: a search that finishes in
two seconds should say nothing, one that takes ten minutes should have spoken
after the first couple. Whether that will happen is knowable by waiting, and
not by guessing from a file size and a throughput that varies with pattern
count, line length and disk."""

import io

import pytest

from clustergrep.progress import DELAY, Progress


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(total=None, enabled=True):
    clock, out = Clock(), io.StringIO()
    return Progress(total, enabled, stream=out, clock=clock), clock, out


def test_a_fast_search_says_nothing_at_all():
    p, clock, out = make(total=1000)
    for _ in range(10):
        clock.t += DELAY / 20
        p.advance(50)
    p.done()
    assert out.getvalue() == ""


def test_a_slow_search_reports():
    p, clock, out = make(total=1000)
    clock.t = DELAY + 1
    p.advance(500)
    assert "50.0%" in out.getvalue()


def test_nothing_is_written_when_disabled():
    """The guarantee for pipelines, cron and CI."""
    p, clock, out = make(total=1000, enabled=False)
    clock.t = DELAY + 100
    p.advance(500)
    p.done()
    assert out.getvalue() == ""


def test_redraws_are_rate_limited():
    p, clock, out = make(total=100000)
    clock.t = DELAY + 1
    for _ in range(50):
        clock.t += 0.001
        p.advance(10)
    assert out.getvalue().count("\r") <= 2


def test_an_unknown_total_still_reports_progress():
    """stdin cannot be sized, and that is the documented pipeline."""
    p, clock, out = make(total=None)
    clock.t = DELAY + 1
    p.advance(5_000_000)
    text = out.getvalue()
    assert "%" not in text
    assert "MB" in text and "/s" in text


def test_a_known_total_gives_a_share_and_an_estimate():
    p, clock, out = make(total=10_000_000)
    clock.t = DELAY + 2
    p.advance(2_000_000)
    text = out.getvalue()
    assert "20.0%" in text
    assert "left" in text


def test_done_erases_so_it_cannot_collide_with_the_output():
    p, clock, out = make(total=1000)
    clock.t = DELAY + 1
    p.advance(500)
    out.truncate(0), out.seek(0)
    p.done()
    assert out.getvalue() == "\r\033[K"


def test_done_is_silent_if_nothing_was_ever_drawn():
    p, clock, out = make(total=1000)
    p.advance(10)
    p.done()
    assert out.getvalue() == ""


def test_the_share_never_exceeds_one():
    """A file that grew while being read must not report 143%."""
    p, clock, out = make(total=100)
    clock.t = DELAY + 1
    p.advance(500)
    assert "100.0%" in out.getvalue()
