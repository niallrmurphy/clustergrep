"""Cutting a readable window out of a line that is far too long to print.

One record of a large JSONL can be pages of text, which makes the usual
grep-shaped output useless: the match is in there somewhere, surrounded by
several kilobytes of everything else. An excerpt keeps the part a reader
actually wants -- the match and enough either side to judge it -- and throws
the rest away.

This is presentation only. Nothing here changes what matched or how far away
it was; it decides what you get shown.
"""

from __future__ import annotations

from dataclasses import dataclass

ELLIPSIS = "…"

# How far to look for a space when avoiding a cut through the middle of a
# word. Beyond this the tidier boundary costs more context than it is worth,
# so the window is cut where it falls.
SNAP = 12


@dataclass(frozen=True)
class Window:
    """A slice of a line, with the match positions rewritten to suit it."""

    text: str
    spans: tuple[tuple[int, int], ...]


def excerpts(line: str, spans, width: int) -> list[Window]:
    """Windows of about ``width`` characters, one per match, overlaps merged.

    Merging matters more than it sounds: a paragraph mentioning the concept
    five times in two sentences should produce one readable excerpt, not five
    nearly identical ones.
    """
    spans = sorted(spans)
    if not spans:
        return []

    bounds = [_window(line, start, end, width) for start, end in spans]

    merged: list[tuple[int, int, list[tuple[int, int]]]] = []
    for (lo, hi), span in zip(bounds, spans):
        if merged and lo <= merged[-1][1]:
            prev_lo, prev_hi, held = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi), held + [span])
        else:
            merged.append((lo, hi, [span]))

    out = []
    for lo, hi, held in merged:
        text = line[lo:hi]
        prefix = ELLIPSIS if lo > 0 else ""
        suffix = ELLIPSIS if hi < len(line) else ""
        shift = lo - len(prefix)
        out.append(
            Window(
                text=prefix + text + suffix,
                spans=tuple((s - shift, e - shift) for s, e in held),
            )
        )
    return out


def _window(line: str, start: int, end: int, width: int) -> tuple[int, int]:
    """Bounds of a window of ``width`` centred on one match.

    A match longer than the window still appears whole -- truncating the thing
    you searched for would be a strange way to show it to you.
    """
    width = max(width, end - start)
    centre = (start + end) // 2
    lo = centre - width // 2
    hi = lo + width

    # Keep the full width when the match sits near either end of the line,
    # rather than returning a half-empty window.
    if lo < 0:
        lo, hi = 0, min(len(line), width)
    if hi > len(line):
        hi = len(line)
        lo = max(0, hi - width)

    lo = min(lo, start)
    hi = max(hi, end)
    return _snap(line, lo, hi, start, end)


def _snap(line: str, lo: int, hi: int, start: int, end: int) -> tuple[int, int]:
    """Move the cuts to whitespace so the window does not begin mid-word."""
    if lo > 0:
        for i in range(lo, min(lo + SNAP, start)):
            if line[i].isspace():
                lo = i + 1
                break
    if hi < len(line):
        for i in range(hi, max(hi - SNAP, end), -1):
            if i < len(line) and line[i].isspace():
                hi = i
                break
    return lo, hi
