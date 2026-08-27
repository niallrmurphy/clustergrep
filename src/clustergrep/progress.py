"""Telling the user how far along a long search is, without being a nuisance.

Two rules decide whether anything is printed at all, and both matter:

  * only when stderr is a terminal, so a pipeline, a cron job or CI never
    finds control characters in its output;
  * only once the search has already been running long enough that the user
    is plainly waiting -- which is knowable by simply waiting, rather than by
    guessing the runtime in advance from a file size and a throughput that
    varies with pattern count, line length and disk.

The second rule is why this is time-triggered rather than size-triggered. A
search that finishes in two seconds should print nothing at all; one that
takes ten minutes should have said something after the first couple.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, TextIO

# Long enough that a quick search stays silent, short enough that a slow one
# does not feel hung. Below about a second, a progress line that appears and
# vanishes is worse than none.
DELAY = 2.0

# Redrawing faster than this buys nothing a reader can perceive and costs
# writes to a terminal that may be slow or remote.
INTERVAL = 0.25


def _clock() -> float:
    return time.monotonic()


class Progress:
    """A one-line, self-erasing progress report on stderr."""

    def __init__(
        self,
        total: int | None,
        enabled: bool,
        stream: TextIO | None = None,
        clock: Callable[[], float] = _clock,
    ) -> None:
        self.total = total or None
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.start = clock()
        self.seen = 0
        self._last = 0.0
        self._drawn = False

    def advance(self, count: int) -> None:
        self.seen += count
        if not self.enabled:
            return
        now = self.clock()
        if now - self.start < DELAY or now - self._last < INTERVAL:
            return
        self._last = now
        self._draw(now)

    def _draw(self, now: float) -> None:
        elapsed = now - self.start
        rate = self.seen / elapsed if elapsed else 0.0
        parts = [f"{_size(self.seen)} at {_size(rate)}/s"]
        if self.total:
            share = min(self.seen / self.total, 1.0)
            parts.insert(0, f"{share * 100:4.1f}%")
            if rate > 0 and share < 1.0:
                parts.append(f"{_time(( self.total - self.seen) / rate)} left")
        line = "  ".join(parts)
        self.stream.write(f"\r\033[K{line}")
        self.stream.flush()
        self._drawn = True

    def done(self) -> None:
        """Erase the report so it never collides with the actual output."""
        if self._drawn:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._drawn = False


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
