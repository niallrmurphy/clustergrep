"""Command line interface.

Deliberately grep-shaped: same flag names where the meaning is the same, same
exit codes (0 matched, 1 nothing matched, 2 error), same file:line: prefix.
The one place it departs is that a match carries two extra fields -- how far
the matching term sits from your query, and which term it was -- because a
result you cannot calibrate is worse than no result.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

from . import __version__
from .cluster import Backend, BackendError, Cluster, normalise
from .excerpt import excerpts
from .progress import Progress
from .matcher import Match, Matcher

EXIT_MATCH = 0
EXIT_NO_MATCH = 1
EXIT_ERROR = 2
# Conventional shell values, so `set -o pipefail` and Ctrl-C behave the way
# they do for every other tool in the pipeline.
EXIT_INTERRUPTED = 130
EXIT_BROKEN_PIPE = 141

DEFAULT_THRESHOLD = 0.4
DEFAULT_MAX_TERMS = 250

# Wide enough to judge a match in context, narrow enough that a screenful of
# them stays a screenful.
DEFAULT_EXCERPT = 100

# Read this much of a file to decide whether it is text, as grep does.
_SNIFF = 8192


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clustergrep",
        description="grep for a concept: match a word and the words that mean "
        "roughly the same thing, reporting how far each match sits from what "
        "you asked for.",
        epilog=(
            "Matching is always word-oriented, like grep -w, because concepts "
            "are words.\nWith --threshold 0 no semantic expansion happens at "
            "all, so\n\n    clustergrep -t 0 --no-inflect -s WORD FILE\n\nis "
            "exactly grep -w -F WORD FILE."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("word", nargs="?", help="the concept to search for")
    p.add_argument("files", nargs="*", help="files to search; omit to read stdin")

    g = p.add_argument_group("cluster")
    g.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        metavar="D",
        help=f"largest distance to accept, 0.0-1.0 (default {DEFAULT_THRESHOLD}); "
        "0 matches only the word itself",
    )
    g.add_argument(
        "-b", "--backend", choices=("wordnet", "vectors", "thesaurus"),
        default=os.environ.get("CLUSTERGREP_BACKEND", "wordnet"),
        help="where the cluster comes from (default wordnet)",
    )
    g.add_argument("--model", default=os.environ.get("CLUSTERGREP_MODEL"),
                   metavar="PATH", help="vector model for --backend vectors")
    g.add_argument("--thesaurus", default=os.environ.get("CLUSTERGREP_THESAURUS"),
                   metavar="PATH", help="TSV file for --backend thesaurus")
    g.add_argument("--pos", choices=("n", "v", "a", "r"),
                   help="restrict to one part of speech (noun/verb/adj/adverb)")
    g.add_argument("--sense", type=int, metavar="N",
                   help="use only WordNet sense N; see --senses")
    g.add_argument("--sense-penalty", type=float, default=None, metavar="D",
                   help="extra distance per less-common sense (default 0.05)")
    g.add_argument("--antonyms", action="store_true",
                   help="include opposites in the cluster")
    g.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS, metavar="N",
                   help=f"cap the cluster size (default {DEFAULT_MAX_TERMS})")
    inf = g.add_mutually_exclusive_group()
    inf.add_argument("--inflect", dest="inflect", action="store_true", default=True,
                     help="also match inflections: escaped, fled (the default)")
    inf.add_argument("--no-inflect", dest="inflect", action="store_false",
                     help="match only the exact patterns in the cluster")

    g = p.add_argument_group("inspect the cluster instead of searching")
    g.add_argument("--explain", action="store_true",
                   help="print the cluster and exit, without searching")
    g.add_argument("--tsv", action="store_true",
                   help="with --explain, emit thesaurus TSV to pin and edit")
    g.add_argument("--senses", action="store_true",
                   help="list the word's WordNet senses and exit")
    g.add_argument("--patterns", action="store_true",
                   help="print every pattern that would match, one per line, "
                        "and exit; feed to rg -Fw -f as a prefilter (not BSD "
                        "grep, which is far slower than no prefilter at all)")

    g = p.add_argument_group("matching")
    case = g.add_mutually_exclusive_group()
    case.add_argument("-i", "--ignore-case", dest="ignore_case",
                      action="store_true", default=True,
                      help="case-insensitive matching (the default)")
    case.add_argument("-s", "--case-sensitive", dest="ignore_case",
                      action="store_false", help="case-sensitive matching")
    g.add_argument("-v", "--invert-match", action="store_true",
                   help="print lines with no match in the cluster")
    g.add_argument("-m", "--max-count", type=int, metavar="N",
                   help="stop after N matching lines per file")
    g.add_argument("-r", "-R", "--recursive", action="store_true",
                   help="search directories recursively")
    g.add_argument("--include", action="append", metavar="GLOB", default=[],
                   help="only search files matching GLOB (repeatable)")
    g.add_argument("--exclude", action="append", metavar="GLOB", default=[],
                   help="skip files matching GLOB (repeatable)")

    g = p.add_argument_group("output")
    g.add_argument("-c", "--count", action="store_true",
                   help="print a count of matching lines per file")
    g.add_argument("-l", "--files-with-matches", action="store_true",
                   help="print only the names of files that matched")
    g.add_argument("-L", "--files-without-match", action="store_true",
                   help="print only the names of files that did not match")
    shown = g.add_mutually_exclusive_group()
    shown.add_argument("-o", "--only-matching", action="store_true",
                       help="print only the matched text")
    shown.add_argument("--excerpt", nargs="?", type=int, const=DEFAULT_EXCERPT,
                       metavar="N",
                       help=f"print about N characters around each match "
                            f"instead of the whole line (default "
                            f"{DEFAULT_EXCERPT}); for corpora where one line "
                            f"is pages of text")
    num = g.add_mutually_exclusive_group()
    num.add_argument("-n", "--line-number", dest="line_number",
                     action="store_true", default=True,
                     help="prefix with line number (the default)")
    num.add_argument("-N", "--no-line-number", dest="line_number",
                     action="store_false", help="omit line numbers")
    g.add_argument("-H", "--with-filename", dest="filename", action="store_true",
                   default=None, help="always prefix with the file name")
    g.add_argument("--no-filename", dest="filename", action="store_false",
                   help="never prefix with the file name")
    g.add_argument("--no-distance", dest="show_distance", action="store_false",
                   help="omit the distance and matched-term columns")
    g.add_argument("--sort", action="store_true",
                   help="buffer output and print nearest matches first")
    g.add_argument("--json", action="store_true",
                   help="emit one JSON object per match")
    g.add_argument("--stats", action="store_true",
                   help="alongside the results, report on stderr which terms fired")
    g.add_argument("--tune", action="store_true",
                   help="drop cluster terms that fire more often than the word "
                        "you searched for, and say on stderr which were dropped")
    g.add_argument("--summary", action="store_true",
                   help="report only which terms fired and how often; print no lines")
    g.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help="colourise output (default auto)")
    g.add_argument("--progress", choices=("auto", "always", "never"),
                   default="auto",
                   help="report how far along a slow search is, on stderr "
                        "(default auto: only at a terminal, and only once it "
                        "has been running a couple of seconds)")

    p.add_argument("--install-data", action="store_true",
                   help="download the WordNet corpus (~10MB, once), then exit")
    p.add_argument("--paths", action="store_true",
                   help="show where downloaded data and caches live, then exit")
    p.add_argument("--version", action="version", version=f"clustergrep {__version__}")
    return p


def parse_argv(parser: argparse.ArgumentParser, argv: Sequence[str] | None):
    """Parse arguments, tolerating options interleaved with file names.

    argparse matches positionals in contiguous runs, so an option sitting
    between two of them splits the run and the second group has nothing left
    to match:

        clustergrep escape a.log --stats b.log

    That fails on every Python version, and before 3.12 even a single option
    between the word and one file is enough to break it. Since grep accepts
    its options anywhere, so must this.

    So positionals are recovered rather than matched: anything argparse could
    not place is a file name, unless it looks like a flag, in which case it is
    a typo and still deserves the usual error rather than being silently
    searched for on disk.
    """
    args, extra = parser.parse_known_args(argv)
    unknown = [a for a in extra if _looks_like_flag(a)]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.files.extend(a for a in extra if not _looks_like_flag(a))
    return args


def _looks_like_flag(token: str) -> bool:
    # A bare "-" is conventionally a file name (stdin), not an option.
    return token.startswith("-") and token != "-"


# ---------------------------------------------------------------- colour


class Ink:
    """ANSI colouring, or nothing at all when the output is not a terminal."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def path(self, t: str) -> str:
        return self._wrap("35", t)

    def lineno(self, t: str) -> str:
        return self._wrap("32", t)

    def sep(self, t: str) -> str:
        return self._wrap("36", t)

    def hit(self, t: str) -> str:
        return self._wrap("1;31", t)

    def distance(self, value: float, t: str) -> str:
        # Graded so that a wall of results reads at a glance: near matches are
        # calm, far ones announce themselves as worth double-checking.
        code = "32" if value <= 0.2 else "33" if value <= 0.4 else "31"
        return self._wrap(code, t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)


def _can_prompt() -> bool:
    """Whether there is a person at a terminal to answer a question."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def offer_install(exc: BackendError, err: TextIO) -> bool:
    """Offer to fetch the corpus, and report whether it is now present.

    A wheel has no post-install hook -- installers unpack files and never
    execute anything -- so the corpus cannot be fetched when the package is
    installed. The next honest moment is the first run.

    Only when someone is at a terminal to say yes, though. In a pipeline, a
    cron job or CI the original error stands unchanged, so clustergrep never
    reaches the network because a script happened to run it.
    """
    from .wordnet import INSTALL_REMEDY

    if exc.remedy != INSTALL_REMEDY or not _can_prompt():
        return False

    from .paths import data_dir

    err.write(f"clustergrep: {exc}\n")
    err.write(f"Download it into {data_dir()} now? [Y/n] ")
    err.flush()
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        err.write("\n")
        return False
    if answer not in ("", "y", "yes"):
        return False

    from .wordnet import install_data

    ok, message = install_data()
    err.write(message + "\n")
    return ok


def build_cluster(args, backend: Backend, err: TextIO) -> Cluster:
    """Expand the query, fetching the corpus first if a person allows it."""
    try:
        terms = backend.expand(args.word, args.threshold)
    except BackendError as exc:
        if not offer_install(exc, err):
            raise
        terms = backend.expand(args.word, args.threshold)
    return Cluster.build(
        query=args.word,
        backend=backend.name,
        terms=terms,
        threshold=args.threshold,
        max_terms=args.max_terms,
    )


# --tune identifies polysemous intruders by how often they fire relative to
# the query word itself. Searching for "escape" in a corpus where "run"
# appears ten times more often than "escape" does means "run" is not being
# used in the escape sense.
#
# The cut is made at the largest multiplicative jump in that ratio rather than
# at a fixed threshold, because no fixed threshold survives contact with a
# second corpus. Measured: on a 6GB corpus the noise sat at 9.5x, 4.6x and
# 3.4x with the nearest genuine term at 0.29x; on another, noise at 80x and
# signal at 2.1x. Any constant separating one pair sits within a hair of
# misclassifying the other, while the jump itself is 11.8x and 38x -- an
# enormous margin in both. The corpus tells us where its own boundary is.
TUNE_MIN_GAP = 3.0

# Never drop a term that fires less often than the query. Without this, a
# corpus containing no noise at all would still have a largest gap somewhere,
# and --tune would invent a boundary to cut at.
TUNE_FLOOR = 1.0

# How much to read before judging. The sample only has to establish which
# terms are common, not count them exactly, so it can be a small fraction of
# a very large file -- deciding that "run" is too common should not cost as
# much as the search it is tuning.
TUNE_MIN_MATCHES = 2000
TUNE_MAX_BYTES = 32 * 1024 * 1024

# Every ratio is measured against the query's own count, so too few of those
# makes the whole calculation noise. Below this, --tune declines to act.
TUNE_MIN_QUERY = 20


def tune(matcher: Matcher, paths, err: TextIO,
         progress: Progress | None = None):
    """Drop cluster terms that are drowning the query, in one bounded pass.

    Returns (matcher, dropped, replay). ``replay`` holds lines already
    consumed from stdin, which cannot be rewound and so must be handed back
    to the real search.
    """
    counts, replay, read, matches = Counter(), [], 0, 0
    query_key = normalise(matcher.cluster.query)

    def enough() -> bool:
        return (
            matches >= TUNE_MIN_MATCHES and counts.get(query_key, 0) >= TUNE_MIN_QUERY
        ) or read >= TUNE_MAX_BYTES

    for path in paths:
        if path is not None and looks_binary(path):
            continue
        try:
            handle = (
                sys.stdin if path is None
                else path.open(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue
        try:
            for raw in handle:
                if path is None:
                    replay.append(raw)
                read += len(raw)
                if progress is not None:
                    progress.advance(len(raw))
                for m in matcher.finditer(raw.rstrip("\n").rstrip("\r")):
                    counts[m.term.text] += 1
                    matches += 1
                if enough():
                    break
        finally:
            if path is not None:
                handle.close()
        if enough():
            break

    if progress is not None:
        progress.done()

    query = counts.get(query_key, 0)
    dropped, gap = _suspects(counts, query_key, query)

    if dropped:
        kept = tuple(t for t in matcher.cluster.terms if t.text not in dropped)
        matcher = Matcher(
            replace(matcher.cluster, terms=kept),
            inflect=matcher.inflect,
            ignore_case=matcher.ignore_case,
            word_variants=matcher.word_variants,
        )

    report(dropped, query, counts, gap, err)
    return matcher, dropped, replay


def _suspects(counts, query_key: str, query: int):
    """Terms above the largest jump in fire-rate. Returns (dropped, gap)."""
    if query < TUNE_MIN_QUERY:
        return {}, 0.0
    ratios = {
        term: n / query for term, n in counts.items() if term != query_key
    }
    ranked = sorted(ratios.items(), key=lambda kv: -kv[1])
    cut, gap = None, 1.0
    for i in range(len(ranked) - 1):
        high, low = ranked[i][1], ranked[i + 1][1]
        if low > 0 and high / low > gap:
            gap, cut = high / low, i
    if cut is None or gap < TUNE_MIN_GAP:
        return {}, gap
    return {
        term: (counts[term], ratio)
        for term, ratio in ranked[: cut + 1]
        if ratio > TUNE_FLOOR
    }, gap


def report(dropped, query: int, counts, gap: float, err: TextIO) -> None:
    """Say what --tune did. Deciding quietly would be the one unacceptable
    way for this to work."""
    if query < TUNE_MIN_QUERY:
        err.write(
            f"clustergrep: --tune saw the query itself only {query} time(s) in "
            f"its sample, too few to measure the others against; "
            f"cluster unchanged\n"
        )
        return
    if not dropped:
        err.write(
            f"clustergrep: --tune kept all {len(counts)} term(s) that fired; "
            f"no clear separation from the query's own rate\n"
        )
        return
    err.write(
        f"clustergrep: --tune dropped {len(dropped)} term(s), cutting at a "
        f"{gap:.0f}x jump in how often they fire versus the query:\n"
    )
    for term, (n, ratio) in sorted(dropped.items(), key=lambda kv: -kv[1][1]):
        err.write(f"  {term:<20} {n:>9}  {ratio:5.1f}x\n")
    err.write("  re-run without --tune to keep them\n")


# ---------------------------------------------------------------- searching


@dataclass
class LineHit:
    path: str
    lineno: int
    line: str
    matches: list[Match]

    @property
    def best(self) -> Match | None:
        return min(self.matches, key=lambda m: m.distance, default=None)


def search_stream(
    stream: Iterable[str],
    matcher: Matcher,
    path: str,
    *,
    invert: bool,
    limit: int | None,
    need_matches: bool = True,
    progress: Progress | None = None,
) -> Iterator[LineHit]:
    """Yield the lines of ``stream`` that match, or that do not under ``invert``.

    ``need_matches`` is the difference between asking "where are all the hits
    on this line, and which terms were they" and asking "is there one". The
    counting and file-listing modes only need the second question answered,
    and on a large file that is most of the work.
    """
    probe = matcher.regex.search
    found = 0
    for lineno, raw in enumerate(stream, 1):
        if progress is not None:
            progress.advance(len(raw))
        line = raw.rstrip("\n").rstrip("\r")
        if need_matches:
            matches = matcher.search(line)
            hit = bool(matches)
        else:
            matches = []
            hit = probe(line) is not None
        if hit == invert:
            continue
        yield LineHit(path=path, lineno=lineno, line=line, matches=matches)
        found += 1
        if limit is not None and found >= limit:
            return


def looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(_SNIFF)
    except OSError:
        return False


def walk(files: Sequence[str], *, recursive: bool, include: list[str],
         exclude: list[str], warn) -> Iterator[Path]:
    def wanted(p: Path) -> bool:
        if include and not any(fnmatch.fnmatch(p.name, g) for g in include):
            return False
        return not any(fnmatch.fnmatch(p.name, g) for g in exclude)

    for name in files:
        path = Path(name)
        if path.is_dir():
            if not recursive:
                warn(f"{name}: is a directory")
                continue
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d != ".git")
                for child in sorted(names):
                    candidate = Path(root) / child
                    if wanted(candidate):
                        yield candidate
        elif path.exists():
            yield path
        else:
            warn(f"{name}: no such file or directory")


# ---------------------------------------------------------------- rendering


class Printer:
    """Formats hits. Holds every decision about what a result line looks like."""

    def __init__(self, args, ink: Ink, out: TextIO) -> None:
        self.args = args
        self.ink = ink
        self.out = out
        self.show_path = args.filename

    def _prefix(self, hit: LineHit, match: Match | None) -> str:
        sep = self.ink.sep(":")
        parts = []
        if self.show_path:
            parts.append(self.ink.path(hit.path))
        if self.args.line_number:
            parts.append(self.ink.lineno(str(hit.lineno)))
        if self.args.show_distance and match is not None:
            parts.append(self.ink.distance(match.distance, f"{match.distance:.2f}"))
            parts.append(self.ink.distance(match.distance, match.term.text))
        return f"{sep.join(parts)}{sep}" if parts else ""

    def _highlight(self, hit: LineHit) -> str:
        if not self.ink.enabled or not hit.matches:
            return hit.line
        out, cursor = [], 0
        for m in sorted(hit.matches, key=lambda m: m.start):
            if m.start < cursor:  # overlapping match, already painted
                continue
            out.append(hit.line[cursor:m.start])
            out.append(self.ink.hit(hit.line[m.start:m.end]))
            cursor = m.end
        out.append(hit.line[cursor:])
        return "".join(out)

    def emit(self, hit: LineHit) -> None:
        if self.args.json:
            for m in hit.matches or [None]:
                record = {
                    "file": hit.path,
                    "line": hit.lineno,
                    "distance": None if m is None else round(m.distance, 4),
                    "term": None if m is None else m.term.text,
                    "matched": None if m is None else m.text,
                }
                # On a corpus where one line is pages of text, echoing the
                # line back is the difference between a usable stream and an
                # unreadable one. -o wants only the match; --excerpt wants it
                # with enough context to judge.
                if self.args.excerpt and m is not None:
                    windows = excerpts(
                        hit.line, [(m.start, m.end)], self.args.excerpt
                    )
                    record["excerpt"] = windows[0].text if windows else ""
                elif not self.args.only_matching:
                    record["text"] = hit.line
                self.out.write(json.dumps(record) + "\n")
            return
        if self.args.only_matching:
            for m in sorted(hit.matches, key=lambda m: m.start):
                self.out.write(f"{self._prefix(hit, m)}{self.ink.hit(m.text)}\n")
            return
        if self.args.excerpt:
            for window in excerpts(
                hit.line, [(m.start, m.end) for m in hit.matches], self.args.excerpt
            ):
                nearest = self._nearest_in(hit, window)
                self.out.write(
                    f"{self._prefix(hit, nearest)}{self._paint(window)}\n"
                )
            return
        self.out.write(f"{self._prefix(hit, hit.best)}{self._highlight(hit)}\n")

    def _nearest_in(self, hit: LineHit, window) -> Match | None:
        """The nearest match this window actually contains.

        Each excerpt is labelled with its own distance rather than the line's,
        so a window showing a 0.4 match is not reported as 0.15 because
        something nearer appeared elsewhere in the same page of text.
        """
        texts = {window.text[a:b].lower() for a, b in window.spans}
        inside = [m for m in hit.matches if m.text.lower() in texts]
        return min(inside or hit.matches, key=lambda m: m.distance, default=None)

    def _paint(self, window) -> str:
        if not self.ink.enabled:
            return window.text
        out, cursor = [], 0
        for start, end in sorted(window.spans):
            if start < cursor:
                continue
            out.append(window.text[cursor:start])
            out.append(self.ink.hit(window.text[start:end]))
            cursor = end
        out.append(window.text[cursor:])
        return "".join(out)


def render_explain(cluster: Cluster, ink: Ink, out: TextIO) -> None:
    out.write(
        f"{cluster.query!r} via {cluster.backend}, threshold {cluster.threshold:g}: "
        f"{len(cluster.terms)} term(s)\n"
    )
    width = max((len(t.text) for t in cluster.terms), default=0)
    for term in cluster.terms:
        line = f"  {ink.distance(term.distance, f'{term.distance:.2f}')}  {term.text:<{width}}"
        if term.via and term.via != "query":
            line += f"  {ink.dim(term.via)}"
        out.write(line.rstrip() + "\n")


def render_summary(matcher, fired, lines: int, out: TextIO, ink: Ink,
                   as_json: bool) -> None:
    """How many lines matched, and which cluster terms were responsible.

    Ordered by distance rather than by count, so the report reads as "how far
    did I have to reach, and what did that buy me" -- the question a threshold
    is chosen to answer. On a corpus whose lines are pages long, this is the
    only output that fits on a screen.
    """
    distances = matcher.cluster.distances()
    rows = sorted(fired.items(), key=lambda kv: (distances.get(kv[0], 1.0), kv[0]))

    if as_json:
        out.write(json.dumps({
            "query": matcher.cluster.query,
            "backend": matcher.cluster.backend,
            "threshold": matcher.cluster.threshold,
            "lines_matched": lines,
            "matches": sum(fired.values()),
            "cluster_terms": len(matcher.cluster.terms),
            "terms_fired": len(fired),
            "terms": [
                {"term": t, "distance": distances.get(t), "count": n} for t, n in rows
            ],
        }) + "\n")
        return

    out.write(
        f"{lines} line(s) matched, {sum(fired.values())} match(es), "
        f"{len(fired)} of {len(matcher.cluster.terms)} cluster term(s) fired\n"
    )
    width = max((len(t) for t, _ in rows), default=0)
    for term, n in rows:
        d = distances.get(term, float("nan"))
        out.write(f"  {ink.distance(d, f'{d:.2f}')}  {term:<{width}}  {n}\n")


def render_senses(backend, word: str, out: TextIO) -> int:
    senses = backend.describe_senses(word)
    if not senses:
        out.write(f"{word!r} is not in WordNet\n")
        return EXIT_NO_MATCH
    for index, name, gloss, lemmas in senses:
        out.write(f"{index:>3}  {name:<24} {', '.join(lemmas)}\n     {gloss}\n")
    return EXIT_MATCH


# ---------------------------------------------------------------- assembly


def build_backend(args) -> Backend:
    if args.backend == "wordnet":
        from .wordnet import SENSE_PENALTY, WordNetBackend

        return WordNetBackend(
            pos=args.pos,
            sense=args.sense,
            sense_penalty=(
                SENSE_PENALTY if args.sense_penalty is None else args.sense_penalty
            ),
            include_antonyms=args.antonyms,
        )
    if args.backend == "thesaurus":
        if not args.thesaurus:
            raise BackendError(
                "--backend thesaurus needs a file",
                remedy="pass --thesaurus PATH or set CLUSTERGREP_THESAURUS",
            )
        from .thesaurus import ThesaurusBackend

        return ThesaurusBackend(args.thesaurus)

    if not args.model:
        raise BackendError(
            "--backend vectors needs a model",
            remedy="pass --model PATH or set CLUSTERGREP_MODEL",
        )
    from .vectors import VectorBackend

    return VectorBackend(args.model)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Ends the way a Unix tool should when interrupted.

    `clustergrep ... | head` closes the pipe as soon as head has what it
    wants, and the next write raises. Reporting that as a traceback would be
    noise: the user got exactly what they asked for. Ctrl-C is the same --
    it is an instruction, not a failure.
    """
    try:
        return _run(argv)
    except BrokenPipeError:
        _erase_progress()
        _discard_stdout()
        return EXIT_BROKEN_PIPE
    except KeyboardInterrupt:
        _erase_progress()
        sys.stderr.write("\n")
        return EXIT_INTERRUPTED


def _erase_progress() -> None:
    """Clear a half-drawn progress line the search never got to erase itself."""
    try:
        if sys.stderr.isatty():
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
    except (OSError, ValueError):
        pass


def _discard_stdout() -> None:
    """Point stdout at nowhere so the interpreter's final flush cannot raise.

    Without this, Python reports a second BrokenPipeError while shutting
    down, after we have already handled the first.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        pass


def _run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parse_argv(parser, argv)
    out, err = sys.stdout, sys.stderr

    if args.paths:
        from .paths import describe

        out.write(describe())
        return EXIT_MATCH
    if args.install_data:
        from .wordnet import install_data

        ok, message = install_data()
        (out if ok else err).write(message + "\n")
        return EXIT_MATCH if ok else EXIT_ERROR
    if args.word is None:
        parser.error("a word to search for is required")
    if args.excerpt is not None and args.excerpt < 1:
        parser.error(f"--excerpt must be at least 1, got {args.excerpt}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold must be between 0.0 and 1.0, got {args.threshold}")

    ink = Ink(args.color == "always" or (args.color == "auto" and out.isatty()))

    def warn(message: str) -> None:
        err.write(f"clustergrep: {message}\n")

    try:
        backend = build_backend(args)
        if args.senses:
            if not hasattr(backend, "describe_senses"):
                parser.error(f"--senses is only meaningful for --backend wordnet")
            return render_senses(backend, args.word, out)

        cluster = build_cluster(args, backend, err)
    except BackendError as exc:
        warn(str(exc))
        if exc.remedy:
            err.write(f"  try: {exc.remedy}\n")
        return EXIT_ERROR
    except ValueError as exc:
        warn(str(exc))
        return EXIT_ERROR

    if args.explain:
        if args.tsv:
            from .thesaurus import to_tsv

            out.write(to_tsv(cluster))
        else:
            render_explain(cluster, ink, out)
        return EXIT_MATCH

    if len(cluster.terms) == 1 and args.threshold > 0:
        warn(
            f"{args.word!r} has no neighbours within {args.threshold:g} "
            f"in {backend.name}; searching for the word alone"
        )

    # Inflection is morphology, not semantics: "escaped" is the same word as
    # "escape", not a more distant one. So it stays on its own axis and does
    # not quietly switch itself off when --threshold is 0.
    from .wordnet import irregular_forms

    matcher = Matcher(
        cluster,
        inflect=args.inflect,
        ignore_case=args.ignore_case,
        # Every backend gets these, not just wordnet: "fled" is a spelling of
        # "flee" no matter which lexicon proposed "flee" in the first place.
        word_variants=irregular_forms,
    )

    if args.patterns:
        if args.tune:
            # Without this the prefilter keeps every line containing the
            # noise, which on a corpus where the noise is the common case
            # means it discards almost nothing and the fast path is not fast.
            matcher, _, _ = tune(matcher, input_paths(args, warn), err)
        out.write("".join(f"{pattern}\n" for pattern in matcher.patterns()))
        return EXIT_MATCH

    return run_search(args, matcher, ink, out, warn)


def _total_bytes(paths) -> int | None:
    """How much there is to read, when that is knowable.

    Unknowable for stdin, which is exactly the case the documented pipeline
    uses, so progress has to stay useful without it -- hence bytes and rate
    rather than only a percentage.
    """
    if any(p is None for p in paths):
        return None
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            return None
    return total or None


def _clock_for(args):
    from .progress import _clock

    return _clock


def input_paths(args, warn) -> "list[Path | None]":
    if not args.files:
        return [None]
    return list(
        walk(args.files, recursive=args.recursive, include=args.include,
             exclude=args.exclude, warn=warn)
    )


def run_search(args, matcher: Matcher, ink: Ink, out: TextIO, warn,
               replay: list[str] | None = None) -> int:
    paths = input_paths(args, warn)

    if args.filename is None:
        args.filename = len(paths) > 1 or args.recursive

    # Both --stats and --summary are claims about which terms fired, so they
    # force the slow path that collects every match rather than the first.
    counting = args.count or args.files_with_matches or args.files_without_match
    quiet = counting or args.summary
    need_matches = args.stats or args.summary or not (quiet or args.invert_match)

    showing = args.progress == "always" or (
        args.progress == "auto" and sys.stderr.isatty()
    )

    replay: list[str] = []
    if args.tune:
        matcher, _, replay = tune(
            matcher, paths, sys.stderr,
            Progress(None, showing, clock=_clock_for(args)),
        )

    progress = Progress(_total_bytes(paths), showing, clock=_clock_for(args))

    printer = Printer(args, ink, out)
    fired: Counter = Counter()
    lines = 0
    matched_any = False
    buffered: list[LineHit] = []

    for path in paths:
        label = "(standard input)" if path is None else str(path)
        if path is not None and looks_binary(path):
            continue
        try:
            handle = (
                sys.stdin if path is None
                else path.open(encoding="utf-8", errors="replace")
            )
        except OSError as exc:
            warn(f"{label}: {exc.strerror}")
            continue
        if path is None and replay:
            # --tune already consumed these from a stream that cannot seek.
            handle = chain(replay, handle)

        count = 0
        try:
            for hit in search_stream(handle, matcher, label,
                                     invert=args.invert_match,
                                     limit=args.max_count,
                                     need_matches=need_matches,
                                     progress=progress):
                count += 1
                lines += 1
                matched_any = True
                for m in hit.matches:
                    fired[m.term.text] += 1
                if quiet:
                    if args.files_with_matches:
                        out.write(f"{ink.path(label)}\n")
                        break
                    if args.files_without_match:
                        break
                    continue
                if args.sort:
                    buffered.append(hit)
                else:
                    printer.emit(hit)
        finally:
            if path is not None:
                handle.close()

        if args.count:
            prefix = f"{ink.path(label)}{ink.sep(':')}" if args.filename else ""
            out.write(f"{prefix}{count}\n")
        if args.files_without_match and count == 0:
            out.write(f"{ink.path(label)}\n")

    progress.done()

    if args.sort:
        buffered.sort(key=lambda h: (
            h.best.distance if h.best else 1.0, h.path, h.lineno))
        for hit in buffered:
            printer.emit(hit)

    if args.summary:
        # The report is the output here, so it goes to stdout where a pipe
        # can reach it -- unlike --stats, which annotates a normal run and
        # must stay on stderr to keep from polluting the results.
        render_summary(matcher, fired, lines, out, ink, as_json=args.json)
    elif args.stats:
        render_summary(matcher, fired, lines, sys.stderr, ink, as_json=False)

    return EXIT_MATCH if matched_any else EXIT_NO_MATCH


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
