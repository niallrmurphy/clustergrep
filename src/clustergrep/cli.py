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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

from . import __version__
from .cluster import Backend, BackendError, Cluster
from .matcher import Match, Matcher

EXIT_MATCH = 0
EXIT_NO_MATCH = 1
EXIT_ERROR = 2

DEFAULT_THRESHOLD = 0.4
DEFAULT_MAX_TERMS = 250

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
                     help="match only the exact surface forms in the cluster")

    g = p.add_argument_group("inspect the cluster instead of searching")
    g.add_argument("--explain", action="store_true",
                   help="print the cluster and exit, without searching")
    g.add_argument("--tsv", action="store_true",
                   help="with --explain, emit thesaurus TSV to pin and edit")
    g.add_argument("--senses", action="store_true",
                   help="list the word's WordNet senses and exit")

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
    g.add_argument("-o", "--only-matching", action="store_true",
                   help="print only the matched text")
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
                   help="after searching, report which terms actually fired")
    g.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help="colourise output (default auto)")

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
) -> Iterator[LineHit]:
    """Yield the lines of ``stream`` that match, or that do not under ``invert``.

    ``need_matches`` is the difference between asking "where are all the hits
    on this line, and which terms were they" and asking "is there one". The
    counting and file-listing modes only need the second question answered,
    and on a large file that is most of the work.
    """
    probe = matcher.pattern.search
    found = 0
    for lineno, raw in enumerate(stream, 1):
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
                self.out.write(json.dumps({
                    "file": hit.path,
                    "line": hit.lineno,
                    "distance": None if m is None else round(m.distance, 4),
                    "term": None if m is None else m.term.text,
                    "matched": None if m is None else m.text,
                    "text": hit.line,
                }) + "\n")
            return
        if self.args.only_matching:
            for m in sorted(hit.matches, key=lambda m: m.start):
                self.out.write(f"{self._prefix(hit, m)}{self.ink.hit(m.text)}\n")
            return
        self.out.write(f"{self._prefix(hit, hit.best)}{self._highlight(hit)}\n")


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

        cluster = Cluster.build(
            query=args.word,
            backend=backend.name,
            terms=backend.expand(args.word, args.threshold),
            threshold=args.threshold,
            max_terms=args.max_terms,
        )
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
    matcher = Matcher(
        cluster,
        inflect=args.inflect,
        ignore_case=args.ignore_case,
        word_variants=getattr(backend, "word_variants", None),
    )

    return run_search(args, matcher, ink, out, warn)


def run_search(args, matcher: Matcher, ink: Ink, out: TextIO, warn) -> int:
    from collections import Counter

    if args.files:
        paths: list[Path | None] = list(
            walk(args.files, recursive=args.recursive, include=args.include,
                 exclude=args.exclude, warn=warn)
        )
    else:
        paths = [None]

    if args.filename is None:
        args.filename = len(paths) > 1 or args.recursive

    # --stats is a claim about which terms fired, so it forces the slow path.
    summarising = args.count or args.files_with_matches or args.files_without_match
    need_matches = args.stats or not (summarising or args.invert_match)

    printer = Printer(args, ink, out)
    fired: Counter = Counter()
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

        count = 0
        try:
            for hit in search_stream(handle, matcher, label,
                                     invert=args.invert_match, limit=args.max_count):
                count += 1
                matched_any = True
                for m in hit.matches:
                    fired[m.term.text] += 1
                if summarising:
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

    if args.sort:
        buffered.sort(key=lambda h: (
            h.best.distance if h.best else 1.0, h.path, h.lineno))
        for hit in buffered:
            printer.emit(hit)

    if args.stats:
        sys.stderr.write(f"\n{sum(fired.values())} match(es) from "
                         f"{len(fired)} of {len(matcher.cluster.terms)} cluster term(s)\n")
        distances = matcher.cluster.distances()
        for term, n in fired.most_common():
            sys.stderr.write(f"  {distances.get(term, float('nan')):.2f}  "
                             f"{term:<24} {n}\n")

    return EXIT_MATCH if matched_any else EXIT_NO_MATCH


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
