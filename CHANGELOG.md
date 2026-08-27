# Changelog

## 0.11.0

- A progress report on stderr for slow searches: percentage, throughput and
  an estimate when the input can be sized, throughput alone when it is a
  pipe. Time-triggered rather than size-triggered, so a quick search stays
  silent, and shown only at a terminal so pipelines and CI never see it.
  `--progress always|never` overrides.
- Fixed: the boolean fast path for `-c`, `-l` and `-v` was never actually
  reached, so those modes had been collecting every match on a line when they
  only needed to know whether one existed.

- `--patterns --tune` tunes the pattern list against the corpus before
  emitting it, so the prefilter drops the noise too. Without it the filter
  keeps every line containing `run`, `break` or `miss` -- which on a corpus
  where those are the common case means it discards almost nothing, and the
  documented fast path is not fast.

- `--excerpt N` prints about N characters around each match instead of the
  whole line, for corpora where one record is pages of text and the usual
  grep-shaped output is unreadable. Overlapping windows in a line are merged,
  each window is labelled with the distance of the match it contains rather
  than the line's nearest, and a match longer than the window is still shown
  whole. Under `--json` it replaces `text` with an `excerpt` field.

## 0.10.0

- `--tune` drops cluster terms that fire far more often than the query word
  itself, which at scale is what polysemous intruders look like: on a 6GB
  corpus, `run`, `break` and `miss` were 91% of all matches for `escape`.
  The cut is made at the largest jump in the ratio rather than at a fixed
  threshold, since no constant separated noise from signal on two different
  corpora while the jump did so by margins of 11.8x and 38x. It declines to
  act when the query is too rare to measure against or when no clear boundary
  exists, never drops a term firing less often than the query, and always
  reports what it dropped on stderr.

## 0.9.1

- Irregular inflections now apply to every backend, not just wordnet. A
  pinned thesaurus holding `flee` matches `fled` again; previously it did
  not, and nothing said so -- the pattern file generated for a prefilter
  dropped them too, so the loss compounded. `--no-inflect` now switches off
  regular and irregular forms together, rather than only the regular ones.

- Corrected the large-file recipe, which recommended `grep -Fw -f`. That
  benchmark was taken in a shell where `grep` was silently aliased to ugrep.
  BSD grep, the default on macOS, takes 432s on the 300MB sample where
  ripgrep takes 1.5s -- making the documented prefilter about eight times
  slower than no prefilter at all. The recipe now uses `rg`, and says so.

## 0.9.0

- `--summary` reports which cluster terms fired and how often, ordered by
  distance, and prints no lines. For a corpus whose lines are pages of text,
  this is the only output that fits on a screen. `--summary --json` gives the
  same as one object.
- `--patterns` prints every pattern the matcher would recognise, inflections
  included, for use as a prefilter: `grep -Fw -f patterns.txt big.jsonl |
  clustergrep ... --summary` is roughly 16x faster than scanning directly and
  produces identical output. A filter built from `--explain` instead would
  silently drop lines containing `fled` while searching for `flee`.
- `--json -o` no longer echoes the whole line back, only what matched.

## 0.8.1

- The first search now offers to download the WordNet corpus, so installing
  and using clustergrep is one command rather than two. You are asked only
  when you are at a terminal to answer: in a pipeline, cron job or CI a
  missing corpus remains an error pointing at `--install-data`, and nothing
  reaches the network unprompted.
- README recommends `uv tool install`, which gives the CLI an isolated
  environment and puts it on PATH.

## 0.8.0

First public release.

- `clustergrep WORD FILE` matches a word and the words that mean roughly the
  same thing, reporting the distance and the term that fired on every line.
- `--threshold 0 --no-inflect -s` is exactly `grep -w -F`, verified against
  real grep in the test suite.
- Three backends behind one distance contract: `wordnet` (default; weighted
  shortest path over the lexical graph, every distance an explainable path),
  `vectors` (cosine distance over GloVe/word2vec), `thesaurus` (a TSV you pin,
  edit and commit).
- `--explain` prints the cluster before you trust it; `--tsv` writes it out as
  a thesaurus file; `--stats` reports which terms actually fired.
- Options may appear anywhere among the file names, as they may in grep.
- The WordNet corpus is downloaded on request with `--install-data`, never
  bundled. It lands in a per-user data directory — `%LOCALAPPDATA%` on Windows,
  `~/Library/Application Support` on macOS, `~/.local/share` elsewhere — and an
  existing copy is reused. `--paths` shows where everything is.
