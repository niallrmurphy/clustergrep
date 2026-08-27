# Changelog

## 0.9.0

- `--summary` reports which cluster terms fired and how often, ordered by
  distance, and prints no lines. For a corpus whose lines are pages of text,
  this is the only output that fits on a screen. `--summary --json` gives the
  same as one object.
- `--forms` prints every surface form the matcher would recognise, inflections
  included, for use as a prefilter: `grep -Fw -f forms.txt big.jsonl |
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
