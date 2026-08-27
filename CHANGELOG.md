# Changelog

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
