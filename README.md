# clustergrep

grep for a concept rather than a string, and tell you how far each match is
from what you asked for.

`grep` is mostly used for fixed or moderately varying text that you can determine
in advance: a fixed log line string (e.g. "reboot"), a well-formed timestamp,
a string which is known to occur around another particular string ("login for niallm
failed"). It is a poor fit for the other common search, the one where you
know the *idea* and not the words — for example the concept "escape".
Today we use something like `grep -E 'escape|flee|jailbreak|breakout'`, which
is inelegant, incomplete, and stale after we invent a new word for something,
which we seem to be doing a lot these days. Handing the whole job to a language model
trades those problems for different, potentially worse ones: hallucination,
skipping lines or an entire file, or answering differently this Tuesday as
opposed to last one.

clustergrep splits the difference. A lexical model decides *what to
look for*, once, up front, where you can read it and argue with it. After that,
finding it is ordinary deterministic matching with nothing in the loop.

```console
$ clustergrep -t 0.25 escape incidents.log
1:0.00:escape:2024-01-02 The prisoner escaped through the laundry chute.
2:0.25:breakout:2024-01-03 Guards reported a breakout on B wing at 0300.
4:0.20:flee:2024-01-05 Two inmates fled across the yard before dawn.
5:0.25:jailbreak:2024-01-06 A jailbreak attempt was foiled by the perimeter fence.
6:0.25:fly the coop:2024-01-07 He flew the coop while the van was being loaded.
8:0.15:elude:2024-01-09 The suspect eluded officers for six hours.
9:0.15:escapism:2024-01-10 Staff described a general air of escapism among the population.
10:0.20:getaway:2024-01-11 Getaway vehicle recovered near the motorway.
12:0.15:break loose:2024-01-13 The detainee broke loose during transfer.
```

Of course, this matching surrenders the previous boolean does/doesn't match
approach, in favour of a notion of the _conceptual distance_ between the term
you searched for and what was found. This means that we report lines that matched,
and also the extent to which they matched. In the above, `0.25:jailbreak` says
*this line matched because "jailbreak" is 0.25 away from "escape"*. You can tweak
`-t`, the flag that sets the conceptual/lexical distance until you're satisfied
with the results.

Note "fled" and "broke loose": clustergrep matches irregular inflections, so
the cluster does not have to enumerate them.

At the moment clustergrep does not handle non-English, but this is very definitely
an ambition. (Technically, the thesaurus is multi-language, but the inflection
rules are not.)

## Install

One of:
```bash
uv tool install clustergrep
pip install clustergrep
pipx install clustergrep
```

`uv tool` gives it an isolated environment, so `nltk` and its dependencies
never collide with anything else, and puts `clustergrep` on your PATH. If the
shell cannot find it afterwards, `uv tool update-shell`. To try it without
installing anything, `uvx clustergrep -t 0.25 escape incidents.log`.

`pip install clustergrep` works too, and `pipx install clustergrep` is the
same idea as `uv tool` if that is what you already have.

For the vectors backend, ask for the extra at install time — these
environments are isolated, so numpy isn't easily added afterwards.

```bash
uv tool install 'clustergrep[vectors]'
```

The first search will offer to fetch the WordNet corpus, or you can get it
over with at the very start:

```bash
clustergrep --install-data
```

You will only be asked to install the data in an interactive terminal; when that's
not available (e.g. pipeline, cron job, CI), you get an error telling you
to run `--install-data`.

**The corpus is not bundled.** As above, WordNet is about 10MB and is licensed separately
by Princeton, so clustergrep downloads it on request into a per-user directory
rather than shipping a copy. If you already have it — from a previous `nltk`
install, a system package, or `NLTK_DATA` — that copy is found first and
nothing is downloaded.

|            | data                                   | cache                                |
|------------|----------------------------------------|--------------------------------------|
| Linux      | `~/.local/share/clustergrep`           | `~/.cache/clustergrep`               |
| macOS      | `~/Library/Application Support/clustergrep` | `~/Library/Caches/clustergrep`  |
| Windows    | `%LOCALAPPDATA%\clustergrep\Data`       | `%LOCALAPPDATA%\clustergrep\Cache`   |

`XDG_DATA_HOME` and `XDG_CACHE_HOME` are honoured wherever they are set, and
`CLUSTERGREP_DATA` / `CLUSTERGREP_CACHE` override everything — necessary for
containers, CI, and read-only home directories.

```console
$ clustergrep --paths
data     /home/you/.local/share/clustergrep
cache    /home/you/.cache/clustergrep
wordnet  /home/you/.local/share/clustergrep/corpora/wordnet.zip
```

The corpus is fetched once per machine, not once per install: it lives in
the data directory rather than the environment, so `uvx` runs reuse it and
switching from `uvx` to `uv tool install` will not download it again.

Uninstalling is `uv tool uninstall clustergrep` (or `pip uninstall`) and
deleting those two directories. Nothing else is written. The cache holds only
files clustergrep can rebuild, so deleting it alone is always safe.

### From source

```bash
git clone https://github.com/niallrmurphy/clustergrep
cd clustergrep
uv pip install -e '.[dev]'
uv run pytest -q
```

## Distance

Distance runs from 0.0 to 1.0.

| | |
|---|---|
| `0.0` | the word you typed, and nothing else |
| `~0.15` | another word for the same thing |
| `~0.25` | a narrower or related kind of the same thing |
| `~0.4` | recognisably connected, plausibly a different concept |
| `1.0` | unrelated |

`--threshold 0` admits only the literal word, so clustergrep becomes plain grep.
That is exact, not approximate:

```console
$ clustergrep -t 0 --no-inflect -s --no-distance Guards incidents.log
$ grep -n -w -F Guards incidents.log            # identical output
```

*Notable distinction from grep*: cluster matching is always word-oriented,
like `grep -w`, because concepts are words, and root/syllabic tokenisation would
result in undue implementation complexity right now.

### Where the numbers come from

The default backend walks WordNet as a weighted graph with costed edges,
and a penalty per "sense".  Every distance is therefore a summed path,
which `--explain` will show:

```console
$ clustergrep --explain -t 0.2 escape
'escape' via wordnet, threshold 0.2: 15 term(s)
  0.00  escape
  0.15  break loose  escape.v.01
  0.15  dodging      evasion.n.03
  0.15  elude        elude.v.02
  0.15  escapism     escape.n.02
  0.15  evasion      evasion.n.03
  0.15  flight       escape.n.01
  0.15  get away     escape.v.01
  0.15  get by       get_off.v.05
  0.15  get off      get_off.v.05
  0.15  get out      get_off.v.05
  0.15  miss         miss.v.09
  0.20  escapee      escape.v.01 -derivation-> escapee
  0.20  flee         escape.n.01 -derivation-> flee
  0.20  getaway      escape.v.01 -derivation-> getaway
```

| relation | cost | |
|---|---|---|
| synonym | 0.15 | another lemma of the same sense |
| similar / derivation / verb group | 0.20 | escape → escapee |
| hyponym | 0.25 | escape → jailbreak (narrower) |
| also-see | 0.35 | |
| hypernym | 0.40 | escape → movement (broader) |
| meronym | 0.45 | part, member, substance |
| antonym | 0.60 | `--antonyms`, off by default |

Narrowing costs less than broadening, because a narrower term keeps you inside
the concept while a broader one leaves it. Each successive dictionary sense of
the word adds 0.05, so the dominant reading dominates the cluster. There is little
that is truly objective about this scoring, but is pragmatically enough to work
with right now, and we are open to other suggestions.

## Backends

```
--backend wordnet      (default)  offline, explainable, English
--backend vectors      --model glove.6B.100d.txt
--backend thesaurus    --thesaurus terms.tsv
```

**wordnet** just knows lexicographic relations: relatively precise, but only
lexically bound, so "escape" and "jail" have no plausible path.

**vectors** is the opposite. Cosine distance over any GloVe or word2vec
text export "knows" that `escape → jail` immediately but isn't comparably
enumerable as a path from an accessible graph search.

**thesaurus** is a patch file you can use to add your own terms (but
_replaces_ WordNet rather than augments it, so you need to seed from the
original):

```
# concept   term          distance   note
escape      jailbreak     0.25
escape      exfil         0.30       our term
```

```bash
clustergrep --explain -t 0.3 escape --tsv > escape.tsv
clustergrep -b thesaurus --thesaurus escape.tsv escape incidents.log
```

From then on the search is fully reproducible and a new term is a one-line diff.

## Options

Familiar from grep: `-i` `-v` `-c` `-l` `-L` `-o` `-n` `-r` `-m` `-H`
`--include` `--exclude` `--color`. Exit codes match too — 0 matched, 1 nothing
matched, 2 error.

Particular to this tool:

| | |
|---|---|
| `-t, --threshold` | how far to reach (default 0.4) |
| `--explain`, `--tsv` | print the cluster and stop |
| `--senses` | list the word's WordNet senses, for `--sense` |
| `--pos n\|v\|a\|r` | one part of speech only |
| `--sense N` | pin one reading of the word |
| `--tune` | drop cluster terms that fire far more often than the query |
| `--summary` | report which terms fired and how often; print no lines |
| `--excerpt N` | print ~N characters around each match, not the whole line |
| `--stats` | the same report on stderr, alongside the normal results |
| `--patterns` | every pattern that would match, for use as a prefilter |
| `--sort` | nearest matches first |
| `--json` | one object per match |
| `--no-inflect` | exact patterns only |
| `--no-distance` | grep-shaped output |

`--stats` is for tuning: 

```console
$ clustergrep -t 0.4 escape incidents.log --stats -c
10 match(es) from 10 of 62 cluster term(s)
  0.00  escape                   1
  0.25  breakout                 1
  0.20  flee                     1
  0.25  jailbreak                1
  0.25  fly the coop             1
  0.30  leakage                  1
  0.15  elude                    1
  0.15  escapism                 1
  0.20  getaway                  1
  0.15  break loose              1
```

## Large files

Two things go wrong when the corpus is measured in gigabytes and a single
line is pages of text: printing matches is useless, and 6MB/s is too slow.

`--summary` answers the first. It prints no lines at all — only which cluster
terms fired and how often, ordered by distance, so you can read the shape of
a search that would otherwise scroll for hours:

```console
$ clustergrep -t 0.25 escape big.jsonl --summary
1370 line(s) matched, 1370 match(es), 5 of 40 cluster term(s) fired
  0.00  escape        289
  0.15  elude         279
  0.20  flee          273
  0.25  breakout      277
  0.25  fly the coop  252
```

Ordering by distance rather than by count is deliberate: it reads as *how far
did I reach, and what did that buy me* -- `-t` being how far, and the output
being what you gained. `--summary --json` gives those results as one JSON object.
When you want the matches themselves rather than a tally, `--excerpt` prints
a window of about 100 characters around each one instead of the whole line:

```console
$ clustergrep -t 0.25 escape big.jsonl --excerpt
1:0.00:escape:…during unusual was wing east the the detainee escaped through a service corridor unusual overnight…
41:0.20:flee:…the sensor although the the two inmates fled across the exercise yard nothing nothing the…
81:0.15:elude:…the and wing the fence duty the suspect eluded officers for several hours shift duty the…
```

Windows overlapping in the same line are merged, so a paragraph mentioning the
concept five times gives one readable excerpt rather than five near-identical
ones, and each excerpt is labelled with the distance of the match it actually
contains. `--excerpt N` sets the width; `-o` is the degenerate case of it,
printing the match and nothing else. Under `--json` the window arrives as an
`excerpt` field in place of `text`.

`--patterns` answers the second. It prints every string the matcher would
recognise — inflections included — which is exactly what a fast tool needs to
throw away the lines that cannot match:

```bash
clustergrep -t 0.25 escape --patterns > patterns.txt
```

```bash
rg -Fw -f patterns.txt big.jsonl | clustergrep -t 0.25 escape --summary
```

ripgrep discards the 98% of lines with nothing in them, and clustergrep does
the distance work on what survives. On a 300MB JSONL that is 54 seconds down
to 1.5, with byte-identical output.

**Use ripgrep, or GNU grep, and not the `grep` that came with macOS.** BSD
grep degrades catastrophically on a `-f` file of a few hundred patterns —
measured on the same 300MB JSONL, all three producing identical output:

| | 300MB | extrapolated to 6GB |
|---|---|---|
| `rg -Fw -f` then clustergrep | 1.5s | **~30s** |
| clustergrep alone, no prefilter | 54s | ~18 min |
| `/usr/bin/grep -Fw -f` then clustergrep | 432s | ~2.5 hours |

That is not a typo: on macOS the obvious prefilter is roughly eight times
*slower* than doing no prefiltering at all. `brew install ripgrep`, or
`brew install grep` for GNU grep as `ggrep`. On Linux the distribution `grep`
is GNU grep and is fine.

You cannot use `--explain` output as the pattern file: it holds `flee` where
the text holds `fled`, so lines are dropped silently. That is what
`--patterns` is for.

### Terms that drown the search

At scale the polysemy problem stops being theoretical. Searching a 6GB corpus
for `escape` returned 395,587 matches, of which `run`, `break` and `miss` were
91%. Different terms may be
conceptually synonymous in a context, but be textually irrelevant elsewhere;
`escape` can be used directly, or implied by `run` or `break`, but `run` can
of course occur in non-escape contexts. Lowering `-t` doesn't help in all
circumstances. However, we can systematically look at the distribution of the
usage, and remove the lower fidelity occurrences.

`--tune` effectively does this:

```bash
clustergrep -t 0.25 escape big.jsonl --summary --tune
```

```
clustergrep: --tune dropped 3 term(s), cutting at a 11x jump in how often they fire versus the query:
  run                        976    8.9x
  break                      515    4.7x
  miss                       340    3.1x
  re-run without --tune to keep them
319 line(s) matched, 319 match(es), 4 of 37 cluster term(s) fired
  0.00  escape     206
  0.15  flight     59
  0.20  flee       44
  0.25  jailbreak  10
```

It samples the corpus, counts how often each cluster term fires relative to
the query word itself, and cuts at the largest jump in that ratio. The
reasoning is that a term appearing ten times more often than the word you
actually searched for is not being used in your sense of it.

The cut is made at the jump rather than at a fixed threshold because corpora
vary, and any constant is unlikely to be relevant across both.

The flag does nothing rather than guess when the query is too rare in the sample
to measure against, or when there is no clear separation, and it never drops a
term firing less often than the query. **It always says what it did, on
stderr.** It does not silently discard terms.

Of course, this is a heuristic, and as such can fail or have edge cases: search for `automobile` in
a corpus that says `car`, and `car` looks exactly like `run` does. That is what
the stderr report is for — if it drops something you wanted, drop `--tune` and
pin a thesaurus instead.

If the interesting text is one field of a JSONL record, cut it out first and
scan less:

```bash
jq -r '.text' big.jsonl | rg -Fw -f patterns.txt | clustergrep -t 0.25 escape --summary
```

One caveat on all of these: `-F` matches literally, so a hyphenated
multi-word term (e.g. `fly-the-coop`) will of course be treated literally.
clustergrep itself accepts spaces, hyphens and underscores
interchangeably, so the second pass is more permissive than the first.

## Known limits

**Polysemy.** A cluster covers every sense of the word. At `-t 0.4`, "escape"
reaches `leakage` and `outflow` via the concept of fluid-discharge, so a line about
reactor coolant will match.

**`--tune` is statistical, not semantic.** It can only see how often a term
fires, so a genuine synonym that happens to be commoner than your query looks
identical to a polysemous intruder. It reports every term it drops for exactly
this reason.

**Distances are not probabilities** and are not comparable between backends.
They only order matches within one search.

**English only**, and only as current as WordNet 3.0.

**Irregular inflections** (`fled`, `broke`, `flew`) come from WordNet's
exception lists and apply to every backend, including a thesaurus you pinned
by hand — which spellings of a word exist is a fact about English, not about
where the cluster came from. They are lost only if the corpus is not
installed; `clustergrep --paths` will tell you. `--no-inflect` switches off
regular and irregular forms together.

**Speed.** Between about 4 and 6MB/s, against 0.1s for `grep -E` on the same
file. This comes from Python's regex engine working over an alternation of a
few hundred patterns, and is what produces the extra columns. Expect the lower
end on a first read of a large file and the upper end once it is in the page
cache; line length and match density barely matter, and non-ASCII text costs
a few percent. See [Large files](#large-files) for a way around it.

## Licence

clustergrep is MIT licensed. The WordNet corpus it downloads is **not** part of
this distribution and is covered by [Princeton's WordNet
licence](https://wordnet.princeton.edu/license-and-commercial-use). Any vector
model you point `--model` at carries whatever licence its publisher gave it.
