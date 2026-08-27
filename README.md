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
*this line matched because "jailbreak" is 0.25 away from "escape"*. You can tune
`-t`, the flag that sets the conceptual/lexical distance until you're satisfied
with the results.

Note "fled" and "broke loose": clustergrep matches irregular inflections, so
the cluster does not have to enumerate them.

At the moment clustergrep does not handle non-English, but this is very definitely
an ambition. (Technically, the thesaurus is multi-language, but the inflection
rules are not.)

## Install

```bash
uv tool install clustergrep
```

`uv tool` gives it an isolated environment, so `nltk` and its dependencies
never collide with anything else, and puts `clustergrep` on your PATH. If the
shell cannot find it afterwards, `uv tool update-shell`. To try it without
installing anything, `uvx clustergrep -t 0.25 escape incidents.log`.

`pip install clustergrep` works too, and `pipx install clustergrep` is the
same idea as `uv tool` if that is what you already have.

For the vectors backend, ask for the extra at install time — these
environments are isolated, so numpy cannot be added to one afterwards:

```bash
uv tool install 'clustergrep[vectors]'
```

The first search will offer to fetch the WordNet corpus, or you can get it
over with up front:

```bash
clustergrep --install-data
```

You are only asked when you are at a terminal to answer. In a pipeline, a cron
job or CI, a missing corpus is an error telling you to run `--install-data`,
so nothing ever reaches the network because a script happened to run
clustergrep. (A wheel has no post-install hook — installers unpack files and
never execute anything — so first run is the earliest honest moment to ask.)

**The corpus is not bundled.** WordNet is about 10MB and is licensed separately
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
| `--stats` | which cluster terms actually fired |
| `--sort` | nearest matches first |
| `--json` | one object per match |
| `--no-inflect` | exact surface forms only |
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

## Known limits

**Polysemy.** A cluster covers every sense of the word. At `-t 0.4`, "escape"
reaches `leakage` and `outflow` via the concept of fluid-discharge, so a line about
reactor coolant will match.

**Distances are not probabilities** and are not comparable between backends.
They only order matches within one search.

**English only**, and only as current as WordNet 3.0.

**Irregular inflections** (`fled`, `broke`) come from WordNet's exception lists,
so they are available under the default backend. The vectors and thesaurus
backends get regular suffix rules only: if you have a specific requirement,
put them in the TSV.

**Speed.** Roughly 8µs per line — about 5s for a 24MB, 300k-line file, against
0.02s for `grep -E`. This is Python's regex engine over a large alternation
and the extra columns. For now, if you need fast loops over lots of data, use 
`--explain` to generate the alternation and hand it to real grep.

## Licence

clustergrep is MIT licensed. The WordNet corpus it downloads is **not** part of
this distribution and is covered by [Princeton's WordNet
licence](https://wordnet.princeton.edu/license-and-commercial-use). Any vector
model you point `--model` at carries whatever licence its publisher gave it.
