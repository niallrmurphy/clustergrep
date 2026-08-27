# clustergrep

grep for a concept rather than a string, and tell you how far each match sat
from what you asked for.

`grep` is built for text you can spell in advance: a request id, a timestamp,
a stack frame. It is a poor fit for the other common search, the one where you
know the *idea* and not the words — every line about an escape, a failure, a
complaint. Today that becomes `grep -E 'escape|flee|jailbreak|breakout'`, which
is tedious, silently incomplete, and stale the moment someone coins a new term.

Handing the whole job to a language model trades those problems for worse ones:
a search that may hallucinate a line, quietly skip a file, or answer differently
on Tuesday. clustergrep splits the difference. A lexical model decides *what to
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

The two extra fields are the point. `0.25:jailbreak` says *this line matched
because "jailbreak" is 0.25 away from "escape"* — so you can judge the hit
without rereading the line, and you can tighten `-t` until the noise stops.

Note "fled" and "broke loose": clustergrep matches irregular inflections, so
the cluster does not have to enumerate them.

## Install

```bash
pip install clustergrep
clustergrep --install-data      # once: fetches the WordNet corpus
```

**The corpus is not bundled.** WordNet is about 10MB and carries Princeton's
own licence, so clustergrep downloads it on request into a per-user directory
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

Uninstalling is `pip uninstall clustergrep` and deleting those two
directories. Nothing else is written. The cache holds only files clustergrep
can rebuild, so deleting it alone is always safe.

### From source

```bash
git clone https://github.com/niallrmurphy/clustergrep
cd clustergrep
uv pip install -e '.[dev]'
uv run pytest -q
```

## Distance

Distance runs from 0.0 to 1.0 and is the whole interface:

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

Matching is always word-oriented, like `grep -w`, because concepts are words.

### Where the numbers come from

The default backend walks WordNet as a weighted graph and charges for each
relation it crosses. Every distance is therefore the cost of a real path, and
`--explain` will show you it:

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
the word adds 0.05, so the dominant reading dominates the cluster.

These costs are judgements, not measurements. They are worth exactly as much as
`--explain` says they are.

## Backends

```
--backend wordnet      (default)  offline, explainable, English
--backend vectors      --model glove.6B.100d.txt
--backend thesaurus    --thesaurus terms.tsv
```

**wordnet** knows relations a lexicographer wrote down. That makes it precise
and auditable, and blind to plain association: "escape" and "jail" have no
WordNet path worth the name.

**vectors** is the opposite trade. Cosine distance over any GloVe or word2vec
text export finds `escape → jail` immediately, and cannot tell you why. Parsed
models are cached, so only the first run pays for loading. Distances are not
comparable with WordNet's — read `--explain` before trusting a threshold.

**thesaurus** is a TSV file you own:

```
# concept   term          distance   note
escape      jailbreak     0.25
escape      exfil         0.30       our term for it
```

This is the answer to staleness and to jargon. Generate a starting point, edit
it, commit it next to the code it searches:

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

`--stats` is the tuning tool. It shows which terms earned their place:

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
reaches `leakage` and `outflow` via the fluid-discharge sense, so a line about
reactor coolant matches. The sense penalty pushes such terms further out — which
is why the distance is on every line — and `--pos v`, `--sense N` or a pinned
TSV removes them entirely.

**Common words.** `run`, `break`, `miss` and `flight` are all legitimately
within 0.25 of "escape", and in real prose they fire constantly. `--stats` finds
the culprit; a pinned thesaurus is the fix.

**Distances are not probabilities** and are not comparable between backends.
They order matches within one search. That is all they claim to do.

**English only**, and only as current as WordNet 3.0 — which is the case the
thesaurus backend exists to cover.

**Irregular inflections** (`fled`, `broke`) come from WordNet's exception lists,
so they are available under the default backend. The vectors and thesaurus
backends get regular suffix rules only; spell irregulars out in the TSV.

**Speed.** Roughly 8µs per line — about 5s for a 24MB, 300k-line file, against
0.02s for `grep -E`. The cost is Python's regex engine over a large alternation,
and it is the price of the extra columns. For a hot loop over gigabytes, use
`--explain` to generate the alternation and hand it to real grep.

## Licence

clustergrep is MIT licensed. The WordNet corpus it downloads is **not** part of
this distribution and is covered by [Princeton's WordNet
licence](https://wordnet.princeton.edu/license-and-commercial-use). Any vector
model you point `--model` at carries whatever licence its publisher gave it.
