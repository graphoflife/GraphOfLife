# Do the brains matter at all?

## In short

Yes — and so does their evolving. Worlds whose agents decide by chance cannot
live at all: all thirty fell apart within three iterations. Worlds whose
brains never change after the founders do live — not one of thirty died —
but each is stuck with the brain of the founder whose descendants fill it,
and they end up nothing alike. Half of them froze: not one birth after
iteration 500, the same agents game after game. A third teem, with about one
child for every agent in every iteration. The worlds that keep evolving all
end up in between, at three births per hundred agents. Without evolution the
number of agents varies from world to world 2.6 times as much, births seven
times as much, and nearly all the variation lies between worlds rather than
within them: the seed, through the founder it picks, decides everything. The
thesis holds. Evolution is what makes the worlds alike — and what makes them
wander.

## Thesis

```thesis E07
```

Everything Chapters 3 to 8 found is a fact about the baseline, with nothing
to hold it against (Meta I). This chapter takes the brains away in two ways.
If the worlds without them look like the baseline, the rules alone make the
world, and the brains are decoration. If they do not, the difference is what
the brains — and their evolution — do.

## Method

```experiment E07
```

Three conditions of thirty seeds each, at 10,000 tokens and for 3,000
iterations, as in Chapter 3. Each changes one setting of the baseline B1.

- **baseline** — the thirty runs of Chapter 3, reused.
- **brains never change** — `mutation_sparsity` 0. A brain still "changes"
  with the same chance of one in five, at birth and after every game, and
  gets a new name when it does — but not one of its weights moves. So the only
  brains a world ever has are its founders', copied; the names still split
  exactly as often as in the baseline, which keeps the family tree of
  Chapter 8 comparable. Once one founder's descendants fill a world, every
  agent in it carries the same brain, and which name spreads from then on is
  pure chance.
- **decisions by chance** — `random_decisions`. Agents never read their
  inputs: every number a brain would have given is drawn at random instead,
  and every decision follows from that.

Two things were seen before this plan was written, in pilots run outside the
lab to size it. Every one of ten worlds deciding by chance died within three
iterations: random stakes leave connections unused, unused connections are
cut, and within two games the world falls apart into pieces too small to
live. The thirty runs of that condition record this properly; it is not the
thesis. And two worlds whose brains never change came out as different as
the thesis says — which is why it says it.

What is compared, from iteration 500 on (the settled life of a run, Meta I):

- **how alike the worlds are** — for the number of agents and the births,
  the spread across the surviving worlds of each condition (the standard
  deviation over the mean), and the share of all variation in a world's
  number of agents that lies between the worlds rather than within each one
  over time, as in Chapter 4;
- **everything Chapters 3 to 8 measured** — agents, connections, births and
  deaths, inequality, the game, the shape of the graph, genotypes and
  families — each condition against the baseline;
- **lineages**, as in Chapter 8. In a world whose brains never change, once
  one founder's descendants fill it, no lineage is better than another, so
  how often the common ancestor of all the living moves forward there is the
  rate of chance — the null that Chapter 8's sweeps were missing. It is
  reported without a thesis.

The lab's estimate for this experiment is high: it cannot know that the
worlds deciding by chance die in seconds, and counts them as full runs. The
work that is really left is the thirty runs whose brains never change, about
seven hours on four workers.

## Results

### Chance cannot keep a world

All thirty worlds deciding by chance died: 25 after their second iteration,
5 after their third. Random stakes leave connections without tokens; a
connection that carries no tokens in a game is cut at its end, and anything
no longer joined to the largest piece of the world is removed. In the last
game of the median world, 62% of the agents were cut off that way and
another 15% starved. The baseline's founders start with brains nobody chose either — random
weights — and their worlds live. The difference is presumably that even a
random brain is a function of what it sees, so what it does on one
connection hangs together with what it does on the next, game after game;
noise has nothing of the kind.

### Without change: frozen or teeming

```figure E07/births
```

The thin lines are three single worlds whose brains never change: one frozen
(seed 2, at nothing), one breeding (seed 1) and one teeming (seed 25). The
evolving worlds of the baseline are the thin band along the bottom, at about
forty births per iteration.

From iteration 500 on, the thirty worlds whose brains never change fall into
kinds the baseline never shows:

- **frozen** — 15 worlds without a single birth. Their last child was born
  between iterations 29 and 439. Fourteen of them then kept exactly the same
  number of agents to the end of the run — between 183 and 2,252, depending
  on the world — playing the game every iteration with no one born and no
  one dying. The fifteenth, seed 20, lost agents without ever replacing
  them, from 1,165 at iteration 300 to 23 by iteration 2,024, and then sat
  at 23, three above the line at which a run counts as extinct, for the
  rest of the run.
- **nearly frozen** — 2 worlds, with a birth in about one iteration in ten.
- **breeding** — 3 worlds, with 8 to 24 births per hundred agents per
  iteration.
- **teeming** — 10 worlds, with 75 to 100 births per hundred agents per
  iteration: about one child for every agent, every iteration, and about as
  many agents lost again in the following game. In seed 12 every child is
  lost the moment it is born, joined to no one: its 1,297 agents try to have
  a child in every iteration, and not one ever lives.

The evolving worlds of the baseline, against that, all breed alike: three
births per hundred agents per iteration, the middle half of the 26 between
2.8 and 3.2. And not one of the thirty worlds without change died out,
against four of the baseline's thirty.

```figure E07/agents
```

Here the thin lines are the frozen world, flat at 1,420 agents from iteration
95 on, and the teeming one.

### How alike the worlds are

| From iteration 500 on | baseline | brains never change | the thesis asked |
|---|---|---|---|
| spread of the number of agents across the worlds (sd ÷ mean) | 0.17 | 0.45 | at least 0.34 |
| spread of births across the worlds | 0.19 | 1.39 | at least twice the baseline's |
| share of the variation in agents that lies between the worlds | 0.17 | 0.96 | more than half |

All three hold, by a wide margin. And a world without change hardly wanders.
Its most crowded hundred iterations hold 1.11 times as many agents as its
emptiest, against 4.65 for an evolving world (paired by seed, a difference of
3.5, 95% interval 2.8 to 4.2). Its first half says almost everything about
its second: the correlation across the thirty worlds is 0.96, against 0.12
for the baseline. The wander of Chapter 4 needs evolution.

### The same on average, in no single world

| From iteration 500 on, median [lowest–highest world] | baseline | brains never change |
|---|---|---|
| inequality of wealth (Gini) | 0.49 [0.42–0.55] | 0.32 [0.15–0.79] |
| share of agents that keep their node | 0.45 [0.40–0.52] | 0.45 [0.15–0.77] |
| share of stakes put on one's own node | 0.30 [0.28–0.35] | 0.30 [0.04–0.51] |
| share of nodes taken by a coalition | 0.52 [0.41–0.56] | 0.44 [0.00–0.83] |
| share of agents in the core | 0.54 [0.31–0.59] | 0.53 [0.10–0.95] |
| share of connections that are bridges | 0.31 [0.26–0.59] | 0.37 [0.04–0.85] |
| clustering | 0.075 [0.033–0.129] | 0.020 [0.000–0.225] |
| genotypes per agent | 0.63 [0.61–0.67] | 0.61 [0.49–0.72] |

On average a world without change looks much like an evolving one: in both,
45% of agents keep their node in a game, and 30% of stakes go on the
staker's own node. But the evolving worlds sit close together, and the
worlds without change spread over almost everything each statistic can be.
One difference holds on average too: wealth is more evenly spread without
change, a Gini coefficient of 0.34 against 0.50 (paired by seed, a
difference of 0.17, 95% interval 0.11 to 0.23), and most evenly in the
teeming worlds (0.22).

```figure E07/gini
```

### Lineages by chance

```figure E07/ancestor
```

In a world whose brains never change, the founders' lines still compete
until one of them fills the world: between iterations 75 and 100 in the
median world (125 in the baseline). From then on every agent in it carries the same brain, and
which genotype spreads is pure chance — the null Chapter 8 was missing. How
often the ancestor all the living share moved forward, after iteration 100:

| | baseline | frozen | teeming |
|---|---|---|---|
| moves of the common ancestor, median (range) | 8 (3–21) | 0 (0–4) | 94 (0–108) |
| how far back all the living share one ancestor, from iteration 500 | 730 iterations | back to the founders | 38 iterations |

In the frozen worlds the family tree hardly moves once one founder's line
has filled the world: nodes still change hands, but no later branch takes the
whole world again. In four of the worlds without change (seeds 7, 14, 18 and
19) the living still descended from more than one founder at the end. In the teeming worlds
chance sweeps the whole world every thirty iterations or so — more than ten
times as often as the baseline's lineages. So chance alone can make one line take a
world over, much faster than in the baseline or never, depending on how
often agents are born and die. Chapter 8's sweeps are not, by themselves,
evidence of selection: a fair null needs a world that is born and dies as
the baseline does while no brain is better than another, and none of these
is one. The three breeding worlds come closest, with 0, 16 and 68 moves.

## Conclusion

The thesis holds on all three counts it set. Evolution is what makes the
worlds alike. Without it, a world is stuck with the policy of whichever founder's line
fills it — usually within a hundred iterations — and the policies a
hundred random founders bring are anything but alike: half of them stop
having children altogether, and a third have a child every iteration. The
seed decides which, and with it nearly everything about the world.

With evolution, the 26 surviving worlds of the baseline, each from a
different set of founders, all end up in the same narrow band: three births per
hundred agents per iteration, inequality near 0.5, the same share of nodes
kept and taken. Arriving at the same place from different starting points
is what selection towards a shared optimum would look like. It is not yet
proof of it: mutation's own churn could also keep brains from settling
anywhere extreme. The next experiment on brains can tell the two apart.
Chapter 11 changes how often brains change: if three births per hundred is
what selection finds, it should hold at other rates of mutation; if it is
mutation's churn, it should move with them.

Three more things this chapter settles or opens:

- **The brains matter.** No world survives on decisions taken at random, and
  even unselected random brains keep a world alive — because they are
  consistent.
- **The wander needs evolution.** Meta I suspected the wander of Chapter 4
  came from lineages sweeping the world. A world without change hardly
  wanders at all, which fits; whether the sweeps themselves drive it, rather
  than mutation in general, is still open.
- **Sweeps happen by chance too.** How often depends on births and deaths.
  The baseline's sweeps need a null with the baseline's own births and
  deaths before they can be read as selection.

## Details

| | |
|---|---|
| Runs | 90: the baseline's 30 of Chapter 3, reused (`B1-10000-s001` … `s030`); brains never change, `B1-10000-a8525d-s001` … `s030`; decisions by chance, `B1-10000-e7f450-s001` … `s030` |
| Survived to iteration 3,000 | baseline 26, brains never change 30, decisions by chance 0 (25 died after 2 iterations, 5 after 3) |
| Strains | `gol-1+brain_kind=float16` (baseline, brains never change: `mutation_sparsity` is a parameter); `gol-1+brain_kind=float16+random_decisions` |
| Engine | snapshot `f820369a1583af53`, commit `4bc7165`, for all 90 runs |
| Environment | Python 3.12.3, numpy 2.5.1, networkx 3.6.1, OpenBLAS (scipy-openblas 0.3.33), Intel Core Ultra 7 258V, numpy dispatch AVX2, one BLAS thread |
| Cost | the new runs: 35 hours of computing, done in 9.1 hours on four workers (2026-10-05 20:31 to 2026-10-06 05:39); 11 GB on disk; at most 746 MB of memory for one run |
| Where a run ended up | its mean from iteration 500 to 3,000, over the worlds that reached iteration 3,000 |
| Kinds of worlds without change | from each world's births from iteration 500 on, in the results by seed: none (frozen); fewer than 0.5 per hundred agents per iteration (nearly frozen); more than 50 (teeming); between (breeding). The iteration of each frozen world's last birth, seed 20's decline, the children lost at birth and how the chance-driven worlds fell apart were read from the runs' statistics for this chapter, outside the analysis |
| Comparisons | paired by seed; intervals by resampling the differences, p by flipping their signs at random, 10,000 times each |
| Results | `book/results/E07.json`: every number above, each world's value by seed, and every world's lineage summary |
| Made with | `python3 gol_lab.py analyse E07` (about 14 minutes: the lineages read every frame of every run) |
