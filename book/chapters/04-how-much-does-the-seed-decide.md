# How much does the seed decide?

## In short

Little. Thirty seeds made thirty worlds of the same kind, and what separates
two worlds at any moment is mostly time, not their seed: a world wanders,
remembers its size for a few hundred iterations, and its first half says
almost nothing about its second. That wander makes where a world stands at
the end of a run noisy. Over the last fifth of the runs, the number of agents
differs from seed to seed by a quarter of its mean — more than the fifth the
thesis allowed — and a change of a tenth in it would take a hundred seeds per
condition to see; inequality is steadier and needs 26. Measured over the
whole settled life of a run instead, from iteration 500, the spread of most
statistics nearly halves, and so do the seeds a change needs.

## Thesis

```thesis E03
```

The answer sets the size of every experiment after this one. If worlds with
different seeds end up much alike, a few dozen seeds can show the effect of a
setting. If they do not, an effect has to be very large to be seen at all —
or the seeds have to be looked at one by one.

## Method

```experiment E03
```

The thirty runs of Chapter 3. For each statistic, *where a run ended up* is
its average over the last fifth of its iterations, from 2,400 to 3,000. Across
the thirty seeds this gives a mean, a spread (the standard deviation), and
their ratio, the *coefficient of variation*: 0.1 means the seeds differ by
about a tenth of the typical value.

From that spread follows **how many seeds a condition needs** for a change of
5%, 10% or 20% in a statistic to be found four times in five, at the usual 5%
level of chance — the number every later chapter is sized by.

The statistics are the basic ones of Chapters 3 to 6: agents, connections,
inequality of wealth, leaves, genotypes and families, births, starvation and
culling, bridges, clustering and the core. Five seeds are also drawn one by
one, to show what the band in a figure hides.

## Results

### One kind of world

```figure E03/agents
```

The five thin lines are five single worlds; the bands are all thirty. Each
line crosses much of the band in the course of its life.

Where the 26 surviving worlds ended up spreads out evenly, with no gaps. Over
iterations 2,400 to 3,000 they held from 762 to 2,009 agents, from 1,043 to
3,083 connections, and their inequality of wealth (the Gini coefficient,
Chapter 5) lay between 0.37 and 0.62. Sorted, these values rise smoothly from
the lowest to the highest: no group of seeds settles at one level while
another settles at a different one.

```figure E03/gini
```

### Time, not the seed

For each statistic, each world's level was measured in 29 stretches of a
hundred iterations, from iteration 100 to 3,000. All the variation in those
levels splits into two parts: how much the worlds' own long-run averages
differ from each other, and how much each world moves around its own
average. The first part is small — between a tenth and a quarter of the
whole. The rest is each world wandering.

| | share of the variation between worlds | alike 100 iterations later | 200 | 300 | first half against second half |
|---|---|---|---|---|---|
| agents | 0.17 | 0.68 | 0.34 | 0.17 | +0.12 (p = 0.28) |
| connections | 0.21 | 0.64 | 0.28 | 0.12 | +0.41 (p = 0.02) |
| connections per agent | 0.10 | 0.54 | 0.18 | 0.04 | −0.04 (p = 0.58) |
| inequality (Gini) | 0.12 | 0.68 | 0.37 | 0.20 | −0.28 (p = 0.92) |
| share of the richest tenth | 0.14 | 0.67 | 0.37 | 0.21 | −0.19 (p = 0.83) |
| leaves | 0.19 | 0.71 | 0.39 | 0.20 | +0.02 (p = 0.45) |
| genotypes | 0.17 | 0.67 | 0.33 | 0.16 | +0.10 (p = 0.30) |
| families | 0.19 | 0.64 | 0.31 | 0.15 | +0.11 (p = 0.30) |
| births | 0.12 | 0.84 | 0.64 | 0.41 | −0.32 (p = 0.96) |
| starved | 0.13 | 0.83 | 0.62 | 0.43 | −0.14 (p = 0.72) |
| cut off | 0.12 | 0.84 | 0.63 | 0.38 | −0.34 (p = 0.97) |
| bridges | 0.17 | 0.70 | 0.40 | 0.24 | −0.17 (p = 0.78) |
| clustering | 0.23 | 0.65 | 0.34 | 0.12 | +0.40 (p = 0.02) |
| share in the core | 0.24 | 0.72 | 0.47 | 0.31 | +0.12 (p = 0.28) |

**A world forgets.** The middle columns are correlations between a world's
level in one stretch and its level 100, 200 and 300 iterations later. A
hundred iterations on, a world is still much where it was (0.68 for the
number of agents); two hundred on, only half as much; after three hundred,
hardly at all, and after five hundred not at all. Births and deaths are
remembered a little longer. Over a run, a world's most crowded hundred
iterations typically hold 4.6 times as many agents as its emptiest.

**The seed leaves little trace.** The last column asks whether a world that
is big in the first half of its life (iterations 100 to 1,499) is big in the
second (1,500 to 2,999) as well. For the number of agents the correlation
across the 26 worlds is 0.12 — chance alone gives that much more than one
time in four. Connections and clustering are the exception: a world with
many connections in its first half tends to have many in its second (0.41
and 0.40, each p = 0.02). That may be a trace of the starting ring that lasts,
or chance among fourteen tries; next to the wander it is small. Even the
share of variation "between worlds" in the first column is partly wander
that 2,900 iterations did not average away.

### How many seeds

| | mean at the end | spread at the end (sd / mean) | seeds for a 10% change | spread from iteration 500 | seeds for a 10% change |
|---|---|---|---|---|---|
| agents | 1,355 | 0.25 | 100 | 0.17 | 46 |
| connections | 2,206 | 0.19 | 58 | 0.19 | 56 |
| connections per agent | 3.35 | 0.17 | 45 | 0.08 | 10 |
| inequality (Gini) | 0.488 | 0.13 | 26 | 0.07 | 8 |
| share of the richest tenth | 0.434 | 0.15 | 36 | 0.09 | 12 |
| leaves | 520 | 0.52 | 420 | 0.24 | 94 |
| genotypes | 862 | 0.27 | 111 | 0.17 | 45 |
| families | 336 | 0.36 | 205 | 0.20 | 61 |
| births | 43.6 | 0.41 | 258 | 0.19 | 59 |
| starved | 8.61 | 0.68 | 718 | 0.35 | 188 |
| cut off | 26.8 | 0.41 | 264 | 0.22 | 79 |
| bridges | 658 | 0.46 | 327 | 0.22 | 74 |
| clustering | 0.076 | 0.46 | 339 | 0.34 | 184 |
| share in the core | 0.540 | 0.20 | 65 | 0.13 | 26 |

**At the end.** The thesis asked for a spread below a fifth of the mean and
at most thirty seeds for a change of a tenth, in the number of agents, the
connections and the inequality of wealth. Inequality passes on both counts
(0.13, 26 seeds). Connections pass the first and fail the second (0.19, 58
seeds). The number of agents fails both (0.25, 100 seeds). With thirty seeds
per condition, the smallest change that can be seen four times in five is
18% in the number of agents, 14% in connections and 9% in inequality.

**Over a longer stretch.** A world forgets its level within a few hundred
iterations, so its average over a longer stretch averages more of its wander
away. Over iterations 500 to 3,000, the spread across the seeds falls for
almost every statistic — for the number of agents from 0.25 to 0.17, for
inequality from 0.13 to 0.07 — and the seeds a change needs fall with the
square of it: from 100 to 46 for the number of agents, from 26 to 8 for
inequality. Connections are the exception, at 0.19 either way: they
are the statistic in which worlds seem to keep something of their own, and a
longer look cannot average that away. This stretch was chosen
after the runs had been seen, so it is a proposal for the experiments to
come, not a test of this one.

### Dying out

Four of the thirty worlds died: 13%, with a 95% interval of about 5% to 30%.
At a rate this low, extinction can only tell conditions apart when they
differ a lot: to see it rise from 13% to 30%, four times in five, would take
about 90 seeds per condition.

## Conclusion

The thesis is refuted, narrowly, by the number of agents and of
connections: their spread at the end of a run needs 100 and 58 seeds to show
a change of a tenth, not thirty. It holds for inequality, and it holds in
what it says about kind: there are no separate groups of worlds. Every seed
makes the same kind of world.

What the thesis missed is where the spread comes from. It is not the seed
but time: each world wanders, slowly, and the end of a run catches it
wherever it happens to be. That changes how the next experiments are
measured:

- **Over the settled life of a run, not its last fifth.** From iteration 500
  on, the spread of most statistics nearly halves, and the seeds they need
  fall to less than half. The next plans name this stretch before their
  runs.
- **Thirty seeds see large effects.** With thirty seeds over that stretch, a
  change of about 12% in the number of agents, 14% in connections or 5% in
  inequality can be seen four times in five. Smaller effects need more seeds.
- **Pairing by seed buys little.** Runs of two conditions with the same seed
  share their starting ring and little else; since the seed hardly decides
  where a world goes, a comparison seed by seed is about as noisy as one
  between unrelated runs.
- **Extinction is counted, not tested,** unless a change is drastic.

## Details

| | |
|---|---|
| Runs | the 30 of Chapter 3: `B1-10000-s001` … `B1-10000-s030`; 26 reached iteration 3,000 |
| Where a run ended up | its mean over iterations 2,400–3,000 — over the 26 worlds that reached iteration 3,000; the graph statistics (bridges, clustering, core) are measured every 25 iterations |
| From iteration 500 | each surviving world's mean over iterations 500–3,000, chosen after the runs were seen |
| Seeds needed | for two groups compared at the 5% level (two-sided) with 80% power, by the normal approximation: 2 × (1.96 + 0.84)² × (spread ÷ change)² |
| Between worlds and over time | each surviving world's mean in 29 stretches of 100 iterations from 100 to 3,000; the variance of the worlds' averages, as a share of it plus the mean variance within a world; correlations between a world's stretches 100, 200 and 300 iterations apart, pooled over the worlds |
| First half against second | the correlation across worlds of their means over iterations 100–1,499 and 1,500–2,999; p is the share of 10,000 random pairings of first and second halves that correlate as well (one-sided) |
| Dying out | Wilson's interval for 4 of 30; seeds for a rise to 30% by the normal approximation for two proportions |
| Results | `book/results/E03.json`: every number above, and each world's value by seed |
| Made with | `python3 gol_lab.py analyse E03` |
