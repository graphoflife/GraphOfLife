# How much does the seed decide?

> [!question] Questions of this page
> - How different do thirty worlds of the same [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") end up?
> - Is the difference something each [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") fixes for good, or does it come and
>   go?
> - How many seeds does an experiment need to see that a change of setting
>   changed something?

## In short

Thirty [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds that differ only in their seed end up alike: at
100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), their sizes vary by about a tenth of the mean. Most of that
difference is not the seed's doing: each world wanders, and forgets its level
within about three hundred [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). Only the shape of their network sets a
few worlds apart: in those, one [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") is joined to a sixth of all agents or
more. With each world measured over its settled life, 19 seeds per condition
reveal a change of a tenth in its number of agents. Experiment 3 had expected
thirty seeds to be enough with worlds measured at the end of their runs; it
was narrowly refuted.

## Same rules, different dice

A Graph of Life world is full of chance. Its [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") start with random
[brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."), ties are broken at random, and the tokens of the dead are dealt out
at random. All of it comes from one number, the run's seed. The seed decides
the starting ring, the founders' brains and every random choice the world
ever makes. So two worlds of the same settings but different seeds share no
single event.

Do they still become the same kind of world? Picture thirty of them side by
side, each summed up by one number, such as how many agents it holds. If the
seed decides little, the thirty numbers lie close together. If it decides
much, they scatter widely, or fall into separate groups.

The answer matters for every experiment in this notebook. An experiment
changes one setting, runs worlds with and without the change, and compares
them. The more worlds differ by chance alone, the more seeds it takes to tell
a real change from luck.

The worlds of this page are the thirty baseline worlds of 100,000 tokens of
[Thirty worlds](a-worlds-life.md#thirty-worlds), seeds 1 to 30. All of them lived to the
end of their 1,500 iterations. The page was first written from the thirty
worlds of 10,000 tokens, and their values are given alongside. Four of those
died out, so their values are for the 26 that lived.

## What we expected

Before the runs of 10,000 tokens, we expected the seed to decide little. A
world holds thousands of agents and lives for thousands of iterations, so its
accidents should average out. Then the thirty worlds' numbers of agents and
[connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."), and their inequality, would differ by less than a fifth of their
typical value. And thirty seeds per condition would be enough to see a setting
change a world by a tenth. That was the thesis of Experiment 3, written down
before any of its runs existed:

<!-- thesis E03 -->
> [!quote] The thesis of Experiment 3, written down before any of its runs existed
> **The claim.** Thirty seeds give thirty different histories, but the same kind of world. Where a world ends up — how many agents, how many connections, how unequal — differs from seed to seed by less than a fifth of its typical value. Thirty seeds per condition are enough to see a change of a tenth in those.
>
> **Why it would be so.** A seed decides the starting ring and every coin the world throws, so no two runs share a single event. But each run is made of thousands of agents and thousands of iterations, and a world with that much going on averages its own accidents away, the way two different shuffles of a large deck still hold the same number of each suit.
>
> **It holds if:** For the population, the connections and the Gini coefficient of wealth at the end of the runs, the standard deviation across the thirty seeds is below a fifth of the mean, and the number of seeds needed to see a 10% change is at most thirty.
>
> **It fails if:** One of these varies across seeds by more than a fifth of its mean, or the endings fall into separate groups — some seeds settling at one level, others at another. Then the seed is part of the story, and every later experiment needs more seeds than planned, or must look at the seeds one by one.
<!-- /thesis -->

The thesis speaks of worlds "at the end of the runs", so it was checked, as
planned, on the last fifth of each run. Its verdict follows the measurements,
under *Was the thesis right?* below.

## Where the worlds settle

Measure each world by its mean over its settled life, iterations 500 to
1,499: its mean number of agents, say. The first few hundred iterations are a
world's youth, unlike the rest of its life, so they are left out. Then divide
each world's mean by the mean of the thirty, so that 1 is the average world.
Each world becomes a dot, and a short column of dots means that the worlds
settle alike:

<!-- figure seed/dots-1 -->
![Where 30 worlds settle: people, wealth, families](../figures/seed/dots-1.svg)

**Where 30 worlds settle: people, wealth, families.** One dot per world: its mean over iterations 500 to 1,499, divided by the mean of that value over the 30 worlds, so that every statistic is on the same scale and 1 is the average world. The bar is the median. A column of dots close to 1 means the worlds settle alike; a tall column, that they do not. The dots are spread sideways only so they do not hide one another.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For each run and statistic, average the statistic over the rows with 500 ≤ `iteration` ≤ 1,499 (rows with `phase` 1 for births, 2 for the rest; `a/b` means the row's `a` divided by its `b`).
> 2. Divide each run's average by the mean of the 30 averages.
> 3. Plot one dot per run, with a bar at the median.
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

<!-- figure seed/dots-2 -->
![Where 30 worlds settle: the shape of the network, genotypes](../figures/seed/dots-2.svg)

**Where 30 worlds settle: the shape of the network, genotypes.** One dot per world: its mean over iterations 500 to 1,499, divided by the mean of that value over the 30 worlds, so that every statistic is on the same scale and 1 is the average world. The bar is the median. A column of dots close to 1 means the worlds settle alike; a tall column, that they do not. The dots are spread sideways only so they do not hide one another.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For each run and statistic, average the statistic over the rows with 500 ≤ `iteration` ≤ 1,499 (rows with `phase` 1 for births, 2 for the rest; `a/b` means the row's `a` divided by its `b`).
> 2. Divide each run's average by the mean of the 30 averages.
> 3. Plot one dot per run, with a bar at the median.
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

How tall a column is can be put in one number, the
[spread](median-and-quantiles.md#spread-between-worlds): the standard deviation of the thirty
worlds' means, divided by their mean. A spread of 0.1 says that a typical
world lies about a tenth of the mean away from the mean.

The columns differ a great deal in height:

- The number of distinct genotypes, or kinds of brain, per agent is almost
  the same in every world: a spread of 0.02.
- Inequality (the [Gini coefficient](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all."), 0.06), connections per agent (0.08) and
  the share of tokens held by the [richest tenth](gini-coefficient.md#the-richest-tenths-share "The share of all tokens held by the richest tenth of the agents.") (0.09) vary little.
- The number of agents (0.11) and of connections (0.12) vary more.
- The number of [families](genotype.md#families), or separate lines of descent
  (0.20), and the number of births in an iteration (0.22) vary more still.

These counts vary *less* than in the worlds of 10,000 tokens. There the
number of agents had a spread of 0.17, and connections 0.19. Larger worlds
average more of their own accidents away.

The shape of the network does the opposite: its columns are taller in the
larger worlds. Here are their spreads, with those at 10,000 tokens in
brackets:

- the share of agents that are leaves, with a single connection: 0.25 (0.12);
- the share of connections that are bridges, whose cut would split the
  network: 0.35 (0.24);
- the share of agents in the [core](bridges.md#core-trees-and-leaves), on loops or
  between them: 0.17 (0.13);
- [clustering](degree.md#clustering "How often an agent's neighbours are also each other's neighbours."), how often two [neighbours](the-world.md#the-network "An agent joined to another by a connection.") of an agent are neighbours of each
  other: 0.62 (0.34).

And these columns are not one even spread. Most worlds sit together, and a
few lie far out. The same few lie far out for leaves, bridges and the core,
and among the lowest for clustering.

## The worlds of a giant hub

Who are the few? They are worlds in which a single agent became a **hub**,
joined to far more agents than most. In each of them, the largest hub was
joined, on average over the settled life, to a sixth of the world or more. In
the world with seed 28, on average 58% of all agents were neighbours of one
agent, and 72% of that world's connections were bridges.

A sixth is one line. The notebook also draws a second, lower one. A world
whose largest hub is joined, on average, to a tenth of the world or more is
called a [star](stars-and-hubs.md). Seven of the thirty worlds were stars.
The two lines differ. Every world past a sixth is also past a tenth, but not
the other way round. So the few far-out worlds are some of the seven stars:
those with the largest hubs.

Across all thirty worlds, the share of agents in the core goes with the share
joined to the largest hub almost exactly. Their
[correlation](correlation.md) is −0.96. A correlation runs from −1 to 1. One
near −1 means that as one quantity goes up, the other goes down, with hardly
an exception. Here, the bigger the hub, the smaller the core.
[What shape does the network take?](what-shape-does-the-network-take.md)
shows what such a world looks like, and
[The largest and the smallest](the-largest-and-the-smallest.md) how a hub
grows and falls.

Hubs of this kind were there in the worlds of 10,000 tokens too. Ten of the
26 were stars, with a hub joined, on average, to a tenth of the world or
more. In three, the hub was joined to a fifth or more. But there the shape of
a world went with its hub far more loosely: a correlation of −0.54. At
100,000 tokens, the hub is almost the whole story of a world's shape.

## A world forgets

The single worlds of [A world's life](a-worlds-life.md) wander: their numbers
drift up and down over hundreds of iterations. How long does a world remember
where it was? Cut each world's life, from iteration 100 on, into 14 stretches
of 100 iterations, and take its mean level in each. Then correlate each
world's level in one stretch with its level some iterations later. This
correlation of a world with its own later self is its
[autocorrelation](autocorrelation.md):

<!-- figure seed/memory -->
![How long a world remembers its level](../figures/seed/memory.svg)

**How long a world remembers its level.** For each statistic, the correlation between a world's level in one stretch of 100 iterations and its level a given number of iterations later, over all pairs of stretches in the 30 worlds. 1 would mean a world stays exactly where it was; 0 that where it was says nothing about where it will be. Because each world's own mean is taken away first, a world cut into only 14 stretches shows negative correlations at long lags even if it simply wandered: the grey dashes are what a random walk, cut and centred the same way, gives (averaged over 200 sets of as many walks), and the grey dots what levels with no memory at all give.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For each run, cut iterations 100 to 1,499 into 14 stretches of 100 and take the mean of the statistic in each.
> 2. Subtract from each run's 14 values their own mean, so that only the run's movement remains.
> 3. For a lag of L stretches, pair every stretch with the one L later, in every run, and take the Pearson correlation of all the pairs together.
> 4. References: 30 series of 1,400 steps, each step normal with mean 0 — summed (a random walk) or not (no memory) — cut, centred and correlated the same way; the mean over 200 such sets (generator seeded with 13).
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

A hundred iterations on, a world is still much where it was. The correlation
is 0.61 for the number of agents and 0.62 for inequality. Two hundred on, it
is about a third of that: 0.21 for both. Three hundred on, nothing is left.
The share of agents in the core is remembered a little longer (0.37 after 200
iterations), births a little less (0.36 after 100).

Then every line goes below zero, to about −0.5 at 700 iterations. This is not
a real anti-memory. Each world was measured against its own average, or
*centred*, so a world above that average in some stretches must be below it
in others. With only 14 stretches per world, that alone pushes distant
stretches to opposite sides.

How far it pushes depends on how the world moves, and the two grey lines
bracket it. One, dashed, is a random walk: a level that adds a random step at
every iteration, and so wanders without limit. Measured the same way, it falls
from 0.83 at 100 iterations to −0.66 at 900. The other, dotted, is a level
with no memory at all; it sits at −0.08 everywhere.

The worlds forget far faster than the random walk: at 100 and 200 iterations
they lie well below it. Their dip for stretches far apart is just what
centring does to a level that drifts over the 1,400 iterations. In the worlds
of 10,000 tokens, with 29 stretches per world, the same dip was only −0.1 to
−0.17 ([Autocorrelation: how long a world remembers](autocorrelation.md)).

## The seed, or time?

The noon temperatures of two towns differ for two reasons. One town may
simply be warmer, by where it lies; and today's weather may favour one. Over a
year the weather averages out, and only the towns' own difference remains.
The differences between worlds can be split the same way.

All the variation in a statistic's 100-iteration levels, 14 stretches in each
of 30 worlds, splits into two parts
([Between worlds and within them](variance-split.md#between-worlds-and-within-them)). One part lies
between the worlds' own averages, like the towns' climates. The other is each
world moving around its own average, like the weather:

<!-- figure seed/between -->
![Variation between worlds, and within each world](../figures/seed/between.svg)

**Variation between worlds, and within each world.** All the variation of a statistic's 100-iteration levels — 14 stretches in each of 30 worlds — split into the part that lies between the worlds' own averages (blue) and the part that is each world moving around its own average (the rest of the column, grey).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take each run's 14 stretch means as for the figure above.
> 2. Between: the variance (n − 1 in the denominator) of the 30 runs' own averages.
> 3. Within: the mean over runs of the variance of each run's 14 values.
> 4. Draw between ÷ (between + within).
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

For the counts and for inequality, about a fifth to a quarter of the
variation lies between the worlds. It is 21% for the number of agents, 27% for
connections and 25% for inequality. The rest, about three quarters, is each
world wandering. Even the part between worlds is partly wandering that 1,400
iterations did not average away.

The share of agents in the core is the exception. More than half of its
variation, 54%, lies between the worlds' own averages (24% at 10,000 tokens).
Which shape a world settles into, with a giant hub or without, is more its
own than how many agents it holds.

A sharper test: is a world that is crowded in the first half of its life also
crowded in the second?

<!-- figure seed/halves -->
![First half against second half](../figures/seed/halves.svg)

**First half against second half.** One dot per world. Left: its mean number of agents over iterations 100–749 (x) against its mean over iterations 750–1,499 (y). Right: the same for connections. A dot on the dashed diagonal had the same mean in both halves. r is the correlation over the 30 dots.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For `nodes` and `edges`, average over the rows with `phase` = 2 and 100 ≤ `iteration` ≤ 749, and over 750 ≤ `iteration` ≤ 1,499.
> 2. Plot the second against the first, one dot per run; r is their Pearson correlation ([correlation](correlation.md)).
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

Hardly. Over the 30 worlds, the correlation between the two halves is 0.17 for
the number of agents and 0.20 for connections. Now pair first and second
halves at random, many times over ([a permutation test](bootstrap.md#permutation-tests)).
Random pairings give a correlation at least that large 18 and 14 times in 100.
So a world's first half says little more about its second than another
world's first half would.

At 10,000 tokens, connections had kept something of their own: a correlation
of 0.41, which random pairings reached only 2 times in 100. At 100,000 tokens
nothing of the kind shows.

## Longer looks, smaller differences

A world forgets its level within a few hundred iterations. So its mean over a
longer stretch averages more of its wandering away, and the worlds look more
alike:

<!-- figure seed/stretch -->
![Longer looks, smaller differences](../figures/seed/stretch.svg)

**Longer looks, smaller differences.** For each statistic, how much the 30 worlds differ — the standard deviation of their averages divided by the mean — when each world is measured by its average over its last L iterations, for L from 100 to 1,400.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. For each length L, average the statistic over the rows with 1,500 − L ≤ `iteration` ≤ 1,499 in each run.
> 2. Divide the standard deviation of the 30 averages (n − 1 in the denominator) by their mean.
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

Measured over its last 100 iterations, the number of agents varies between
worlds by 0.24 of the mean. Over the last 300, the last fifth of a run, it
varies by 0.17; over the last 1,400, by 0.10. Inequality falls from 0.12 to
0.05, families from 0.31 to 0.16, and connections from 0.23 to 0.11. Births
fall most slowly, from 0.27 to 0.18, and at every length but the shortest
they vary most.

## How many seeds an experiment needs

From the spread follows how many worlds each condition needs. Say a setting
changes the mean of a statistic by a share Δ: Δ = 0.1 for a change of a
tenth. An experiment should find that change four times in five, with a test
at the 5% level. If the worlds of a condition vary with a spread *c*, each
condition needs about

$$
n = \frac{2\,(z_1 + z_2)^2\, c^2}{\Delta^2} \approx \frac{15.7\, c^2}{\Delta^2}
$$

worlds, rounded up. Here *z*₁ = 1.960 stands for the 5% level, and
*z*₂ = 0.842 for the four times in five.
[How many seeds an experiment needs](bootstrap.md#how-many-seeds-an-experiment-needs) shows where the formula
comes from. The seeds needed grow with the square of the spread, and fall with
the square of the change. For the number of agents over the settled life,
*c* = 0.11. So a change of a tenth needs *n* = 15.7 × 0.11² / 0.1² ≈ 19
seeds:

<!-- figure seed/seeds-needed -->
![Seeds a condition needs](../figures/seed/seeds-needed.svg)

**Seeds a condition needs.** How many worlds each of two conditions needs for a change of a given size in the average to be found four times in five at the 5% level, if worlds vary as much as these do. Dashed: worlds measured over their last fifth (iterations 1,200–1,499); solid: over iterations 500–1,499. The grey line is 30 seeds.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Measure each run by its average over the stretch, and compute the spread c = sd ÷ mean of the 30 averages.
> 2. For a change Δ (as a share of the mean), n = 2 (z₁ + z₂)² c² / Δ², with z₁ = 1.960 (5%, two-sided) and z₂ = 0.842 (power 80%); round up.
>
> **To make it again:** `python3 book_figures.py seed`.
<!-- /figure -->

Measured over the last fifth of the runs (dashed), seeing a change of 10% in
the number of agents takes 48 seeds per condition. Measured over the settled
life (solid), it takes 19. The worlds of 10,000 tokens needed 100 and 46. For
inequality it takes 14 and 7.

Turn this around, and take 30 seeds per condition with the settled-life
measure. Then the smallest change seen four times in five is about 8% in the
number of agents, 9% in connections and 5% in inequality. Larger worlds make
sharper experiments, at a price: each costs about eight times the computing
([Thirty worlds](a-worlds-life.md#thirty-worlds)).

Two more things follow for experiments:

- **Pairing by seed helps little.** Two conditions run with the same seed
  share their starting ring and founders. But the seed hardly decides where a
  world goes, so they share little else. A comparison seed by seed is about as
  noisy as one between unrelated worlds.
- **Shape needs more seeds than size.** Only a few worlds in thirty grow a
  giant hub. A change of setting that made giant hubs twice as common would
  need far more than thirty seeds to see.

## Was the thesis right?

The thesis was checked as planned, on the last fifth of each run of 10,000
tokens: iterations 2,400 to 2,999. Inequality passed both of its tests: its
spread was 0.13, and 26 seeds would see a change of a tenth. Connections
passed the first test and failed the second: a spread of 0.19, but 58 seeds.
The number of agents failed both: a spread of 0.25, and 100 seeds. So the
thesis was refuted, narrowly. But it held in what it says about *kind*: there
were no separate groups of worlds.

At 100,000 tokens, on the last fifth of each run (iterations 1,200 to 1,499),
the thesis is refuted again, more narrowly. All three spreads are now below a
fifth: the number of agents 0.17, connections 0.19, inequality 0.09. But
seeing a change of a tenth still takes more than thirty seeds for the first
two (48 and 55), and 14 for inequality. On kind, the thesis holds less well
than before. In agents, connections and inequality the worlds spread evenly.
But in the shape of their network a few lie far from the rest: the worlds of a
giant hub.

What the thesis missed, at both sizes, is where the spread comes from. It is
mostly not the seed but time. Each world wanders, and the end of a run catches
it wherever it happens to be. That is why every later page measures worlds
over their settled life, from iteration 500 on.

## What we learned

- Worlds that differ only in their seed settle alike in size and inequality.
  At 100,000 tokens they vary by about a tenth of the mean or less: less than
  at 10,000 tokens.
- Most of what tells such worlds apart is time, not the seed: a world forgets
  its level within about three hundred iterations.
- The shape of the network is the exception: a few worlds grow a giant hub,
  and stay apart from the rest.
- Measure a world over its settled life, not at its end. A change of a tenth
  in the number of agents then takes 19 seeds, not 48.

## Where to go next

- [A world's life](a-worlds-life.md): how a single world wanders.
- [Stars and hubs](stars-and-hubs.md): the worlds that one agent holds
  together.
- [How many seeds an experiment needs](bootstrap.md#how-many-seeds-an-experiment-needs): the formula behind
  the seed counts, step by step.
- [Thirty worlds](a-worlds-life.md#thirty-worlds): the thirty worlds of this page, and the
  other questions asked of them.

## How this was made

This page reads two sets of runs. The thirty worlds of 100,000 tokens are the
runs of Experiment 10:

<!-- runs E10 -->
> [!info]- The runs behind this page
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`hidden_activation`](settings.md#hidden_activation) | sigmoid | The curve each hidden unit applies to its sum: `sigmoid`, which passes on at most a quarter of a change, so that deep brains barely respond to what they see, or `tanh`, the same S-shape centred on zero, through which a change still passes in several layers. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`contest`](settings.md#contest) | largest | Who takes a contested node: the `largest` stake as the game weighs it, or a staker drawn by `lottery` with a chance in proportion to its stake (needs revolutions off). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none); `local` lets each estate flow through the network to the nearest living agents, split evenly at every step. |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`link_token`](settings.md#link_token) | off | Whether every link holds one token: its maker pays for it, a lapsing link gives it back to one of its ends, and a link of a dead agent to the end that lives on, so the links can never outnumber the tokens. |
> | [`link_upkeep`](settings.md#link_upkeep) | off | Whether an agent may pay a token an iteration to keep a link, each brain deciding for each neighbour: `paid_or_used` keeps a link that was paid for or used, `paid_only` only one that was paid for; `off`, only use keeps a link. |
> | [`upkeep_to`](settings.md#upkeep_to) | end | Where a link's upkeep goes: to the link's other end (`end`), or into the estate of the dead (`estate`), shared out by the world's redistribution rule. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`mate_pays`](settings.md#mate_pays) | off | Under mix_at_birth, whether the mate pays the child as many tokens as the parent gave it, as far as it can and never its last. |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

The thirty worlds of 10,000 tokens, on which the thesis was first tested, are
the runs of Experiment 2:

<!-- runs E03 -->
> [!info]- The runs behind this page
> These are the runs of Experiment 2; nothing new was run for this page.
>
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 3,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 10,000 tokens. Runs `B1-10000-s001` … `B1-10000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E02`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`hidden_activation`](settings.md#hidden_activation) | sigmoid | The curve each hidden unit applies to its sum: `sigmoid`, which passes on at most a quarter of a change, so that deep brains barely respond to what they see, or `tanh`, the same S-shape centred on zero, through which a change still passes in several layers. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`contest`](settings.md#contest) | largest | Who takes a contested node: the `largest` stake as the game weighs it, or a staker drawn by `lottery` with a chance in proportion to its stake (needs revolutions off). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none); `local` lets each estate flow through the network to the nearest living agents, split evenly at every step. |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`link_token`](settings.md#link_token) | off | Whether every link holds one token: its maker pays for it, a lapsing link gives it back to one of its ends, and a link of a dead agent to the end that lives on, so the links can never outnumber the tokens. |
> | [`link_upkeep`](settings.md#link_upkeep) | off | Whether an agent may pay a token an iteration to keep a link, each brain deciding for each neighbour: `paid_or_used` keeps a link that was paid for or used, `paid_only` only one that was paid for; `off`, only use keeps a link. |
> | [`upkeep_to`](settings.md#upkeep_to) | end | Where a link's upkeep goes: to the link's other end (`end`), or into the estate of the dead (`estate`), shared out by the world's redistribution rule. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`mate_pays`](settings.md#mate_pays) | off | Under mix_at_birth, whether the mate pays the child as many tokens as the parent gave it, as far as it can and never its last. |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 3,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

Every figure of this page is drawn again by `python3 book_figures.py seed`.
The check of the thesis at 10,000 tokens is in `book/results/E03.json`,
written by `python3 gol_lab.py analyse E03`.

## How our view changed

- **11 October 2026.** The page was first written from the worlds of 10,000
  tokens. It said that no column of its dot figures falls into separate
  clumps, "as one kind of world would". It also said that the spread between
  worlds is "not the seed but time". The worlds of 100,000 tokens showed
  otherwise for the shape of the network. There a few worlds, those of a
  giant hub, lie far from the rest. And more than half (54%) of the variation
  in the share of agents in the core lies between the worlds. So the page now
  says that the spread is *mostly* time. It changed when it was rewritten
  from the thirty worlds of Experiment 10 ([Thirty worlds](a-worlds-life.md#thirty-worlds)).
- **11 October 2026.** The page said that a world keeps something of its own
  in how densely it is connected, "perhaps a trace of its starting ring". At
  10,000 tokens, the two halves of a world's life agreed in connections with
  a correlation of 0.41. And the spread of connections stopped falling at
  about 0.19, however long the look. In the worlds of 100,000 tokens neither
  holds. The halves agree only as well as chance would make them (0.20), and
  the spread of connections falls on, to 0.11. The page changed in the same
  rewrite.

<!-- turns -->
---

← [The life of a B1 world](a-worlds-life.md) · [The baselines B1 and B2](../README.md#the-baselines-b1-and-b2) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (15)
> [Seeds and the random stream](random-numbers.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [How the game is played: what agents stake, and who wins the nodes](how-the-game-is-played.md) · [Power laws, real and apparent, and how to fit one](power-laws-real-and-apparent.md) · [Stars and hubs](stars-and-hubs.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [Do the brains matter?](do-the-brains-matter.md) · [How worlds are measured](how-worlds-are-measured.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [Many worlds at once: the median, bands, the settled life and the spread](median-and-quantiles.md) · [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md) · [How long a world remembers: autocorrelation and the power spectrum](autocorrelation.md) · [How sure can we be? The bootstrap, permutation tests, the Wilson interval and how many seeds](bootstrap.md) · [Splitting variation: where, who and chance, and between worlds and within them](variance-split.md)
<!-- /turns -->
