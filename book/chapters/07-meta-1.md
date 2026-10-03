# Meta I · What the first five experiments say

## In short

The baseline world can be made again exactly, and every seed makes the same
kind of world — but not a calm one. A world has a short, violent youth, and
then wanders for thousands of iterations: its size, its inequality and its
shape drift together, each remembering where it was for about three hundred
iterations, while births slowly become rarer and one world in eight dies.
The theses written before the runs were mostly right about averages and
mostly wrong about how the world works: agents attack far more than they
defend, the network keeps a core of loops, and the seed hardly matters. The
book now measures differently — over the settled life of a run — and moves one experiment to the front: the controls in which brains decide
nothing or never change, which are the null every result so far is missing.

## The record so far

| Chapter | Question | Thesis | What was found |
|---|---|---|---|
| 2 | Is a run reproducible? | holds, but for one clause | Yes, exactly — on one thread for the matrix library, which every run now uses |
| 3 | The life of a world | refuted | A youth of a hundred iterations, then a long wander; births fall by 41%; 4 of 30 worlds die |
| 4 | How much does the seed decide? | refuted, narrowly | Little: worlds differ mostly because each one wanders, not because of their seeds |
| 5 | Where do the tokens go? | refuted | Inequality is moderate (Gini 0.5) and steady, and agents stake 70% of their tokens on their neighbours |
| 6 | What shape does the graph take? | refuted, narrowly | Half tree, half web: a third of connections are bridges, half of the agents sit in a core of loops |

Four of five theses failed. That is what a thesis written before its runs is
for: each failure is something about the world that was not known before.

## What holds across the chapters

### One slow wander, in everything

Chapter 4 measured how long a world remembers its level, statistic by
statistic. They all forget at much the same pace: a hundred iterations
later, a world's level is still clearly alike (correlations of 0.64 to 0.72
for the number of agents, connections, inequality, leaves, genotypes,
families, bridges, clustering and the core); three hundred later, little is left
(0.12 to 0.31); five hundred later, nothing. Only births and deaths are
remembered a little longer. Size, wealth and shape do not each wander on
their own schedule; they look like one slow process seen through many
windows.

Chapter 8, from the same runs, found a process with about that rhythm: every
350 iterations or so, one branch of the family tree outlives all the others
and the whole world descends from it. A sweep replaces the brains of a whole
world, and with them how it plays. **Hypothesis:** the wander is the sweeps.
It can be tested in two ways — whether a world's level moves faster around a
sweep than between sweeps, and whether a world whose brains decide nothing
still wanders.

### A world of attack

Chapter 5 found that agents put only 30% of their stakes on their own node,
and Chapter 8 that 55% of all nodes change hands in every game, while
coalitions of smaller stakers take half of all nodes. Over the first five hundred iterations
the share of agents that keep their node rose from 32% to 45%, while the
share of stakes they put at home barely moved (27% to 30%). Defence got
better without agents staking more on their own node; what changed instead
is not yet known.

### A world that ages

Births fell by 41% between iterations 100–200 and the last two hundred, and
the median age of the living rose from 26 iterations to 80. The baseline is
still changing at iteration 3,000. That could be a world slowly settling
towards something, or a trend — brains that have fewer children, or
children that last — and a lasting trend is what the second rung of the
ladder in Chapter 1 looks for: adaptation that accumulates.

### What the world is

Put together, the baseline world at 10,000 tokens is: about 1,300 agents
with 3.3 connections each; a Gini coefficient of wealth near 0.5, the
richest tenth holding 44%; a third of all connections bridges, a third of
all agents leaves, half of all agents in a core of loops, and ten steps
from one agent to another; a genotype for every 1.6 agents, each lasting a
couple of iterations; and lineages that take over the whole world, one after
another.

## What this changes in the method

- **The youth is left out.** Every thesis from now on is checked from
  iteration 500 on, the settled life of a run.
- **Long stretches, not the last fifth.** A run's average over its settled
  life varies across seeds by much less than its last fifth does — for the
  number of agents 0.17 of the mean instead of 0.25 — so it needs fewer than
  half the seeds.
- **Thirty seeds see changes of about a tenth.** Over the settled life, thirty
  seeds per condition can see a change of 12% in the number of agents, 14% in
  connections and 5% in inequality, four times in five. Smaller effects are
  reported as not seen, not as absent.
- **The dead are counted apart.** About one world in eight dies at this size.
  Extinction is reported for every condition, and only a large change in it —
  from 13% to 30% would take about ninety seeds per condition — can be seen.
- **Pairing by seed helps little.** Two conditions with the same seed share
  their starting ring and little else.
- **Lineages, not genotypes.** A genotype lasts two iterations; the thing that
  lasts, spreads and takes over is a lineage, and the chapters to come follow
  lineages.

## What comes next

Every number in Chapters 3 to 8 is a number about the baseline, with nothing
to hold it against. The rules of this book ask for a null next to every
number, and the most important null is still missing: a world in which the
brains make no difference. Without it, nothing so far can say which of what
was seen comes from the brains and which from the rules alone — not the
wander, not the sweeps, not the attack, not the ageing.

So the roadmap changes. The control, planned as the last experiment of
Part I, comes first — and with a second one beside it, in which the brains
still decide but never change, so that what evolution adds can be told from
what the brains do at all:

| Chapter | Question | One change |
|---|---|---|
| 9 | Do the brains matter at all? | brains that never change (`mutation_sparsity` 0); every decision drawn at random (`random_decisions`) |
| 10 | How fast should brains change? | the mutation probability |
| 11 | What keeps wealth spread? | no coalitions (`allow_revolutions` off) |
| 12 | How big should a brain be? | the hidden layers |
| 13 | Meta II | |
| 14 | Does size change the dynamics? | the number of tokens |

Each of Chapters 9 to 12 is measured against the thirty baseline runs of
Chapter 3, which it reuses, with thirty seeds of its own and the settled life
of a run as its measure. Size needs worlds of 100,000 tokens and more, and
comes after Meta II, when the measures are settled.
