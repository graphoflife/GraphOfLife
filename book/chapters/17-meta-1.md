# Meta I · What the baseline world is

Part II looked at thirty baseline worlds from every side. This chapter puts
the pieces together: what the theses predicted, what was found instead, what
seems to hold across the chapters, and what that changed in how the book
measures. It was first written after Experiments 1 to 5 and is brought up to
date here, with everything Part II found.

> [!question] Questions of this chapter
> - What is a settled baseline world, in numbers?
> - What do the chapters of Part II, taken together, suggest about how it
>   works?
> - What do the experiments still need — and in what order?

## The record

- [Chapter 7](07-is-a-run-reproducible.md) — *Is a run reproducible?* —
  thesis **refuted**, but only by the number of threads of the matrix
  library, in the last bits. Every run now uses one thread, and is then
  exactly reproducible.
- [Chapter 9](09-a-worlds-life.md) — *A world's life* — thesis **refuted**.
  A youth of a hundred iterations, then a long wander; births fall by 41%;
  4 of 30 worlds die.
- [Chapter 12](12-how-much-does-the-seed-decide.md) — *How much does the seed
  decide?* — thesis **refuted, narrowly**. Little: worlds differ mostly
  because each one wanders, not because of their seeds.
- [Chapter 13](13-where-do-the-tokens-go.md) — *Where do the tokens go?* —
  thesis **refuted**. Inequality is moderate (a Gini near 0.5) and steady,
  and agents stake 70% of their tokens on their neighbours.
- [Chapter 15](15-what-shape-does-the-network-take.md) — *What shape does the
  network take?* — thesis **refuted, narrowly**. Half tree, half web.
- [Chapter 16](16-genotypes-and-lineages.md) — *Genotypes and lineages* —
  thesis **refuted**. Genotypes last two iterations; lines take the world
  over, again and again.

Six of six theses failed, two of them narrowly. That is what a thesis
written before its runs is for: each failure is something about the world
that was not known before.

## The baseline world in numbers

A settled baseline world of 10,000 tokens — each number the median over the
26 worlds that lived to the end of each world's mean over iterations 500 to
2,999:

- **about 1,300 agents** with **3.3 connections** each
  ([Chapter 9](09-a-worlds-life.md), [Chapter 15](15-what-shape-does-the-network-take.md));
- **3.1 births** per hundred agents per iteration, and as many deaths, three
  in four of them agents cut off from the network
  ([Chapter 11](11-births-deaths-and-ages.md));
- a **Gini coefficient** of wealth near **0.5**, the richest tenth holding
  **44%** ([Chapter 13](13-where-do-the-tokens-go.md));
- **45%** of agents keep their node in a game; **half** of all nodes are won
  by coalitions ([Chapter 14](14-how-the-game-is-played.md));
- **31%** of connections bridges, **36%** of agents leaves, **54%** in a core
  of loops, **10 steps** from one agent to another
  ([Chapter 15](15-what-shape-does-the-network-take.md));
- a genotype for every **1.6 agents**, each lasting a couple of iterations;
  lines that take the whole world over, about **every 350 iterations**
  ([Chapter 16](16-genotypes-and-lineages.md)).

## What seems to hold across the chapters

### One slow wander, in everything

[Chapter 12](12-how-much-does-the-seed-decide.md) measured how long a world
remembers its level, statistic by statistic. They all forget at much the same
pace: a hundred iterations later, a world's level is still clearly alike
(correlations of 0.64 to 0.72 for the number of agents, connections,
inequality, leaves, genotypes, families, bridges, clustering and the core);
three hundred later, little is left (0.12 to 0.31); five hundred later,
nothing. Size, wealth and shape do not seem to wander each on its own
schedule; they look like one slow process seen through many windows.

[Chapter 16](16-genotypes-and-lineages.md), from the same runs, found a
process with about that rhythm: every 350 iterations or so, one branch of the
family tree outlives all the others and the whole world descends from it. A
sweep replaces the brains of a whole world, and with them how it plays.

> [!question] An open question
> Is the wander the sweeps? Two tests would tell: whether a world's level
> moves faster around a sweep than between sweeps, and whether a world whose
> brains never change still wanders. [Chapter 30](30-do-the-brains-matter.md)
> did the second: a world without change hardly wanders at all.

### A world of attack

Agents put only 30% of their staked tokens on their own node
([Chapter 13](13-where-do-the-tokens-go.md)), more than half of all nodes
change hands in every game, and coalitions of smaller stakers take half of
all nodes ([Chapter 14](14-how-the-game-is-played.md)). Over the first five
hundred iterations, the share of agents that keep their node rose, while the
share of tokens they put at home barely moved. Defence got better without
agents staking more on their own node; what changed instead is not yet known.

### A world that ages

Births fell by 41% between iterations 100–200 and the last two hundred, and
the median age of the living rose from 26 iterations to 80
([Chapter 11](11-births-deaths-and-ages.md)). The baseline is still changing
at iteration 3,000. That could be a world slowly settling towards something,
or a trend — brains that have fewer children, or children that last — and a
lasting trend is what the second rung of the ladder in
[Chapter 1](01-what-this-book-is-about.md) looks for: adaptation that
accumulates.

## What this changed in the method

- **The youth is left out.** Every thesis from here on is checked from
  iteration 500 on, the settled life of a run ([The settled life](../notes/settled-life.md)).
- **Long stretches, not the last fifth.** A run's mean over its settled life
  varies across seeds by much less than its last fifth does — for the number
  of agents 0.17 of the mean instead of 0.25 — so it needs fewer than half the
  seeds ([How many seeds](../notes/seeds-needed.md)).
- **Thirty seeds see changes of about a tenth.** Over the settled life,
  thirty seeds per condition can see a change of 12% in the number of agents,
  14% in connections and 5% in inequality, four times in five. Smaller effects
  are reported as not seen, not as absent.
- **The dead are counted apart.** About one world in eight dies at this
  size. Extinction is reported for every condition; only a large change in it
  — from 13% to 30% would take about ninety seeds per condition — can be seen
  ([When a world ends](../notes/extinction.md)).
- **Pairing by seed helps little.** Two conditions with the same seed share
  their starting ring and little else.
- **Lines, not genotypes.** A genotype lasts two iterations; what lasts,
  spreads and takes over is a line, and the chapters to come follow lines.

## What came next

Part III then measured the same thirty worlds through every statistic the
viewer offers — entropy, gains and losses, the flow of tokens, children, power
laws, scaling, geometry, how worlds break, pictures, cooperation — and closed
with a chapter questioning the rules themselves and a second meta chapter,
[Meta II](29-meta-2.md), on what all of it means for open-ended evolution.

Every number of Part II is a number about the baseline, with nothing to hold
it against. The rules of this book ask for a null beside every number, and
the most important one was missing: a world in which the brains make no
difference. So the control came first in Part IV:
[Chapter 30](30-do-the-brains-matter.md) runs worlds whose brains never
change, and worlds in which every decision is drawn at random.
[Chapter 31](31-how-does-a-worlds-size-follow-its-tokens.md) then asks, at
the author's request, how a world's size follows its tokens. The chapters
planned after them — the rate of mutation, coalitions switched off, the size
of a brain — are listed in [the contents](../README.md).

<!-- turns -->
---

← [Chapter 16 · Genotypes and lineages](16-genotypes-and-lineages.md) · [Contents](../README.md) · [Chapter 18 · How even is a world?](18-how-even-is-a-world.md) →
<!-- /turns -->
