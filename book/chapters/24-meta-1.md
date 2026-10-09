# Meta I · What the baseline world is

Part II looked at thirty baseline worlds from every side. This chapter puts
the pieces together: what the theses predicted, what was found instead, what
seems to hold across the chapters, and what that changed in how the book
measures. It was first written after Experiments 1 to 5 and is brought up to
date here, with everything Part II found — last with its six chapters on the
brains, the messages, mobility, growth, likeness and dimension
(Chapters 18 to 23).

> [!question] Questions of this chapter
> - What is a settled baseline world, in numbers?
> - What do the chapters of Part II, taken together, suggest about how it
>   works?
> - What do the experiments still need — and in what order?

## The record

- [Chapter 8](08-is-a-run-reproducible.md) — *Is a run reproducible?* —
  thesis **refuted**, but only by the number of threads of the matrix
  library, in the last bits. Every run now uses one thread, and is then
  exactly reproducible.
- [Chapter 10](10-a-worlds-life.md) — *A world's life* — thesis **refuted**.
  A youth of a hundred iterations, then a long wander; births fall by 41%;
  4 of 30 worlds die.
- [Chapter 13](13-how-much-does-the-seed-decide.md) — *How much does the seed
  decide?* — thesis **refuted, narrowly**. Little: worlds differ mostly
  because each one wanders, not because of their seeds.
- [Chapter 14](14-where-do-the-tokens-go.md) — *Where do the tokens go?* —
  thesis **refuted**. Inequality is moderate (a Gini near 0.5) and steady,
  and agents stake 70% of their tokens on their neighbours.
- [Chapter 16](16-what-shape-does-the-network-take.md) — *What shape does the
  network take?* — thesis **refuted, narrowly**. Half tree, half web.
- [Chapter 17](17-genotypes-and-lineages.md) — *Genotypes and lineages* —
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
  ([Chapter 10](10-a-worlds-life.md), [Chapter 16](16-what-shape-does-the-network-take.md));
- **3.1 births** per hundred agents per iteration, and as many deaths, three
  in four of them agents cut off from the network
  ([Chapter 12](12-births-deaths-and-ages.md));
- a **Gini coefficient** of wealth near **0.5**, the richest tenth holding
  **44%** ([Chapter 14](14-where-do-the-tokens-go.md));
- **45%** of agents keep their node in a game; **half** of all nodes are won
  by coalitions ([Chapter 15](15-how-the-game-is-played.md));
- **31%** of connections bridges, **36%** of agents leaves, **54%** in a core
  of loops, **10 steps** from one agent to another
  ([Chapter 16](16-what-shape-does-the-network-take.md));
- a genotype for every **1.6 agents**, each lasting a couple of iterations;
  lines that take the whole world over, about **every 350 iterations**
  ([Chapter 17](17-genotypes-and-lineages.md));
- brains with a **gain of 0.0015**: a change in what a brain sees reaches
  what it does at a seven-hundredth of its size. Selection has set their
  constants alike in every world — a child in 2.6% of cases, the stake
  spread in 98% — and in the median world 58% of agents score every
  candidate at zero or below, so the rules split their tokens evenly
  ([Chapter 18](18-what-the-brains-are-like.md));
- messages that are **names**: one agent's messages to two readers differ
  by 0.0005, two agents' by 0.55, and agents of one genotype write the same
  ([Chapter 19](19-what-agents-say-to-each-other.md));
- an order of wealth that is **half forgotten in fifty games**, and rich
  agents with one connection that die three times as often as other leaves
  ([Chapter 20](20-do-the-rich-stay-rich.md));
- connections gained in proportion to the connections an agent has
  (k^0.94) and lost slightly faster (k^1.11), and **no new connection ever
  bringing two agents closer** ([Chapter 21](21-how-the-network-grows.md));
- likeness that fades within **two steps** — tokens, age, connections and
  kin alike ([Chapter 22](22-like-next-to-like.md));
- a **finite dimension**: about 3 by the room within *r* steps (3.3 in
  worlds of 60,000 agents), about 2 by how a walker spreads, where random
  networks with the same connections have none
  ([Chapter 23](23-how-many-dimensions-does-a-world-have.md)).

## What seems to hold across the chapters

### One slow wander, in everything

[Chapter 13](13-how-much-does-the-seed-decide.md) measured how long a world
remembers its level, statistic by statistic. They all forget at much the same
pace: a hundred iterations later, a world's level is still clearly alike
(correlations of 0.64 to 0.72 for the number of agents, connections,
inequality, leaves, genotypes, families, bridges, clustering and the core);
three hundred later, little is left (0.12 to 0.31); five hundred later,
nothing. Size, wealth and shape do not seem to wander each on its own
schedule; they look like one slow process seen through many windows.

[Chapter 17](17-genotypes-and-lineages.md), from the same runs, found a
process with about that rhythm: every 350 iterations or so, one branch of the
family tree outlives all the others and the whole world descends from it. A
sweep replaces the brains of a whole world, and with them how it plays.

> [!question] An open question
> Is the wander the sweeps? Two tests would tell: whether a world's level
> moves faster around a sweep than between sweeps, and whether a world whose
> brains never change still wanders. [Chapter 37](37-do-the-brains-matter.md)
> did the second: a world without change hardly wanders at all.

### A world of attack

Agents put only 30% of their staked tokens on their own node
([Chapter 14](14-where-do-the-tokens-go.md)), more than half of all nodes
change hands in every game, and coalitions of smaller stakers take half of
all nodes ([Chapter 15](15-how-the-game-is-played.md)). Over the first five
hundred iterations, the share of agents that keep their node rose, while the
share of tokens they put at home barely moved. Defence got better without
agents staking more on their own node; what changed instead is not yet known.

### A world that ages

Births fell by 41% between iterations 100–200 and the last two hundred, and
the median age of the living rose from 26 iterations to 80
([Chapter 12](12-births-deaths-and-ages.md)). The baseline is still changing
at iteration 3,000. That could be a world slowly settling towards something,
or a trend — brains that have fewer children, or children that last — and a
lasting trend is what the second rung of the ladder in
[Chapter 1](01-what-this-book-is-about.md) looks for: adaptation that
accumulates.

### Brains that cannot see

[Chapter 18](18-what-the-brains-are-like.md) opened the brains and found
them nearly blind. Five sigmoid layers pass a change in the inputs on at a
quarter of its size each, about a thousandth in all. Much of the rest of
Part II follows from that one number:

- the stakes are nearly even, and the game moves tokens almost as a random
  walk would ([Chapter 27](27-where-the-tokens-flow.md)), down to the
  two-game rhythm of wealth that a random walk shows on hubs and leaves
  ([Chapter 20](20-do-the-rich-stay-rich.md));
- an agent's tokens are set by its place, connections plus one, and not by
  how it plays ([Chapter 30](30-how-properties-scale-together.md)); the rich
  are those the walk's resting state favours;
- messages carry a name and nothing else
  ([Chapter 19](19-what-agents-say-to-each-other.md)), and kin, who could
  tell each other by it, cannot use it
  ([Chapter 34](34-do-agents-cooperate.md));
- selection has acted, the same way in every world, but on constants: have
  few children, spread the stake, make every token revolutionary.

Evolution so far is the evolution of a few constants. That is the first
obstacle to open-ended evolution, and it sits in the architecture, not in
the world.

### A space, not a small world

New connections are only ever made inside a neighbourhood, and none brings
two agents closer ([Chapter 21](21-how-the-network-grows.md)). So a world
grows as a body does. It has a finite dimension
([Chapter 23](23-how-many-dimensions-does-a-world-have.md)), and its width
grows with its size ([Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md)),
where a random network of the same connections would be a small world. Yet
nothing in the space keeps regions apart: likeness fades within two steps
([Chapter 22](22-like-next-to-like.md)). A world has the room for regions
that differ, and nothing to make them.

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
  size — falls, that is, to 20 agents, which [Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md)
  later found is not always the end. Extinction is reported for every condition; only a large change in it
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
[Meta II](36-meta-2.md), on what all of it means for open-ended evolution.

Every number of Part II is a number about the baseline, with nothing to hold
it against. The rules of this book ask for a null beside every number, and
the most important one was missing: a world in which the brains make no
difference. So the control came first in Part IV:
[Chapter 37](37-do-the-brains-matter.md) runs worlds whose brains never
change, and worlds in which every decision is drawn at random.
[Chapter 38](38-how-does-a-worlds-size-follow-its-tokens.md) then asks, at
the author's request, how a world's size follows its tokens. The chapters
planned after them — the rate of mutation, coalitions switched off, the size
of a brain — are listed in [the contents](../README.md). After
[Chapter 18](18-what-the-brains-are-like.md), the size and shape of a brain
moves up: a brain that cannot see is the first thing to change.
[Chapter 51](51-worlds-of-100000-tokens.md) prepares a second baseline of
thirty worlds of 100,000 tokens, for the questions that need room.

<!-- turns -->
---

← [Chapter 23 · How many dimensions does a world have?](23-how-many-dimensions-does-a-world-have.md) · [Contents](../README.md) · [Chapter 25 · How even is a world?](25-how-even-is-a-world.md) →
<!-- /turns -->
