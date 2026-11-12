# The world: tokens, agents and a network

> [!question] Questions of this page
> - What exactly is a world: what does it consist of, and what stays fixed?
> - How is a world built at the start, from its settings and its seed?
> - What is written down about it as it runs?

## In short

A world is a network of nodes joined by connections, with one agent on every
node. Every agent holds a whole number of tokens and a brain that takes all
its decisions. Tokens move between agents but are never made or destroyed,
so their total, *T*, never changes. A world starts as *T*/100 founders of
100 tokens each, on a ring with a few random shortcuts. Everything after that
follows from the world's settings and its seed, and the world is written down
after every phase.

## A world at a glance

Picture a fishing net spread out on the ground. Each knot is a **node**, and
each string between two knots is a **connection**. On every knot stands a
player, an **agent**. It holds a purse of coins, its **tokens**, and a small
rulebook, its **brain**. An agent deals only with the agents tied to its own
knot. Coins pass from hand to hand, but no coin is ever minted or melted.

Time runs in iterations of two phases, reproduction and the game
([One iteration](one-iteration.md#the-picture)).

This page says exactly what a world is made of and how it begins.
[One iteration, step by step](one-iteration.md) says how it changes, and
[The brain](the-brain.md) how its agents decide.

## The network

Draw a few dots on paper and join some pairs of them with lines. That is a
**graph**: the dots are its nodes, and the lines its connections (in graph
theory, *edges*). Here is a small one. Take four nodes *a*, *b*, *c* and
*d*, and four connections: {*a*, *b*}, {*a*, *c*}, {*b*, *c*} and
{*c*, *d*}. So *a*, *b* and *c* form a triangle, and *d* hangs on *c*.

In symbols, a world is a graph *G* = (*V*, *E*): a set *V* of nodes and a set
*E* of connections. Three rules hold for every connection:

- it joins two different nodes;
- it has no direction, so the connection {*u*, *v*} is the same as
  {*v*, *u*};
- two nodes are joined at most once.

The **neighbours** of a node *u* are the nodes it is joined to:

$$
N(u) = \{\, v \in V : \{u, v\} \in E \,\}.
$$

In words: *N*(*u*) is the set of every node *v* whose connection with *u* is
in *E*. The **degree** of *u* is how many neighbours it has,
deg(*u*) = |*N*(*u*)|. In the example, *N*(*c*) = {*a*, *b*, *d*}, so
deg(*c*) = 3; *a* and *b* have degree 2, and *d* has degree 1.

Every connection has two ends, so the degrees add up to twice the number of
connections. In the example, 2 + 2 + 3 + 1 = 8 = 2 · 4. So the **mean
degree**, the number of connections an agent has on average, is twice the
number of connections divided by the number of nodes:

$$
\bar k = \frac{1}{|V|} \sum_{u \in V} \deg(u) = \frac{2\,|E|}{|V|} .
$$

In the example, *k̄* = 8 / 4 = 2. [Degree](measuring-networks.md#an-agents-connections-degree-and-clustering) says more about
degrees.

Every node carries a whole-number **id**. Ids are handed out in increasing
order and never reused. So an id names one agent for its whole life. And an
agent with a smaller id is never younger than one with a larger id.

## Agents and tokens

On every node lives one **agent**; in this notebook "agent" and "node" mean
the same thing. An agent has four things:

- **tokens**: a whole number τ(*u*) (τ is the Greek letter tau). After every
  phase, every living agent holds at least one: an agent left with none dies
  ([One iteration, step by step](one-iteration.md));
- **a brain** *B*(*u*): a neural network that takes all its decisions
  ([The brain](the-brain.md));
- **an age**: the number of iterations since it was born
  ([Age and lifetime](measuring-lives.md#age-lifetime-and-survival-curves));
- **a genotype**, the identity of its brain. An agent can lose its genotype.
  When another agent wins its node in the game, the node takes on a copy of
  the winner's brain ([Genotypes and the family tree](measuring-evolution.md#genotypes-families-and-the-common-ancestor)).

The world holds a fixed number of tokens, *T*. Tokens move between agents,
but they are never made or destroyed. So after every phase of every
iteration, the tokens of all the agents add up to *T*:

$$
\sum_{u \in V} \tau(u) = T .
$$

[Why the tokens add up](one-iteration.md#why-the-tokens-add-up) checks that
every rule keeps this sum. Tokens are the only thing of value. An agent needs
them to have a child, and in the game they are what it fights with. And since
every living agent holds at least one token, a world can never hold more than
*T* agents. A world of 10,000 tokens holds at most 10,000 agents, and in
practice far fewer.

## How a world begins

A world begins as a circle of equal founders. Each is joined to its nearest
neighbours on either side, and a few connections reach across the circle
instead. Every founder holds the same number of tokens, and a brain of its
own, drawn at random.

A world is fixed by its **settings** and its **seed**. The settings are 41
numbers and switches, chosen before the run starts.
[Every setting](settings.md) lists them all, with their values in the baseline
B1. A **baseline** is a set of rules that an experiment starts from and
changes one thing of. B1 is the first
([B1 and B2: the two baselines](b1-and-b2.md)). Eighteen of the settings were
added later, in the search for open-ended evolution: evolution that keeps
producing new things without end. B1 keeps every one of them at its old rule,
so its rules are exactly the ones these pages describe. The **seed** is a
whole number that fixes every random choice of the run.

The settings that shape the start are the number of tokens *T*, the number of
founders and their ring. These pages teach with worlds of *T* = 10,000 tokens.
The study of the baseline reads the same baseline at *T* = 100,000. That is
ten times as many founders of 100 tokens each, on a denser ring. [How the game is played](b1-the-game-and-its-flows.md#how-the-game-is-played-what-agents-stake-and-who-wins-the-nodes) shows what the
denser ring changes in the first game.

From the settings and the seed: *T*/100 founders of 100 tokens, each with a
random brain, on a Watts–Strogatz ring whose connections are rewired with
probability 0.2. Every later choice comes from one random stream, so a run
depends on its settings and its seed alone
([The starting ring](rules-in-detail.md#step-by-step),
[Seeds and the random stream](rules-in-detail.md#seeds-and-the-random-stream);
[Is a run reproducible?](is-a-run-reproducible.md) tests it). Here is the real
starting network of one baseline run:

<!-- figure world/ring -->
![The 100 founders of the world with seed 1](../figures/world/ring.svg)

**The 100 founders of the world with seed 1.** The starting graph of `B1-10000-s001`: 100 founders placed around a circle in the order of the ring, each joined to its two nearest neighbours on either side; 37 of the 200 connections were moved to a founder chosen at random, and cross the circle.

> [!example]- How to make this figure
> **Runs.** `B1-10000-s001`.
>
> **Data.** From its settings.
>
> 1. Build the graph exactly as the engine does: `networkx.watts_strogatz_graph(n=100, k=5, p=0.2, seed=1)` — with k = 5 networkx joins every node to ⌊5/2⌋ = 2 neighbours on either side.
> 2. Place node i at angle 2πi / 100 on a circle and draw every connection.
>
> **To make it again:** `python3 book_figures.py world`.
<!-- /figure -->

Rewired, 37 of its 200 connections are shortcuts: the clustering falls from
0.5 to 0.27, and the average distance from 12.9 steps to 4.4.

## What is recorded

A run is kept like a film: a still picture of the whole world after every
phase. Each picture is a **frame**: frame 2*t* after the reproduction phase
of iteration *t*, and frame 2*t* + 1 after its game. Iteration 0, for
example, gives frames 0 and 1. A frame holds every living agent's id, tokens,
genotype, parent genotype and age; every connection; and every decision taken
in the phase.

From each frame, a row of about ninety **statistics** is computed: the number
of agents, of connections, the inequality of wealth, and so on. The rows are
kept in the run's file `stats.jsonl`. [What a run records](runs-and-records.md#what-a-run-records)
lists the fields, and links each statistic to the page that defines it.

A baseline run of 10,000 tokens and 3,000 iterations writes 6,000 frames,
about 300 megabytes on disk. One of 100,000 tokens and 1,500 iterations
writes 3,000 frames, about 2 gigabytes.

## Where to go next

- [One iteration, step by step](one-iteration.md): how a world changes, in
  the exact order the program runs.
- [The brain](the-brain.md): how an agent decides.
- [The starting ring](rules-in-detail.md#the-starting-ring): the founders' ring, built step by
  step.
- [Every setting](settings.md): every number and switch of a run.

<!-- turns -->
---

[How the simulation works](../README.md#how-the-simulation-works) · [One iteration, step by step](one-iteration.md) →

> [!info]- Pages that link here (11)
> [What this notebook is about](what-this-notebook-is-about.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [Every setting](settings.md) · [Glossary](glossary.md) · [The rules in detail](rules-in-detail.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) · [How runs are made, recorded and costed](runs-and-records.md) · [B1: lives and wealth](b1-lives-and-wealth.md) · [B1: the game and its flows](b1-the-game-and-its-flows.md)
<!-- /turns -->
