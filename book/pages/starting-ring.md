# The starting ring

> [!question] Questions of this page
> - What does a world look like before its first [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")?
> - How exactly is that first network built from the settings and the seed?
> - Why a ring with a few shortcuts — a "small world"?

## In short

Every world starts as a **ring of founders with a few shortcuts**, the
"small-world" network of Watts and Strogatz (1998). The founders, the agents a
world begins with, sit around a circle, each joined to its nearest founders on
either side. Then a few connections are moved to founders chosen at random. In
the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), 100 founders of 100 tokens each start with 4 neighbours, and
about 40 of the 200 connections become shortcuts. The ring is clustered like a
neighbourhood, yet every founder is only a few steps from every other.

## The picture

Picture houses standing round a lake, where every household knows the two
households on either side. The people you know mostly know each other too.
But news passed on by word of mouth needs many steps to reach the far shore.
Now swap a few of these acquaintances, chosen at random, for someone across
the lake. The village stays close-knit, yet news now reaches anyone in a few
steps. Such a network is called a **small world**.

![The founders' ring](../diagrams/ring.svg)

**The founders' ring, in a small example.** Left: a ring of 16 founders, each
joined to the 2 nearest on either side. Right: the same ring after each
connection was moved with probability 0.2. The 6 connections drawn in yellow
were moved to a founder chosen at random: they are the shortcuts.

## Step by step

Let *T* be the number of tokens ([`total_tokens`](settings.md#total_tokens)).

1. **The founders.** There are *n* = ⌊*T*/100⌋ of them
   ([`n_nodes`](settings.md#n_nodes)): one for every hundred tokens, rounded
   down. They are numbered 0 to *n* − 1, and these numbers are also their
   agent ids. Picture them in order around a circle.
2. **The ring.** Let *k* = max(⌊*n*/100⌋, 5)
   ([`k_neighbors`](settings.md#k_neighbors)). Every founder *i* is joined to
   the founders *i* ± 1, *i* ± 2, …, *i* ± ⌊*k*/2⌋. The count goes round the
   circle, so founder 0 is next to founder *n* − 1. That makes *n*·⌊*k*/2⌋
   connections, and every founder has 2⌊*k*/2⌋ of them.
3. **The shortcuts.** Go once round the circle through the connections
   {*i*, *i* + 1}, for *i* = 0, 1, …, *n* − 1. Then go round once more
   through the connections {*i*, *i* + 2}, and so on up to
   {*i*, *i* + ⌊*k*/2⌋}. On the way, each connection {*i*, *i* + *j*} is
   **rewired** with probability *p* = 0.2
   ([`rewire_p`](settings.md#rewire_p)). Its far end *i* + *j* is replaced by
   a founder chosen uniformly at random. The new end is neither *i* itself nor
   a founder already joined to *i*. Rewiring moves connections; it never adds
   or removes one.
4. **Tokens.** Every founder gets ⌊*T*/*n*⌋ tokens. If *n* does not divide
   *T*, then *T* − *n*⌊*T*/*n*⌋ tokens are left over. They are dealt out by
   one [multinomial draw](random-numbers.md#how-a-draw-becomes-a-decision), in
   which every founder has the same chance at every token.
5. **Brains.** Every founder gets a brain whose weights are drawn at random
   ([The brain](the-brain.md)). Each is a [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") of its own, numbered 1 to
   *n*.

Steps 1–3 are done by the networkx function
`watts_strogatz_graph(n, k, p, seed)`. It draws from a random generator of its
own, started from the run's seed. Steps 4–5 draw from the world's random
stream ([Seeds and the random stream](random-numbers.md)).

The ring needs more founders than *k*. With *k* = 5 that means at least six
founders. So with one founder per hundred tokens, a world needs at least 600
tokens.

## The baseline's ring

With *T* = 10,000: *n* = 100 founders of 100 tokens each, and *k* = 5. So
every founder is joined to 2 founders on each side — [degree](degree.md "The number of an agent's connections: how many neighbours it has.") 4 — and the ring
has 100 · 2 = 200 connections. On average 0.2 · 200 = 40 of them are
rewired.

## Why a small world?

A plain ring is **clustered**: two neighbours of a founder are often
neighbours of each other. With 2 neighbours on each side, a founder's 4
neighbours make 6 pairs, and 3 of these pairs are joined. So the clustering is
3/6 = 0.5 ([Clustering](clustering.md)). Had the same 200 connections joined
founders at random, the clustering would be near 0.04. Two neighbours of a
founder would then be joined with a chance of only 4 in 99.

But a plain ring is **long**. To get from one founder to the one opposite
takes about *n*/(2·2) = 25 steps. Halfway round the circle is *n*/2 = 50
founders, and each step covers at most 2. The average distance between two
founders is about 12.9 steps ([Path length](path-length.md)).

A few random shortcuts change that. Every shortcut joins two far-apart parts
of the circle, and a handful are enough to make every distance short. In the
real ring of the run with seed 1, 37 of the 200 connections were rewired. The
average distance fell to 4.4 steps, about a third of what it was. The
clustering fell to 0.27: about half was lost, but it is still far above the
0.04 of a random network. Clustered and short at once: that is a small world.
[The world: tokens, agents and a network](the-world.md) draws this ring.

## What becomes of the ring

In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") cuts the connections that carried no tokens in it, so the
ring is forgotten within a few iterations. It matters mainly as a fair,
connected start: every founder can reach every other.
[The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations) follows what
the founders make of it.

## Where to go next

- [The world: tokens, agents and a network](the-world.md): the world this
  ring starts, with the real ring of seed 1 drawn.
- [Seeds and the random stream](random-numbers.md): where the ring's chance
  comes from.
- [Clustering](clustering.md) and [Path length](path-length.md): the two
  measures a small world is judged by.
- [The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations): how the
  founders become a world.

<!-- turns -->
---

← [Every setting](settings.md) · [How the simulation works](../README.md#how-the-simulation-works) · [Seeds and the random stream](random-numbers.md) →

> [!info]- Pages that link here (12)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [Every setting](settings.md) · [Seeds and the random stream](random-numbers.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [The life of a B1 world](a-worlds-life.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [Clustering](clustering.md) · [Path length](path-length.md) · [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
