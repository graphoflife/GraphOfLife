# The invasion test

A [cross-time tournament](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md))
asks whether a world's later [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") beat its earlier ones. The **invasion
test** asks something else: is there play *better* than what a world has
evolved? It plants a strategy in the evolved world and watches whether it
spreads.

## How it is played

1. Take an evolved world as its archive keeps it, its graph, its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") and
   its brains (`GraphOfLife.from_checkpoint`).
2. Choose a **patch**: an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") drawn at random, then the agents nearest to
   it, step by step through the network, until the patch holds the share of
   the world asked for (5% in [Strategies written by hand](strategies-written-by-hand.md)),
   or a single agent.
3. Give every agent of the patch the strategy in place of its brain. A
   strategy is a function of exactly the inputs a brain reads, and it writes
   exactly the outputs a brain writes, so it can do nothing an evolved brain
   could not (`research/oee/pilots/manual.py`). It spreads as a brain does,
   into children and into nodes its agents take, and it never mutates.
4. Play on under the world's own rules and count, after every [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), the
   agents that carry the strategy, as a share of all agents.

## The control

A patch can grow or shrink because of where it sits: next to rich agents or
poor ones, in the middle of the world or at an edge. So the share a strategy
reaches means little on its own. Every patch is also played as a **control**.
The same agents get a frozen copy of one resident brain, that of an agent of
the patch drawn at random, marked so it can be counted. A strategy beats the
residents if it ends with more of the world than this control, planted in the
same place on the same [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.").

## Reading the result

- **Growth from a patch** says that the strategy beats the residents *when it
  has a crowd*: better play exists than the play evolved.
- **Growth from a single agent** says more: the strategy could be found by
  one [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") and spread. A strategy that grows from a patch but not from one
  agent has an **Allee effect**, and evolution, which tries one agent at a
  time, can miss it although it is better.
- A strategy's parts can be tested one at a time by dropping them: the raider
  without its password ([Strategies written by hand](strategies-written-by-hand.md))
  shows which part wins.

A strategy that dies out says only that *this* strategy is worse, never that
nothing better exists: an invasion test can show that a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") is not solved,
but not that it is.

<!-- turns -->
---

← [Why the raiders stopped winning](does-evolution-close-the-door.md) · [Strategies written by hand](../README.md#strategies-written-by-hand)

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (8)
> [Glossary](glossary.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [Strategies written by hand](strategies-written-by-hand.md) · [What a brain can compute](what-a-brain-can-compute.md) · [How large a crowd a raider needs](how-large-a-crowd-a-raider-needs.md) · [Why the raiders stopped winning](does-evolution-close-the-door.md) · [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
