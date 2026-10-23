# The invasion test

A cross-time tournament ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md))
asks whether a world's later brains beat its earlier ones. The **invasion
test** asks something else: is there play *better* than what a world has
evolved? It plants a strategy in the evolved world and watches whether it
spreads.

## How it is played

1. Take an evolved world as its archive keeps it, its graph, its tokens and
   its brains (`GraphOfLife.from_checkpoint`).
2. Choose a **patch**: an agent drawn at random, then the agents nearest to
   it, step by step through the network, until the patch holds the share of
   the world asked for (5% in [Strategies written by hand](strategies-written-by-hand.md)),
   or a single agent.
3. Give every agent of the patch the strategy in place of its brain. A
   strategy is a function of exactly the inputs a brain reads, and it writes
   exactly the outputs a brain writes, so it can do nothing an evolved brain
   could not (`research/oee/pilots/manual.py`). It spreads as a brain does,
   into children and into nodes its agents take, and it never mutates.
4. Play on under the world's own rules and count, after every iteration, the
   agents that carry the strategy, as a share of all agents.

## The control

A patch can grow or shrink because of where it sits: next to rich agents or
poor ones, in the middle of the world or at an edge. So the share a strategy
reaches means little on its own. Every patch is also played as a **control**.
The same agents get a frozen copy of one resident brain, that of an agent of
the patch drawn at random, marked so it can be counted. A strategy beats the
residents if it ends with more of the world than this control, planted in the
same place on the same seed.

## Reading the result

- **Growth from a patch** says that the strategy beats the residents *when it
  has a crowd*: better play exists than the play evolved.
- **Growth from a single agent** says more: the strategy could be found by
  one mutation and spread. A strategy that grows from a patch but not from one
  agent has an **Allee effect**, and evolution, which tries one agent at a
  time, can miss it although it is better.
- A strategy's parts can be tested one at a time by dropping them: the raider
  without its password ([Strategies written by hand](strategies-written-by-hand.md))
  shows which part wins.

A strategy that dies out says only that *this* strategy is worse, never that
nothing better exists: an invasion test can show that a game is not solved,
but not that it is.

<!-- turns -->
---

← [Hand-written strategies in B1](hand-written-strategies-in-b1.md) · [Strategies written by hand](../README.md#strategies-written-by-hand)
<!-- /turns -->
