# Strategies written by hand

> [!question] Questions of this chapter
> - Is the play evolution found the best there is, or only the best it could
>   reach?
> - Can a strategy written by hand, for one agent or for a team, beat evolved
>   brains in their own world?
> - What would a team need in order to grow?

## Why ask

[Chapter 50](50-why-evolution-stops.md) read the end of evolution as the end of
the game: a Colonel Blotto contest has a best way to play, to be
unpredictable, and once evolution has found it there is nothing better left
to find. That reading makes a prediction that can be tested without waiting
for evolution at all. If the evolved play is the answer, then no strategy
anyone could write should beat it by much. If one does, evolution did not stop
at the top. It stopped where it could no longer climb, and the game still had
more to give.

There is a second reason to write strategies by hand. Evolution finds what is
one small step from what it has. A strategy that needs many agents to act
together — some to hold a frontier, others to feed it, all knowing which is
which — may be far better and still never be found, because no single
mutation makes it. The author of the simulation suggested exactly such a
strategy: a *factory*, a structured group in which every agent has its job in
the pipeline of token flow, extracting tokens from its surroundings like a
colony of parasites. If a factory written by hand can grow in an evolved
world, the game rewards organisation, and the question becomes how evolution
could get there.

## The invasion test

Take an evolved world: one of the B2 worlds of
[Experiment 16](50-why-evolution-stops.md#the-runs), 10,000 tokens, as it
stood at iteration 2,000. Choose a **patch**: an agent at random and the
agents nearest to it, step by step, until the patch holds 5% of the world.
Give every agent of the patch a strategy instead of its brain, and play on
under the world's own rules. Follow the share of the world's agents that carry
the strategy. A strategy spreads as a brain does: into a child, and into a
node its owner takes.

A strategy's share means nothing on its own. A patch might grow or shrink
whatever it does, because of where it was placed. So every patch is also
played as a **control**. The same agents are given a copy of one resident's
brain, the brain of an agent of the patch drawn at random. The copy is marked
so it can be counted, and it never mutates. A strategy beats the residents if
it holds more of the world than the resident patch played from the same place
on the same seed.

A strategy reads exactly what a brain reads and writes exactly what a brain
writes (`research/oee/pilots/manual.py`). It sees each candidate's tokens, its
connections and its messages. Its decisions go through the same output rows
the engine turns into a child, its links, the stakes and the messages. It can
do nothing an evolved brain could not do.

## Three strategies

- **The turtle** keeps everything at home. When it has 30 tokens or more it
  has a child with a third of them.
- **The raider** looks for the poorest neighbour that is not a teammate.
  Under B2 it can stake at most 70% of its tokens, since 30% stay home. If
  that is more than one and a half times the neighbour's tokens, it is enough
  to take the node even if the neighbour kept everything at home, and the
  raider stakes it all there. Otherwise it keeps everything at home. When it
  has 30 tokens or more it has a child with a third of them, joined to every
  teammate next door.
- **The factory** is a team with a pipeline. Every member works out how far
  it is from the team's **frontier**. Its distance is 0 if a stranger is next
  door, and otherwise one more than the smallest distance a teammate reported.
  It writes the distance into its messages. Members on the frontier attack as
  the raider does: everything they can stake goes on the poorest stranger next
  door if that takes it, and otherwise they hold. Members inside stake
  everything they can on the teammate nearest the frontier. That pumps their
  tokens downhill, towards where the team meets the rest of the world. Inside
  members breed when rich, and join their children to their teammates.

Teammates know each other by what they write. The first four numbers of every
message a strategy sends are a fixed signature, and the fifth carries its
distance. This is a password, not a privilege: an evolved brain could write
the same numbers.

<!-- turns -->
---

← [Chapter 50 · Why evolution stops](50-why-evolution-stops.md) · [Contents](../README.md) · [Chapter 52 · How fast should brains change?](52-how-fast-should-brains-change.md) →
<!-- /turns -->
