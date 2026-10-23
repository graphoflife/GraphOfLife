# Path length

How far apart are the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") of a world? The **distance** between two
agents is the smallest number of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") on a path from one to the
other. The **mean path length** is the average distance between agents.

## Breadth-first search

The distances from one agent *s* to every other are found by a
**breadth-first search**: *s* is at distance 0; its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") at distance 1;
their neighbours not yet reached at distance 2; and so on, until every agent
is reached. It takes time proportional to the number of agents plus the
number of connections.

## The estimate the notebook uses

The exact mean over every pair of agents needs a search from every agent:
too slow for worlds of thousands of agents, every 25 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). So
`meanPathLength` is estimated from a spread of sources:

1. Let *n* be the number of agents, listed in increasing order of id, and
   |*E*| the number of connections. The number of searches wanted is
   *S* = max(8, min(16, round(250,000 / (*n* + 2|*E*|)))) — between 8 and
   16, fewer for bigger worlds.
2. Let *step* = max(1, ⌊*n* / *S*⌋). A search is run from the agents at
   positions 0, *step*, 2·*step*, … of the list.
3. The mean path length is the average of the distances from these sources
   to every other agent (each source's distance to itself left out).

After the [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") the world is one connected piece, so every agent can be
reached from every source.

## What the numbers mean

Averaged over the [settled life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured.") of each [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world (iterations 500 to
2,999), the mean path length is 9.8 in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world and between 6.5 and
13.0 across the 26 worlds that lived to the end. A random network with the
same number of agents *N* and mean [degree](degree.md "The number of an agent's connections: how many neighbours it has.") *k̄* would have, by the usual rough
estimate ln *N* / ln *k̄*, about 6.0
([What shape does the network take?](what-shape-does-the-network-take.md)). The
founders' ring of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 started at 4.4 ([The starting ring](starting-ring.md)).

Since an agent can only stake on its neighbours, a [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") needs about ten
[games](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") to cross a typical distance of the world.

How distances grow with the size of a world says what kind of network it
is: in a lattice they grow like a power of the number of agents, in a
"small world" like its logarithm.

<!-- turns -->
---

← [Clustering](clustering.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Core, trees and leaves](core-trees-leaves.md) →

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (11)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [The starting ring](starting-ring.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [The geometry of a world](the-geometry-of-a-world.md) · [Radius and diameter](radius-and-diameter.md) · [Physical inspiration](physical-inspiration.md) · [The random twin: is a world a space?](random-twin.md) · [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
