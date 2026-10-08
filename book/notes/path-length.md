# Path length

How far apart are the agents of a world? The **distance** between two
agents is the smallest number of connections on a path from one to the
other. The **mean path length** is the average distance between agents.

## Breadth-first search

The distances from one agent *s* to every other are found by a
**breadth-first search**: *s* is at distance 0; its neighbours at distance 1;
their neighbours not yet reached at distance 2; and so on, until every agent
is reached. It takes time proportional to the number of agents plus the
number of connections.

## The estimate the book uses

The exact mean over every pair of agents needs a search from every agent:
too slow for worlds of thousands of agents, every 25 iterations. So
`meanPathLength` is estimated from a spread of sources:

1. Let *n* be the number of agents, listed in increasing order of id, and
   |*E*| the number of connections. The number of searches wanted is
   *S* = max(8, min(16, round(250,000 / (*n* + 2|*E*|)))) — between 8 and
   16, fewer for bigger worlds.
2. Let *step* = max(1, ⌊*n* / *S*⌋). A search is run from the agents at
   positions 0, *step*, 2·*step*, … of the list.
3. The mean path length is the average of the distances from these sources
   to every other agent (each source's distance to itself left out).

After the cleanup the world is one connected piece, so every agent can be
reached from every source.

## What the numbers mean

Averaged over the settled life of each baseline world (iterations 500 to
2,999), the mean path length is 9.8 in the median world and between 6.5 and
13.0 across the 26 worlds that lived to the end. A random network with the
same number of agents *N* and mean degree *k̄* would have, by the usual rough
estimate ln *N* / ln *k̄*, about 6.0
([Chapter 15](../chapters/15-what-shape-does-the-network-take.md)). The
founders' ring of seed 1 started at 4.4 ([The starting ring](starting-ring.md)).

Since an agent can only stake on its neighbours, a token needs about ten
games to cross a typical distance of the world.

How distances grow with the size of a world says what kind of network it
is: in a lattice they grow like a power of the number of agents, in a
"small world" like its logarithm.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
