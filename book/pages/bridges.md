# Bridges, cut risk, loops and the core: how a network can be cut

## In short

A **bridge** is a [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") on no loop: cut it, and the network falls into
two pieces. In a settled [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), a quarter of the
connections are bridges. The **cut risk** is the largest share of the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")
that one bridge holds to the rest: a ninth, in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world. The **cycle
rank** counts the independent **loops**: how many connections, the right
ones, a network could lose and still be one piece. Peel away every agent with
a single connection, again and again, and the **core** is left. Every loop
lies in it, and **trees** of agents hang off it.

## Bridges

Think of a network as towns joined by roads. Most roads have a detour: close
one, and traffic goes round. A road with no detour is a **bridge**. Close it,
and the towns on its far side are cut off. Bridges are where a world is
fragile.

### What a bridge is

Take a network in one piece. A connection {*u*, *v*} is a **bridge** if, once
it is removed, *u* and *v* can no longer reach each other. Put another way:
it lies on no [loop](#loops). A loop through it would be a second way from *u*
to *v*.

Three rules follow. Every connection of a tree is a bridge, and so is every
leaf's single connection. In the 2-core, every agent lies on a loop or on a
path between loops ([Core, trees and leaves](#core-trees-and-leaves)). There,
only the connections on a path that joins two loops are bridges.

**Example.** This network of 9 agents and 9 connections is the example of
the whole page:

```
  a — b — c — d
  |       |
  e ——————f — g — h
              |
              i
```

The connections a–b, b–c, c–f, f–e and e–a form a loop, so none of them is a
bridge. The other four are bridges. Three of them, c–d, g–h and g–i, each hold
a single leaf. The fourth, f–g, holds g, h and i.

### How bridges are found

Removing each connection in turn and checking what is still joined would take
a walk through the whole network for every connection. **Tarjan's algorithm**
(1974) finds every bridge in a single depth-first walk.

The walk numbers every agent in the order it first reaches it: disc(*u*). It
also gives each agent a "low-link", low(*u*). Look at *u* and every agent
below it in the walk's tree. The low-link of *u* is the smallest disc number
among them, and among the agents they reach by one connection outside the
walk's tree. The connection from *u* down to its child *v* in the walk is a
bridge exactly when

$$
\text{low}(v) > \text{disc}(u),
$$

that is, when nothing below *v* reaches back to *u* or above it. The walk
takes time proportional to the number of agents plus the number of
connections.

In the example, let the walk start at a. It goes to b, c and d, back to c, on
to f and e, back to f, and last to g, h and i:

| agent | a | b | c | d | f | e | g | h | i |
|---|---|---|---|---|---|---|---|---|---|
| disc | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| low | 0 | 0 | 0 | 3 | 0 | 0 | 6 | 7 | 8 |

From e the walk sees a, numbered 0, so e, f, c and b all get the low-link 0.
Below g, nothing reaches back above g: low(g) = 6 is more than disc(f) = 4, so
f–g is a bridge. By the same test, so are c–d, g–h and g–i, and no other
connection.

### Bridges in the worlds

A run records `bridges`, the number of bridges, in its statistics file, on the
rows of every 25th iteration. The notebook divides it by the number of
connections, `bridges/edges`: the share of all connections that are bridges.
Count them after the game, over a B1 world's [settled life](median-and-quantiles.md#the-settled-life-of-a-world "A world from iteration 500 to its end, after its youth, where it is measured."). At 100,000 tokens
about a quarter of the connections are bridges: 26%, some 6,400 of them. At
10,000 tokens it is 31%
([Core, leaves and bridges](what-shape-does-the-network-take.md#core-leaves-and-bridges)).
Most bridges hold a single leaf.

Experiment 5 predicted that more than a fifth of all connections would be
bridges, and in every world they were
([Was the thesis right?](what-shape-does-the-network-take.md#was-the-thesis-right)).
The viewer can colour every connection by whether it is a bridge
([What the viewer can colour by](one-world-many-colours.md#what-the-viewer-can-colour-by)).

A count of bridges does not say how much hangs on each. For every bridge, the
program also works out how many agents lie behind it. The largest share of
the world a single cut can sever is the [cut risk](#cut-risk).

## Cut risk

Counting bridges says how many connections lie on no loop. It does not say how
badly the network would break if one were cut. A bridge to a single leaf costs
one agent; a bridge between two halves costs half the world. The **cut risk**
measures the worst case.

### What the cut risk is

Removing a bridge splits the network into two pieces. Call the smaller one the
part **behind** the bridge. The cut risk is the largest such part, over all
bridges, as a share of all agents:

$$
\texttt{cutRisk} = \max_{\text{bridges } e} \frac{\min\bigl(|A_e|, |B_e|\bigr)}{|V|} ,
$$

where *A*ₑ and *B*ₑ are the two pieces left by removing *e*, and |*V*| is the
number of agents. In words: for each bridge, take its smaller side and divide
by all agents; the cut risk is the largest of these. It is 0 when there is no
bridge, and at most ½ — a network split down the middle by one connection.

**Example.** In the network drawn under [Bridges](#bridges), c–d, g–h and g–i
each hold one agent of 9. The bridge f–g holds three: g, h and i. So the cut
risk is 3/9, a third.

The formula assumes a network in one piece, as a world is after every cleanup.
Read before the cleanup, the network can be in several pieces. The program
then takes *A*ₑ to be the agents the walk reached through the bridge. *B*ₑ is
every other agent of the world, in the same piece or not.

### When the cut risk is read

A run records the cut risk at two moments of a phase:

| field | read on | when | on which rows |
|---|---|---|---|
| `cutRisk` | the [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."): the world as the run records it | at the end of the phase, after its cleanup, when only the largest piece is left | every 25th iteration |
| `cutRiskBefore` | the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s network, as it runs | just before the cleanup, when nobody has been removed yet | every row |

In a game's row, `cutRiskBefore` is read after the game has been played and
after its unused connections have been cut. In B1, every connection that
carried no token in the game is cut at that point
([`prune_after`](settings.md#prune_after)). Only after the reading does the
cleanup remove anybody. In a reproduction row, it is read after the births,
just before that phase's cleanup.

Why two readings? A frame shows the world after the cull: the cleanup's
removal of every piece but the largest. A cut that removes a whole side of a
bridge changes `cutRisk` and the death toll in the same row. Which was cause
and which effect, the row cannot say. The reading `cutRiskBefore` is taken on
a network the cull has not touched yet. So it can be read as a predictor of
the cull rather than a consequence of it.

It is not read before the game, though. The game's unused connections are cut
first, so it cannot foresee those cuts. A region whose every connection to the
rest went unused is already a piece of its own by then. Measured as above, the
piece can count as if one connection held it. For the network the game starts
from, read the reproduction row of the same iteration. Its `cutRiskBefore` is
read after the births, and its `cutRisk` on that very network.
[How a world breaks](how-a-world-breaks.md#does-a-fragile-world-lose-more)
uses the game's reading. In the worlds of 10,000 tokens it checked it against
a reading at the start of the game, and the picture is the same.

### Why the cut risk matters

The cleanup keeps only the largest piece of the network. Take a bridge with a
tenth of the world behind it. In B1, that tenth of the world dies the moment
no token crosses the bridge in a game. How often does that happen? In the
median of the thirty B1 worlds of 100,000 tokens (Experiment 10), the cut risk
is 11% (12% at 10,000 tokens). Yet over 30,000 settled games, its [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.")
with the share of agents a game cut off is only 0.09
([How a world breaks](how-a-world-breaks.md)). A bridge with much behind it is
busy, and a busy connection is almost never left unused.

The [health bar](health-bar.md) asks that no single link's loss cut off more
than 1% of a world. It asks the same of agents: no agent's loss may cut off
more than 2%. [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") never lets a bridge lapse. There, an unused connection is cut
only where its two ends share at least two [neighbours](the-world.md#the-network "An agent joined to another by a connection.")
([`prune_needs`](settings.md#prune_needs)). The two ends of a bridge share
none, since a shared neighbour would put the bridge on a triangle
([A world that cannot break](a-world-that-cannot-break.md)).

### How the cut risk is computed

Tarjan's walk finds every bridge
([How bridges are found](#how-bridges-are-found)), and the same walk counts
the agents below each one. So the size of the part behind every bridge costs
nothing extra. In the example, the walk counts 1 agent below c–d, g–h and
g–i, and 3 below f–g. The code is `bridge_splits` and `worst_cut_share` in
`GraphOfLifeSimple.py`. `gol_series.py` applies them to every frame of a 25th
iteration, for `cutRisk`. The engine applies them at the start of
`_cleanup_and_redistribute`, for `cutRiskBefore`.

## Loops

A **loop** (or cycle) is a path through the network that comes back to where
it started without using a connection twice. Loops are what make a network
robust: along a loop there are always two ways between any two of its agents.
Counting them needs care, because the number of different loops in a network
can be astronomically large.

### Independent loops: the cycle rank

Take a **spanning tree**: a set of connections that reaches every agent and
contains no loop. A connected network of *n* agents has a spanning tree of
exactly *n* − 1 connections. Every connection **not** in the tree closes
exactly one loop with the tree: its **fundamental cycle**. These loops
generate all the others: any loop is a combination of them. (Two loops
combine into the connections that only one of them uses.) So the number of
independent loops is

$$
\text{cycle rank} = |E| - |V| + c .
$$

In words: the number of connections, minus the number of agents, plus the
number of connected pieces, *c*. In every recorded frame, after the cleanup,
*c* is 1. This is `cycleRank` in the viewer.

**Example.** A square of four agents with one diagonal: |*E*| = 5,
|*V*| = 4, *c* = 1, so 2 independent loops — the two triangles. The square
itself is their combination: the two triangles share the diagonal, and
without it they make the square. The network drawn under [Bridges](#bridges)
has 9 connections, 9 agents and one piece, so 1 loop: a–b–c–f–e–a.

Take the world of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 of Experiment 10, after its last game. There, 26,166
connections join 14,870 agents in one piece: 26,166 − 14,870 + 1 = 11,297
independent loops.

### Loop density

`loopDensity` = `cycleRank` / |*E*|: the share of connections that are
"spare", beyond what a tree would need. A tree has 0; a densely interwoven
network has values near 1. The example network has 1/9, about 0.11, and the
world of seed 1 has 11,297 / 26,166, about 0.43. A settled B1 world has about
0.42 at 100,000 tokens, and 0.38 at 10,000 tokens. The median world of
100,000 tokens has about 10,000 independent loops
([Where the loops run](what-shape-does-the-network-take.md#where-the-loops-run)).

Every loop lies in the core. A settled B1 world has 0.82 independent loops per
agent, and 1.34 per agent of its core. Both are counted at the start of a game
([The loops live in the core](what-shape-does-the-network-take.md#the-loops-live-in-the-core)).

### Pieces

`components` is the number of separate pieces. The cleanup keeps only the
largest, so after every phase it is 1 — anything else would be a bug.

### Loops through one agent

The viewer can colour every agent (and every connection) by how many loops
pass through it. It counts the fundamental cycles of a **breadth-first**
spanning tree. For every connection outside the tree, it walks both ends up
the tree until they meet. Every agent on the way, and the meeting point, lies
on that loop. Breadth-first trees keep the loops short and local. A different
tree would give a different basis, so this is a fair sample of where the
loops run rather than a canonical count.

An agent on no loop at all hangs on a tree, or lies in the core on a path
between loops ([Core, trees and leaves](#core-trees-and-leaves)). In the world
of seed 1, after its last game, 37% of the agents lie on none of its 11,297
loops. Only 36% are off its core, so 114 agents lie in the core, on paths
between loops. One agent, a [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."), lies on 1,012 of the loops: about a tenth of
them.

## Core, trees and leaves

Every connected network can be cut into two kinds of part. There is a
**core**, where every agent lies on a loop or on a path between loops. And
there are **trees** that hang off it. This section says how the cut is made,
and why it matters for Graph of Life.

### Peeling

Take the network and repeat:

> remove every agent that has one connection or none,

until no such agent is left. Removing a leaf can make its neighbour a leaf,
so the peeling goes on until it stops by itself. What is left is the
**2-core**: the largest part of the network in which every agent has at least
two connections *within that part*. It can be empty: a tree peels away
completely.

Every agent then has one of three **roles**:

- **core**: it is in the 2-core;
- **leaf**: it has exactly one connection;
- **tree**: neither. It was peeled away, but it has two or more connections.
  It lies on a tree hanging off the core, between the core and the leaves.

A run records `coreShare`, the share of agents in the 2-core, every 25
iterations. It also records `leaves`, the number of agents with exactly one
connection, after every phase.

### An example of peeling

Take the network of [What a bridge is](#what-a-bridge-is) again:

```
  a — b — c — d
  |       |
  e ——————f — g — h
              |
              i
```

The loop a–b–c–f–e–a is the core, with its members' connections among
themselves. Peeling removes d, h and i, which have one connection each, in
the first round. Then it removes g, which has only f left. So the core is
{a, b, c, e, f}, the leaves are {d, h, i}, and the tree is {g}.

### Why the core matters

In B1's game, every connection that carries no tokens is cut
([One iteration, step by step](one-iteration.md)). Cutting a connection on a
loop never disconnects anything: there is always another way round the loop.
Every other connection is a [bridge](#bridges). A bridge of a tree cuts off
everything beyond it. A bridge on a path between two loops of the core cuts
the core itself in two. The cleanup then removes whatever is no longer joined
to the largest piece. So a single cut can lose any tree whole, and, through a
bridge of the core, part of the core too. B2 cuts an unused connection only
where its two ends share two neighbours, so it never cuts a bridge
([`prune_needs`](settings.md#prune_needs)).

In a settled B1 world of 100,000 tokens, about 57% of the agents are in the
core, and 35% are leaves. The rest lie in trees. At 10,000 tokens it is 54%
and 36%
([The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world)).

### How far a tree hangs

Every agent off the core belongs to exactly one tree. And every tree joins
the core at exactly one core agent. Two ways into the core would close a
loop, and the agent would then be in the core. So each agent off the core has a
**depth**, its number of steps to the core. It also has a **root**, the core
agent its tree hangs from.

In the example, d hangs from c at depth 1. g hangs from f at depth 1, and h
and i at depth 2. The tree hanging from f holds three agents (g, h and i),
and the one hanging from c holds one (d). A breadth-first search out of every
core agent at once finds both for every agent.

### Deeper cores: the onion

The peeling can go further. Remove every agent with fewer than **three**
connections, again and again, and what is left is the **3-core**. With fewer
than *k*, it is the ***k*-core**. Each *k*-core lies inside the one before,
like the layers of an onion. An agent's **core number** is the largest *k*
whose *k*-core still holds it (Seidman 1983). An agent on a tree or a leaf
has core number 1. An agent of the 2-core that the 3-core no longer holds has
core number 2, and so on.

In the example, every core agent has two connections inside the core, so the
3-core is empty. So a, b, c, e and f have core number 2, and d, g, h and i
core number 1. A network in which many agents have high core numbers is held
together by many overlapping loops, not by single rings.

A run records only the share of agents in the 2-core (`coreShare`). The
deeper cores, the trees and the [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") of the core are measured in
[The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world).

## Where to go next

- [How a world breaks](how-a-world-breaks.md): whether the worst bridge gives
  way, and what severs a region instead.
- [The shape of the network](what-shape-does-the-network-take.md): the core,
  the bridges and the loops of the thirty worlds.
- [Distances in a network](path-length.md#the-spectral-gap): the spectral
  gap, which weighs every way of cutting a network in two at once.
- [The health bar](health-bar.md): the two cuts a world must keep small to
  pass it.

## How our view changed

- **24 October 2026.** When the notes on bridges, cut risk and loops became
  one page, their statements were checked against the engine:
  - The note on the cut risk said that `cutRiskBefore` is measured "before
    the phase's cleanup removed anybody". So, it said, it can be read as a
    predictor of the cull rather than a consequence of it. In a game's row,
    though, it is read after the game's unused connections are cut, as
    [How a world breaks](how-a-world-breaks.md#does-a-fragile-world-lose-more)
    now says too. It cannot foresee those cuts, and a region they severed is
    already a piece of its own when it is read.
  - The note on loops said that an agent on no loop at all hangs on a tree.
    It may also lie in the core, on a path between loops. In the world of
    seed 1 after its last game, 114 agents do.

## References

- Seidman, S. B. (1983). Network structure and minimum degree. *Social
  Networks* 5, 269–287.
- Tarjan, R. E. (1974). A note on finding the bridges of a graph.
  *Information Processing Letters* 2, 160–161.

<!-- turns -->
---

← [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network)

> [!info]- Pages that link here (18)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [Measuring the flow: token flow, lightning and circulation](token-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [How a world breaks](how-a-world-breaks.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Stars and hubs](stars-and-hubs.md) · [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [The health bar](health-bar.md) · [One world, many colours: what the viewer can colour by](one-world-many-colours.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
