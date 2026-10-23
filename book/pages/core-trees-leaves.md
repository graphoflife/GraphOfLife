# Core, trees and leaves

Every connected network can be cut into two kinds of part: a **core**, where
every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") lies on a [loop](loops.md "A path through the network that comes back to where it started without using a connection twice.") or on a path between loops, and **trees** that hang
off it. This note says how the cut is made and why it matters for Graph of
Life.

## Peeling

Take the network and repeat:

> remove every agent that has one [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") or none,

until no such agent is left. Removing a leaf can make its [neighbour](the-world.md#the-network "An agent joined to another by a connection.") a leaf,
so the peeling goes on until it stops by itself. What is left is the
**2-core**: the largest part of the network in which every agent has at least
two connections *within that part*. It can be empty (a tree peels away
completely).

Every agent then has one of three **roles**:

- **core** — it is in the 2-core;
- **leaf** — it has exactly one connection;
- **tree** — neither: it was peeled away, but it has two or more
  connections. It lies on a tree hanging off the core, between the core and
  the leaves.

The statistics record `coreShare` — the share of agents in the 2-core, every
25 iterations — and `leaves`, the number of agents with exactly one
connection, after every phase.

## An example

```
  a — b — c — d
  |       |
  e ——————f — g — h
              |
              i
```

The loop a–b–c–f–e–a is the core (and so are its members' connections among
themselves). Peeling removes d, h and i (one connection each) in the first
round, and then g, which has only f left. So: core = {a, b, c, e, f};
leaves = {d, h, i}; tree = {g}.

## Why it matters here

In the game, every connection that carries no [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") is cut
([One iteration, step by step](one-iteration.md)). Cutting a connection on a
loop never disconnects anything — there is always another way round the
loop. Every other connection is a [bridge](bridges.md): one of a tree cuts
off everything beyond it, and one on a path between two loops of the core
cuts the core itself in two. The cleanup then removes whatever is no longer
joined to the largest piece. So a single cut can lose any tree whole — and,
through a bridge of the core, part of the core too.

In a settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world of 100,000 tokens about 57% of agents are in the
core, 35% are leaves and the rest in trees; at 10,000 tokens, 54% and 36%
([What shape does the network take?](what-shape-does-the-network-take.md),
[The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world)).

## How far a tree hangs

Every agent off the core belongs to exactly one tree, and every tree joins
the core at exactly one core agent — two ways into the core would close a
loop, and the agent would be in the core. So each agent off the core has a
**depth**, its number of steps to the core, and a **root**, the core agent
its tree hangs from. In the example, d hangs from c at depth 1; g hangs from
f at depth 1, and h and i at depth 2. The tree hanging from f holds three
agents (g, h and i), the one hanging from c one (d). A breadth-first search
out of every core agent at once finds both for every agent.

## Deeper cores: the onion

The peeling can go further. Remove every agent with fewer than **three**
connections, again and again, and what is left is the **3-core**; with
fewer than *k*, the ***k*-core**. Each *k*-core lies inside the one before,
like the layers of an onion, and an agent's **core number** is the largest
*k* whose *k*-core still holds it (Seidman 1983). An agent on a tree or a
leaf has core number 1; an agent of the 2-core that the 3-core no longer
holds has core number 2; and so on.

In the example every core agent has two connections inside the core, so the
3-core is empty: a, b, c, e and f have core number 2, and d, g, h and i
core number 1. A network in which many agents have high core numbers is
held together by many overlapping loops, not by single rings.

The viewer records only the share of agents in the 2-core (`coreShare`);
the deeper cores, the trees and the [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") of the core are measured in
[The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world).

<!-- turns -->
---

← [Path length](path-length.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Bridges](bridges.md) →

> [!info]- Pages that link here (14)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Stars and hubs](stars-and-hubs.md) · [Bridges](bridges.md) · [Loops](loops.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
