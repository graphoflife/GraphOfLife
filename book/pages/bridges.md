# Bridges

A **bridge** is a [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") that lies on no [loop](loops.md "A path through the network that comes back to where it started without using a connection twice."). Cut it, and the network
falls into two pieces. Bridges are where a world is fragile.

## Definition

A connection {*u*, *v*} of a connected network is a bridge if, after removing
it, *u* and *v* can no longer reach each other. Equivalently: it is not part
of any cycle.

Every connection of a tree is a bridge. No connection inside the 2-core is,
except those joining two loops by a single path ([Core, trees and
leaves](core-trees-leaves.md)). Every leaf's connection is a bridge.

## How they are found

By **Tarjan's algorithm** (1974), in one depth-first walk of the network.
Every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") gets a number in the order the walk first reaches it, disc(*u*),
and a "low-link", low(*u*): the smallest disc number reachable from *u*'s
part of the walk by going down the walk's tree and then along at most one
connection back up. The connection from *u* down to its child *v* in the walk
is a bridge exactly when

$$
\text{low}(v) > \text{disc}(u),
$$

that is, when nothing below *v* reaches back to *u* or above. The walk takes
time proportional to agents plus connections.

## What is recorded

`bridges`, the number of bridges, every 25 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). The notebook divides it by
the number of connections, `bridges/edges`: the share of all connections
that are bridges. In a settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world about a quarter of the
connections are bridges at 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), 31% at 10,000
([What shape does the network take?](what-shape-does-the-network-take.md)).

A count of bridges does not say how much hangs on each. The program also
works out, for every bridge, how many agents lie behind it; `cutRisk` is the
largest share of the world a single cut can sever ([Cut risk](cut-risk.md)).
[How a world breaks](how-a-world-breaks.md) asks whether that risk
foretells how many agents a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") cuts off.

<!-- turns -->
---

← [Core, trees and leaves](core-trees-leaves.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Loops](loops.md) →

> [!info]- Pages that link here (10)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [The core of a world](the-core-of-a-world.md) · [How a world breaks](how-a-world-breaks.md) · [Core, trees and leaves](core-trees-leaves.md) · [Cut risk](cut-risk.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
