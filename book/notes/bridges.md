# Bridges

A **bridge** is a connection that lies on no loop. Cut it, and the network
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
Every agent gets a number in the order the walk first reaches it, disc(*u*),
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

`bridges`, the number of bridges, every 25 iterations. The book divides it by
the number of connections, `bridges/edges`: the share of all connections
that are bridges. In a settled baseline world about 31% of connections are
bridges ([Chapter 16](../chapters/16-what-shape-does-the-network-take.md)).

A count of bridges does not say how much hangs on each. The program also
works out, for every bridge, how many agents lie behind it; `cutRisk` is the
largest share of the world a single cut can sever ([Cut risk](cut-risk.md)).
[Chapter 26](../chapters/26-how-a-world-breaks.md) asks whether that risk
foretells how many agents a game cuts off.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
