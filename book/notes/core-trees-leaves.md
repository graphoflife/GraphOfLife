# Core, trees and leaves

Every connected network can be cut into two kinds of part: a **core**, where
every agent lies on a loop or on a path between loops, and **trees** that hang
off it. This note says how the cut is made and why it matters for Graph of
Life.

## Peeling

Take the network and repeat:

> remove every agent that has one connection or none,

until no such agent is left. Removing a leaf can make its neighbour a leaf,
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

In the game, every connection that carries no tokens is cut
([Chapter 3](../chapters/03-one-iteration.md)). Cutting a connection inside
the core never disconnects anything — there is always another way round the
loop. Cutting a connection of a tree cuts off everything beyond it, and the
cleanup then removes all of it. The share of a world that hangs in trees is
the share that can be lost to a single cut.

In a settled baseline world about 54% of agents are in the core, 36% are
leaves and the rest in trees ([Chapter 16](../chapters/16-what-shape-does-the-network-take.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
