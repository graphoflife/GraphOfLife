# Loops

A **loop** (cycle) is a path through the network that returns to where it
started without using a [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") twice. Loops are what make a network
robust: along a loop there are always two ways between any two of its [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.").
Counting them needs care, because the number of different loops in a network
can be astronomically large.

## Independent loops: the cycle rank

Take a **spanning tree**: a set of connections that reaches every agent and
contains no loop. A connected network of *n* agents has a spanning tree of
exactly *n* − 1 connections. Every connection **not** in the tree closes
exactly one loop with the tree — its **fundamental cycle** — and these loops
generate all the others (any loop is a combination of them). So the number
of independent loops is

$$
\text{cycle rank} = |E| - |V| + c ,
$$

with *c* the number of connected pieces (1 in every recorded [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."), after the
[cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.")). This is `cycleRank` in the viewer.

**Example.** A square of four agents with one diagonal: |*E*| = 5,
|*V*| = 4, *c* = 1, so 2 independent loops — the two triangles. The square
itself is their combination.

## Loop density

`loopDensity` = `cycleRank` / |*E*|: the share of connections that are
"spare", beyond what a tree would need. 0 is a tree; values near 1 a densely
interwoven network. A settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world has about 0.42 at 100,000
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), 0.38 at 10,000.

## Pieces

`components` is the number of separate pieces. The cleanup keeps only the
largest, so after every phase it is 1 — anything else would be a bug.

## Loops through one agent

The viewer can colour every agent (and every connection) by how many loops
pass through it. It counts the fundamental cycles of a **breadth-first**
spanning tree: for every connection outside the tree, walk both its ends up
the tree until they meet; every agent on the way, and the meeting point, lies
on that loop. Breadth-first trees keep the loops short and local. A different
tree would give a different basis, so this is a fair sample of where the
loops run rather than a canonical count. An agent on no loop at all hangs on
a tree ([Core, trees and leaves](core-trees-leaves.md)).

<!-- turns -->
---

← [Bridges](bridges.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Cut risk](cut-risk.md) →

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [The shape of the network](what-shape-does-the-network-take.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
