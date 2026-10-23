# Clustering

Are the [neighbours](the-world.md#the-network "An agent joined to another by a connection.") of an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") also neighbours of each other? In a circle of
friends they often are; in a tree they never are. **Clustering** measures
how often.

## Triangles and triples

A **triangle** is three agents all joined to each other. A **connected
triple** centred on *u* is a pair of *u*'s neighbours — two [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") that
meet at *u*. An agent of [degree](degree.md "The number of an agent's connections: how many neighbours it has.") *d* is the centre of

$$
\binom{d}{2} = \frac{d(d-1)}{2}
$$

triples. Every triangle closes three triples, one at each corner.

## Transitivity

The clustering this notebook uses is the **transitivity**: the share of all
connected triples in the network that are closed into triangles,

$$
C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}
  = \frac{3\,t}{\sum_u \binom{\deg(u)}{2}} .
$$

*C* is between 0 and 1: 0 when there is no triangle at all (a tree, for
example), 1 when every pair of neighbours is joined (every connected piece is
complete). It is recorded as `transitivity`, on every 25th [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.").

## Examples

- **A triangle** of three agents: one triangle, three triples (one at each
  corner), *C* = 3·1/3 = 1.
- **A [star](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.")**: one centre joined to 4 leaves, no other connection. The centre
  has 6 triples, the leaves none, no triangle: *C* = 0.
- **The founders' ring** of the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), before rewiring: every founder has
  degree 4, so 6 triples, of which 3 are closed. Over the ring that is
  *C* = 0.5. Rewiring 37 connections of the real ring of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 lowered it to
  0.27 ([The starting ring](starting-ring.md)).
- **A random network** with the same number of agents and connections, each
  connection placed at random, has *C* close to the probability that two
  given agents are joined, *k̄*/(|*V*| − 1): for 1,300 agents of mean degree
  3.3, about 0.003.

## Why it matters

Triangles are what makes a network hard to cut: in a triangle, every
connection has a way round it. A tree-like network, with few triangles,
falls apart when a connection is cut. A settled baseline world is in
between: clustered far above a random network, but with long tree-like
stretches hanging off its clustered core
([What shape does the network take?](what-shape-does-the-network-take.md)).

<!-- turns -->
---

← [Preferential attachment and its kernel](preferential-attachment.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Path length](path-length.md) →

> [!info]- Pages that link here (8)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [The starting ring](starting-ring.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [How properties scale together](how-properties-scale-together.md) · [What a run records](frames-and-stats.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
