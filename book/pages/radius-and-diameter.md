# Radius and diameter

Three numbers describe how far apart the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") of a world are. All three
start from the **distance** d(*u*, *v*): the fewest [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") on a path from
*u* to *v* ([Path length](path-length.md)).

## Eccentricity

The **eccentricity** of an agent is its distance to the agent furthest from
it:

$$
\varepsilon(u) = \max_{v} d(u, v) .
$$

An agent at the centre of the network has a small eccentricity; one at the
tip of a long branch a large one.

## Diameter and radius

- The **diameter** is the largest eccentricity — the distance between the two
  agents furthest apart: diam = max_u ε(*u*).
- The **radius** is the smallest eccentricity — how far the most central agent
  has to reach to touch everyone: rad = min_u ε(*u*).

They are always related by rad ≤ diam ≤ 2·rad: going from any agent to any
other through the most central one takes at most twice the radius.

**Example.** A path of 5 agents a–b–c–d–e: the eccentricities are 4, 3, 2, 3,
4, so the diameter is 4 and the radius 2 (at c).

## How the viewer estimates them

Exact values need a breadth-first search from every agent — too slow for a
world of thousands, every 25 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). The program searches from 8 to 16
agents spread through the list (the same searches as for the mean path
length), then once more from the furthest agent any of them reached — the
"double sweep", which is exact on a tree. So:

- `diameter` can only **under**-report: the true furthest pair might not have
  been found;
- `radius` can only **over**-report: a more central agent might never have
  been tried.

In a settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") the diameter is about 60
steps, the radius about 36 and the mean path length about 18; in one of
10,000 tokens, 29, 17 and 10 ([The geometry of a world](the-geometry-of-a-world.md)).

<!-- turns -->
---

← [Cut risk](cut-risk.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [The spectral gap](spectral-gap.md) →

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (4)
> [Glossary](glossary.md) · [The geometry of a world](the-geometry-of-a-world.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
