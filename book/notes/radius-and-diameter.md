# Radius and diameter

Three numbers describe how far apart the agents of a world are. All three
start from the **distance** d(*u*, *v*): the fewest connections on a path from
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
world of thousands, every 25 iterations. The program searches from 8 to 16
agents spread through the list (the same searches as for the mean path
length), then once more from the furthest agent any of them reached — the
"double sweep", which is exact on a tree. So:

- `diameter` can only **under**-report: the true furthest pair might not have
  been found;
- `radius` can only **over**-report: a more central agent might never have
  been tried.

In a settled baseline world the diameter is about 29 steps, the radius about
17 and the mean path length about 10 ([Chapter 24](../chapters/24-the-geometry-of-a-world.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
