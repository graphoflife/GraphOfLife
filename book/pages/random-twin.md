# The random twin: is a world a space?

## In short

The [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") of a world have no positions, only links. Can such a network
still be a space, in which far places are far apart? To find out, the
notebook sets each world beside its **random twin**: the same agents, each
with as many links as in the world, but wired at random. A random network is
a *small world*, in which everyone is a few steps from everyone; a space is
wide. A world counts as a space when its typical distance is at least 1.5
times its twin's.

## How the twin is made

Give every agent as many link ends as it has links, put all the ends in one
pile, and join them in pairs drawn at random: a *configuration model*. A pair
that would join an agent to itself is dropped, and so is a second link
between the same two agents (`book_graph.degree_preserving`, [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 23). Apart
from those, every agent keeps its number of links, its [degree](degree.md),
and nothing else of the world is kept. So whatever a world has and its twin
lacks was built by the world's rules.

## Why a random network is near

In a random network a link is as likely to be a long jump as a short one. So
the number of agents within *r* steps multiplies at every step, and distances
grow only like the logarithm of the network's size. In a space of *d*
dimensions, about *r*^*d* agents lie within *r* steps of an agent
([Dimension and curvature from ball growth](ball-dimension-and-curvature.md)),
and distances grow as a power of the size.

A twin has no dimension at all. [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") read 2.8 by
ball growth; their twins read 6.5, and are still climbing when, after three
or four steps, a quarter of the twin is reached
([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)).

## The typical distance

From 40 agents drawn at random, count how many agents lie within *r* steps,
for every *r*, and take the geometric mean of the 40 counts: the *typical*
ball, which the few agents beside a [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") do not pull up. The **typical
distance** is the number of steps at which the typical ball holds half the
world, read between whole steps. Do the same on the twin, and divide:

$$
\text{width} = \frac{\text{the world's typical distance}}{\text{its twin's typical distance}} .
$$

A small world has a width of about 1; a space is wider. The notebook's line
for a space is a width of at least 1.5. Both networks are measured on their
largest piece (`research/atlas/space.py`).

## Examples

From [What folds a world](what-folds-a-world.md#b1-and-b2-side-by-side), the
[median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of each kind of world:

| worlds | typical distance | the twin's | width |
|---|---|---|---|
| B1 at 10,000 tokens (Experiment 2) | 10.5 steps | 4.5 steps | 2.3 |
| [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") at 10,000 tokens, [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 2,000 (Experiment 16) | 5.0 steps | 4.2 steps | 1.18 |
| B2 with the [estate](estate.md "The tokens a dead agent leaves, and who gets them.") of the dead to the newborns or the parents ([pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds."), batch R) | 11.2 steps | 4.7 steps | 2.4 |

B1's width grows with its size, as a space's must: 2.3 at 1,500 agents, 3.7
at 13,800 and 5.1 at 64,000. B2's does not: 1.24 at 1,600 agents, 1.18 at
3,400 and 1.14 at 30,700.

## Folded

A world is **folded** when its balls grow over the first few steps like those
of a space of 2 to 4 dimensions, yet it is hardly wider than its twin. B2's
worlds of 5,000 tokens (Experiment 14) are an example: by the
[health bar](health-bar.md)'s
measure, over 1 to 5 steps, they read 3.45 to 4.19, but their typical
distance is only about 1.25 times their twins'. Followed further, the local
dimension of a B2 world of 3,400 agents climbs 2.4, 3.9, 5.0 until, four
steps out, a typical agent's ball holds a quarter of the world. A small world
has no dimension to find, which is why the health bar's dimension rose with
the size of B2's worlds.

The word says how it happens. Every B2 world starts as a space, 1.68 times as
wide as its twin at iteration 100, and folds to 1.24 by iteration 300, once
its first generation dies of age. Each death is mended with a ring of links
among the dead agent's [neighbours](the-world.md#the-network "An agent joined to another by a connection.")
([`mend_on_death`](settings.md#mend_on_death)), which joins agents that were
two steps apart. Worlds without old age, without mending, or whose unused
links lapse keep their room.

Read at iteration 1,000, 395 worlds fell into two groups: 318 folded, with a
median width of 1.26, and 77 spaces, with medians of 2.22 to 3.80 by the rule
that made them. The line of 1.5 lies between the two.

## Where it is used

- [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)
  and [The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world): B1's worlds and their
  cores have a dimension; their twins have none.
- [The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world): a B1 world with a
  mean [path length](path-length.md) of 27.9 steps, against 6.0 when wired at
  random.
- [What folds a world](what-folds-a-world.md): the width of every kind of
  world, and what folds it.
- [Healthy and rich](healthy-and-rich.md), [Rules in company](rules-in-company.md)
  and [The candidates, played on](the-candidates-played-on.md): which worlds
  are spaces.
- [Room for the steadiest world](room-for-the-steadiest-world.md),
  [Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours) and
  [The search's best world, in new seeds](the-candidates-played-on.md#the-searchs-best-world-in-new-seeds):
  theses that ask for a width of at least 1.5.
- [What makes a world interesting](what-makes-a-world-interesting.md) and
  [A screen of random worlds](a-screen-of-random-worlds.md): the criterion
  *almost three-dimensional*.

## Where to go next

- [What folds a world](what-folds-a-world.md): what makes and unmakes space.
- [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md):
  the two rulers of dimension.
- [Path length](path-length.md): distances in a network.
- [The health bar](health-bar.md): the marks a healthy network meets.

<!-- turns -->
---

← [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md) · [Space and dimension](../README.md#space-and-dimension) · [What folds a world](what-folds-a-world.md) →

**Also about:** [The shape and health of the network](../README.md#the-shape-and-health-of-the-network).

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [The estate of the dead](estate.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Healthy and rich](healthy-and-rich.md) · [The health bar](health-bar.md)
<!-- /turns -->
