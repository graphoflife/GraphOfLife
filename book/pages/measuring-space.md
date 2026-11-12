# Measuring space: dimension and the random twin

> [!question] Questions of this page
> - How can a network without coordinates have a dimension?
> - What would a world be if it were wired at random?

## In short

How the notebook reads a network as a space: three ways to measure its dimension, and the random twin, the same [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") wired at random, that every reading is set against.

Its sections:

- [Three ways to measure a network's dimension](#three-ways-to-measure-a-networks-dimension)
- [The random twin: is a world a space?](#the-random-twin-is-a-world-a-space)

## Three ways to measure a network's dimension

A network has no coordinates, only links, yet it can be shaped like a line, a
sheet or a solid. Three rulers read its dimension: the room within a few
steps, how a random walker spreads, and how many boxes it takes to cover it.
On a grid all three agree; on irregular networks they need not, and where
they differ says what kind of space the network is.
[Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md)
lays all three on the worlds.

### Three rulers

Stand on an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") of a network and look around. There is no map
and no ruler, only links. The distance between two agents is the number of
links on the shortest path between them ([Path length](measuring-networks.md#distances-in-a-network-path-length-radius-and-diameter-the-spectral-gap)). That
is enough to tell a line from a sheet or a solid, in three ways. On a grid of
*d* dimensions, each of them gives *d*:

| the ruler | on a line | on a sheet | in a solid | in *d* dimensions |
|---|---|---|---|---|
| **room**: the agents within *r* steps | grow like *r* | like *r*² | like *r*³ | like *r*^*d* |
| **a walker**: the chance it is back at its start after *t* steps | falls like *t*^(−1/2) | like *t*^(−1) | like *t*^(−3/2) | like *t*^(−*d*/2) |
| **boxes**: how many of size ℓ it takes to cover it | fall like 1/ℓ | like 1/ℓ² | like 1/ℓ³ | like ℓ^(−*d*) |

On other networks the rulers can disagree, and how they disagree says what
kind of space a network is
([Two dimensions that need not agree](#two-dimensions-that-need-not-agree)).
Some networks have no dimension at all. In a random network everyone is a few
steps from everyone: it is a *small world*, and its readings never settle. To
tell a space from a small world, the notebook sets each world beside its
[random twin](measuring-space.md#the-random-twin-is-a-world-a-space). The twin has the same agents, each with as many
links, wired at random.

### Dimension and curvature from ball growth

How many agents are within *r* steps of a given agent? The answer, as a
function of *r*, says what kind of space a network is: a line, a sheet, a
tree. Two statistics of the Graph of Life viewer read it off, `dimension`
and `ricciCurvature`, on every 25th [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") of a run
([What a run records](runs-and-records.md#what-a-run-records)). The same idea lets Wolfram
(2020) read space and its curvature off his network models of physics.

#### Balls and shells

The **ball** of radius *r* around agent *u* is every agent at distance at most
*r* from it. The **shell** of radius *r* is every agent at distance exactly
*r*: the ball's outer layer. In a space of dimension *d*, a ball holds about
*r*^*d* points and a shell about *r*^(*d*−1):

- on a line, the shell at distance *r* has 2 points (*d* = 1);
- on a square grid, 4*r* (*d* = 2);
- in a cubic grid, 4*r*² + 2: that is 6, 18, 38, 66 and 102 points for
  *r* = 1 to 5 (*d* = 3).

In a **tree** in which every agent has *b* children, the shell grows as
*b*^*r*. It grows by the same factor *b* at every step, faster than any power
of *r*. A network like that has no finite dimension.

#### The dimension

The viewer starts from about 24 agents spread evenly through the world. Around
each, it counts the shells out to *r* = 5, and it averages the shell sizes
*S*(*r*) over the agents. It drops every radius at which the ball already
holds half the world. Beyond that, the growth measures the edge of the network,
not its shape. Then it fits a straight line to ln *S*(*r*) against
ln *r* by [least squares](statistics.md#fitting-a-straight-line). A shell that grows like
*r*^(*d*−1) gives a line of slope *d* − 1, so

$$
\texttt{dimension} = \text{slope} + 1 .
$$

Checked on grids, it gives 1.00 for a line, 1.92 for a square grid and 2.56
for a cubic grid. It falls short above one dimension for two reasons. First,
agents near the edge of a finite grid have smaller shells. A square grid
wrapped round, so that it has no edge, has shells of exactly 4*r* and reads
exactly 2. Second, a cubic grid's shells, 6, 18, 38, 66 and 102, grow more
slowly than *r*² over the first five steps. Fitted as above, they give a slope
of 1.76, a dimension of 2.76, even with no edge. Read the dimension as a rough
index.

#### The curvature

On a curved surface, balls grow differently from flat space. On a sphere
(positive curvature) they grow more slowly; on a saddle (negative curvature),
faster. For a ball in *d* dimensions with **Ricci scalar** *R*, the shell
grows as

$$
S(r) \propto r^{d-1} \left( 1 - \frac{R\, r^2}{6 d} + \dots \right) .
$$

In words: the power of *r* that flat space would give, times a correction
that grows with *r*² and has the opposite sign to *R*. So the viewer fits
ln *S*(*r*) against **both** ln *r* and *r*². The first coefficient gives
*d* − 1, the second −*R*/(6*d*), and `ricciCurvature` is that *R*. The fit
needs at least four radii: three would fit its three numbers exactly, and
prove nothing.

- **Negative**: neighbourhoods hold more than flat space allows. Branching,
  tree-like and expander-like networks give this.
- **Positive**: neighbourhoods close back on themselves and hold less.

In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") the curvature is +1.4 in the median world,
from −1.7 in the starting ring after the first [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). There it mostly means
that the bushy neighbourhoods of hubs fill a ball's first steps, and the
growth slows after them. Read it as "the growth slows", not as a sphere
([Four measures of a world's shape](b1-the-network-size-and-scaling.md#four-measures-of-a-worlds-shape)).

#### A caution

Both numbers are fitted to five or fewer points, on networks that are not
lattices. They are indices for comparing worlds and moments, not exact
geometry. Five steps cannot even tell a space from a network with no
dimension. A tree whose shells double at every step (2, 4, 8, 16 and 32
agents) reads 2.68. A finite dimension shows only as a plateau, a reading
that holds steady over many scales
([The rulers, tried on known spaces](how-many-dimensions-does-a-world-have.md#the-rulers-tried-on-known-spaces)).

#### Where the ball dimension is used

The viewer reads B1's worlds of 100,000 tokens as 3.4, from the mean ball
over radii 1 to 5. The typical ball between radii 3 and 10 reads them as
2.8 ([Two rulers](how-many-dimensions-does-a-world-have.md#two-rulers)).

<!-- used dimension-and-curvature-from-ball-growth -->
No page links here yet.
<!-- /used -->

### The spectral dimension

How many dimensions does a space have? One way to find out is to count the
room in it. In *d* dimensions, the points within *r* steps of a point grow
like *r*^*d*
([Dimension and curvature from ball growth](#dimension-and-curvature-from-ball-growth)).
Another way is to **get lost in it**. Release a walker that steps at random,
and see how often it comes back to where it started. The more dimensions, the
more directions there are to wander off in, and the rarer the return.

#### Why returning measures dimension

After *t* random steps, a walker has typically wandered about √*t* steps
away, in every direction open to it. In *d* dimensions it is then spread
over a region of about (√*t*)^*d* = *t*^(*d*/2) points, more or less evenly.
So the probability that it is back at its start is about one over that:

$$
P(t) \propto t^{-d_s/2} .
$$

The exponent defines the **spectral dimension** *d*ₛ. On a line, *P*(*t*)
falls like *t*^(−1/2); on a plane like *t*^(−1); in our three-dimensional
space like *t*^(−3/2). To read it off, compare two times, *t* and 2*t*:

$$
d_s(t) = -2 \, \frac{\ln P(2t) - \ln P(t)}{\ln 2} .
$$

In words: make the walk twice as long, and count how many times the chance of
being back halves; the spectral dimension is twice that. Say a walker is back
at its start with probability 0.02 after 100 steps, and 0.01 after 200. The
chance halves once, so *d*ₛ = 2: the walk spreads as on a plane.

The name comes from the network's Laplacian, the matrix behind diffusion on
it. Its spectrum sets *P*(*t*) (Alexander and Orbach 1982). The heat
equation of [Physical inspiration](physical-inspiration.md#agents-as-little-calculators)
spreads heat exactly as the walker's probability spreads.

#### The walk used here

No run records this dimension: the notebook computes it where a page needs
it, with this walk:

- **Lazy:** at every step the walker stays where it is with probability ½,
  and otherwise moves to a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") drawn at random. Some networks,
  such as a grid or a tree, have points of two alternating classes. Without
  the laziness, a walk on them could return only at even steps, and *P*(*t*)
  would jump between zero and twice its value.
- **Exact:** instead of simulating walkers, the notebook propagates the whole
  probability distribution, step by step, from a few dozen starting points.
- **Minus the resting state:** in a finite world, the walker ends up spread
  over the whole world, at each point in proportion to its connections. The
  notebook subtracts that resting probability from *P*(*t*), so that only the
  approach to rest is measured. It stops when what is left falls below 5/*n*,
  for a world of *n* points.

#### Two dimensions that need not agree

On a regular grid, the ball dimension and the spectral dimension are the
same. On irregular spaces they differ, and how they differ is informative:

- On a **tree**, balls grow quickly, but a walker keeps running into dead
  ends and coming back. A tree drawn at random has a ball dimension of 2 and
  a spectral dimension of 4/3 (Aldous 1991; Durhuus, Jonsson and Wheater
  2007).
- On a **random network** in which every point has the same few neighbours,
  balls grow exponentially and the walker almost never returns. Both
  dimensions are infinite. Measured at growing scales, the readings keep
  rising.
- In **quantum gravity**, *d*ₛ is a favourite probe of space-time built
  from discrete pieces. The causal dynamical triangulations of Ambjørn,
  Jurkiewicz and Loll (2005) give *d*ₛ ≈ 4 at large scales and ≈ 2 at
  small ones.

#### Where the spectral dimension is used

[Is a world a space?](how-many-dimensions-does-a-world-have.md#how-many-dimensions-does-a-world-have)
calibrates it on rings and grids, and reads B1's worlds as 1.9 by walks
against 2.8 by balls: they have room, but dead ends slow a walker.

<!-- used the-spectral-dimension -->
- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md#two-rulers)

> [!info]- And in the Archive (1)
> [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#how-many-dimensions-the-core-has)
<!-- /used -->

### Box dimension

A third way to give a network a dimension: cover it with boxes and count
them (Song, Havlin and Makse 2005). It is the network version of the
box-counting dimension of fractals.

#### Covering a shape with boxes

Cover a shape with boxes of side ℓ and count how many are needed, *N*(ℓ). For
a line of length 1, *N* = 1/ℓ; for a square, (1/ℓ)²; for a cube, (1/ℓ)³.
Halve the boxes, and the line needs twice as many, the square four times as
many, the cube eight times. In general

$$
N(\ell) \propto \ell^{-d_B} ,
$$

and *d*_B is the **box dimension**: the power by which the number of boxes
falls as they grow. A fractal can have a *d*_B that is not a whole number. A
Sierpiński triangle is made of three half-size copies of itself. So it needs
three times as many boxes each time they are halved: *d*_B = ln 3 / ln 2 ≈ 1.58.

#### Boxes on a network

A **box of size ℓ** is a set of agents no two of which are more than ℓ − 1
steps apart. The viewer builds boxes as balls: a centre and every agent within
(ℓ − 1)/2 steps of it. Two agents in such a ball are at most ℓ − 1 steps
apart. Covering a network with the fewest boxes is a hard problem, so it is
done greedily:

1. Order the agents by number of connections, most first.
2. Go down the list. Every agent not yet in a box becomes the centre of a new
   box. The box takes every agent within (ℓ − 1)/2 steps of the centre that no
   box holds yet.
3. Count the boxes, for ℓ = 1, 3, 5, 9, 17, 33 (stopping once one box holds
   everything).

Then fit ln *N*(ℓ) against ln ℓ by least squares. The slope is minus the box
dimension, and the R² says how close the counts lie to a straight line:

$$
\texttt{boxDimension} = -\text{slope}, \qquad \texttt{boxDimensionR2} = R^2 .
$$

**Example.** Take the world of [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 at 10,000 tokens (`B1-10000-s001`),
after its last game. Of size 1, every agent is a box alone:

| size of the boxes, ℓ | 1 | 3 | 5 | 9 | 17 | 33 |
|---|---|---|---|---|---|---|
| boxes needed | 1,336 | 460 | 122 | 41 | 5 | 2 |

On logarithmic axes these lie close to a line of slope about −2: a box
dimension of about 2. The world of seed 1 at 100,000 tokens, 14,870 agents,
reads 1.70
([Covering a world with boxes](b1-the-network-size-and-scaling.md#covering-a-world-with-boxes)).

#### Reading the R²

Small-world networks are covered in a number of boxes that falls
**exponentially** in ℓ, rather than as a power. Over a short range, even an
exponential looks respectable on log–log axes. So a high R² is necessary for
the dimension to mean something, but not sufficient.

#### Where the box dimension is used

The viewer records `boxDimension` and `boxDimensionR2` on every 25th
iteration. In the median B1 world it is about 1.9, far below the 3.4 of
ball growth, because large boxes count the long, thin branches
([Four measures of a world's shape](b1-the-network-size-and-scaling.md#four-measures-of-a-worlds-shape)).

<!-- used box-dimension -->
- [Physical inspiration](physical-inspiration.md#how-to-read-a-dimension-off-a-network)
- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md#the-rulers-on-the-worlds)
- [How runs are made, recorded and costed](runs-and-records.md#the-statistics-file)

> [!info]- And in the Archive (1)
> [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#four-measures-of-a-worlds-shape)
<!-- /used -->

## The random twin: is a world a space?

The agents of a world have no positions, only links. Can such a network
still be a space, in which far places are far apart? To find out, the
notebook sets each world beside its **random twin**: the same agents, each
with as many links as in the world, but wired at random. A random network is
a *small world*, in which everyone is a few steps from everyone; a space is
wide. A world counts as a space when its typical distance is at least 1.5
times its twin's.

### How the twin is made

Give every agent as many link ends as it has links, put all the ends in one
pile, and join them in pairs drawn at random: a *configuration model*. A pair
that would join an agent to itself is dropped, and so is a second link
between the same two agents (`book_graph.degree_preserving`, seed 23). Apart
from those, every agent keeps its number of links, its [degree](measuring-networks.md#an-agents-connections-degree-clustering-assortativity-attachment-and-scaling),
and nothing else of the world is kept. So whatever a world has and its twin
lacks was built by the world's rules.

### Why a random network is near

In a random network a link is as likely to be a long jump as a short one. So
the number of agents within *r* steps multiplies at every step, and distances
grow only like the logarithm of the network's size. In a space of *d*
dimensions, about *r*^*d* agents lie within *r* steps of an agent
([Dimension and curvature from ball growth](measuring-space.md#three-ways-to-measure-a-networks-dimension)),
and distances grow as a power of the size.

A twin has no dimension at all. B1's worlds of 100,000 tokens read 2.8 by
ball growth; their twins read 6.5, and are still climbing when, after three
or four steps, a quarter of the twin is reached
([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)).

### The typical distance

From 40 agents drawn at random, count how many agents lie within *r* steps,
for every *r*, and take the geometric mean of the 40 counts: the *typical*
ball, which the few agents beside a hub do not pull up. The **typical
distance** is the number of steps at which the typical ball holds half the
world, read between whole steps. Do the same on the twin, and divide:

$$
\text{width} = \frac{\text{the world's typical distance}}{\text{its twin's typical distance}} .
$$

A small world has a width of about 1; a space is wider. The notebook's line
for a space is a width of at least 1.5. Both networks are measured on their
largest piece (`research/atlas/space.py`).

### Examples

From [What folds a world](how-many-dimensions-does-a-world-have.md#b1-and-b2-side-by-side), the
median of each kind of world:

| worlds | typical distance | the twin's | width |
|---|---|---|---|
| B1 at 10,000 tokens (Experiment 2) | 10.5 steps | 4.5 steps | 2.3 |
| [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") at 10,000 tokens, iteration 2,000 (Experiment 16) | 5.0 steps | 4.2 steps | 1.18 |
| B2 with the [estate](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") of the dead to the newborns or the parents (pilots, batch R) | 11.2 steps | 4.7 steps | 2.4 |

B1's width grows with its size, as a space's must: 2.3 at 1,500 agents, 3.7
at 13,800 and 5.1 at 64,000. B2's does not: 1.24 at 1,600 agents, 1.18 at
3,400 and 1.14 at 30,700.

### Folded

A world is **folded** when its balls grow over the first few steps like those
of a space of 2 to 4 dimensions, yet it is hardly wider than its twin. B2's
worlds of 5,000 tokens (Experiment 14) are an example: by the
[health bar](measuring-health.md#the-health-bar)'s
measure, over 1 to 5 steps, they read 3.45 to 4.19, but their typical
distance is only about 1.25 times their twins'. Followed further, the local
dimension of a B2 world of 3,400 agents climbs 2.4, 3.9, 5.0 until, four
steps out, a typical agent's ball holds a quarter of the world. A small world
has no dimension to find, which is why the health bar's dimension rose with
the size of B2's worlds.

The word says how it happens. Every B2 world starts as a space, 1.68 times as
wide as its twin at iteration 100, and folds to 1.24 by iteration 300, once
its first generation dies of age. Each death is mended with a ring of links
among the dead agent's neighbours
([`mend_on_death`](settings.md#mend_on_death)), which joins agents that were
two steps apart. Worlds without old age, without mending, or whose unused
links lapse keep their room.

Read at iteration 1,000, 498 worlds fell into two groups: 371 folded, with
median widths of 1.26 and 1.27, and 127 spaces, with medians of 2.09 to 3.63
by the rule that made them. The line of 1.5 lies between the two.

### Where it is used

<!-- used the-random-twin-is-a-world-a-space -->
- [Which worlds are healthy? Three orders](which-worlds-are-healthy.md#three-ways-to-put-the-criteria-together)
- [Measuring lives: births, deaths, ages and estates](measuring-lives.md#an-example-paying-for-births)
- [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md#where-degree-is-used)
- [Physical inspiration](physical-inspiration.md#what-the-worlds-turn-out-to-be)
- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md#the-rulers-on-the-worlds)
- [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md#where-it-is-used-1)
- [Measuring health: criteria, the health bar and diversity](measuring-health.md#read-it-with-care)

> [!info]- And in the Archive (5)
> [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#how-many-dimensions-the-core-has) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#two-halves) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md#nearer-the-authors-picture) · [Space at scale (side study)](space-at-scale.md#the-rulers-at-every-size) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md#nulls-and-controls)
<!-- /used -->

## Where to go next

- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md):
  the rulers on the worlds, and what folds a world into a small world.
- [The geometry of a world](b1-the-network-size-and-scaling.md#the-geometry-of-a-world):
  the viewer's measures on the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds.
- [Physical inspiration](physical-inspiration.md#space-made-of-relations): why
  the notebook asks how many dimensions a world has.
- [Path length](measuring-networks.md#distances-in-a-network-path-length-radius-and-diameter-the-spectral-gap): distances in a network.
- [The health bar](measuring-health.md#the-health-bar): the marks a healthy network meets.

## How our view changed

- *Three ways to measure a network's dimension*:
  - **24 October 2026.** From 8 October, this page said that a shell in a cubic
    grid holds about 6*r*² points. It holds exactly 4*r*² + 2. The page also
    said that the dimension falls short on grids because five steps are too few
    for the growth to reach its limit. That holds for a cubic grid, whose shells
    read 2.76 even with no edge. It does not hold for a square grid, whose shells
    of 4*r* read exactly 2. Its 1.92 comes from the edges of a finite grid.
    And the page said that the viewer starts from up to 24 agents. It takes
    every ⌊*n*/24⌋-th of a world's *n* agents, which in any world of a thousand
    agents or more is 24 or 25. All three were found when this page and two
    others were merged into one, by counting and with the viewer's code
    (`gol_series.py`).
- *The random twin: is a world a space?*:
  - **24 October 2026.** Since 23 October, this page had said that 395 worlds
    read at iteration 1,000 fell into 318 folded and 77 spaces, the spaces with
    medians of 2.22 to 3.80. That reading lacked batches L and V and the second
    round of the search by elimination. With them, 498 worlds fall into 371 and
    127, with medians of 2.09 to 3.63
    ([Every world at iteration 1,000](how-many-dimensions-does-a-world-have.md#every-world-at-iteration-1000)).
    The two groups stay apart.

## References

- Aldous, D. (1991). The continuum random tree II: an overview. In
  *Stochastic Analysis*, Cambridge University Press, 23–70.
- Alexander, S. and Orbach, R. (1982). Density of states on fractals:
  "fractons". *Journal de Physique Lettres* 43, L625–L631.
- Ambjørn, J., Jurkiewicz, J. and Loll, R. (2005). Spectral dimension of the
  universe is scale dependent. *Physical Review Letters* 95, 171301.
- Durhuus, B., Jonsson, T. and Wheater, J. F. (2007). The spectral dimension
  of generic trees. *Journal of Statistical Physics* 128, 1237–1260.
- Song, C., Havlin, S. and Makse, H. A. (2005). Self-similarity of complex
  networks. *Nature* 433, 392–395.
- Wolfram, S. (2020). A class of models with the potential to represent
  fundamental physics. *Complex Systems* 29(2), 107–536.

<!-- turns -->
---

← [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) →

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (12)
> [Glossary](glossary.md) · [Measuring lives: births, deaths, ages and estates](measuring-lives.md) · [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) · [How runs are made, recorded and costed](runs-and-records.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [Space at scale (side study)](space-at-scale.md) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md)
<!-- /turns -->
