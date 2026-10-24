# Three ways to measure a network's dimension

## In short

A network has no coordinates, only links, yet it can be shaped like a line, a
sheet or a solid. Three rulers read its dimension: the room within a few
steps, how a random walker spreads, and how many boxes it takes to cover it.
On a grid all three agree; on irregular networks they need not, and where
they differ says what kind of space the network is.
[Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md)
lays all three on the worlds.

## Three rulers

Stand on an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") of a network and look around. There is no map
and no ruler, only links. The distance between two agents is the number of
links on the shortest path between them ([Path length](path-length.md)). That
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
[random twin](random-twin.md). The twin has the same agents, each with as many
links, wired at random.

## Dimension and curvature from ball growth

How many agents are within *r* steps of a given agent? The answer, as a
function of *r*, says what kind of space a network is: a line, a sheet, a
tree. Two statistics of the Graph of Life viewer read it off, `dimension`
and `ricciCurvature`, on every 25th [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") of a run
([What a run records](frames-and-stats.md)). The same idea lets Wolfram
(2020) read space and its curvature off his network models of physics.

### Balls and shells

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

### The dimension

The viewer starts from about 24 agents spread evenly through the world. Around
each, it counts the shells out to *r* = 5, and it averages the shell sizes
*S*(*r*) over the agents. It drops every radius at which the ball already
holds half the world. Beyond that, the growth measures the edge of the network,
not its shape. Then it fits a straight line to ln *S*(*r*) against
ln *r* by [least squares](least-squares.md). A shell that grows like
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

### The curvature

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

In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") the curvature is +1.4 in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world,
from −1.7 in the starting ring after the first [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). There it mostly means
that the bushy neighbourhoods of [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") fill a ball's first steps, and the
growth slows after them. Read it as "the growth slows", not as a sphere
([Four measures of a world's shape](what-shape-does-the-network-take.md#four-measures-of-a-worlds-shape)).

### A caution

Both numbers are fitted to five or fewer points, on networks that are not
lattices. They are indices for comparing worlds and moments, not exact
geometry. Five steps cannot even tell a space from a network with no
dimension. A tree whose shells double at every step (2, 4, 8, 16 and 32
agents) reads 2.68. A finite dimension shows only as a plateau, a reading
that holds steady over many scales
([The rulers, tried on known spaces](how-many-dimensions-does-a-world-have.md#the-rulers-tried-on-known-spaces)).

### Where the ball dimension is used

- **The [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds.** In B1's worlds of 100,000 tokens the dimension is
  3.4 in the median world (3.0 at 10,000 tokens).
  [The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world)
  reads it beside a picture of the ball growth itself.
- **The health bar** asks for a dimension of 2.3 to 4
  ([The health bar](health-bar.md)). It cannot tell a space from a small
  world. [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s worlds of 5,000 tokens (Experiment 14) read 3.45 to 4.19, yet
  they are hardly wider than their random twins
  ([Folded](random-twin.md#folded)).
- **The two rulers** of
  [Is a world a space?](how-many-dimensions-does-a-world-have.md#two-rulers)
  read the same growth radius by radius, and look for a plateau. They use the
  typical ball, the geometric mean over 40 to 100 agents. Between radii 3 and
  10 they read B1's worlds of 100,000 tokens as 2.8. The viewer reads 3.4 from
  the mean ball over radii 1 to 5, which the few agents next to a hub pull
  up.

## The spectral dimension

How many dimensions does a space have? One way to find out is to count the
room in it. In *d* dimensions, the points within *r* steps of a point grow
like *r*^*d*
([Dimension and curvature from ball growth](#dimension-and-curvature-from-ball-growth)).
Another way is to **get lost in it**. Release a walker that steps at random,
and see how often it comes back to where it started. The more dimensions, the
more directions there are to wander off in, and the rarer the return.

### Why returning measures dimension

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

### The walk used here

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
  over the whole world, at each point in proportion to its [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."). The
  notebook subtracts that resting probability from *P*(*t*), so that only the
  approach to rest is measured. It stops when what is left falls below 5/*n*,
  for a world of *n* points.

### Two dimensions that need not agree

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

### Where the spectral dimension is used

[Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md#how-many-dimensions-does-a-world-have)
measures both dimensions on the worlds, after calibrating them on spaces
whose dimension is known, of about 2,000 points each. By walks, a ring reads
1 there, a square grid 2.1, a cubic grid 3.2 and a tree drawn at random 1.2.
B1's worlds of 100,000 tokens read 1.9 by walks, flat from 4 to 160 steps,
against 2.8 by balls. They have room, but dead ends and narrow passages slow
a walker. In the game, tokens move almost as a random walk would
([Where the tokens flow](where-the-tokens-flow.md#the-game-as-a-random-walk)).
So a difference in tokens spreads slowly through these worlds too.

## Box dimension

A third way to give a network a dimension: cover it with boxes and count
them (Song, Havlin and Makse 2005). It is the network version of the
box-counting dimension of fractals.

### Covering a shape with boxes

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

### Boxes on a network

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

**Example.** Take the world of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 at 10,000 tokens (`B1-10000-s001`),
after its last game. Of size 1, every agent is a box alone:

| size of the boxes, ℓ | 1 | 3 | 5 | 9 | 17 | 33 |
|---|---|---|---|---|---|---|
| boxes needed | 1,336 | 460 | 122 | 41 | 5 | 2 |

On logarithmic axes these lie close to a line of slope about −2: a box
dimension of about 2. The world of seed 1 at 100,000 tokens, 14,870 agents,
reads 1.70
([Covering a world with boxes](what-shape-does-the-network-take.md#covering-a-world-with-boxes)).

### Reading the R²

Small-world networks are covered in a number of boxes that falls
**exponentially** in ℓ, rather than as a power. Over a short range, even an
exponential looks respectable on log–log axes. So a high R² is necessary for
the dimension to mean something, but not sufficient.

### Where the box dimension is used

The viewer records `boxDimension` and `boxDimensionR2` on every 25th
iteration. In the median B1 world of 100,000 tokens, the box dimension is
1.88, with R² = 0.96. That is far below the 3.4 that ball growth gives the
same worlds. Over the thirty worlds it runs from 1.66 to 2.58, and in worlds
of 10,000 tokens it is 1.85. The two dimensions look at different scales. Box
covering runs to size 33, and at large sizes it counts the long, thin
branches
([Four measures of a world's shape](what-shape-does-the-network-take.md#four-measures-of-a-worlds-shape)).

## Where to go next

- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md):
  the rulers on the worlds, and what folds a world into a small world.
- [The random twin: is a world a space?](random-twin.md): how the notebook
  tells a space from a small world.
- [The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world):
  the viewer's measures on the baseline worlds.
- [Physical inspiration](physical-inspiration.md#space-made-of-relations): why
  the notebook asks how many dimensions a world has.

## How our view changed

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

← [The random twin: is a world a space?](random-twin.md) · [Space and dimension](../README.md#space-and-dimension)

> [!info]- Pages that link here (8)
> [Glossary](glossary.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [The random twin: is a world a space?](random-twin.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
