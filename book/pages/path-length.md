# Distances in a network: path length, radius and diameter, the spectral gap

## In short

The **distance** between two [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") is the fewest [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") on a path from
one to the other. The **mean path length** is its average over pairs, the
**diameter** its largest value, and the **radius** the reach of the most
central agent. The **spectral gap** says how hard the network is to cut in
two, and how fast a random walker on it forgets where it began. In a settled
[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), two agents are a [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") 17.7 steps apart, the
radius is 36 and the diameter 60.

## Path length

How far apart are two agents of a world? Agents have no positions, only
connections. A token moves along them, at most one connection in a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). So
the natural measure of distance is a count of steps along connections.

The **distance** d(*u*, *v*) between two agents is the smallest number of
connections on a path from one to the other. The **mean path length** is
the average distance between agents, over every pair of them.

**Example.** Five agents stand in a row, a–b–c–d–e, each joined to the next.
The distance from a to e is 4, and from b to d it is 2. Of the 10 pairs of
agents, four are 1 step apart and three are 2 steps apart. Two pairs are 3
steps apart, and one pair is 4. The distances add up to 20, so the mean path
length is 20/10 = 2.

### Breadth-first search

The distances from one agent *s* to every other are found by a
**breadth-first search**:

1. *s* is at distance 0;
2. its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") are at distance 1;
3. their neighbours not yet reached are at distance 2;
4. and so on, until every agent is reached.

It takes time proportional to the number of agents plus the number of
connections. In the row a–b–c–d–e, a search from c finds b and d at
distance 1, and a and e at distance 2.

### The estimate the notebook uses

The exact mean over every pair of agents needs a search from every agent.
That is too slow for worlds of thousands of agents, every 25 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.").
So `meanPathLength`, which a run records on the rows of every 25th iteration
([What a run records](frames-and-stats.md)), is estimated from a spread of
sources (`gol_series.py`):

1. Let *n* be the number of agents, listed in increasing order of id, and
   |*E*| the number of connections. The number of searches wanted is
   *S* = max(8, min(16, round(250,000 / (*n* + 2|*E*|)))): between 8 and 16,
   fewer for bigger worlds. A search visits the *n* agents and the 2|*E*|
   ends of connections. So *S* searches make about 250,000 visits in all,
   unless a bound applies.
2. Let *step* = max(1, ⌊*n* / *S*⌋). A search is run from the agents at
   positions 0, *step*, 2·*step*, … of the list, as far as it goes. In a world
   of more than a few hundred agents, that makes *S* searches. It makes one
   more when *n* is not a multiple of *S*.
3. The mean path length is the average of the distances from these sources
   to every other agent (each source's distance to itself left out).

After the [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") the world is one connected piece, so every agent can be
reached from every source.

**Example.** The world of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 of Experiment 10 ended with 14,870 agents
and 26,166 connections. So *n* + 2|*E*| = 67,202, and 250,000 / 67,202 ≈ 3.7
rounds to 4: *S* = 8, the least allowed. The step is ⌊14,870 / 8⌋ = 1,858.
The searches start at positions 0, 1,858, 3,716, … up to 8 × 1,858 = 14,864,
which is still in the list: 9 searches.

### Near or far?

How many steps count as far? That depends on what the network could have
been.

In a **random network**, a connection is as likely to be a long jump as a
short one. Let *k̄* be the mean [degree](degree.md "The number of an agent's connections: how many neighbours it has."), the mean number of connections per
agent. Each step outward then reaches roughly *k̄* times as many agents as the
step before. So *r* steps reach about *k̄*^*r* agents, and all *N* agents are
reached when *k̄*^*r* ≈ *N*. That gives the usual rough estimate of the
distance in a random network of *N* agents:

$$
d_{\text{random}} \approx \frac{\ln N}{\ln \bar k} .
$$

In words: the number of steps *r* at which *k̄*^*r* reaches *N*. So distances
in a random network grow only like the logarithm of its size: a hundred times
the agents adds only a few steps. Such a network is called a **small world**.

In a lattice, a regular grid of *d* [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), a lump of *N* agents is about
*N*^(1/*d*) agents wide. A cube of a thousand agents, for example, is ten
agents wide. There, distances grow like a power of the number of agents. So
how distances grow with the size of a world says what kind of network it is.

The founders' ring a world starts on is a small world by design. In the world
of seed 1 at 10,000 tokens, a plain ring of its 100 founders would have
them 12.9 steps apart on average. With its shortcuts, they started 4.4 steps
apart ([The starting ring](starting-ring.md)).

### Where path length is used

- **How stretched a world is.** Take B1's thirty worlds of 100,000 tokens
  (Experiment 10), over their [settled life](median-and-quantiles.md#the-settled-life-of-a-world "A world from iteration 500 to its end, after its youth, where it is measured."), iterations 500 to 1,499. Two
  agents are a median 17.7 steps apart. A random network with the
  same *N* and *k̄* would have about 7.8. So the agents are 2.3 times as far
  apart
  ([How far apart are two agents?](what-shape-does-the-network-take.md#how-far-apart-are-two-agents)).
  A token moves at most one connection in a game, so it needs about 18
  games to go from one agent to another, on average.
- **Smaller worlds.** In B1's worlds of 10,000 tokens (Experiment 2), the
  settled life runs from iteration 500 to 2,999. Averaged over it, the mean
  path length is 9.8 in the median world. It lies between 6.5 and 13.0 across
  the 26 worlds that lived to the end, against about 6.0 for a random
  network. A token there needs about 10 games to go from one agent to
  another.
- **Against the random twin.** Computed exactly after its last game, the
  world of seed 1 of Experiment 10 has a mean path length of 27.9 steps. Its
  random twin, the same agents with as many connections each but wired at
  random, has 6.0
  ([Distances](what-shape-does-the-network-take.md#distances)).
- **As worlds grow.** Their distances grow like a power of the number of
  agents, not like its logarithm. That is how distances grow in a body of
  about three dimensions
  ([Distances grow like a power](how-does-a-worlds-size-follow-its-tokens.md#distances-grow-like-a-power)).

## Radius and diameter

The mean path length is a typical distance. Two more numbers describe the
extremes. How far apart are the two agents furthest from each other? And how
far must the most central agent reach to touch everyone? Both start from the
distance d(*u*, *v*): the fewest connections on a path from *u* to *v*
([Path length](#path-length)).

### Eccentricity

The **eccentricity** of an agent is its distance to the agent furthest from
it:

$$
\varepsilon(u) = \max_{v} d(u, v) .
$$

In words: search outward from *u*, and count the steps to the last agent
reached. An agent at the centre of the network has a small eccentricity; one
at the tip of a long branch a large one.

### The largest and the smallest eccentricity

- The **diameter** is the largest eccentricity — the distance between the two
  agents furthest apart: diam = max_u ε(*u*).
- The **radius** is the smallest eccentricity — how far the most central agent
  has to reach to touch everyone: rad = min_u ε(*u*).

They are always related by rad ≤ diam ≤ 2·rad. Going from any agent to any
other through the most central one takes at most twice the radius.

**Example.** In the row of 5 agents a–b–c–d–e, the eccentricities are 4, 3, 2,
3 and 4. So the diameter is 4 and the radius 2, at c.

### How the viewer estimates them

Exact values need a breadth-first search from every agent: too slow for a
world of thousands, every 25 iterations. So the program reuses the searches
of the mean path length
([The estimate the notebook uses](#the-estimate-the-notebook-uses)). Then it
searches once more, from the furthest agent any of them reached. This
"double sweep" is exact on a tree. There, whichever agent a search starts
from, the agent furthest from it is one end of a longest path. So:

- `diameter` can only **under**-report: the true furthest pair might not have
  been found;
- `radius` can only **over**-report: a more central agent might never have
  been tried.

In the world of seed 1 after its last game, that makes the 9 searches of the
example above. A tenth serves the diameter alone.

### Where radius and diameter are used

- **The thirty worlds.** Over the settled life of B1's worlds of 100,000
  tokens, the median world has a radius of 36 and a diameter of 60. In
  the worlds of 10,000 tokens they were 17 and 29, with a mean path length of
  9.8: all three about half
  ([Distances](what-shape-does-the-network-take.md#distances)).
- **Against the random twin.** Computed exactly after its last game, the
  world of seed 1 of Experiment 10 has a radius of 48 and a diameter of 96.
  Its random twin has 8.6 and 16.
- **As worlds grow.** The diameter grows like a power of the number of
  agents, and more cleanly than the mean path length
  ([Distances grow like a power](how-does-a-worlds-size-follow-its-tokens.md#distances-grow-like-a-power)).

## The spectral gap

How hard is a network to cut into two halves? Counting bridges answers for
cuts of a single connection. The **spectral gap** λ₂ answers for every way of
dividing the network at once. It is also the speed at which anything that
spreads along the connections — a random walker, heat, tokens — forgets where
it started.

### Step by step

1. **The random walk.** A walker at agent *u* moves to a neighbour chosen
   uniformly at random. After many steps, where is it? On a connected network
   it ends up at agent *u* with probability proportional to deg(*u*), its
   number of connections, whatever it started from. There is one exception:
   when the agents fall into two classes and every connection joins one class
   to the other, as in a tree. Then the walker swings between the classes at
   every step, and settles only on average over two steps. A network with a
   single triangle is not like that, and the worlds have many.
2. **How fast?** The difference from that resting distribution shrinks, step
   by step, roughly by a fixed factor (1 − λ₂). After about 1/λ₂ steps the
   start is forgotten. When λ₂ is near 1, a few steps suffice. When it is near
   0, the walker stays in its own region for a long time.
3. **Why a region traps it.** A walker leaves a region only through the
   connections that lead out of it. If a big region has few such connections —
   a bottleneck — the walk lingers. So a small λ₂ means there is a cheap cut
   somewhere.

### The definition

λ₂ is the second-smallest eigenvalue of the **normalised Laplacian**. An
eigenvalue of a matrix *M* is a number λ for which *Mx* = λ*x* holds for some
vector *x* other than zero. The matrix only stretches that vector. The
normalised Laplacian is the matrix

$$
\mathcal{L} = I - D^{-1/2} A D^{-1/2} .
$$

Here *I* is the identity matrix, and *D* the diagonal matrix of degrees.
*A* is the adjacency matrix: *A*ᵤᵥ = 1 if *u* and *v* are joined, and 0 if
not. The eigenvalues of the Laplacian lie between 0 and 2; the smallest is
always 0, and λ₂ > 0 exactly when the network is connected.

Why this matrix? At every step, the walker's chances of being at each agent
are multiplied by the matrix *D*⁻¹*A*. Its eigenvalues are 1 − λ, for every
eigenvalue λ of the Laplacian. The largest, 1, belongs to the resting
distribution. The next largest, 1 − λ₂, is the factor of step 2 above.

### Cheeger's inequality

**Cheeger's inequality** ties λ₂ to the network's worst bottleneck *h*. For a
set *S* of agents, count the connections that leave it, |∂*S*|. Divide by
vol(*S*), the sum of its agents' degrees: the number of connection ends in
it. The worst bottleneck is the smallest such ratio, over every set that
holds at most half of all connection ends. With *V* the set of all agents:

$$
h = \min_{\mathrm{vol}(S) \,\le\, \mathrm{vol}(V)/2} \frac{|\partial S|}{\mathrm{vol}(S)} .
$$

Then λ₂ and *h* bound each other (Chung 1997):

$$
\frac{h^2}{2} \;\le\; \lambda_2 \;\le\; 2h .
$$

A small λ₂ promises that a good cut exists; a large one proves that none does.

**Example.** Cut a ring of an even number *n* of agents into two halves. Two
connections leave each half, and each half holds *n* connection ends: a ratio
of 2/*n*. No set of at most half the ring has a smaller ratio, so
*h* = 2/*n*. Cheeger's inequality then puts λ₂ between 2/*n*² and 4/*n*. The
true value, about 2π²/*n*² ([Examples](#examples)), lies near the bottom.

### Examples

- A complete network of *n* agents, each joined to every other:
  λ₂ = *n*/(*n* − 1), about 1.
- A ring of *n* agents: λ₂ = 1 − cos(2π/*n*) ≈ 2π²/*n*², tiny for large *n*.
  A ring is easy to cut (twice) into two arcs.
- A grid of *d* dimensions with *N* agents: λ₂ shrinks like *N*^(−2/*d*); the
  ring is the case *d* = 1. A world grown into a well-knit body of three
  dimensions would show this
  ([Physical inspiration](physical-inspiration.md)).

### How it is computed

By the Lanczos method, on the largest connected piece (`gol_spectral.py`).
The method multiplies a vector by the Laplacian, 48 times over, and keeps
each result at right angles to all the earlier ones. From these it builds a
small tridiagonal matrix: one whose entries lie only on its diagonal and next
to it. Its eigenvalues approach the Laplacian's at the ends of the spectrum.
The smallest of them is then found by bisection, halving an interval again
and again. The direction of the eigenvalue 0 is taken out at every step, so
the smallest value found is λ₂.

The value found is an upper bound on λ₂, exact on well-connected networks. On
nearly divided ones it is a small number whose smallness matters more than its
last digit.

### Where the spectral gap is used

- **The thirty worlds.** A settled B1 world has a spectral gap of about
  0.0012 at 100,000 tokens and 0.0026 at 10,000
  ([Four measures of a world's shape](what-shape-does-the-network-take.md#four-measures-of-a-worlds-shape)).
  So a random walk on it needs on the order of 1/0.0012 ≈ 800 steps to forget
  its start (or 1/0.0026 ≈ 400).
- **Bottlenecks.** After its last game, the world of seed 1 of Experiment 10
  has λ₂ = 0.00095, and its random twin 0.031: thirty-three times larger. By Cheeger's inequality, part of the world
  is joined to the rest by few connections
  ([How easily a world could be cut](what-shape-does-the-network-take.md#how-easily-a-world-could-be-cut)).
  [How a world breaks](how-a-world-breaks.md) finds those bottlenecks.
- **Tokens.** A game moves tokens almost like a random walk
  ([The game as a random walk](where-the-tokens-flow.md#the-game-as-a-random-walk)).
  In the world of seed 1, such a walk needs about 1/0.00095, a thousand
  steps, to forget its start. So tokens settle within a neighbourhood in a few
  games, but could spread evenly across the whole world only over about a
  thousand.

## Where to go next

- [The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world):
  the distances and the spectral gap of the thirty worlds, beside their
  random twins.
- [The random twin: is a world a space?](random-twin.md): how the notebook
  tells a space from a small world.
- [How a world breaks](how-a-world-breaks.md): where the bottlenecks that a
  small spectral gap promises give way.
- [Bridges, cut risk and loops: how a network can be cut](bridges.md): the
  cuts of a single connection, one by one.

## How our view changed

- **24 October 2026.** Three notes became one page: on path length, on radius
  and diameter, and on the spectral gap. Their statements were checked against
  the code (`gol_series.py`, `gol_spectral.py`) and the theory, and three were
  corrected:
  - From 8 October, the note on the spectral gap defined the bottleneck *h* of
    Cheeger's inequality differently. It divided the connections leaving a set
    by the connections inside it, over sets of at most half the network. The
    inequality holds for *h* as defined [above](#cheegers-inequality): the
    connections leaving a set over the sum of its agents' degrees. The sets
    are those holding at most half of all connection ends (Chung 1997).
  - The same note said that on any connected network, a random walker ends
    up at each agent with a probability proportional to its degree. When the
    agents fall into two classes and every connection joins one to the other,
    as in a tree, the walker swings between them. It settles only on average
    over two steps.
  - The note on radius and diameter said that the program searches from 8 to
    16 agents. That is how many searches it wants. In a world of more than a
    few hundred agents, it makes that many or one more. In the world of seed 1
    after its last game it makes 9, then a tenth for the diameter.

## References

- Chung, F. R. K. (1997). *Spectral Graph Theory*. CBMS Regional Conference
  Series in Mathematics 92, American Mathematical Society.

<!-- turns -->
---

← [Stars and hubs](stars-and-hubs.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Bridges, cut risk, loops and the core: how a network can be cut](bridges.md) →

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (11)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [The starting ring](starting-ring.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Bridges, cut risk, loops and the core: how a network can be cut](bridges.md) · [Physical inspiration](physical-inspiration.md) · [The random twin: is a world a space?](random-twin.md) · [Three ways to measure a network's dimension](ball-dimension-and-curvature.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
