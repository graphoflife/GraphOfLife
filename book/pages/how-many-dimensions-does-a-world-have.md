# Is a world a space? Dimension, and what folds it

> [!question] Questions of this page
> - What does it mean for a network to have a dimension, and how is it
>   measured?
> - How many dimensions do the worlds have? Is the answer the same at every
>   scale, and by every ruler?
> - Is the dimension made by how the worlds are wired, or would any network
>   with the same numbers of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") have it?
> - Are [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s worlds, and the others the notebook has made since, spaces too?
> - When does a world become a space, and when does it fold into a small
>   world? Which rules decide it?

## In short

The notebook's second aim is a world that grows a space of its own, like a
plane or a room, from simple local rules. The worlds of the first [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."),
are spaces: about three dimensions by the room they hold, and two by how a
random walker spreads. Random networks with the same links have no dimension:
they are *small worlds*, in which everyone is a few steps from everyone. B2's
worlds start as spaces but fold into near small worlds once [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") die of age,
each death mended with a ring of links. Where newborns breed almost at once,
the dying hold fewer links, and a world stays a space of about three
dimensions, though a poor one.

## How many dimensions does a world have?

A network has no space around it. Its agents have no coordinates, only
connections. And yet a network can be shaped like a line, a sheet or a solid.
Or it can be like nothing of the kind: a tangle in which everyone is a few
steps from everyone. Such a tangle is called a **small world**. A space is the
opposite. Far places are far apart in it, and the farther out you look, the
more room there is.

**Why it matters.** Graph of Life was built with physics in mind
([Physical inspiration](physical-inspiration.md)). Several attempts to unite
gravity with quantum theory build space itself out of discrete pieces and the
relations between them. A world whose simple, local rules grow a space of a
few dimensions, three if possible, would be a small model of that idea. No
geometry would have been put in by hand. This is the notebook's second aim,
beside [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), and the two aims meet. In a space, distant
regions are far apart, so they could evolve apart. In a small world,
everything is near everything.

This half of the page measures what the worlds of the first baseline, B1, have
grown into. They are the thirty worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") of
[Thirty worlds](a-worlds-life.md#thirty-worlds), about 13,800 agents each at
their end. Besides them, it measures the three worlds of 409,600 tokens of
[Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md)
(`B1-409600-s001` … `-s003`, `python3 gol_lab.py run E08`). They held about
60,000 agents each after 600 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). And it measures the worlds of 10,000
tokens it was first written from: the 26 of thirty that lived to the end. They
held about 1,500 agents each at their end (1,300 over their [settled life](median-and-quantiles.md#the-settled-life-of-a-world "A world from iteration 500 to its end, after its youth, where it is measured.")).
That makes three sizes, each four to ten times the one before.
[How this was made](#how-this-was-made) lists every run.

### Two rulers

**The first ruler counts room.** In a space of *d* dimensions, the number of
points within a distance *r* of a point grows like *r*^*d*. It grows like 2*r*
on a line, like π*r*² in a plane, like (4/3)π*r*³ in a solid. On a network,
distance is the number of steps, and the same count can be made. Count the
agents within *r* steps of an agent: its **ball** of radius *r*. If that
number *N*(*r*) grows like *r*^*d*, the network has dimension *d* at those
scales. Read between two neighbouring radii,

$$
d(r) = \frac{\ln N(r+1) - \ln N(r)}{\ln (r+1) - \ln r}
$$

is the **local dimension** at radius *r*. In words: from *r* to *r* + 1 steps,
the ball grows as fast as *r* to the power *d*(*r*) would. If *N*(*r*) were
exactly *r*^*d*, it would read *d* at every radius.
[Dimension and curvature from ball growth](ball-dimension-and-curvature.md)
explains the idea in full.

**The second ruler asks a walker.** A walker that steps at random through a
space of *d* dimensions has, after *t* steps, spread over a region of about
*t*^(*d*/2) points. So the chance that it is back where it started falls like
*t*^(−*d*/2). The exponent read off that chance is the **spectral dimension**
([The spectral dimension](ball-dimension-and-curvature.md#the-spectral-dimension)). It measures how easily
things spread through a space, where the first ruler measures how much room
there is.

On a regular grid the two rulers agree. On irregular spaces they need not, and
how they differ says what kind of space it is.

### The rulers, tried on known spaces

Before measuring anything unknown, a ruler is tried on things whose dimension
is known. Here: five networks of about 2,000 points, a little larger than a
world of 10,000 tokens. That is the smallest of the three sizes, and the
hardest for the rulers.

<!-- figure dimensions/rulers -->
![Two rulers, tried on spaces whose dimension is known](../figures/dimensions/rulers.svg)

**Two rulers, tried on spaces whose dimension is known.** Five networks of about 2,000 points each: a ring, a square grid and a cubic grid, each wrapped round so that it has no edge (dimension 1, 2 and 3); a tree drawn at random among all trees on 2,000 points; and a random network in which every point has three neighbours. Left: the local dimension from how balls grow, d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), where N(r) is the typical number of points within r steps (the geometric mean over 40 points drawn at random), until a ball holds a quarter of the network. Right: the spectral dimension from a random walk, d(t) = −2 · ln(P(2t)/P(t)) / ln 2, where P(t) is how much more often than at rest a walk is back at its start after t steps (averaged over 24 starts), until P(2t) falls below 5/n.

> [!example]- How to make this figure
> **Runs.** No runs: the five networks are built by `book_chapters.space` (`ring`, `torus`, `tree` and `regular`, the last two with networkx, seed 23).
>
> **Data.** From the networks themselves.
>
> 1. Balls: a breadth-first search out of each of 40 points drawn at random (seed 23); N(r) the geometric mean, over the 40, of the number within r steps.
> 2. Walk: a lazy random walk — stay with probability ½, else step to a neighbour drawn at random — started at each of 24 points, its distribution propagated exactly step by step; P(t) is the probability of being at the start minus the probability of being there at rest (the start's connections over twice all connections).
> 3. Read d(r) and d(t) as in the caption; t runs over a grid of ratio √2.
>
> **To make it again:** `python3 book_figures.py dimensions`.
<!-- /figure -->

- The **ring** reads 1 by both rulers. The **square grid** reads 1.9 by balls
  and 2.1 by walks: 2.
- The **cubic grid** reads 2.6 by balls and 3.2 by walks. At 2,000 points a
  cube is only 13 points along each edge, and both rulers feel it. Balls fill
  a quarter of it by radius 7, before *N*(*r*) has reached its *r*³ growth,
  and walkers come round the edges. The rulers are least reliable where they
  are most wanted, at three dimensions in a world of few agents. The large
  worlds below are needed for that.
- A **tree** drawn at random reads 1.7 by balls and 1.2 by walks. Theory says
  2 and 4/3 (Aldous 1991; Durhuus, Jonsson and Wheater 2007). A tree has room,
  but a walker keeps running into its dead ends.
- A **random network**, in which every point has three [neighbours](the-world.md#the-network "An agent joined to another by a connection."), has no
  finite dimension. Its readings never settle: by balls they climb past 4
  within six steps, by walks past 4 within forty.

The lesson to carry on: **a finite dimension shows as a plateau**, a reading
that holds steady over a range of scales. A reading that keeps rising is the
mark of a network that is not a space at all.

### The rulers, on the worlds

<!-- figure dimensions/worlds -->
![The same rulers, on the worlds](../figures/dimensions/worlds.svg)

**The same rulers, on the worlds.** The two rulers of the figure above, on the largest piece of each world after its last game. Blue: the 30 worlds of 100,000 tokens (about 13,800 agents each) — the line is the median of the worlds, the darker band the middle half and the paler band nine in ten. Red: for each, its random twin, a network with exactly as many connections at every agent but wired at random. Yellow: the three worlds of 409,600 tokens of [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) (about 60,000 agents each), measured from 100 balls and 48 walks of up to 1,024 steps; dashed orange, their random twins. Green dots: the median of the 26 worlds of 10,000 tokens that lived to the end, read the same way.

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Thirty worlds](a-worlds-life.md#thirty-worlds); `python3 gol_lab.py run E10`). The three worlds of 409,600 tokens of Experiment 8 (`B1-409600-s001` … `-s003`, [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md); `python3 gol_lab.py run E08`), after their last game. The 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, [A world's life](a-worlds-life.md); `python3 gol_lab.py run E02`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read each run's last frame; build its network from `edges` and keep its largest piece.
> 2. The rulers as in the figure above (`book_chapters.space.measure`); the twin by `book_chapters.structure.degree_preserving` (seed 23).
>
> **To make it again:** `python3 book_figures.py dimensions`.
<!-- /figure -->

**The worlds of 100,000 tokens** (blue) read **2.8** by balls, between radii 3
and 10 in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world. By walks they read **1.9**, flat from 4 to 160
steps. Their reading by balls rises to about 3 at radius 5. It holds there
until the balls near a quarter of the world, at about 12 steps. **The worlds
of 10,000 tokens** (green dots) read less: 2.3 by balls between radii 2 and 6,
and 1.8 by walks. A world of 1,500 agents gives the balls little room. In the
median world a quarter of it lies within about seven steps, where the reading
stops.

**Their [random twins](random-twin.md)** (red) are the decisive comparison. A
world's random twin has the same agents, each with exactly as many connections
as in the world, but wired at random. If the dimension came from the numbers
of connections, the twins would read the same. They do not. By balls they read
6.5 and are still climbing when a quarter of the twin is reached, after only
three or four steps. By walks they climb past 3 and keep going. A twin's balls
grow by a factor at every step, faster than any power of *r*. The twins are
small worlds without a dimension; the worlds are not.

**The worlds of 409,600 tokens** (yellow) are four times larger again, and
here the rulers have the most room. By balls, two of the three read **3.3**,
flat from radius 3 to radius 14; the third wanders between 2.1 and 3.3. By
walks, all three stay between 1.5 and 2.3 from 4 steps to 400, with medians of
**1.8 to 1.9**. In one world the reading stays between 1.8 and 2.1 at every
scale from 6 steps to 724, two decades. Their twins read 8 by balls within
four steps, and climb past 6 by walks.

<!-- figure dimensions/growth -->
![How a large world fills out](../figures/dimensions/growth.svg)

**How a large world fills out.** The typical number of agents within r steps (the geometric mean over 100 agents drawn at random) in the three worlds of 409,600 tokens after their last game (yellow), and in their random twins (dashed orange), on logarithmic axes, until the ball holds the whole world. In a space of dimension d, the number grows like r^d, a straight line of slope d: in grey, r² (dotted, the shallower) and r³ (dashed, the steeper), drawn through the worlds' typical ball of radius 4.

> [!example]- How to make this figure
> **Runs.** The three worlds of 409,600 tokens of Experiment 8 (`B1-409600-s001` … `-s003`, [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md); `python3 gol_lab.py run E08`), after their last game.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read each run's last frame; build its network from `edges` and keep its largest piece; make its random twin as above.
> 2. From each of 100 agents drawn at random (seed 23), a breadth-first search; N(r) the geometric mean of the number within r steps.
>
> **To make it again:** `python3 book_figures.py dimensions`.
<!-- /figure -->

Drawn directly, the count of agents within *r* steps of the largest worlds
runs along a straight line of slope about 3.3 on logarithmic axes. From radius
3 to about 15 it is steeper than *r*³, and far steeper than *r*². Then it
bends, as the balls reach the edges of the world, around 60,000 agents. The
twins' counts shoot up and hold the whole world within seven steps.

**The three sizes at the same scales.** Read over the same radii, 3 to 6, and
the same walks, 4 to 64 steps, the three sizes can be set side by side. The
worlds of 10,000, 100,000 and 409,600 tokens read 2.4, 2.9 and 3.3 by balls,
and 1.8, 1.8 and 1.9 by walks. The larger the world, the more room it holds at
the same few steps, and the longer its plateau. The walker's reading hardly
changes. Whether the reading by balls goes on rising with the size of a world,
or settles near 3.3, these three sizes cannot tell.

**The viewer's own dimensions.** The Graph of Life viewer records a dimension
of its own. In these worlds it is about 3.4, and 3.0 at 10,000 tokens
([The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world)).
It is fitted to the **mean** ball over radii 1 to 5. The mean is pulled up by
the few agents next to a [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."). The rulers here use the **typical** ball over 40
to 100 agents. That is the geometric mean of their balls: the mean taken on a
logarithmic scale, which a few huge balls cannot pull up. The typical ball
grows more slowly at small radii. The two answer slightly different questions:
how much lies around the average agent, and how much around a typical one.

The viewer also records a **box dimension**. It covers a world with boxes,
each holding the agents within a few steps of its centre, and counts how many
it takes. In *d* dimensions, boxes half as wide take 2^*d* times as many
([Box dimension](ball-dimension-and-curvature.md#box-dimension)). It reads 1.88 in the median world, for at
large sizes it counts the long, thin branches
([Four measures of a world's shape](what-shape-does-the-network-take.md#four-measures-of-a-worlds-shape)).

### Two dimensions that disagree

So the worlds have a dimension, and it depends on the ruler. It is about **3**
by the room they hold, and about **2** by how a walker spreads through them.
On a grid the two would agree. Where they disagree, the space is irregular in
a particular way: it has room, but the room is hard to get around in. Dead
ends trap a walker, and narrow passages slow it.

The ratio says how much. A walker's typical distance after *t* steps grows
like *t* to the power *d*ₛ / 2*d*_ball. Here *d*ₛ is the walker's dimension
and *d*_ball the balls'. That is about 1.9 / 5.6 ≈ 0.34 in these worlds, and
1.9 / 6.6 ≈ 0.29 in the largest. On a grid it is 0.5. Physicists call such
slowed spreading **anomalous diffusion**. It is what is found on fractals,
shapes with holes at every scale. An example is the clusters of percolation:
fill the sites of a grid at random, just densely enough for one cluster to
reach across it. In three dimensions such clusters have a ball dimension of
about 2.5 and a spectral dimension close to 4/3 (Alexander and Orbach 1982).

Where are the dead ends? Prune every agent with a single connection, again and
again until none is left. What remains is the **core** of a world, its 2-core
([Core, trees and leaves](bridges.md#core-trees-and-leaves)), more than half of its
agents. In these worlds the core reads 2.5 by balls and 1.9 by walks, against
2.8 and 1.9 for the whole world. The trees that hang off the core add room,
not ways round. The core of the worlds of 409,600 tokens reads 3.1 by balls
and 2.0 by walks at short range. That is over radii 3 to 6, and walks of up to
64 steps. Over the whole range of scales it reads 2.7 and 1.5. Its loops close
at short range, and over long distances it seems to branch more than it
meshes.
[The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world)
takes the core apart.

This matters beyond geometry. In the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), tokens move almost as a random walk
would ([Where tokens flow](where-the-tokens-flow.md)). On these worlds that
walk is slow. A difference in tokens spreads through a world more slowly than
through a grid of the same size. It spreads much more slowly than through a
random network. Lines of descent spread in the same way
([Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)).
Two agents up to about 7 steps apart share a family of 25 iterations back more
often than chance. For a family of 200 iterations back, the excess over chance
reaches about 12 steps. The patches widen as a random walk on these worlds
would.

### Over a world's life

<!-- figure dimensions/life -->
![Dimension over a world's life](../figures/dimensions/life.svg)

**Dimension over a world's life.** The two rulers, each read at one scale, after the games of iterations 10, 25, 50, 100, 200, 300, 500, 750, 1,000 and 1,499 of the 30 worlds, on a logarithmic axis of time. Left: the local dimension from balls between radius 2 and 3; right: the spectral dimension from walks of 4 to 8 steps. The line is the median of the worlds, the darker band the middle half of them and the paler band nine in ten. The viewer's own dimensions, every 25 iterations, are drawn in [The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world).

> [!example]- How to make this figure
> **Runs.** The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Thirty worlds](a-worlds-life.md#thirty-worlds); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frame 2·t + 1 for each t; take the largest piece of its network.
> 2. The rulers as in the first figure, the walk run for 128 steps (`book_chapters.space.life_dimensions`); read the ball ruler at r = √6 and the walk at t = √32.
>
> **To make it again:** `python3 book_figures.py dimensions`.
<!-- /figure -->

Read each ruler at one small scale: balls between radius 2 and 3, and walks of
4 to 8 steps. Then the dimension of a world rises through its youth, and
holds. By balls the median world reads 1.8 at iteration 10, 2.6 by iteration
50, and between 2.4 and 2.7 from then to the end. By walks it reads 1.6 at
iteration 10, 1.9 by iteration 50, and between 1.85 and 1.97 after. A B1 world
takes its shape in its first fifty iterations of growth. It keeps it through
1,500 iterations of births, deaths and conquests. At 10,000 tokens the same
small scales are a larger part of a world. There the readings were about 2 by
balls, and 1.6 to 1.9 by walks, throughout. A world of the second baseline,
below, does not keep its shape.

## What folds a world

B1's worlds are spaces, but full of dead ends. What rule would make the two
rulers agree? Perhaps rules that close loops at long range, or that let dead
ends wither, or that give every place the same number of neighbours.

The second baseline, B2, has rules that close loops and leave no dead ends.
They were made to keep a world from breaking
([Building B2: a world that cannot break](a-world-that-cannot-break.md)), and
B2 has been the baseline of everything since:

- a newborn is joined to its parent and to two of its parent's neighbours;
- when an agent dies, its former neighbours are joined up in a ring round the
  gap, so that no death can split the world. This is **mending**
  ([`mend_on_death`](settings.md#mend_on_death));
- an unused link lapses only where its two ends share two neighbours;
- a [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") may choose at most 8 links.

In a model without brains, the first two rules alone put 97% of agents in the
core, with a ball dimension of 3.7. In a first run of B2 with all four, every
agent was in the core: no dead ends at all. Its ball dimension read 3.1 to
3.7.

But those readings are of the kind the health bar uses. The
[health bar](health-bar.md) is seven marks a healthy network must meet. One of
them is a dimension, read off the mean ball over radii 1 to 5. Until this
section, B2's network had been read only through that one number. And it rose
with a world's size
([What holds a world's size](what-holds-a-worlds-size.md)), as a small world's
would. A small world has no dimension to find; the number read off its first
few steps only grows with the room it has.

So this section reads B2's worlds, and the others the notebook has made since,
with the full rulers above (`research/atlas/space.py`). It adds one number
that sums a world up, its **width**. A world's typical distance is the number
of steps within which a typical agent's ball holds half the world. Its width
is its typical distance over its random twin's; the figures call it "typical
distance, over the twin's". A small world is about as near as its twin, a
width of about 1. A space is wider, because its distances grow as a power of
its size, the twin's only as the logarithm. The notebook counts a world as a
space when its width is at least 1.5 ([The random twin](random-twin.md)).

### B1 and B2 side by side

<!-- figure spaces/balls -->
![B1 and B2 side by side](../figures/spaces/balls.svg)

**B1 and B2 side by side.** Left: the share of its world a typical agent's ball holds within r steps (the geometric mean over 40 agents drawn at random), the median of each kind of world, on logarithmic axes; dashed, the same for the worlds' random twins, which have exactly the links of their world at every agent, wired at random. Right: the local dimension the balls show, d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), until a ball holds a quarter of the world, the median of each kind; dashed, the twins.

> [!example]- How to make this figure
> **Runs.** B1: the 26 worlds of 10,000 tokens of Experiment 2 that lived to the end, after their last game; B2: Experiment 16's ten worlds at iteration 2,000; mixed at birth: Experiment 17's worlds at 10,000; the estates to the newborns or the parents: batch R's B2 worlds (`R/B2-newborns-s1`, `-s2`, `R/B2-parents-s1`, `-s2`) at 2,000.
>
> **Data.** From each world's whole world kept then (`research/atlas/space.py`, `GraphOfLifeRuns/.pilots/atlas/space.json`).
>
> 1. Build the network of the world's largest piece; make its random twin.
> 2. From 40 agents drawn at random, a breadth-first search; N(r) the geometric mean of the number within r steps.
> 3. Read the share N(r)/n and the local dimension d(r).
>
> **To make it again:** `python3 book_figures.py spaces`.
<!-- /figure -->

- **B1 is a space**, as
  [the first half of this page](#how-many-dimensions-does-a-world-have) found.
  In its 26 worlds of 10,000 tokens, the local dimension holds between 2.0 and
  2.5 from radius 1 to 6. A typical agent needs 10.5 steps to reach half of
  its world, where its twin needs 4.5: a width of 2.3.
- **B2 is nearly a small world.** Its local dimension climbs (2.4, 3.9, 5.0)
  until, four steps out, a typical agent's ball already holds a quarter of the
  world's 3,400 agents. It needs 5.0 steps to reach half the world, its twin
  4.2: a width of 1.18. Each step multiplies its ball by four to six, its
  twin's by five to seven.
- **Brains mixed at birth do not change that.** Experiment 17's worlds read
  1.21, at 630 agents. There each newborn's brain is mixed from its parent's
  and a neighbour's
  ([Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md)).
- **But B2 is a space, and one of about three dimensions, when the [estate](estate.md "The tokens a dead agent leaves, and who gets them.") of
  the dead goes to the newborns or to the parents.** The estate is the tokens
  a dead agent leaves. Here it goes to the iteration's newborns, or to their
  parents ([Who inherits?](who-inherits.md)). These worlds' local dimension
  rises to 3.1 at radius 3 and 3.5 at 4 to 5, then falls to 2.7 by radius 7. A
  typical agent needs 11.2 steps to reach half of its 5,200 agents, its twin
  4.7: a width of 2.4, as wide as B1. The twins' dimension climbs past 5.7
  within four steps.

### A world nine times larger

A small world's rulers run out of room within a few steps. And a space only
shows its dimension where it has room
([The rulers, on the worlds](#the-rulers-on-the-worlds)). So a B2 world of
100,000 tokens was played for 1,000 iterations (a [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds."), [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1,
`GraphOfLifeRuns/.pilots/BIG/B2-100000-s1`). At the end it held 30,686 agents,
nine times B2's usual 3,400.

<!-- figure spaces/size -->
![Does a world fold the same way at every size?](../figures/spaces/size.svg)

**Does a world fold the same way at every size?** Each dot is a world: its typical distance over its random twin's, against the agents in its largest piece, both on logarithmic axes. Blue: B1's worlds of 10,000, 100,000 and 409,600 tokens after their last game ([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)'s readings). Grey: B2's worlds of 5,000 tokens at iteration 1,000 (Experiment 14) and of 10,000 at 2,000 (Experiment 16), and one B2 world of 100,000 tokens at iteration 1,000 (a pilot). A space grows wider than its twin as it grows; a small world stays as near.

> [!example]- How to make this figure
> **Runs.** B1: Experiments 2, 10 and 8 (`B1-10000-…`, `B1-100000-…`, `B1-409600-…`); B2: Experiments 14 and 16, and the pilot world `GraphOfLifeRuns/.pilots/BIG/B2-100000-s1` (B2's settings at 100,000 tokens, seed 1, 1,000 iterations).
>
> **Data.** From [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)'s readings (`GraphOfLifeRuns/.book/<run>.dims2.json`) and `research/atlas/space.py` (`GraphOfLifeRuns/.pilots/atlas/space.json`).
>
> 1. Measure each world and its random twin as in the first figure.
> 2. Divide the radius at which its typical ball holds half the world by its twin's.
>
> **To make it again:** `python3 book_figures.py spaces`.
<!-- /figure -->

**It is folded as B2 at 10,000 tokens is.** A typical agent's ball holds half
of its 30,686 agents within 5.5 steps, its twin's within 4.8: a width of 1.14.
Its local dimension climbs from 2.9 to 5.5 to 7.1 in three steps, as its
twin's does (3.1, 5.8, 7.8), with no plateau. B1's width grows with its size,
as a space's must: 2.3 at 1,500 agents, 3.7 at 13,800, 5.1 at 64,000. B2's
does not: 1.24 at 1,600 agents (5,000 tokens), 1.18 at 3,400, 1.14 at 30,700.
**At every size measured, B2 is a small world.**

### A world through its life

Every one of these worlds starts the same way. In B2 every agent dies at age
100, so until the first agents reach a hundred, nobody dies. The first hundred
iterations of a B2 world are the same whatever its rule for estates or for
age. How does a world become what it is?

<!-- figure spaces/life -->
![A world through its life](../figures/spaces/life.svg)

**A world through its life.** The typical distance of a world — the steps within which a typical agent's ball holds half of it — over the same for its random twin, at moments of its life; the median of two or three seeds of each kind. About 1: a small world, as near as its twin. Above: a space, wider than its twin. Above, worlds of 10,000 tokens: B2 itself (Experiment 16), without old age, with the estate of the dead to the newborns or to the parents (batch R), and with every unused link lapsing (batch S). Below, worlds of 5,000 tokens from Experiment 14, each without one of B2's rules, or with handover.

> [!example]- How to make this figure
> **Runs.** Experiment 16's B2 worlds and their sister experiment without old age (`B2-10000-s001` … `s003`, `B2-10000-3ec24f-s001` … `s003`), batch R's and batch S's pilot worlds, and Experiment 14's worlds at 5,000 tokens (`B2-5000-…-s001` … `s003`).
>
> **Data.** From the whole worlds kept every 100 iterations (`research/atlas/space.py --life`, `GraphOfLifeRuns/.pilots/atlas/space_life.json`).
>
> 1. Measure each world as in the figure above, at each moment.
> 2. Divide the radius at which its typical ball holds half the world by its twin's.
>
> **To make it again:** `python3 book_figures.py spaces`.
<!-- /figure -->

- **Every B2 world starts as a space**: at iteration 100 its width is 1.68.
  Birth makes space. A newborn is joined to its parent and to two of the
  parent's neighbours, next to where it was born.
- **B2 folds when its agents start to die of age.** From iteration 100 on, the
  first generation dies, and by iteration 300 the width has fallen to 1.24. It
  stays near 1.2 for as long as the world is played.
- **Without old age, nothing folds**: the world keeps the space it was born
  with, 1.8 to 1.9.
- **With the estate to the newborns or to the parents, the dying build space
  instead.** The width rises to 2.0 to 2.2 with the estate to the newborns,
  and to 2.4 to 2.6 with the estate to the parents.
- **With every unused link lapsing, the world opens wide**: 4.3 to 4.7 times
  its twin's distance from iteration 200 on.

The lower panel holds the worlds of Experiment 14, at 5,000 tokens, each
without one of B2's rules
([A world that cannot break](a-world-that-cannot-break.md#a-world-that-cannot-break)).
B2 itself folds there too, from 1.54 to about 1.25. Without links at birth it
folds a little less (1.35 to 1.45), and without the cap on links a little more
(1.13 to 1.28). With handover, in which a parent passes some of its links to
its newborn, it holds at about 1.5. **Without mending, it does not fold**: 1.6
to 1.7 through iteration 700, and 2.0 at 1,000.

**How mending folds a world.** Mending joins up the neighbours of an agent
that dies: they close in a ring round the gap, as the edge of a hole closes.
Precisely, the dead agent's former neighbours are grouped by the links among
them. Then the groups are joined in a ring, each through its member with the
fewest links. Each death joins agents that were two steps apart through the
dead one. A world in which everyone dies at a hundred does so for every agent,
generation after generation.

One reading, consistent with everything above: **death with mending folds a
world, and birth unfolds it.** Without old age nobody dies, and nothing folds.
Without mending a death leaves a hole, and the world keeps its room. And where
every unused link lapses, the links mending makes are cut again as soon as
nobody stakes on them.

The worlds whose estate goes to the newborns or the parents mend as B2 does,
yet they unfold. What lets them is not settled here, but two differences point
one way:

- **Who breeds.** With an estate in hand, a newborn has its first child within
  a few iterations. At a birth, the median parent is two or three iterations
  old, against 18 and 28 in B2. Agents under 30 have nearly as many living
  children as the old: 0.74 to 0.78 each, against 0.30 to 0.38 in B2. The
  world grows as chains of short generations.
- **Who dies.** In B2 an agent gathers links all its life, from its children
  and from its neighbours' children. At 90 or older it holds 8.1 to 8.4 links,
  in 7.6 to 7.8 groups of neighbours not linked to one another. Its death
  closes a ring of about eight links among them. Where the estate goes to
  births, the old hold about six links in 5.4 groups.

The estate to an agent's own children reaches heirs of every age, and changes
neither. There parents are 26 iterations old at a birth, as in B2, and the
worlds fold as B2's do, to 1.18 and 1.19. The neighbours of the unfolded
worlds are also more alike in age: the [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of birth times across links
is +0.8, against +0.45 in B2. And their links per agent are a little fewer,
4.7 to 4.9 against 5.4 to 5.7. Which of these is the cause, worlds that differ
in so much cannot say; worlds without brains can
([With no brains at all](#with-no-brains-at-all)).

### Every world at iteration 1,000

<!-- figure spaces/every -->
![Every world at iteration 1,000](../figures/spaces/every.svg)

**Every world at iteration 1,000.** Each dot is a world of the notebook read at iteration 1,000 — the pilot batches, the search by elimination and the lab's B2 families at 10,000 tokens — its typical distance over its random twin's (logarithmic). Top, for the worlds whose agents die of age and are mended round: against the number of groups the neighbours of its agents within ten iterations of their death fall into — groups linked to one another directly, which mending joins in a ring. Middle: against its links per agent; bottom: against the share of its agents holding three tokens or fewer. Colours: what stands out in its rules.

> [!example]- How to make this figure
> **Runs.** Every world of `GraphOfLifeRuns/.pilots/atlas/at1000.json` (`research/atlas/at.py`): the pilot batches K, M, P, Q, R, S, T, U, W, X, Y and YB, the search by elimination's worlds (Z), and the lab's B2 families of Experiments 15 to 17.
>
> **Data.** From each world's whole world kept at iteration 1,000 (`research/atlas/space.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/space_at1000.json`).
>
> 1. Measure each world and its twin as above.
> 2. Split the neighbours of each agent within ten iterations of its death into groups linked directly (`research/atlas/rings.py`).
> 3. Set its typical distance over the twin's against the mean number of groups, its links per agent and its share of poor agents.
>
> **To make it again:** `python3 book_figures.py spaces`.
<!-- /figure -->

Read at iteration 1,000, 395 worlds fall into two groups. They are every pilot
batch, the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s B2 families at 10,000 tokens, and the first round of the
search by elimination. That search drew every setting at random. Round after
round, it kept the values that did not make worlds worse
([The search's best world, in new seeds](the-candidates-played-on.md#the-searchs-best-world-in-new-seeds)).

- **Folded**: the 235 worlds of B2 with other rules (a median width of 1.26)
  and the 83 with brains mixed at birth (1.26).
- **Spaces**: the 37 worlds whose estate goes to the newborns or the parents
  (2.22), and the 14 in which every unused link lapses (3.80). And the 26
  whose deaths are not mended, on B1's rules or drawn so by the search (2.30).

Across them all, the width goes with fewer links per agent: a rank correlation
of −0.58. That is the correlation of the worlds' ranks, not of their values.
The width goes hardly at all with how poor the agents are (+0.12), how many
they are (+0.07) or how many tokens each holds (−0.09). Being a space is not a
matter of poverty or size.

**The ring a death closes tells it best.** Take the 336 worlds whose agents
die of age and are mended round. For every agent within ten iterations of its
death, count the groups its neighbours fall into: the groups mending will join
in a ring (`research/atlas/rings.py`). The more groups, the more folded the
world (a rank correlation of −0.69):

| groups of neighbours of the agents about to die | worlds | width, the typical distance over the twin's (median) |
|---|---|---|
| fewer than 5 | 50 | 1.88 |
| 5 to 6 | 15 | 2.12 |
| 6 to 7 | 55 | 1.34 |
| 7 to 9 | 193 | 1.24 |
| 9 or more | 23 | 1.08 |

The most folded worlds are those whose children may be of any size, down to a
single token, where B2 asks for at least 10. There the parents are 60
iterations old at a birth (a median), and the dying hold 12 to 14 links in as
many groups. Where the old breed, they gather links till they die, and every
death closes a wide ring.

Experiment 14's worlds, at 5,000 tokens and not among the 336, follow the same
rule. B2 there leaves rings of 6.7 to 7.5 groups and reads 1.25 to 1.33. With
handover, the dying leave rings of 2 to 5 groups, and the world reads 1.50 to
1.84. Without the cap on links, the rings have 10 to 24 groups, and two of the
three worlds read 1.14. The third, at 2.40, is the one exception.

### With no brains at all

The worlds above differ in many things at once. To see whether who breeds is
enough, the rules of space were played alone, as in
[Space alone](a-world-that-cannot-break.md#space-alone). There are no brains
and no tokens. A newborn is joined to its parent and two of the parent's
neighbours, and every agent dies at age 100, as in B2. Births balance deaths
at about 1,500 agents. The one thing changed is who may have a child. It is
any agent, only the young (at the median age of the living or younger), or
only the old. The model is `research/oee/fold.py`, with five seeds of 600
iterations each:

| rules | width (median, and range) | groups of neighbours of the oldest tenth | worst cut by one agent's loss |
|---|---|---|---|
| deaths mended in a ring, anyone breeds | 1.16 (1.06 to 1.28) | 6.1 | 1% |
| deaths mended in a ring, only the young breed | 1.71 (1.50 to 2.05) | 3.8 | 3% |
| deaths mended in a ring, only the old breed | 1.02 (1.02 to 1.06) | 7.6 | 0.1% |
| deaths not mended, anyone breeds | 2.36 (2.09 to 2.52) | 1.2 | 37% |
| the model of [Space alone](a-world-that-cannot-break.md#space-alone): agents die by chance, mended in a ring | 1.24 (1.13 to 1.31) | 7.4 | 1.5% |
| only the pieces a death would split joined, anyone breeds | 5.1 to 5.6 (two seeds of five: beyond reach) | 1.9 | 47% |
| the same, only the young breed | 4.22 (3.62 to 4.83) | 2.0 | 48% |
| the same, only the old breed | beyond reach in all five | 1.8 | 46% |

With anyone breeding, the brainless world folds as B2 does. Let only the young
breed, and it stays a space. Let only the old breed, and it folds completely,
as the worlds whose children may be of any size do. **Who breeds decides it,
with nothing else changed**, through the ring a death closes: 3.8 groups where
the young breed, 7.6 where the old do. And mending is what keeps a world
whole. Without it the brainless world is a space, but the loss of one agent
can cut off a third of it. The cull, which removes every piece cut off from
the largest, leaves half as many agents.

**How much mending makes how much room.** The ring joins every group of a
dying agent's neighbours that are not linked directly, even where they stay
joined through others. The least mending that still keeps a world whole joins
only the pieces a death would split it into (`--mend-split` in the model).
After most deaths, that is none. With it the brainless world never breaks, and
never folds: it grows into a string. In seed 1 of each, the string is 126
steps from end to end where anyone breeds. It is 54 where the young do, and
175 where the old do. Its dimension is between 1.3 and 1.9 by the health bar's
measure. A typical agent's ball reaches half of it five times later than its
twin's, or not within 40 steps at all ("beyond reach" above). Not mended, a
world breaks; mended in a ring, it folds; mended as little as it must be, it
is a line. Three dimensions lie somewhere between, and the worlds whose young
do the breeding come nearest.

### With brains: only the young, or only the old

The same rule was then given to worlds with brains (pilot batch YB, a switch
of the pilots' prototype engine, `breeders`). Only agents at the median age of
the living or younger may have a child, or only those older than it. It was
played in B2, two seeds to 2,000 iterations, and in B2 with brains mixed at
birth. [Founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") are all of one age, so in a world started afresh nobody is
older than the median, and nobody would breed. So the old were given B2's
world of seed 1 at iteration 500, and played on from there, beside the young
from the same world.

| world | parents' age at a birth (median) | groups of neighbours of the dying | width |
|---|---|---|---|
| B2, seed 1, at 1,000 / 1,500 / 2,000 | 16 / 15 / 18 | 7.5 / 7.3 / 7.6 | 1.22 / 1.15 / 1.27 |
| from it at 500, only the old breed | 52 / 63 / 59 | 8.9 / 8.8 / 9.0 | 1.12 / 1.04 / 1.07 |
| from it at 500, only the young breed | 16 / 17 / 18 | 7.4 / 7.3 / 7.6 | 1.30 / 1.26 / 1.34 |
| B2 afresh, only the young, seeds 1 and 2, at 2,000 | 13; 14 | 6.7; 7.0 | 1.34; 1.23 (B2: 1.27; 1.17) |
| mixed at birth, only the young, at 2,000 | 19; 26 | 7.5; 7.2 | 1.30; 1.30 (mixing alone: 1.36; 1.21) |

**Let the old breed, and a world with brains folds as far as the brainless
one.** The parents are 52 to 63 iterations old, the dying hold 9.3 to 9.6
links, and the world reads 1.04 to 1.12. It is smaller as well (1,300 to 1,650
agents), and a little richer. In it 59% of the agents hold three tokens or
fewer, against 77% to 79%. **Let only the young breed, and almost nothing
changes**, because in B2 the young already do most of the breeding. B2's
parents are 15 to 28 iterations old at a birth, and the rule leaves them 13 to
26. The dying still hold about eight links, gathered from their neighbours'
children as much as from their own, and the worlds stay folded.

Two things unfold the worlds whose estate goes to the newborns or the parents.
Their parents are young, two or three iterations old, and their dying hold few
links. They hold about six, in 5.4 groups, where the table of rings above puts
the line between folded and not.

## What we learned

- **B1's worlds are spaces.** Their dimension shows as a plateau, from radius
  3 to 10–14 and over two decades of walking times. Their random twins, with
  the same connections per agent, have none. The dimension is made by how the
  worlds are wired. Every new connection is made inside a neighbourhood, and
  none ever brings two agents closer together
  ([How the network grows](how-agents-have-children.md#how-the-network-grows)).
  So a world can only grow as a body grows, outwards, never by shortcuts.
- **About three dimensions of room.** At 14,000 agents the worlds hold about
  as much within *r* steps as a three-dimensional body does, from 5 to 10
  steps. At 60,000 agents they hold somewhat more, 3.3, steady from 3 to 14
  steps. Their [diameters](path-length.md#radius-and-diameter "The longest of the shortest paths between two agents of a world.") grow as agents^0.30
  ([Distances grow like a power](how-does-a-worlds-size-follow-its-tokens.md#distances-grow-like-a-power)).
  It is the first sign that something like three-dimensional space can emerge
  from these rules, with no geometry put in.
- **But not our kind of space.** A walker reads only two dimensions. The
  worlds are full of dead ends and narrow passages, a fractal more than a
  solid. In our space, both rulers read 3 at every scale we can measure. In
  the causal dynamical triangulations of quantum gravity, the dimension
  depends on the scale (Ambjørn, Jurkiewicz and Loll 2005). Whether it does
  here, in still larger worlds, is for a page still to be written, *Dimensions
  at scale* ([the contents](../README.md#space-and-dimension)).
- **B2 is not a space**, at any size measured, up to 30,700 agents. Its worlds
  are nearly small worlds: a typical agent is about as near to everyone as in
  a random network with the same links. That is why the health bar's dimension
  rose with their size and failed the large ones.
- **Space is made and unmade by life and death.** Birth makes it, and every B2
  world starts as one. Death with mending folds it, once the first generation
  dies. The larger the ring each death closes, the more it folds, and the
  rings are largest where the old do the breeding. Worlds without old age,
  without mending, or whose unused links lapse keep their room. Without
  brains, who breeds is enough to fold a world or to keep it a space. With
  brains, letting only the old breed folds it further, and letting only the
  young breed changes little, for B2's young already do.
- **Room and resilience pull apart.** Mending keeps a death from splitting a
  world, and it is what folds it. Mended in a ring, a world folds into a near
  small world; mended only where it would split, it stretches into a line. A
  world that is a three-dimensional space, and that no single loss can cut,
  has to sit between the two.
- **The three-dimensional worlds are the poor ones.** B2 with the estate to
  the newborns or the parents is a space of about three dimensions, as the
  notebook's second aim asks. But nine in ten of its agents hold three tokens
  or fewer, its agents are alike, and its evolution stops
  ([Who inherits?](who-inherits.md)). A world that is a three-dimensional
  space and rich, and that keeps evolving, is still to be found
  ([Healthy and rich](healthy-and-rich.md)).
- **Room for open-ended evolution.** In a space, distant places are far apart.
  The diameter of a world grows as a power of its size, a small world's only
  as its logarithm. So a large world has room for regions that evolve apart.
  Lines of descent do hold patches, a dozen steps across. But what the agents
  hold, their tokens, ages and [genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), is alike only within three steps
  ([Like next to like](like-next-to-like.md)).
- **Open**: rules that would keep a world's dying from gathering wide rings in
  a rich world, or keep a world whole without folding it. Mending would add
  nothing where a dying agent's neighbours are already linked to one another,
  as on a surface of triangles. But B2's rule for unused links cuts exactly
  those triangles. Letting only the young breed is not enough with brains:
  B2's young already breed.

## Where to go next

- [The random twin: is a world a space?](random-twin.md): the width, and the
  line of 1.5 between a space and a folded world.
- [Building B2: a world that cannot break](a-world-that-cannot-break.md): the
  rules of B2, mending among them, and what each one does.
- [Who inherits?](who-inherits.md): the estates that keep a world a space, and
  why their worlds stay poor.
- [Physical inspiration](physical-inspiration.md): the physics behind the aim,
  and what else a world might grow.

## How this was made

Every figure of this page says, folded beneath it, which runs it reads and
how. To draw the figures again: `python3 book_figures.py dimensions` for the
first half of the page, and `python3 book_figures.py spaces` for the second.

The first half reads three sets of B1 worlds:

- the thirty of Experiment 10 (below);
- the three worlds of 409,600 tokens of Experiment 8
  (`python3 gol_lab.py run E08`);
- the worlds of 10,000 tokens of Experiment 2 (`python3 gol_lab.py run E02`,
  [The life of a B1 world](a-worlds-life.md)).

The second half measures each world and its random twin with
`research/atlas/space.py`, and counts the groups of a dying agent's neighbours
with `research/atlas/rings.py`. It reads Experiment 2's B1 worlds, Experiments
14 to 17 of the lab, and the pilot batches. And it reads one pilot world of
100,000 tokens on B2's settings, seed 1, 1,000 iterations
(`GraphOfLifeRuns/.pilots/BIG/B2-100000-s1`). That world was still being
played when What folds a world was first written, and its reading was added
the same day, 17 October 2026. The worlds without brains come from the model
of [Space alone](a-world-that-cannot-break.md#space-alone),
`research/oee/fold.py`.

### The runs of Experiment 10

Experiment 10 is 30 runs: B1 at 100,000 tokens, seeds 1 to 30, 1,500
iterations each.

<!-- runs E10 -->
> [!info]- The runs behind this page
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 1,500 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 100,000 tokens. Runs `B1-100000-s001` … `B1-100000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E10`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 100,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 30 | one run per seed |
> | iterations | 1,500 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## How our view changed

- **14 October 2026.** From 9 October, the first half of this page said that
  nothing held regions of a world apart.
  [Like next to like](like-next-to-like.md) had found likeness gone within
  three steps.
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)
  found that lines of descent do hold regions: patches of kin up to about 12
  steps wide ([Two dimensions that disagree](#two-dimensions-that-disagree)).
- **17 October 2026.** From 14 October, this page said that B2's rules leave
  no dead ends: every agent is in the core. It gave B2 a ball dimension of 3.1
  to 3.7. These were rules of the kind it had asked for, to make the two
  rulers agree. It asked whether the walker's reading would rise to meet the
  balls'. [What folds a world](#what-folds-a-world) found that B2's worlds are
  not spaces at all, but nearly small worlds, folded by deaths mended in a
  ring. The dimension read off their first few steps grows with a world's
  size.
- **17 October 2026.** Earlier that day, worlds without brains had shown that
  letting only the young breed keeps a world a space. What folds a world then
  named rules that let the young breed in a rich world as the way open. Pilot
  batch YB tried the rule with brains, and it changed little
  ([With brains: only the young, or only the old](#with-brains-only-the-young-or-only-the-old)).
  B2's young already do most of the breeding.
- **23 October 2026.** Since 9 October, this page had said that one world of
  409,600 tokens read 1.8 by walks at every scale from 6 steps to 724. Its
  readings (`book/results/dimensions.json`) are 2.1 near 6 steps, and fall to
  1.8 by about 30 steps. Over the two decades they stay between 1.8 and 2.1,
  as [The rulers, on the worlds](#the-rulers-on-the-worlds) now says.

<!-- turns -->
---

← [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [Space and dimension](../README.md#space-and-dimension) · [The random twin: is a world a space?](random-twin.md) →

**Also about:** [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world).

> [!info]- Pages that link here (25)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [How births build the network](how-agents-have-children.md) · [Where tokens flow](where-the-tokens-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Physical inspiration](physical-inspiration.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [The random twin: is a world a space?](random-twin.md) · [Three ways to measure a network's dimension](ball-dimension-and-curvature.md) · [What holds a world's size](what-holds-a-worlds-size.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Rules in company](rules-in-company.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [Who inherits?](who-inherits.md) · [Links to strangers](links-to-strangers.md) · [Healthy and rich](healthy-and-rich.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The health bar](health-bar.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [The candidates, played on and played again](the-candidates-played-on.md) · [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [A screen of random worlds](a-screen-of-random-worlds.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
