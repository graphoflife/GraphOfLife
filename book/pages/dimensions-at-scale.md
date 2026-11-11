# Dimensions at scale

> [!question] Questions of this page
> - Is a world's dimension the same over a few steps as over many, or does
>   it change with the scale, as it does in some models of quantum gravity?
> - How much of what the rulers read is the edge of a world of finite size?
> - Does a larger world read more at the same radius, and why?
> - How does a world's width grow with its [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), and which dimension does
>   that give?
> - What do the largest worlds hint at about larger ones, and how large would
>   a world have to be to settle it?

## In short

Within one world, [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s dimension changes with the scale no more than a
finite body's does. By balls it rises over the first few steps, holds (at
about 3.1 from radius 3 to 14 in the largest worlds, of 64,000 agents) and
falls at the world's edge, as in a ball cut out of a cubic grid; by walks it
reads 2 or less, from a few steps to several thousand. What changes with a
world's size is the reading at a fixed radius: worlds of 15,000 agents and
more read about half a dimension more over their first steps than worlds of
1,500, more than any cubic grid. An edge only ever lowers a reading, so the
cause is not the edge but bigger hubs, which bring agents close; above
15,000 agents the rise slows, but it has not been seen to stop. By its width
a world reads about four dimensions, and only worlds ten times larger, or
worlds whose make-up does not change with size, could tell a dimension that
grows at the largest scales from hubs that grow with the world.

## Does the dimension change with the scale?

Look at a ball of wool from across a room, and it is a point. Come closer,
and it is a solid ball, three-dimensional. Closer still, it is a tangle of
threads, each of them a line; and under a magnifying glass a thread is a
solid again. What dimension a thing has can depend on the scale at which you
look (Mandelbrot 1982). Every measured dimension is a dimension *at a scale*.

Physics takes this seriously. In the causal dynamical triangulations of
quantum gravity, space-time is glued together from small pieces by local
rules. A random walker in it reads four dimensions over long distances, but
only about two over the shortest (Ambjørn, Jurkiewicz and Loll 2005). Several
other approaches to quantum gravity find the same fall towards two at the
smallest scales (Carlip 2017). The worlds of Graph of Life grow a space of
their own from local rules ([Physical inspiration](physical-inspiration.md)),
so it is fair to ask the same of them.

[Is a world a space?](how-many-dimensions-does-a-world-have.md#two-rulers)
gave the notebook two rulers. The first counts **room**: the agents within *r*
steps of an agent grow like *r*^*d* in a space of *d* dimensions, and the
local dimension *d*(*r*) is read between neighbouring radii. The second asks a
**walker**: its chance of being back where it started falls like
*t*^(−*d*/2) after *t* steps, and the spectral dimension is read between *t*
and 2*t*. Both are dimensions at a scale: *d*(*r*) at a radius, the spectral
dimension at a length of walk. On B1's largest worlds, of about 60,000
agents, the balls of two of the three read **3.3** from radius 3 to 14, and
walks read 1.8 to 1.9 in the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") from 4 steps to 400
([The rulers, on the worlds](how-many-dimensions-does-a-world-have.md#the-rulers-on-the-worlds)).
Neither changed much with the scale.

But the size of a world did change them. Read over the same radii, 3 to 6,
worlds of 10,000, 100,000 and 409,600 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") read 2.4, 2.9 and 3.3. Three
explanations fit that rise, and each says something different about larger
worlds:

1. **The dimension grows with the scale.** A larger world lets the rulers
   look further, where there is more room, as in quantum gravity's
   triangulations. Then still larger worlds would read still more.
2. **The edge.** A finite world ends somewhere. A ball that reaches the end
   stops growing there, and a small world's balls reach it sooner. Then the
   readings would settle, as worlds grow, at the dimension they have.
3. **Larger worlds are made differently.** They have a few more links per
   agent ([Worlds of many sizes](b1-the-network-size-and-scaling.md#agents-and-connections-follow-the-tokens))
   and bigger hubs, and so more room within a few steps. Then the dimension
   belongs to the size.

This page tells them apart. It reads the B1 worlds of three experiments, at
eleven sizes from 130 agents to 70,000. And it reads bodies whose dimension is
known, which end at an edge as a world does, at the same sizes and with the
same code.

## The worlds, and the bodies beside them

**The worlds** are B1's, from three experiments:

- Experiment 8 ran the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") at ten sizes, from 800 to 409,600 tokens,
  three [seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") each, for 600 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")
  ([Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes)).
  Of its 30 worlds, 25 lived to the end: one of 1,600 tokens and every world
  of 3,200 tokens and more.
- Experiment 2's worlds of 10,000 tokens, 26 of which lived through their
  3,000 iterations ([The life of a B1 world](b1-lives-and-wealth.md#the-life-of-a-b1-world)).
- Experiment 10's thirty worlds of 100,000 tokens, 1,500 iterations each
  ([Worlds of 100,000 tokens](b1-the-network-size-and-scaling.md#worlds-of-100000-tokens)).

Each world is read in its largest piece, after its last [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), by both rulers:
balls grown all the way out from 100 agents drawn at random, and a lazy
random walk from 48 agents for up to 1,024 steps. Its **typical ball** at
radius *r* is the geometric mean of the hundred balls, the mean taken on a
logarithmic scale, which a few huge balls cannot pull up much; its
**typical distance** is the radius at which that ball holds half the world
([The typical distance](measuring-space.md#the-typical-distance)). Its random
twin, the same agents with as many links each but wired at random, is read
beside it.

A world's shape at large scales is restless. Its typical distance can
double or halve within a few hundred iterations, as games cut off parts of it
and lines sweep through it ([How a world breaks](b1-the-network-size-and-scaling.md#how-a-world-breaks)). One
look at a world is one draw from that restlessness. So each world's balls,
and its twin's, are also read at five moments spread evenly over its
[settled life](statistics.md#the-settled-life-of-a-world "A world from iteration 500 to its end, after its youth, where it is measured."),
from iteration 500 to its last, and the world is summed up by the median of
the five.

Many of the smaller worlds are **stars**, held together by one hub joined to
a tenth of all agents or more ([Stars and hubs](measuring-networks.md#stars-and-hubs)). Through
its hub nearly every agent of a star is two steps from every other, and there
is little room to measure. Five of Experiment 8's seven worlds below 10,000
tokens are stars, 10 of the 26 of Experiment 2 and 7 of the 30 of Experiment
10; of Experiment 8's eighteen worlds from 12,800 tokens up, one is. Stars are
drawn in grey and left out of every median and every fit.

## What an edge does to the rulers

A world is finite, and that alone bends the rulers. To see how, they are
tried first on bodies whose dimension is known and which end at an edge, as a
world does: a disc cut out of a square grid, and a ball cut out of a cubic
grid. Each is cut at 250, 500, 1,000 … 64,000 points, each twice the last,
about as many as the worlds hold.

<!-- figure scaledims/bodies -->
![What an edge does to the rulers](../figures/scaledims/bodies.svg)

**What an edge does to the rulers.** Bodies whose dimension is known and which end at an edge, as a world does: a disc cut out of a square grid (dimension 2) and a ball cut out of a cubic grid (dimension 3), of 250, 500, 1,000 … 64,000 points, each twice the last, from dark to bright, read by the same code as the worlds. Top: the local dimension from balls, until a typical ball holds a quarter of the body; dashed, the same grid with no edge at all, counted exactly. Bottom: the spectral dimension from a lazy random walk of up to 1,024 steps.

> [!example]- How to make this figure
> **Runs.** No runs: the bodies are built by `book_chapters.scaledims.body`.
>
> **Data.** From the bodies themselves.
>
> 1. Bodies: the n points of a square or cubic grid nearest its centre (and any as near as the last), each linked to the points beside it that are inside too (`book_chapters.scaledims.body`), for n = 250, 500, … 64,000; read as the worlds are. The endless grids' balls are counted exactly: Σₖ 2ᵏ·C(d, k)·C(r, k) points within r steps.
> 2. Balls and walks as for the worlds: 100 balls grown all the way out and 48 walks of up to 1,024 steps, from points drawn at random (seed 23).
>
> **To make it again:** `python3 book_figures.py scaledims`.
<!-- /figure -->

By balls, three things show.

- **Close in, the grid holds the reading down.** Within *r* steps of a point
  of an endless cubic grid lie 7, 25, 63 and 129 points for *r* = 1 to 4: an
  octahedron, which grows like *r*³ only slowly. So the endless grid reads
  2.49 between radius 3 and 4, 2.80 between 8 and 9 and 2.90 between 15 and
  16. It nears 3 only far out. A square grid nears 2 the same way, from
  below.
- **Far out, the edge holds it down.** A ball that reaches the edge grows no
  further there, and the reading falls. The ball of 64,000 points reads at
  most 2.65, between radius 8 and 11; the ball of 2,000 points at most 2.30,
  between radius 4 and 6, and a typical ball holds a quarter of it by radius
  8. The smaller the body, the lower and shorter its plateau.
- **No body reads more than its dimension.** Neither the grid nor its edge
  adds room. A reading above 3 cannot come from a three-dimensional body
  seen too small.

At one radius, the edge matters less the larger the body. Between radius 2
and 3 the ball reads 2.11 at 2,000 points, 2.20 at 16,000 and 2.22 at
64,000, against 2.28 with no edge. Between radius 4 and 5 it reads 2.30,
2.45 and 2.51, against 2.61. From 16,000 points on, a body's reading over
its first few steps hardly changes with its size: from 16,000 points to
64,000 it gains 0.02 between radius 2 and 3, and 0.06 between 4 and 5.

By walks, the bodies read their dimension sooner. The disc reads 1.85 to
2.1 from four steps on, at every size. The ball reads 2.8 to 3.1 between 6 and
16 steps, from 2,000 points on. Then the edge shows here too: a walker that
comes up against it is hemmed in and comes back more often, and the reading
sinks, in the ball of 64,000 points from 3.05 at about 8 steps to 2.7 at
500. A small body's walk ends sooner, once the walker is spread over all of
it. Over its first few steps a walk on a grid overshoots the grid's
dimension, by up to 0.08, and then never reads more.

## The rulers at every size

Read the same way as the bodies, B1's worlds of every size show the same
three parts by balls: a rise over the first two or three steps, a plateau,
and a fall once a typical ball nears a quarter of the world.

<!-- figure scaledims/rulers -->
![The two rulers, at every size](../figures/scaledims/rulers.svg)

**The two rulers, at every size.** The two rulers of [Is a world a space?](how-many-dimensions-does-a-world-have.md#two-rulers) on B1's worlds that are not stars, each line the median of the worlds of one size. From dark to bright: Experiment 8's worlds of 12,800 to 409,600 tokens (two or three of each size; below 12,800 tokens most of its worlds are stars). Grey: the 16 worlds of 10,000 tokens of Experiment 2 (dotted) and the 23 of 100,000 tokens of Experiment 10 (dashed). Left: the local dimension from balls, until a typical ball holds a quarter of its world, over the worlds and five moments of their settled lives. Right: the spectral dimension from a lazy random walk after their last game, of up to 1,024 steps, and of up to 8,192 in the worlds of 409,600 tokens, until it is back no more than 5/n more often than at rest.

> [!example]- How to make this figure
> **Runs.** The 25 baseline worlds of Experiment 8 that lived to iteration 600 — B1 at 1,600 to 409,600 tokens, seeds 1 to 3, of which one world of 1,600 tokens and all of 3,200 tokens and more (`B1-1600-s002`, `B1-3200-s001` … `B1-409600-s003`; [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E08`), after the game of iteration 599. The 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, [The life of a B1 world](b1-lives-and-wealth.md#the-life-of-a-b1-world); `python3 gol_lab.py run E02`). The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read the frame after each world's last game; build its network from `edges` and keep its largest piece (`book_chapters.space.largest_piece`). Its random twin: the same numbers of links per agent wired at random (`book_graph.degree_preserving`, seed 23), its largest piece.
> 2. Balls: a breadth-first search all the way out from each of 100 agents drawn at random (seed 23); N(r) the geometric mean of the numbers within r steps; d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), read while N(r+1) is less than a quarter of the piece (`book_chapters.scaledims.reach` and `ball_curve`).
> 3. Walks: a lazy random walk (stay with probability ½, else step to a neighbour drawn at random) from each of 48 agents drawn at random, its distribution propagated exactly, for up to 1,024 steps; P(t) the probability of being back at the start minus that of being there at rest (the walk is reversible, so the chance to be back after 2m steps follows from where it is after m, and half the steps are propagated); d(t) = −2 · ln(P(2t)/P(t)) / ln 2 on a grid of t of ratio √2, read while P(2t) is above 5/n (`walk` and `walk_curve`).
> 4. A world whose best-linked agent was joined, on average over iterations 500 to its end, to a tenth of its agents or more is a star ([Stars and hubs](measuring-networks.md#stars-and-hubs); `maxDegree`/`nodes` of its rows in `stats.jsonl`).
> 5. The same balls, of each world and of its twin, after the games of five iterations evenly spaced over its settled life, from 500 to its last (500, 525, 550, 574 and 599 in Experiment 8; 500, 750, 1,000, 1,249 and 1,499 in Experiment 10; 500, 1,125, 1,750, 2,374 and 2,999 in Experiment 2); each reading the median of the five (`book_chapters.scaledims.settled`).
> 6. In the three worlds of 409,600 tokens, the same 48 walks for up to 8,192 steps (`book_chapters.scaledims.long_reading`).
> 7. Each line: at every radius (or walk), the median over the worlds of that size (and, for the balls, their five moments) that read there, where at least half of them do.
>
> **To make it again:** `python3 book_figures.py scaledims`.
<!-- /figure -->

**The plateau grows longer with the size of a world, and higher.** Over five
moments of their settled lives, a typical ball holds a quarter of the median
world by radius 7 at 1,400 agents (Experiment 2), 13 at 14,600 (Experiment
10) and 21 at 64,000 (409,600 tokens): at every size, two thirds to three
quarters of the way to the world's typical distance, as in the bodies. Over
radii 3 to 6, the median world reads 2.37 at 1,400 agents, 2.95 at 14,600,
and 3.04 to 3.26 at 18,000 to 64,000. (At 64,000 agents that is 3.04 over
five moments of each world, where their last game alone gave the 3.3 of the
opening: one look at a world is one draw.) At 1,400 agents that is about what
a ball cut out of a cubic grid of the same size reads there, 2.30 at 2,000
points; from 14,000 agents on it is well above it, 2.45 at 16,000 points and
2.51 at 64,000.

**In the median, the plateau does not climb.** Over the three worlds of
409,600 tokens and five moments of each, the median reading is 2.98 between
radius 3 and 4, 3.24 between 9 and 11, and 3.01 between 13 and 14; then it
falls, as the balls near a quarter of the world. In the median world of
Experiment 10 it holds at 2.85 to 2.98 from radius 3 to 8, and then falls.
Single worlds differ, and the way they differ is telling.

- In the world of seed 3, the reading is lowest close in and climbs with
  the radius, at every moment: about 2.5 within eight steps, and up to 3.3
  to 4.4 beyond fourteen. That world has a hub of 1,700 to 3,200 links, with
  a seventh of all agents within four steps of it, and its median agent is
  19 steps away. A typical ball meets that crowd only some fifteen steps
  out, and then takes in thousands of agents at once. The climb is the hub,
  seen from afar.
- In the world of seed 2, whose best-linked agent has only 150 to 270 links,
  the reading falls from about 3 to below 2 beyond fifteen to twenty steps,
  along long branches. Its typical distance is 33 to 55 steps, the widest of
  the three.
- The world of seed 1, with a hub of 800 to 1,400 links, reads between 3.0
  and 3.8 from radius 3 to 12 at every moment, with no tilt.

A hub, or long branches, give a world a scale of its own. Neither is a
dimension that grows with the scale in every world.

**By walks, the worlds read about 2 at every size.** Between 4 and 64
steps, after their last game, the median world of each size reads 1.67 to
1.96, with no trend from 1,400 agents to 64,000: 1.71 in Experiment 2, 1.80
in Experiment 10 and 1.89 at 409,600 tokens. Beyond a few hundred steps the
worlds part: in the median world of 204,800 tokens the reading rises to
2.3, in those of 12,800 and 25,600 tokens it falls to about 1.6.

Walked eight times as long, to 8,192 steps, the worlds of 409,600 tokens
read no more. Between 700 and 5,800 steps they read 1.3 to 1.8, and their
median sinks from 1.74 to 1.48, below its 1.8 to 1.9 over the first hundred
steps. In these worlds a walker gets about *t*^0.29 steps from its start
([Two dimensions that disagree](how-many-dimensions-does-a-world-have.md#two-dimensions-that-disagree)):
some 7 steps in 724, and 12 in 5,800, as far out as the balls' plateau
reaches. Over the same distances, then, a walker reads under two where the
balls read three. The walker strays into dead ends and long branches, which
the balls count as room; in the world of long branches (seed 2) the walk
reads 1.3 to 1.5 beyond 700 steps.

**The twins have no plateau at any size.** A typical ball holds a quarter of
a twin within three or four steps, and on the way the twin reads 4.5 at
1,400 agents and 8.0 at 64,000, more the larger it is: the mark of a small
world ([The random twin](measuring-space.md#the-random-twin-is-a-world-a-space)).

## One radius, every size

Hold the radius fixed and let the size grow. An edge can only take room
away, and less of it the larger the body. So if the edge were what held the
smaller worlds down, their readings at a fixed radius would rise with size
only as a body's do, and stop where a body's stop.

<!-- figure scaledims/fixed -->
![One radius, every size](../figures/scaledims/fixed.svg)

**One radius, every size.** The local dimension from balls, read between the same two radii at every size, against the agents of a world's largest piece (logarithmic). Each dot is a world, its median over five moments of its settled life: blue, Experiment 8's worlds of 1,600 to 409,600 tokens; green, Experiment 2's worlds of 10,000 tokens; yellow, Experiment 10's of 100,000 tokens; grey, the stars among them. A world whose typical ball holds a quarter of it before the outer radius, at most of its moments, has no dot. Lines, read the same way at 250 to 64,000 points: red, the ball cut out of a cubic grid; violet, the disc cut out of a square grid. Dashed: the same grids with no edge.

> [!example]- How to make this figure
> **Runs.** The 25 baseline worlds of Experiment 8 that lived to iteration 600 — B1 at 1,600 to 409,600 tokens, seeds 1 to 3, of which one world of 1,600 tokens and all of 3,200 tokens and more (`B1-1600-s002`, `B1-3200-s001` … `B1-409600-s003`; [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E08`), after the game of iteration 599. The 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, [The life of a B1 world](b1-lives-and-wealth.md#the-life-of-a-b1-world); `python3 gol_lab.py run E02`). The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read the frame after each world's last game; build its network from `edges` and keep its largest piece (`book_chapters.space.largest_piece`). Its random twin: the same numbers of links per agent wired at random (`book_graph.degree_preserving`, seed 23), its largest piece.
> 2. Balls: a breadth-first search all the way out from each of 100 agents drawn at random (seed 23); N(r) the geometric mean of the numbers within r steps; d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), read while N(r+1) is less than a quarter of the piece (`book_chapters.scaledims.reach` and `ball_curve`).
> 3. A world whose best-linked agent was joined, on average over iterations 500 to its end, to a tenth of its agents or more is a star ([Stars and hubs](measuring-networks.md#stars-and-hubs); `maxDegree`/`nodes` of its rows in `stats.jsonl`).
> 4. The same balls, of each world and of its twin, after the games of five iterations evenly spaced over its settled life, from 500 to its last (500, 525, 550, 574 and 599 in Experiment 8; 500, 750, 1,000, 1,249 and 1,499 in Experiment 10; 500, 1,125, 1,750, 2,374 and 2,999 in Experiment 2); each reading the median of the five (`book_chapters.scaledims.settled`).
> 5. Bodies: the n points of a square or cubic grid nearest its centre (and any as near as the last), each linked to the points beside it that are inside too (`book_chapters.scaledims.body`), for n = 250, 500, … 64,000; read as the worlds are. The endless grids' balls are counted exactly: Σₖ 2ᵏ·C(d, k)·C(r, k) points within r steps.
>
> **To make it again:** `python3 book_figures.py scaledims`.
<!-- /figure -->

They do not. Between radius 2 and 3, the worlds of up to 5,000 agents read
1.8 to 2.4 (the median of each size), about what a ball cut out of a cubic
grid of their size reads, 2.1 to 2.2. From 14,000 agents on they read 2.5 to
2.8: more than any cubic grid reads there, even one with no edge at all
(2.28). The bodies' readings rise by a tenth over the whole range of sizes,
and hardly at all from 16,000 points on. The worlds' rise by three to six
tenths, most of it between 5,000 and 15,000 agents, past anything a cubic
grid reads. Between radius 4 and 5, the worlds read 2.2 to 2.9
below 5,000 agents and 2.9 to 3.3 from 14,000 on, against 2.30 to 2.51 for
the balls of a grid and 2.61 for the endless grid. Between radius 8 and 9,
which only worlds of 3,000 agents and more reach, they read 2.3 to 3.0,
against 2.3 to 2.65 for the balls of a grid and 2.80 with no edge.

So the edge is not what lifts the larger worlds. What does? Within one size,
the worlds that read more over their first few steps are those with bigger
hubs or more links. Across Experiment 10's 23 worlds that are not stars, the
rank [correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes "Whether two quantities go up and down together, from −1 to 1.") of the reading over radii 3 to 6 with the links of the
best-linked agent is +0.61; across Experiment 2's 16, with the links per
agent, +0.40. And the larger worlds have both. The best-linked agent of the
median world has 60 links at 1,400 agents, 410 at 14,600 and 1,215 at
64,000, and the links per agent rise from 3.2 to 3.6.

A hub of a thousand links puts a thousand agents two steps from one another.
It adds room close in, as the links of a random network do, and it shortens
the distances. So the worlds that read most over their first steps are the
narrowest: across Experiment 10's worlds, the reading over radii 3 to 6 and
the typical distance go against each other, at −0.68.

One more thing makes a single reading unreliable. A world's shape changes
from one moment to the next. Between the five moments of its settled life,
the median world's reading over radii 3 to 6 varies by 0.8, and its typical
distance by a factor of 1.8; in a quarter of the worlds by more than 2.4,
and in one by 6. That is why each world is summed up by the median of five.

## How wide a world is

A third ruler needs no radius at all. Pack *N* points into a body of *d*
dimensions, and it is about *N*^(1/*d*) points across: a cube of a thousand
points is ten wide, a square of a thousand about 32. So how a world's width
grows with its agents is a dimension too, read at the largest scale there is,
the whole world ([Distances grow like a power](b1-the-network-size-and-scaling.md#distances-grow-like-a-power)).
Here the width is the typical distance.

<!-- figure scaledims/width -->
![How wide a world is, at every size](../figures/scaledims/width.svg)

**How wide a world is, at every size.** The typical distance of each world — the number of steps within which a typical agent's ball holds half of the world's largest piece — against the agents in that piece, both the medians over five moments of its settled life, on logarithmic axes. Blue: Experiment 8's worlds of 1,600 to 409,600 tokens, and dashed, the power law fitted to those of 3,200 tokens and more that are not stars, agents^0.20 (95% interval 0.15 to 0.27); green: Experiment 2's of 10,000 tokens; yellow: Experiment 10's of 100,000 tokens; grey: the stars among them; orange: the random twins of the worlds that are not stars. Lines: the disc and the ball cut out of a grid. On these axes a body of d dimensions is a line of slope 1/d, and a random network a curve that flattens.

> [!example]- How to make this figure
> **Runs.** The 25 baseline worlds of Experiment 8 that lived to iteration 600 — B1 at 1,600 to 409,600 tokens, seeds 1 to 3, of which one world of 1,600 tokens and all of 3,200 tokens and more (`B1-1600-s002`, `B1-3200-s001` … `B1-409600-s003`; [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E08`), after the game of iteration 599. The 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, [The life of a B1 world](b1-lives-and-wealth.md#the-life-of-a-b1-world); `python3 gol_lab.py run E02`). The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E10`).
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read the frame after each world's last game; build its network from `edges` and keep its largest piece (`book_chapters.space.largest_piece`). Its random twin: the same numbers of links per agent wired at random (`book_graph.degree_preserving`, seed 23), its largest piece.
> 2. Balls: a breadth-first search all the way out from each of 100 agents drawn at random (seed 23); N(r) the geometric mean of the numbers within r steps; d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), read while N(r+1) is less than a quarter of the piece (`book_chapters.scaledims.reach` and `ball_curve`).
> 3. A world whose best-linked agent was joined, on average over iterations 500 to its end, to a tenth of its agents or more is a star ([Stars and hubs](measuring-networks.md#stars-and-hubs); `maxDegree`/`nodes` of its rows in `stats.jsonl`).
> 4. The same balls, of each world and of its twin, after the games of five iterations evenly spaced over its settled life, from 500 to its last (500, 525, 550, 574 and 599 in Experiment 8; 500, 750, 1,000, 1,249 and 1,499 in Experiment 10; 500, 1,125, 1,750, 2,374 and 2,999 in Experiment 2); each reading the median of the five (`book_chapters.scaledims.settled`).
> 5. Bodies: the n points of a square or cubic grid nearest its centre (and any as near as the last), each linked to the points beside it that are inside too (`book_chapters.scaledims.body`), for n = 250, 500, … 64,000; read as the worlds are. The endless grids' balls are counted exactly: Σₖ 2ᵏ·C(d, k)·C(r, k) points within r steps.
> 6. The typical distance: the radius at which N(r) first reaches half the piece, between whole steps by the logarithm of N (`book_chapters.scaledims.half_reach`, as `research/atlas/space.py` reads it).
> 7. Fit log₁₀(typical distance) = log₁₀ a + b · log₁₀(agents) by least squares over Experiment 8's worlds of 3,200 tokens and more that are not stars; the interval of b from 10,000 resamplings of the worlds within each size (`gol_analysis.power_fit`).
>
> **To make it again:** `python3 book_figures.py scaledims`.
<!-- /figure -->

In Experiment 8, over their settled lives, B1's worlds widen as
agents^0.20 (95% interval 0.15 to 0.27); with Experiments 2 and 10 beside
them, as agents^0.23 (0.19 to 0.27). Their diameters, the longest distances
found, widen a little faster: as agents^0.26 (0.22 to 0.30), and 0.27 (0.24
to 0.31). The ruler itself is sound. The ball of a cubic grid widens as
points^0.34 to 0.35 and the disc as points^0.51, close to 1/3 and 1/2. And
the random twins hardly widen at all, as agents^0.02 to 0.05: a small
world's distances grow only like a logarithm.

Read as a dimension, one over these exponents is 3.7 to 4.9: about four,
more than the three or so the balls hold. At 1,400 agents a B1 world is about
as wide as a ball cut out of a cubic grid with as many points, 10 or 11
steps. At 64,000 agents the median world is narrower, 27 steps against 39,
though one of the three is wider (42).

There are two ways to read this. A dimension that grows towards four at the
largest scales would do it. So would a world that gains room close in as it
grows: hubs that grow with a world bring its agents nearer to one another
everywhere, and its distances then grow more slowly than they would in a
body of a fixed make-up. The balls say that the second happens. The width
alone cannot tell the two apart, and its interval is wide.

## Round 3's worlds, beside B1's

Round 3 of the [screen of random worlds](a-screen-of-random-worlds.md)
judges, among much else, whether a world is almost three-dimensional, and
one of its two measures is the width: a world's typical distance over its
random twin's ([Which worlds are healthy?](which-worlds-are-healthy.md)). But
a space's width grows with its size, and round 3's worlds hold from about
1,100 agents to 21,000. A width that is wide for the smallest is narrow for
the largest. B1's worlds, read at many sizes, give a yardstick: the width B1
has at each size.

<!-- figure scaledims/round3 -->
![Round 3's worlds, beside B1's at their size](../figures/scaledims/round3.svg)

**Round 3's worlds, beside B1's at their size.** The width of a world — its typical distance over its random twin's — against its agents, on logarithmic axes. Blue: B1's worlds that are not stars, each the median over five moments of its settled life, and dashed, the power law fitted to them. Green and orange: the worlds of round 3 of the [screen of random worlds](a-screen-of-random-worlds.md) that lived to the end of their runs, each averaged from iteration 100 on as [Which worlds are healthy?](which-worlds-are-healthy.md) reads them, by whether their dead are mended (`mend_on_death`).

> [!example]- How to make this figure
> **Runs.** The 25 baseline worlds of Experiment 8 that lived to iteration 600 — B1 at 1,600 to 409,600 tokens, seeds 1 to 3, of which one world of 1,600 tokens and all of 3,200 tokens and more (`B1-1600-s002`, `B1-3200-s001` … `B1-409600-s003`; [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E08`), after the game of iteration 599. The 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, [The life of a B1 world](b1-lives-and-wealth.md#the-life-of-a-b1-world); `python3 gol_lab.py run E02`). The 30 baseline worlds that lived to the end (`B1-100000-s001` … `-s030`, [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); `python3 gol_lab.py run E10`). Round 3 of the screen: every world of Experiments 40 to 42 that lived to the end of its run, 30,000 tokens, 3,000 iterations.
>
> **Data.** From B1's frames, as above, and for round 3 `research/screen/health.json` (made by `research/screen/health.py` from the screen's readings).
>
> 1. Read the frame after each world's last game; build its network from `edges` and keep its largest piece (`book_chapters.space.largest_piece`). Its random twin: the same numbers of links per agent wired at random (`book_graph.degree_preserving`, seed 23), its largest piece.
> 2. Balls: a breadth-first search all the way out from each of 100 agents drawn at random (seed 23); N(r) the geometric mean of the numbers within r steps; d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), read while N(r+1) is less than a quarter of the piece (`book_chapters.scaledims.reach` and `ball_curve`).
> 3. A world whose best-linked agent was joined, on average over iterations 500 to its end, to a tenth of its agents or more is a star ([Stars and hubs](measuring-networks.md#stars-and-hubs); `maxDegree`/`nodes` of its rows in `stats.jsonl`).
> 4. The same balls, of each world and of its twin, after the games of five iterations evenly spaced over its settled life, from 500 to its last (500, 525, 550, 574 and 599 in Experiment 8; 500, 750, 1,000, 1,249 and 1,499 in Experiment 10; 500, 1,125, 1,750, 2,374 and 2,999 in Experiment 2); each reading the median of the five (`book_chapters.scaledims.settled`).
> 5. Width: the typical distance over the twin's, at each moment; its median over the five. Fit log width against log agents over B1's worlds that are not stars (`gol_analysis.power_fit`).
> 6. Round 3: each world's `averages.width` and `averages.agents` in `health.json`, and `set.mend_on_death`; nothing is measured again.
>
> **To make it again:** `python3 book_figures.py scaledims`.
<!-- /figure -->

B1's width grows as agents^0.19 (95% interval 0.16 to 0.23): about 2.2 at
1,500 agents, 3.2 at 10,000 and 4.5 at 60,000. Set beside it at its own
size, a round 3 world is wider or narrower than B1 would be.

Of round 3's 82 worlds that lived to the end, 33 are at least as wide as B1
at their size. Two thirds of them leave their dead unmended: of the 38
worlds that do not mend, 22 are as wide as B1 or wider, against 11 of the 44
that mend. In the median, a world that does not mend is as wide as B1 at its
size (1.04 times), and one that mends 0.73 times as wide, folded as
[What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)
found. The widest for its size is seed 3044 of Experiment 40, at 1.77 times
B1's width, though its dead are mended. Next come the two plays of the draw
of seed 4003, a token at a time weakest first and strongest first
(Experiments 41 and 42), at 1.6 times.

A world's width over B1's at its size takes the size out of the comparison,
which a plain width does not. Width is not a dimension, though: a long,
thin world is wide too.

## What larger worlds would show

The three explanations of the opening make different forecasts, and the
worlds there are already rule on two of them.

- **A dimension that grows with the scale** would tilt the plateau of the
  largest worlds upwards in the median, and lift a walker's reading as its
  walks grow longer. Neither happens
  ([The rulers at every size](#the-rulers-at-every-size)). Where a single
  world's reading climbs, a hub explains it. *Not seen, within reach.*
- **The edge** would let a world's readings over its first few steps rise
  with its size only as a body's do, by about a tenth, and never above what
  the same body reads with no edge. The worlds' rise by three to six tenths,
  and end above any cubic grid's
  ([One radius, every size](#one-radius-every-size)). *Ruled out as the main
  cause.*
- **Larger worlds made differently** remains: bigger hubs and more links
  give them more room close in. *Indicative*: it rests on the correlations
  within one size, over 16 and 23 worlds, and on Experiment 8's two or three
  worlds a size.

Will the make-up settle, and the readings with it? The worlds there are give
a mixed answer. The readings at a few steps rose most between 5,000 and
15,000 agents, as the best-linked agent grew from under a hundred links to
several hundred. Above that, the median of each size changed little: 2.5 to
2.8 between radius 2 and 3, and 3.0 to 3.3 over radii 3 to 6, from 15,000
agents to 64,000. But across the 32 worlds of 10,000 agents and more, the
larger still read a little more (a rank correlation of +0.49 over radii 3 to
6), and their hubs are still growing, to 1,215 links. The rise may be
slowing; it has not been seen to stop.

How large would a world have to be to settle it? A world's plateau ends
where its balls hold a quarter of it, two thirds to three quarters of the way
to its typical distance, and in Experiment 8 that distance grows as
agents^0.20. So to read the balls half as far out again as on the largest
worlds, to radius 30 or so, a world would need about seven times their
tokens, some 3 million. By the notebook's own estimate of what a run costs
([What a simulation costs](runs-and-records.md#what-a-simulation-costs)), 600 iterations of
it would take 2.8 days and 65 GB of memory. To read them twice as far out,
to radius 40, it would need 12 million tokens: 11 days and 260 GB. And it
would need several seeds of each, since one world's hub can bend its whole
curve. There is a cheaper way to the same answer: worlds whose rules keep
their make-up the same at every size, with no hub that grows with the world,
would show at sizes the notebook can afford whether their dimension stays
put.

## Space, and room to evolve

**For the physics aim** ([Physical inspiration](physical-inspiration.md)),
B1's worlds stay what [Is a world a space?](how-many-dimensions-does-a-world-have.md)
found: spaces grown from local rules, with about three dimensions of room and
a walker that sees two. This page adds that, within reach, no scale is
singled out. There is no small-scale two that becomes a large-scale four, as
in the triangulations of quantum gravity. But the room a world holds within a
few steps grows with the world, through its hubs. A dimension that belongs to
the size of a world is not yet the dimension of a space. A world whose rules
gave it the same make-up at every size would be the test.

**For [open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution "Evolution that keeps producing new things without end, not only up to a plateau.")**, what counts is room for regions to evolve
apart. In the largest worlds, a typical agent has half its world within 27
steps (the median of their settled lives), and in the worlds of 100,000
tokens a line of descent holds a patch out to about 12 steps
([Where the lines live](b1-the-network-size-and-scaling.md#where-the-lines-live)).
So even the largest worlds hold only a few such patches side by side. And
B1's worlds widen slowly: ten times the agents make a world only
10^0.20 ≈ 1.6 times as wide. A world that is to hold many regions far apart
needs either far more agents than the notebook can afford, or rules under
which it widens faster as it grows.

## What we learned

- **Within a world, the dimension does not change with the scale beyond what
  any finite body shows.** By balls, worlds of every size rise over their
  first steps, hold on a plateau and fall at their edge, as a ball cut out of
  a cubic grid does. In the median of the largest worlds the plateau holds at
  3.0 to 3.2 from radius 3 to 14. By walks the worlds read 1.7 to 2.0 at
  every size, and the largest no more over walks of up to 8,192 steps.
  Within reach, there is no small-scale two
  that becomes a large-scale four. Where a single world's reading climbs, a
  hub explains it.
- **An edge only lowers a reading.** A ball cut out of a cubic grid reads at
  most 2.30 at 2,000 points and 2.65 at 64,000; a disc of a square grid, 1.73
  and 1.89. By balls, no body reads more than its dimension.
- **Larger worlds read more at the same radius, and the edge is not why.**
  Between radius 2 and 3 the worlds read 1.8 to 2.4 up to 5,000 agents and
  2.5 to 2.8 from 14,000 on, more than any cubic grid; a body's reading
  there changes by a tenth. Larger worlds hold bigger hubs, and more room
  close in (*indicative*). Above 15,000 agents the rise slows.
- **By their width, the worlds read about four.** Their typical distance
  grows as agents^0.20 to 0.23, their diameter as agents^0.26 to 0.27, where
  a ball of a cubic grid widens as points^0.35: 3.7 to 4.9 dimensions. Hubs
  that grow with a world would do the same, so the width alone cannot say
  whether the dimension grows at the largest scales (*indicative*).
- **One look at a world is not enough.** A world's reading over radii 3 to 6
  varies by 0.8 between five moments of its settled life, and its typical
  distance by a factor of 1.8, in the median world.
- **Round 3's worlds that do not mend their dead are as wide as B1 at their
  size**, in the median; those that mend are 0.73 times as wide. Seed 3044
  of Experiment 40 and seed 4003 of Experiments 41 and 42 are the widest for
  their size, 1.6 to 1.8 times B1's.
- **Settling it needs larger worlds, or steadier ones.** To read the balls
  twice as far out, a world needs about 12 million tokens: by the notebook's
  own costs, 11 days and some 260 GB of memory for 600 iterations. Worlds
  whose make-up does not change with their size would answer the question at
  the sizes there are.

## Where to go next

- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md):
  the two rulers, tried on known spaces, and what folds a world into a small
  world.
- [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes): the
  experiment whose worlds this page reads at nine of its ten sizes.
- [Stars and hubs](measuring-networks.md#stars-and-hubs): the hubs that give larger worlds more
  room close in.
- [Physical inspiration](physical-inspiration.md): the physics behind the
  hope for a space of three dimensions.

## How this was made

Every figure of this page says, folded beneath it, which runs it reads and
how. No world was played for it: it reads saved runs only. To draw its
figures again: `python3 book_figures.py scaledims`; the code is
`book_chapters/scaledims.py`.

- **The worlds.** Experiment 8's 25 baseline worlds that lived to iteration
  600 (`B1-1600-s002` and `B1-3200-s001` … `B1-409600-s003`;
  `python3 gol_lab.py run E08`), Experiment 2's 26 worlds of 10,000 tokens
  that lived to iteration 3,000 (`B1-10000-s001` … `-s030`;
  `python3 gol_lab.py run E02`) and Experiment 10's 30 worlds of 100,000
  tokens (`B1-100000-s001` … `-s030`; `python3 gol_lab.py run E10`). The
  [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") of Experiments 8 and 10 are listed below, Experiment 2's in
  [The life of a B1 world](b1-lives-and-wealth.md#the-life-of-a-b1-world).
- **After the last game.** Each world's largest piece and its random twin,
  by balls from 100 agents drawn at random, all the way out, and by a lazy
  random walk from 48 agents for up to 1,024 steps; in the three worlds of
  409,600 tokens, the same walks for up to 8,192 steps, and how far their
  agents are from their best-linked agent. Kept in
  `GraphOfLifeRuns/.book/<run>.scaledims.json`, `<run>.scaledims-long.json`
  and `<run>.scaledims-hub.json`.
- **Over the settled life.** The same balls after the games of five
  iterations evenly spaced from 500 to the last, kept in
  `<run>.scaledims-settled.json`.
- **The bodies.** Discs and balls cut out of square and cubic grids, 250 to
  64,000 points, read the same way and kept in
  `GraphOfLifeRuns/.book/bodies/`.
- **Stars.** A world whose best-linked agent was joined, on average from
  iteration 500 to its end, to a tenth of its agents or more
  ([Stars and hubs](measuring-networks.md#stars-and-hubs)).
- **Round 3.** Its 82 worlds that lived to the end of their runs, as
  `research/screen/health.json` holds them (made by
  `research/screen/health.py`); nothing of them was measured again.

Every random choice is seeded with 23, so the same runs give the same
figures.

### The runs of Experiment 8

<!-- runs E08 -->
> [!info]- The runs behind this page
> **42 runs.** Every condition below is run once for every seed (1 to 3) and every number of tokens it lists, for 600 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 800, 1,600, 3,200, 6,400, 12,800, 25,600, 51,200, 102,400, 204,800, 409,600 tokens. Runs `B1-800-s001` … `B1-409600-s003`.
> - **stopped only when empty** — changes `extinction_threshold` = 0; 800, 1,600, 3,200, 6,400 tokens. Runs `B1-800-57b358-s001` … `B1-6400-57b358-s003`.
>
> To make these runs again: `python3 gol_lab.py run E08`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | stopped only when empty | what it is |
> |---|---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 800, 1,600, 3,200, 6,400, 12,800, 25,600, 51,200, 102,400, 204,800, 409,600 | 800, 1,600, 3,200, 6,400 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | 50, 45, 40, 35, 30 | The widths of the brain's hidden layers, in order from input to output. |
> | [`hidden_activation`](settings.md#hidden_activation) | sigmoid | sigmoid | The curve each hidden unit applies to its sum: `sigmoid`, which passes on at most a quarter of a change, so that deep brains barely respond to what they see, or `tanh`, the same S-shape centred on zero, through which a change still passes in several layers. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`contest`](settings.md#contest) | largest | largest | Who takes a contested node: the `largest` stake as the game weighs it, or a staker drawn by `lottery` with a chance in proportion to its stake (needs revolutions off). |
> | [`allocation`](settings.md#allocation) | full | full | How the game's stakes are placed: all at once (`full`), or a token at a time in rounds with the game as it stands in view, the poorest done first (`weakest_first`) or the richest starting alone (`strongest_first`), as the original engine played. |
> | [`allow_gifting`](settings.md#allow_gifting) | off | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none); `local` lets each estate flow through the network to the nearest living agents, split evenly at every step. |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`link_token`](settings.md#link_token) | off | off | Whether every link holds one token: its maker pays for it, a lapsing link gives it back to one of its ends, and a link of a dead agent to the end that lives on, so the links can never outnumber the tokens. |
> | [`link_upkeep`](settings.md#link_upkeep) | off | off | Whether an agent may pay a token an iteration to keep a link, each brain deciding for each neighbour: `paid_or_used` keeps a link that was paid for or used, `paid_only` only one that was paid for; `off`, only use keeps a link. |
> | [`upkeep_to`](settings.md#upkeep_to) | end | end | Where a link's upkeep goes: to the link's other end (`end`), or into the estate of the dead (`estate`), shared out by the world's redistribution rule. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`mate_pays`](settings.md#mate_pays) | off | off | Under mix_at_birth, whether the mate pays the child as many tokens as the parent gave it, as far as it can and never its last. |
> | [`self_message`](settings.md#self_message) | own | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | **0** | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 3 | 1 to 3 | one run per seed |
> | iterations | 600 | 600 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

### The runs of Experiment 10

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
> | [`hidden_activation`](settings.md#hidden_activation) | sigmoid | The curve each hidden unit applies to its sum: `sigmoid`, which passes on at most a quarter of a change, so that deep brains barely respond to what they see, or `tanh`, the same S-shape centred on zero, through which a change still passes in several layers. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 30 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | on | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | on | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`contest`](settings.md#contest) | largest | Who takes a contested node: the `largest` stake as the game weighs it, or a staker drawn by `lottery` with a chance in proportion to its stake (needs revolutions off). |
> | [`allocation`](settings.md#allocation) | full | How the game's stakes are placed: all at once (`full`), or a token at a time in rounds with the game as it stands in view, the poorest done first (`weakest_first`) or the richest starting alone (`strongest_first`), as the original engine played. |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none); `local` lets each estate flow through the network to the nearest living agents, split evenly at every step. |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | off | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.0 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 0 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | off | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 0 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 0 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`link_token`](settings.md#link_token) | off | Whether every link holds one token: its maker pays for it, a lapsing link gives it back to one of its ends, and a link of a dead agent to the end that lives on, so the links can never outnumber the tokens. |
> | [`link_upkeep`](settings.md#link_upkeep) | off | Whether an agent may pay a token an iteration to keep a link, each brain deciding for each neighbour: `paid_or_used` keeps a link that was paid for or used, `paid_only` only one that was paid for; `off`, only use keeps a link. |
> | [`upkeep_to`](settings.md#upkeep_to) | end | Where a link's upkeep goes: to the link's other end (`end`), or into the estate of the dead (`estate`), shared out by the world's redistribution rule. |
> | [`keep_home`](settings.md#keep_home) | 0.0 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 0 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 1 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | off | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`mate_pays`](settings.md#mate_pays) | off | Under mix_at_birth, whether the mate pays the child as many tokens as the parent gave it, as far as it can and never its last. |
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

## References

- Ambjørn, J., Jurkiewicz, J. and Loll, R. (2005). Spectral dimension of the universe is scale dependent. *Physical Review Letters* 95, 171301.
- Carlip, S. (2017). Dimension and dimensional reduction in quantum gravity. *Classical and Quantum Gravity* 34, 193001.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. San Francisco: W. H. Freeman.

<!-- turns -->
---

← [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Space and dimension](../README.md#space-and-dimension) · [Which properties have no scale?](which-properties-have-no-scale.md) →

> [!info]- Pages that link here (3)
> [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md)
<!-- /turns -->
