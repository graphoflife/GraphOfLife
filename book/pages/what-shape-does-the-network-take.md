# The shape of the network

> [!question] Questions of this page
> - What becomes of the founders' ring? Is the network that replaces it a
>   tree, a web, or something in between?
> - How many [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") does an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") have, and how unequally are they
>   spread?
> - What is left of a world when its hanging trees are peeled away? How much
>   of the agents and [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") does it hold, and who dies where?
> - How far apart are two agents, and is that far or near for a network of
>   its kind?
> - What kind of space is a world: a line, a sheet, a tree? How many
>   dimensions do it and its core have?
> - How easily could a world be cut in two, and where do its loops run?

## In short

Within a hundred iterations, the founders' ring with its few shortcuts becomes
a mesh, the **core**, with small trees hanging off it. The core holds more
than half of the agents, three quarters of the tokens and every loop. Most
agents have one or two connections, and a few [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") have hundreds or thousands.
Experiment 5 had predicted a thinner, more tree-like world, and was refuted on
the size of the core. A child joins only its parent's neighbourhood, so the
world is stretched, with bottlenecks that make it slow to mix and easy to
break. By the room it holds, a world and its core have two to three
dimensions; by how a walker spreads, a little under two.

## What shape does the network take?

The founders start on a small-world ring
([The starting ring](starting-ring.md)). Each is joined to its nearest
founders on a circle, and some connections are moved at random to make
shortcuts. From then on the agents make and lose every connection themselves.
A child is joined to some of its parent's candidates, and a parent may hand
connections to it. In every game, a connection that carries no tokens is cut
when the game ends ([One iteration, step by step](one-iteration.md)). Nothing
in the rules aims at any shape. So what shape comes out?

Every section of this page reads the thirty worlds of the first [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."),
at 100,000 tokens ([Thirty worlds](a-worlds-life.md#thirty-worlds)). Much of
it was first written from the thirty worlds of 10,000 tokens; where those
differ, their numbers are given beside.

### What was expected

Why should the ring not last? A new connection can only be made at a birth,
between the newborn and its parent's neighbourhood. And any connection that
carries no tokens in a game is cut, a new one too. Nothing ever joins two
distant parts of a world, so the ring's long shortcuts are a budget that is
only spent. Children should hang off their parents like leaves off a branch,
and a world should become a tree of families.

How tree-like is a network? Three measures say. A **bridge** is a connection
on no loop: cut it, and the world falls in two ([Bridges](bridges.md)).
**Clustering** says how often two [neighbours](the-world.md#the-network "An agent joined to another by a connection.") of an agent are neighbours of
each other too; in a tree, never ([Clustering](degree.md#clustering)). And the
**core** is what is left when agents with one connection or none are peeled
away, again and again. In it, every agent lies on a loop or on a path between
loops ([Core, trees and leaves](bridges.md#core-trees-and-leaves)); a tree has no core at
all. Before the runs of 10,000 tokens, Experiment 5 wrote down its thesis:

<!-- thesis E05 -->
> [!quote] The thesis of Experiment 5, written down before any of its runs existed
> **The claim.** The small-world ring the founders start on is gone within a hundred iterations, and what replaces it is thin and tree-like: more than a fifth of all connections are bridges — connections on no loop — clustering stays below 0.1, and less than half of all agents sit in the core where every agent is on some loop.
>
> **Why it would be so.** A new connection can only be made at birth, between a newborn and its parent's neighbourhood, and any connection that carries no tokens in a game is cut at its end — a new one included. Nothing ever joins distant parts of the world, so the long-range shortcuts of the starting ring are a budget that is only spent. Children hang off their parents like leaves off a branch.
>
> **It holds if:** From iteration 100 on, the median share of connections that are bridges is above 0.2, the median clustering is below 0.1, and the median core share is below 0.5, across the thirty runs.
>
> **It fails if:** The graph keeps or rebuilds loops: bridges below a tenth of connections, or clustering above 0.2, or a core holding most agents. Then something in the rules closes loops, and the picture of a world as a tree of families is wrong.
<!-- /thesis -->

The thesis was tested in the thirty worlds of 10,000 tokens, and again in the
thirty of 100,000 ([Was the thesis right?](#was-the-thesis-right)). First,
here is what the large worlds did, with the small ones beside them.

### One world, four moments

<!-- figure shape/worlds -->
![One world at four moments](../figures/shape/worlds.svg)

**One world at four moments.** The world with seed 1 after the games of iterations 0, 100, 500 and 1,499. Every agent alive is a dot, every connection a line (at iteration 0, every one of its 1,389 agents; at iteration 100, the 748 of its 16,099 agents within 2 steps of agent 415, the oldest, 100 iterations old — a piece, since a world this large is a tangle when drawn whole; at iteration 500, the 685 of its 15,889 agents within 10 steps of agent 424, the oldest, 500 iterations old — a piece, since a world this large is a tangle when drawn whole; at iteration 1,499, the 2,495 of its 14,870 agents within 25 steps of agent 660,973, the oldest, 806 iterations old — a piece, since a world this large is a tangle when drawn whole). Blue: agents in the core, which is what is left when agents with one connection or none are removed again and again until none remains ([Core, trees and leaves](bridges.md#core-trees-and-leaves)); yellow: the rest of the hanging trees; red: leaves, agents with a single connection. Where a dot is drawn means nothing in itself: a layout (networkx's ForceAtlas2, seed 1) pulls joined agents together and pushes the rest apart.

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`, one of the 30 baseline runs.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frames `1`, `201`, `1001` and `2999` (iterations 0, 100, 500 and 1,499, after the game): `ids` and `edges`.
> 2. Peel: remove every agent with one connection or none, again and again, until none is left; the agents never removed are the core. Leaves are agents with exactly one connection.
> 3. In a world of more than 2,500 agents, keep the agents within r steps of the oldest agent, for the largest r that keeps at most 2,500 (`book_chapters.society.shown`).
> 4. Lay the graph out with `networkx.forceatlas2_layout(G, max_iter=200, seed=1)`.
>
> **To make it again:** `python3 book_figures.py shape`.
<!-- /figure -->

The pictures colour every agent by its place in the network. Blue marks the
agents of the core. Red marks the **leaves**, agents with a single connection.
Yellow marks the agents in between, on the trees that hang off the core.

After the first game, the founders and their first children, 1,389 agents,
still form one web. It is the founders' ring with its shortcuts, and 97% of
its agents are in the core. A hundred iterations later the world has 16,099
agents, too many to draw. The picture holds the 748 within two steps of its
oldest agent. The world has changed character. It is now dense knots of core,
joined by thin strands, with leaves bunched at their ends. At iterations 500
and 1,499 it looks the same in kind. More than half of all agents are in the
core (58% and 64% of the whole world). About a third or less are leaves (36%
and 29%).

### Connections per agent

<!-- figure shape/degree -->
![How many connections agents have](../figures/shape/degree.svg)

**How many connections agents have.** Of all agents alive after the last game of the 30 worlds, the share that have at least d connections, for every d. At the start a founder had 10 on average (the grey line) — every one of them exactly that many before the ring's rewiring ([The starting ring](starting-ring.md)).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Count every agent's connections in the last frame's `edges`; pool the 30 runs.
> 2. For d = 1, 2, …, the share of agents with at least d.
>
> **To make it again:** `python3 book_figures.py shape`.
<!-- /figure -->

An agent's **degree** is its number of connections ([Degree](degree.md)). The
figure shows, for every *d*, the share of agents with at least *d* connections
(a complementary cumulative distribution). Both axes are logarithmic: each
step along them multiplies by the same factor
([Logarithmic axes](correlation.md#logarithmic-axes-and-power-laws)). After the last game of the thirty
worlds, 35% of all agents have exactly one connection and 23% have two. The
mean is 3.5. But 4.7% have ten or more, and the curve reaches out to an agent
with 10,735. That agent is the centre of a star, a world held together by a
single hub ([The largest and the smallest](the-largest-and-the-smallest.md)).
Every founder started with ten. At 10,000 tokens, 37% had one connection and
23% two, and the mean was 3.2. There, 4% had ten or more, the largest had
1,399, and the founders started with four.

So the network is very unequal in connections. Most agents have one or two,
and a few are **hubs**, joined to hundreds. On the log–log axes the curve
bends downward instead of running straight, so it is not a clean [power law](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones.").

### Core, leaves and bridges

<!-- figure shape/structure -->
![Core, leaves and bridges](../figures/shape/structure.svg)

**Core, leaves and bridges.** Three shares, after every game of the 30 baseline worlds. Left: the share of agents in the core ([Core, trees and leaves](bridges.md#core-trees-and-leaves)), measured every 25 iterations. Middle: the share of agents with exactly one connection, after every game. Right: the share of connections that are bridges ([Bridges](bridges.md)), every 25 iterations. In each, the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and the statistics `coreShare`, `leaves/nodes` and `bridges/edges` (`a/b` is the row's `a` divided by its `b`).
> 2. For `leaves/nodes`, make the band as for every other band ([Bands](median-and-quantiles.md#bands-many-worlds-in-one-figure)): cut the 1,500 iterations into 600 stretches of 2.5 and take each world's mean in each stretch.
> 3. For `coreShare` and `bridges/edges`, which are measured every 25 iterations, use stretches of 25.
>
> **To make it again:** `python3 book_figures.py shape`.
<!-- /figure -->

Three shares settle within the first hundred or so iterations, and then hold:

- **The core** holds 97% of the agents after the first game. Over the settled
  life, iterations 500 to 1,499, it settles near 57% in the median world (54%
  at 10,000 tokens). Across the thirty worlds it lies between 19% and 64%.
- **Leaves** rise from 3% to about 35% of all agents (36% at 10,000 tokens).
- **Bridges** rise from 1% to about 26% of all connections (31%).

All three are counted after the game, when the connections that carried
nothing have just been cut. [Peeling a world](#peeling-a-world) counts them at
the start of a game, and finds a slightly larger core.

The worlds at the far ends of these ranges are the stars. In the world with
[seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 28, the largest hub is joined to more than half of all agents. Only 19%
of its agents are in the core, 72% are leaves, and 72% of its connections are
bridges.

### Clustering

<!-- figure shape/clustering -->
![Clustering](../figures/shape/clustering.svg)

**Clustering.** Three times the number of triangles divided by the number of connected triples ([Clustering](degree.md#clustering)), every 25 iterations. The founders' ring starts at 0.34 in the median world. The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and the statistic `transitivity`.
> 2. Bands as above, in stretches of 25.
>
> **To make it again:** `python3 book_figures.py shape`.
<!-- /figure -->

Clustering is taken over the whole world. Of all the pairs of neighbours that
agents have, it is the share that are joined to each other. A triangle closes
three such pairs, so it is three times the number of triangles over the number
of pairs ([Clustering](degree.md#clustering)).

In the founders' ring, about a third of the pairs of neighbours of a founder
are neighbours themselves (0.34). Within the first game that halves (0.17). In
a settled world it is 0.039, between 0.002 and 0.114 across the thirty worlds:
half of the 0.075 of the worlds of 10,000 tokens.

That is low. But it is still more than a hundred times what a random network
of the same size would have, about 0.0002. There are triangles, and they are
not there by chance. A child joined both to its parent and to one of its
parent's neighbours closes a triangle the moment it is born.

### How far apart are two agents?

<!-- figure shape/distance -->
![Steps between two agents](../figures/shape/distance.svg)

**Steps between two agents.** For each world (blue): its mean number of agents and the mean number of steps along connections from one agent to another, both averaged over iterations 500–1,499. Grey: the rough distance in a random network with the same number of agents N and connections per agent k, ln N / ln k.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Average `nodes`, `meanDegree` and `meanPathLength` over the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,499 (`meanPathLength` is measured every 25 iterations, from breadth-first searches out of 8 to 16 evenly spread agents; [Path length](path-length.md)).
> 2. Plot one dot per run, and ln N / ln k for the same run.
>
> **To make it again:** `python3 book_figures.py shape`.
<!-- /figure -->

The distance between two agents is the fewest connections on a path between
them ([Path length](path-length.md)). Two agents of a settled world are a
median 17.7 steps apart. Take a random network with the same number of agents
*N* and connections per agent *k*. There, two agents are about ln *N* / ln *k*
steps apart: about 7.8. So the agents of a world are 2.3 times as far apart.
In the worlds of 10,000 tokens they were 1.6 times as far (9.8 steps against
6). The worlds are stretched out. A token moves at most one connection in a
game, so it needs about eighteen games to go from one agent to another, on
average.

Among these thirty worlds, the larger ones do not have their agents further
apart. The worlds hold 12,300 to 18,100 agents, and the
[correlation](correlation.md) between size and distance is −0.35. Without the
seven stars, which are both large and short, it is −0.18. In a star, the
agents joined to its centre are two steps from one another through it. The
shortest world, seed 28, has its agents 4.2 steps apart. Among the others,
distances range from 12 to 26 steps, for reasons other than size.
[Distances](#distances) measures one world exactly, against a sharper null.

### Was the thesis right?

The thesis was checked, as planned, from iteration 100 on. In the worlds of
10,000 tokens it holds in three of its four clauses. The ring is gone almost
at once. More than a fifth of all connections are bridges: a median of 32%.
They are more than a fifth in every one of the 26 worlds that lived to the
end. Clustering stays below 0.1: a median of 0.077, and above 0.2 in no world.
But the thesis fails in its fourth clause. The core holds about half of all
agents, not less: a median of 53%, and less than half in only 6 of the 26
worlds. Over the settled life, from iteration 500 on, the numbers are almost
the same: 31%, 0.075 and 54%. So the thesis is **refuted**, narrowly, on its
last clause.

At 100,000 tokens, checked the same way from iteration 100 on, it is refuted
again on the same clause, and more clearly. Bridges are a median 26% of all
connections, and more than a fifth in every one of the thirty worlds.
Clustering is 0.043, above 0.1 in one world and above 0.2 in none. But the
core holds a median 57% of all agents, and less than half in only 5 of the 30.
A larger world is, if anything, more of a web.

### What closes the loops

The thesis said what its failure would mean. Something in the rules closes
loops, and the picture of a world as a tree of families is wrong. Births close
them. A child is joined to some of its parent's candidates. If two of them are
neighbours of each other, the child closes a triangle at once
([Clustering](#clustering)).

The measurements of the baseline found how this shapes a world. It grows only
locally, by copying. An agent gains a connection whenever a neighbour has a
child that is joined to it too. So the well-connected gain connections in
proportion to those they have, and each connection a hub gains this way comes
with about one triangle. Clustering then falls as one over an agent's number
of connections. The result is a **hierarchy of stars**, with leaves hanging on
hubs ([Power laws, real and apparent](power-laws-real-and-apparent.md),
[How properties scale together](how-properties-scale-together.md)).

## The core of a world

Peel away every agent with one connection or none, again and again, and what
is left is a world's **core**
([Core, trees and leaves](bridges.md#core-trees-and-leaves)). Its agents lie on a loop, or
on a path between loops. Everything peeled hangs off the core in trees.
[Core, leaves and bridges](#core-leaves-and-bridges) found that the core holds
more than half of the agents. This section looks inside it. How much of the
tokens does it hold? How far do the trees hang from it, and how deep does it
go? Where are the loops, and who lives where?
[How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)
measured the dimensions of a whole world; this section asks the same of its
core.

### Why the core matters

The core is where a world can hold anything together. Every connection in it
lies on a loop or on a path between loops. Cutting one on a loop splits
nothing: there is another way round. But a connection on a path between loops
is a bridge, and its loss splits the core in two ([Bridges](bridges.md)). The
**cut risk** of a world is the largest share of it that one bridge holds to
the rest ([Cut risk](bridges.md#cut-risk)). In one world of
[How a world breaks](how-a-world-breaks.md), a third to almost half of the
world hung on a single connection, for 140 iterations. A tree is cut off whole
by the loss of the one connection that holds it. So anything larger than an
agent that might last and evolve can only live in the core. Such a thing would
be a group of agents that holds together while its members come and go.

And the core is the better candidate for a **space**, in the sense of
[Physical inspiration](physical-inspiration.md). Trees branch. A tree of depth
*d* can hold a number of agents that grows exponentially with *d*. So the
agents within *r* steps of an agent, its **ball** of radius *r*, swell in
number with every tree the ball meets. Such growth makes a network look
higher-dimensional than the space it hangs from. If the worlds have a
dimension of their own, it should show most cleanly in their cores. The
difference between a world's dimension and its core's says how much the trees
inflate it.

### Peeling a world

Every 25 iterations, at the start of the game, each world is peeled. Its
agents and tokens are counted in three parts: the core, the hanging trees and
the leaves. The hanging trees are the agents that the peeling removes,
although they have two connections or more.

<!-- figure core/parts -->
![Core, trees and leaves](../figures/core/parts.svg)

**Core, trees and leaves.** At the start of the game of every 25th iteration of the 30 worlds: the share of the agents (left) and of the tokens (right) in the core, on the hanging trees (agents with two connections or more that the peeling removes), and on leaves (agents with one connection). The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`.
>
> 1. Peel every agent with one connection or none, again and again; what is never peeled is the core; leaves are agents with one connection; the rest of the peeled are on trees.
> 2. Count the agents and add up the tokens of each part; divide by the world's.
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

Over the settled life, the core holds **60%** of the agents in the median
world. In the middle half of the worlds it holds between 56% and 63%. The
hanging trees hold 7.5% and the leaves 32%. These shares are counted at the
start of a game, just after the births. After the game, when the connections
that carried nothing have just been cut, the core holds 57% and the leaves 35%
([Core, leaves and bridges](#core-leaves-and-bridges)). The core holds more of
the tokens than of the agents, **73%**. The leaves hold less: 20% of the
tokens, on 32% of the agents. The shares settle within the first hundred
iterations and do not move after.

### How far the trees hang

Search outward from all core agents at once, one step at a time: a
breadth-first search. It gives every other agent its distance from the core,
and the core agent its tree hangs from.

<!-- figure core/trees -->
![The trees that hang off the core](../figures/core/trees.svg)

**The trees that hang off the core.** Left: of all agents off the core at the start of the games of every 25th iteration from 500 on, pooled over the worlds, the share at each number of steps from the core, on a logarithmic axis: a straight falling line is a share that shrinks by the same factor with every step. Right: every 25 iterations, the number of agents hanging from the core agent that holds the most, the steps from the core to the farthest agent, and the mean number hanging from a core agent that holds any. The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`.
>
> 1. Peel as above; a breadth-first search out of every core agent at once gives every other agent its steps from the core and the core agent its tree hangs from (`book_chapters.core.hanging`).
> 2. Count the agents per step and per core agent.
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

Most of what hangs off the core hangs close to it. Of the agents off the core,
77% are one step from it, mostly leaves joined straight to a core agent.
Another 15% are two steps away, 5% three and 1.8% four. From the third step
on, each step holds about 40% of the agents of the one before. On the
logarithmic axis the points fall along a straight line: the mark of a share
that shrinks by the same factor at every step. The deepest agent of a world is
about **9 steps** from the core. A core agent from which anything hangs
carries 3 agents on average. But the largest tree of a world holds some 345
agents in the median world, and between 190 and 650 in the middle half. The
trees are small, with a few large ones.

### A core within the core

Peeling can go on. Remove, again and again, every agent with fewer than three
connections, and what is left is the **3-core**; with fewer than *k*, the
*k*-core. The core of this page is the 2-core. An agent's **core number** is
the deepest core it belongs to. It is 1 for the trees and leaves, which the
first peeling removes. It is 2 for the agents of the core that the 3-core no
longer holds, and so on. A world is an onion:

<!-- figure core/onion -->
![A world as an onion](../figures/core/onion.svg)

**A world as an onion.** An agent's core number is the largest k such that it belongs to the k-core: what is left when every agent with fewer than k connections is removed, again and again. Core number 1 is everything peeled to reach the 2-core (trees and leaves), 2 the agents of the 2-core that the 3-core no longer holds, and so on. Left: the share of agents with each core number, at the start of every 25th game from 500 on, pooled over the worlds. Right: the deepest core number in a world every 25 iterations; the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`.
>
> 1. `networkx.core_number` of the network at the start of the game.
> 2. Count the agents of each core number (8 and more together); and the largest.
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

Two agents in five have core number 1, three in ten core number 2, 15% core
number 3 and 7% core number 4. From there, each shell holds less than half the
one before. In the median world, the deepest core has core number **8**. It is
a small group of agents, each joined to at least eight others of the group:
the world's densest knot.

### The loops live in the core

A world after cleanup is one piece. How much more than a tree is it? Take any
tree of its connections that reaches every agent, a **spanning tree**. It has
one connection fewer than the world has agents. Every other connection closes
one loop with it, and none of these loops is a combination of the others. So a
world has connections − agents + 1 **independent loops**, its **cycle rank**
([Loops](bridges.md#loops)). That many connections, the right ones, could be cut
without splitting the world. All of those loops are in the core:

<!-- figure core/loops -->
![Loops, per agent and per core agent](../figures/core/loops.svg)

**Loops, per agent and per core agent.** Every 25 iterations at the start of the game: the number of independent loops of a world — connections minus agents plus one, its cycle rank ([Loops](bridges.md#loops)) — divided by its agents and by the agents of its core, which holds every loop; and the mean number of connections a core agent has to other core agents. The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`.
>
> 1. Cycle rank = connections − agents + 1 (the world is one piece after every cleanup).
> 2. Divide by the agents, and by the core's agents; connections with both ends in the core, twice, over the core's agents.
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

A settled world has **0.82** independent loops per agent, and **1.34** per
agent of its core. Inside the core, an agent has on average 4.7 connections to
other core agents. Each connection has two ends, so the core has 4.7 / 2,
about 2.3, connections per agent, where a tree would have one. Each of the 1.3
more closes a loop: 4.7 / 2 − 1 = 1.35 loops per agent, as many as are
counted.

### Who lives where

Which part an agent is in at the start of a game, and what it has:

<!-- figure core/where -->
![Who lives where](../figures/core/where.svg)

**Who lives where.** At the start of the games of every 25th iteration from 500 on: for the agents of the core, of the trees and the leaves, the share no longer alive after that game, their mean tokens, and their mean age. One dot per world; the bar is the median.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`.
>
> 1. Peel as above; for each part, the agents of frame `2·t` missing from frame `2·t + 1`, and the means of `tokens` and `ages`.
> 2. Pool the looks of a world; one dot per world.
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

A core agent holds 8.8 tokens on average, an agent of a tree 6.7 and a leaf
4.4. Core agents are a little younger than tree agents, 82 iterations against
86 on average, and older than leaves (77). But in a game they all die alike.
Of the core agents at the start of a game, **2.7%** are gone after it; of the
tree agents, 2.8%; of the leaves, 2.7%. Hanging from the world by a single
connection does not make a leaf more likely to die. A tree is cut off whole by
the loss of one connection. But agents in the core are lost in other ways just
as often. They starve, and whole parts of the core go quiet at once
([How a world breaks](how-a-world-breaks.md)).

### How many dimensions the core has

Two rulers measure a network's dimension
([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)).
The first counts the agents in a ball of radius *r*. In a space of *d*
dimensions, their number grows like *r*^*d*
([Dimension and curvature](ball-dimension-and-curvature.md)). The second
releases a walker that steps at random. It asks how often the walker is back
where it started after *t* steps: the more dimensions, the rarer
([The spectral dimension](ball-dimension-and-curvature.md#the-spectral-dimension)). Both rulers are laid on the
whole world, on its core alone, and on its **random twin**
([The random twin: is a world a space?](random-twin.md)). The twin has the
same agents, each with as many connections, wired at random.

<!-- figure core/dimension -->
![How many dimensions the core has](../figures/core/dimension.svg)

**How many dimensions the core has.** The two rulers of [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md) — the local dimension from how balls grow, and the spectral dimension from how a random walk spreads — on the 30 worlds after their last game (blue), on their cores alone (green), and on their random twins (red); each line is the median of the worlds, the darker band the middle half of them and the paler band nine in ten. Yellow: the cores of the three worlds of 409,600 tokens of [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`. The three worlds of 409,600 tokens of Experiment 8 (`B1-409600-s001` … `-s003`, [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md); `python3 gol_lab.py run E08`), after their last game.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read each run's last frame; build its network from `edges`; its core by peeling; its random twin by `book_graph.degree_preserving` (seed 23).
> 2. Each network's largest piece, measured as in [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md) (`book_chapters.space.measure`).
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

By balls the core reads **2.5** in the median world, and between 2.2 and 2.9
in the middle half; the whole world reads 2.8. By walks the two read the same,
**1.9**. The random twins read 6.5 by balls and 3.2 by walks. That is no
dimension at all in the sense of a space: a random network's balls grow
exponentially, and their reading only rises with *r*.

So the core of a world is a space, as the world is: two to three dimensions by
balls, a little under two by walks. By balls it reads a little lower than the
world. The trees inflate how fast a ball grows, by about 0.3 dimensions, as
[Why the core matters](#why-the-core-matters) supposed. They add agents to
every ball without adding a direction to move in. The random walk is hardly
affected. It spends a few steps in a tree and comes back.

The size of a world matters more than its trees. In the worlds of 10,000
tokens the core reads 1.9 by balls, and the whole world 2.3. The three worlds
of 409,600 tokens of
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
have about 42,000 core agents each. There the core reads 2.7 over radii 3 to
14 (the whole world 3.3), and 3.1 at radii 3 to 6. By walks the cores read
1.6, 1.9 and 1.5 at 10,000, 100,000 and 409,600 tokens. The core's dimension
by balls grows with the world, as the world's does, and has not settled at any
of these sizes. *Dimensions at scale*, planned (see
[the contents](../README.md)), is to follow it further.

### The core's dimension over a world's life

The same readings, after the games of five moments of the settled life, for
the whole world, its core and its 3-core:

<!-- figure core/over-life -->
![The core's dimension over a world's life](../figures/core/over-life.svg)

**The core's dimension over a world's life.** After the games of iterations 500, 750, 1,000, 1,250 and 1,499 of the 30 worlds: each ruler's typical reading — the median of its values over the scales of [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)'s plateau — on the whole world, on its 2-core and on its 3-core (what is left when agents with fewer than three connections inside the core are peeled away too). The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`.
>
> 1. At each moment, the network after the game, its 2-core, and its 3-core (`networkx.core_number` ≥ 3).
> 2. Both rulers on the largest piece of each (`book_chapters.space.measure`); the median of the readings over the plateau's scales.
>
> **To make it again:** `python3 book_figures.py core`.
<!-- /figure -->

Nothing moves much. By balls, the whole world reads 3.1 at iteration 500 and
2.8 at the end. Its core reads 2.8 and 2.5, and its 3-core 2.3 and 2.1. By
walks all three read between 1.7 and 2.0 at every moment. The bands of the
worlds are far wider than any change. A world's dimension is set by iteration
500, and each world keeps its own.

## The geometry of a world

A network has no space around it: agents are not anywhere, they are only
joined. Still, a network has a **geometry**: distances between its agents, a
way of growing outward from each, and thin places where it could be cut. The
viewer measures it in nine ways. This section reads them on the thirty worlds.
It holds the most telling ones against a **null**, the world's random twin:
exactly the same number of connections at every agent, wired at random. What
the world has and its twin lacks is what the rules built.

### Distances

[How far apart are two agents?](#how-far-apart-are-two-agents) found two
agents of a settled world a median 17.7 steps apart: that is the world's
**mean path length**. Two more numbers describe its distances
([Radius and diameter](path-length.md#radius-and-diameter)). The **diameter** is the
distance between the two agents furthest apart. The **radius** is how far the
most central agent has to reach to touch everyone.

<!-- figure geometry/distances -->
![Three distances](../figures/geometry/distances.svg)

**Three distances.** Every 25 iterations, in the 30 baseline worlds: the diameter, the radius and the mean path length, all estimated from breadth-first searches out of 8 to 16 spread agents ([Radius and diameter](path-length.md#radius-and-diameter)). For each, the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `diameter`, `radius` and `meanPathLength` of the rows with `phase` = 2.
> 2. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py geometry`.
<!-- /figure -->

Over the settled life, the median world has a radius of 36 and a diameter of
60. With the mean path length, all three are about twice those of the worlds
of 10,000 tokens (9.8, 17 and 29).

Are those long? Take the world of seed 1 after its last game, with 14,870
agents, and compute every distance exactly. Its mean path is 27.9 steps, its
radius 48 and its diameter 96. Now keep every agent's number of connections,
and wire them at random, five times over: a "configuration model", the world's
random twin. The mean path falls to **6.0**, the radius to 8.6 and the
diameter to 16.

So the world is **four and a half times as long** as its connections require.
The world of seed 1 at 10,000 tokens was twice as long (8.8 against 4.2). The
gap widens with size. A random network's distances grow with the logarithm of
its number of agents, a world's with a power of it. In worlds of up to 60,000
agents, they grow like the number of agents to the power 0.25 to 0.30. That is
how distances grow in a body of about three dimensions
([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)).

Random networks are "small worlds" because every connection is as likely to be
a long jump as a short one. In these worlds there are no long jumps. A child
is joined to its parent's neighbourhood
([How agents have children](how-agents-have-children.md#how-agents-have-children)),
so the network only ever grows locally, branch by branch.

### How a world grows around an agent

Stand on an agent and count the agents within *r* steps. This is the ball of
[How many dimensions the core has](#how-many-dimensions-the-core-has), seen up
close in one world. In a space of *d* dimensions the count grows like *r*^*d*.
In a tree, or in a random network, it grows exponentially: by about the same
factor with every step
([Dimension and curvature](ball-dimension-and-curvature.md)).

<!-- figure geometry/ball -->
![How many agents are within r steps](../figures/geometry/ball.svg)

**How many agents are within r steps.** The world with seed 1 after its last game (14,870 agents). From 24 agents spread evenly through the list of ids, a breadth-first search counts how many agents lie within r steps; the dots are the means over the 24. Left, both axes logarithmic: a power law r^d — growth like a d-dimensional space — is a straight line of slope d; red is the line least squares fits up to the radius where half the world is reached. Right, only the vertical axis logarithmic: exponential growth, as in a tree or a random network, would be a straight line there ([Dimension and curvature](ball-dimension-and-curvature.md)).

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frame `2999`; build the network from `edges`.
> 2. From every ⌊n/24⌋-th agent of the id-sorted list, breadth-first search; count the agents at distance ≤ r for r = 1 … 40; average over the 24 sources.
> 3. Fit ln V(r) = a + d·ln r by least squares for r with V(r) ≤ n/2.
>
> **To make it again:** `python3 book_figures.py geometry`.
<!-- /figure -->

In the world of seed 1, one step holds 6 agents on average: the agent and its
neighbours. Two steps hold 21, three 62, five 200 and ten 1,270. The count
grows as *r*^2.38 until half the world is reached, at 24 steps. Then it
flattens, as the balls run into the edges of the world. In the world of seed 1
at 10,000 tokens, two steps already held 70 agents, because most agents were
two steps from a hub. Half that world lay within seven steps. A power law
would run straight in the left panel, an exponential in the right one. The
growth here is closer to a power law, and
[How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)
reads it as a dimension.

### Four measures of a world's shape

From ball growth and from coverings come four numbers that the viewer records
every 25 iterations. Each is given for the median world, with the range of the
thirty worlds in brackets:

<!-- figure geometry/measures -->
![Four measures of a world's geometry](../figures/geometry/measures.svg)

**Four measures of a world's geometry.** Every 25 iterations, in the 30 baseline worlds: the dimension and the curvature read off how balls grow ([Dimension and curvature](ball-dimension-and-curvature.md)), the box dimension ([Box dimension](ball-dimension-and-curvature.md#box-dimension)), and the spectral gap, on a logarithmic axis ([The spectral gap](path-length.md#the-spectral-gap)). In each, the line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Thirty worlds](a-worlds-life.md#thirty-worlds)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](frames-and-stats.md)).
>
> 1. Take `dimension`, `ricciCurvature`, `boxDimension` and `spectralGap` of the rows with `phase` = 2 (measured every 25 iterations).
> 2. In each stretch, take the median, the 25th and 75th and the 5th and 95th percentiles of those means over the worlds that still have a value there (a world that died has none after its death).
>
> **To make it again:** `python3 book_figures.py geometry`.
<!-- /figure -->

- **Dimension from ball growth**: the viewer's reading of how balls grow over
  their first five steps. It is 3.4 (2.9 to 6.9; 3.0 at 10,000 tokens),
  against 3.7 in the founders' ring after the first game. Over their first few
  steps, balls grow about as fast as in three-dimensional space, or faster.
- **Curvature**: +1.4 (0.1 to 23), from −1.7 at the start. Positive curvature
  means that balls grow more slowly at larger *r* than at smaller, as on a
  sphere ([Dimension and curvature](ball-dimension-and-curvature.md)). Here
  that is mostly the bushy neighbourhoods of hubs, which fill the first steps
  of a ball. In a star, most agents are two steps from most others, through
  the centre (the 23). Read it as "the growth slows", not as a sphere.
- **Box dimension**, from how many boxes it takes to cover the world
  ([Covering a world with boxes](#covering-a-world-with-boxes)). It is 1.88
  (1.66 to 2.58; 1.85 at 10,000 tokens), from 2.48 in the ring. A straight
  line fits the counts well, with R² = 0.96.
- **Spectral gap**, which says how hard the world is to cut in two
  ([How easily a world could be cut](#how-easily-a-world-could-be-cut)). It is
  0.0012 (0.0010 to 0.0019), half that of the worlds of 10,000 tokens
  (0.0026), down from 0.080 in the ring.

Two dimensions, 3.4 and 1.9, for the same worlds. They do not contradict each
other; they look at different scales. Ball growth is fitted at radii 1 to 5,
where the bushy neighbourhoods of hubs make the world look high-dimensional.
Box covering runs from size 1 to 33, and at large sizes it counts the long,
thin branches, which make the world look low-dimensional. Neither number is
"the" dimension of a network that has no single one; both are indices for
comparing worlds. The rulers of
[How many dimensions the core has](#how-many-dimensions-the-core-has) read the
same worlds as 2.8 by balls. They use the typical ball over radii 3 to 10. The
viewer uses the mean ball over radii 1 to 5, which the few agents next to a
hub pull up.

### Covering a world with boxes

Cover the world with boxes. A box of size ℓ is every agent within (ℓ − 1)/2
steps of a centre. The larger the boxes, the fewer are needed: in a space of
*d* dimensions, their number falls like ℓ^−*d*. Halve the size of the boxes,
and a line needs twice as many, a sheet four times as many
([Box dimension](ball-dimension-and-curvature.md#box-dimension)).

<!-- figure geometry/boxes -->
![Covering one world with boxes](../figures/geometry/boxes.svg)

**Covering one world with boxes.** The world with seed 1 after its last game, covered by boxes: a box of size ℓ is every agent within (ℓ − 1)/2 steps of a centre, and centres are taken greedily, best-connected first, among the agents no box holds yet. Dots: how many boxes it takes for ℓ = 1, 3, 5, 9, 17 and 33; red: the least-squares line on logarithmic axes, whose slope is minus the box dimension ([Box dimension](ball-dimension-and-curvature.md#box-dimension)).

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frame `2999`; order the agents by degree, highest first.
> 2. For each ℓ: go down the order; every agent not yet in a box starts a new box, which takes every agent within (ℓ − 1)/2 steps of it; count the boxes.
> 3. Fit ln(boxes) on ln ℓ by least squares.
>
> **To make it again:** `python3 book_figures.py geometry`.
<!-- /figure -->

The world of seed 1 needs this many boxes of each size (of size 1, every agent
is a box alone):

| size of the boxes, ℓ | 1 | 3 | 5 | 9 | 17 | 33 |
|---|---|---|---|---|---|---|
| boxes needed | 14,870 | 5,135 | 2,004 | 756 | 185 | 42 |

On logarithmic axes these lie close to a line of slope −1.70: a box dimension
of about 1.7, between a line and a sheet. The world of seed 1 at 10,000 tokens
read 2.0.

### How easily a world could be cut

The **spectral gap** λ₂ measures, in one number, how hard a network is to cut
into two large parts. It also says how quickly a random walk on it forgets
where it started: after about 1/λ₂ steps
([The spectral gap](path-length.md#the-spectral-gap)).

The world of seed 1 has λ₂ = 0.00095. Its random twin has 0.031:
**thirty-three times larger** (thirty-two at 10,000 tokens). By Cheeger's
inequality, a small λ₂ guarantees a bottleneck: a large part of the world
joined to the rest by few connections.
[How a world breaks](how-a-world-breaks.md) finds those bottlenecks: whole
branches hanging on single connections. A world breaks, when it breaks, along
its poorest regions.

The gap also says something about the tokens. A game moves tokens almost like
a random walk ([Where the tokens flow](where-the-tokens-flow.md)). On this
network, a random walk needs about a thousand steps to forget its start. So
tokens could spread evenly across a whole world only over about a thousand
games, most of a run. Meanwhile the network changes in every game. About 2,300
connections are made and 1,600 cut per iteration, and the hubs change [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.")
in nearly every game. **Locally**, tokens settle within a few games, which is
why every agent's tokens follow its connections
([How properties scale together](how-properties-scale-together.md)).
**Globally**, the world is never at rest: far-apart regions are only loosely
coupled.

### Where the loops run

A world's independent loops are counted by its cycle rank
([The loops live in the core](#the-loops-live-in-the-core)). Over the settled
life, the median world has 10,000 of them. Its **loop density**, independent
loops per connection, is 0.42 (0.38 at 10,000 tokens). So 42% of the
connections are "spare", beyond what holding the world together needs. Where
do the loops run?

<!-- figure geometry/loops -->
![How many loops pass through an agent](../figures/geometry/loops.svg)

**How many loops pass through an agent.** A breadth-first spanning tree of the world with seed 1 after its last game leaves 11,297 connections over, each closing one loop of a cycle basis ([Loops](bridges.md#loops)). For every count x, the share of agents that at least x of those loops pass through. 37% of the agents lie on none of them.

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read frame `2999`; build a breadth-first spanning tree.
> 2. For every connection not in the tree, walk both ends up the tree to where they meet; every agent on the way, and the meeting point, lies on that loop.
> 3. Count the loops per agent; draw the share with at least x.
>
> **To make it again:** `python3 book_figures.py geometry`.
<!-- /figure -->

A breadth-first spanning tree of the world of seed 1 leaves 11,297 connections
over, and each closes one loop. Of its agents, 37% lie on none of those loops:
they hang on tree-like branches. One agent lies on 1,012 of them, so a tenth
of all the loops of the world run through a single hub. In the smaller world
of seed 1, at 10,000 tokens, half of them did. The number of loops through an
agent is heavy-tailed: most agents are on a few, a handful on hundreds.

## What we learned

- **The ring is gone at once, and a mesh with trees replaces it.** Within a
  hundred iterations, more than half of the agents are in the core: 57% after
  a game, 60% before it. Small trees hang off the core. A third of all agents
  are leaves, and a quarter of all connections are bridges. At a tenth of the
  size, a world was half tree and half web.
- **Experiment 5's thesis was refuted, on one clause.** Bridges and clustering
  were as tree-like as it predicted. But the core held most of the agents:
  narrowly so at 10,000 tokens, more clearly at 100,000. Births close loops: a
  child joined to two agents that are neighbours closes a triangle as it is
  born.
- **Connections are very unequal.** Most agents have one or two, and a few
  hubs have hundreds or thousands. Growth by copying makes a hierarchy of
  stars, with leaves hanging on hubs.
- **The core is a mesh within meshes.** It holds three quarters of the tokens
  and every loop. Inside it is a denser mesh, and inside that a denser one
  still, down to a knot of agents each joined to eight others. The trees
  around it are small, mostly a single step deep.
- **The mesh is a space.** By balls the core has two to three dimensions, a
  little fewer than the whole world, because trees add room without adding
  directions. By walks it reads a little under two (1.5 to 1.9 at the three
  sizes), as the world does. Its random twin has no dimension at all. So a
  world's dimension is not an artefact of its trees: it is the dimension of
  its core. By balls, it grows with the world's size, over the sizes run so
  far.
- **The world is stretched.** Two agents are a median 17.7 steps apart, 2.3
  times as far as in a random network of the same size. A token needs about
  eighteen games to go between them. Measured exactly, one world is four and a
  half times as long as its random twin (twice at 10,000 tokens). Nothing
  makes long jumps, and the gap widens with size.
- **It mixes slowly, through bottlenecks.** Its spectral gap is a thirty-third
  of its twin's. Tokens settle within neighbourhoods and not across the world,
  and far-apart regions are only loosely coupled.
- **For evolution, that cuts both ways.** A network that mixes slowly lets
  regions differ for a long time. That is how spatial structure protects
  variety, and how cooperation can grow among neighbours (Nowak and May 1992;
  [Do agents cooperate?](do-agents-cooperate.md)). But a network with
  bottlenecks also breaks easily, and whole branches can be lost at once
  ([How a world breaks](how-a-world-breaks.md)).
- **Anything lasting would have to live in the core, and the core is not
  safe.** A group of agents that holds together while its members change (P3
  in [What this notebook is about](what-this-notebook-is-about.md)) needs
  loops. The core is the only part of a world where a group is joined by more
  than one path. There, losing one connection does not cut it in two. Yet core
  agents die as often as leaves, 2.7% in every game. A part of the world that
  lasted would show as core agents that die less often than the rest. There
  are none but the hubs, which die about half as often as other agents
  ([Do the rich stay rich?](where-do-the-tokens-go.md#do-the-rich-stay-rich)).
  And a hub is a place, not an individual: the brain on it changes in nearly
  every game ([How the game is played](how-the-game-is-played.md)).

## Where to go next

- [How a world breaks](how-a-world-breaks.md): where the bottlenecks give way,
  and what a world loses when they do.
- [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md):
  the two rulers, tried on known spaces and then on the worlds.
- [How births build the network](how-agents-have-children.md): how a child is
  joined, and how births and games make and cut every connection.
- [Power laws, real and apparent](power-laws-real-and-apparent.md): whether
  the connections per agent follow a power law, and what makes the hubs.

## How this was made

Every figure of this page reads the thirty runs of Experiment 10, listed
below. They are worlds of the baseline B1 at 100,000 tokens, seeds 1 to 30,
run for 1,500 iterations each. The command `python3 gol_lab.py run E10` makes
them. [How many dimensions the core has](#how-many-dimensions-the-core-has)
also reads the three worlds of 409,600 tokens of Experiment 8. It quotes the
worlds of 10,000 tokens of Experiment 2 too. From the runs, three commands
draw the figures:

- `python3 book_figures.py shape` those of
  [What shape does the network take?](#what-shape-does-the-network-take);
- `python3 book_figures.py core` those of
  [The core of a world](#the-core-of-a-world);
- `python3 book_figures.py geometry` those of
  [The geometry of a world](#the-geometry-of-a-world).

Each figure says, folded beneath it, which runs it reads and how.

The thesis of Experiment 5 was first checked in the thirty worlds of 10,000
tokens of Experiment 2, listed last. There `python3 gol_lab.py analyse E05`
checks it. In the worlds of 100,000 tokens, `python3 book_figures.py shape`
checks it again.

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

### The runs of Experiment 5

<!-- runs E05 -->
> [!info]- The runs behind this page
> These are the runs of Experiment 2; nothing new was run for this page.
>
> **30 runs.** Every condition below is run once for every seed (1 to 30), for 3,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it.
>
> - **baseline** — changes nothing: B1 as it is; 10,000 tokens. Runs `B1-10000-s001` … `B1-10000-s030`.
>
> To make these runs again: `python3 gol_lab.py run E02`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | baseline | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
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
> | iterations | 3,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B1.
<!-- /runs -->

## How our view changed

- **11 October 2026.**
  [What shape does the network take?](#what-shape-does-the-network-take) and
  [The geometry of a world](#the-geometry-of-a-world) were first written from
  the thirty worlds of 10,000 tokens. On this day they were rewritten from the
  thirty of 100,000 tokens, with the small worlds' numbers beside the new
  ones. Some claims changed with the size of a world:
  - A world was "half tree, half web": a core of about half the agents, and a
    third of the connections bridges. At 100,000 tokens it is more of a web:
    the core holds more than half the agents, and a quarter of the connections
    are bridges.
  - The growth of balls was "neither a clean power law nor a clean
    exponential", in worlds too small for either. In the large worlds it is
    closer to a power law, and is read as a dimension.
  - The positive curvature was put down to the world's finite size: balls run
    out of world before they reach half of it. In the large worlds it comes
    mostly from the bushy neighbourhoods of hubs.
  - The world of seed 1 had a box dimension of "about 2, like a sheet". At
    100,000 tokens it has about 1.7, between a line and a sheet.
- **14 October 2026.** A pass through the whole notebook checked every
  statement against the numbers it rested on, and corrected these:
  - Experiment 5's page said that a world is "thirty times easier to cut in
    two" than a random network with the same connections. What was measured is
    the spectral gap, a thirty-third of the random network's. It shows that
    the world has bottlenecks; it is not a factor of how easily it is cut.
  - The page on the core said that every connection of the core lies on a
    loop, "so cutting one connection never splits the core". A connection on a
    path between two loops is a bridge, and cutting it splits the core in two.
  - It said that the core has between two and three dimensions by both rulers.
    By walks it reads a little under two, as the whole world does.
  - It said that no part of the core dies less often than the rest. The hubs
    do, about half as often as other agents; but a hub is a place, not an
    individual.
- **23 October 2026.** When the three pages became one, their numbers were
  checked against the results the figures are drawn from (`book/results/`),
  and these were corrected:
  - The core of the worlds of 10,000 tokens reads 1.9 by balls (1.94), not
    2.0. The random twins read 3.2 by walks (3.248), not 3.3. And the middle
    half of the cores of 100,000 tokens reads 2.2 to 2.9 by balls (2.9496 at
    its top), not 2.2 to 3.0.
  - The box counts of the world of seed 1 fall along a slope of −1.70
    (−1.7045), not −1.71.
  - The page said that news and tokens "take about eighteen games to cross" a
    world. But 17.7 steps is the mean distance between two agents; to cross a
    world takes up to its diameter, 60 steps. A token needs about eighteen
    games to go from one agent to another, on average.

<!-- turns -->
---

[The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Power laws, real and apparent](power-laws-real-and-apparent.md) →

**Also about:** [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (19)
> [Questioning the mechanics](questioning-the-mechanics.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [How births build the network](how-agents-have-children.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Where tokens flow](where-the-tokens-flow.md) · [How a world breaks](how-a-world-breaks.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Stars and hubs](stars-and-hubs.md) · [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) · [Bridges, cut risk, loops and the core: how a network can be cut](bridges.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [The random twin: is a world a space?](random-twin.md) · [Three ways to measure a network's dimension](ball-dimension-and-curvature.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [One world, many colours](one-world-many-colours.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
