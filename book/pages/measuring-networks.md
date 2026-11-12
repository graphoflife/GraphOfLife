# Measuring a network: connections, hubs, distances and cuts

> [!question] Questions of this page
> - How many connections do agents have, and how do they cluster?
> - What are hubs and stars?
> - How far apart are two agents, and how wide is a world?
> - Where can a network be cut?

## In short

How the notebook measures the shape of a world's network: how many connections agents have and how they cluster, the hubs and stars, how far apart agents are, and where a network can be cut.

Its sections:

- [An agent's connections: degree and clustering](#an-agents-connections-degree-and-clustering)
- [Stars and hubs](#stars-and-hubs)
- [Distances in a network: path length, radius and diameter, the spectral gap](#distances-in-a-network-path-length-radius-and-diameter-the-spectral-gap)
- [Bridges, cut risk, loops and the core: how a network can be cut](#bridges-cut-risk-loops-and-the-core-how-a-network-can-be-cut)

## An agent's connections: degree and clustering

An agent's **degree** is the number of its connections; in a settled
[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world of 100,000 tokens, the mean is about 3.5. Most agents have one
or two, and a few **hubs** hundreds or thousands. **Clustering** says how
the connections are arranged: whether an agent's neighbours are joined to each
other. Whether hubs are joined to hubs, who gains new connections and how an
agent's tokens and triangles grow with its connections were measured only by
studies now in the Archive. Their sections, assortativity, preferential
attachment and scaling relations, are in
[Measures of the archived studies](measures-of-the-archived-studies.md#networks).

### Degree

The **degree** of an agent is the number of connections it has, which is the
number of its neighbours. It is the simplest thing to know about a place in a
network, and much depends on it. An agent stakes only on its own node and its
neighbours' nodes, and only its neighbours can stake on its node.

#### Degree, mean degree and density

For an agent *u* with neighbours *N*(*u*), the degree is
deg(*u*) = |*N*(*u*)|: the number of agents in that set. Every connection has
two ends, so summing the degrees counts every connection twice:

$$
\sum_{u} \deg(u) = 2\,|E| ,
$$

where |*E*| is the number of connections. The **mean degree** of a world with
|*V*| agents and |*E*| connections is therefore

$$
\bar k = \frac{1}{|V|}\sum_u \deg(u) = \frac{2\,|E|}{|V|} ,
$$

recorded as `meanDegree`.

**Example.** Take the small network of
[The network](the-world.md#the-network): the agents *a*, *b* and *c* form a
triangle, and *d* hangs on *c*. Their degrees are 2, 2, 3 and 1. These add up
to 8, twice the 4 connections, so the mean degree is 8 / 4 = 2.

A settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world has about 1.7 connections for every agent at
100,000 tokens (1.6 at 10,000). Each connection has two ends, so the mean
degree is about 3.5 (3.3). This is what the notebook means by *links per
agent*.

Three more numbers describe the degrees of a frame: `medianDegree`, the
[median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread); `minDegree`, the fewest; and `maxDegree`,
the most, which the largest hub has. In a world of one piece with more
than one agent, `minDegree` is at least 1. It is 1 while some agent is a leaf,
and about a third of a settled B1 world's agents are
([Hubs and leaves](#hubs-and-leaves)). Under all of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s rules, in
the worlds of Experiment 14, it was almost always 2. There, nearly every agent
sat on a loop or between loops, and leaves were rare
([Nobody cut off](building-b2-and-the-search-for-a-healthy-rich-world.md#nobody-cut-off)). In the small
example, the median is 2, the fewest 1 and the most 3.

The **density** is the share of all possible pairs of agents that are joined:

$$
\texttt{density} = \frac{|E|}{|V|(|V| - 1)/2} = \frac{\bar k}{|V| - 1} .
$$

In words: |*V*| agents make |*V*|(|*V*| − 1)/2 pairs, and |*E*| of them are
joined. In the small example, 4 of the 6 pairs are joined, a density of 0.67.
A world of 1,300 agents with mean degree 3.3 has a density of 0.0025: of
every 400 pairs of agents, one is joined. Density falls as a world grows, even
when every agent keeps the same number of connections. That is why the
notebook mostly uses the mean degree.

#### The degree distribution

How many agents have degree 1, 2, 3, …? There are two ways of drawing it:

- **The histogram**: for each *d*, the share of agents with degree exactly
  *d*.
- **The complementary cumulative distribution** (CCDF): for each *d*, the
  share of agents with degree **at least** *d*,

  $$
  P(\deg \ge d) = \frac{|\{u : \deg(u) \ge d\}|}{|V|} .
  $$

  In words: count the agents with *d* connections or more, and divide by all
  agents. It starts at 1 for *d* = 1, since every agent of a world in one
  piece has at least one neighbour. It falls to 1/|*V*| at the largest
  degree, if a single agent has it.

In the small example, the shares with at least 1, 2 and 3 connections are 1,
3/4 and 1/4.

The CCDF is the better picture for wide distributions. It needs no choice of
bins, it is smooth where the histogram is noisy, and on
[logarithmic axes](statistics.md#logarithmic-axes-and-power-laws) it shows the shape of the tail.
Suppose the share of agents with degree *d* falls like *d*^−γ, a
power law. Then the CCDF falls like *d*^−(γ−1). On log–log axes it is
a straight line, with slope −(γ − 1).

#### Hubs and leaves

A **leaf** is an agent of degree 1. It hangs on the network by a single
connection, and the cut of that one connection cuts it off (`leaves` in the
statistics). A **hub** is an agent of very high degree. There is no exact line
for a hub, but there is one for a star, a world held together by one
([Stars and hubs](measuring-networks.md#stars-and-hubs)).

A settled baseline world has both. After the last
game of the thirty baseline worlds of 100,000 tokens, 35% of all
agents had exactly one connection and 23% two. Another 4.7% had ten or more,
and the largest hub had 10,735
([Connections per agent](b1-the-network-size-and-scaling.md#connections-per-agent)).
At 10,000 tokens, these were 37%, 23%, 4% and 1,399.

#### Where degree is used

Degree enters the [health bar](measuring-health.md#the-health-bar), the [even split](measuring-tokens.md#the-even-split), the
[random twin](measuring-space.md#the-random-twin-is-a-world-a-space) and
[what a brain sees](rules-in-detail.md#what-a-brain-sees).

<!-- used degree -->
- [One iteration, step by step](one-iteration.md#phase-1-reproduction)
- [Statistics: the tools of the notebook](statistics.md#distributions-on-loglog-axes)

> [!info]- And in the Archive (4)
> [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#what-a-power-law-is-and-why-it-is-hard-to-see) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md#what-it-would-take) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#children-as-missiles) · [Measures of the archived studies](measures-of-the-archived-studies.md#the-idea-of-assortativity)
<!-- /used -->

### Clustering

Are the neighbours of an agent also neighbours of each other? In a circle of
friends they often are; in a tree they never are. **Clustering** measures how
often.

#### Triangles and triples

A **triangle** is three agents all joined to each other. A **connected
triple** centred on *u* is a pair of *u*'s neighbours: two connections that
meet at *u*. An agent of degree *d* is the centre of

$$
\binom{d}{2} = \frac{d(d-1)}{2}
$$

triples. Each of its *d* neighbours pairs with each of the other *d* − 1, and
that counts every pair twice. Every triangle closes three triples, one at each
corner.

#### Transitivity

The clustering this notebook uses is the **transitivity**: the share of all
connected triples in the network that are closed into triangles,

$$
C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}
  = \frac{3\,t}{\sum_u \binom{\deg(u)}{2}} .
$$

Here *t* is the number of triangles, and the factor 3 counts each triangle at
each of its corners. *C* is between 0 and 1. It is 0 when there is no triangle
at all (a tree, for example). It is 1 when every pair of neighbours is joined
(every connected piece is complete). It is recorded as `transitivity`, with
the number of triangles as `triangles`, on every 25th
iteration.

An agent's own **clustering coefficient** asks the same of its neighbours
alone. For an agent with *k* ≥ 2 neighbours, it is its triangles divided by
its *k*(*k* − 1)/2 pairs of neighbours. The transitivity is the mean of these
coefficients, each weighted by the agent's pairs, so the hubs weigh most.
[Scaling relations](measures-of-the-archived-studies.md#scaling-relations-between-agents-properties) follow how
the coefficient falls with *k*.

#### Examples of clustering

- **A triangle** of three agents: one triangle, three triples (one at each
  corner), *C* = 3·1/3 = 1.
- **A star**: one centre joined to 4 leaves, and no other connection.
  The centre has 6 triples, the leaves none, and there is no triangle:
  *C* = 0.
- **The small network** of [Degree](#degree-mean-degree-and-density): *a*,
  *b* and *c* form a triangle, and *d* hangs on *c*. Its 5 triples are one at
  *a*, one at *b* and three at *c*. The triangle closes three of them:
  *C* = 3·1/5 = 0.6.
- **The founders' ring** of a world of 10,000 tokens, before rewiring. Every
  founder has degree 4, so 6 triples, of which 3 are closed. Over the ring
  that is *C* = 0.5. Rewiring 37 connections of the real ring of seed 1
  lowered it to 0.27 ([The starting ring](rules-in-detail.md#the-starting-ring)). At 100,000
  tokens a founder starts with 10 neighbours, and the rewired rings read 0.34
  in the median world.
- **A random network** has the same number of agents and connections, each
  connection placed at random. Two neighbours of an agent are then joined
  about as often as any two agents. So *C* is close to the probability that
  two given agents are joined: the density, *k̄*/(|*V*| − 1). For 1,300
  agents of mean degree 3.3, that is about 0.003.

#### Why clustering matters

Loops make a network hard to cut, and a triangle is the shortest loop: in a
triangle, every connection has a way round it. A tree-like network, with few
loops, falls apart when a connection is cut ([Bridges](measuring-networks.md#bridges-cut-risk-loops-and-the-core-how-a-network-can-be-cut)). B2
builds on this: it cuts an unused connection only where its two ends share at
least two neighbours. Each shared neighbour closes a triangle with the
connection, so the cut leaves two ways round
([`prune_needs`](settings.md#prune_needs)).

A settled baseline world lies between a tree and a web. At 100,000 tokens its
clustering is low, 0.039. Yet that is more than a hundred times what a random
network of its size would have, about 0.0002. At 10,000 tokens it is 0.075.
Births close triangles: a child joined both to its parent and to one of its
parent's neighbours closes one the moment it is born
([Clustering](b1-the-network-size-and-scaling.md#clustering)). Small trees
hang off the clustered core, and most of their agents are a single step from
it ([How far the trees hang](b1-the-network-size-and-scaling.md#how-far-the-trees-hang)).

#### Where clustering is used

In B1's worlds clustering stays below 0.1, as Experiment 5 predicted
([Was the thesis right?](b1-the-network-size-and-scaling.md#was-the-thesis-right-the-shape-of-the-network)).

<!-- used clustering -->
- [The ideas this builds on](the-ideas-this-builds-on.md#networks)
- [The world: tokens, agents and a network](the-world.md#how-a-world-begins)
- [The rules in detail](rules-in-detail.md#why-a-small-world)
- [How runs are made, recorded and costed](runs-and-records.md#the-statistics-file)
- [How worlds are measured](how-worlds-are-measured.md#step-1--a-row-of-statistics-per-phase)

> [!info]- And in the Archive (4)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#what-the-second-edition-changed) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#what-was-expected-the-shape-of-the-network) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md#agents) · [Measures of the archived studies](measures-of-the-archived-studies.md#the-four-relations)
<!-- /used -->

## Stars and hubs

Most agents have one or two connections; a few, the **hubs**, have hundreds or
thousands. A world whose best-connected agent is joined to a tenth of it or
more is a **star**, and seven of the thirty baseline worlds of 100,000 tokens
were. A star's centre is a place, not an individual: the [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") on it changes
nearly every game, while its connections pass from parent to newborn child. The
[health bar](measuring-health.md#the-health-bar) counts any agent joined to more than 5% of the others as a fault.

### Hubs

An agent's [degree](measuring-networks.md#an-agents-connections-degree-and-clustering) is its number of connections. In a settled
baseline world of 100,000 tokens the median agent has 2 and the mean is 3.5, but
the best-connected agent of the median world has 640 (from 130 to 11,000
between worlds; [The largest and the smallest](b1-lives-and-wealth.md#the-largest-and-the-smallest)).
An agent that far above the rest is a **hub**. There is no exact line for a hub;
there is one for a star.

Hubs are places. Under the near-[even split](measuring-tokens.md#the-even-split), the agent on a node
with *d* connections is one of *d* + 1 stakers of about equal size, and keeps
its node about as often as one of them would: with fifty connections or more,
in 2.0% of games
([Where the tokens flow](b1-the-game-and-its-flows.md#who-keeps-their-node)).

### Stars

A world is a **star** when its best-connected agent is joined, on average over
the world's [settled life](statistics.md#the-settled-life-of-a-world), to at least a tenth of the agents
alive (in the code, `STAR = 0.1` in `book_chapters/common.py`). By this measure
7 of the 30 worlds of Experiment 10, at 100,000 tokens, were stars, and 10 of
the 26 surviving worlds of 10,000 tokens of Experiment 2. In the large worlds a star sets the
shape: across the thirty, the share of agents in the
[2-core](measuring-networks.md#core-trees-and-leaves) falls as the share joined to the largest hub
rises, with a correlation of −0.96
([How much does the seed decide?](b1-the-network-size-and-scaling.md#how-much-does-the-seed-decide)).

The largest star is the world with seed 28, whose centre was joined to 58% of
all agents on average, for a thousand iterations. At iteration 1,122 it was
joined to 20,666 of the 25,958 agents alive and held 20,286 tokens, a fifth of
all; 94% of its neighbours were leaves, holding 2.8 tokens each on average. A
coalition of neighbours won its node in each of the hundred games from then
on, and the node carried 78 different [genotypes](measuring-evolution.md#genotypes-families-and-the-common-ancestor "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.") in them
([The largest and the smallest](b1-lives-and-wealth.md#how-a-star-works)).

### How a star is handed down

An agent gains connections as its neighbours have children, about in
proportion to those it has, but loses them a little faster, so the well
connected drift back down
([How the network grows](b1-lives-and-wealth.md#who-gains-and-who-loses)).

A star's centre is kept another way. Now and then it has a child and hands it
nearly all its connections ([How agents have children](b1-lives-and-wealth.md#how-births-build-the-network)):
at iteration 1,153 the centre of seed 28 went from 18,913 connections to 219,
and its child, of age 0, took 18,865. Over the settled lives of the thirty
worlds, first place in connections changed hands 1,299 times, 21% of them to a
child of the agent before and 16% to a newborn one. A line of first agents,
each the child of the one before, is a **dynasty**; dynasties lasted a median
of 7 iterations, and the longest 942.

### Why the health bar counts a hub as a fault

The author's picture of a world that might keep evolving asks for a network
that is resilient, decentralised and close to three-dimensional, with few
single points of failure. So the
[health bar](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers)
allows no agent to be joined to more than 5% of the others. If one agent holds
links to many others, the world depends on it: when that hub dies, everything
changes at once ([What makes a world interesting](rounds-1-and-2-of-the-screen.md#the-six-criteria)).
And a star is mostly leaves, and short: in seed 28 only 19% of the agents were
in the core and 72% were leaves, and two agents were 4.2 steps apart on
average, against 12 to 26 steps in the worlds without a star
([What shape does the network take?](b1-the-network-size-and-scaling.md#the-shape-of-the-network)).

[B2](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break) keeps hubs down: a newborn's links are drawn
at random, a brain may choose at most 8 links
([`valence`](settings.md#valence)), and no parent hands its links to a child.

### Where it is used

<!-- used stars-and-hubs -->
- [Welcome](../README.md#graph-of-life--the-research-notebook)
- [What this notebook is about](what-this-notebook-is-about.md#in-short)
- [How this notebook is written](how-this-notebook-is-written.md#the-rules-of-the-research)
- [Which worlds are healthy? Three orders](which-worlds-are-healthy.md#in-short)
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md#in-short)
- [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md#11--roles-in-the-network)
- [Questioning the mechanics](questioning-the-mechanics.md#hubs-are-places)
- [Every setting](settings.md#birth_links)
- [What keeps wealth spread?](what-keeps-wealth-spread.md#without-coalitions)
- [Measuring tokens: inequality and flow](measuring-tokens.md#a-lazy-random-walk)
- [Particles: patterns that outlast what carries them](particles.md#in-short)
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md#room-to-go-on)
- [Rules without numbers](rules-without-numbers.md#why-ask)
- [Statistics: the tools of the notebook](statistics.md#what-thirty-seeds-can-see)

> [!info]- And in the Archive (11)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#brains-that-cannot-see) · [B1: lives and wealth](b1-lives-and-wealth.md#stars) · [B1: the game and its flows](b1-the-game-and-its-flows.md#two-agents-of-one-module-games-later) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#the-worlds-of-a-giant-hub) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md#but-some-genotypes-get-big) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#what-the-pilots-showed) · [Rounds 1 and 2 of the screen, and the criteria they were judged by](rounds-1-and-2-of-the-screen.md#the-six-criteria) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md#what-the-switches-did-to-the-world) · [Cooperation, help and gliders (side study)](cooperation-help-and-gliders.md#what-was-tried-to-make-helping-pay) · [Space at scale (side study)](space-at-scale.md#the-worlds-and-the-bodies-beside-them) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md#between-worlds)
<!-- /used -->

## Distances in a network: path length, radius and diameter, the spectral gap

The **distance** between two agents is the fewest connections on a path from
one to the other. The **mean path length** is its average over pairs, the
**diameter** its largest value, and the **radius** the reach of the most
central agent. The **spectral gap** says how hard the network is to cut in
two, and how fast a random walker on it forgets where it began. In a settled
B1 world of 100,000 tokens, two agents are a median 17.7 steps apart, the
radius is 36 and the diameter 60.

### Path length

How far apart are two agents of a world? Agents have no positions, only
connections. A token moves along them, at most one connection in a game. So
the natural measure of distance is a count of steps along connections.

The **distance** d(*u*, *v*) between two agents is the smallest number of
connections on a path from one to the other. The **mean path length** is
the average distance between agents, over every pair of them.

**Example.** Five agents stand in a row, a–b–c–d–e, each joined to the next.
The distance from a to e is 4, and from b to d it is 2. Of the 10 pairs of
agents, four are 1 step apart and three are 2 steps apart. Two pairs are 3
steps apart, and one pair is 4. The distances add up to 20, so the mean path
length is 20/10 = 2.

#### Breadth-first search

The distances from one agent *s* to every other are found by a
**breadth-first search**:

1. *s* is at distance 0;
2. its neighbours are at distance 1;
3. their neighbours not yet reached are at distance 2;
4. and so on, until every agent is reached.

It takes time proportional to the number of agents plus the number of
connections. In the row a–b–c–d–e, a search from c finds b and d at
distance 1, and a and e at distance 2.

#### The estimate the notebook uses

The exact mean over every pair of agents needs a search from every agent.
That is too slow for worlds of thousands of agents, every 25 iterations.
So `meanPathLength`, which a run records on the rows of every 25th iteration
([What a run records](runs-and-records.md#what-a-run-records)), is estimated from a spread of
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

After the cleanup the world is one connected piece, so every agent can be
reached from every source.

**Example.** The world of seed 1 of Experiment 10 ended with 14,870 agents
and 26,166 connections. So *n* + 2|*E*| = 67,202, and 250,000 / 67,202 ≈ 3.7
rounds to 4: *S* = 8, the least allowed. The step is ⌊14,870 / 8⌋ = 1,858.
The searches start at positions 0, 1,858, 3,716, … up to 8 × 1,858 = 14,864,
which is still in the list: 9 searches.

#### Near or far?

How many steps count as far? That depends on what the network could have
been.

In a **random network**, a connection is as likely to be a long jump as a
short one. Let *k̄* be the mean degree, the mean number of connections per
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

In a lattice, a regular grid of *d* dimensions, a lump of *N* agents is about
*N*^(1/*d*) agents wide. A cube of a thousand agents, for example, is ten
agents wide. There, distances grow like a power of the number of agents. So
how distances grow with the size of a world says what kind of network it is.

The founders' ring a world starts on is a small world by design. In the world
of seed 1 at 10,000 tokens, a plain ring of its 100 founders would have
them 12.9 steps apart on average. With its shortcuts, they started 4.4 steps
apart ([The starting ring](rules-in-detail.md#the-starting-ring)).

#### Where path length is used

Two agents of a settled B1 world of 100,000 tokens are a median 17.7 steps
apart, 2.3 times as far as in a random network
([Distances](b1-the-network-size-and-scaling.md#distances)). At 10,000
tokens it is 9.8 in the median world, and 6.5 to 13.0 across the 26 worlds
that lived.

<!-- used path-length -->
No page links here yet.
<!-- /used -->

### Radius and diameter

The mean path length is a typical distance. Two more numbers describe the
extremes. How far apart are the two agents furthest from each other? And how
far must the most central agent reach to touch everyone? Both start from the
distance d(*u*, *v*): the fewest connections on a path from *u* to *v*
([Path length](#path-length)).

#### Eccentricity

The **eccentricity** of an agent is its distance to the agent furthest from
it:

$$
\varepsilon(u) = \max_{v} d(u, v) .
$$

In words: search outward from *u*, and count the steps to the last agent
reached. An agent at the centre of the network has a small eccentricity; one
at the tip of a long branch a large one.

#### The largest and the smallest eccentricity

- The **diameter** is the largest eccentricity — the distance between the two
  agents furthest apart: diam = max_u ε(*u*).
- The **radius** is the smallest eccentricity — how far the most central agent
  has to reach to touch everyone: rad = min_u ε(*u*).

They are always related by rad ≤ diam ≤ 2·rad. Going from any agent to any
other through the most central one takes at most twice the radius.

**Example.** In the row of 5 agents a–b–c–d–e, the eccentricities are 4, 3, 2,
3 and 4. So the diameter is 4 and the radius 2, at c.

#### How the viewer estimates them

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

#### Where radius and diameter are used

Over the settled life of B1's worlds of 100,000 tokens, the median world
has a radius of 36 and a diameter of 60, about twice those at 10,000
tokens ([Distances](b1-the-network-size-and-scaling.md#distances)).

<!-- used radius-and-diameter -->
- [How runs are made, recorded and costed](runs-and-records.md#the-statistics-file)

> [!info]- And in the Archive (2)
> [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#distances) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#mending-a-death)
<!-- /used -->

### The spectral gap

How hard is a network to cut into two halves? Counting bridges answers for
cuts of a single connection. The **spectral gap** λ₂ answers for every way of
dividing the network at once. It is also the speed at which anything that
spreads along the connections — a random walker, heat, tokens — forgets where
it started.

#### Step by step

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

#### The definition

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

#### Cheeger's inequality

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

#### Examples

- A complete network of *n* agents, each joined to every other:
  λ₂ = *n*/(*n* − 1), about 1.
- A ring of *n* agents: λ₂ = 1 − cos(2π/*n*) ≈ 2π²/*n*², tiny for large *n*.
  A ring is easy to cut (twice) into two arcs.
- A grid of *d* dimensions with *N* agents: λ₂ shrinks like *N*^(−2/*d*); the
  ring is the case *d* = 1. A world grown into a well-knit body of three
  dimensions would show this
  ([Physical inspiration](physical-inspiration.md)).

#### How it is computed

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

#### Where the spectral gap is used

A settled B1 world has a spectral gap of about 0.0012, so a random walk on
it needs some 800 steps to forget its start
([Four measures of a world's shape](b1-the-network-size-and-scaling.md#four-measures-of-a-worlds-shape)).

<!-- used the-spectral-gap -->
- [Physical inspiration](physical-inspiration.md#a-well-knit-world)
- [How runs are made, recorded and costed](runs-and-records.md#the-statistics-file)

> [!info]- And in the Archive (2)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#the-baseline-world-in-one-paragraph) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#four-measures-of-a-worlds-shape)
<!-- /used -->

## Bridges, cut risk, loops and the core: how a network can be cut

A **bridge** is a connection on no loop: cut it, and the network falls into
two pieces. In a settled B1 world of 100,000 tokens, a quarter of the
connections are bridges. The **cut risk** is the largest share of the agents
that one bridge holds to the rest: a ninth, in the median world. The **cycle
rank** counts the independent **loops**: how many connections, the right
ones, a network could lose and still be one piece. Peel away every agent with
a single connection, again and again, and the **core** is left. Every loop
lies in it, and **trees** of agents hang off it.

### Bridges

Think of a network as towns joined by roads. Most roads have a detour: close
one, and traffic goes round. A road with no detour is a **bridge**. Close it,
and the towns on its far side are cut off. Bridges are where a world is
fragile.

#### What a bridge is

Take a network in one piece. A connection {*u*, *v*} is a **bridge** if, once
it is removed, *u* and *v* can no longer reach each other. Put another way:
it lies on no [loop](#loops). A loop through it would be a second way from *u*
to *v*.

Three rules follow. Every connection of a tree is a bridge, and so is every
leaf's single connection. In the 2-core, every agent lies on a loop or on a
path between loops ([Core, trees and leaves](#core-trees-and-leaves)). There,
only the connections on a path that joins two loops are bridges.

**Example.** This network of 9 agents and 9 connections is the example of
the whole page:

```
  a — b — c — d
  |       |
  e ——————f — g — h
              |
              i
```

The connections a–b, b–c, c–f, f–e and e–a form a loop, so none of them is a
bridge. The other four are bridges. Three of them, c–d, g–h and g–i, each hold
a single leaf. The fourth, f–g, holds g, h and i.

#### How bridges are found

Removing each connection in turn and checking what is still joined would take
a walk through the whole network for every connection. **Tarjan's algorithm**
(1974) finds every bridge in a single depth-first walk.

The walk numbers every agent in the order it first reaches it: disc(*u*). It
also gives each agent a "low-link", low(*u*). Look at *u* and every agent
below it in the walk's tree. The low-link of *u* is the smallest disc number
among them, and among the agents they reach by one connection outside the
walk's tree. The connection from *u* down to its child *v* in the walk is a
bridge exactly when

$$
\text{low}(v) > \text{disc}(u),
$$

that is, when nothing below *v* reaches back to *u* or above it. The walk
takes time proportional to the number of agents plus the number of
connections.

In the example, let the walk start at a. It goes to b, c and d, back to c, on
to f and e, back to f, and last to g, h and i:

| agent | a | b | c | d | f | e | g | h | i |
|---|---|---|---|---|---|---|---|---|---|
| disc | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| low | 0 | 0 | 0 | 3 | 0 | 0 | 6 | 7 | 8 |

From e the walk sees a, numbered 0, so e, f, c and b all get the low-link 0.
Below g, nothing reaches back above g: low(g) = 6 is more than disc(f) = 4, so
f–g is a bridge. By the same test, so are c–d, g–h and g–i, and no other
connection.

#### Bridges in the worlds

A run records `bridges`, the number of bridges, in its statistics file, on the
rows of every 25th iteration. The notebook divides it by the number of
connections, `bridges/edges`: the share of all connections that are bridges.
Count them after the game, over a B1 world's settled life. At 100,000 tokens
about a quarter of the connections are bridges: 26%, some 6,400 of them. At
10,000 tokens it is 31%
([Core, leaves and bridges](b1-the-network-size-and-scaling.md#core-leaves-and-bridges)).
Most bridges hold a single leaf.

Experiment 5 predicted that more than a fifth of all connections would be
bridges, and in every world they were
([Was the thesis right?](b1-the-network-size-and-scaling.md#was-the-thesis-right-the-shape-of-the-network)).
The viewer can colour every connection by whether it is a bridge
([What the viewer can colour by](measures-atlases-and-assembly.md#what-the-viewer-can-colour-by)).

A count of bridges does not say how much hangs on each. For every bridge, the
program also works out how many agents lie behind it. The largest share of
the world a single cut can sever is the [cut risk](#cut-risk).

### Cut risk

Counting bridges says how many connections lie on no loop. It does not say how
badly the network would break if one were cut. A bridge to a single leaf costs
one agent; a bridge between two halves costs half the world. The **cut risk**
measures the worst case.

#### What the cut risk is

Removing a bridge splits the network into two pieces. Call the smaller one the
part **behind** the bridge. The cut risk is the largest such part, over all
bridges, as a share of all agents:

$$
\texttt{cutRisk} = \max_{\text{bridges } e} \frac{\min\bigl(|A_e|, |B_e|\bigr)}{|V|} ,
$$

where *A*ₑ and *B*ₑ are the two pieces left by removing *e*, and |*V*| is the
number of agents. In words: for each bridge, take its smaller side and divide
by all agents; the cut risk is the largest of these. It is 0 when there is no
bridge, and at most ½ — a network split down the middle by one connection.

**Example.** In the network drawn under [Bridges](#bridges), c–d, g–h and g–i
each hold one agent of 9. The bridge f–g holds three: g, h and i. So the cut
risk is 3/9, a third.

The formula assumes a network in one piece, as a world is after every cleanup.
Read before the cleanup, the network can be in several pieces. The program
then takes *A*ₑ to be the agents the walk reached through the bridge. *B*ₑ is
every other agent of the world, in the same piece or not.

#### When the cut risk is read

A run records the cut risk at two moments of a phase:

| field | read on | when | on which rows |
|---|---|---|---|
| `cutRisk` | the frame: the world as the run records it | at the end of the phase, after its cleanup, when only the largest piece is left | every 25th iteration |
| `cutRiskBefore` | the engine's network, as it runs | just before the cleanup, when nobody has been removed yet | every row |

In a game's row, `cutRiskBefore` is read after the game has been played and
after its unused connections have been cut. In B1, every connection that
carried no token in the game is cut at that point
([`prune_after`](settings.md#prune_after)). Only after the reading does the
cleanup remove anybody. In a reproduction row, it is read after the births,
just before that phase's cleanup.

Why two readings? A frame shows the world after the cull: the cleanup's
removal of every piece but the largest. A cut that removes a whole side of a
bridge changes `cutRisk` and the death toll in the same row. Which was cause
and which effect, the row cannot say. The reading `cutRiskBefore` is taken on
a network the cull has not touched yet. So it can be read as a predictor of
the cull rather than a consequence of it.

It is not read before the game, though. The game's unused connections are cut
first, so it cannot foresee those cuts. A region whose every connection to the
rest went unused is already a piece of its own by then. Measured as above, the
piece can count as if one connection held it. For the network the game starts
from, read the reproduction row of the same iteration. Its `cutRiskBefore` is
read after the births, and its `cutRisk` on that very network.
[How a world breaks](b1-the-network-size-and-scaling.md#does-a-fragile-world-lose-more)
uses the game's reading. In the worlds of 10,000 tokens it checked it against
a reading at the start of the game, and the picture is the same.

#### Why the cut risk matters

The cleanup keeps only the largest piece of the network. Take a bridge with a
tenth of the world behind it. In B1, that tenth of the world dies the moment
no token crosses the bridge in a game. How often does that happen? In the
median of the thirty B1 worlds of 100,000 tokens (Experiment 10), the cut risk
is 11% (12% at 10,000 tokens). Yet over 30,000 settled games, its correlation
with the share of agents a game cut off is only 0.09
([How a world breaks](b1-the-network-size-and-scaling.md#how-a-world-breaks)). A bridge with much behind it is
busy, and a busy connection is almost never left unused.

The [health bar](measuring-health.md#the-health-bar) asks that no single link's loss cut off more
than 1% of a world. It asks the same of agents: no agent's loss may cut off
more than 2%. B2 never lets a bridge lapse. There, an unused connection is cut
only where its two ends share at least two neighbours
([`prune_needs`](settings.md#prune_needs)). The two ends of a bridge share
none, since a shared neighbour would put the bridge on a triangle
([A world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break)).

#### How the cut risk is computed

Tarjan's walk finds every bridge
([How bridges are found](#how-bridges-are-found)), and the same walk counts
the agents below each one. So the size of the part behind every bridge costs
nothing extra. In the example, the walk counts 1 agent below c–d, g–h and
g–i, and 3 below f–g. The code is `bridge_splits` and `worst_cut_share` in
`GraphOfLife.py`. `gol_series.py` applies them to every frame of a 25th
iteration, for `cutRisk`. The engine applies them at the start of
`_cleanup_and_redistribute`, for `cutRiskBefore`.

### Loops

A **loop** (or cycle) is a path through the network that comes back to where
it started without using a connection twice. Loops are what make a network
robust: along a loop there are always two ways between any two of its agents.
Counting them needs care, because the number of different loops in a network
can be astronomically large.

#### Independent loops: the cycle rank

Take a **spanning tree**: a set of connections that reaches every agent and
contains no loop. A connected network of *n* agents has a spanning tree of
exactly *n* − 1 connections. Every connection **not** in the tree closes
exactly one loop with the tree: its **fundamental cycle**. These loops
generate all the others: any loop is a combination of them. (Two loops
combine into the connections that only one of them uses.) So the number of
independent loops is

$$
\text{cycle rank} = |E| - |V| + c .
$$

In words: the number of connections, minus the number of agents, plus the
number of connected pieces, *c*. In every recorded frame, after the cleanup,
*c* is 1. This is `cycleRank` in the viewer.

**Example.** A square of four agents with one diagonal: |*E*| = 5,
|*V*| = 4, *c* = 1, so 2 independent loops — the two triangles. The square
itself is their combination: the two triangles share the diagonal, and
without it they make the square. The network drawn under [Bridges](#bridges)
has 9 connections, 9 agents and one piece, so 1 loop: a–b–c–f–e–a.

Take the world of seed 1 of Experiment 10, after its last game. There, 26,166
connections join 14,870 agents in one piece: 26,166 − 14,870 + 1 = 11,297
independent loops.

#### Loop density

`loopDensity` = `cycleRank` / |*E*|: the share of connections that are
"spare", beyond what a tree would need. A tree has 0; a densely interwoven
network has values near 1. The example network has 1/9, about 0.11, and the
world of seed 1 has 11,297 / 26,166, about 0.43. A settled B1 world has about
0.42 at 100,000 tokens, and 0.38 at 10,000 tokens. The median world of
100,000 tokens has about 10,000 independent loops
([Where the loops run](b1-the-network-size-and-scaling.md#where-the-loops-run)).

Every loop lies in the core. A settled B1 world has 0.82 independent loops per
agent, and 1.34 per agent of its core. Both are counted at the start of a game
([The loops live in the core](b1-the-network-size-and-scaling.md#the-loops-live-in-the-core)).

#### Pieces

`components` is the number of separate pieces. The cleanup keeps only the
largest, so after every phase it is 1 — anything else would be a bug.

#### Loops through one agent

The viewer can colour every agent (and every connection) by how many loops
pass through it. It counts the fundamental cycles of a **breadth-first**
spanning tree. For every connection outside the tree, it walks both ends up
the tree until they meet. Every agent on the way, and the meeting point, lies
on that loop. Breadth-first trees keep the loops short and local. A different
tree would give a different basis, so this is a fair sample of where the
loops run rather than a canonical count.

An agent on no loop at all hangs on a tree, or lies in the core on a path
between loops ([Core, trees and leaves](#core-trees-and-leaves)). In the world
of seed 1, after its last game, 37% of the agents lie on none of its 11,297
loops. Only 36% are off its core, so 114 agents lie in the core, on paths
between loops. One agent, a hub, lies on 1,012 of the loops: about a tenth of
them.

### Core, trees and leaves

Every connected network can be cut into two kinds of part. There is a
**core**, where every agent lies on a loop or on a path between loops. And
there are **trees** that hang off it. This section says how the cut is made,
and why it matters for Graph of Life.

#### Peeling

Take the network and repeat:

> remove every agent that has one connection or none,

until no such agent is left. Removing a leaf can make its neighbour a leaf,
so the peeling goes on until it stops by itself. What is left is the
**2-core**: the largest part of the network in which every agent has at least
two connections *within that part*. It can be empty: a tree peels away
completely.

Every agent then has one of three **roles**:

- **core**: it is in the 2-core;
- **leaf**: it has exactly one connection;
- **tree**: neither. It was peeled away, but it has two or more connections.
  It lies on a tree hanging off the core, between the core and the leaves.

A run records `coreShare`, the share of agents in the 2-core, every 25
iterations. It also records `leaves`, the number of agents with exactly one
connection, after every phase.

#### An example of peeling

Take the network of [What a bridge is](#what-a-bridge-is) again:

```
  a — b — c — d
  |       |
  e ——————f — g — h
              |
              i
```

The loop a–b–c–f–e–a is the core, with its members' connections among
themselves. Peeling removes d, h and i, which have one connection each, in
the first round. Then it removes g, which has only f left. So the core is
{a, b, c, e, f}, the leaves are {d, h, i}, and the tree is {g}.

#### Why the core matters

In B1's game, every connection that carries no tokens is cut
([One iteration, step by step](one-iteration.md)). Cutting a connection on a
loop never disconnects anything: there is always another way round the loop.
Every other connection is a [bridge](#bridges). A bridge of a tree cuts off
everything beyond it. A bridge on a path between two loops of the core cuts
the core itself in two. The cleanup then removes whatever is no longer joined
to the largest piece. So a single cut can lose any tree whole, and, through a
bridge of the core, part of the core too. B2 cuts an unused connection only
where its two ends share two neighbours, so it never cuts a bridge
([`prune_needs`](settings.md#prune_needs)).

In a settled B1 world of 100,000 tokens, about 57% of the agents are in the
core, and 35% are leaves. The rest lie in trees. At 10,000 tokens it is 54%
and 36%
([The core of a world](b1-the-network-size-and-scaling.md#the-core-of-a-world)).

#### How far a tree hangs

Every agent off the core belongs to exactly one tree. And every tree joins
the core at exactly one core agent. Two ways into the core would close a
loop, and the agent would then be in the core. So each agent off the core has a
**depth**, its number of steps to the core. It also has a **root**, the core
agent its tree hangs from.

In the example, d hangs from c at depth 1. g hangs from f at depth 1, and h
and i at depth 2. The tree hanging from f holds three agents (g, h and i),
and the one hanging from c holds one (d). A breadth-first search out of every
core agent at once finds both for every agent.

#### Deeper cores: the onion

The peeling can go further. Remove every agent with fewer than **three**
connections, again and again, and what is left is the **3-core**. With fewer
than *k*, it is the ***k*-core**. Each *k*-core lies inside the one before,
like the layers of an onion. An agent's **core number** is the largest *k*
whose *k*-core still holds it (Seidman 1983). An agent on a tree or a leaf
has core number 1. An agent of the 2-core that the 3-core no longer holds has
core number 2, and so on.

In the example, every core agent has two connections inside the core, so the
3-core is empty. So a, b, c, e and f have core number 2, and d, g, h and i
core number 1. A network in which many agents have high core numbers is held
together by many overlapping loops, not by single rings.

A run records only the share of agents in the 2-core (`coreShare`). The
deeper cores, the trees and the dimension of the core are measured in
[The core of a world](b1-the-network-size-and-scaling.md#the-core-of-a-world).

## Where to go next

- [The shape of the network](b1-the-network-size-and-scaling.md#the-shape-of-the-network): the
  connections, the clustering and the core of the thirty B1 worlds.
- [How properties scale together](b1-the-network-size-and-scaling.md#how-properties-scale-together): the
  agents behind the four scaling relations, and whom the hubs are joined to.
- [How births build the network](b1-lives-and-wealth.md#how-the-network-grows):
  how agents gain connections, and how they lose them.
- [The largest and the smallest](b1-lives-and-wealth.md#the-largest-and-the-smallest): stars and
  dynasties in full.
- [A world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break): the rules that
  keep hubs down.
- [The random twin: is a world a space?](measuring-space.md#the-random-twin-is-a-world-a-space): how the notebook
  tells a space from a small world.
- [How a world breaks](b1-the-network-size-and-scaling.md#how-a-world-breaks): where the bottlenecks that a
  small spectral gap promises give way.
- [The health bar](measuring-health.md#the-health-bar): the two cuts a world must keep small to
  pass it.

## How our view changed

- *An agent's connections: degree, clustering, assortativity, attachment and scaling*:
  - **24 October 2026.** When the five notes on an agent's connections became
    one page, their statements were checked against the code and the results:
    - The note on degree said that `minDegree` is 1 in any world of one piece
      with more than one agent. It is at least 1, and 1 only while some agent
      is a leaf. In the worlds of Experiment 14, under all of B2's rules, it
      was almost always 2. The note also said that the share with at least *d*
      connections falls to 1/|*V*| at the largest degree. It does only if a
      single agent has that degree.
    - The note on assortativity said that friendship networks are typically
      assortative, at about +0.1 to +0.3. It also said that the internet, the
      web and networks of proteins read about −0.1 to −0.2. The assortative
      networks of Newman (2002) are networks of collaboration, at +0.12 to
      +0.36. And the web reads −0.065, nearer zero than that range.
    - The note on clustering said that triangles are what makes a network hard
      to cut. Loops are, and a triangle is only the shortest: a square grid has
      no triangle, and no bridge either. It also said that a settled world has
      long tree-like stretches hanging off its core. In the worlds of 100,000
      tokens the trees are small. Of the agents off the core, 77% are a single
      step from it
      ([How far the trees hang](b1-the-network-size-and-scaling.md#how-far-the-trees-hang)).
    - The note on scaling relations said that `clusteringVsDegree` ≈
      `trianglesVsDegree` − 2 when the triangles follow a power law. The two
      fits run over the same agents. So the difference is exactly the exponent
      of an agent's pairs of neighbours, *k*(*k* − 1)/2, over them. It is 2
      only for large *k*. In B1's worlds of 100,000 tokens it is 2.27
      ([How closely knit a neighbourhood is](b1-the-network-size-and-scaling.md#how-closely-knit-a-neighbourhood-is)).
- *Distances in a network: path length, radius and diameter, the spectral gap*:
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
- *Bridges, cut risk, loops and the core: how a network can be cut*:
  - **24 October 2026.** When the notes on bridges, cut risk and loops became
    one page, their statements were checked against the engine:
    - The note on the cut risk said that `cutRiskBefore` is measured "before
      the phase's cleanup removed anybody". So, it said, it can be read as a
      predictor of the cull rather than a consequence of it. In a game's row,
      though, it is read after the game's unused connections are cut, as
      [How a world breaks](b1-the-network-size-and-scaling.md#does-a-fragile-world-lose-more)
      now says too. It cannot foresee those cuts, and a region they severed is
      already a piece of its own when it is read.
    - The note on loops said that an agent on no loop at all hangs on a tree.
      It may also lie in the core, on a path between loops. In the world of
      seed 1 after its last game, 114 agents do.
- **12 November 2026.** Moved to [Measures of the archived
  studies](measures-of-the-archived-studies.md): three sections that only pages
  of the Archive use: assortativity, preferential attachment and its kernel, and
  scaling relations between agents' properties. The family heading above them
  now names degree and clustering alone.

## References

- Newman, M. E. J. (2002). Assortative mixing in networks. *Physical Review
  Letters* 89, 208701.

- Chung, F. R. K. (1997). *Spectral Graph Theory*. CBMS Regional Conference
  Series in Mathematics 92, American Mathematical Society.

- Seidman, S. B. (1983). Network structure and minimum degree. *Social
  Networks* 5, 269–287.
- Tarjan, R. E. (1974). A note on finding the bridges of a graph.
  *Information Processing Letters* 2, 160–161.

<!-- turns -->
---

← [Measuring tokens: inequality and flow](measuring-tokens.md) · [Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [Measuring space: dimension and the random twin](measuring-space.md) →

**Also about:** [Wealth: who holds the tokens](../README.md#wealth-who-holds-the-tokens), [Space and dimension](../README.md#space-and-dimension).

> [!info]- Pages that link here (20)
> [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [Glossary](glossary.md) · [The rules in detail](rules-in-detail.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Measuring tokens: inequality and flow](measuring-tokens.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Measuring space: dimension and the random twin](measuring-space.md) · [Do the brains matter?](do-the-brains-matter.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [A source and a sink](a-source-and-a-sink.md) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) · [How runs are made, recorded and costed](runs-and-records.md) · [B1: lives and wealth](b1-lives-and-wealth.md) · [B1: the game and its flows](b1-the-game-and-its-flows.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [Space at scale (side study)](space-at-scale.md) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md) · [Measures of the archived studies](measures-of-the-archived-studies.md)
<!-- /turns -->
