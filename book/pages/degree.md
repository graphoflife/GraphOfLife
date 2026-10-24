# An agent's connections: degree, clustering, assortativity, attachment and scaling

## In short

An agent's **degree** is the number of its connections; in a settled
[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world of 100,000 tokens, the mean is about 3.5. Most agents have one
or two, and a few **hubs** hundreds or thousands. **Clustering** and
**assortativity** say how the connections are arranged: whether an agent's
neighbours are joined to each other, and whether hubs are joined to hubs.
The **attachment kernel** says who gains new connections, and **scaling
relations** how an agent's tokens and triangles grow with its connections.

## Degree

The **degree** of an agent is the number of connections it has, which is the
number of its neighbours. It is the simplest thing to know about a place in a
network, and much depends on it. An agent stakes only on its own node and its
neighbours' nodes, and only its neighbours can stake on its node.

### Degree, mean degree and density

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

Three more numbers describe the degrees of a [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."): `medianDegree`, the
[median](median-and-quantiles.md); `minDegree`, the fewest; and `maxDegree`,
the most, which the largest hub has. In a world of one piece with more
than one agent, `minDegree` is at least 1. It is 1 while some agent is a leaf,
and about a third of a settled B1 world's agents are
([Hubs and leaves](#hubs-and-leaves)). Under all of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s rules, in
the worlds of Experiment 14, it was almost always 2. There, nearly every agent
sat on a loop or between loops, and leaves were rare
([Nobody cut off](a-world-that-cannot-break.md#nobody-cut-off)). In the small
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

### The degree distribution

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
[logarithmic axes](correlation.md#logarithmic-axes-and-power-laws) it shows the shape of the tail.
Suppose the share of agents with degree *d* falls like *d*^−γ, a
[power law](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones."). Then the CCDF falls like *d*^−(γ−1). On log–log axes it is
a straight line, with slope −(γ − 1).

### Hubs and leaves

A **leaf** is an agent of degree 1. It hangs on the network by a single
connection, and the cut of that one connection cuts it off (`leaves` in the
statistics). A **hub** is an agent of very high degree. There is no exact line
for a hub, but there is one for a star, a world held together by one
([Stars and hubs](stars-and-hubs.md)).

A settled baseline world has both. After the last
[game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") of the thirty baseline worlds of 100,000 tokens, 35% of all
agents had exactly one connection and 23% two. Another 4.7% had ten or more,
and the largest hub had 10,735
([Connections per agent](what-shape-does-the-network-take.md#connections-per-agent)).
At 10,000 tokens, these were 37%, 23%, 4% and 1,399.

### Where degree is used

- **The [health bar](health-bar.md)** asks for 3 to 12 links per agent: a
  mean degree in that range.
- **The [even split](even-split.md).** An agent with *d* connections has
  *d* + 1 [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."): its own node and its neighbours'. Under the even split,
  it stakes about 1/(*d* + 1) of its tokens on each.
- **The [random twin](random-twin.md)** keeps every agent's degree, and wires
  the connections at random. What a world has and its twin lacks, the world's
  rules built.
- **[Brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.")** read the logarithm of one plus the agent's own degree, and of
  each candidate's ([What a brain sees](brain-inputs.md)).
- **How evenly connections are spread**: the entropy and evenness of the
  degrees ([Entropy and evenness](gini-coefficient.md#entropy-and-evenness)).

## Clustering

Are the neighbours of an agent also neighbours of each other? In a circle of
friends they often are; in a tree they never are. **Clustering** measures how
often.

### Triangles and triples

A **triangle** is three agents all joined to each other. A **connected
triple** centred on *u* is a pair of *u*'s neighbours: two connections that
meet at *u*. An agent of degree *d* is the centre of

$$
\binom{d}{2} = \frac{d(d-1)}{2}
$$

triples. Each of its *d* neighbours pairs with each of the other *d* − 1, and
that counts every pair twice. Every triangle closes three triples, one at each
corner.

### Transitivity

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
[iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.").

An agent's own **clustering coefficient** asks the same of its neighbours
alone. For an agent with *k* ≥ 2 neighbours, it is its triangles divided by
its *k*(*k* − 1)/2 pairs of neighbours. The transitivity is the mean of these
coefficients, each weighted by the agent's pairs, so the hubs weigh most.
[Scaling relations](#scaling-relations-between-agents-properties) follow how
the coefficient falls with *k*.

### Examples of clustering

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
  that is *C* = 0.5. Rewiring 37 connections of the real ring of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1
  lowered it to 0.27 ([The starting ring](starting-ring.md)). At 100,000
  tokens a founder starts with 10 neighbours, and the rewired rings read 0.34
  in the median world.
- **A random network** has the same number of agents and connections, each
  connection placed at random. Two neighbours of an agent are then joined
  about as often as any two agents. So *C* is close to the probability that
  two given agents are joined: the density, *k̄*/(|*V*| − 1). For 1,300
  agents of mean degree 3.3, that is about 0.003.

### Why clustering matters

Loops make a network hard to cut, and a triangle is the shortest loop: in a
triangle, every connection has a way round it. A tree-like network, with few
loops, falls apart when a connection is cut ([Bridges](bridges.md)). B2
builds on this: it cuts an unused connection only where its two ends share at
least two neighbours. Each shared neighbour closes a triangle with the
connection, so the cut leaves two ways round
([`prune_needs`](settings.md#prune_needs)).

A settled baseline world lies between a tree and a web. At 100,000 tokens its
clustering is low, 0.039. Yet that is more than a hundred times what a random
network of its size would have, about 0.0002. At 10,000 tokens it is 0.075.
Births close triangles: a child joined both to its parent and to one of its
parent's neighbours closes one the moment it is born
([Clustering](what-shape-does-the-network-take.md#clustering)). Small trees
hang off the clustered core, and most of their agents are a single step from
it ([How far the trees hang](what-shape-does-the-network-take.md#how-far-the-trees-hang)).

### Where clustering is used

- **Experiment 5** predicted that clustering would stay below 0.1, in a world
  of trees. It did: from iteration 100 on, a median of 0.077 at 10,000
  tokens, and 0.043 at 100,000.
  The thesis failed on the size of the core instead
  ([Was the thesis right?](what-shape-does-the-network-take.md#was-the-thesis-right)).
- **The starting ring** is a small world: clustered like a neighbourhood, yet
  short ([Why a small world?](starting-ring.md#why-a-small-world)).
- **An agent's own coefficient** falls about as one over its connections, the
  mark of a hierarchy of stars
  ([How closely knit a neighbourhood is](how-properties-scale-together.md#how-closely-knit-a-neighbourhood-is)).

## Assortativity

Do well-connected agents connect to other well-connected agents, or to poorly
connected ones? **Assortativity** answers with one number between −1 and 1
(Newman 2002).

### The idea of assortativity

Look at every connection and write down the [degrees](#degree) of its two
ends: a pair (*j*, *k*). Then ask whether *j* and *k* go together: the
[correlation](correlation.md) of the two ends over all connections.

- **Positive** (assortative): like joins like, hubs to hubs and the sparsely
  connected among themselves. Networks of people often are. In networks of
  scientists who wrote papers together, or of actors who played in the same
  films, *r* is about +0.1 to +0.4 (Newman 2002).
- **Negative** (disassortative): hubs sit among the sparsely connected, like
  the centre of a star. The internet, networks of proteins and nerve cells,
  and food webs are so, at about −0.15 to −0.3. The web reads −0.07.
- **Zero**: the degree of one end says nothing about the other.

### The formula for assortativity

A connection has no first and second end, so every connection is counted
both ways round, as (*j*, *k*) and (*k*, *j*). With *M* connections, the
correlation of these 2*M* pairs is

$$
r = \frac{\frac{1}{M}\sum_e j_e k_e - \Bigl[\frac{1}{M}\sum_e \tfrac12 (j_e + k_e)\Bigr]^2}
         {\frac{1}{M}\sum_e \tfrac12 (j_e^2 + k_e^2) - \Bigl[\frac{1}{M}\sum_e \tfrac12 (j_e + k_e)\Bigr]^2} .
$$

The numerator is the covariance of the two ends: the mean product of their
degrees, less the square of the mean degree at an end. The denominator is the
variance of the degree at one end of a connection: the mean of its square,
less the same square. The viewer's `assortativity` is *r* over a
frame's connections, recorded on every 25th iteration.

### Two examples of assortativity

- **A star**: one hub joined to three leaves. Every connection is (3, 1), so
  Σ *jk* / *M* = 3. The mean end degree is (3 + 1)/2 = 2, and the mean
  squared end degree is (9 + 1)/2 = 5. So *r* = (3 − 4)/(5 − 4) = **−1**:
  perfectly disassortative.
- **A path of four**, a–b–c–d, degrees 1, 2, 2, 1. The connections are
  (1, 2), (2, 2) and (2, 1), so Σ *jk* / *M* = 8/3. The mean end degree is
  10/6, and the mean squared end degree 18/6 = 3. So
  *r* = (8/3 − 25/9)/(3 − 25/9) = **−0.5**.

### A caution about assortativity

Suppose a few hubs hold many of the connections. Then there are not enough
hubs for every hub to have hubs as neighbours: most of a hub's neighbours
*must* be small. So a network with a heavy-tailed degree distribution tends
to come out disassortative, even if it is wired at random given its degrees.
A negative *r* is only interesting beyond that pull.

### Where assortativity is used

A settled baseline world has an assortativity of about −0.07 at 100,000
tokens, and −0.11 at 10,000. Across the thirty worlds of 100,000 tokens it
runs from −0.47 to +0.02. That is mildly negative, and partly the pull above.
A curve says more: the mean degree of an agent's neighbours, for each class
of its own degree. It shows leaves hanging on hubs, and hubs surrounded by
leaves
([Whom the well-connected are joined to](how-properties-scale-together.md#whom-the-well-connected-are-joined-to)).
Taken of the logarithm of the degree, which the hubs sway less, the same
correlation reads −0.20 in the median world
([Like next to like](like-next-to-like.md#neighbours)).
[How alike at a distance](like-next-to-like.md#how-alike-at-a-distance) carries it on to agents
several steps apart.

## Preferential attachment and its kernel

In many growing networks, the well connected gain new connections faster than
the poorly connected. Popular web pages are linked to more, and much-cited
papers are cited more. Suppose the rate at which an agent gains connections
is proportional to the connections it has. Then the network grows by
**preferential attachment** (Barabási and Albert 1999). It is the same idea
as Price's cumulative advantage (Price 1976), and as Simon's model of word
frequencies (Simon 1955).

### The attachment kernel

The general form is the **attachment kernel** Π(*k*): the expected number of
new connections an agent with *k* connections gains in one step. It is
measured directly. Take all agents at the start of a step, and sort them by
*k*. Count the connections each class gains during the step, and divide by
the agents in the class (Newman 2001; Jeong, Néda and Barabási 2003). A
kernel that grows as a power of *k*, Π(*k*) ∝ *k*^α, is a straight line of
slope α on logarithmic axes:

- α = 0: every agent gains alike, whatever it has. The degrees stay narrow,
  as in a random network.
- α = 1: **linear** preferential attachment. A growing network gets a
  power-law tail of degrees, with exponent 3 if nothing is ever lost
  (Barabási and Albert 1999).
- α < 1 (sublinear): the tail is cut off and bends down, close to a
  stretched exponential.
- α > 1 (superlinear): one agent ends up joined to nearly everyone
  (Krapivsky, Redner and Leyvraz 2000).

### A kernel, worked out

Say that at the start of a step, 1,000 agents have one connection and 100
have ten. During the step the first class gains 50 new connections in all,
and the second class 50 too. Then Π(1) = 50 / 1,000 = 0.05, and
Π(10) = 50 / 100 = 0.5. The two classes gained alike, but each agent with ten
connections gained ten times as much as each agent with one. Ten times the
connections bring ten times the gains: α = 1, linear preferential attachment.

### Losses too

A network that also loses connections has a **loss kernel** as well: the
expected connections lost per agent with *k*. If both gains and losses are
proportional to *k*, every agent's connections change by a random factor
from step to step, multiplied rather than added. This is Gibrat's law of
proportionate growth, applied to connections
([Reading an exponent](#reading-an-exponent)). Whether a heavy tail forms
then depends on how gains and losses balance, and on what stops the poorest
from vanishing.

### The kernels in the worlds

[Who gains and who loses](how-agents-have-children.md#who-gains-and-who-loses)
measures both kernels in the thirty B1 worlds of 100,000 tokens. It counts
what each class of agents gains in a reproduction phase, sorted by their
connections at its start. And it counts what they lose in the game that
follows, sorted by their connections at the start of that game. Gains grow as
*k*^1.02, and losses a little faster, as *k*^1.20. So the well connected
drift back down, and no agent runs away with the network.

One caution holds for the gains. Say a parent hands a connection over to its
newborn child. Then the neighbour at the other end is counted as gaining one,
though the connection only changed hands. The parent's loss is not counted.
So part of what the leaves gain may be handovers.

### Where preferential attachment is used

- **The tail of the connections.** A child is joined to some of its parent's
  candidates. So an agent gains a connection whenever a neighbour has a child
  that is joined to it too. Twice the neighbours bring twice the chances.
  [Power laws, real and apparent](power-laws-real-and-apparent.md) had
  proposed preferential attachment, reached by copying, as the source of the
  heavy tail of the connections. There it is still a hypothesis.
- **B2's links at birth.** A brain that chooses links meets the well
  connected among its candidates more often, whatever it favours. That is
  preferential attachment, and it grows hubs. So B2 draws two of a newborn's
  links at random, and caps the links a brain may choose
  ([What the game adds](a-world-that-cannot-break.md#what-the-game-adds)).
- **Age.** In the model of Barabási and Albert, the oldest nodes are the
  hubs. In B1, age gives an agent no connections; connections give it age
  ([Connections and age](how-agents-have-children.md#connections-and-age)).

## Scaling relations between agents' properties

How does one property of an agent grow with another: its tokens with its
connections, say? On every 25th iteration, the viewer fits four **scaling
relations** of the form

$$
y \approx a\, x^{b} .
$$

It fits them by least squares on the logarithms of all agents of the frame.
That is, it fits the straight line ln *y* = ln *a* + *b* ln *x*
([Fitting a straight line](correlation.md#fitting-a-straight-line)). In words: drawn on
logarithmic axes, the agents form a cloud, and the fit is the straight line
through it. Its slope, the **exponent** *b*, says how *y* grows with *x*. R²
says how tightly the agents follow the line. Agents with *x* = 0 or *y* = 0
are left out, since a logarithm of 0 does not exist.

### The four relations

| field | *y* | *x* | what an exponent would mean |
|---|---|---|---|
| `tokensVsDegree` | tokens | connections | 1: tokens in proportion to connections; 0: connections buy nothing |
| `trianglesVsDegree` | triangles the agent is a corner of | connections | 2: neighbourhoods wired at random (pairs of neighbours grow as k²) |
| `clusteringVsDegree` | [clustering](#clustering) coefficient (agents with ≥ 2 connections) | connections | −1: a hierarchical network — dense small groups inside sparser large ones (Ravasz and Barabási 2003) |
| `changeVsTokens` | the size \|Δ\| of the change in tokens over the phase | tokens after the phase | 1: everyone risks the same fraction of what they hold (Gibrat's law) |

Each is reported with its R² (`tokensVsDegreeR2`, …).

### Reading an exponent

- For **tokens and connections**: take *b* = 0.6. An agent with ten times the
  connections then holds 10^0.6 ≈ 4 times the tokens: more, but less than in
  proportion.
- For **clustering**: an agent's clustering coefficient is its triangles
  divided by its *k*(*k* − 1)/2 pairs of neighbours. The two fits run over the
  same agents, those in at least one triangle. So `clusteringVsDegree` is
  exactly `trianglesVsDegree` minus the exponent of the pairs over those
  agents. That exponent is 2 only for large *k*: from two connections to
  three, the pairs triple, where *k*² grows 2.25 times. Over the agents of
  B1's worlds of 100,000 tokens it is 2.27. Triangles grow as *k*^0.99, and
  clustering falls as *k*^−1.28
  ([How closely knit a neighbourhood is](how-properties-scale-together.md#how-closely-knit-a-neighbourhood-is)).
- For **changes**: Gibrat's law of proportionate growth (Gibrat 1931) says the
  change of a firm's size is proportional to its size. So *b* < 1 means that
  the rich change by a smaller fraction than the poor.

### Two cautions about the fits

- **R² is about scatter, not about power laws.** Many curved relations look
  straight over a decade or two of logarithmic axes. Read R² as "how much of
  the spread does this line account for", not as proof of a law.
- **The fit is over agents, and most agents are small.** Thousands of agents
  with one or two connections dominate the least-squares sum; a handful of
  hubs barely moves it.

**An example.** In B1's worlds, agents hold on average tokens in proportion
to *k* + 1, the number of their candidates. That is not a power of *k*. From
*k* = 1 to *k* = 2, *k* doubles, but *k* + 1 grows by a factor of only 1.5.
That is a local exponent of ln 1.5 / ln 2 = 0.58. Most agents have one or two
connections. So even if every agent held exactly its share, the fit over them
would report 0.73, not 1. It reports 0.58, because the poorly connected also
hold a little more per candidate
([Tokens and connections](how-properties-scale-together.md#tokens-and-connections)).
An exponent fitted to a relation that is not a power law reports where most
of the points are.

### Where scaling relations are used

- [How properties scale together](how-properties-scale-together.md) shows the
  agents behind each of the four fits, in B1's thirty worlds of 100,000
  tokens. All four are nearly the same in every world: they belong to the
  rules, not to what the brains compute.
- [What goes with what](what-goes-with-what.md#between-worlds): how clustering
  and triangles grow with an agent's connections is the second of the
  components that tell those thirty worlds apart.

## Where to go next

- [The shape of the network](what-shape-does-the-network-take.md): the
  connections, the clustering and the core of the thirty B1 worlds.
- [How properties scale together](how-properties-scale-together.md): the
  agents behind the four scaling relations, and whom the hubs are joined to.
- [How births build the network](how-agents-have-children.md#how-the-network-grows):
  how agents gain connections, and how they lose them.
- [Stars and hubs](stars-and-hubs.md): the best-connected agents, and the
  worlds that one of them holds together.

## How our view changed

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
    ([How far the trees hang](what-shape-does-the-network-take.md#how-far-the-trees-hang)).
  - The note on scaling relations said that `clusteringVsDegree` ≈
    `trianglesVsDegree` − 2 when the triangles follow a power law. The two
    fits run over the same agents. So the difference is exactly the exponent
    of an agent's pairs of neighbours, *k*(*k* − 1)/2, over them. It is 2
    only for large *k*. In B1's worlds of 100,000 tokens it is 2.27
    ([How closely knit a neighbourhood is](how-properties-scale-together.md#how-closely-knit-a-neighbourhood-is)).

## References

- Barabási, A.-L. and Albert, R. (1999). Emergence of scaling in random
  networks. *Science* 286, 509–512.
- Gibrat, R. (1931). *Les inégalités économiques*. Paris: Sirey.
- Jeong, H., Néda, Z. and Barabási, A.-L. (2003). Measuring preferential
  attachment in evolving networks. *Europhysics Letters* 61, 567–572.
- Krapivsky, P. L., Redner, S. and Leyvraz, F. (2000). Connectivity of
  growing random networks. *Physical Review Letters* 85, 4629–4632.
- Newman, M. E. J. (2001). Clustering and preferential attachment in growing
  networks. *Physical Review E* 64, 025102.
- Newman, M. E. J. (2002). Assortative mixing in networks. *Physical Review
  Letters* 89, 208701.
- Price, D. de S. (1976). A general theory of bibliometric and other
  cumulative advantage processes. *Journal of the American Society for
  Information Science* 27, 292–306.
- Ravasz, E. and Barabási, A.-L. (2003). Hierarchical organization in complex
  networks. *Physical Review E* 67, 026112.
- Simon, H. A. (1955). On a class of skew distribution functions.
  *Biometrika* 42, 425–440.

<!-- turns -->
---

← [How a world breaks](how-a-world-breaks.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Stars and hubs](stars-and-hubs.md) →

**Also about:** [Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages).

> [!info]- Pages that link here (16)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [The starting ring](starting-ring.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [How births build the network](how-agents-have-children.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [The even split](even-split.md) · [The shape of the network](what-shape-does-the-network-take.md) · [How properties scale together](how-properties-scale-together.md) · [Stars and hubs](stars-and-hubs.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [The random twin: is a world a space?](random-twin.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
