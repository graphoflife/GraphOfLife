# Measures of the archived studies

> [!question] Questions of this page
> - How is a power law fitted properly, and where does its tail start?
> - How can the spread of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")' decisions be split into where, who and chance?
> - Are hubs joined to hubs, who gains new [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."), and how do an agent's
>   properties grow together?
> - Do the rich stay rich, and do [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") go round?
> - Why are the living older than the dead, and how risky is each age?

## In short

Some measures are used only by pages of the Archive: the studies of [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") and of
the pilots. Until 12 November 2026 they were sections of the reference pages.
They are kept here as they were written, so that every archived page can still
be read. A measure that a page outside the Archive uses stays on its reference
page.

Its sections:

- [Statistics](#statistics): fitting a power law, splitting a decision's variance, the participation ratio
- [Networks](#networks): assortativity, preferential attachment, scaling relations
- [Tokens](#tokens): mobility and lightning
- [Lives and ages](#lives-and-ages): why the living are older, steady ages, the risk at each age, cohorts
- [Assembly index and assembly](#assembly-index-and-assembly)

## Statistics

From [Statistics: the tools of the notebook](statistics.md).

### Fitting a power law properly

Say the connections of a world follow a power law above some size. How is
its exponent γ found, and where does the law begin? This section gives the
method the literature recommends (Clauset, Shalizi and Newman 2009), with a
small example. [Power laws, real and apparent](b1-the-network-size-and-scaling.md#power-laws-real-and-apparent-and-how-to-fit-one)
applies it to the worlds of B1.

#### The obvious way, and its problem

The obvious way is to draw the CCDF on log–log axes and fit a straight line
through it. The viewer's `degreeExponent` and `tokenExponent` are exactly
that. Each is the [least-squares line](statistics.md#fitting-a-straight-line)
of ln *P*(≥ *k*) on ln *k*, one point for each distinct value, and the
exponent is 1 − slope.

The trouble is in the points:

- Neighbouring points of a cumulative distribution are not independent: each
  contains the next.
- The line counts every distinct value once. In the far tail each value
  belongs to one agent or two, so a handful of agents make many of the
  points.

So the estimate wanders, and R² looks good whatever the truth. In the median
B1 world of 100,000 tokens, the line through the whole degree distribution has
R² = 0.96. Yet the distribution is not straight
([The connections of one world](b1-the-network-size-and-scaling.md#the-connections-of-one-world)).

#### Maximum likelihood

Maximum likelihood asks a different question. If the values came from a
power law with exponent γ, how probable would the values we see be? The best
γ is the one that makes them most probable.

Precisely: suppose the values *x*₁ … *x*ₙ from some *x*_min upward follow the
density p(*x*) = (γ − 1) *x*_min^(γ−1) *x*^−γ. That is a power law, scaled
so that its shares above *x*_min add up to one. The γ that makes the
observed values most probable is

$$
\hat\gamma = 1 + n \Bigl[\sum_{i=1}^{n} \ln \frac{x_i}{x_{\min}}\Bigr]^{-1} .
$$

In words: take the logarithm of how many times over each value exceeds
*x*_min, add these up, divide *n* by the sum, and add one. Values far above
*x*_min make the sum large and the exponent small: that is a heavy tail. The
estimate is uncertain by about (γ̂ − 1)/√*n*, its standard error.

For whole numbers, like a degree, *x*_min is replaced by *x*_min − ½. That
corrects most of the error of treating them as continuous.

**Example.** Take a tail of four degrees, 2, 3, 4 and 8, with *k*_min = 2,
so that the formula divides by 1.5. The logarithms of 2/1.5, 3/1.5, 4/1.5
and 8/1.5 add up to 3.636. So γ̂ = 1 + 4/3.636 = 2.10. It is known to about
±1.1/√4 = ±0.55, which is to say hardly at all. A tail needs many values.

#### Where does the tail start?

Real distributions are a power law, if at all, only above some *x*_min.
Below it, the body bends away. So the fit tries many starts, and keeps the
one whose fitted law sits closest to the data:

1. Take up to 24 values of *k*_min, spread evenly over the distinct values.
2. For each with at least 25 values at or above it, estimate γ̂ as above.
3. Measure the **Kolmogorov–Smirnov distance**: the largest vertical gap
   between the cumulative distribution of the tail and that of the fitted
   law.
4. Keep the *k*_min with the smallest distance.

The viewer reports five numbers from this fit:

- γ̂ (`degreeGamma`);
- the *k*_min (`degreeKMin`);
- the share of all agents in the tail (`degreeTailShare`);
- the distance (`degreeGammaKS`);
- R² of the CCDF over the tail (`degreeGammaR2`), which describes the fit
  but does not test it.

#### What this does not tell you

The method finds the best power law. It does not prove there is one. Two
more checks are needed.

**Is the gap small enough?** Draw many samples from the fitted law, and fit
each the same way. Then ask how often a sample fits its own law **worse**
than the data fit theirs: how often its distance is at least the data's.
This is the test of Clauset, Shalizi and Newman, and that share is its
p-value. If it is 10% or less, the data are too far from a power law to be
one. If it is larger, a power law is a plausible description, though another
curve might be too.

The viewer does not run this test; the analysis behind
[Power laws, real and apparent](b1-the-network-size-and-scaling.md#power-laws-real-and-apparent-and-how-to-fit-one)
does. Each
sample draws its values below *k*_min from the data's own, and its tail from
the fitted law. The world of [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 gets 200 samples, and every other fit 100.

**Does a different law fit as well?** A **log-normal**, a distribution whose
logarithm is normal, often fits as well over the range one has. So does a
power law with an exponential cut-off, which bends down at the top.

And a tail is a small part of a world. In B1, the tail of the connections holds 5%
of its agents at 10,000 tokens, and 1% at 100,000 tokens. A power law there
says nothing about the rest.

### Where, who and chance: splitting a decision's variance

Agents decide differently from one another. There are four possible reasons:
- they are in different situations (**where**);
- their [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") differ (**who**);
- each brain reacts differently to the situation it is in (**who, there**);
- or the random inputs every brain reads differ (**chance**).

This section shows how the variance of a decision splits into those four
parts.

#### The design

Take *B* brains and *S* situations. Put every brain into every situation *K*
times, each time with fresh random inputs. But use the *same* *K* draws for
every brain, so that the brains meet exactly the same inputs. Write *d*_{bsk}
for what brain *b* decides in situation *s* on draw *k*. It may be a share of
tokens, or a yes or no as 1 or 0. Then take the averages:

- *d̄*_b: brain *b* over all situations and draws;
- *d̄*_s: situation *s* over all brains and draws;
- *d̄*_{bs}: brain *b* in situation *s*, over its draws;
- *d̄*: everything.

They split the sum of squared deviations from the grand mean exactly into
four parts. Statisticians call this a two-way analysis of variance with
replicates:

$$
\begin{aligned}
\text{situation} &= B K \sum_s (\bar d_s - \bar d)^2 \\
\text{brain} &= S K \sum_b (\bar d_b - \bar d)^2 \\
\text{brain in situation} &= K \sum_{b,s} (\bar d_{bs} - \bar d_b - \bar d_s + \bar d)^2 \\
\text{chance} &= \sum_{b,s,k} (d_{bsk} - \bar d_{bs})^2
\end{aligned}
$$

In words: the first two parts say how far each situation's average lies
from the grand mean, and each brain's. The third says how far each brain in
each situation lies from what those two would predict. The last says how far
each draw lies from its brain's average in its situation. Each part divided by their sum is a **share** of the variance.

#### How to read the shares

- **Situation** large: every brain does the same thing in the same place,
  and agents differ because their places do.
- **Brain** large: some brains always do more of it than others, wherever
  they are. It is a trait, like a temperament.
- **Brain in situation** large: brains differ in *how they respond*. One
  gives more when rich, another when poor.
- **Chance** large: the random inputs decide, and a brain does not do the
  same thing twice.

"Do agents behave the same in the same circumstance?" is answered by the last
three together, against chance alone. If brain and brain in situation are
small, agents in the same place act alike, and their differences are the
dice.

#### Units as well as shares

Shares hide sizes. A decision every agent makes the same way has a tiny
variance, and its shares can be anything. So the decision's spread is also
given in its own units. That is how far apart two agents in the same
situation and draw are, as the standard deviation over agents. It is set
against how far one brain is from itself over the draws.

[How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents) uses this
split, with a small worked example.

### How many directions a cloud of points uses

A [message](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") of B1 has 30 numbers, so the messages of a world are points in a
space of 30 [dimensions](measuring-space.md#three-ways-to-measure-a-networks-dimension "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."). They need not use all 30. Suppose every message were
the same except for one number. Then the messages would vary along a single
direction, and the other 29 numbers would carry nothing. The **participation
ratio** counts how many directions the variety is really spread over.

#### The recipe

1. **Covariance.** Take the points *x*₁, …, *x*ₙ, each a list of *d*
   numbers. Compute the *d* × *d* covariance matrix *C*. Its entry *Cᵢⱼ* is
   the covariance of number *i* with number *j* over the points.
2. **Principal directions.** The eigenvectors of *C* are the directions along
   which the cloud of points spreads, independently of each other: the
   principal components. The eigenvalue λₖ of each is the variance of the
   points along it. The eigenvalues add up to the total variance.
3. **Count them, weighted:**

$$
\text{PR} = \frac{\left(\sum_k \lambda_k\right)^2}{\sum_k \lambda_k^2} .
$$

In words: the square of the total variance, over the sum of the squared
variances of the directions. It is the effective number *D* above, with each
direction's share of the variance as its *p*.

#### Reading the participation ratio

- If the variance is spread equally over *m* directions and the rest have
  none, then PR = *m*. For example, *m* eigenvalues of 1 give *m*²/*m* = *m*.
- If one direction holds almost everything, PR is close to 1.
- In between, PR is an **effective number** of directions, in the same sense
  as 2^*H* is an effective number of kinds
  ([Entropy and evenness](measuring-tokens.md#entropy-and-evenness)). It is
  how many equally important directions would spread the cloud as unevenly
  as it is spread.

So PR runs from 1 to *d*. For random points with independent numbers of equal
variance, it is close to *d*, and a little less because of sampling noise.
For 30 independent numbers measured on a few thousand points, it comes out
near 28 or 29.

#### Where the participation ratio is used

[What agents say to each other](b1-brains-lineages-and-messages.md#what-agents-say-to-each-other) uses it on
the messages agents write. There it counts how many of the 30 message numbers
a world's messages actually use. The measure comes from physics, where it
counts over how many sites a wave is spread (Bell and Dean 1970). In
neuroscience it measures the dimension of neural activity (Gao and others
2017).

## Networks

From [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md).

### Assortativity

Do well-connected agents connect to other well-connected agents, or to poorly
connected ones? **Assortativity** answers with one number between −1 and 1
(Newman 2002).

#### The idea of assortativity

Look at every connection and write down the [degrees](measuring-networks.md#degree) of its two
ends: a pair (*j*, *k*). Then ask whether *j* and *k* go together: the
[correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes) of the two ends over all connections.

- **Positive** (assortative): like joins like, hubs to hubs and the sparsely
  connected among themselves. Networks of people often are. In networks of
  scientists who wrote papers together, or of actors who played in the same
  films, *r* is about +0.1 to +0.4 (Newman 2002).
- **Negative** (disassortative): hubs sit among the sparsely connected, like
  the centre of a star. The internet, networks of proteins and nerve cells,
  and food webs are so, at about −0.15 to −0.3. The web reads −0.07.
- **Zero**: the degree of one end says nothing about the other.

#### The formula for assortativity

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
frame's connections, recorded on every 25th [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.").

#### Two examples of assortativity

- **A star**: one hub joined to three leaves. Every connection is (3, 1), so
  Σ *jk* / *M* = 3. The mean end degree is (3 + 1)/2 = 2, and the mean
  squared end degree is (9 + 1)/2 = 5. So *r* = (3 − 4)/(5 − 4) = **−1**:
  perfectly disassortative.
- **A path of four**, a–b–c–d, degrees 1, 2, 2, 1. The connections are
  (1, 2), (2, 2) and (2, 1), so Σ *jk* / *M* = 8/3. The mean end degree is
  10/6, and the mean squared end degree 18/6 = 3. So
  *r* = (8/3 − 25/9)/(3 − 25/9) = **−0.5**.

#### A caution about assortativity

Suppose a few hubs hold many of the connections. Then there are not enough
hubs for every hub to have hubs as [neighbours](the-world.md#the-network "An agent joined to another by a connection."): most of a hub's neighbours
*must* be small. So a network with a heavy-tailed degree distribution tends
to come out disassortative, even if it is wired at random given its degrees.
A negative *r* is only interesting beyond that pull.

#### Where assortativity is used

A settled [baseline](b1-and-b2.md "The stated rules an experiment starts from: B1 or B2. Drawn worlds are stored as changes from B1, but judged against each other.") world has an assortativity of about −0.07 at 100,000
tokens, and −0.11 at 10,000. Across the thirty worlds of 100,000 tokens it
runs from −0.47 to +0.02. That is mildly negative, and partly the pull above.
A curve says more: the mean degree of an agent's neighbours, for each class
of its own degree. It shows leaves hanging on hubs, and hubs surrounded by
leaves
([Whom the well-connected are joined to](b1-the-network-size-and-scaling.md#whom-the-well-connected-are-joined-to)).
Taken of the logarithm of the degree, which the hubs sway less, the same
correlation reads −0.20 in the median world
([Like next to like](b1-the-network-size-and-scaling.md#neighbours)).
[How alike at a distance](b1-the-network-size-and-scaling.md#how-alike-at-a-distance) carries it on to agents
several steps apart.

### Preferential attachment and its kernel

In many growing networks, the well connected gain new connections faster than
the poorly connected. Popular web pages are linked to more, and much-cited
papers are cited more. Suppose the rate at which an agent gains connections
is proportional to the connections it has. Then the network grows by
**preferential attachment** (Barabási and Albert 1999). It is the same idea
as Price's cumulative advantage (Price 1976), and as Simon's model of word
frequencies (Simon 1955).

#### The attachment kernel

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

#### A kernel, worked out

Say that at the start of a step, 1,000 agents have one connection and 100
have ten. During the step the first class gains 50 new connections in all,
and the second class 50 too. Then Π(1) = 50 / 1,000 = 0.05, and
Π(10) = 50 / 100 = 0.5. The two classes gained alike, but each agent with ten
connections gained ten times as much as each agent with one. Ten times the
connections bring ten times the gains: α = 1, linear preferential attachment.

#### Losses too

A network that also loses connections has a **loss kernel** as well: the
expected connections lost per agent with *k*. If both gains and losses are
proportional to *k*, every agent's connections change by a random factor
from step to step, multiplied rather than added. This is Gibrat's law of
proportionate growth, applied to connections
([Reading an exponent](#reading-an-exponent)). Whether a heavy tail forms
then depends on how gains and losses balance, and on what stops the poorest
from vanishing.

#### The kernels in the worlds

[Who gains and who loses](b1-lives-and-wealth.md#who-gains-and-who-loses)
measures both kernels in the thirty B1 worlds of 100,000 tokens. It counts
what each class of agents gains in a reproduction phase, sorted by their
connections at its start. And it counts what they lose in the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") that
follows, sorted by their connections at the start of that game. Gains grow as
*k*^1.02, and losses a little faster, as *k*^1.20. So the well connected
drift back down, and no agent runs away with the network.

One caution holds for the gains. Say a parent hands a connection over to its
newborn child. Then the neighbour at the other end is counted as gaining one,
though the connection only changed hands. The parent's loss is not counted.
So part of what the leaves gain may be handovers.

#### Where preferential attachment is used

- **The tail of the connections.** A child is joined to some of its parent's
  [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."). So an agent gains a connection whenever a neighbour has a child
  that is joined to it too. Twice the neighbours bring twice the chances.
  [Power laws, real and apparent](b1-the-network-size-and-scaling.md#power-laws-real-and-apparent-and-how-to-fit-one) had
  proposed preferential attachment, reached by copying, as the source of the
  heavy tail of the connections. There it is still a hypothesis.
- **[B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s links at birth.** A brain that chooses links meets the well
  connected among its candidates more often, whatever it favours. That is
  preferential attachment, and it grows hubs. So B2 draws two of a newborn's
  links at random, and caps the links a brain may choose
  ([What the game adds](building-b2-and-the-search-for-a-healthy-rich-world.md#what-the-game-adds)).
- **Age.** In the model of Barabási and Albert, the oldest nodes are the
  hubs. In B1, age gives an agent no connections; connections give it age
  ([Connections and age](b1-lives-and-wealth.md#connections-and-age)).

### Scaling relations between agents' properties

How does one property of an agent grow with another: its tokens with its
connections, say? On every 25th iteration, the viewer fits four **scaling
relations** of the form

$$
y \approx a\, x^{b} .
$$

It fits them by least squares on the logarithms of all agents of the frame.
That is, it fits the straight line ln *y* = ln *a* + *b* ln *x*
([Fitting a straight line](statistics.md#fitting-a-straight-line)). In words: drawn on
logarithmic axes, the agents form a cloud, and the fit is the straight line
through it. Its slope, the **exponent** *b*, says how *y* grows with *x*. R²
says how tightly the agents follow the line. Agents with *x* = 0 or *y* = 0
are left out, since a logarithm of 0 does not exist.

#### The four relations

| field | *y* | *x* | what an exponent would mean |
|---|---|---|---|
| `tokensVsDegree` | tokens | connections | 1: tokens in proportion to connections; 0: connections buy nothing |
| `trianglesVsDegree` | triangles the agent is a corner of | connections | 2: neighbourhoods wired at random (pairs of neighbours grow as k²) |
| `clusteringVsDegree` | [clustering](measuring-networks.md#clustering) coefficient (agents with ≥ 2 connections) | connections | −1: a hierarchical network — dense small groups inside sparser large ones (Ravasz and Barabási 2003) |
| `changeVsTokens` | the size \|Δ\| of the change in tokens over the phase | tokens after the phase | 1: everyone risks the same fraction of what they hold (Gibrat's law) |

Each is reported with its R² (`tokensVsDegreeR2`, …).

#### Reading an exponent

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
  ([How closely knit a neighbourhood is](b1-the-network-size-and-scaling.md#how-closely-knit-a-neighbourhood-is)).
- For **changes**: Gibrat's law of proportionate growth (Gibrat 1931) says the
  change of a firm's size is proportional to its size. So *b* < 1 means that
  the rich change by a smaller fraction than the poor.

#### Two cautions about the fits

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
([Tokens and connections](b1-the-network-size-and-scaling.md#tokens-and-connections)).
An exponent fitted to a relation that is not a power law reports where most
of the points are.

#### Where scaling relations are used

- [How properties scale together](b1-the-network-size-and-scaling.md#how-properties-scale-together) shows the
  agents behind each of the four fits, in B1's thirty worlds of 100,000
  tokens. All four are nearly the same in every world: they belong to the
  rules, not to what the brains compute.
- [What goes with what](measures-atlases-and-assembly.md#between-worlds): how clustering
  and triangles grow with an agent's connections is the second of the
  components that tell those thirty worlds apart.

## Tokens

From [Measuring tokens: inequality and flow](measuring-tokens.md).

### Mobility

A world can be unequal in two very different ways. In one, the same agents
are rich game after game. In the other, wealth is just as unequal at every
moment, but who holds it keeps changing. The Gini coefficient cannot tell the
two apart, because it looks at one moment. **Mobility** compares two moments.
Economists measure it for incomes across years and generations (Shorrocks
1978; Chetty and others 2014). The notebook uses three measures.

#### Rank correlation over time

Take the agents alive at a moment *t* and again *k* games later. Rank them by
their tokens at each moment, the poorest first; agents with equal tokens share
the mean of their ranks. **Spearman's rank correlation** ρ is the ordinary
[correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes) of the two lists of ranks:

- ρ = 1: every agent kept its place in the order of wealth;
- ρ = 0: the later order has nothing to do with the earlier;
- ρ < 0: the order turned round, and the poor became the rich.

Ranks rather than tokens make ρ deaf to how unequal the world is. They also
keep a few very rich agents from deciding it, as they would decide an ordinary
correlation of the tokens themselves. Drawn against *k*, ρ shows how long
wealth is remembered.

#### A table of moves

Sort the agents at *t* into fifths by their rank of wealth, from the poorest
fifth to the richest. Do the same *k* games later. For every pair of fifths,
count the agents that went from the one to the other, and add a column for
those no longer alive. Divide each row by its total. Then each row says where
the agents of one fifth went.

This is a **transition matrix** *P*. Its entry *Pᵢⱼ* is the share of the
agents of fifth *i* found in fifth *j* later. A world where nobody moves has 1
on the diagonal and 0 elsewhere.

#### One number: the Shorrocks index

For *n* classes, with the diagonal entries *Pᵢᵢ* computed among the agents
still alive,

$$
M = \frac{n - \sum_i P_{ii}}{n - 1}
$$

(Shorrocks 1978). In words: count how far the diagonal falls short of *n*,
the sum it would have if nobody moved, and scale it.

If nobody moves, every *Pᵢᵢ* is 1 and *M* = 0. Suppose instead that where an
agent ends up has nothing to do with where it started. Then every row is
(⅕, ⅕, ⅕, ⅕, ⅕), the diagonal adds up to 1, and *M* = 1. So *M* runs from 0,
a rigid world, to 1, a world without memory of wealth. It can exceed 1 when
the order tends to reverse.

#### Where mobility is used

[Do the rich stay rich?](b1-lives-and-wealth.md#do-the-rich-stay-rich)
measures all three in the baseline worlds.

### Lightning: tokens that go round

Does the flow of a game go round in circles? Picture tokens leaving an
agent, crossing several connections and arriving back where they started,
like current round a circuit. The viewer calls such a closed loop of flow a
**lightning**. Ecologists measure the same thing in food webs as a *cycling
index* (Finn 1976). This section explains what is measured, and why only
bounds can be given.

#### Loops of flow

Think of the flow of a game ([Token flow](measuring-tokens.md#token-flow)) as a directed
network: an arrow from *u* to *v* carrying *a*(*u* → *v*) tokens. A **loop**
of length *L* is a closed path along *L* arrows,
*u*₁ → *u*₂ → … → *u*_L → *u*₁. Each of its arrows carries at least one
token. Taking one token off every arrow of the loop removes *L* tokens of
flow that went round. A token taken off cannot be counted in a second loop.
A loop of length *L* **scores** *L*², so that one long circuit counts for
more than many short ones.

#### The greedy peel

Finding the loops that use the most flow is as hard as finding the longest
loop in a network. No fast exact method is known for that. So the program
(`gol_lightning.py`) peels loops greedily:

1. Start at an agent, in increasing order of id. Walk along arrows that
   still carry tokens, preferring an agent not yet visited in this walk.
2. As soon as the walk reaches an agent already on it, a loop is closed.
   Remove one token from each arrow of the loop, and add *L*² to the score.
3. Walk again from the same agent while its walks keep closing loops. As
   soon as a walk ends at an agent with no arrow left to follow, move on to
   the next agent. The walk from an agent always takes the same path: the
   first unvisited target in order of id. So an agent is given up even when
   another path from it would have closed a loop.

Two limits keep the search short. A walk is given up after 4,096 steps, and
the peel of one flow stops after 20,000 loops.

| field | definition |
|---|---|
| `lightningScore` | Σ *L*² over the loops peeled |
| `cyclingShare` | the tokens of flow the loops used, ÷ `totalFlow` |
| `lightningLongest` | the longest loop found |

Being greedy, these are **lower bounds**: the true maximum is at least this.

#### The ceiling: what conservation forbids

Every token that goes round a loop leaves an agent and comes back to it, so
loops change no agent's balance. Whatever net imbalance there is — agents
that received more than they sent, or the reverse — cannot be in loops.
Write in(*u*) for the tokens *u*'s neighbours staked on its node, and
out(*u*) for the tokens *u* staked on theirs. Then

$$
\texttt{flowImbalance} = \frac{\tfrac12 \sum_u \bigl| \text{in}(u) - \text{out}(u) \bigr|}{\texttt{totalFlow}} .
$$

In words: take the gap between what each agent received and what it sent,
counted positive either way. Add up the gaps, halve the sum, and divide by
the whole flow. It is halved because every stranded token appears twice:
once as a surplus, once as a deficit. This is exact, and an **upper bound**:
at most 1 − `flowImbalance` of the flow can go round. The truth lies between
`cyclingShare` and that ceiling.

#### The net version

Two neighbours staking on each other form a loop of length 2:
[circulation](measuring-tokens.md#flows-that-go-round) of the most trivial kind. The `net…`
statistics repeat everything on the net flow, in which each pair's two
directions have cancelled ([Cancelling: the net flow](measuring-tokens.md#cancelling-the-net-flow)).
There, loops of length 2 cannot exist. The statistics are
`netLightningScore`, `netCyclingShare` (÷ the net total) and
`netLightningLongest`. A loop that survives cancellation is tokens genuinely
going *round*.

Cancelling changes no agent's balance, so the same stranded tokens bound the
net flow too. A share `flowImbalance` ÷ `netFlowShare` of the net flow must
run one way, and at most the rest can go round.

#### An example: three agents

Take agents A, B and C, as in [Token flow](measuring-tokens.md#the-flow-of-a-game). A stakes 3
on B, B 3 on C, C 3 on A, and B 2 on A. The total flow is 11. The loop
A → B → C → A can be peeled three times. That uses 9 tokens, and each peel
scores 3² = 9, so the score is 3 · 9 = 27. What remains is B → A (2), with no
loop, so `cyclingShare` = 9/11.

The imbalance: A received 3 + 2 = 5 and sent 3 (+2). B received 3 and sent
5 (−2). C received 3 and sent 3 (0). Half of 2 + 2 is 2, so
`flowImbalance` = 2/11, and the ceiling is 9/11. The greedy peel found
everything there was.

After cancelling, the net flow is A → B (1), B → C (3) and C → A (3): 7
tokens. The loop can now be peeled only once, so `netCyclingShare` = 3/7
and `netLightningLongest` = 3. The same 2 tokens are stranded, so the
ceiling allows 5 of the 7. But this time the floor, 3/7, is the truth. The 2
stranded tokens run B → C → A, two steps each, so they use up 4 tokens of the
net flow, not 2.

#### Where lightning is used

A run records these statistics after the games of every 25th iteration
([What a run records](runs-and-records.md#what-a-run-records)).

- **Do tokens go round?** In B1's worlds of 100,000 tokens, every matched
  stake is a loop of two. So between 72% and 77% of the flow as staked goes
  round, yet the greedy peel finds only 13%: a weak floor. After cancelling,
  it finds 0.3% of the net flow on loops, where the ceiling allows 16%
  ([Do tokens go round?](b1-the-game-and-its-flows.md#do-tokens-go-round)).
- **Every world.** [Lightning in every world](b1-the-game-and-its-flows.md#lightning-in-every-world)
  reads the floor in every world the notebook has made. One rule decides how
  much goes round: whether a connection must carry tokens to survive. And the
  longest loop found is as long as a world is wide.
- **Over many games.** The greedy count of single games is set beside the
  exact split of fifty games in
  [A greedy count and an exact one](b1-the-game-and-its-flows.md#a-greedy-count-and-an-exact-one).
  The split is explained under [Flows that go round](measuring-tokens.md#flows-that-go-round).
- **A yardstick for evolution.** How much of the flow goes round is one
  statistic that might keep rising in a world that keeps evolving
  ([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).

## Lives and ages

From [Measuring lives: births, deaths, ages and estates](measuring-lives.md).

### Why the living are older than the dead

Picture a hospital where most patients go home after a day, and a few stay a
month. On any day you visit, the long stays fill far more beds than their
number suggests. A patient who stays thirty days is there on thirty days; one
who stays a day, on one.

Agents are the same. Suppose half of all agents live 1 iteration and half live
99. The **average lifetime** is 50. But look at the world at any moment. An
agent of the second kind is alive at 99 times as many moments as one of the
first. So 99 in 100 of the agents you see are long-lived ones: the **living
are a sample weighted by lifetime**.

That is why the living look so old. In the 30 B1 worlds of 100,000 tokens
(Experiment 10), about half of the agents that live through a game are gone
within ten iterations. Two thirds are, if the newborns that never do are
counted too. Yet the median age of the agents alive at the end is 54
([Births, deaths and ages](b1-lives-and-wealth.md#births-deaths-and-ages)).

In general, take a world in a steady state, where lifetimes *L* occur with
frequency *p*(*L*) and have mean μ. The agents alive at a random moment have
lifetimes with frequency

$$
\frac{L \, p(L)}{\mu} :
$$

each lifetime's frequency multiplied by its length, and divided by the mean so
that the frequencies add up to one. So long lives are over-represented in
proportion to their length, and each such agent is equally likely to be at any
point of its life. This is the *inspection paradox* of renewal theory.

The B1 worlds of 10,000 tokens are not exactly in a steady state: their births
slow down for over a thousand iterations. But the effect is the same in kind.
Those of 100,000 tokens are in one from iteration 500 on
([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)).

### The ages of a steady world

The same idea, turned into a formula for the ages of the living. Suppose that
in every iteration, always, *B* newborns live through their first game.
Suppose too that every agent's life follows the same survival curve *S*(*L*),
the share of lives lasting at least *L* iterations. The agents of age *a* alive
after a game were born *a* iterations earlier, and their lives lasted at least
*a* + 1 games. There are *B* · *S*(*a* + 1) of them. Summed over all ages, the
living number *B* · Σ *S*(*L*), and Σ *S*(*L*) over *L* ≥ 1 is the mean
lifetime μ. So in such a world

$$
\text{share of the living of age } a = \frac{S(a+1)}{\mu} ,
$$

and in particular the newborns, of age 0, are a share 1/μ of the living. This
is the **stationary population** of demography. Its ages follow from the
lifetimes alone. A world whose living are older than its lifetimes say has had
more births in the past than now.

Two examples. In the B1 worlds of 100,000 tokens, the lives begun from
iteration 500 to 1,399 last 41.3 iterations on average. So the newborns should
be 1/41.3 = 2.4% of the living, and the median age of the living 55. Over
iterations 500 to 1,499 the worlds had 2.3 to 2.6% newborns, and a median age
of 55 ([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)).

In B2 every agent lives exactly 100 iterations. So in a steady B2 world, each
age from 0 to 99 holds one agent in a hundred. Such a world keeps its size only
if one agent in a hundred has a child in every iteration. B2's worlds in
Experiment 14 had 1.00 to 1.05 births per hundred agents from iteration 200 to
1,000 ([Building B2: a world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#old-age)).

### The risk of dying at each age

The survival curve *S*(*L*) is built from one number per age. An agent
**reaches** age *a* when its lifetime is at least *a*, and its life **ends
there** when its lifetime is exactly *a*. Of the agents whose lives reach age
*a*, the share whose lives end there is

$$
h(a) = \frac{d(a)}{n(a)} ,
$$

with *n*(*a*) the lives lasting at least *a* (finished or not) and *d*(*a*)
the finished lives of exactly *a*. This is the **hazard**, the chance of dying
at age *a* for an agent that got there. To last at least *L* iterations, a
life must get past every age before *L*, so

$$
S(L) = \prod_{a < L} \bigl(1 - h(a)\bigr) .
$$

If the hazard is the same at every age, an agent's past says nothing about its
future: lives are **memoryless**. Then *S*(*L*) falls along a straight line
when *S* is drawn on a logarithmic axis against a plain axis of *L*.

A hazard that falls with age means that those who survive are safer than the
young. The young die most, as in a population with high infant mortality.
Either each agent grows safer as it lives, or the fragile die first and leave
the robust. A hazard that rises with age is ageing in the usual sense.

The worlds show both ends. In the settled B1 worlds of 100,000 tokens the
hazard falls at every age. It is 21% at age 1, 2.7% at 10 and 1.1% at 100
([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)). In B2 it is 0 at
every age below 100, and every agent that reaches 100 dies at it: ageing at
its sharpest.

### Cohorts

Agents born in the same stretch of iterations are a **cohort**. Following each
cohort's lives says how life expectancy changes over a world's life. Cohorts
born late can only be followed to the end of the run. The Kaplan–Meier
estimate leaves their unfinished lives open, rather than cutting them short.

To compare cohorts on equal terms, each is measured by how many of its first
100 iterations an agent lives on average. That is the **restricted mean
lifetime**,

$$
\sum_{L=1}^{100} S(L) ,
$$

the area under the survival curve from *L* = 1 to 100. Every cohort born at
least 100 iterations before the end can be measured by it in full. In the
median B1 world of 100,000 tokens, an agent born in iterations 1 to 24 lived 5
of its first 100 iterations. One born from iteration 500 on lived 28 to 31
([How ages change over a world's life](b1-lives-and-wealth.md#how-ages-change-over-a-worlds-life)).

## Assembly index and assembly

How many steps does it take to make a thing, and how many copies of it are
there? **Assembly theory** asks both (Marshall and others 2021; Sharma and
others 2023). It builds an object from basic blocks by **joins**: a join lays
two pieces already made together, and whatever is made can be used again for
free. The **assembly index** *a* of an object is the fewest joins that make
it. Its **copy number** *n* is how many identical copies of it a collection
holds. The **assembly** of the collection adds up its kinds of object,
weighing each by e to the power of its index and counting only its copies
beyond the first:

$$
A = \sum_i e^{a_i} \, \frac{n_i - 1}{N_T},
$$

with *N_T* the objects in all. The theory reads many copies of objects with a
high index as the mark of selection: chance alone seldom makes the same
complicated thing twice.

### An example of the index

In a network the block is one link, and an object can be an agent's **ego
network**: the agent, its neighbours and the links among them. A star of
eight links takes three joins: two links make a star of two, two of those a
star of four, two of those a star of eight. A star of seven links takes four,
for each join can only add two sizes already made (1, 2, 3, 4, 7): the fewest
joins for *m* links is the length of the shortest **addition chain** to *m*,
written l(*m*). In a triangular lattice every agent's ego network is the same
wheel of six spokes, twelve links built in four joins, so its 10,000 copies
give A = e⁴ · 9,999 / 10,000 ≈ 54.6.

### Where it is used

- [Assembly theory](measures-atlases-and-assembly.md#assembly-theory-how-many-steps-to-make-a-thing-and-how-many-copies): the index and the assembly of the
  ego networks of B1, B2 and round 3, each against its [random twin](measuring-space.md#the-random-twin-is-a-world-a-space "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.") and four
  networks with no history; and why a [genotype](measuring-evolution.md#genotypes-families-and-the-common-ancestor "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.")'s path in the family tree
  gives no index worth the name.

## Where to go next

- [Statistics: the tools of the notebook](statistics.md)
- [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md)
- [Measuring tokens: inequality and flow](measuring-tokens.md)
- [Measuring lives: births, deaths, ages and estates](measuring-lives.md)

## How our view changed

- **12 November 2026.** Gathered, unchanged, from five reference pages, since
  only pages of the Archive use them: from Statistics, fitting a power law,
  where, who and chance, and the participation ratio; from Measuring a network,
  assortativity, preferential attachment and scaling relations; from Measuring
  tokens, mobility and lightning; from Measuring lives, four sections on age;
  and from Measuring evolution, assembly.

## References

- Clauset, A., Shalizi, C. R. and Newman, M. E. J. (2009). Power-law
  distributions in empirical data. *SIAM Review* 51, 661–703.
- Bell, R. J. and Dean, P. (1970). Atomic vibrations in vitreous silica.
  *Discussions of the Faraday Society* 50, 55–61.
- Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K. and
  Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
  measurement. *bioRxiv* 214262.
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
- Chetty, R., Hendren, N., Kline, P. and Saez, E. (2014). Where is the land
  of opportunity? The geography of intergenerational mobility in the United
  States. *Quarterly Journal of Economics* 129, 1553–1623.
- Shorrocks, A. F. (1978). The measurement of mobility. *Econometrica* 46,
  1013–1024.
- Marshall, S. M., Mathis, C., Carrick, E. and others (2021). Identifying
  molecules as biosignatures with assembly theory and mass spectrometry.
  *Nature Communications* 12, 3033.
- Sharma, A., Czégel, D., Lachmann, M., Kempes, C. P., Walker, S. I. and
  Cronin, L. (2023). Assembly theory explains and quantifies selection and
  evolution. *Nature* 622, 321–328.

<!-- turns -->
---

← [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md) · [Side studies](../README.md#side-studies)

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (16)
> [Glossary](glossary.md) · [Measuring lives: births, deaths, ages and estates](measuring-lives.md) · [Measuring tokens: inequality and flow](measuring-tokens.md) · [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Brains replaced by their average](brains-replaced-by-their-average.md) · [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [How runs are made, recorded and costed](runs-and-records.md) · [Statistics: the tools of the notebook](statistics.md) · [B1: lives and wealth](b1-lives-and-wealth.md) · [B1: the game and its flows](b1-the-game-and-its-flows.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [Space at scale (side study)](space-at-scale.md) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md)
<!-- /turns -->
