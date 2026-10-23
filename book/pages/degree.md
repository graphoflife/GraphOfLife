# Degree

The **degree** of an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") is the number of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") it has — the number
of its [neighbours](the-world.md#the-network "An agent joined to another by a connection."). It is the simplest thing to know about a place in a
network, and much else depends on it: an agent can only stake on its
neighbours, and only its neighbours can stake on it.

## Definitions

For an agent *u* with neighbours *N*(*u*), the degree is
deg(*u*) = |*N*(*u*)|. Every connection has two ends, so summing the degrees
counts every connection twice:

$$
\sum_{u} \deg(u) = 2\,|E| .
$$

The **mean degree** of a world with |*V*| agents and |*E*| connections is
therefore

$$
\bar k = \frac{1}{|V|}\sum_u \deg(u) = \frac{2\,|E|}{|V|} ,
$$

recorded as `meanDegree`. A settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world has about 1.7
connections for every agent at 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") (1.6 at 10,000). Each connection has
two ends, so the mean degree is about 3.5 (3.3): this is what the notebook means by *links per agent*.

Three more numbers describe the degrees of a [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."): `medianDegree`, the
[median](median-and-quantiles.md); `minDegree`, which is 1 in a connected
world of more than one agent; and `maxDegree`, the largest [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."). And the
**density** is the share of all possible pairs of agents that are joined:

$$
\texttt{density} = \frac{|E|}{|V|(|V| - 1)/2} = \frac{\bar k}{|V| - 1} .
$$

A world of 1,300 agents with mean degree 3.3 has a density of 0.0025: of
every 400 pairs of agents, one is joined. Density falls as a world grows even
when every agent keeps the same number of connections, which is why the notebook
mostly uses the mean degree.

## The degree distribution

How many agents have degree 1, 2, 3, …? Two ways of drawing it:

- **The histogram**: for each *d*, the share of agents with degree exactly
  *d*.
- **The complementary cumulative distribution** (CCDF): for each *d*, the
  share of agents with degree **at least** *d*,

  $$
  P(\deg \ge d) = \frac{|\{u : \deg(u) \ge d\}|}{|V|} .
  $$

  It starts at 1 for *d* = 1 (every agent of a connected world has at least
  one neighbour) and falls to 1/|*V*| at the largest degree.

The CCDF is the better picture for wide distributions: it needs no choice of
bins, it is smooth where the histogram is noisy, and on
[logarithmic axes](logarithmic-axes.md) it shows the shape of the tail. If
the share of agents with degree *d* falls like *d*^−γ — a [power law](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones.") — the CCDF
falls like *d*^−(γ−1) and is a straight line on log–log axes, with slope
−(γ − 1).

## Hubs and leaves

A **leaf** is an agent of degree 1: it hangs on the network by a single
connection, and the cut of that one connection cuts it off (`leaves` in the
statistics). A **hub** is an agent of very high degree. A settled baseline world has
both. After the last [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") of the thirty baseline worlds of 100,000 tokens,
35% of all agents had exactly one connection and 23% two; 4.7% had ten or
more; and the largest hub had 10,735
([What shape does the network take?](what-shape-does-the-network-take.md); at 10,000
tokens, 37%, 23%, 4% and 1,399).

<!-- turns -->
---

← [How a world breaks](how-a-world-breaks.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Stars and hubs](stars-and-hubs.md) →

> [!info]- Pages that link here (11)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [The even split](even-split.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [Stars and hubs](stars-and-hubs.md) · [Assortativity](assortativity.md) · [The random twin: is a world a space?](random-twin.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
