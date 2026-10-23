# Assortativity

Do well-connected [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") connect to other well-connected agents, or to
poorly connected ones? **Assortativity** answers with one number between −1
and 1 (Newman 2002).

## The idea

Look at every [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") and write down the [degrees](degree.md) of its two
ends: a pair (*j*, *k*). Then ask whether *j* and *k* go together — the
[correlation](correlation.md) of the two ends over all connections.

- **Positive** (assortative): like joins like — [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") to hubs, the sparsely
  connected among themselves. Friendship networks are typically so
  (about +0.1 to +0.3).
- **Negative** (disassortative): hubs sit among the sparsely connected, like
  the centre of a star. The internet, the web and networks of proteins are
  typically so (about −0.1 to −0.2).
- **Zero**: the degree of one end says nothing about the other.

## The formula

A connection has no first and second end, so every connection is counted
both ways round, as (*j*, *k*) and (*k*, *j*). With *M* connections, the
correlation of these 2*M* pairs is

$$
r = \frac{\frac{1}{M}\sum_e j_e k_e - \Bigl[\frac{1}{M}\sum_e \tfrac12 (j_e + k_e)\Bigr]^2}
         {\frac{1}{M}\sum_e \tfrac12 (j_e^2 + k_e^2) - \Bigl[\frac{1}{M}\sum_e \tfrac12 (j_e + k_e)\Bigr]^2} .
$$

The numerator is the covariance of the two ends, the denominator the variance
of the degree at one end of a connection.

## Two examples

- **A star**: one hub joined to three leaves. Every connection is (3, 1).
  Σ *jk* / *M* = 3; the mean end degree is (3 + 1)/2 = 2; the mean squared end
  degree is (9 + 1)/2 = 5. *r* = (3 − 4)/(5 − 4) = **−1**: perfectly
  disassortative.
- **A path of four**, a–b–c–d, degrees 1, 2, 2, 1. The connections are (1, 2),
  (2, 2), (2, 1). Σ *jk* / *M* = 8/3; the mean end degree 10/6; the mean
  squared end degree 18/6 = 3. *r* = (8/3 − 25/9)/(3 − 25/9) = **−0.5**.

## A caution

When a few hubs hold many of the connections, there are not enough hubs for
every hub to have hubs as [neighbours](the-world.md#the-network "An agent joined to another by a connection."): most of a hub's neighbours *must* be
small. So a network with a heavy-tailed degree distribution tends to come out
disassortative even if it is wired at random given its degrees. A negative
*r* is only interesting beyond that pull.

The viewer's `assortativity` is *r* over the [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.")'s connections. A settled
[baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world has about −0.07 at 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") and −0.11 at 10,000
([How properties scale together](how-properties-scale-together.md)).

<!-- turns -->
---

← [The spectral gap](spectral-gap.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Scaling relations between agents' properties](scaling-relations.md) →

> [!info]- Pages that link here (4)
> [Glossary](glossary.md) · [How properties scale together](how-properties-scale-together.md) · [How alike at a distance](correlation-function.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
