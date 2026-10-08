# Degree

The **degree** of an agent is the number of connections it has — the number
of its neighbours. It is the simplest thing to know about a place in a
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

recorded as `meanDegree`. A settled baseline world has about 1.6
connections per agent, so a mean degree of about 3.3.

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
the share of agents with degree *d* falls like *d*^−γ — a power law — the CCDF
falls like *d*^−(γ−1) and is a straight line on log–log axes, with slope
−(γ − 1).

## Hubs and leaves

A **leaf** is an agent of degree 1: it hangs on the network by a single
connection, and the cut of that one connection cuts it off (`leaves` in the
statistics). A **hub** is an agent of very high degree. A settled baseline world has
both. After the last game of the 26 baseline worlds that lived to the end,
37% of all agents had exactly one connection and 23% two; 4% had ten or more;
and the largest hub had 1,399
([Chapter 15](../chapters/15-what-shape-does-the-network-take.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
