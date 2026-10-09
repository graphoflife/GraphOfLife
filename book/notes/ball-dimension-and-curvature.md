# Dimension and curvature from ball growth

How many agents are within *r* steps of a given agent? The answer, as a
function of *r*, says what kind of space a network is — a line, a sheet, a
tree — and two of the viewer's statistics read it off.

## Balls and shells

The **ball** of radius *r* around agent *u* is every agent at distance at most
*r* from it; the **shell** of radius *r* is every agent at distance exactly
*r*. In a space of dimension *d*, a ball holds about *r*^*d* points and a shell
about *r*^(*d*−1):

- on a line, the shell at distance *r* has 2 points (*d* = 1);
- on a square grid, about 4*r* (*d* = 2);
- in a cubic grid, about 6*r*² (*d* = 3).

In a **tree** in which every agent has *b* children, the shell grows as
*b*^*r* — faster than any power of *r*. A network like that has no finite
dimension.

## The dimension

The viewer walks out from up to 24 agents spread through the network,
averages the shell sizes *S*(*r*) for *r* = 1 … 5, drops the radii at which the
ball already holds half the world (beyond that the growth measures the edge of
the network rather than its shape), and fits a straight line to
ln *S*(*r*) against ln *r* by [least squares](least-squares.md):

$$
\texttt{dimension} = \text{slope} + 1 .
$$

Checked on lattices, it gives 1.00 for a line, 1.92 for a square grid and
2.56 for a cube: exact in one dimension and increasingly short above, since
five steps are too few for the growth to reach its limit. Read it as a rough
index.

## The curvature

On a curved surface balls grow differently from flat space: on a sphere
(positive curvature) more slowly, on a saddle (negative curvature) faster. For
a ball in *d* dimensions with **Ricci scalar** *R*, the shell grows as

$$
S(r) \propto r^{d-1} \left( 1 - \frac{R\, r^2}{6 d} + \dots \right) .
$$

So the program fits ln *S*(*r*) against **both** ln *r* and *r*² (at least four
radii are needed): the first coefficient gives *d* − 1, the second −*R*/(6*d*).
`ricciCurvature` is that *R*.

- **Negative**: neighbourhoods hold more than flat space allows — branching,
  tree-like, expander-like networks.
- **Positive**: neighbourhoods close back on themselves and hold less.

## A caution

Both are fitted to five or fewer points, on networks that are not
lattices. They are indices for comparing worlds and moments, not exact
geometry. [Chapter 31](../chapters/31-the-geometry-of-a-world.md) reads them on
the baseline worlds, beside a picture of the ball growth itself.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
