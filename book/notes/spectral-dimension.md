# The spectral dimension

How many dimensions does a space have? One way to find out is to measure it
([Dimension and curvature](ball-dimension-and-curvature.md)): count how many
points lie within *r* steps of a point. In *d* dimensions the count grows
like *r*^*d*. Another way is to **get lost in it**. Release a walker that
steps at random, and see how often it comes back to where it started.

## Why returning measures dimension

After *t* random steps, a walker has typically wandered about √*t* steps
away, in every direction open to it. In *d* dimensions it is then spread
over a region of about (√*t*)^*d* = *t*^(*d*/2) points, more or less evenly.
So the probability that it is back at its start is about one over that:

$$
P(t) \propto t^{-d_s/2} .
$$

The exponent defines the **spectral dimension** *d*ₛ. On a line, *P*(*t*)
falls like *t*^(−1/2); on a plane like *t*^(−1); in our three-dimensional
space like *t*^(−3/2). Read off two times, *t* and 2*t*:

$$
d_s(t) = -2 \, \frac{\ln P(2t) - \ln P(t)}{\ln 2} .
$$

The name comes from the fact that *P*(*t*) is set by the spectrum of the
network's Laplacian, the matrix behind diffusion on it (Alexander and Orbach
1982). The heat equation of [Chapter 7](../chapters/07-physical-inspiration.md)
spreads heat exactly as the walker's probability spreads.

## The walk used here

- **Lazy:** at every step the walker stays where it is with probability ½,
  and otherwise moves to a neighbour drawn at random. Without the laziness,
  a walk on a network whose points fall into two alternating classes — a
  grid, a tree — could return only at even steps, and *P*(*t*) would jump
  between zero and twice its value.
- **Exact:** instead of simulating walkers, the book propagates the whole
  probability distribution, step by step, from a few dozen starting points.
- **Minus the resting state:** in a finite world, the walker ends up spread
  over the whole world, at each point in proportion to its connections. The
  book subtracts that resting probability from *P*(*t*), so that only the
  approach to rest is measured, and stops when what is left falls below 5/*n*
  for a world of *n* points.

## Two dimensions that need not agree

On a regular grid, the ball dimension and the spectral dimension are the
same. On irregular spaces they differ, and how they differ is informative:

- On a **tree**, balls grow quickly but a walker keeps running into dead
  ends and coming back. A tree drawn at random has a ball dimension of 2 and
  a spectral dimension of 4/3 (Aldous 1991; Durhuus, Jonsson and Wheater
  2007).
- On a **random network** in which every point has the same few neighbours,
  balls grow exponentially and the walker almost never returns: both
  dimensions are infinite. Measured at growing scales, the readings keep
  rising.
- In **quantum gravity**, *d*ₛ is a favourite probe of space-time built
  from discrete pieces. The causal dynamical triangulations of Ambjørn,
  Jurkiewicz and Loll (2005) give *d*ₛ ≈ 4 at large scales and ≈ 2 at
  small ones.

## Where it is used

[Chapter 23](../chapters/23-how-many-dimensions-does-a-world-have.md)
measures both dimensions on the worlds, after calibrating them on spaces
whose dimension is known.

## References

- Aldous, D. (1991). The continuum random tree II: an overview. In
  *Stochastic Analysis*, Cambridge University Press, 23–70.
- Alexander, S. and Orbach, R. (1982). Density of states on fractals:
  "fractons". *Journal de Physique Lettres* 43, L625–L631.
- Ambjørn, J., Jurkiewicz, J. and Loll, R. (2005). Spectral dimension of the
  universe is scale dependent. *Physical Review Letters* 95, 171301.
- Durhuus, B., Jonsson, T. and Wheater, J. F. (2007). The spectral dimension
  of generic trees. *Journal of Statistical Physics* 128, 1237–1260.
