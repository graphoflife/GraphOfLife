# Between worlds and within them

The worlds of thirty seeds differ, and each world also changes over time.
How much of all the variation is **between** the worlds — something each seed
fixes — and how much is **within** each world, its wander over time? This
note splits it.

## The split

As for the [autocorrelation](autocorrelation.md), cut each of *N* worlds'
lives (iterations 100 to 2,999) into *m* = 29 stretches of 100 iterations
and take the mean of the statistic in each: levels *x*ᵢⱼ for world *i* and
stretch *j*. Let *x̄*ᵢ be world *i*'s own average of its 29 levels.

- **Between:** how much the worlds' own averages differ — the variance of
  *x̄*₁, …, *x̄*_N (with *N* − 1 in the denominator):

  $$
  B = \frac{1}{N-1}\sum_{i=1}^{N} \bigl(\bar x_i - \bar{\bar x}\bigr)^2 .
  $$

- **Within:** how much a world moves around its own average — the variance of
  each world's 29 levels (with *m* − 1 in the denominator), averaged over the
  worlds:

  $$
  W = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{m-1}\sum_{j=1}^{m} \bigl(x_{ij} - \bar x_i\bigr)^2 .
  $$

The **share between** is *B* / (*B* + *W*).

## How to read it

- Close to 1: worlds differ, but each stays where it is. Knowing the seed
  tells you the level.
- Close to 0: every world covers much the same range over time. The seed
  tells you little.

In the baseline the share between is between a tenth and a quarter for every
statistic looked at — 17% for the number of agents
([Chapter 13](../chapters/13-how-much-does-the-seed-decide.md)). So most of
the variation is time, not seed.

## A caution

Even *B* is partly wander: the average of 29 levels that are correlated
over several stretches still carries some of the noise of the wander. A
share between of 17% is therefore an upper bound on what the seed itself
decides. The sharper test is whether a world's first half predicts its
second half, across worlds ([Correlation](correlation.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
