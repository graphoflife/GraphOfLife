# Fitting a power law properly

Is the number of connections, or of tokens, distributed as a **power law**,
P(*k*) ∝ *k*^−γ? And if so, with which γ? The obvious method — draw the
distribution on log–log axes and fit a line — is biased. This note gives the
method the literature recommends (Clauset, Shalizi and Newman 2009) and says
how the viewer uses both.

## The obvious way, and its problem

The complementary cumulative distribution — the share *P*(≥ *k*) of values at
least *k* — falls as *k*^−(γ−1) for a power law, so a straight line through it
on log–log axes has slope 1 − γ ([Logarithmic axes](logarithmic-axes.md)).
The viewer's `degreeExponent` and `tokenExponent` are exactly that:
[least squares](least-squares.md) of ln *P*(≥ *k*) on ln *k* over the distinct
values, exponent = 1 − slope.

The trouble: neighbouring points of a cumulative distribution are not
independent (each contains the next), the far tail is a handful of agents,
and the least-squares line weights every distinct value equally. The
estimate wanders, and R² looks good whatever the truth.

## Maximum likelihood

Suppose the values *x*₁ … *x*ₙ from some *x*_min upward follow the density
p(*x*) = (γ − 1) *x*_min^(γ−1) *x*^−γ. The γ that makes the observed values
most probable is

$$
\hat\gamma = 1 + n \Bigl[\sum_{i=1}^{n} \ln \frac{x_i}{x_{\min}}\Bigr]^{-1} ,
$$

with a standard error of about (γ̂ − 1)/√*n*. For whole numbers like a degree,
*x*_min is replaced by *x*_min − ½, which corrects most of the error of
treating them as continuous.

**Example.** A tail of four degrees 2, 3, 4, 8 with *k*_min = 2: the
logarithms of 2/1.5, 3/1.5, 4/1.5, 8/1.5 add up to 3.636, so
γ̂ = 1 + 4/3.636 = 2.10 — known to about ±1.1/√4 = ±0.55, which is to say
hardly at all. A tail needs many values.

## Where does the tail start?

Real distributions are a power law, if at all, only above some *x*_min. So
the program tries candidates and keeps the one whose fitted law sits closest
to the data:

1. Take up to 24 candidate *k*_min from the distinct values.
2. For each with at least 25 values at or above it, estimate γ̂ as above.
3. Measure the **Kolmogorov–Smirnov distance**: the largest vertical gap
   between the cumulative distribution of the tail and that of the fitted law.
4. Keep the *k*_min with the smallest distance.

The viewer reports γ̂ (`degreeGamma`), the *k*_min (`degreeKMin`), the share
of all agents in the tail (`degreeTailShare`), the distance
(`degreeGammaKS`), and R² of the cumulative distribution over the tail
(`degreeGammaR2`, a description, not a test).

## What this does not tell you

The method finds the best power law; it does not prove there is one. Two
further checks would be needed:

- **Is the gap small enough?** Generate many samples from the fitted law,
  fit each the same way, and see how often their distance is as large as the
  data's (the "goodness-of-fit p-value").
- **Does a different law fit as well?** A **log-normal** — a distribution
  whose logarithm is normal — or a power law with an exponential cut-off
  often fits as well over the range one has.

And a tail of 5% of a world is a small part of it: a power law there says
nothing about the other 95%. [Chapter 29](../chapters/29-power-laws-real-and-apparent.md)
fits both methods to the baseline worlds.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
