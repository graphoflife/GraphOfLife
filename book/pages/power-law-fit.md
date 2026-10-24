# Fitting a power law properly

Is the number of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."), or of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), distributed as a **power law**,
P(*k*) ∝ *k*^−γ? And if so, with which γ? The obvious method — draw the
distribution on log–log axes and fit a line — is biased. This note gives the
method the literature recommends (Clauset, Shalizi and Newman 2009) and says
how the viewer uses both.

## The obvious way, and its problem

The complementary cumulative distribution — the share *P*(≥ *k*) of values at
least *k* — falls as *k*^−(γ−1) for a power law, so a straight line through it
on log–log axes has slope 1 − γ ([Logarithmic axes](correlation.md#logarithmic-axes-and-power-laws)).
The viewer's `degreeExponent` and `tokenExponent` are exactly that:
[least squares](correlation.md#fitting-a-straight-line) of ln *P*(≥ *k*) on ln *k* over the distinct
values, exponent = 1 − slope.

The trouble: neighbouring points of a cumulative distribution are not
independent (each contains the next), the far tail is a handful of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."),
and the least-squares line weights every distinct value equally. The
estimate wanders, and R² looks good whatever the truth.

## Maximum likelihood

Suppose the values *x*₁ … *x*ₙ from some *x*_min upward follow the density
p(*x*) = (γ − 1) *x*_min^(γ−1) *x*^−γ. The γ that makes the observed values
most probable is

$$
\hat\gamma = 1 + n \Bigl[\sum_{i=1}^{n} \ln \frac{x_i}{x_{\min}}\Bigr]^{-1} ,
$$

with a standard error of about (γ̂ − 1)/√*n*. For whole numbers like a [degree](degree.md "The number of an agent's connections: how many neighbours it has."),
*x*_min is replaced by *x*_min − ½, which corrects most of the error of
treating them as continuous.

**Example.** A tail of four degrees 2, 3, 4, 8 with *k*_min = 2: the
logarithms of 2/1.5, 3/1.5, 4/1.5, 8/1.5 add up to 3.636, so
γ̂ = 1 + 4/3.636 = 2.10 — known to about ±1.1/√4 = ±0.55, which is to say
hardly at all. A tail needs many values.

## Where does the tail start?

Real distributions are a power law, if at all, only above some *x*_min. So
the program tries [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.") and keeps the one whose fitted law sits closest
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

And a tail of 5% of a world — or, at 100,000 tokens, of 1% — is a small part
of it: a power law there says nothing about the rest. [Power laws, real and apparent](power-laws-real-and-apparent.md)
fits both methods to the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds.

<!-- turns -->
---

← [Muller plots](muller-plot.md) · [Methods and statistics](../README.md#methods-and-statistics) · [The power spectrum of a time series](power-spectrum.md) →

> [!info]- Pages that link here (4)
> [Glossary](glossary.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [What a run records](frames-and-stats.md) · [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md)
<!-- /turns -->
