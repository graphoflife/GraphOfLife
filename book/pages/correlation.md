# Correlation

Do two quantities go up and down together? The **correlation coefficient**
*r* answers with a number between −1 and 1.

## Definition

For *n* pairs (*x*₁, *y*₁), …, (*x*ₙ, *y*ₙ), with means *x̄* and *ȳ*, the
**Pearson correlation** is

$$
r = \frac{\sum_{i} (x_i - \bar x)(y_i - \bar y)}
         {\sqrt{\sum_{i} (x_i - \bar x)^2}\ \sqrt{\sum_{i} (y_i - \bar y)^2}} .
$$

The numerator adds up, pair by pair, whether *x* and *y* are on the same
side of their means (positive) or on opposite sides (negative). The
denominator scales the sum so that *r* is between −1 and 1.

- *r* = 1: the points lie exactly on a rising straight line.
- *r* = −1: exactly on a falling straight line.
- *r* = 0: no straight-line relation (there can still be a curved one).

*r*² is the share of the variance of *y* that a straight line in *x*
accounts for: *r* = 0.5 means a quarter.

## Example

Three worlds: (*x*, *y*) = (1, 2), (2, 2), (3, 5). Means 2 and 3. Deviations
(−1, −1), (0, −1), (1, 2). Products 1, 0, 2: sum 3. Sums of squares 2 and
6. *r* = 3 / (√2 · √6) = 3/√12 ≈ 0.87.

## Is a correlation real?

With few pairs, a sizeable *r* can come about by chance. The notebook checks
this with a [permutation test](permutation-test.md): shuffle the *y* values
against the *x* values many times, and see how often the shuffled pairs give
an *r* at least as large. In [How much does the seed decide?](how-much-does-the-seed-decide.md),
for 26 worlds, the correlation 0.12 between the number of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") in the first
and second half of the run is reached by shuffled pairings 28 times in 100 —
nothing; 0.41 for [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") only 2 times in 100.

## Correlation is not cause

[Where do the tokens go?](where-do-the-tokens-go.md) finds that agents with
more connections hold more [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") (*r* = 0.50 between the number of
connections and the logarithm of the tokens). That alone cannot say whether
connections make an agent rich, riches bring connections, or something else
brings both.

<!-- turns -->
---

← [Spread between worlds](spread-between-worlds.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Autocorrelation: how long a world remembers](autocorrelation.md) →

> [!info]- Pages that link here (13)
> [Glossary](glossary.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Assortativity](assortativity.md) · [How alike at a distance](correlation-function.md) · [Do the brains matter?](do-the-brains-matter.md) · [How worlds are measured](how-worlds-are-measured.md) · [Autocorrelation: how long a world remembers](autocorrelation.md) · [Mobility](mobility.md) · [Between worlds and within them](between-and-within.md) · [Permutation tests](permutation-test.md) · [Fitting a straight line, and R²](least-squares.md)
<!-- /turns -->
