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

With few pairs, a sizeable *r* can come about by chance. The book checks
this with a [permutation test](permutation-test.md): shuffle the *y* values
against the *x* values many times, and see how often the shuffled pairs give
an *r* at least as large. In [Chapter 12](../chapters/12-how-much-does-the-seed-decide.md),
for 26 worlds, the correlation 0.12 between the number of agents in the first
and second half of the run is reached by shuffled pairings 28 times in 100 —
nothing; 0.41 for connections only 2 times in 100.

## Correlation is not cause

[Chapter 13](../chapters/13-where-do-the-tokens-go.md) finds that agents with
more connections hold more tokens (*r* = 0.50 between the number of
connections and the logarithm of the tokens). That alone cannot say whether
connections make an agent rich, riches bring connections, or something else
brings both.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
