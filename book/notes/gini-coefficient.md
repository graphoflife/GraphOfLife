# The Gini coefficient and the Lorenz curve

How unequal is the wealth of a world? The founders start equal; later, some
agents hold one token and some hold thousands. Two standard tools measure
this: the **Lorenz curve**, a picture, and the **Gini coefficient**, one
number (Lorenz 1905; Gini 1912).

## The Lorenz curve

Sort the *n* agents from poorest to richest, *x*₍₁₎ ≤ *x*₍₂₎ ≤ … ≤ *x*₍ₙ₎,
and let *S* = Σ *x*₍ᵢ₎ be all their tokens (in a world of *T* tokens, *S* = *T*).
For every *i*, mark the point

$$
\left( \frac{i}{n},\ \frac{x_{(1)} + \dots + x_{(i)}}{S} \right):
$$

the share of agents counted from the poorest, against the share of all tokens
they hold. Join the points, starting at (0, 0). The curve ends at (1, 1).

- If everyone holds the same, the poorest *i*/*n* of the agents hold *i*/*n*
  of the tokens: the curve is the **diagonal**.
- The more unequal the world, the further the curve **sags** below the
  diagonal. If one agent holds everything, it runs along the bottom and
  jumps to 1 at the end.

![How the Gini coefficient is read off a Lorenz curve](../diagrams/gini.svg)

## The Gini coefficient

The Gini coefficient is the area between the diagonal and the Lorenz curve,
divided by the whole area under the diagonal (which is ½):

$$
G = \frac{A}{A + B} = 1 - 2B ,
$$

where *A* is the area between the diagonal and the curve and *B* the area
under the curve. It is 0 when everyone holds the same, and (*n* − 1)/*n* —
almost 1 — when one agent holds everything.

For sorted values it can be computed directly:

$$
G = \frac{\sum_{i=1}^{n} (2i - n - 1)\, x_{(i)}}{n \sum_{i=1}^{n} x_{(i)}} .
$$

## What a value means

There is a second, equivalent definition that makes values easier to read:
*G* is half the **mean absolute difference** between two agents, relative to
the mean,

$$
G = \frac{1}{2\bar x} \cdot \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \lvert x_i - x_j \rvert .
$$

So a Gini of 0.5 says: two agents picked at random (independently, the same
one possibly twice) differ on average by as much as the average agent holds.

## A worked example

Five agents with 1, 1, 2, 4 and 12 tokens: *n* = 5, *S* = 20.

| *i* | *x*₍ᵢ₎ | 2*i* − *n* − 1 | product |
|---|---|---|---|
| 1 | 1 | −4 | −4 |
| 2 | 1 | −2 | −2 |
| 3 | 2 | 0 | 0 |
| 4 | 4 | 2 | 8 |
| 5 | 12 | 4 | 48 |
| | | sum | 50 |

*G* = 50 / (5 · 20) = **0.5**. Check with the second definition: the mean is
4; the sum of |*xᵢ* − *xⱼ*| over all 25 ordered pairs is 100; 100/25 = 4, and
4/(2 · 4) = 0.5. ✓

The Lorenz points are (0.2, 0.05), (0.4, 0.10), (0.6, 0.20), (0.8, 0.40),
(1, 1): the poorest 80% hold 40% of the tokens, the richest agent 60%.

## In the code

`_gini(values)` in `gol_series.py` computes
(*n* + 1 − 2 Σᵢ *C*ᵢ / *C*ₙ) / *n*, where *C*ᵢ = *x*₍₁₎ + … + *x*₍ᵢ₎ are the
running sums of the sorted values — the same number as the formula above,
rearranged. It is recorded as `gini` after every phase.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
