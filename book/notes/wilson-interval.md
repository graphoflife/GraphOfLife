# The Wilson interval

Four of thirty baseline worlds died out: 13%. With only thirty worlds, the
true rate could easily be lower or higher. The **Wilson interval** (Wilson
1927) gives a range for a share estimated from a count.

## The formula

If *k* of *n* trials are "successes", the observed share is *p̂* = *k*/*n*.
With *z* = 1.960 for 95%, the Wilson interval is

$$
\frac{\hat p + \dfrac{z^2}{2n} \;\pm\; z \sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}} .
$$

## Example

*k* = 4, *n* = 30: *p̂* = 0.1333, *z*²/*n* = 0.1280.

- centre: (0.1333 + 0.0640) / 1.1280 = 0.1749;
- half-width: 1.960 · √(0.1333 · 0.8667/30 + 3.8415/3,600) / 1.1280
  = 1.960 · √(0.003852 + 0.001067) / 1.1280 = 1.960 · 0.07013 / 1.1280 = 0.1219.

The interval is **5.3% to 29.7%**: thirty worlds say only that between about
one world in twenty and one in three dies.

## Why not the simple ± formula?

The textbook interval *p̂* ± *z*√(*p̂*(1 − *p̂*)/*n*) gives 1.2% to 25.5% here,
and for small counts it is badly wrong: with *k* = 0 it gives the "interval"
0 to 0. The Wilson interval stays inside 0 to 1, is never empty, and covers
the true share about as often as it promises.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
