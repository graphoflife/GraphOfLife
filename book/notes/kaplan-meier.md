# Survival curves (Kaplan–Meier)

*What share of agents live at least L iterations?* If every life had ended
by the end of the run, the answer would be a simple count. But some are
still going, and for them we know only that they lasted **at least** so
long. The **Kaplan–Meier estimate** (Kaplan and Meier 1958) uses exactly that
much and no more.

## The idea: survive one step at a time

To live at least *L* + 1 iterations, a life must first live at least *L*,
and then not end at *L*. So the share surviving to *L* + 1 is the share
surviving to *L* times the chance of not ending at *L*, among the lives that
got that far. Chaining the steps:

$$
S(L) = \prod_{\ell < L} \left( 1 - \frac{d_\ell}{n_\ell} \right),
$$

where, for each length ℓ,

- *n*ℓ is the number of lives **at risk** at ℓ: lives that lasted at least ℓ
  — ended at ℓ or later, or still going after having lasted ℓ or more;
- *d*ℓ is the number of lives that **ended** at exactly ℓ.

A life still going at the end of the run, having lasted *L*₀, counts in
*n*ℓ for every ℓ ≤ *L*₀ and is then dropped without ever counting as an end.
That is how "at least *L*₀" is used: no more, no less.

## A worked example

Six agents' lives, in iterations; a + means still alive at the end of the
run: 1, 1, 2, 3+, 4, 6+.

| ℓ | at risk *n*ℓ | ended *d*ℓ | 1 − *d*/*n* | share living longer than ℓ |
|---|---|---|---|---|
| 1 | 6 | 2 | 4/6 | 0.667 |
| 2 | 4 | 1 | 3/4 | 0.500 |
| 3 | 3 | 0 (the 3+ leaves here) | 1 | 0.500 |
| 4 | 2 | 1 | 1/2 | 0.250 |
| 6 | 1 | 0 (the 6+ leaves here) | 1 | 0.250 |

So half of all lives last at least 3 iterations, and a quarter at least 5.
Ignoring the unfinished lives would have given a quarter for "at least 3"
(only the 4 of the four finished lives); counting them as if they ended
would understate the long lives too.

## Drawing it

The curve *S*(*L*) starts at 1 and falls in steps. The book draws it on
[logarithmic axes](logarithmic-axes.md), so that both the many short lives
and the few long ones are visible ([Chapter 12](../chapters/12-births-deaths-and-ages.md),
[Chapter 17](../chapters/17-genotypes-and-lineages.md)).

## The median lifetime

The **median lifetime** is the smallest *L* with *S*(*L* + 1) ≤ ½: half of
all lives end by then.

## In the code

`kaplan_meier(lives)` in `book_figures.py`, where a life is (first
iteration, last iteration, still going).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
