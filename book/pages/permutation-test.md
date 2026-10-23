# Permutation tests

A difference between two conditions, or a correlation between two
measurements, could be real — or chance. A **permutation test** asks: if
there were no real difference, how often would chance alone produce one at
least this large? It does so by **shuffling**, with no assumption about
distributions.

## The logic

Suppose a [setting](settings.md "One number or switch of a run's rules, fixed before it starts.") changes nothing. Then the label "condition A" or
"condition B" on each world is arbitrary: any relabelling of the worlds is as
likely as the one we have. So relabel at random many times, recompute the
difference each time, and see where the real one falls among the shuffled
ones. If it is larger than almost all of them, "no difference" is hard to
believe.

The share of shuffles that give a difference at least as large (in size) as
the real one is the **p-value**. The notebook computes it as

$$
p = \frac{1 + \#\{\text{shuffles at least as extreme}\}}{1 + \text{number of shuffles}} ,
$$

with 10,000 shuffles; the 1s count the real labelling as one of the
possibilities, so *p* is never exactly 0.

## Three versions in this notebook

- **Two independent groups.** Pool the values of both conditions, deal them
  at random into two groups of the original sizes, and take the difference of
  the means.
- **Pairs.** If each world of condition B has a partner in A with the same
  [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."), take the differences *dᵢ* of the pairs. Under "no difference", each
  *dᵢ* is as likely to be +*dᵢ* as −*dᵢ*: so flip the sign of each at random
  and take the mean.
- **A correlation.** To test whether a world's first half predicts its
  second, keep the first-half values in place and shuffle the second-half
  values among the worlds; recompute the [correlation](correlation.md). Here
  the test is one-sided: how often is the shuffled *r* at least as large as
  the real one.

## Reading a p-value

A *p*-value is not the probability that the difference is real. It is how
surprising the data would be if it were not. The notebook reports it, together
with the size of the difference and its [bootstrap](bootstrap.md) interval,
and calls a difference **indicative** rather than an effect when fewer than
thirty seeds per condition went into it.

## In the code

`compare_conditions` and `wandering` in `gol_analysis.py`.

<!-- turns -->
---

← [The bootstrap](bootstrap.md) · [Methods and statistics](../README.md#methods-and-statistics) · [How many seeds an experiment needs](seeds-needed.md) →

> [!info]- Pages that link here (7)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Do the brains matter?](do-the-brains-matter.md) · [How worlds are measured](how-worlds-are-measured.md) · [Correlation](correlation.md) · [The bootstrap](bootstrap.md)
<!-- /turns -->
