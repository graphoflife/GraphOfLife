# Spread between worlds

How different are worlds that differ only in their [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.")? The notebook answers
with one number per statistic: the **coefficient of variation** of the
worlds' levels, which it calls the **spread** *c*.

## Definition

Measure each of *N* worlds by its level *x*₁, …, *x*_N
([The settled life of a world](settled-life.md)). Their mean and their
standard deviation are

$$
\bar x = \frac{1}{N}\sum_{i=1}^{N} x_i ,
\qquad
s = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N} (x_i - \bar x)^2 } .
$$

(Dividing by *N* − 1 rather than *N* makes *s*² an unbiased estimate of the
variance of the population the worlds are drawn from.) The spread is

$$
c = \frac{s}{\bar x} .
$$

It has no unit: a spread of 0.17 says the worlds typically lie about 17% of
the mean away from the mean, whatever the statistic is measured in. That is
what lets the notebook compare the spread of the number of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") with the spread
of the [Gini coefficient](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.").

## Example

Five worlds settle at 1,000, 1,200, 1,300, 1,400 and 1,600 agents. The mean
is 1,300; the deviations are −300, −100, 0, 100, 300; their squares add to
200,000; divided by 4 that is 50,000; *s* = √50,000 ≈ 224. So *c* = 224/1,300
≈ 0.17.

## A picture of it

The notebook's "dot columns" show the same thing without summing it up: each
world is a dot at its level divided by the mean of all worlds, so 1 is the
average world, and a column's height shows the spread
([How much does the seed decide?](how-much-does-the-seed-decide.md)).

## What it is used for

- To say how much the seed decides ([How much does the seed decide?](how-much-does-the-seed-decide.md)).
- To work out how many seeds an experiment needs: the number grows with *c*²
  ([How many seeds an experiment needs](seeds-needed.md)).

<!-- turns -->
---

← [The settled life of a world](settled-life.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Correlation](correlation.md) →

> [!info]- Pages that link here (5)
> [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Do the brains matter?](do-the-brains-matter.md) · [Every world side by side](every-world-side-by-side.md) · [How worlds are measured](how-worlds-are-measured.md) · [How many seeds an experiment needs](seeds-needed.md)
<!-- /turns -->
