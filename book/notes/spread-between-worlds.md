# Spread between worlds

How different are worlds that differ only in their seed? The book answers
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
what lets the book compare the spread of the number of agents with the spread
of the Gini coefficient.

## Example

Five worlds settle at 1,000, 1,200, 1,300, 1,400 and 1,600 agents. The mean
is 1,300; the deviations are −300, −100, 0, 100, 300; their squares add to
200,000; divided by 4 that is 50,000; *s* = √50,000 ≈ 224. So *c* = 224/1,300
≈ 0.17.

## A picture of it

The book's "dot columns" show the same thing without summing it up: each
world is a dot at its level divided by the mean of all worlds, so 1 is the
average world, and a column's height shows the spread
([Chapter 12](../chapters/12-how-much-does-the-seed-decide.md)).

## What it is used for

- To say how much the seed decides ([Chapter 12](../chapters/12-how-much-does-the-seed-decide.md)).
- To work out how many seeds an experiment needs: the number grows with *c*²
  ([How many seeds an experiment needs](seeds-needed.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
