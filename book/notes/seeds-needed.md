# How many seeds an experiment needs

An experiment changes one setting and asks whether the worlds change. Worlds
differ by chance anyway, so a small change can hide in the noise. How many
worlds — seeds — does each condition need for a change of a given size to be
**found**, most of the time?

## The formula

Suppose each condition runs *n* worlds, the worlds of a condition vary with a
[spread](spread-between-worlds.md) *c* (standard deviation divided by mean),
and the setting changes the mean by a share Δ (Δ = 0.1 for a change of 10%).
To find such a change with a two-sided test at the 5% level, four times in
five, each condition needs about

$$
n = \frac{2\,(z_{1} + z_{2})^2\, c^2}{\Delta^2}
$$

worlds, rounded up, where *z*₁ = 1.960 and *z*₂ = 0.842 are the points of the
standard normal distribution beyond which lie 2.5% and 20% of it.

## Where it comes from

The difference of two means of *n* worlds each, with standard deviation *s*
in both, has standard error *s*√(2/*n*). A two-sided test at the 5% level
calls a difference real when it exceeds *z*₁ standard errors. For a true
difference *D* to exceed that four times in five, *D* must lie *z*₂ standard
errors further out:

$$
D = (z_1 + z_2)\, s \sqrt{2/n}
\quad\Longleftrightarrow\quad
n = \frac{2\,(z_1 + z_2)^2\, s^2}{D^2} .
$$

With *s* = *c* · mean and *D* = Δ · mean, the means cancel and the formula
above follows. It assumes the means are roughly normal, which with tens of
worlds they are.

## Example

The number of agents in a settled baseline world has *c* = 0.17. To see a
change of 10%:

$$
n = \frac{2 \times (1.960 + 0.842)^2 \times 0.17^2}{0.1^2}
  = \frac{2 \times 7.851 \times 0.0289}{0.01} \approx 45.4 ,
$$

so 46 seeds per condition. Turned around, with 30 seeds the smallest change
found four times in five is Δ = (*z*₁ + *z*₂)·*c*·√(2/30) ≈ 0.12: 12%.

## Two lessons

- *n* grows with the **square** of the spread: halving *c* — by measuring
  worlds over a longer stretch ([The settled life of a world](settled-life.md)) —
  quarters the seeds needed.
- *n* grows with one over the **square** of the change: a change half as
  big needs four times the seeds.

## In the code

`seeds_needed` in `gol_analysis.py`;
[Chapter 13](../chapters/13-how-much-does-the-seed-decide.md) draws *n* against Δ.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
