# Autocorrelation: how long a world remembers

A world wanders: its size, its inequality, its births go up and down over
hundreds of iterations. How long does it **remember** where it was? If a
world is crowded now, is it still crowded 100 iterations later? 500? The
**autocorrelation** answers: the [correlation](correlation.md) of a world
with itself, some time later.

## How the book measures it

1. **Levels.** Cut each world's life from iteration 100 to 2,999 into 29
   stretches of 100 iterations, and take the mean of the statistic in each:
   29 levels per world, *x*₁, …, *x*₂₉.
2. **Centre.** Subtract the world's own average of its 29 levels from each,
   so that what remains is how far the world is above or below *its own*
   typical level.
3. **Pairs.** For a lag ℓ (in stretches), collect, over all worlds, every
   pair (*x*ⱼ, *x*ⱼ₊ℓ) of centred levels ℓ stretches apart.
4. **Correlate.** The correlation of those pairs is the autocorrelation at
   a distance of 100·ℓ iterations.

Step 2 matters: without it, a world that is simply bigger than the others
throughout would look like a world with a long memory.

## How to read it

- Close to 1: a world is still where it was.
- Close to 0: where a world was says nothing about where it will be.
- In the baseline: 0.68 for the number of agents after 100 iterations, about
  half that after 200, little after 300, nothing after 500
  ([Chapter 12](../chapters/12-how-much-does-the-seed-decide.md)).

## Why it dips below zero

At lags of 500 iterations and more, the measured autocorrelation is slightly
**negative**. That is mostly not a real anti-memory but a known bias of
step 2. Each world was measured against its own average of only *m* = 29
levels; a world above its own average in some stretches must be below it in
others, since its centred levels add up to zero. This pulls every measured
autocorrelation down by about

$$
\frac{S}{m}, \qquad S = 1 + 2\sum_{\ell \ge 1} \rho_\ell ,
$$

where ρℓ are the true autocorrelations — a classic result (Marriott and Pope
1954; Kendall 1954). With the baseline's
values for the number of agents, *S* ≈ 1 + 2·(0.68 + 0.34 + 0.17 + 0.04) ≈
3.5, so the pull is about 3.5/29 ≈ 0.12 — about the dip seen.

## What it means for measuring

Levels 500 iterations apart are nearly independent. A mean over *k*
correlated levels is about as precise as a mean over *k*/*S* independent
ones. A world's settled life, iterations 500 to 2,999, is 25 stretches of
100; with *S* ≈ 3.5 that is worth about seven independent looks — against
one look for a single stretch at the end. That is why the book measures
worlds over their settled life ([The settled life of a world](settled-life.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
