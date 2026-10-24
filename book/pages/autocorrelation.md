# How long a world remembers: autocorrelation and the power spectrum

> [!question] Questions of this page
> - If a world is crowded now, is it still crowded 100 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") later?
>   500?
> - Why does a measured memory dip below zero, and what does that mean?
> - Is a world's wander a slow drift or a quick jitter, and how much of it
>   happens at each pace?

## In short

A world wanders: its size, its inequality and its births go up and down over
hundreds of iterations. The **autocorrelation** says how long it remembers
where it was: the correlation of a world with itself, some time later. In
[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") it falls from about 0.6 after 100 iterations to nothing after 300, so
levels 500 iterations apart are nearly independent. The **power spectrum**
splits the wander into rhythms and says how much of it happens at each. A
world's number of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") moves like a random walk on short scales, and is
pulled back towards its level on long ones.

## Autocorrelation

How long does a world **remember** where it was? If a world is crowded now,
is it still crowded 100 iterations later? 500? The **autocorrelation**
answers: the [correlation](correlation.md) of a world with itself, some time
later.

### How the notebook measures it

1. **Levels.** Cut each world's life, from iteration 100 on, into stretches
   of 100 iterations, and take the mean of the statistic in each. A world of
   10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), run to iteration 2,999, gives 29 levels, *x*₁, …, *x*₂₉. A
   world of 100,000 tokens, run to 1,499, gives 14.
2. **Centre.** Subtract the world's own average of its levels from each. What
   remains is how far the world is above or below *its own* typical level.
3. **Pairs.** For a lag of ℓ stretches, collect, over all worlds, every pair
   (*x*ⱼ, *x*ⱼ₊ℓ) of centred levels ℓ stretches apart.
4. **Correlate.** The correlation of those pairs is the autocorrelation at a
   distance of 100·ℓ iterations.

The second step matters. Without it, a world that is simply bigger than the
others throughout would look like a world with a long memory.

### How to read it

- Close to 1: a world is still where it was.
- Close to 0: where a world was says nothing about where it will be.
- In the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), for the number of agents: 0.61 after 100 iterations in the
  worlds of 100,000 tokens (0.68 in those of 10,000). About a third of that
  is left after 200, and nothing after 300
  ([How much does the seed decide?](how-much-does-the-seed-decide.md)).

### Why it dips below zero

At lags of 500 iterations and more, the measured autocorrelation is
**negative**. That is mostly not a real anti-memory, but a known bias of the
second step. Each world was measured against its own average of only *m*
levels. Its centred levels add up to zero, so a world above its own average
in some stretches must be below it in others. This pulls every measured
autocorrelation down by about

$$
\frac{S}{m}, \qquad S = 1 + 2\sum_{\ell \ge 1} \rho_\ell ,
$$

where the ρℓ are the true autocorrelations. It is a classic result (Marriott
and Pope 1954; Kendall 1954).

**In the worlds of 10,000 tokens**, with *m* = 29, take the baseline's values
for the number of agents. Then *S* ≈ 1 + 2·(0.68 + 0.34 + 0.17 + 0.04) ≈ 3.5,
so the pull is about 3.5/29 ≈ 0.12. That is about the dip seen there, −0.1
to −0.17.

**In the worlds of 100,000 tokens**, with only *m* = 14, the dip is far
deeper: about −0.5 at 700 iterations. The formula assumes that the memory
dies within a few stretches. A level that also drifts slowly over the 1,400
iterations is pushed further by the centring.
[How much does the seed decide?](how-much-does-the-seed-decide.md#a-world-forgets)
brackets it with two made-up levels measured the same way. A random walk
falls to −0.66 at 900 iterations, and a level with no memory at all sits at
−0.08.

### What it means for measuring

Levels 500 iterations apart are nearly independent. A mean over *k*
correlated levels is about as precise as a mean over *k*/*S* independent
ones. A world's settled life, iterations 500 to 2,999, is 25 stretches of
100. With *S* ≈ 3.5 that is worth about seven independent looks, against one
look for a single stretch at the end. That is why the notebook measures
worlds over their settled life
([The settled life of a world](median-and-quantiles.md#the-settled-life-of-a-world)).

## The power spectrum of a time series

A world's number of agents goes up and down. Is that a slow drift over
hundreds of iterations, or a quick jitter from one iteration to the next? The
**power spectrum** splits the motion into rhythms, and says how much of it
happens at each.

### Waves

Take any series of *N* values *x*₀ … *x*_{N−1}. It can be written exactly as
a sum of sine waves of the frequencies *f* = *k*/*N*, for
*k* = 0, 1, …, *N*/2. The wave of frequency *k*/*N* makes *k* full cycles
over the series: it has a **period** of *N*/*k* steps. The discrete Fourier
transform

$$
X_k = \sum_{t=0}^{N-1} x_t\, e^{-2\pi i k t / N}
$$

gives each wave's size and timing, and |*X*ₖ|² is its **power**. The powers
add up, apart from a constant factor, to the series' variance. So the
spectrum says how the variance is shared among the rhythms.

### Three preparations

1. **Remove the trend.** A slow overall rise is not a rhythm, but it would
   put much power at the lowest frequencies. So the notebook first subtracts
   the [least-squares](correlation.md#fitting-a-straight-line) straight line.
2. **Taper the ends.** The transform treats the series as if it repeated. The
   jump from its last value back to its first would spread false power over
   all frequencies, which is called **leakage**. Multiplying by a **Hann
   window**, ½(1 − cos 2π*t*/(*N*−1)), which rises from 0 to 1 and back,
   removes the jump.
3. **Normalise.** Dividing by the total power makes worlds of different sizes
   comparable before they are averaged.

### Reading the slope

On log–log axes, many natural series give a straight line: power ∝ *f*^β.

| β | name | what it means |
|---|---|---|
| 0 | white noise | every value independent of the last; all rhythms equally strong |
| −1 | pink or 1/*f* noise | memory on every time scale at once |
| −2 | brown noise, a random walk | each value the last plus an independent step |

Picture a random walk held near a level by a restoring force: the
**Ornstein–Uhlenbeck process**. Its spectrum is

$$
S(f) \propto \frac{1}{1 + (2\pi f \tau)^2} .
$$

Here τ is the time the force needs to pull a deviation back by a factor *e*.
In words: for periods shorter than 2πτ the spectrum falls like a random
walk's (β = −2), and for longer ones it is flat. Its autocorrelation after a
time *s* is *e*^(−*s*/τ) ([Autocorrelation](#autocorrelation)).

### The spectrum in the baseline

The number of agents of the baseline worlds of 10,000 tokens has β ≈ −2.1
for periods between 4 and 500 iterations. Its spectrum levels off for periods
of 800 iterations and more. In the worlds of 100,000 tokens, β ≈ −1.8
([Power laws, real and apparent](power-laws-real-and-apparent.md)). So a
world moves like a random walk on short scales, and is pulled back on long
ones. Its τ is a few hundred iterations. In the small worlds, 70% of the
variance lies at periods of 500 iterations or more, and 0.4% at periods
shorter than 10.

## Where to go next

- [How much does the seed decide?](how-much-does-the-seed-decide.md): how
  long thirty worlds remember their levels, statistic by statistic.
- [Many worlds at once](median-and-quantiles.md): the settled life that this
  memory argues for.
- [Power laws, real and apparent](power-laws-real-and-apparent.md): the
  spectrum of a world's wander, and other slopes on log–log axes.
- [Two numbers together](correlation.md): the correlation this page takes of
  a world with itself.

## How our view changed

- **24 October 2026.** The worlds of 100,000 tokens were added to this page
  on 11 October. Since then it said that the measured autocorrelation dips
  *slightly* below zero, by about *S*/*m* ≈ 0.12. That holds for the worlds of
  10,000 tokens, with 29 stretches each. In the worlds of 100,000 tokens, with
  14, the dip reaches about −0.5 at 700 iterations
  ([How much does the seed decide?](how-much-does-the-seed-decide.md#a-world-forgets)).

## References

- Kendall, M. G. (1954). Note on bias in the estimation of autocorrelation.
  *Biometrika* 41, 403–404.
- Marriott, F. H. C. and Pope, J. A. (1954). Bias in the estimation of
  autocorrelations. *Biometrika* 41, 390–402.

<!-- turns -->
---

← [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Survival curves (Kaplan–Meier)](kaplan-meier.md) →

> [!info]- Pages that link here (8)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [Physical inspiration](physical-inspiration.md) · [How worlds are measured](how-worlds-are-measured.md) · [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md) · [Splitting variation: where, who and chance, and between worlds and within them](variance-split.md)
<!-- /turns -->
