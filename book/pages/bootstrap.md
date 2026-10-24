# How sure can we be? The bootstrap, permutation tests, the Wilson interval and how many seeds

> [!question] Questions of this page
> - Thirty worlds give a mean. How far might another thirty give a different
>   one?
> - Two conditions differ. Could chance alone have made the difference?
> - Four worlds in thirty died. What range does the true share lie in?
> - How many [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") does an experiment need to find a change of a given size?

## In short

Every number the notebook reports comes from a few dozen worlds, and another
few dozen would give a somewhat different number. Four tools say how much
that matters. The **bootstrap** redraws the worlds we have, many times, to
show how much a mean could vary. A **permutation test** shuffles which world
belongs to which condition, to show how often chance alone gives a difference
as large as the real one. The **Wilson interval** gives the range a share
probably lies in when it was counted in few worlds. And a simple formula says
how many seeds an experiment needs to find a change of a given size.

## The bootstrap

Thirty worlds give a mean. Another thirty would give a somewhat different
one. How uncertain is the mean? The **bootstrap** (Efron 1979) answers
without any formula about distributions, by re-sampling the data we have.

### The idea of the bootstrap

The thirty values we measured are our best picture of the population they
come from. So draw new "samples of thirty" **from them**: thirty values
picked at random **with replacement**. The same world may be picked twice,
and another not at all. Compute the mean of each such sample. The spread of
those means shows how much the mean would vary from one set of thirty worlds
to another.

### Exactly how

Take the values *x*₁, …, *x*ₙ.

1. Repeat *B* = 10,000 times: draw *n* indices uniformly at random from
   1, …, *n*, with replacement. Compute the mean of the values at them.
2. The **95% interval** for the mean runs from the 2.5th to the 97.5th
   percentile of the 10,000 means
   ([Median and quantiles](median-and-quantiles.md#quantiles)).

The random draws are seeded, so the interval comes out the same every time.

**Example.** Take the values 2, 4 and 9. One resample might be (4, 4, 9),
with mean 5.67; another (2, 9, 2), with mean 4.33. Over 10,000 resamples, the
means range from 2 (all three picks are the 2) to 9. The middle 95% of them
form the interval.

### For a difference

To compare two conditions, the notebook bootstraps the **difference** of
their means. If the runs are paired by seed, it resamples the paired
differences *dᵢ* = *yᵢ* − *xᵢ*. Otherwise it resamples each condition
separately and subtracts. An interval that does not contain 0 means that the
data are hard to reconcile with "no difference". That is then tested directly
with a [permutation test](#permutation-tests).

In the code, this is `bootstrap` and `compare_conditions` in
`gol_analysis.py`.

## Permutation tests

A difference between two conditions, or a correlation between two
measurements, could be real, or it could be chance. A **permutation test**
asks: if there were no real difference, how often would chance alone produce
one at least this large? It answers by **shuffling**, with no assumption
about distributions.

### The logic

Suppose a [setting](settings.md "One number or switch of a run's rules, fixed before it starts.") changes nothing. Then the label "condition A" or
"condition B" on each world is arbitrary. Any relabelling of the worlds is as
likely as the one we have. So relabel at random many times, and recompute
the difference each time. Then see where the real difference falls among the
shuffled ones. If it is larger than almost all of them, "no difference" is
hard to believe.

The share of shuffles that give a difference at least as large (in size) as
the real one is the **p-value**. The notebook computes it as

$$
p = \frac{1 + \#\{\text{shuffles at least as extreme}\}}{1 + \text{number of shuffles}} ,
$$

with 10,000 shuffles. In words: count the shuffles at least as extreme as the
real labelling, and the real labelling itself, and divide by all labellings
tried. The 1s count the real labelling as one of the possibilities, so *p* is
never exactly 0.

### Three versions in this notebook

- **Two independent groups.** Pool the values of both conditions. Deal them
  at random into two groups of the original sizes, and take the difference of
  the means.
- **Pairs.** Each world of condition B may have a partner in A with the same
  seed. Then take the differences *dᵢ* of the pairs. Under "no difference",
  each *dᵢ* is as likely to be +*dᵢ* as −*dᵢ*. So flip the sign of each at
  random, and take the mean.
- **A correlation.** Does a world's first half predict its second? Keep the
  first-half values in place, and shuffle the second-half values among the
  worlds; then recompute the [correlation](correlation.md). Here the test is
  one-sided: how often is the shuffled *r* at least as large as the real one?

### Reading a p-value

A *p*-value is not the probability that the difference is real. It is how
surprising the data would be if it were not. The notebook reports it together
with the size of the difference and its [bootstrap](#the-bootstrap) interval.
And it calls a difference **indicative**, rather than an effect, when fewer
than thirty seeds per condition went into it.

In the code, this is `compare_conditions` and `wandering` in
`gol_analysis.py`.

## The Wilson interval

Four of the thirty [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds of 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") died out: 13%. With only
thirty worlds, the true rate could easily be lower or higher. The **Wilson
interval** (Wilson 1927) gives a range for a share estimated from a count.

### The formula of the Wilson interval

Suppose *k* of *n* trials are "successes". The observed share is
*p̂* = *k*/*n*. With *z* = 1.960 for 95%, the Wilson interval is

$$
\frac{\hat p + \dfrac{z^2}{2n} \;\pm\; z \sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}} .
$$

In words: the interval is centred a little nearer to one half than the
observed share. And its width allows for how few trials there were.

**Example.** Take *k* = 4 and *n* = 30. Then *p̂* = 0.1333, and
*z*²/*n* = 0.1280.

- The centre is (0.1333 + 0.0640) / 1.1280 = 0.1749.
- The half-width is 1.960 · √(0.1333 · 0.8667/30 + 3.8415/3,600) / 1.1280
  = 1.960 · √(0.003852 + 0.001067) / 1.1280 = 1.960 · 0.07013 / 1.1280
  = 0.1219.

The interval is **5.3% to 29.7%**. Thirty worlds say only that between about
one world in twenty and one in three dies.

### Why not the simple ± formula?

The textbook interval *p̂* ± *z*√(*p̂*(1 − *p̂*)/*n*) gives 1.2% to 25.5%
here. For small counts it is badly wrong: with *k* = 0 it gives the
"interval" 0 to 0. The Wilson interval stays inside 0 to 1 and is never
empty. And it covers the true share about as often as it promises.

## How many seeds an experiment needs

An experiment changes one setting and asks whether the worlds change. Worlds
differ by chance anyway, so a small change can hide in the noise. How many
worlds, that is seeds, does each condition need for a change of a given size
to be **found**, most of the time?

### The formula for the seeds

Suppose each condition runs *n* worlds. The worlds of a condition vary with a
[spread](median-and-quantiles.md#spread-between-worlds) *c*: their standard
deviation divided by their mean. And the setting changes the mean by a share
Δ (Δ = 0.1 for a change of 10%). To find such a change with a two-sided test
at the 5% level, four times in five, each condition needs about

$$
n = \frac{2\,(z_{1} + z_{2})^2\, c^2}{\Delta^2}
$$

worlds, rounded up. Here *z*₁ = 1.960 and *z*₂ = 0.842 are the points of the
standard normal distribution beyond which lie 2.5% and 20% of it.

### Where it comes from

Take the difference of two means of *n* worlds each, with standard deviation
*s* in both. Its standard error is *s*√(2/*n*). A two-sided test at the 5%
level calls a difference real when it exceeds *z*₁ standard errors. For a
true difference *D* to exceed that four times in five, *D* must lie *z*₂
standard errors further out:

$$
D = (z_1 + z_2)\, s \sqrt{2/n}
\quad\Longleftrightarrow\quad
n = \frac{2\,(z_1 + z_2)^2\, s^2}{D^2} .
$$

With *s* = *c* · mean and *D* = Δ · mean, the means cancel, and the formula
above follows. It assumes that the means are roughly normal, which with tens
of worlds they are.

### An example of the seeds needed

Take the number of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") in a settled baseline world of 10,000 tokens,
measured over the last fifth of its run. It has *c* = 0.17. To see a change
of 10%:

$$
n = \frac{2 \times (1.960 + 0.842)^2 \times 0.17^2}{0.1^2}
  = \frac{2 \times 7.851 \times 0.0289}{0.01} \approx 45.4 ,
$$

so 46 seeds per condition. Turned around: with 30 seeds, the smallest change
found four times in five is Δ = (*z*₁ + *z*₂)·*c*·√(2/30) ≈ 0.12, that is
12%. In the worlds of 100,000 tokens, measured over their settled life, *c*
is 0.11, and 19 seeds are enough
([How much does the seed decide?](how-much-does-the-seed-decide.md)).

### Two lessons

- *n* grows with the **square** of the spread. Halving *c*, by measuring
  worlds over a longer stretch
  ([The settled life of a world](median-and-quantiles.md#the-settled-life-of-a-world)),
  quarters the seeds needed.
- *n* grows with one over the **square** of the change. A change half as big
  needs four times the seeds.

In the code, this is `seeds_needed` in `gol_analysis.py`.
[How much does the seed decide?](how-much-does-the-seed-decide.md) draws *n*
against Δ.

## Where to go next

- [Many worlds at once](median-and-quantiles.md): the median, the bands and
  the spread these tools build on.
- [How much does the seed decide?](how-much-does-the-seed-decide.md): how far
  thirty worlds that differ only in their seed lie apart.
- [Correlation](correlation.md): the correlation a permutation test can test.
- [When a world ends](extinction.md): the deaths of whole worlds, counted in
  shares like the one above.

## How our view changed

- **24 October 2026.** The four notes became one page. The note on how many
  seeds an experiment needs had said that *c* is about 0.10 in the worlds of
  100,000 tokens. It also said that 19 seeds are enough. A spread of 0.10 would give
  16 seeds. The spread
  [How much does the seed decide?](how-much-does-the-seed-decide.md) measures
  is 0.11, and that gives 19.

## References

- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *The Annals of Statistics* 7, 1–26.
- Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. *Journal of the American Statistical Association*
  22, 209–212.

<!-- turns -->
---

← [How long a world remembers: autocorrelation and the power spectrum](autocorrelation.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Muller plots](muller-plot.md) →

> [!info]- Pages that link here (14)
> [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [Seeds and the random stream](random-numbers.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Do the brains matter?](do-the-brains-matter.md) · [What would count as open-ended evolution](open-ended-test.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [How worlds are measured](how-worlds-are-measured.md) · [Pilot worlds](pilot-worlds.md) · [Many worlds at once: the median, bands, the settled life and the spread](median-and-quantiles.md) · [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md)
<!-- /turns -->
