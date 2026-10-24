# Two numbers together: correlation, a fitted line and logarithmic axes

> [!question] Questions of this page
> - Do two quantities go up and down together, and how strongly?
> - Which straight line fits a cloud of points best, and how much of their
>   spread does it explain?
> - How does one read an axis on which every factor of ten gets the same
>   room, and why does a power law become a straight line there?

## In short

Much of the notebook asks how one number goes with another. The
**correlation** *r* says, from −1 to 1, how closely two quantities rise and
fall together along a straight line. The **least-squares line** is the
straight line that misses a cloud of points least. Its **R²**, the square of
*r*, is the share of their spread it accounts for. A **logarithmic axis**
gives every factor of ten the same room, so that numbers of very different
sizes fit on one chart. On two such axes a power law is a straight line,
whose slope is its exponent.

## Correlation

Do two quantities go up and down together? The **correlation coefficient**
*r* answers with a number between −1 and 1.

### The definition of correlation

Take *n* pairs (*x*₁, *y*₁), …, (*x*ₙ, *y*ₙ), with means *x̄* and *ȳ*. The
**Pearson correlation** is

$$
r = \frac{\sum_{i} (x_i - \bar x)(y_i - \bar y)}
         {\sqrt{\sum_{i} (x_i - \bar x)^2}\ \sqrt{\sum_{i} (y_i - \bar y)^2}} .
$$

The numerator adds up, pair by pair, whether *x* and *y* are on the same side
of their means (positive) or on opposite sides (negative). The denominator
scales the sum so that *r* lies between −1 and 1.

- *r* = 1: the points lie exactly on a rising straight line.
- *r* = −1: they lie exactly on a falling straight line.
- *r* = 0: there is no straight-line relation, though there can still be a
  curved one.

*r*² is the share of the variance of *y* that a straight line in *x* accounts
for. So *r* = 0.5 means a quarter
([How much the line explains](#how-much-the-line-explains)).

**Example.** Take three worlds, (*x*, *y*) = (1, 2), (2, 2) and (3, 5). The
means are 2 and 3. The deviations are (−1, −1), (0, −1) and (1, 2), so the
products are 1, 0 and 2, with a sum of 3. The sums of squares are 2 and 6.
So *r* = 3 / (√2 · √6) = 3/√12 ≈ 0.87.

### Is a correlation real?

With few pairs, a sizeable *r* can come about by chance. The notebook checks
this with a [permutation test](bootstrap.md#permutation-tests). It shuffles
the *y* values against the *x* values many times, and sees how often the
shuffled pairs give an *r* at least as large.

In [How much does the seed decide?](how-much-does-the-seed-decide.md), take
26 worlds. Their numbers of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") in the first and the second half of a run
correlate at 0.12. Shuffled pairings reach that 28 times in 100: it is
nothing. For [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") the correlation is 0.41, which shuffled pairings
reach only 2 times in 100.

### Correlation is not cause

[How unequal is a world, and who stays rich](where-do-the-tokens-go.md) finds
that agents with more connections hold more [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). The correlation between
the number of connections and the logarithm of the tokens is *r* = 0.50. That alone cannot say
which way it works. Connections may make an agent rich, riches may bring
connections, or something else may bring both.

## Fitting a straight line

Many figures of this notebook draw a straight line through a cloud of points.
Many statistics of the viewer are the slope of such a line. This section says
which line, and what its R² means.

### The line that misses least

Take *n* points (*x*₁, *y*₁), …, (*x*ₙ, *y*ₙ). The **least-squares line**
*y* = *a* + *b x* is the one that makes the sum of squared vertical misses

$$
\sum_i \bigl(y_i - a - b x_i\bigr)^2
$$

as small as possible. Set the derivatives in *a* and *b* to zero. With the
means *x̄* and *ȳ*, that gives

$$
b = \frac{S_{xy}}{S_{xx}}, \qquad a = \bar y - b \bar x ,
$$

$$
S_{xx} = \sum_i (x_i - \bar x)^2, \quad
S_{yy} = \sum_i (y_i - \bar y)^2, \quad
S_{xy} = \sum_i (x_i - \bar x)(y_i - \bar y) .
$$

In words: the slope is how much *x* and *y* vary together, over how much *x*
varies alone. The line always passes through the point of means
(*x̄*, *ȳ*).

### How much the line explains

How much better is the line than no line at all? Without a line, the best
single guess for every *y* is *ȳ*, with squared misses *S*_yy. With the line,
the squared misses shrink to *S*_yy − *S*_xy²/*S*_xx. The share removed is

$$
R^2 = \frac{S_{xy}^2}{S_{xx}\, S_{yy}} ,
$$

which is the square of the [correlation](#correlation) *r*. R² = 1 means that
every point lies on the line. R² = 0 means that the line explains nothing.

**Example.** Take the points (1, 2), (2, 3) and (3, 7). The means are 2 and
4. The deviations of *x* are −1, 0 and 1; those of *y* are −2, −1 and 3. So
*S*_xx = 2, *S*_yy = 14 and *S*_xy = 2 + 0 + 3 = 5. That gives *b* = 2.5,
*a* = 4 − 5 = −1, and R² = 25/28 ≈ 0.89.

### Lines through logarithms

A power law *y* = *A x*^*b* becomes a straight line in the logarithms. It
reads ln *y* = ln *A* + *b* ln *x*
([Logarithmic axes and power laws](#logarithmic-axes-and-power-laws)). So the
notebook and the viewer fit power laws by least squares on (ln *x*, ln *y*).
The program (`_power_fit` in `gol_series.py`) keeps only the points with both
values positive. A zero cannot be logged, and it is a real observation, not a
small one. The program wants at least 8 points.

Two things to keep in mind:

- **R² is about scatter.** A curve that bends gently has a high R² over a
  short range; a true law with noisy points has a low one. R² says how much of
  the spread a line accounts for, not whether a law holds.
- **The line is pulled by where the points are.** Every point counts the
  same, so a crowd of points at small *x* decides the slope more than a few at
  large *x*. For distributions, this makes least squares a poor way to find an
  exponent. [Fitting a power law](power-law-fit.md) says what to do instead.

## Logarithmic axes and power laws

Many quantities in Graph of Life range over several orders of magnitude. An
agent may hold 1 token or 3,000; a life may last 1 [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") or 1,500. On an
ordinary axis, everything below a few hundred is squeezed into a sliver. A
**logarithmic axis** gives every factor of ten the same room.

### How to read a logarithmic axis

On a logarithmic axis, the distance between two values is the difference of
their logarithms. So 1 to 10, 10 to 100 and 100 to 1,000 are equally long.
And so are 1 to 2, 10 to 20 and 500 to 1,000. **Equal distances are equal
ratios.** The ticks are usually at the powers of ten.

Two things follow:

- Take a quantity that grows by the same **factor** every step, doubling
  every three iterations say. Against time, on a logarithmic value axis, it is
  a **straight line**
  ([The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations)).
- Zero and negative numbers cannot be drawn: log 0 does not exist. Figures
  with logarithmic axes leave such values out, and say so.

### Log–log plots and power laws

When **both** axes are logarithmic, a **power law**

$$
y = a\, x^{b}
$$

becomes a straight line, because taking logarithms gives

$$
\log y = \log a + b \log x :
$$

a line with the **slope** *b*. So a power law is recognised by a straight
line on log–log axes, and its exponent is read off as the slope. Examples:

- *b* = 1: *y* grows in proportion to *x*: twice the tokens, twice the
  agents. That is the question of
  [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md).
- *b* = −1: *y* halves when *x* doubles.
- A curve that **bends** downward on log–log axes is not a power law. It falls
  faster and faster, as an exponential tail does.

### Distributions on log–log axes

To see the shape of a wide distribution, of [degrees](degree.md "The number of an agent's connections: how many neighbours it has."), tokens or [lifetimes](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death."), the
notebook draws its **complementary cumulative distribution**. That is the
share of values at least *x*, against *x*, with both axes logarithmic.
Suppose the share of values equal to *x* follows a power law with the
exponent −γ. Then the share at least *x* follows one with the exponent
−(γ − 1), and is a straight line. A survival curve
([Kaplan–Meier](kaplan-meier.md)) is the same kind of picture for lifetimes.

Clauset, Shalizi and Newman (2009) give a caution. A stretch of a curve that
looks straight on log–log axes is weak evidence of a power law. Many
distributions look straight over a decade or two. The notebook calls
something a power law only when it holds over a wide range, and a bent
alternative fits no better.

## Where to go next

- [Fitting a power law](power-law-fit.md): how to find an exponent from a
  distribution, and where its tail starts.
- [How sure can we be?](bootstrap.md): the permutation test that asks whether
  a correlation is real.
- [How properties scale together](how-properties-scale-together.md): four
  power laws fitted to every agent of a world.
- [Autocorrelation](autocorrelation.md): the correlation of a world with
  itself, some time later.

## References

- Clauset, A., Shalizi, C. R. and Newman, M. E. J. (2009). Power-law
  distributions in empirical data. *SIAM Review* 51, 661–703.

<!-- turns -->
---

← [Many worlds at once: the median, bands, the settled life and the spread](median-and-quantiles.md) · [Methods and statistics](../README.md#methods-and-statistics) · [How long a world remembers: autocorrelation and the power spectrum](autocorrelation.md) →

> [!info]- Pages that link here (24)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Measuring inequality: the Gini coefficient, the richest tenth, entropy and mobility](gini-coefficient.md) · [The even split](even-split.md) · [Where tokens flow](where-the-tokens-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [How properties scale together](how-properties-scale-together.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [Three ways to measure a network's dimension](ball-dimension-and-curvature.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [Do the brains matter?](do-the-brains-matter.md) · [Eras, contests and invasions: does a world keep getting better?](eras-and-contests.md) · [How worlds are measured](how-worlds-are-measured.md) · [How long a world remembers: autocorrelation and the power spectrum](autocorrelation.md) · [Between worlds and within them](between-and-within.md) · [Survival curves (Kaplan–Meier)](kaplan-meier.md) · [How sure can we be? The bootstrap, permutation tests, the Wilson interval and how many seeds](bootstrap.md) · [Fitting a power law properly](power-law-fit.md)
<!-- /turns -->
