# Logarithmic axes and power laws

Many quantities in Graph of Life range over several orders of magnitude: an
[agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") may hold 1 [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") or 3,000; a life may last 1 [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") or 1,500.
On an ordinary axis, everything below a few hundred is squeezed into a
sliver. A **logarithmic axis** gives every factor of ten the same room.

## How to read one

On a logarithmic axis, the distance between two values is the difference of
their logarithms. So 1 to 10, 10 to 100 and 100 to 1,000 are equally long;
and so are 1 to 2, 10 to 20 and 500 to 1,000. **Equal distances are equal
ratios.** The ticks are usually at the powers of ten.

Two consequences:

- A quantity that grows by the same **factor** every step — doubling every
  three iterations, say — is a **straight line** against time on a
  logarithmic value axis ([The first hundred iterations](the-first-hundred-iterations.md)).
- Zero and negative numbers cannot be drawn: log 0 does not exist. Figures
  with log axes leave such values out, and say so.

## Log–log plots and power laws

When **both** axes are logarithmic, a **[power law](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones.")**

$$
y = a\, x^{b}
$$

becomes a straight line, because taking logarithms gives

$$
\log y = \log a + b \log x :
$$

a line with **slope** *b*. So a power law is recognised by a straight line on
log–log axes, and its exponent read off as the slope. Examples:

- *b* = 1: *y* grows in proportion to *x* (twice the tokens, twice the
  agents) — the question of [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md).
- *b* = −1: *y* halves when *x* doubles.
- A curve that **bends** downward on log–log axes is not a power law: it falls
  faster and faster, as an exponential tail does.

## Distributions on log–log axes

To see the shape of a wide distribution — of [degrees](degree.md "The number of an agent's connections: how many neighbours it has."), tokens or [lifetimes](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.") —
the notebook draws its **complementary cumulative distribution**: the share of
values at least *x*, against *x*, both axes logarithmic. If the share of
values equal to *x* follows a power law with exponent −γ, the share at least
*x* follows one with exponent −(γ − 1), and is a straight line. A survival
curve ([Kaplan–Meier](kaplan-meier.md)) is the same kind of picture for
lifetimes.

A caution, from Clauset, Shalizi and Newman (2009): a stretch of a curve that
looks straight on log–log axes is weak evidence of a power law. Many
distributions look straight over a decade or two. The notebook calls something a
power law only when it holds over a wide range and a bent alternative fits
no better.

<!-- turns -->
---

← [The Wilson interval](wilson-interval.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Muller plots](muller-plot.md) →

> [!info]- Pages that link here (11)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [Lightning in every world](lightning-in-every-world.md) · [What shape does the network take?](what-shape-does-the-network-take.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [Degree](degree.md) · [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) · [How worlds are measured](how-worlds-are-measured.md) · [Survival curves (Kaplan–Meier)](kaplan-meier.md) · [Fitting a straight line, and R²](least-squares.md) · [Fitting a power law properly](power-law-fit.md)
<!-- /turns -->
