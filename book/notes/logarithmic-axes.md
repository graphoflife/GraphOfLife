# Logarithmic axes and power laws

Many quantities in Graph of Life range over several orders of magnitude: an
agent may hold 1 token or 3,000; a life may last 1 iteration or 1,500.
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
  logarithmic value axis ([Chapter 11](../chapters/11-the-first-hundred-iterations.md)).
- Zero and negative numbers cannot be drawn: log 0 does not exist. Figures
  with log axes leave such values out, and say so.

## Log–log plots and power laws

When **both** axes are logarithmic, a **power law**

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
  agents) — the question of [Chapter 38](../chapters/38-how-does-a-worlds-size-follow-its-tokens.md).
- *b* = −1: *y* halves when *x* doubles.
- A curve that **bends** downward on log–log axes is not a power law: it falls
  faster and faster, as an exponential tail does.

## Distributions on log–log axes

To see the shape of a wide distribution — of degrees, tokens or lifetimes —
the book draws its **complementary cumulative distribution**: the share of
values at least *x*, against *x*, both axes logarithmic. If the share of
values equal to *x* follows a power law with exponent −γ, the share at least
*x* follows one with exponent −(γ − 1), and is a straight line. A survival
curve ([Kaplan–Meier](kaplan-meier.md)) is the same kind of picture for
lifetimes.

A caution, from Clauset, Shalizi and Newman (2009): a stretch of a curve that
looks straight on log–log axes is weak evidence of a power law. Many
distributions look straight over a decade or two. The book calls something a
power law only when it holds over a wide range and a bent alternative fits
no better.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
