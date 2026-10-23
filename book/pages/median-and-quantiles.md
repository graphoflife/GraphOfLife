# Median and quantiles

The book describes sets of numbers — thirty worlds' sizes, a million
lifetimes — by a few **quantiles** rather than by their mean alone. This note
says what they are and exactly how they are computed.

## The median

Sort the numbers. The **median** is the one in the middle: half of the
numbers are at or below it, half at or above. With an even count it is the
average of the two middle ones. The median of 3, 9, 1, 4 and 100 is 4 — the
mean is 23.4, pulled up by the single 100. That is why the book prefers the
median for "the typical world": one extreme world cannot move it far.

## Quantiles

The **quantile** *Q*(*q*), for a share *q* between 0 and 1, is the value
below which a share *q* of the numbers lie. *Q*(0.5) is the median;
*Q*(0.25) and *Q*(0.75) are the **quartiles**, a quarter and three quarters
of the way up; *Q*(0.05) and *Q*(0.95) cut off the lowest and highest
twentieth. The 25th **percentile** is the same thing as *Q*(0.25).

## Exactly how

For *n* numbers sorted from smallest, *x*₀ ≤ *x*₁ ≤ … ≤ *x*ₙ₋₁ (counted from
0), the quantile is read at the position *h* = (*n* − 1)·*q*, interpolating
linearly between the two numbers around it:

$$
Q(q) = x_{\lfloor h \rfloor} + \bigl(h - \lfloor h \rfloor\bigr)\,\bigl(x_{\lfloor h \rfloor + 1} - x_{\lfloor h \rfloor}\bigr) .
$$

This is the default of numpy's `quantile` (statisticians call it "type 7"),
and it is what every quantile in this book uses. The brain's six
neighbourhood quantiles are computed the same way
([What a brain sees](brain-inputs.md)).

**Example.** The five numbers 1, 3, 4, 9, 100: *Q*(0.25) is at *h* = 1, so
3; the median at *h* = 2, so 4; *Q*(0.9) at *h* = 3.6, so
9 + 0.6·(100 − 9) = 63.6.

## The middle half and nine in ten

Two ranges appear in almost every figure:

- **the middle half**: from *Q*(0.25) to *Q*(0.75) — half of the values lie
  in it;
- **nine in ten**: from *Q*(0.05) to *Q*(0.95) — 90% of the values lie in it.

With thirty worlds, nine in ten leaves out the lowest one or two and the
highest one or two. See [Bands](bands.md) for how this is drawn over time.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
