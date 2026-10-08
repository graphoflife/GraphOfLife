# The bootstrap

Thirty worlds give a mean. Another thirty would give a somewhat different
one. How uncertain is the mean? The **bootstrap** (Efron 1979) answers
without any formula about distributions, by re-sampling the data we have.

## The idea

The thirty values we measured are our best picture of the population they
come from. So draw new "samples of thirty" **from them**: thirty values
picked at random **with replacement** (the same world may be picked twice,
another not at all). Compute the mean of each such sample. The spread of
those means shows how much the mean would vary from one set of thirty worlds
to another.

## Exactly how

Given values *x*₁, …, *x*ₙ:

1. Repeat *B* = 10,000 times: draw *n* indices uniformly at random from
   1, …, *n*, with replacement, and compute the mean of the values at them.
2. The **95% interval** for the mean runs from the 2.5th to the 97.5th
   percentile of the 10,000 means ([Median and quantiles](median-and-quantiles.md)).

The random draws are seeded, so the interval comes out the same every time.

## Example

Values 2, 4, 9. One resample might be (4, 4, 9), mean 5.67; another
(2, 9, 2), mean 4.33. Over 10,000 resamples, the means range from 2 (all
three picks the 2) to 9, and the middle 95% of them form the interval.

## For a difference

To compare two conditions, the book bootstraps the **difference** of their
means. If the runs are paired by seed, it resamples the paired differences
*dᵢ* = *yᵢ* − *xᵢ*; otherwise it resamples each condition separately and
subtracts. An interval that does not contain 0 means the data are hard to
reconcile with "no difference" — which is then tested directly with a
[permutation test](permutation-test.md).

## In the code

`bootstrap` and `compare_conditions` in `gol_analysis.py`.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
