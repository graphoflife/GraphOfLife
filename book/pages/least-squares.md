# Fitting a straight line, and R²

Many figures in this book draw a straight line through a cloud of points, and
many statistics of the viewer are the slope of such a line. This note says
which line, and what its R² means.

## The line that misses least

Given *n* points (*x*₁, *y*₁), …, (*x*ₙ, *y*ₙ), the **least-squares line**
*y* = *a* + *b x* is the one that makes the sum of squared vertical misses

$$
\sum_i \bigl(y_i - a - b x_i\bigr)^2
$$

as small as possible. Setting the derivatives in *a* and *b* to zero gives,
with means *x̄* and *ȳ*,

$$
b = \frac{S_{xy}}{S_{xx}}, \qquad a = \bar y - b \bar x ,
$$

$$
S_{xx} = \sum_i (x_i - \bar x)^2, \quad
S_{yy} = \sum_i (y_i - \bar y)^2, \quad
S_{xy} = \sum_i (x_i - \bar x)(y_i - \bar y) .
$$

The line always passes through the point of means (*x̄*, *ȳ*).

## R²

How much better is the line than no line at all? Without a line, the best
single guess for every *y* is *ȳ*, with squared misses *S*_yy. With the
line, the squared misses shrink to *S*_yy − *S*_xy²/*S*_xx. The share removed
is

$$
R^2 = \frac{S_{xy}^2}{S_{xx}\, S_{yy}} ,
$$

which is the square of the [correlation](correlation.md) *r*. R² = 1: every
point on the line. R² = 0: the line explains nothing.

## Example

Points (1, 2), (2, 3), (3, 7). Means 2 and 4. Deviations of *x*: −1, 0, 1;
of *y*: −2, −1, 3. *S*_xx = 2, *S*_yy = 14, *S*_xy = 2 + 0 + 3 = 5. So
*b* = 2.5, *a* = 4 − 5 = −1, and R² = 25/28 ≈ 0.89.

## On logarithms

A power law *y* = *A x*^*b* becomes a straight line in the logarithms,
ln *y* = ln *A* + *b* ln *x* ([Logarithmic axes](logarithmic-axes.md)). So the
book and the viewer fit power laws by least squares on (ln *x*, ln *y*). The
program (`_power_fit` in `gol_series.py`) keeps only the points with both
values positive — a zero cannot be logged and is a real observation, not a
small one — and wants at least 8 of them.

Two things to keep in mind:

- **R² is about scatter.** A curve that bends gently has a high R² over a
  short range; a true law with noisy points a low one. R² says how much of
  the spread a line accounts for, not whether a law holds.
- **The line is pulled by where the points are.** Every point counts the
  same, so a crowd of points at small *x* decides the slope more than a few
  at large *x*. For distributions this makes least squares a poor way to
  find an exponent; [Fitting a power law](power-law-fit.md) says what to do
  instead.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
