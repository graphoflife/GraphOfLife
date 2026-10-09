# The power spectrum of a time series

A world's number of agents goes up and down. Is that a slow drift over
hundreds of iterations, or a quick jitter from one iteration to the next?
The **power spectrum** splits the motion into rhythms and says how much of it
happens at each.

## Waves

Any series of *N* values *x*₀ … *x*_{N−1} can be written exactly as a sum of
sine waves of the frequencies *f* = *k*/*N*, *k* = 0, 1, …, *N*/2 — *k* full
cycles over the series, a **period** of *N*/*k* steps. The discrete Fourier
transform

$$
X_k = \sum_{t=0}^{N-1} x_t\, e^{-2\pi i k t / N}
$$

gives each wave's size and timing; |*X*ₖ|² is its **power**. The powers add
up (apart from a constant factor) to the series' variance: the spectrum says
how the variance is shared among the rhythms.

## Three preparations

1. **Remove the trend.** A slow overall rise is not a rhythm, but would put
   much power at the lowest frequencies. The book subtracts the least-squares
   straight line first.
2. **Taper the ends.** The transform treats the series as if it repeated, and
   the jump from its last value back to its first would spread false power
   over all frequencies ("leakage"). Multiplying by a **Hann window**,
   ½(1 − cos 2π*t*/(*N*−1)), which rises from 0 to 1 and back, removes the
   jump.
3. **Normalise.** Dividing by the total power makes worlds of different sizes
   comparable before they are averaged.

## Reading the slope

On log–log axes, many natural series give a straight line, power ∝ *f*^β:

| β | name | what it means |
|---|---|---|
| 0 | white noise | every value independent of the last; all rhythms equally strong |
| −1 | pink or 1/*f* noise | memory on every time scale at once |
| −2 | brown noise, a random walk | each value the last plus an independent step |

A random walk held near a level by a restoring force — the
**Ornstein–Uhlenbeck process** — has a spectrum

$$
S(f) \propto \frac{1}{1 + (2\pi f \tau)^2} :
$$

like a random walk (β = −2) for periods shorter than 2πτ, flat for longer
ones, where τ is the time the force needs to pull a deviation back by a
factor *e*. Its autocorrelation after a time *s* is *e*^(−*s*/τ)
([Autocorrelation](autocorrelation.md)).

## In the baseline

The number of agents of the baseline worlds has β ≈ −2.1 for periods
between 4 and 500 iterations, and a spectrum that levels off for periods of
800 iterations and more ([Chapter 29](../chapters/29-power-laws-real-and-apparent.md)):
a random walk on short scales, pulled back on long ones, with τ of a few
hundred iterations. 70% of the variance lies at periods of 500 iterations or
more, and 0.4% at periods shorter than 10.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
