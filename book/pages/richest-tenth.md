# The richest tenth's share

A second, plainer measure of inequality than the
[Gini coefficient](gini-coefficient.md): **what share of all tokens do the
richest ten percent of agents hold?** If everyone held the same, it would be
10%.

## Definition

With *n* agents, the richest tenth is the top

$$
m = \max\bigl(1,\ \lfloor 0.1\,n + 0.5 \rfloor\bigr)
$$

agents — a tenth of *n*, rounded to the nearest whole number (halves rounded
up), and at least one. Sort the agents' tokens from largest to smallest and
add up the first *m*:

$$
\text{topDecileShare} = \frac{x_{[1]} + x_{[2]} + \dots + x_{[m]}}{S},
$$

where *S* is all the tokens. Recorded as `topDecileShare` after every phase.

## Example

Twelve agents: 0.1 · 12 + 0.5 = 1.7, so *m* = 1: the share of the single
richest agent. Fifteen agents: 2.0, so *m* = 2. With 1,300 agents, *m* = 130.

## How it relates to the Lorenz curve

The richest tenth hold 1 minus what the poorest 90% hold — one minus the
height of the [Lorenz curve](gini-coefficient.md) at 0.9 (up to the rounding
of *m*). The Gini coefficient sums up the whole curve; this share reads one
point of it, the one people most often ask about.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
