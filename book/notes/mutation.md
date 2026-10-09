# How a brain changes

Brains change — this is the **variation** evolution needs
([Chapter 1](../chapters/01-what-this-book-is-about.md)). This note gives the
exact rule, with the baseline's numbers.

## When

A brain is offered a change at two moments:

1. **At birth.** A child's brain is a copy of its parent's, and the copy is
   offered a change.
2. **After every game.** Every brain in the world — after the cleanup, so
   only the survivors' — is offered a change, each independently.

Each time, the brain changes **with probability** *p* = 0.2
([`mutation_probability`](settings.md#mutation_probability)): a uniform
random number *U* is drawn, and the brain changes if *U* ≤ *p*.

So in one iteration, an agent's brain is offered at least one change (after
the game), and a newborn's two (at birth and after the game). The chance
that a given brain is changed at least once in an iteration is
1 − 0.8 = 0.2 for an old agent and 1 − 0.8² = 0.36 for a newborn.

## What a change does

A brain is a list of **weight matrices** *W* and **bias vectors** *b*, one of
each per layer ([Chapter 4](../chapters/04-the-brain.md)). Every one of them
is changed separately, in two steps. Let *s* = 0.1
([`mutation_sparsity`](settings.md#mutation_sparsity)), σ = 0.2
([`mutation_noise_std`](settings.md#mutation_noise_std)), and let *k* be the
**fan-in** of the layer: the number of inputs each of its neurons has.

**Step 1 — jitter.** Every number of the matrix (or vector) is, independently
with probability *s*, moved by a normal random amount:

$$
w \;\leftarrow\; w + \varepsilon, \qquad \varepsilon \sim \mathcal{N}\!\left(0,\ \left(\frac{\sigma}{\sqrt{k}}\right)^{2}\right).
$$

**Step 2 — a rare reset.** With probability *s*, the matrix is also reset in
part: every number of it is, independently with probability *s*, replaced by
a fresh draw — for a weight matrix from the distribution the founders' weights
were drawn from, 𝒩(0, (1/√*k*)²), for a bias vector from 𝒩(0, (σ/√*k*)²).

So a change moves about one number in ten by a little, and now and then
redraws about one number in ten of a matrix from scratch.

## How big "a little" is

A founder's weights are drawn from 𝒩(0, (1/√*k*)²) — standard deviation
1/√*k*. A jitter has standard deviation σ/√*k* = 0.2/√*k*: **a fifth of the
typical size of a weight**. For the first layer of the baseline brain,
*k* = 154, so a weight is typically about 1/√154 ≈ 0.081 in size and a
jitter about 0.016.

Per change, the expected number of jittered weights is
*s* · 15,550 ≈ 1,555 of the brain's 15,550 weights. Per matrix, a reset happens
with probability 0.1 and then touches about a tenth of its numbers.

## A new genotype

A brain that changed gets a **new genotype number**, and records the
genotype it came from as its parent ([Genotypes and the family tree](genotype.md)).
A brain that was offered a change and did not change keeps its number.

One special case matters in [Chapter 31](../chapters/31-do-the-brains-matter.md):
with *s* = 0, step 1 and step 2 change nothing — but a brain drawn to change
(with probability *p*) still gets a new genotype number. Its weights are
those of its parent exactly.

## Precision

In the baseline, weights are stored as 16-bit numbers
([`brain_kind`](settings.md#brain_kind)). The change is computed in 64 bits
and the result stored in 16, so a jitter much smaller than a weight's last
stored digit can be lost in the rounding.

## In the code

`Brain.mutate` and `Brain._perturb` in `GraphOfLifeSimple.py`. Every draw comes
from the world's random stream ([Seeds and the random stream](random-numbers.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
