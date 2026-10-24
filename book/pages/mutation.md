# How a brain changes

## In short

Brains change: this is the **variation** that evolution needs
([What this notebook is about](what-this-notebook-is-about.md)). A brain is
offered a change whenever it is copied into a newborn, and it changes with
probability 0.2. [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") also offers every brain a change after every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."); [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")
instead offers one whenever a [node](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") won by a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") takes on a copy of the
winner's brain. A change nudges about one number in ten by a fifth of a
weight's typical size, and now and then redraws some numbers from scratch. A
brain that changed is a new genotype.

## A copy with slips

Think of a brain as a long table of numbers, copied out by hand. Most copies
are perfect. One in five has slips: about one number in ten is a little off,
and now and then part of the table is written out afresh, from scratch. The
slips are blind. Whether one helps or harms is for the world to find out.

## When

A brain is offered a change at two moments:

1. **At birth**, in both [baselines](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."). A child's brain is a copy of its
   parent's, and the copy is offered a change. (In some experiments the copy
   is first mixed with a neighbour's brain:
   [`mix_at_birth`](settings.md#mix_at_birth).)
2. **After every game, in B1.** Every brain in the world — after the
   [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out."), so only the survivors' — is offered a change, each
   independently.

B2 replaces the second moment by another copy
([`mutate_on_replication`](settings.md#mutate_on_replication)). When a
neighbour wins a node, the node takes on a copy of the winner's brain, and
that copy is offered a change, as a newborn's is. A brain that stays where it
is never changes. The reason: in B1, 97% of new genotypes are made in brains
that are not being copied, so a successful brain is worn away where it sits
([What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)).

Each time, the brain changes **with probability** *p* = 0.2
([`mutation_probability`](settings.md#mutation_probability)). A uniform
random number *U* between 0 and 1 is drawn, every value equally likely, and
the brain changes if *U* ≤ *p*.

So in one [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") of B1, an agent's brain is offered at least one change
(after the game), and a newborn's two (at birth and after the game). The
chance that a given brain is changed at least once in an iteration is
1 − 0.8 = 0.2 for an old agent and 1 − 0.8² = 0.36 for a newborn.

## What a change does

A brain is a list of **weight matrices** *W* and **bias vectors** *b*, one of
each per layer: the numbers it computes with ([The brain](the-brain.md)).
Every one of them is changed separately, in two steps. Let *s* = 0.1
([`mutation_sparsity`](settings.md#mutation_sparsity)), σ = 0.2
([`mutation_noise_std`](settings.md#mutation_noise_std)), and let *k* be the
**fan-in** of the layer: the number of inputs each of its neurons has.

**Step 1: jitter.** Every number of the matrix (or vector) is, independently
with probability *s*, moved by a normal random amount. The amount is a draw
from a bell curve centred on 0, with standard deviation σ/√*k*:

$$
w \;\leftarrow\; w + \varepsilon, \qquad \varepsilon \sim \mathcal{N}\!\left(0,\ \left(\frac{\sigma}{\sqrt{k}}\right)^{2}\right).
$$

Here 𝒩(0, *d*²) stands for that bell curve, with standard deviation *d*.

**Step 2: a rare reset.** With probability *s*, the matrix is also reset in
part: every number of it is, independently with probability *s*, replaced by
a fresh draw. A weight is drawn from the distribution the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' weights
were drawn from, 𝒩(0, (1/√*k*)²); a bias from 𝒩(0, (σ/√*k*)²).

So a change moves about one number in ten by a little, and now and then
redraws about one number in ten of a matrix from scratch.

## How big "a little" is

A founder's weights are drawn from 𝒩(0, (1/√*k*)²): their standard deviation
is 1/√*k*. A jitter has standard deviation σ/√*k* = 0.2/√*k*: **a fifth of
the typical size of a weight**. For the first layer of the B1 brain,
*k* = 154, so a weight is typically about 1/√154 ≈ 0.081 in size, and a
jitter about 0.016.

Per change, the expected number of jittered weights is
*s* · 15,550 ≈ 1,555 of the B1 brain's 15,550 weights. B2's smaller brain has
66 · 16 + 16 · 19 = 1,360 weights, of which about 136 are jittered. Per
matrix, a reset happens with probability 0.1, and then touches about a tenth
of its numbers.

Jitters and resets pull a weight's size in opposite directions. The jitters
of a change add to a weight's variance, on average, *s* · σ² = 0.1 · 0.04 =
0.004 times a founder's variance: 0.4% of it. The same change redraws a given
weight from the founders' distribution with probability *s*² = 0.01. That
takes away, on average, 1% of the variance the weight has above a founder's.
The two balance where a weight's variance is 1.40 times a founder's, since
0.01 · (1.40 − 1) = 0.004. That is the size at which the first-layer weights
of evolved B1 brains are found: mutation alone sets it
([What the brains are like](what-the-brains-are-like.md)).

## A new genotype

A brain that changed gets a **new genotype number**, and records the
genotype it came from as its parent
([Genotypes and the family tree](genotype.md)). A brain that was offered a
change and did not change keeps its number.

One special case matters in [Do the brains matter?](do-the-brains-matter.md).
With *s* = 0, step 1 and step 2 change nothing. But a brain drawn to change
(with probability *p*) still gets a new genotype number. Its weights are
those of its parent, exactly.

## Precision

In B1 and B2, weights are stored as 16-bit numbers
([`brain_kind`](settings.md#brain_kind)), which keep about three significant
decimal digits. The change is computed in 64 bits and the result stored in
16, so a jitter much smaller than a weight's last stored digit can be lost in
the rounding.

## Where it is used

- [Do the brains matter?](do-the-brains-matter.md): B1 worlds whose brains
  never change (*s* = 0), against B1 itself (Experiment 7).
- [How fast should brains change?](how-fast-should-brains-change.md): B2
  worlds at five chances of change, *p* from 0.05 to 0.8 (Experiment 9).
- [Genotypes and the family tree](genotype.md): how the new genotype numbers
  make a family tree.
- [What the brains are like](what-the-brains-are-like.md): what mutation
  alone does to the size of an evolved brain's weights.

In the code it is `Brain.mutate` and `Brain._perturb` in
`GraphOfLifeSimple.py`. Every draw comes from the world's random stream
([Seeds and the random stream](random-numbers.md)).

## Where to go next

- [The brain](the-brain.md): the network whose numbers change.
- [Genotypes and lineages](genotypes-and-lineages.md): the family trees that
  the changes grow.
- [Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops): what changing brains find,
  and where they stop.

<!-- turns -->
---

← [How a coalition takes a node](revolution.md) · [How the simulation works](../README.md#how-the-simulation-works) · [When a world ends](extinction.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (19)
> [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Every setting](settings.md) · [Seeds and the random stream](random-numbers.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [What the brains are like](what-the-brains-are-like.md) · [Do the brains matter?](do-the-brains-matter.md) · [Genotypes, families and the common ancestor](genotype.md) · [Eras, contests and invasions: does a world keep getting better?](eras-and-contests.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md)
<!-- /turns -->
