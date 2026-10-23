# Families

How many separate lines of descent does a world hold? Counting genotypes
gives too many: one [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") makes a new genotype, but parent and child are
still one family. The **families** count, recorded as `cladesInWindow`, asks
instead how many different **ancestors of eight [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") ago** the living
descend from.

## Definition

At iteration *t*, let the **anchor** be *t* − 8. For every living [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), take
its genotype *g* and climb the family tree ([Genotypes](genotype.md)):

1. If *g* was born at or before the anchor — it first appeared in a [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") of
   iteration ≤ *t* − 8 — stop: *g* is this agent's family.
2. If *g* has no parent (a [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s genotype), stop: *g* is the family.
3. Otherwise move to *g*'s parent and go back to step 1.

The **number of families** is the number of different genotypes the climbs
stop at.

## An example

At iteration 100, the living carry genotypes 5,021, 5,022, 5,100 and 4,870.
Suppose 5,021 and 5,022 were born at iteration 96 and 97, both from 4,900,
born at iteration 95, whose parent 4,610 was born at 88. Genotype 5,100 was
born at 99 from 4,870, and 4,870 itself at 90.

- 5,021 → 4,900 (95, after the anchor 92) → 4,610 (88 ≤ 92): family **4,610**.
- 5,022 → 4,900 → 4,610: family **4,610**.
- 5,100 → 4,870 (born at 90 ≤ 92): family **4,870**.
- 4,870: born at 90 ≤ 92: family **4,870**.

Two families.

## Why eight iterations?

Short enough to be a count of *current* lines, long enough that a single
mutation does not count as a family of its own. With a mutation probability
of 0.2 per [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") and [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), a line gathers on average about 1.6 mutations in
eight iterations, so the climb usually goes a few steps.

## Early in a run

Before iteration 8, "eight iterations ago" is before the world began: the
climb ends at the founders, and the count is how many of the founders still
have descendants ([The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations)).

## In the code

`CladeWindow` in `gol_series.py`. It needs every frame in order, because
the family tree is a chain; it is computed while a run records, and kept for
both phases' rows. It holds a genotype's place in the tree for sixteen
iterations after it first appears, and for as long as it lives if that is
longer.

Until 2026-10-09 it let go of living genotypes too. One that outlived sixteen
iterations was then taken for a newborn the next time it was seen, and for
eight iterations was counted into its parent's family. Runs recorded before
that date carry counts that are slightly low: by 0.2% on average, and by
2% at most, over the 3,000 iterations of `B1-10000-s001`. That is small
beside the differences between [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."), but it also made the count depend on
when a run had last been paused.

<!-- turns -->
---

← [Brain diversity](brain-diversity.md) · [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are) · [The common ancestor of the living](common-ancestor.md) →

> [!info]- Pages that link here (9)
> [Glossary](glossary.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Genotypes and the family tree](genotype.md) · [Brain diversity](brain-diversity.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Who helps whom](who-helps-whom.md) · [One world, many colours](one-world-many-colours.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
