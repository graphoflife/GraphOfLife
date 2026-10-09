# Families

How many separate lines of descent does a world hold? Counting genotypes
gives too many: one mutation makes a new genotype, but parent and child are
still one family. The **families** count, recorded as `cladesInWindow`, asks
instead how many different **ancestors of eight iterations ago** the living
descend from.

## Definition

At iteration *t*, let the **anchor** be *t* − 8. For every living agent, take
its genotype *g* and climb the family tree ([Genotypes](genotype.md)):

1. If *g* was born at or before the anchor — it first appeared in a frame of
   iteration ≤ *t* − 8 — stop: *g* is this agent's family.
2. If *g* has no parent (a founder's genotype), stop: *g* is the family.
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
of 0.2 per brain and game, a line gathers on average about 1.6 mutations in
eight iterations, so the climb usually goes a few steps.

## Early in a run

Before iteration 8, "eight iterations ago" is before the world began: the
climb ends at the founders, and the count is how many of the founders still
have descendants ([Chapter 11](../chapters/11-the-first-hundred-iterations.md)).

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
beside the differences between seeds, but it also made the count depend on
when a run had last been paused.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
