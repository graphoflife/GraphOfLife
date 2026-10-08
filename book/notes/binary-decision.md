# A yes-or-no decision

Some decisions are a plain yes or no: *should my child be joined to this
neighbour? should I hand this connection to my child?* A brain answers each
with **four numbers**, and the rule below turns them into the answer.

## The four numbers

For one candidate (one column of the brain's outputs, see
[What a brain says](brain-outputs.md)):

- *y* — how strongly the brain says **yes**;
- *n* — how strongly it says **no**;
- *m*₁ and *m*₂ — the **mode**: how the brain wants *y* and *n* to be read.

## The rule

1. If *m*₁ > *m*₂, the pair is read as a **probability**: draw a uniform
   random number *U* between 0 and 1, and answer yes if
   *U* < *f*(*y*, *n*) ([The share function](share-function.md)). So the
   answer is yes with probability *f*(*y*, *n*).
2. Otherwise (*m*₁ ≤ *m*₂), the pair is read **sharply**: the answer is yes
   if *y* > *n* and no if *y* < *n*. If *y* = *n* exactly, a fair coin
   decides.

## Examples

| *y* | *n* | *m*₁ | *m*₂ | how it is read | answer |
|---|---|---|---|---|---|
| 0.9 | 0.3 | 1.0 | −0.2 | probability | yes with probability 0.9/1.2 = 0.75 |
| 0.9 | 0.3 | −1.0 | 0.4 | sharp | yes, since 0.9 > 0.3 |
| −0.5 | −0.1 | 0.2 | 0.1 | probability | yes with probability 0.5 (neither positive) |
| −0.5 | −0.1 | 0.1 | 0.2 | sharp | no, since −0.5 < −0.1 |

## Why a mode?

Because then the brain decides not only *what* it prefers but also *whether
to act on its preference every time or only sometimes* — and since the mode
outputs are part of the brain, that choice is inherited and changes over
generations like any other. A lineage can evolve to be decisive or to be
erratic.

The same idea is used for the stake: a pair of mode outputs decides whether
an agent spreads its tokens over its candidates or puts them all on one
([Chapter 3](../chapters/03-one-iteration.md)).

## Where it is used

- **Linking a newborn**: one decision per candidate of the parent, the
  parent itself included — outputs 2–3 with mode 4–5.
- **Handing over a connection**: one decision per neighbour of the parent —
  outputs 11–12 with mode 13–14.

In the code it is `_choose_binary(yes, no, mode_yes, mode_no, rng)` in
`GraphOfLifeSimple.py`. The uniform number comes from the world's random
stream ([Seeds and the random stream](random-numbers.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
