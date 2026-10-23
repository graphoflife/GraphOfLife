# A yes-or-no decision

Some decisions are a plain yes or no: *should my child be joined to this
[neighbour](the-world.md#the-network "An agent joined to another by a connection.")? should I hand this [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") to my child?* A [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") answers each
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
generations like any other. A [lineage](families.md "A line of descent: the agents whose genotypes go back to one ancestor.") can evolve to be decisive or to be
erratic.

The same idea is used for the stake: a pair of mode outputs decides whether
an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") spreads its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") over its candidates or puts them all on one
([One iteration, step by step](one-iteration.md)).

## Where it is used

- **Linking a newborn**: one decision per candidate of the parent, the
  parent itself included — outputs 2–3 with mode 4–5.
- **Handing over a connection**: one decision per neighbour of the parent —
  outputs 11–12 with mode 13–14.

In the code it is `_choose_binary(yes, no, mode_yes, mode_no, rng)` in
`GraphOfLifeSimple.py`. The uniform number comes from the world's random
stream ([Seeds and the random stream](random-numbers.md)).

<!-- turns -->
---

← [The share function f(a, b)](share-function.md) · [How the simulation works](../README.md#how-the-simulation-works) · [Splitting tokens into whole numbers](largest-remainder.md) →

> [!info]- Pages that link here (6)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Seeds and the random stream](random-numbers.md) · [The share function f(a, b)](share-function.md) · [What a brain says](brain-outputs.md)
<!-- /turns -->
