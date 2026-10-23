# A yes-or-no decision

## In short

Some decisions are a plain yes or no, such as whether to join a newborn to
a [neighbour](the-world.md#the-network "An agent joined to another by a connection."), or to hand it a [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."). A [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") answers each with **four
numbers**. Two say how strongly it means yes and how strongly no; the other
two, the **mode**, say how to read that pair. Read as a probability, chance
draws the answer; read sharply, the stronger one always wins. The mode is
part of the brain, so whether a [lineage](families.md "A line of descent: the agents whose genotypes go back to one ancestor.") decides firmly or by chance is
inherited, and can evolve.

## Firm, or by chance

Two people can lean the same way and still act differently. Say both lean
three to one towards yes. The first always says yes. The second says yes
three times in four and no the fourth time, as if drawing lots weighted by
its leaning. A brain can be either of the two, and it says which, decision by
decision.

## The four numbers

For one candidate — one column of the brain's outputs, see
[What a brain says](brain-outputs.md) — the brain gives:

- *y*: how strongly it says **yes**;
- *n*: how strongly it says **no**;
- *m*₁ and *m*₂: the **mode**, how it wants *y* and *n* to be read.

## The rule

1. If *m*₁ > *m*₂, the pair is read as a **probability**. Draw a uniform
   random number *U* between 0 and 1: every value in that range is equally
   likely. Answer yes if *U* < *f*(*y*, *n*), where *f* is
   [the share function](share-function.md). So the answer is yes with
   probability *f*(*y*, *n*).
2. Otherwise (*m*₁ ≤ *m*₂), the pair is read **sharply**: the answer is yes
   if *y* > *n*, and no if *y* < *n*. If *y* = *n* exactly, a fair coin
   decides.

## Examples

| *y* | *n* | *m*₁ | *m*₂ | how it is read | answer |
|---|---|---|---|---|---|
| 0.9 | 0.3 | 1.0 | −0.2 | probability | yes with probability 0.9/1.2 = 0.75 |
| 0.9 | 0.3 | −1.0 | 0.4 | sharp | yes, since 0.9 > 0.3 |
| −0.5 | −0.1 | 0.2 | 0.1 | probability | yes with probability 0.5 (neither positive) |
| −0.5 | −0.1 | 0.1 | 0.2 | sharp | no, since −0.5 < −0.1 |

Step by step, the first row. The mode compares *m*₁ = 1.0 with *m*₂ = −0.2.
The first is larger, so the pair is a probability. Both *y* = 0.9 and
*n* = 0.3 are positive, so *f*(0.9, 0.3) = 0.9 / (0.9 + 0.3) = 0.9/1.2 =
0.75. The answer is yes three times in four.

The second row has the same *y* and *n*, but now *m*₁ = −1.0 is smaller than
*m*₂ = 0.4. The pair is read sharply, and since 0.9 > 0.3 the answer is yes,
every time.

The last two rows hold two negative numbers. Read as a probability, both
count as zero, and the answer is a coin toss. Read sharply, the numbers are
compared as they are, negative or not, and −0.5 < −0.1 gives a firm no.

## Why a mode?

With a mode, a brain decides not only *what* it prefers, but also *whether
to act on its preference every time, or only sometimes*. The mode outputs
are part of the brain, so that choice is inherited and changes over
generations like any other. A lineage can evolve to be decisive, or to be
erratic.

The same idea is used for the stake. A pair of mode outputs decides whether
an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") spreads its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") over its candidates or puts them all on one
([One iteration, step by step](one-iteration.md)).

## Where it is used

- **Linking a newborn**: one decision per candidate of the parent, the
  parent itself included — outputs 2–3 with mode 4–5. In the [settled life](settled-life.md "A world from iteration 500 to its end, after its youth, where it is measured.") of
  the 30 [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") worlds of 100,000 tokens, 11.5% of children are joined to none of
  their parent's candidates at birth
  ([How agents have children](how-agents-have-children.md)). In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") a newborn
  is first joined to its parent and to two of the parent's neighbours, drawn
  at random, and the brain decides about the others.
- **Handing over a connection**: one decision per neighbour of the parent —
  outputs 11–12 with mode 13–14. In those B1 worlds, 75% of parents hand over
  no connection at all. B2 has no handovers.
- **Rules of some experiments**: whether to give a gift to a neighbour
  ([`allow_gifting`](settings.md#allow_gifting)), and whether to accept a
  link with a stranger met in a search ([`link_search`](settings.md#link_search)).

In the code it is `_choose_binary(yes, no, mode_yes, mode_no, rng)` in
`GraphOfLifeSimple.py`. The uniform number comes from the world's random
stream ([Seeds and the random stream](random-numbers.md)).

## Where to go next

- [The share function f(a, b)](share-function.md): how a pair of numbers
  becomes a probability.
- [What a brain says](brain-outputs.md): every output of a brain, and its
  job.
- [How agents have children](how-agents-have-children.md): the decisions of
  a birth, in the worlds.

<!-- turns -->
---

← [The share function f(a, b)](share-function.md) · [How the simulation works](../README.md#how-the-simulation-works) · [Splitting tokens into whole numbers](largest-remainder.md) →

> [!info]- Pages that link here (6)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Seeds and the random stream](random-numbers.md) · [The share function f(a, b)](share-function.md) · [What a brain says](brain-outputs.md)
<!-- /turns -->
