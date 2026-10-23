# What a brain says

For every candidate it looks at — itself and each [neighbour](the-world.md#the-network "An agent joined to another by a connection.") — a brain
produces one column of **45 numbers** (in the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")). Each group of rows
has one job. This note lists them, rows counted from 0 as in the code, and
says how each is used. The decisions themselves are explained in
[One iteration, step by step](one-iteration.md).

## The outputs

| rows | name | used in | how it is read |
|---|---|---|---|
| 0–1 | child share | reproduction | averaged over all candidates' columns, then *f*(row 0, row 1) = the share of its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") the [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") gives a child |
| 2–3 | link | reproduction | per candidate: (yes, no) — join the child to this candidate? |
| 4–5 | link mode | reproduction | per candidate: read rows 2–3 as a probability (row 4 > row 5) or sharply |
| 6 | stake score | game | per candidate: how much to stake here, relative to the others |
| 7–8 | spread mode | game | averaged over all columns: spread the stake by score (row 7 > row 8) or put it all on the best-scored candidate |
| 9–10 | [revolution](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") | game | per candidate: *f*(row 9, row 10) = the share of the stake on it marked revolutionary |
| 11–12 | handover | reproduction | per neighbour: (yes, no) — give this [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") to the child? |
| 13–14 | handover mode | reproduction | per neighbour: probability or sharp, as for links |
| 15–44 | [message](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") | both | per candidate: the 30 numbers of the message to it, each squashed into (−1, 1) by tanh |

Rows 0–14 are 15 numbers; with the 30 message numbers that is **45**.
*f* is [the share function](share-function.md); "probability or sharp" is
[the yes-or-no rule](binary-decision.md).

## Averaged or per candidate?

Two decisions are about the agent as a whole — *how much do I give my
child?* and *do I spread my stake?* — but the brain answers once per
candidate. These outputs are therefore **averaged over all of the agent's
columns** (itself and every neighbour) before they are used. The other
decisions are about one candidate each and use that candidate's column.

## The output layer is linear

The hidden layers pass their sums through the sigmoid function, but the last
layer does not: an output can be any real number, positive or negative. Only
the message rows are squashed afterwards, by tanh, so a message is always 30
numbers between −1 and 1. See [The brain](the-brain.md).

## When some mechanics are off

The layout shrinks with the rules: without revolutions there are no rows
9–10, without handovers no rows 11–14, and the rows after them move up. In
every run of this notebook both are on and the layout is the one above.

<!-- turns -->
---

← [What a brain sees](brain-inputs.md) · [How the simulation works](../README.md#how-the-simulation-works) · [Messages](messages.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (7)
> [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [A yes-or-no decision](binary-decision.md) · [What a brain sees](brain-inputs.md) · [Messages](messages.md) · [How the game is played](how-the-game-is-played.md) · [What a brain can compute](what-a-brain-can-compute.md)
<!-- /turns -->
