# What a brain says

## In short

For every candidate it looks at — itself and each [neighbour](the-world.md#the-network "An agent joined to another by a connection.") — a brain
produces one column of numbers: **45** in B1, 19 in B2. Each group of rows
has one job: how much to give a child, whom the child joins, where and how
to stake, how much of a stake is revolutionary, which [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") to hand
over, and what to say. Two decisions concern the [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") as a whole, and their
rows are averaged over all its columns first; the others are taken candidate
by candidate. An output can be any number, positive or negative; only the
message rows are squeezed into the range from −1 to 1.

## One answer sheet per candidate

[What a brain sees](brain-inputs.md) pictures each column of inputs as a
form about one candidate. The column of outputs is the answer sheet the brain
hands back for it. Every sheet has the same lines, in the same order. Some
lines make sense only once per agent, such as how much to give a child; for
those, the program reads the average over all of the agent's sheets.

## The outputs

This is B1's layout, rows counted from 0 as in the code. The decisions
themselves are explained in [One iteration, step by step](one-iteration.md).

| rows | name | used in | how it is read |
|---|---|---|---|
| 0–1 | child share | reproduction | averaged over all candidates' columns, then *f*(row 0, row 1) = the share of its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") the agent gives a child |
| 2–3 | link | reproduction | per candidate: (yes, no) — join the child to this candidate? |
| 4–5 | link mode | reproduction | per candidate: read rows 2–3 as a probability (row 4 > row 5) or sharply |
| 6 | stake score | game | per candidate: how much to stake here, relative to the others |
| 7–8 | spread mode | game | averaged over all columns: spread the stake by score (row 7 > row 8) or put it all on the best-scored candidate |
| 9–10 | [revolution](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") | game | per candidate: *f*(row 9, row 10) = the share of the stake on it marked revolutionary |
| 11–12 | handover | reproduction | per neighbour: (yes, no) — give this connection to the child? |
| 13–14 | handover mode | reproduction | per neighbour: probability or sharp, as for links |
| 15–44 | message | both | per candidate: the 30 numbers of the message to it, each squashed into (−1, 1) by tanh |

Rows 0–14 are 15 numbers; with the 30 message numbers that is **45**. *f* is
[the share function](share-function.md); "probability or sharp" is
[the yes-or-no rule](binary-decision.md); a spread stake is split by
[the largest-remainder method](largest-remainder.md).

## Averaged or per candidate?

Two decisions are about the agent as a whole: *how much do I give my child?*
and *do I spread my stake?* But the brain answers once per candidate. So
these outputs, rows 0–1 and 7–8, are **averaged over all of the agent's
columns** (itself and every neighbour) before they are used. The other
decisions are about one candidate each, and use that candidate's column.

## The output layer is linear

The hidden layers pass their sums through the sigmoid, a function that
squeezes any number into the range from 0 to 1. The last layer does not: an
output can be any real number, positive or negative. Only the message rows
are squashed afterwards, by tanh, which squeezes any number into the range
from −1 to 1. So a message is always 30 numbers between −1 and 1. See
[The brain](the-brain.md).

## An example: what an evolved brain says

Take the 30 B1 worlds of 100,000 tokens (Experiment 10) at their end. Average
each output over a world's agents and their columns, and take the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.")
world ([What the brains are like](what-the-brains-are-like.md)):

- **Rows 0–1**, the child share: −0.53 and +0.70. The first is negative and
  the second positive, so *f* = 0: no child.
- **Row 6**, the stake score: +0.08 (the worlds range from −1.1 to +0.6),
  nearly the same for every candidate.
- **Rows 7–8**: +0.61 against −0.45. Row 7 is the larger, so the stake is
  spread.
- **Rows 9–10**: +0.64 and −0.61, so *f* = 1: all of a stake is marked
  revolutionary. In these worlds, 97% of staked tokens are.

Read together: have no child, spread the stake almost evenly, and mark all of
it revolutionary. A [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s brain, for comparison, averages 0 on every one
of these rows.

## When some rules are off, or added

The layout follows the rules. Without revolutions there are no rows 9–10;
without handovers, no rows 11–14; and the rows after them move up. B1 has
both, and the layout above. B2 has no handovers: its rows 0–10 are as above,
and its message of 8 numbers fills rows 11–18, 19 rows in all. Rules that
some experiments switch on add rows before the message: gifts add six, a
search for strangers four ([Every setting](settings.md)).

## Where it is used

- [What the brains are like](what-the-brains-are-like.md): where selection
  pushed each row in B1's worlds, and why the stake score barely changes
  from one candidate to the next.
- [How the game is played](how-the-game-is-played.md): what rows 6–10 come to
  in B1's worlds. From iteration 500 on, 98% of the agents spread their stake.
- [What a brain can compute](what-a-brain-can-compute.md): a B2 brain whose
  weights are set by hand so that its 19 outputs play a rule written in
  Python.

## Where to go next

- [What a brain sees](brain-inputs.md): the inputs, row by row.
- [Messages](messages.md): what becomes of the message rows.
- [The brain](the-brain.md): the network between the inputs and the outputs.

## How our view changed

- **8 October 2026.** This page said that every run of the notebook has
  both revolutions and handovers on, so that the layout above is the only
  one.
- **12 October 2026.** B2 was made without handovers
  ([B1 and B2: the two baselines](b1-and-b2.md)): its brains have 19 outputs.
  Later experiments switch other rules on and off as well. The page now says
  how each of them changes the layout.

<!-- turns -->
---

← [What a brain sees](brain-inputs.md) · [How the simulation works](../README.md#how-the-simulation-works) · [Messages](messages.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (10)
> [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Every setting](settings.md) · [The share function f(a, b)](share-function.md) · [A yes-or-no decision](binary-decision.md) · [What a brain sees](brain-inputs.md) · [Messages](messages.md) · [How the game is played: what agents stake, and who wins the nodes](how-the-game-is-played.md) · [Measuring the flow: token flow, lightning and circulation](token-flow.md) · [What a brain can compute](what-a-brain-can-compute.md)
<!-- /turns -->
