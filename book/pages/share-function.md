# The share function f(a, b)

## In short

A brain often answers with **two numbers** where the rules need **one
share** between 0 and 1. The share may be of its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") for a child, of a
stake to mark revolutionary, or the probability of saying yes. One function
turns every such pair into its share. Negative numbers count as zero, and the
share is the first number's part of the total; if neither is positive, the
share is one half. So a brain can say "all", "nothing" or "half" outright,
and brains of random weights often do.

## Two claims on one cake

Picture two people who divide a cake by what each of them claims. A claim
below zero is no claim at all. The first person gets the part of the cake
that their claim is of everything claimed. If neither claims anything, they
cut the cake in half.

The share function does exactly this with a brain's two numbers. The first
number, *a*, is the first claim, and the second, *b*, is the other. The share
is what the first claim gets.

## The rule

For two real numbers *a* and *b*,

$$
f(a, b) =
\begin{cases}
\dfrac{\max(a, 0)}{\max(a, 0) + \max(b, 0)} & \text{if } \max(a, 0) + \max(b, 0) > 0, \\[2ex]
\dfrac{1}{2} & \text{if } a \le 0 \text{ and } b \le 0 .
\end{cases}
$$

In words: negative numbers count as zero, and the share is *a*'s part of the
total. If neither number is positive, there is no preference, and the share
is one half.

## Examples

Take *a* = 0.8 and *b* = 1.2. Both are positive, so both count in full. The
total is 0.8 + 1.2 = 2, and *a*'s part of it is 0.8 / 2 = 0.4.

Now take *a* = 3 and *b* = −1. The negative *b* counts as zero, so the total
is 3, and *a* has all of it: the share is 1. The table holds these two
examples and four more.

| *a* | *b* | *f*(*a*, *b*) | why |
|---|---|---|---|
| 0.8 | 1.2 | 0.4 | 0.8 / (0.8 + 1.2) |
| 3 | −1 | 1 | *b* counts as 0, so *a* has all of it |
| −1 | 3 | 0 | *a* counts as 0 |
| −2 | −5 | 0.5 | neither is positive |
| 0 | 0 | 0.5 | neither is positive |
| 5 | 5 | 0.5 | equal parts |

## Three properties worth knowing

- **It is a share.** Always 0 ≤ *f*(*a*, *b*) ≤ 1, and swapping the two
  numbers gives the rest of the whole: *f*(*a*, *b*) + *f*(*b*, *a*) = 1.
- **Only the ratio counts.** Only the ratio of the positive parts matters:
  *f*(2, 1) = *f*(20, 10) = 2/3. Making both numbers larger does not make
  the share larger.
- **It has corners.** The share is exactly 0 or exactly 1 whenever one of the
  two numbers is zero or negative and the other positive. It is exactly ½
  whenever both are zero or negative. So a brain can say "all", "nothing" or
  "half" outright, without computing any ratio.

## All, nothing or half

Brains of random weights land on these corners often. A [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s weights
are drawn at random around zero, and the last layer of its brain does not
squash its sums ([The brain](the-brain.md)). So each of its two numbers is as
likely to be negative as positive, and their signs fall like two independent
coins. Each of four cases then has a chance of one in four:

| *a* | *b* | *f*(*a*, *b*) | the parent's child gets |
|---|---|---|---|
| positive | positive | between 0 and 1 | a part of its tokens |
| positive | zero or negative | 1 | all of them |
| zero or negative | positive | 0 | nothing: there is no child |
| zero or negative | zero or negative | ½ | half |

So about three founders in four should have a child. Of those, a third
should give it everything and a third exactly half. That is what happened in
the first reproduction phase of the 30 B1 worlds of 10,000 tokens
(Experiment 2). Of their 3,000 founders, 2,241 had a child. Of these, 33%
gave it every token they had, and died for it, holding none; another 33% gave
exactly half ([The brain](the-brain.md)).

Selection then pushes evolved brains firmly into the corners. Look at the 30
B1 worlds of 100,000 tokens (Experiment 10). In every one of them, the two
child-share numbers end up, on average over the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), the first negative
and the second positive. *f* reads that as 0: no child. The two numbers for
the revolutionary part of a stake end up the other way round, which *f*
reads as 1 ([What the brains are like](what-the-brains-are-like.md)).

## Where it is used

- **The child's tokens.** A parent with τ tokens gives its child
  ⌊*f*(*ā*, *b̄*) · τ⌋: the share of its tokens, rounded down to a whole
  number. Here *ā* and *b̄* are its two reproduction outputs, averaged over
  all its candidates ([How agents have children](how-agents-have-children.md),
  [One iteration, step by step](one-iteration.md)). B2 adds limits: a parent
  keeps at least one token, a child needs at least 10, and an agent with 8
  [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") or more has no child
  ([B1 and B2: the two baselines](b1-and-b2.md)).
- **The revolutionary part of a stake.** Of *a* tokens staked on a node,
  ⌊*f* · *a*⌋ are marked revolutionary ([How a coalition takes a node](revolution.md)).
- **A probability.** A yes-or-no decision read as a probability says yes
  with probability *f*(*y*, *n*) ([A yes-or-no decision](binary-decision.md)).
- **Rules of some experiments**, read the same way: how much of its tokens an
  agent gives away as gifts ([`allow_gifting`](settings.md#allow_gifting)),
  and how far it searches for strangers
  ([`link_search`](settings.md#link_search)).

In the code it is `_share_of_first(a, b)` in `GraphOfLifeSimple.py`.

## Where to go next

- [A yes-or-no decision](binary-decision.md): the share read as a
  probability.
- [Splitting tokens into whole numbers](largest-remainder.md): the other tool
  that turns a brain's numbers into whole tokens.
- [What a brain says](brain-outputs.md): which outputs come in pairs, and
  what each pair decides.

<!-- turns -->
---

← [Seeds and the random stream](random-numbers.md) · [How the simulation works](../README.md#how-the-simulation-works) · [A yes-or-no decision](binary-decision.md) →

> [!info]- Pages that link here (11)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [A yes-or-no decision](binary-decision.md) · [Splitting tokens into whole numbers](largest-remainder.md) · [What a brain says](brain-outputs.md) · [How a coalition takes a node](revolution.md) · [How births build the network](how-agents-have-children.md) · [What the brains are like](what-the-brains-are-like.md) · [What a brain can compute](what-a-brain-can-compute.md)
<!-- /turns -->
