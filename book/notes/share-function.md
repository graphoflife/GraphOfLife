# The share function f(a, b)

Many decisions in Graph of Life come out of a brain as **two numbers**, and
must be turned into **one share** between 0 and 1: how much of my tokens to
give my child, how much of a stake to mark revolutionary, how likely I am to
say yes. One function does it everywhere.

## Definition

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

| *a* | *b* | *f*(*a*, *b*) | why |
|---|---|---|---|
| 0.8 | 1.2 | 0.4 | 0.8 / (0.8 + 1.2) |
| 3 | −1 | 1 | *b* counts as 0, so *a* has all of it |
| −1 | 3 | 0 | *a* counts as 0 |
| −2 | −5 | 0.5 | neither is positive |
| 0 | 0 | 0.5 | neither is positive |
| 5 | 5 | 0.5 | equal parts |

## Properties worth knowing

- 0 ≤ *f*(*a*, *b*) ≤ 1 always, and *f*(*a*, *b*) + *f*(*b*, *a*) = 1.
- Only the **ratio** of the positive parts matters: *f*(2, 1) = *f*(20, 10) =
  2/3. Making both outputs larger does not make the share larger.
- The share is exactly 0 or exactly 1 whenever one of the two numbers is
  zero or negative and the other positive, and exactly ½ whenever both are.
  A brain can therefore say "all", "nothing" or "half" outright — and random
  brains do: in the first reproduction phase of the 30 baseline worlds, of
  the 2,241 founders that had a child, 33% gave it every token they had (and
  died for it, holding none), and another 33% gave exactly half
  ([Chapter 4](../chapters/04-the-brain.md)).

## Where it is used

- **The child's tokens.** A parent with τ tokens gives its child
  ⌊*f*(*ā*, *b̄*) · τ⌋, where *ā* and *b̄* are its two reproduction outputs
  averaged over all its candidates ([Chapter 3](../chapters/03-one-iteration.md)).
- **The revolutionary part of a stake.** Of *a* tokens staked on a node,
  ⌊*f* · *a*⌋ are marked revolutionary ([How a coalition takes a node](revolution.md)).
- **A probability.** A yes-or-no decision read as a probability says yes
  with probability *f*(*y*, *n*) ([A yes-or-no decision](binary-decision.md)).

In the code it is `_share_of_first(a, b)` in `GraphOfLifeSimple.py`.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
