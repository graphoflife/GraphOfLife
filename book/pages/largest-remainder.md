# Splitting tokens into whole numbers

## In short

[Tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") are whole: an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") cannot stake 3.7 tokens. So when an agent spreads
its τ tokens over several candidates in proportion to scores, the shares have
to be rounded, and rounded so that **no token is lost or made**. The rule is
the **largest-remainder method**, known from sharing out the seats of a
parliament among parties (Hamilton's method). Every candidate first gets its
exact share rounded down, and the tokens left over go one each to the
candidates that lost the most in the rounding.

## Seats in a parliament

Think of a parliament whose seats are shared out among parties by their
votes. A party's exact share is rarely a whole number of seats, and a seat
cannot be cut. So each party first gets the whole seats it has earned. The
few seats left over go one each to the parties that lost the most by
rounding down. Every seat is given, and none twice.

The program shares an agent's tokens among its candidates the same way. The
tokens are the seats, the candidates are the parties, and the [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.")'s scores
are the votes.

## The rule

Given τ tokens and scores *w*₁, …, *w*ₘ, one per candidate:

1. **Negative scores count as zero:** *v*ⱼ = max(*w*ⱼ, 0).
2. **Shares:** *q*ⱼ = *v*ⱼ / Σᵢ *v*ᵢ, each candidate's part of the total. If
   every *v*ⱼ is zero, every share is 1/*m*.
3. **Exact amounts:** *r*ⱼ = *q*ⱼ · τ. They add up to τ, but are not whole.
4. **Round down:** every candidate gets ⌊*r*ⱼ⌋, its exact amount rounded
   down to a whole number.
5. **The leftovers:** what rounding down lost, *L* = τ − Σⱼ ⌊*r*ⱼ⌋, is a
   whole number between 0 and *m* − 1. The *L* candidates with the largest
   **remainders** *r*ⱼ − ⌊*r*ⱼ⌋ get one more token each.

The result is *m* whole numbers that add up to exactly τ, each within one
token of its exact share.

## A worked example

Split τ = 11 tokens by the scores (2.0, 1.0, −0.5, 0.5):

| | candidate 1 | 2 | 3 | 4 | sum |
|---|---|---|---|---|---|
| score *w* | 2.0 | 1.0 | −0.5 | 0.5 | |
| *v* = max(*w*, 0) | 2.0 | 1.0 | 0 | 0.5 | 3.5 |
| exact *r* = 11 · *v*/3.5 | 6.286 | 3.143 | 0 | 1.571 | 11 |
| rounded down | 6 | 3 | 0 | 1 | 10 |
| remainder | 0.286 | 0.143 | 0 | 0.571 | |
| + leftover (*L* = 1) | | | | +1 | |
| **tokens** | **6** | **3** | **0** | **2** | **11** |

Row by row: the third score, −0.5, counts as zero, so the scores that count
add up to 3.5. Candidate 1's exact amount is 11 · 2.0/3.5 = 6.286 tokens, and
so on along the row. Rounded down, the four candidates get 6, 3, 0 and 1:
10 tokens, so *L* = 1 token is left over. The largest remainder is
candidate 4's, 0.571, and candidate 4 gets the eleventh token.

## Zero scores, and ties

- **A score of zero or below gets nothing**, unless every score is zero or
  below. Then the tokens are split evenly, every share 1/*m*.
- **Equal remainders.** If two remainders are exactly equal, the order in
  which numpy's `argsort` returns them decides. (numpy is the library of
  numerical tools the program is written with, and `argsort` is its sorting
  function.) It is the same order every time for the same numbers, so the
  rule stays reproducible. In an even split every remainder is the same, and
  the leftovers go down the list of candidates in order. The agent's own node
  is first in that list, so it gets the first leftover token
  ([Where the tokens flow](where-the-tokens-flow.md)).

An agent with 3 tokens and one [neighbour](the-world.md#the-network "An agent joined to another by a connection.") that splits evenly shows both.
Each of its two candidates has an exact share of 1.5 tokens, and rounded
down, each gets 1. The one token left over goes to the first candidate, the
agent's own node: 2 at home, 1 on the neighbour
([The even split](even-split.md)).

## Where it is used

- **In the game**, when an agent **spreads** its stake: its τ tokens are
  split over its candidates by their stake scores
  ([One iteration, step by step](one-iteration.md)). Evolved [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") brains spread
  in 98% of cases. In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") nearly every agent puts all its tokens on one
  candidate, spreading in only 0.1–4% of cases ([The even split](even-split.md)).
  In B2, 30% of an agent's tokens (rounded up, at least one) also stay at
  home before anything is split, and only the rest is staked by the brain.
- **The even split.** In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") B1 world of 100,000 tokens, 42% of the
  agents score every candidate at zero or below, and the rule splits their
  tokens evenly ([The even split](even-split.md)).
- **Gifts**, in the experiments that allow them: the tokens an agent gives
  away are split over the neighbours it chose
  ([`allow_gifting`](settings.md#allow_gifting)).

In the code it is `_apportion(weights, total)` in `GraphOfLifeSimple.py`.

## Where to go next

- [The even split](even-split.md): what the rule does when no score is above
  zero, and why that is common.
- [Where the tokens flow](where-the-tokens-flow.md): how the rounding tilts
  stakes towards home.
- [The share function f(a, b)](share-function.md): the other tool that turns
  a brain's numbers into shares.

<!-- turns -->
---

← [A yes-or-no decision](binary-decision.md) · [How the simulation works](../README.md#how-the-simulation-works) · [What a brain sees](brain-inputs.md) →

> [!info]- Pages that link here (9)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [The share function f(a, b)](share-function.md) · [What a brain says](brain-outputs.md) · [The even split](even-split.md) · [Where the tokens flow](where-the-tokens-flow.md) · [What the brains are like](what-the-brains-are-like.md) · [How different are the agents?](how-different-are-the-agents.md)
<!-- /turns -->
