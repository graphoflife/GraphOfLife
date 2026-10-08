# Splitting tokens into whole numbers

Tokens are whole: an agent cannot stake 3.7 tokens. When an agent spreads
its τ tokens over several candidates in proportion to scores, the shares
have to be rounded — and rounded so that **no token is lost or made**. The
rule used is the **largest-remainder method**, known from apportioning seats
in parliaments to parties (Hamilton's method).

## The rule

Given τ tokens and scores *w*₁, …, *w*ₘ (one per candidate):

1. **Negative scores count as zero:** *v*ⱼ = max(*w*ⱼ, 0).
2. **Shares:** *q*ⱼ = *v*ⱼ / Σᵢ *v*ᵢ. If every *v*ⱼ is zero, every share is
   1/*m*.
3. **Exact amounts:** *r*ⱼ = *q*ⱼ · τ. They add up to τ, but are not whole.
4. **Round down:** every candidate gets ⌊*r*ⱼ⌋.
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

## Details

- If two remainders are exactly equal, the order numpy's `argsort` returns
  for them decides — the same order every time for the same numbers, so the
  rule stays reproducible.
- A candidate with a negative or zero score gets nothing (unless every score
  is zero or negative, when the tokens are split evenly).

## Where it is used

In the game, when an agent **spreads** its stake: its τ tokens are split
over its candidates by their stake scores ([Chapter 3](../chapters/03-one-iteration.md)).
In the code it is `_apportion(weights, total)` in `GraphOfLifeSimple.py`.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
