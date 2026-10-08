# How a coalition takes a node

In the game every node goes to someone. Usually that is whoever staked the
most on it. But a **coalition** of smaller stakers can take a node from the
largest one, if together they marked enough of their stakes as
**revolutionary**. This note gives the rule exactly, shows that it is the
same as a simpler statement, and works an example.

## What each staker brings

For a node *v*, collect every stake placed on it — by its own agent and by
its neighbours. A staker *i* who put *a*ᵢ tokens on *v* also marked a part of
them revolutionary:

$$
\rho_i = \bigl\lfloor f(r_{i,1}, r_{i,2}) \cdot a_i \bigr\rfloor ,
$$

where *r*ᵢ,₁ and *r*ᵢ,₂ are rows 9 and 10 of *i*'s brain output for the
candidate *v* and *f* is [the share function](share-function.md).

## The rule

1. **The hegemon.** Let *H* = maxᵢ *a*ᵢ be the largest stake on *v*, and
   *h* the staker who made it. If several made it, *h* is chosen among them
   uniformly at random.
2. **The coalition** is every *other* staker with a revolutionary part
   ρᵢ > 0. (The hegemon's own revolutionary part never counts: it is the one
   being fought.) Let *R* be the sum of their revolutionary parts. If there is
   no coalition, *h* wins.
3. **Climbing.** Sort the coalition from the smallest ρ to the largest.
   Members with equal ρ form one **rung**. Go up rung by rung, adding the
   rungs' ρ to a running sum *L*, the "lower class". At each rung ask:
   does the lower class outweigh everyone above it plus the hegemon?

   $$
   L > (R - L) + H \;?
   $$

   At the first rung where it does, **the coalition wins**, and the winner is
   one member of that rung, chosen uniformly at random. If no rung does, the
   hegemon wins.

Whoever wins, node *v* gets a copy of the winner's brain and **all** the
tokens staked on it ([Chapter 3](../chapters/03-one-iteration.md)).

## The simpler statement

The inequality *L* > (*R* − *L*) + *H* is the same as

$$
L > \frac{R + H}{2} .
$$

*L* grows rung by rung and reaches *R* at the top rung. So a rung where the
inequality holds exists exactly when *R* > (*R* + *H*)/2, that is, when

$$
R > H .
$$

**The coalition wins if and only if its revolutionary tokens together exceed
the largest single stake.** And the member who wins is the one at whose rung
the running sum first passes the halfway mark (*R* + *H*)/2 — a kind of
weighted median of the coalition, shifted up by the hegemon's weight.

Note the asymmetry: the hegemon is weighed with its **whole** stake *H*, the
coalition only with the parts its members marked revolutionary.

## A worked example

Four agents stake on node *v*:

| staker | stake *a* | revolutionary part ρ |
|---|---|---|
| A | 10 | 0 |
| B | 4 | 4 |
| C | 3 | 3 |
| D | 6 | 5 |

The hegemon is A, with *H* = 10. The coalition is B, C and D (A's own part
would not count anyway), with *R* = 4 + 3 + 5 = 12 > 10, so the coalition
wins. Sorted by ρ: C (3), B (4), D (5); the halfway mark is (12 + 10)/2 = 11.

| rung | *L* | *L* > 11? |
|---|---|---|
| C | 3 | no |
| B | 7 | no |
| D | 12 | **yes** |

D wins node *v*: the node takes on a copy of D's brain, and holds all
10 + 4 + 3 + 6 = 23 tokens staked on it.

If D had marked only 2 tokens revolutionary, *R* would be 9 < 10, and A would
have won.

## When there is no contest

Most nodes are not contested by a coalition. An agent who staked on its own
node alone keeps it, with all it staked. A node on which nobody staked
anything — not even its own agent — is left with no tokens, and its agent
starves in the cleanup.

## In the code

`_resolve_winner(offers, revolutionaries, rng)` in `GraphOfLifeSimple.py`.
With [`allow_revolutions`](settings.md#allow_revolutions) off, nobody marks
anything revolutionary and the hegemon always wins.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
