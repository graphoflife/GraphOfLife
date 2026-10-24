# How a coalition takes a node

## In short

In the game every [node](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") goes to someone, usually whoever staked the most on
it: the **hegemon**. But a **coalition** of smaller stakers takes the node if
the parts of their stakes they marked **revolutionary** add up to more than
the hegemon's whole stake. The node then goes to one member of the
coalition: the one at whom a running sum of their revolutionary parts,
smallest first, passes the halfway mark. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") coalitions decide about half
of all nodes; in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") they hardly ever win.

## The largest army, and a band against it

Think of a hill that goes to the largest army marching on it. Smaller armies
can band together against that army: each swears part of its troops to the
band. If the sworn troops together outnumber the largest army, the band
takes the hill.

Which member of the band then holds it? Line the band up from the smallest
sworn part to the largest, and count the sworn troops along the line. The
hill goes to the member at whom the count first passes half of everything
that matters on the field: all the sworn troops, plus the largest army.

The rest of this page states the rule as the program has it, shows that it
says the same as this picture, and works an example.

## What each staker brings

For a node *v*, collect every stake placed on it — by its own agent and by
its [neighbours](the-world.md#the-network "An agent joined to another by a connection."). A staker *i* who put *a*ᵢ [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") on *v* also marked a part of
them revolutionary:

$$
\rho_i = \bigl\lfloor f(r_{i,1}, r_{i,2}) \cdot a_i \bigr\rfloor ,
$$

where *r*ᵢ,₁ and *r*ᵢ,₂ are rows 9 and 10 of *i*'s [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") output for the
candidate *v*, *f* is [the share function](share-function.md), and ⌊ ⌋ rounds
down to a whole number of tokens.

## The rule

1. **The hegemon.** Let *H* = maxᵢ *a*ᵢ be the largest stake on *v*, and
   *h* the staker who made it. If several made it, *h* is chosen among them
   uniformly at random.
2. **The coalition** is every *other* staker with a revolutionary part
   ρᵢ > 0. The hegemon's own revolutionary part never counts: it is the one
   being fought. Let *R* be the sum of the coalition's revolutionary parts.
   If there is no coalition, *h* wins.
3. **Climbing.** Sort the coalition from the smallest ρ to the largest.
   Members with equal ρ form one **rung**. Go up rung by rung, adding the
   rungs' ρ to a running sum *L*, the "lower class". At each rung ask: does
   the lower class outweigh everyone above it plus the hegemon?

   $$
   L > (R - L) + H \;?
   $$

   At the first rung where it does, **the coalition wins**, and the winner is
   one member of that rung, chosen uniformly at random. If no rung does, the
   hegemon wins.

Whoever wins, node *v* gets a copy of the winner's brain and **all** the
tokens staked on it ([One iteration, step by step](one-iteration.md)).

## The simpler statement

Add *L* to both sides of the inequality *L* > (*R* − *L*) + *H*, and divide
both by 2: it is the same as

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
the running sum first passes the halfway mark (*R* + *H*)/2. That makes the
winner a kind of weighted [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of the coalition, shifted up by the
hegemon's weight.

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
wins. Sorted by ρ, the coalition reads C (3), B (4), D (5), and the halfway
mark is (12 + 10)/2 = 11. Climb:

| rung | *L* | *L* > 11? |
|---|---|---|
| C | 3 | no |
| B | 7 | no |
| D | 12 | **yes** |

D wins node *v*: the node takes on a copy of D's brain, and holds all
10 + 4 + 3 + 6 = 23 tokens staked on it.

If D had marked only 2 tokens revolutionary, *R* would be 9 < 10, and A would
have won.

**The same stakes in B2.** B2 gives a defender a home advantage
([`incumbency`](settings.md#incumbency) = 1.5): an agent's stake on its own
node counts 1.5 times when the hegemon is chosen and *H* is measured.
Suppose A is the agent living on *v*. Its 10 tokens then count as *H* = 15,
and *R* = 12 < 15, so A keeps its node. The node still holds the 23 tokens
staked on it: only who wins is weighed.

## How often a coalition wins

Two cases need no climbing. An agent who staked on its own node alone keeps
it, with all it staked. A node on which nobody staked anything — not even
its own agent — is left with no tokens, and its agent starves in the
cleanup.

Everywhere else, how often a coalition wins depends on the world. In B1 it
decides about half of all nodes. In the last game of the 30 B1 worlds of
100,000 tokens (Experiment 10), coalitions won 52% of the nodes on which
anyone staked. The winning member was the agent living on the node for 12%
of all those nodes, and a neighbour for 40%. By then 97% of all staked tokens
are marked revolutionary
([How the game is played](how-the-game-is-played.md)). In B2's [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") worlds
of 10,000 tokens, coalitions won about one node in three hundred a game. B2
with and without them was the same world in every measure
([The switches the world was built with](the-switches-the-world-was-built-with.md)).

## Where it is used

- [How the game is played](how-the-game-is-played.md): who wins the nodes of
  B1's worlds, and how.
- [The switches the world was built with](the-switches-the-world-was-built-with.md):
  B1 and B2 without revolutions. B1 without them becomes a world of mayflies,
  whose agents are 7 to 13 iterations old on average.
- [Revolutions, with every other rule](revolutions-with-every-other-rule.md):
  the worlds near B2 in which revolutions do matter.
- [A crew of raiders](does-evolution-close-the-door.md#a-crew-of-raiders): two raiders of two tokens each
  stake one on a stranger of one token. In B2 the stranger's home stake
  counts as 1.5, so neither raider can take the node alone, but together,
  with *R* = 2, they do.

In the code it is `_resolve_winner(offers, revolutionaries, rng)` in
`GraphOfLifeSimple.py`. With
[`allow_revolutions`](settings.md#allow_revolutions) off, nobody marks
anything revolutionary, and the hegemon always wins.

## Where to go next

- [One iteration, step by step](one-iteration.md): where the rule sits in
  the game.
- [The share function f(a, b)](share-function.md): how a brain marks part of
  a stake revolutionary.
- [Staking at home, and keeping one's node](home-stake.md): how often an
  agent keeps its own node.

## How our view changed

- **8 October 2026.** This page said that most nodes are not contested by a
  coalition.
- **23 October 2026.** Corrected. [How the game is played](how-the-game-is-played.md),
  written the same day as that sentence, measured the opposite in B1: in its
  settled worlds coalitions decide about half of all nodes. In B2's pilot
  worlds they won only about one node in three hundred a game
  ([The switches the world was built with](the-switches-the-world-was-built-with.md),
  15 October 2026).

<!-- turns -->
---

← [Messages](messages.md) · [How the simulation works](../README.md#how-the-simulation-works) · [How a brain changes](mutation.md) →

> [!info]- Pages that link here (19)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Every setting](settings.md) · [Seeds and the random stream](random-numbers.md) · [The share function f(a, b)](share-function.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Gains and losses](gains-and-losses.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [How the game is played](how-the-game-is-played.md) · [Where tokens flow](where-the-tokens-flow.md) · [Staking at home, and keeping one's node](home-stake.md) · [How different are the agents?](how-different-are-the-agents.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Why the raiders stopped winning](does-evolution-close-the-door.md) · [Do agents cooperate?](do-agents-cooperate.md) · [The switches the world was built with](the-switches-the-world-was-built-with.md) · [Revolutions, with every other rule](revolutions-with-every-other-rule.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
