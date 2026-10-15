# Another game

> [!question] Questions of this chapter
> - Is it the Colonel Blotto contest that makes evolution stop, or would any
>   contest for nodes?
> - What changes when chance decides who takes a node, so that being
>   unpredictable is worth nothing?
> - What changes when agents with the same brain fight as one, so that a
>   lineage can act as a body?

## Why ask

[Chapter 50](50-why-evolution-stops.md) blamed the contest. A node goes to
whoever stakes most on it, which makes the agents' game a Colonel Blotto game,
and the best play in such a game is to be unpredictable. Evolution finds that
quickly, through the random inputs every brain reads, and then nothing better
is left to select.

If that is the reason, changing the contest should change the outcome, and
two changes are worth making first. Neither adds a number to the rules, which
since this chapter's predecessors is a rule of its own
([Chapter 57](57-rules-without-numbers.md)).

**A contest without a mixed answer.** Let chance decide instead of the largest
stake. In a **lottery contest** every staked token is a ticket: the node goes
to one of its stakers drawn at random, with a chance in proportion to what it
staked, as the game weighs it. An owner's own stake still counts one and a
half times, as in B2. This is Tullock's contest (1980), the economist's model
of rent seeking and lobbying. Its best play is a plain amount, not a gamble.
Two contestants valuing a prize equally each stake a quarter of its value,
and neither gains by staking otherwise, nor by surprising the other. If evolution stops under Blotto because the answer is to roll dice,
then under a lottery the brains should stop using their random inputs. If
evolution still stops, the reason is more general than Blotto: any contest
whose best play can be found will be found.

**A contest that lets a lineage act as one.** Let stakes count by **genotype**.
Every stake on a node made by agents with exactly the same brain counts as one
side's. The side with the largest total takes the node, and the owner keeps it
if its own genotype wins. Each agent also sees, for every candidate, whether it
carries the same brain. Under B2 a brain is copied whole into a newborn or into
a won node, and changes with probability 0.2 when it is. So a lineage lives as
a patch of identical agents with mutants at its edges. With stakes pooled by
genotype, such a patch is one body: a stake on a clone's node defends it, and
the clones' stakes on a stranger's node are added together. This is the rule
the factories of [Chapter 51](51-strategies-written-by-hand.md) would need,
since in the engine as it is a stake on a teammate's node counts against the
teammate. It is also kin pooling at its narrowest
([Chapter 56](56-alliances.md)): no alliance to agree, and no threshold of
relatedness to choose.

## The worlds

Pilots in the engine's own code (`--proto engine=true`, see
[Pilot worlds](../notes/pilot-worlds.md)): B2 at 10,000 tokens, revolutions
off, 1,000 iterations, two seeds, with a whole world kept every 100
iterations. The control is B2 without revolutions from the pilots of
[Chapter 54](54-the-switches-the-world-was-built-with.md). Revolutions are off
in all three because a revolution is itself a rule about who takes a node, and
would mix with the rule under test.

- **lottery** (`contest=lottery`): the node goes to a staker drawn with a
  chance in proportion to its weighed stake.
- **by genotype** (`pool=genotype`): stakes pooled by genotype, and one more
  input row per candidate, *same brain as mine*.

<!-- turns -->
---

← [Chapter 54 · The switches the world was built with](54-the-switches-the-world-was-built-with.md) · [Contents](../README.md) · [Chapter 56 · Alliances](56-alliances.md) →
<!-- /turns -->
