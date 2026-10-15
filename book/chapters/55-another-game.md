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
- **kin sense** (`kin_sense=true`): the input row alone, with no pooling. An
  agent can tell its clones, and nothing in the rules makes it matter. This is
  the free version of the raiders' password in
  [Chapter 51](51-strategies-written-by-hand.md).

## What the contests did

<!-- figure games/fighting -->
![How much changes hands](../figures/games/fighting.svg)

**How much changes hands.** One line per world, means over 50 iterations.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py` in the engine's own code, kept in `GraphOfLifeRuns/.pilots/S`, two seeds each, at 10,000 tokens for 1,000 iterations with a whole world every 100: B2 without revolutions (`--set hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 keep_home=0.3 max_age=100 min_child=10 allow_revolutions=false --proto engine=true`, worlds `S1-norev`), the same with a lottery contest (`contest=lottery`, `G1-lottery`), with stakes pooled by genotype (`pool=genotype`, `G2-genotype`), with only the sense of which neighbours carry one's own brain (`kin_sense=true`, `G3-kinsense`), and with family tags (`tag_rate=0.05`) sensed (`kin_sense=true`, `G4-familysense`) or pooled (`pool=family`, `G5-familypool`).
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Count the nodes whose winner was not their owner in each game, per 100 agents.
>
> **To make it again:** `python3 book_figures.py games`.
<!-- /figure -->

<!-- figure games/groups -->
![How large a group of clones is](../figures/games/groups.svg)

**How large a group of clones is.** One line per world, every 10th iteration: the agents of an agent's connected group of neighbours with the same brain, averaged over agents.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py` in the engine's own code, kept in `GraphOfLifeRuns/.pilots/S`, two seeds each, at 10,000 tokens for 1,000 iterations with a whole world every 100: B2 without revolutions (`--set hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 keep_home=0.3 max_age=100 min_child=10 allow_revolutions=false --proto engine=true`, worlds `S1-norev`), the same with a lottery contest (`contest=lottery`, `G1-lottery`), with stakes pooled by genotype (`pool=genotype`, `G2-genotype`), with only the sense of which neighbours carry one's own brain (`kin_sense=true`, `G3-kinsense`), and with family tags (`tag_rate=0.05`) sensed (`kin_sense=true`, `G4-familysense`) or pooled (`pool=family`, `G5-familypool`).
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Every 10th iteration, join neighbours with the same brain and take the size of each agent's group, averaged over agents (`organisation`, `cloneMean`).
>
> **To make it again:** `python3 book_figures.py games`.
<!-- /figure -->

<!-- figure games/elo -->
![How much better than the founders](../figures/games/elo.svg)

**How much better than the founders.** One line per world, its eras played against each other, four contests per pair.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py` in the engine's own code, kept in `GraphOfLifeRuns/.pilots/S`, two seeds each, at 10,000 tokens for 1,000 iterations with a whole world every 100: B2 without revolutions (`--set hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 valence=8 keep_home=0.3 max_age=100 min_child=10 allow_revolutions=false --proto engine=true`, worlds `S1-norev`), the same with a lottery contest (`contest=lottery`, `G1-lottery`), with stakes pooled by genotype (`pool=genotype`, `G2-genotype`), with only the sense of which neighbours carry one's own brain (`kin_sense=true`, `G3-kinsense`), and with family tags (`tag_rate=0.05`) sensed (`kin_sense=true`, `G4-familysense`) or pooled (`pool=family`, `G5-familypool`).
>
> **Data.** From each world's checkpoints, `ckpt_<iteration>.npz`.
>
> 1. For every pair of the eras 0, 100, 200, 500 and 1,000 play four contests, two in each era's world: every agent gets a brain drawn at random from one era or the other, mutation is switched off, and 20 iterations are played under the world's own rules (`research/oee/pilots/tourney.py --T 20 --reps 2`).
> 2. Fit one rating per era (Bradley–Terry, `research/oee/pilots/elo.py`), the founders at 0.
>
> **To make it again:** `python3 book_figures.py games`.
<!-- /figure -->


**The lottery leaves almost nothing to learn.** Under a lottery the brains of
iteration 1,000 rated +23 and +14 against the founders, against +139 and +100
under Blotto. Each era tied with the one before it almost from the start,
with step scores of 0.47 to 0.56. Every staked token is a ticket, so where it
is staked matters less. More nodes changed hands, 10 and 12 per hundred agents
a game against 9 and 7. The worlds held more agents, about 4,300 against
3,250 to 3,600, and more genotypes, about 2,000 against 1,450 to 1,700.

And the dice stayed. Redrawing the random inputs of brains evolved under the
lottery changed 48 to 67% of their divisions of stakes. Under Blotto it was
59%, in the same world without revolutions. Randomising is worthless under a
lottery, and the brains roll the dice all the same. So the lottery confirms
one half of [Chapter 50](50-why-evolution-stops.md) and refutes the other.
Evolution stops under a lottery too, sooner and lower, but not because brains
learn to roll dice: they roll them anyway.

**Pooling by genotype builds bodies, and peace.** Counting stakes by genotype
cut the fighting to about a third, 3.2 nodes per hundred agents a game against
9.0 and 6.7. It made groups of clones two to three times larger. At the end an
agent's group of clones held 8 and 11 agents on average, the largest 39 and
57, against 3 under Blotto. In one world the groups kept growing, from 7.8
agents at iterations 300 to 400 to 10.6 at 900 to 1,000; in the other they
barely grew, from 7.1 to 8.1. The ratings stopped where Blotto's did: +146 and
+138 by iteration 200, +112 and +121 at 1,000, and 0.49 for iteration 1,000
against 500 in both worlds.

A contest of coins cannot see a body of clones, since it deals brains out
one agent at a time. The **turf** contest can
([Pilot worlds](../notes/pilot-worlds.md)): the era whose world it is keeps
its clones in place. On turf, the brains of iteration 1,000 held 0.35 of the
world against those of iteration 500, in the one world measured so far. In
the same contest without pooling they held 0.45. The bodies grew larger but
not better.

**A sense of one's clones made no groups.** With the kin sense alone, groups
of clones stayed at 3, as under Blotto. In one world the brains improved
slowly to the end, the only steady climb in this chapter. Its steps were 0.70,
0.57, 0.54 and 0.53, and the brains of iteration 1,000 held 0.67 against those
of iteration 100, against 0.54 and 0.55 under Blotto. In the other world they
stopped as under Blotto, with steps of 0.69, 0.55, 0.48 and 0.50. The brains
made little use of the sense. Flipping which neighbours are clones changed 12
to 16% of their divisions, less than it changed the founders' (26%).

## A password and a crowd, for free

The raiders of [Chapter 51](51-strategies-written-by-hand.md) needed a
password and a crowd that knew it. A sense of one's clones gives the password,
but clones seldom sit together, so it brings no crowd. The last pilots give
both. Every brain carries a **family tag**. A copy keeps its parent's tag; a
brain that mutates takes a new tag with probability 0.05 (`tag_rate`), so one
copy in a hundred starts a new family. That probability is a parameter of
mutation, the kind of number the author's rule allows. A family is then every
genotype descended from one tag. It is far larger than a group of clones and
far smaller than a founder's whole line. Two more pairs of worlds ran:

- **family sense** (`kin_sense=true`, `tag_rate=0.05`): one input row per
  candidate, *same family as mine*, and nothing else;
- **pooled by family** (`pool=family`, `tag_rate=0.05`): stakes counted by
  family, as by genotype above, and the same input row.

With the sense alone, families held 30 to 39 agents on average, and after
iteration 500 the largest held 3 to 12% of the world. The fighting stayed as under Blotto, 8
nodes per hundred agents a game. Pooled by family, families grew to 150 and
490 agents on average and the fighting fell to 1.3 to 1.6. In one world a
single family held 57% of the agents at iteration 410, then broke up again
into families of its own as its tags mutated, to 12% at 710 and about 30% at
the end. Families rise and fall there like empires, which is more movement
than one alliance holding the world.

But the brains did not keep improving. Iteration 1,000 against 500 scored
0.48 and 0.54 with the family sense, and 0.43 and 0.52 pooled by family. The
steps after iteration 200 were 0.43 to 0.54. A password and a crowd for free
were not enough. The raiders of Chapter 51 also had their play written for
them, and that is the part evolution would still have to find.

## What it teaches

**Another contest moves where evolution stops, not whether it does.** A
lottery stops it almost where it starts. Pooling by genotype and a sense of
kin stop it where Blotto does. Every one of these worlds kept a network that
holds, with no cut and five or six links per agent. As in B2, their ball
dimension, 4.1 to 4.4, is a little above the bar's 4.

**The mixed strategy was not the reason.** Where dice are worthless, the
brains keep them. Whatever stops evolution in these worlds, it is not that
the brains have found the one unbeatable way to play.

**Bodies are not organisms.** Pooling makes clones fight as one, and the
clones gather into bodies two to three times larger, which keep growing
slowly in one world. But a body of clones whose members do not divide the
work among themselves is only a crowd. The factory of
[Chapter 51](51-strategies-written-by-hand.md) is what an organism would be:
the frontier fights, the inside feeds it. Under pooling the inside's tokens
already defend the frontier, yet nothing better evolved.

**A password and a crowd given for free are not enough.** The raiders of
Chapter 51 won with a password and a crowd that already used it. Given the
password as a sense of clones, evolving brains find no crowd to use it with:
one copy in five is a new genotype, so clones seldom sit together. Given
families large enough to be a crowd, they still did not find the raiders'
play, and stopped improving where the others did. What evolution cannot find
is not the parts but the play itself. Whether a slower rate of change helps is
a question for [Chapter 52](52-how-fast-should-brains-change.md).

<!-- turns -->
---

← [Chapter 54 · The switches the world was built with](54-the-switches-the-world-was-built-with.md) · [Contents](../README.md) · [Chapter 56 · Alliances](56-alliances.md) →
<!-- /turns -->
