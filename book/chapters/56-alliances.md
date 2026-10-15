# Alliances

> [!question] Questions of this chapter
> - Can cooperation pay in a game in which tokens are only ever moved, never
>   made?
> - What is an alliance here, and what does it change in the game?
> - What do alliances do to a world over thousands of iterations?
> - Does a world of allies keep evolving?

## Why ask

[Chapter 50](50-why-evolution-stops.md) ended on a diagnosis. The contest for
nodes is a Colonel Blotto game, its best play is to be unpredictable, and
evolution finds that in a few hundred iterations and then has nothing left to
find. A game that does not run out of answers is one whose payoffs depend on
what the others have become, and the richest of those are games in which
agents can do better *together*: then whom to trust, whom to join and how to
coordinate are questions whose answers change as the population changes.

But cooperation needs a **surplus**. In this world tokens are never made;
every token an agent gives away is a token it no longer has. The two kinds of
help the pilots of Chapter 50 tried were both gifts without a return. Support
counted for less than a stake at home, and the yield on returned stakes went
to whoever had most partners. Evolution gave both up.

There is one place where the game itself makes a surplus, and it is the rule
that makes it a Blotto game: **a node goes to whoever stakes most on it.**
That rule is *convex*: two stakes of 30 each lose against one of 40, but
together, as 60, they win. A group whose members' stakes counted together
could take nodes that none of them could take alone, and hold nodes that none
of them could hold alone. That is the surplus this chapter gives the agents a
way to reach.

## What an alliance is

The pilots of this chapter add one decision and one thing to see
(`research/oee/pilots/proto.py`, `allies=true`).

**The decision.** In every game, an agent says of each neighbour whether it
wants it as an ally: two more output rows per candidate, and *yes* when the
first is larger than the second. Two neighbours that both say yes are
**allied** for that game. Nobody can be made an ally against its will, and an
alliance lasts one game; it is renewed, or not, in the next.

**The contest.** The stakes on a node are counted by **side**. The owner and
every bidder allied with it are one side. The other bidders are grouped by
being allied with each other: two bidders allied with each other, or joined
through a chain of bidders who are, are one side. The side with the largest
total takes the node. If that is the owner's side, the owner keeps it;
otherwise the largest bidder of the winning side takes it. The owner's own
stake still counts one and a half times, as in B2. The tokens go where they
always went, to the node they were staked on. So a stake on an ally's node is
a gift, but one that defends.

**What an agent sees.** One more input row per candidate: was it my ally in
the last game?

Revolutions are off in these worlds. Sides replace them: a revolution was the
one way smaller stakers could count together, and an alliance is a way to do
it by choice.

Two other ways to cooperate ran beside it, as controls:

- **Kin pooled** (`kin_pool=2`): sides are made by descent instead of by
  choice. Two bidders are on one side if their brains are within two mutations
  of each other in the family tree. Nobody decides anything.
- **Support counted twice** (`support=true`, `support_gain=2`): a stake on a
  neighbour can be marked, in part, as support. Support adds to the owner's
  defence and counts twice, which makes it a better use of a token than a
  stake at home.

## The worlds

All of these are pilots: two seeds per kind where an experiment would have
ten, so they show a lead and not a law. Every world runs under B2's rules as
the pilots' prototype states them: births joined to their parent and two of
its neighbours, deaths mended, only redundant links lapse, a valence of 8, a
child of at least 10 tokens, 30% of every agent's tokens kept at home, death
at age 100, a brain copied with mutation only when it is copied, and an
agent's own stake counting one and a half times. Brains have one hidden layer
of 16 and a memory of the last game's stakes. Revolutions are off.

- at 10,000 tokens for 1,000 iterations: no alliances, alliances, kin pooled,
  support counted twice;
- alliances at 10,000 tokens for 3,000 iterations, and at 20,000 tokens for
  2,000;
- alliances without the home advantage, an agent's own stake counting once
  (incumbency 1), at 10,000 tokens for 1,000 iterations.

## One alliance takes the world

<!-- figure allies/bloc -->
![The largest alliance's share of the world](../figures/allies/bloc.svg)

**The largest alliance's share of the world.** One line per world. An alliance is a group of agents joined by connections between allies: two neighbours that both chose, in the last game, to be allied.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py`, two seeds each, kept in `GraphOfLifeRuns/.pilots`: at 10,000 tokens for 1,000 iterations without alliances (`A10k/ctrl-s1`, `-s2`), with them (`A10k/allies-s1`, `-s2`), with kin pooled (`A10k/kin-s1`, `-s2`) and with support counted twice (`A10k/gain-s1`, `-s2`); with alliances for 3,000 iterations (`AL/allies10k-s3`, `-s4`), at 20,000 tokens for 2,000 (`AL/allies20k-s1`, `-s2`), and without the home advantage, incumbency 1, for 1,000 (`N/N0-s1`, `-s2`). Every one under B2's rules as the pilots' prototype states them (`research/oee/pilots/proto.py`: mor=all incumbency=1.5 birth_links=2 repair=true safe_prune=2 max_age=100 valence=8 valence_mode=optional min_child=10 keep_frac=0.3), with one hidden layer of 16, messages of 8 numbers and no pre-pass, revolutions off, and a memory of the last game's stakes (memory=true).
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Every 10th iteration, join every two neighbours that were allied in the game just played (`ProtoWorld.same_side`) and find the largest connected group (`research/oee/pilots/run.py`, `organisation`, `sideLargest`).
> 2. Divide by the agents.
>
> **To make it again:** `python3 book_figures.py allies`.
<!-- /figure -->

The founders have random brains, so each wants about half of its neighbours as
allies, and a quarter to two fifths of all connections are alliances in the
first game. That is enough to join much of a world into one alliance by chance
alone: at iteration 60, the largest alliance held between 16 and 62% of the
agents.

That first alliance is an accident. In some worlds it broke up as evolution
sorted the founders' brains. In the two shorter worlds of 10,000 tokens the
share of neighbours an agent wanted as allies fell to a quarter or a third by
iteration 300, and the largest alliance shrank to 20 and 32% of the world. In
others it grew from the start. Then, in every world with the home advantage,
one alliance grew. The share of neighbours wanted as allies rose to between
0.42 and 0.85 at iteration 1,000, and to 0.99 by iteration 2,000 in the
longest world. The largest alliance grew with it, to 83 and 44% of the world
at iteration 1,000 in the two shorter worlds. In the two longer ones it held
87 and 89% at 1,000, and 99 and 97% at 2,000. In the end almost every
connection is an alliance: 98% of them at iteration 2,000 in the longest
world.

**The alliance spreads as a habit, not as a family.** One might expect it to be
one lineage that took the world, allying only with its own. It is not. In the
longest world at iteration 3,000 the alliance held 99% of the agents. The
largest group of near relatives held 2%: agents joined to neighbours whose
brains are within four mutations of their own. A world whose agents nearly
all say yes to every neighbour is one alliance, whatever their ancestry. The
habit spreads because an agent that says yes is defended by every neighbour
that says yes too, and stands with them when they attack.

<!-- figure allies/fighting -->
![How much changes hands](../figures/allies/fighting.svg)

**How much changes hands.** One line per world, each a mean over 50 iterations.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py`, two seeds each, kept in `GraphOfLifeRuns/.pilots`: at 10,000 tokens for 1,000 iterations without alliances (`A10k/ctrl-s1`, `-s2`), with them (`A10k/allies-s1`, `-s2`), with kin pooled (`A10k/kin-s1`, `-s2`) and with support counted twice (`A10k/gain-s1`, `-s2`); with alliances for 3,000 iterations (`AL/allies10k-s3`, `-s4`), at 20,000 tokens for 2,000 (`AL/allies20k-s1`, `-s2`), and without the home advantage, incumbency 1, for 1,000 (`N/N0-s1`, `-s2`). Every one under B2's rules as the pilots' prototype states them (`research/oee/pilots/proto.py`: mor=all incumbency=1.5 birth_links=2 repair=true safe_prune=2 max_age=100 valence=8 valence_mode=optional min_child=10 keep_frac=0.3), with one hidden layer of 16, messages of 8 numbers and no pre-pass, revolutions off, and a memory of the last game's stakes (memory=true).
>
> **Data.** From each world's `series.jsonl`.
>
> 1. Count the nodes whose winner was not their owner in each game (`conquests`), per 100 agents, and average over consecutive stretches of 50 iterations.
>
> **To make it again:** `python3 book_figures.py allies`.
<!-- /figure -->

**With the alliance, the fighting ends.** Without alliances, 7.6 to 8.0 nodes
in a hundred changed hands in every game from iteration 500 to 1,000. With
them it was 2.2 and 3.4. In the two long worlds it was 1.8 to 1.9, falling to
0.2 and 0.4 in their last 500 iterations. Allies cannot take each other's
nodes, and a world that is one alliance has nobody left to fight. Kin pooling
brought peace sooner and without any decision: 1.8 and 0.6 in a hundred from
iteration 500. Its sides were small, though, 2 to 14% of the world, because
kinship within two mutations reaches only so far.

**Support counted twice was given up anyway.** In the first game the founders
marked about 2,700 tokens as support; by iteration 100 it was 81 in one world
and 175 in the other. In the first world it stayed near 16 to the end. In the
second it came back for a while, to nearly 2,000 tokens a game at iteration
500, and was given up again, to 114 at 1,000. Doubling a gift does not make it
pay if whoever receives it owes nothing back. An
alliance differs from support in exactly that: both ends have chosen, and
each defends the other.

### Without the home advantage

An agent's own stake counts one and a half times in B2, a number chosen by
hand ([Chapter 49](49-who-dies-and-why.md)). Do alliances need it? Not to
form. In two worlds where an agent's own stake counts once, the largest
alliance held 37 and 71% of the world at iteration 1,000, against 83 and 44%
in the two worlds of the same length with the home advantage; two seeds of
each cannot tell those apart. What the home advantage may change is how much
fighting goes on inside and around an alliance: 6.0 and 3.1 nodes per hundred
agents changed hands in a game from iteration 500 without it, against 2.2 and
3.4 with it. One world of the two says that clearly and the other does not.

## Does a world of allies keep evolving?

> [!warning] Added later: these contests were 20 iterations long
> Every tournament in this chapter was played with contests of 20 iterations.
> In B2's worlds such contests miss what pays off over an agent's life of 100
> iterations: in contests of 100 the brains of iteration 2,000 beat those of
> 500 (0.73) where in contests of 20 they tied
> ([Chapter 50](50-why-evolution-stops.md#would-we-notice-if-it-went-on)). Where
> this chapter says that evolution stops, read: it stops *as far as a short
> contest can see*.
>
> The longer contests have since been played, 100 iterations each, once in
> each era's world (`research/oee/checks/rejudge.py`, seed 17). In every one of
> these worlds the brains of iteration 1,000 held more than half of the world
> against those of 500: 0.62 and 0.59 with alliances, 0.70 and 0.62 without,
> 0.80 and 0.59 with kin pooled, 0.69 and 0.97 with support counted twice. In
> the two long worlds of alliances the brains of 3,000 beat those of 1,000
> (0.69 and 0.88), and in one of them those of 2,000 as well (0.69; the other
> tied, 0.52). The same eras played against themselves held 0.44 to 0.56. So a
> world of allies does go on evolving, as far as 3,000 iterations in one world
> of two: the alliance spreads once, and the brains inside it keep improving,
> which the tournaments below were too short to see.

<!-- figure allies/elo -->
![How much better than the founders](../figures/allies/elo.svg)

**How much better than the founders.** One world of each kind, seed 1 (seed 3 for the long one), its eras played against each other two contests per pair in each era's world.

> [!example]- How to make this figure
> **Runs.** Pilot worlds of `research/oee/pilots/run.py`, two seeds each, kept in `GraphOfLifeRuns/.pilots`: at 10,000 tokens for 1,000 iterations without alliances (`A10k/ctrl-s1`, `-s2`), with them (`A10k/allies-s1`, `-s2`), with kin pooled (`A10k/kin-s1`, `-s2`) and with support counted twice (`A10k/gain-s1`, `-s2`); with alliances for 3,000 iterations (`AL/allies10k-s3`, `-s4`), at 20,000 tokens for 2,000 (`AL/allies20k-s1`, `-s2`), and without the home advantage, incumbency 1, for 1,000 (`N/N0-s1`, `-s2`). Every one under B2's rules as the pilots' prototype states them (`research/oee/pilots/proto.py`: mor=all incumbency=1.5 birth_links=2 repair=true safe_prune=2 max_age=100 valence=8 valence_mode=optional min_child=10 keep_frac=0.3), with one hidden layer of 16, messages of 8 numbers and no pre-pass, revolutions off, and a memory of the last game's stakes (memory=true).
>
> **Data.** From each world's checkpoints, `ckpt_<iteration>.npz`, every 100 iterations.
>
> 1. For every pair of eras play four contests, two in each era's world: every agent gets a brain drawn from one era or the other, a coin each, mutation is switched off, and 20 iterations are played (`research/oee/pilots/tourney.py --T 20 --reps 2`).
> 2. Fit one rating per era to the later era's shares (Bradley–Terry, `research/oee/pilots/elo.py`), the founders at 0, in Elo points.
>
> **To make it again:** `python3 book_figures.py allies`.
<!-- /figure -->

The tournaments of [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)
were played in one world of each kind:

| world | rating of the eras against the founders | step scores, each era against the one before |
|---|---|---|
| no alliances | +126, +204, +230, +196 (100, 200, 500, 1,000) | 0.72, 0.62, 0.56, 0.48 |
| alliances | +113, +127, +135, +143 | 0.69, 0.53, 0.51, 0.52 |
| kin pooled | +92, +141, +116, +117 | 0.69, 0.56, 0.48, 0.50 |
| support counted twice | +222, +294, +306, +284 | 0.86, 0.66, 0.56, 0.49 |
| alliances, 3,000 iterations | +178, +198, +190, +209 (500, 1,000, 1,500, 2,000) | 0.74, 0.53, 0.52, 0.52 |

The world of alliances is the only one whose step scores stayed a little
above one half to the end: 0.51 to 0.53, and 0.52 for each of the last three
steps of the long one. Its brains of iteration 1,000 held 0.57 of a mixed
world against those of iteration 100. That is a slow creep, and its ratings
moved by about 10 points per 500 iterations, which is less than a tournament
of this size can tell from nothing.

A contest of coins has a weakness here, though. It gives every agent a brain
from one era or the other at random, so every alliance of the contested world
is cut through the middle, and whatever an era's brains know about standing
together is tested against strangers. A **territorial** contest is fairer to
alliances: each era gets a contiguous half of the world, the half nearest an
agent drawn at random (`tourney.py --layout halves`), and plays 50 iterations.
In it, the brains of iteration 1,000 held 0.41 of the world against those of
iteration 500, in the world of alliances and in the control alike. A **turf**
contest goes one step further: the era whose world it is keeps its own brains
where they are, so its alliances stand as they stood
([Pilot worlds](../notes/pilot-worlds.md)). On turf the brains of iteration
1,000 held 0.39 against those of iteration 500 in the world of alliances, and
0.42 in the control. The later brains lost every way they were played.
Whatever they had become, it was not better at war, alone or together.

## What it teaches

**A rising number is not open-ended evolution.** The share of the largest
alliance rises for a thousand iterations and more, and its ceiling is the
world, so a larger world has further to go. On paper that is the statistic
this part of the book has been looking for. It is the wrong kind of rise. It
is one convention, *say yes to your neighbours*, spreading through a
population until it holds everywhere, as a single advantageous mutation
would. When it has spread, nothing new follows. The population found the
obvious way to survive, as the author of the simulation put it, and that is
not the same as an open-ended process.

**Peace ends selection by conquest.** A world that is one alliance copies
brains almost only by birth, and births balance deaths, which in these worlds
come from the age limit: every agent dies at 100, so one agent in a hundred
dies, and one is born, in every iteration. Whatever selection remains is on
how an agent breeds, and the tournaments show it doing little.

**Cooperation is now possible and pays.** It is the first rule of Part V that
evolution took up rather than gave up. The surplus it reaches is real: pooled
stakes win contests that single stakes lose. What is missing is a reason for
the cooperation to stay *contested*: a way to cheat an alliance that pays for
a while, a cost to being in one, or a limit to how far one can reach, so that
alliances compete with each other instead of merging into one.

**The network holds.** In the last 500 iterations of every world of
alliances, every agent was in the 2-core. In the median frame no agent or
connection could cut off anyone at all, no agent held more than 2% of the
others, and an agent held about five connections (4.9 to 5.5). The worlds
failed the health bar only on its last line: their ball dimension was 4.1 to
4.8, a little above the bar's 4, and highest in the worlds of 20,000 tokens.

> [!info] Kept as a lead
> These are pilots. Two seeds of each kind cannot say how the time an
> alliance takes to fill a world grows with the world. In one world of
> 20,000 tokens the largest alliance held half the agents first at iteration
> 720, in the other at 120. The cost of an alliance, cheating, and alliances
> that stay many are where the next pilots go. So does the question the
> author asked next: what is left of an alliance in a world whose rules have
> no hand-chosen numbers at all ([Chapter 57](57-rules-without-numbers.md)).

<!-- turns -->
---

← [Chapter 55 · Another game](55-another-game.md) · [Contents](../README.md) · [Chapter 57 · Rules without numbers](57-rules-without-numbers.md) →
<!-- /turns -->
