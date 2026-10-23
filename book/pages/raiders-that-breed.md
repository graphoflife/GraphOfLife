# Raiders that breed

> [!question] Questions of this page
> - The raiders written by hand wait for 30 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") before a child, and in a
>   late world almost nobody holds that many. Does a raider that breeds as
>   the world's own [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") do take hold where the barren one could not?
> - Is it the raid that does it, or the breeding alone?

<!-- experiment E34 -->

## In short

A raider written by hand took hold in none of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s late worlds, and it never
had a child: it waits for 30 tokens, which almost no agent there holds.
Experiment 34 gave the raider, and a turtle that never raids, the world's own
way of breeding: a child of ten tokens whenever it holds more. It changed
nothing. The breeding raiders spread a little further and lived a little
longer, but took hold in none of 30 plantings at 5,000 and 10,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."),
just as the barren raiders. The world explains it: only one agent in thirty
holds the eleven tokens a child of ten needs, so a breeding raider can rarely
breed either.

## Why ask

The raider's door closed in B2's late worlds
([Does evolution close the door?](does-evolution-close-the-door.md)), and a
crew of raiders did not open it ([A crew of raiders](a-crew-of-raiders.md)).
Both pages found the same thing beside the raid itself: the raiders had no
children.
- **Their rule:** a child of a third, once the raider holds 30 tokens, as
  [Strategies written by hand](strategies-written-by-hand.md) wrote it.
- **The late worlds:** of the 7,587 agents the patches of
  [A crew of raiders](a-crew-of-raiders.md) were planted on, 4 held that
  many.

A patch that cannot breed lives only as long as the nodes it holds and the
nodes it takes, and in B2 every agent dies at age 100. The crews died with
the youngest agent they were planted on.

The world's own agents breed as soon as they can give a child ten tokens,
B2's least child ([`min_child`](settings.md#min_child)). This experiment gives
the raider the same. A *stranger*, for these strategies, is a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") that
is not one of their own.
- **The breeding raider** raids as the raider does. It stakes everything on
  the weakest stranger next door if seven tenths of its tokens outweigh what
  that stranger holds, counted one and a half times; otherwise everything at
  home. It has a child of ten tokens whenever it holds more, joined to its
  teammates as the raider's children are. Each child can raid at once: seven
  tokens staked take a stranger of up to four.
- **The breeding turtle** breeds the same way and never raids. It keeps
  everything at home. It tells breeding alone from breeding and raiding.

Both are planted where the raiders of
[Does evolution close the door?](does-evolution-close-the-door.md) and
[A crew of raiders](a-crew-of-raiders.md) were: patches of 5% of B2's five
worlds played on, at 2,000, 5,000 and 10,000 iterations, in the same three
places. Beside them are those pages' lone raiders and controls on the same
patches. A control is the patch given frozen copies of one of its own agents'
[brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."): what the patch would do anyway.

Like those pages, this one plays B2's rules only, and its least child of ten
tokens is B2's. [Hand-written strategies in B1](hand-written-strategies-in-b1.md)
plants the same strategies in [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), whose least child is a single token.

## What we expected

We expected breeding to open the door. A raider that breeds replaces its
dead, so its patch need not die with the agents it was planted on, and every
child can raid at once. The breeding turtle would show whether the raid adds
anything to the breeding.

<!-- thesis E34 -->
> [!quote] The thesis of Experiment 34, written down before any of its runs existed
> **The claim.** A raider that breeds as the world's own agents can takes hold where the barren raider could not, and it is the raid that does it. In B2's worlds played on, at iterations 5,000 and 10,000, where patches of 5% of Chapter [51](strategies-written-by-hand.md)'s raider, which has a child only when it holds 30 tokens, took hold in none of 30 plantings, patches of 5% of a raider that has a child of ten tokens whenever it holds more take hold in at least 5 of the same 30, and in more of them than patches of a turtle that breeds the same way but never raids.
>
> **Why it would be so.** In Experiment 33 neither kind of raider had a child: of the 7,587 agents the patches were planted on, 4 held the 30 tokens their rules wait for. A barren patch lives only as long as the nodes it holds and the nodes it takes, and in B2 every agent dies at age 100; the crews died with the youngest agent they were planted on. The world's own agents have a child as soon as they can give it ten tokens (B2's min_child). A raider that does the same replaces its dead, and each of its children can raid at once: seven tokens staked take a stranger of up to four. The breeding turtle breeds the same way and never raids, and so tells breeding alone from breeding and raiding.
>
> **It holds if:** Breeding raiders take hold in at least 5 of the 30 plantings at 5,000 and 10,000, and in more of them than breeding turtles do: alive after 300 iterations, more than the patch began with, and more than the control planted in the same place.
>
> **It fails if:** Breeding raiders take hold in 4 or fewer of the 30, or in no more of them than breeding turtles: breeding does not open the door, or the raid adds nothing to it; the chapter says which.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## What happened

### Breeding opened nothing

<!-- figure breeders/hold -->
![Breeding raiders, beside barren raiders, breeding turtles and controls](../figures/breeders/hold.svg)

**Breeding raiders, beside barren raiders, breeding turtles and controls.** Of the fifteen plantings at each era (five worlds, three places). Left: how many took hold — alive after 300 iterations, with more agents than the patch began with and than the control planted in the same place: breeding raiders (orange), lone raiders (violet), breeding turtles (green). Right: how many were still alive at all, the controls (grey) beside them.

> [!example]- How to make this figure
> **Runs.** Experiment 34's 180 invasions (`python3 gol_lab.py run E34`): B2's worlds of Experiment 16 played on to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the breeding raider's rule, the breeding turtle's, the lone raider's ([Strategies written by hand](strategies-written-by-hand.md)'s) or frozen copies of one patch member's brain (the control); the lone raiders' and the controls' plantings are those of Experiments 30 and 33.
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the patch's strategy's agents and their tokens after every iteration (each trial's `result.json`).
> 3. A planting takes hold when its strategy is alive after 300 iterations, holds more agents than the patch began with, and more than the control planted in the same place; count the plantings of each era that took hold, and those alive at all.
>
> **To make it again:** `python3 book_figures.py breeders`.
<!-- /figure -->

At 2,000 iterations, breeding raiders took hold in 4 of 15 plantings: as many
as the barren raiders. At 5,000 and 10,000 they took hold in none of 30, again
as the barren raiders. The breeding turtles took hold nowhere, at any [era](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?").

A planting takes hold when, after 300 iterations, its strategy is alive, with
more agents than the patch began with and more than the control planted in
the same place. Counted by tokens instead of agents, every number is the same.

The breeding raiders did a little better on the way. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") planting
their patch grew to 1.50, 1.28 and 1.18 times its size at the three eras,
against 1.36, 1.21 and 1.14 for the barren raiders. And it lived a little
longer: its last agent died 243, 166 and 167 iterations after planting,
against 206, 151 and 155. But of the 30 breeding plantings at 5,000 and
10,000, one was alive after 300 iterations.

The breeding turtles lived on, but shrank. After 300 iterations 10, 8 and 6
of their 15 plantings were alive, about as many as of the controls (11, 10
and 7). None ever held more agents than it was planted with.

### Why: almost nobody can afford a child

<!-- figure breeders/patches -->
![What became of the patches planted at 5,000 and 10,000](../figures/breeders/patches.svg)

**What became of the patches planted at 5,000 and 10,000.** The median over the thirty plantings at 5,000 and 10,000 of each kind: breeding raiders (orange), lone raiders (violet), breeding turtles (green), controls (grey). Left: the strategy's agents as a share of the patch planted (0 once it has died out). Right: their tokens per agent, over the plantings still alive.

> [!example]- How to make this figure
> **Runs.** Experiment 34's 180 invasions (`python3 gol_lab.py run E34`): B2's worlds of Experiment 16 played on to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the breeding raider's rule, the breeding turtle's, the lone raider's ([Strategies written by hand](strategies-written-by-hand.md)'s) or frozen copies of one patch member's brain (the control); the lone raiders' and the controls' plantings are those of Experiments 30 and 33.
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the patch's strategy's agents and their tokens after every iteration (each trial's `result.json`).
> 3. At each iteration after planting, take the median over the thirty plantings of the strategy's agents over the patch planted, and of its tokens per agent over the plantings still alive.
>
> **To make it again:** `python3 book_figures.py breeders`.
<!-- /figure -->

A child of ten tokens needs a parent with eleven. In the worlds the patches
were planted in, 3.3%, 3.1% and 3.3% of the agents held that many, at 2,000,
5,000 and 10,000 iterations: one in thirty. The median patch began with about
two tokens per agent (2.26, 2.03 and 1.99). So a breeding raider could almost
never breed, any more than the raider that waits for 30.

The raids do spread the raiders: every node they take gets their rule. But
each raid stakes everything on a poor node, so the raiders spread their
tokens thinner as they spread. Within ten iterations their tokens per agent
fell from about 2 to 1.6. Then the old die at 100, and nothing replaces them.

The turtle shows the other side. It keeps everything at home, so it keeps
its nodes longer, and its patch lives. But it cannot afford children either,
so it shrinks as its agents die of age. The turtles that remain are the
richer ones, since a turtle of one token is the easiest to take: about 6
tokens each after a hundred iterations, and 7.6 after 300.

### Verdict: refuted

The thesis is refuted on both counts. Breeding raiders took hold in none of
the 30 plantings at 5,000 and 10,000, where the thesis asked for at least 5.
And they took hold in no more of them than the breeding turtles, which took
hold in none either. Breeding does not open the door: in B2's late worlds a
raider can no more afford the world's least child than its own.

## What we learned

- **A strategy cannot breed where the world does not let it.** In B2's late
  worlds one agent in thirty can afford a child of ten tokens. A raider that
  breeds as the world's agents do breeds as rarely as they do.
- **A raid spreads a rule, not tokens.** Every node a raider takes gets its
  rule, so its patch grows at first. But it spreads its tokens thinner, and
  its agents die of age unreplaced.
- **Keeping still keeps a patch alive, not growing.** The breeding turtle
  survives about as often as the control, and never gains.
- **The door stays closed in B2's late worlds** to every strategy written by
  hand so far: lone raiders, crews, and now breeding raiders and turtles.

## Where to go next

- [Hand-written strategies in B1](hand-written-strategies-in-b1.md): the same
  strategies, and crews, in B1's long worlds, which keep nothing at home, have
  no old age and let a child start with a single token.
- [Does evolution close the door?](does-evolution-close-the-door.md): when
  the raiders stopped taking hold.
- [A crew of raiders](a-crew-of-raiders.md): raiders that pool their stakes.
- [What holds a world's size](what-holds-a-worlds-size.md): why B2's agents
  are so poor. They breed as soon as they can, and fill their world.

## How this was made

Each figure says, folded beneath it, how it is drawn from the trials;
`python3 book_figures.py breeders` draws them again. The share of agents that
could afford a child is read from the five worlds as they were kept at each
era (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`,
`ckpt_<iteration>.npz`), and written with every other number of this page to
`book/results/breeders.json`.

<!-- runs E34 -->
> [!info]- The trials behind this page
> **180 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **breeding raiders** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies breeding-raider; patches of 5% of the world; places 1, 2, 3.
> - **breeding turtles** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies breeding-turtle; patches of 5% of the world; places 1, 2, 3.
> - **lone raiders** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E34`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- turns -->
---

← [A crew of raiders](a-crew-of-raiders.md) · [Strategies written by hand](../README.md#strategies-written-by-hand) · [Hand-written strategies in B1](hand-written-strategies-in-b1.md) →

> [!info]- Pages that link here (1)
> [Hand-written strategies in B1](hand-written-strategies-in-b1.md)
<!-- /turns -->
