# Does evolution close the door?

> [!question] Questions of this page
> - When a world has evolved for longer, can a raid that once took half of
>   it still take hold?
> - If evolution closes one door, does another open?

<!-- experiment E30 -->

## Why ask

A raid written by hand took half of some evolved [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") worlds at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")
2,000 ([Strategies written by hand](strategies-written-by-hand.md)). Those worlds had
never met a raider. Do worlds that evolve longer get harder to raid?

Three answers are possible, and each says something different about
[open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."):
- **The door stays open.** Later worlds are raided as easily as early ones.
  Evolution does not find the defence any more than it found the raid, so
  better play lies across a valley it does not cross. The next step is to
  ask what would let it cross: a crowd, or a rule.
- **The door closes, and others open.** Later worlds resist this raid but
  lose to others. That is an arms race, which may go round in circles or
  climb.
- **The door closes, and nothing opens.** Later worlds resist every raid
  tried: a top, at least for these [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.").

The way evolution's own contests fade
([Every era against three](every-era-against-three.md)) says which of these to
expect. So does how much a single [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") could gain by its [best reply](exploitability.md "How much better than a world's brains the best reply to them would do: zero only at the top."),
which stays about one node a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") at every [era](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") from 2,500 to 50,000 in
every long world.

This experiment plants raiders, 5% of a world in three places each, in
B2's five worlds played on to 10,000 iterations, at iterations 500,
2,000, 5,000 and 10,000. Each planting has its control.

## The thesis

<!-- thesis E30 -->
> [!quote] The thesis of Experiment 30, written down before any of its runs existed
> **The claim.** Evolution in B2 does not learn to resist the raider. In B2's worlds played on to 10,000 iterations, patches of 5% of raiders take hold as often at iterations 5,000 and 10,000 as at 500 and 2,000: a one-sided Fisher exact test finds no fewer late than early at the 5% level. A planting takes hold as in Experiment 29: after 300 iterations its raiders are alive, more than the patch began with, and more than the control patch of residents holds there.
>
> **Why it would be so.** In every long world the book has, an agent's best reply to the rest would gain about one node a game, as much at 50,000 iterations as at 2,500 (Chapter [95](every-era-against-three.md)): evolution does not close in on any equilibrium. The raider exploits what B2's brains do not do, and nothing in a world that has never met a raider selects against one. Later brains beat earlier ones by less and less, which a world learning to defend itself would not show.
>
> **It holds if:** The plantings that take hold at 5,000 and 10,000 together (30, in five worlds at three places each) are not fewer than at 500 and 2,000 together by a one-sided Fisher exact test at the 5% level.
>
> **It fails if:** Fewer take hold late than early by that test: evolution closed the door. The chapter then asks what opened instead (other raids that later brains lose to) or whether nothing did.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## The trials

The worlds are B2's own: Experiment 16's worlds, played on to 10,000
iterations as [pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") of the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s own rules. Each trial plays from a
pilot's checkpoint, which holds a whole world, as a run's archive does.
The test is the [invasion test](invasion-test.md "A strategy planted as a patch in an evolved world: does it spread, or die out?") of [How large a crowd a raider needs](how-large-a-crowd-a-raider-needs.md).

<!-- runs E30 -->
> [!info] The trials behind this page
> **120 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **raiders** — 60 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 500, 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 60 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 500, 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E30`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

## What the raiders did

<!-- figure door/door -->
![The same raid, planted in older worlds](../figures/door/door.svg)

**The same raid, planted in older worlds.** Left: of the fifteen plantings at each era (five worlds, three places), how many took hold. Right: each planting's raiders as a share of the world after 300 iterations (0 where they died out).

> [!example]- How to make this figure
> **Runs.** Experiment 30's 120 invasions (`python3 gol_lab.py run E30`): B2's worlds of Experiment 16 played on to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`), at iterations 500, 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, once with the raider's rule and once as a control; and, in the second chart, every patch of 5% of Experiment 29 too (Experiment 16's ten worlds at 2,000).
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the raider ([Strategies written by hand](strategies-written-by-hand.md)'s rule), or the control, frozen copies of one patch member's brain.
> 3. Play 300 iterations under the world's own rules; a planting takes hold when its raiders are then alive, more than the patch began with, and more than the control holds.
> 4. A patch's wealth is its tokens over its agents when it is planted.
>
> **To make it again:** `python3 book_figures.py door`.
<!-- /figure -->

**The door closed.**
- **At iteration 500,** all fifteen plantings took hold. The raiders held
  37% to 79% of their world 300 iterations later.
- **At 2,000,** four of fifteen took hold: worlds 2 and 3 again, as in
  [How large a crowd a raider needs](how-large-a-crowd-a-raider-needs.md).
- **At 5,000 and at 10,000,** none of fifteen did. Every patch grew a
  little in its first five iterations, shrank from then on, and was gone by
  about iteration 150.

**But not by a defence.** One planting at iteration 500 and one at 10,000,
in the same world and place, were played again with every decision kept.
Over 120 iterations:
- **At 500,** residents took 3,233 nodes from the raiders and the raiders
  took 7,123 from residents.
- **At 10,000,** residents took 29 nodes from the raiders. The raiders
  rarely raided at all: 470 raids against 11,226 at 500. They had no
  children, and died where they stood, of age.

**What changed is what a raid starts from.**

<!-- figure door/wealth -->
![What a raid starts from](../figures/door/wealth.svg)

**What a raid starts from.** Each dot is one planting of 5% of a world, from Experiments 29 and 30, by the patch's tokens per agent when it was planted and the raiders' share of the world 300 iterations later.

> [!example]- How to make this figure
> **Runs.** Experiment 30's 120 invasions (`python3 gol_lab.py run E30`): B2's worlds of Experiment 16 played on to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`), at iterations 500, 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, once with the raider's rule and once as a control; and, in the second chart, every patch of 5% of Experiment 29 too (Experiment 16's ten worlds at 2,000).
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the raider ([Strategies written by hand](strategies-written-by-hand.md)'s rule), or the control, frozen copies of one patch member's brain.
> 3. Play 300 iterations under the world's own rules; a planting takes hold when its raiders are then alive, more than the patch began with, and more than the control holds.
> 4. A patch's wealth is its tokens over its agents when it is planted.
>
> **To make it again:** `python3 book_figures.py door`.
<!-- /figure -->

The raider stakes only where it is sure to win: 70% of its [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") has to
come to more than one and a half times the [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s. Against a neighbour
of one token it needs three tokens to raid at all. How often a patch of 5%
took hold, by its tokens per agent when planted, over all 120 such
plantings of Experiments 29 and 30:

| tokens per agent | took hold |
|---|---|
| under 2 | 0 of 24 |
| 2 to 2.5 | 8 of 58 |
| 2.5 to 3 | 7 of 16 |
| 3 to 4 | 16 of 16 |
| 4 or more | 6 of 6 |

A patch planted at 500 held a [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of 3.85 tokens per agent, at 2,000
2.35, and at 5,000 and 10,000 2.0. The worlds hold about as many tokens per
agent at 10,000 as at 1,000, 2.7 to 3.3. But a patch, an agent and its
nearest neighbours, is poorer than the average agent, and it got poorer as
the worlds aged. Whether the residents also learned something these
plantings cannot say. Only five late patches were as rich as the 30 at
2,000 that took hold nine times, and none of the five did.

## The verdict

**The thesis fails.** The plantings at 5,000 and 10,000 took hold 0 times
in 30, against 19 times in 30 at 500 and 2,000. By a one-sided Fisher exact
test that is fewer beyond any doubt (p below 0.0001). The door closed.

It closed because the patches got poorer, not, as far as can be told,
because the residents learned to resist a raid. A raid that needs three
tokens a raider cannot start among agents that hold two.

## What it teaches

- **A world of many small agents shuts out every lone raider.** A node goes
  to the largest stake, and a holder's own stake counts one and a half
  times. With a token or two each, no single agent can take anything that
  holds as much as it does.
- **What it does not shut out is a group.** Several raiders who stake on
  the same neighbour add their stakes up. Raiders that each see the same
  stranger as the weakest next door would raid it together without a word
  passing. The poorer the world, the more such a group is worth. This is a
  reason for groups that comes from the game itself, and the next rung of
  the ladder: a crew of raiders against a lone one, in worlds too poor for
  either alone.

> [!info] Added later (2026-10-21, [A crew of raiders](a-crew-of-raiders.md))
> It shut out the group too. Crews planted in the same places took hold
> nowhere, at 2,000, 5,000 or 10,000, and died before the lone raiders in
> 42 of 45 places ([A crew of raiders](a-crew-of-raiders.md)). Seven agents
> in ten of a late patch hold a single token and can stake nothing, a
> stranger borders 1.6 agents of a patch, and a stake that finds no partner
> goes to the stranger. Neither raider can have a child in a late world:
> its rule waits for 30 tokens, which almost no agent there holds.

- **The door has to be tested with the world's own wealth.** A strategy
  written by hand carries its own thresholds, the raider's three tokens
  among them. Before saying that evolution closed a door, the next tests
  will plant patches of the same wealth at every era.


<!-- turns -->
---

← [How large a crowd a raider needs](how-large-a-crowd-a-raider-needs.md) · [Strategies written by hand](../README.md#strategies-written-by-hand) · [A crew of raiders](a-crew-of-raiders.md) →

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (3)
> [A crew of raiders](a-crew-of-raiders.md) · [Raiders that breed](raiders-that-breed.md) · [Hand-written strategies in B1](hand-written-strategies-in-b1.md)
<!-- /turns -->
