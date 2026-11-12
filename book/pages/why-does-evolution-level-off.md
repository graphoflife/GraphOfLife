# Why does evolution level off?

> [!question] Questions of this page
> - Why have the worlds measured so far stopped getting better?
> - Is it the size of the population, copies that are not faithful, or the
>   [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") itself?
> - Which one change to the rules lets evolution go on longer?

<!-- experiment E46 -->

## In short

This is a plan, ready to play: Experiment 46. The worlds measured so far got
better for a while and then stopped: the notebook's two longest worlds, of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."),
climbed for 20,000 to 25,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") and then only wandered
([Every era against three](every-era-against-three.md)). Experiment 45 asks
the same of round 3
([Did the brains get better?](round-3-eras-against-each-other.md)). Four
causes could stop it, and each points to a different fix. This experiment takes the
rules of one world of round 3, s3058. It plays them at 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") once as
they are, and five more times with one thing changed. Each arm plays three
[seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") for 20,000 iterations and keeps its whole world every 500. Experiment 47
then plays each run's eras against each other, to find where it levelled off.

The thesis, written before any run: evolution stops because of the game, not
because of the population. Four times the tokens and [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that change half
as often level off no later than the control; at least one of the three
changes to the game levels off later.

## The world

s3058 (`B1-30000-4108dd-s3058`) is 28th of the 82 worlds of round 3 that lived
by the evolution rank
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)),
and is on health's shortlist. It is the best-ranked world for which every arm
changes something and whose runs fit in about four days on one worker:
- It plays with [revolutions](rules-in-detail.md#how-a-coalition-takes-a-node "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake."), so revolutionary tokens can be made to cost
  something. Nine of the rank's first ten do not.
- Its brains change with a probability of 0.2, so they can change less often.
  The 10th and 25th worlds change theirs with 0.1, the least round 3 drew.
- Two better-ranked worlds would take 142 and 515 worker-hours.

On its brain-copy tree, conquests make 80% of the copies, and the living meet
one ancestor 224 iterations back, against 2,356 in its shadow worlds of pure
luck.

## Four reasons it could stop

- **Drift.** In a small population chance decides which lines spread, and
  selection sees only large differences. Counted on the brain-copy tree, the
  [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world of round 3 has an effective population of about 100, against
  about 3,500 in its shadow worlds; s3058 has 70.
- **Unfaithful copies.** If a child's brain plays unlike its parent's, what
  selection found is lost again as fast as it is found. Round 3's worlds whose
  brains change more often copy them less faithfully: the median heredity is
  0.99 at a probability of 0.1, 0.97 at 0.2, 0.93 at 0.4 and 0.74 at 0.8.
- **Valleys.** Better play may need a crowd to start: a raider needed about a
  hundred of its kind together
  ([Can a strategy written by hand beat evolution?](strategies-written-by-hand.md)).
  A single new brain cannot cross the valley alone.
- **A narrow game.** A brain scores each [neighbour](the-world.md#the-network "An agent joined to another by a connection.") from that neighbour's
  column alone, so it cannot compare neighbours
  ([What a brain can compute](what-a-brain-can-compute.md)). And flagging a
  token as revolutionary costs nothing, so every brain can make the same
  free move. Once the few good moves are found, nothing is left to find.

## The arms

| Arm | The change | The cause it tests |
|---|---|---|
| Control | none | |
| 4 times the tokens | 40,000 tokens on 800 [nodes](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), against 10,000 on 200 | drift: a population four times larger |
| Faithful copies | brains change half as often: `mutation_probability` 0.1, the least round 3 drew, against 0.2 | unfaithful copies |
| The other [message](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") to oneself | `self_message` = largest, against own | a narrow game: what a brain can tell its neighbours |
| Revolutionary tokens for the coalition alone | `revolt_tokens` = separate | a narrow game: the free flag |
| Stakes pooled by family | `pool` = family, a new family on one [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") in twenty (`tag_rate` 0.05, as in the [pilots](runs-and-records.md#pilot-worlds "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")) | valleys: kin stake as one |

Every change is a setting the [engine](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") already has, and none needs a number
tuned for it ([Every setting](settings.md)). The family-pool arm was to wait
for a second wave, until family pools were in the engine; they are now
([`pool`](settings.md#pool)), and a contest keeps each brain's family.

## How it is read

When the runs are done, their kept worlds play contests, as the eras of round 3
do ([Did the brains get better?](round-3-eras-against-each-other.md)), as
Experiment 47. In every run:

- **Steps:** the brains of 1,000 against those of 500, then 2,500 against
  1,000, 5,000 against 2,500, 10,000 against 5,000 and 20,000 against 10,000.
  Each step plays on both eras' worlds with three coins each, and its score
  is the mean of the six.
- **The noise:** 1,000, 5,000 and 20,000 each play themselves, four coins each.
  A step above 0.5 plus twice the spread of these twelve is a **step up**.
- **Where it levelled off:** the earliest of the eras 500, 1,000, 2,500, 5,000
  and 10,000 after which no step is a step up. A run whose last step, from
  10,000 to 20,000, is still a step up has not levelled off by 20,000.
- **Patches:** the last step is also played with each era's brains planted as
  one region of the world, not by coin, so that kin who stake as one start
  together ([A contest, step by step](measuring-evolution.md#a-contest-step-by-step)).

An arm levels off where the middle one of its three runs does.

<!-- thesis E46 -->
> [!quote] The thesis of Experiment 46, written down before any of its runs existed
> **The claim.** Evolution stops because of the game, not because of the population. Under the rules of round 3's world s3058 at 10,000 tokens, evolution has levelled off by 5,000 iterations. Four times the tokens, and brains that change half as often, level off no later than the control does. At least one of the three changes to the game, the other message to oneself, revolutionary tokens that count for the coalition alone, or stakes pooled by family, levels off later.
>
> **Why it would be so.** On round 3's brain-copy trees conquests make most copies (58% in the median world, 80% in s3058), and the living meet one ancestor far sooner than in shadow worlds of random parents: 224 iterations back in s3058, against 2,356 in its shadows. The effective population is small because winners sweep, not because the world is small (70 in s3058, against 743 in its shadows), so a world four times larger should sweep as fast. Children already play like their parents (heredity 0.96 in s3058 and in the median world), so brains that change less have little to save. Selection on what the brains do is strong almost everywhere, yet the behaviours do not follow it, and the line of descent moves less than mutation alone would move it in 79 of round 3's 82 worlds (Did evolution happen? Fourteen ways to find out): the brains are held where they are. Better play exists that a single new brain cannot reach: in young worlds of B2, strategies written by hand took over from evolved brains, but only with about a hundred of their kind starting together (Can a strategy written by hand beat evolution?). A brain scores each neighbour from that neighbour's column alone and cannot compare neighbours (What a brain can compute), and flagging a token as revolutionary costs nothing, so every brain can make the same free move. These are limits of the game. A change that lets a brain tell its neighbours about a third agent, gives the flag a price, or lets kin stake as one should leave more to be found.
>
> **It holds if:** All three, with every arm read as How it is read says, in Experiment 47's contests; an arm levels off where the middle one of its three runs does. (1) The control has levelled off by 5,000. (2) The arm with four times the tokens and the arm whose brains change half as often each level off no later than the control. (3) At least one of the arms with the other message to oneself, with revolutionary tokens for the coalition alone, and with stakes pooled by family levels off later than the control.
>
> **It fails if:** Any of these. (1) The control has not levelled off by 10,000: its step from 10,000 to 20,000 is still a step up. (2) The arm with four times the tokens, or the arm whose brains change half as often, levels off later than the control. (3) None of the three arms that change the game levels off later than the control. If the control levels off by 10,000 but not by 5,000, the thesis is undecided. A run that dies out, or whose kept worlds cannot be played, decides nothing; an arm with fewer than two runs left decides nothing.
<!-- /thesis -->

## What it costs

At s3058's own pace, the 18 runs take about 34 worker-hours, one at a time:
about a day and a half. The lab keeps them to one worker, so that Experiment
45's contests go on beside them
([Did the brains get better?](round-3-eras-against-each-other.md)).
Experiment 47's 864 contests then take about 8 worker-hours more, as the lab's
model reckons them.

## Where to go next

- [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md):
  Experiment 45, whether round 3's worlds levelled off, and when.
- [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md):
  the tests this experiment draws on.
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the evolution rank, and the brain trees of round 3.
- [A source and a sink](a-source-and-a-sink.md): an open economy, the next
  change to the game if no arm here moves the plateau.

## How this was made

The plans are `book/experiments/E46.json` and `E47.json`.
`research/screen/evolution_plans.py` wrote them on 12 November 2026 from the
evolution rank, `research/screen/evolution.json`. Nothing was played before
the plans were written. Experiment 47 can be played only once Experiment 46's
runs are done: its contests start from their kept worlds.

<!-- runs E46 -->
> [!info]- The runs behind this page
> **18 runs.** Every condition below is run once for every seed (4601 to 4603), for 20,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **control** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-122402-s4601` … `B1-10000-122402-s4603`.
> - **4 times the tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-bbbccd-s4601` … `B1-40000-bbbccd-s4603`.
> - **faithful copies** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_probability` = 0.1, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-0ff18b-s4601` … `B1-10000-0ff18b-s4603`.
> - **message to oneself: largest** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `self_message` = largest, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-4034e2-s4601` … `B1-10000-4034e2-s4603`.
> - **revolutionary tokens for the coalition alone** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `revolt_tokens` = separate, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-97e9c3-s4601` … `B1-10000-97e9c3-s4603`.
> - **stakes pooled by family** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `pool` = family, `tag_rate` = 0.05, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-4819e3-s4601` … `B1-10000-4819e3-s4603`.
>
> To make these runs again: `python3 gol_lab.py run E46`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

<!-- runs E47 -->
> [!info]- The trials behind this page
> **864 trials**, contests: the world of one era, its agents given brains drawn by coin from two eras, every mutation off, and the later era's share of the agents counted at the end; each plays 100 iterations, from a world another run kept in its archive.
>
> - **each era against the one before** — 540 trials: worlds `B1-10000-0ff18b-s4601`, `B1-10000-0ff18b-s4602`, `B1-10000-0ff18b-s4603`, `B1-10000-122402-s4601`, `B1-10000-122402-s4602`, `B1-10000-122402-s4603`, `B1-10000-4034e2-s4601`, `B1-10000-4034e2-s4602`, `B1-10000-4034e2-s4603`, `B1-10000-4819e3-s4601`, `B1-10000-4819e3-s4602`, `B1-10000-4819e3-s4603`, `B1-10000-97e9c3-s4601`, `B1-10000-97e9c3-s4602`, `B1-10000-97e9c3-s4603`, `B1-40000-bbbccd-s4601`, `B1-40000-bbbccd-s4602`, `B1-40000-bbbccd-s4603`; eras 500, 1,000, 2,500, 5,000, 10,000, 20,000; coins 17, 18, 19.
> - **the last step, in patches** — 108 trials: worlds `B1-10000-0ff18b-s4601`, `B1-10000-0ff18b-s4602`, `B1-10000-0ff18b-s4603`, `B1-10000-122402-s4601`, `B1-10000-122402-s4602`, `B1-10000-122402-s4603`, `B1-10000-4034e2-s4601`, `B1-10000-4034e2-s4602`, `B1-10000-4034e2-s4603`, `B1-10000-4819e3-s4601`, `B1-10000-4819e3-s4602`, `B1-10000-4819e3-s4603`, `B1-10000-97e9c3-s4601`, `B1-10000-97e9c3-s4602`, `B1-10000-97e9c3-s4603`, `B1-40000-bbbccd-s4601`, `B1-40000-bbbccd-s4602`, `B1-40000-bbbccd-s4603`; eras 10,000, 20,000; coins 17, 18, 19.
> - **each against itself** — 216 trials: worlds `B1-10000-0ff18b-s4601`, `B1-10000-0ff18b-s4602`, `B1-10000-0ff18b-s4603`, `B1-10000-122402-s4601`, `B1-10000-122402-s4602`, `B1-10000-122402-s4603`, `B1-10000-4034e2-s4601`, `B1-10000-4034e2-s4602`, `B1-10000-4034e2-s4603`, `B1-10000-4819e3-s4601`, `B1-10000-4819e3-s4602`, `B1-10000-4819e3-s4603`, `B1-10000-97e9c3-s4601`, `B1-10000-97e9c3-s4602`, `B1-10000-97e9c3-s4603`, `B1-40000-bbbccd-s4601`, `B1-40000-bbbccd-s4602`, `B1-40000-bbbccd-s4603`; eras 1,000, 5,000, 20,000; coins 17, 18, 19, 20.
>
> To play these trials again: `python3 gol_lab.py run E47`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- turns -->
---

← [Every era against three](every-era-against-three.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going) · [A source and a sink](a-source-and-a-sink.md) →

**Also about:** [Changing the rules](../README.md#changing-the-rules).

> [!info]- Pages that link here (1)
> [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md)
<!-- /turns -->
