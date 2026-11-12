# Did the brains get better? Round 3's eras against each other

> [!question] Questions of this page
> - Do the later [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") of round 3's best worlds beat their earlier brains?
> - If they do, did they keep getting better to the end of their runs, or
>   stop by the middle?
> - What would tell a real improvement from a contest's own noise?

<!-- experiment E44 -->

## In short

This is a plan, ready to play: Experiment 44, the first of the fourteen tests
of [Did evolution happen?](did-evolution-happen.md), and the one the author
liked best. Every world of round 3 kept its whole state every 250 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."),
brains included. In each of the 21 worlds of round 3's shortlist, the brains
of every era are played against the brains of three fixed eras, 250, 1,500
and 3,000, in the world itself: 1,512 contests of 100 iterations, about two
and a half days on the [lab](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s three workers, after Experiment 43. The thesis, written
before any contest: most worlds got better early and had stopped by the
middle of their runs.

## Why play eras against each other

A world's numbers can settle while its brains still improve, and they can
keep moving while nothing is selected
([Eras, contests and invasions](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better)).
A contest asks the brains themselves. Take the world of one era, give each of
its [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") a brain drawn by coin from two eras, switch [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") off so that
every copy keeps its era, and let the world play 100 iterations under its own
rules. The later era's **score** is the share of the agents carrying its
brains at the end: 0.5 is a tie, more means its brains spread at the earlier
era's expense ([A contest, step by step](measuring-evolution.md#a-contest-step-by-step)).
A hundred iterations is a [lifetime](measuring-lives.md#age-lifetime-and-survival-curves "How many iterations an agent lives, from its birth to its death.") or more in round 3's worlds, where most
agents live a few [games](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.").

Round 3's worlds evolve fast and narrowly: hundreds of generations in 3,000
iterations, but an effective population of about fourteen
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)).
So a line can sweep a world by luck, and a world's numbers alone cannot say
whether its brains got better. The contests can.

## The plan

- **The worlds:** the 21 of round 3's shortlist, every candidate among the
  first 13 of any of health's three orders
  ([The shortlist](candidates-for-open-ended-evolution.md#the-shortlist)),
  each with its own rules and its own kept worlds.
- **The eras:** the brains of iterations 250, 500, …, 3,000, twelve a world.
- **Every era against three:** each era against 250 (did it get better at
  all?), against 1,500 (was it still getting better at the middle?) and
  against 3,000 (how far from the end is it?), as Experiment 28 played the
  notebook's longest worlds ([Every era against three](every-era-against-three.md)).
  Each pair is played twice, once on each era's world, so that neither plays
  only at home; an era's score is the mean of the two.
- **The noise:** each of the three fixed eras is also played against itself,
  four times with different coins. A contest of one era against itself should
  end at 0.5; how far its four end from 0.5 is the noise of a contest in that
  world. A score counts as a win only beyond 0.5 plus twice that spread.

That makes 72 contests a world and 1,512 in all. Contests keep no [frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections."), and
the kept worlds they start from are already on disk, so nothing is played but
the contests themselves.

<!-- thesis E44 -->
> [!quote] The thesis of Experiment 44, written down before any of its runs existed
> **The claim.** Round 3's shortlisted worlds got better early and had stopped getting better by the middle of their runs. In most of the 21 worlds the brains of every later era beat the brains of iteration 250 beyond a contest's own noise; against the brains of iteration 1,500, the eras after it do no better than that noise.
>
> **Why it would be so.** Round 3's worlds evolve fast and narrowly: hundreds of generations in 3,000 iterations, an effective population of about 14, and a line of descent that keeps moving a little beyond the noise (Candidates for open-ended evolution). The notebook's two longest worlds, of B2 with brains mixed at birth, climbed for 20,000 to 25,000 iterations and then only wandered (Every era against three, Experiment 28); round 3's worlds live through generations several times faster, so they should reach their plateau within their 3,000 iterations. Against this: their brains respond far more than B1's (How big should a brain be?), and some lines never return to behaviour they had shown, so some worlds may still climb at the end.
>
> **It holds if:** Both. (1) In at least 14 of the 21 worlds, the median score of the eras 500 to 3,000 against 250 (each era's score the mean of its two contests, one on each era's world) lies above the world's noise band, 0.5 plus twice the standard deviation of its twelve contests of an era against itself (250, 1,500 and 3,000, four coins each). (2) In at least 14 of the 21 worlds, the median score of the eras 1,750 to 3,000 against 1,500 lies within that band.
>
> **It fails if:** Either: (1) in fewer than 11 of the 21 worlds do the later eras score above the band against 250, so most worlds did not get better at all; or (2) in 11 or more of the 21 the eras after 1,500 score above the band against 1,500, so most worlds were still getting better at the middle of their runs. Between the two, the thesis is undecided. A world whose contests cannot be played (its kept worlds unreadable) decides nothing, and is left out of both counts.
<!-- /thesis -->

## What it costs

A contest plays 100 iterations of one world. Played as round 3 recorded them,
the 21 worlds took from 0.6 to 14 seconds an iteration in their last thousand
iterations, 3.1 in the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above."); the worlds that stake a [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") at a time are
among the slowest. All told, about 170 hours of computing: about two and a half
days on the lab's three workers, a little less since contests write no
frames. The lab plays
it after Experiment 43, whose runs come first in its queue.

## Where to go next

- [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md):
  the other tests, and what together would say that evolution happened.
- [Every era against three](every-era-against-three.md): the same tournament on
  the notebook's two longest worlds of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.").
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the 21 worlds, and how they evolve.

## How this was made

The plan is `book/experiments/E44.json`: 21 runs of round 3 (Experiments 40,
41 and 42), named by the shortlist in `research/screen/candidates.json` on
9 November 2026, after health came to count agents on [loops](measuring-networks.md#loops "A path through the network that comes back to where it started without using a connection twice.") per link. One
contest of 20 iterations was played as a check on two of the worlds before
the plan was written, one of them staking a token at a time; nothing else was
played.

<!-- runs E44 -->
> [!info] The trials behind this page
> None: withdrawn unplayed on 12 November 2026, with the author's agreement, after an independent review (11 November): health, which chose its 21 worlds, runs against heredity, generations and fertility in round 3, and the family trees behind that choice counted births only, though conquests copy brains more often. Experiment 45 plays the same tournament with three coins on each era's world, on the worlds the evolution rank puts first and on the healthiest and least healthy.
<!-- /runs -->
