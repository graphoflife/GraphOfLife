# Did the brains get better? Round 3's eras against each other

> [!question] Questions of this page
> - Do the later [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") of round 3's worlds beat their earlier brains?
> - If they do, did they keep getting better to the end of their runs, or
>   stop by the middle?
> - Does the evolution rank tell which worlds got better? Does health?

<!-- experiment E45 -->

## In short

This is a plan, ready to play: Experiment 45, the first of the fourteen tests
of [Did evolution happen?](did-evolution-happen.md). It plays 17 worlds of
round 3: the ten the evolution rank puts first
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)),
and the five healthiest and the five least healthy that lived; three are in
two of these groups. In each, the
brains of [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 500 to 3,000 play those of 250 and those of 1,500, in the
world itself. Each pair plays six times, with three coins on each era's world:
1,326 contests of 100 iterations. The thesis, written before any contest:
the brains got better early and had stopped by the middle of the runs, and how
much a world's brains got better follows its evolution rank, not its health.

It replaces Experiment 44, the same tournament with one coin on each era's
world, played on the shortlist chosen by health. Experiment 44 was withdrawn
unplayed ([below](#experiment-44-withdrawn-unplayed)).

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

Round 3's worlds evolve fast and narrowly. Counted on the brain-copy tree,
where conquests make most of the copies, the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world lives through about
2,700 generations in 3,000 iterations, with an effective population of about
100, against about 3,500 in shadow worlds of pure luck
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)).
So a line can sweep a world by luck, and a world's numbers alone cannot say
whether its brains got better. The contests can.

## The plan

- **The worlds:** the ten worlds the evolution rank puts first, leaving out
  those health's filter marks (s3037, eighth, stands still), and the five
  healthiest and the five least healthy of those that lived. A world in two of
  these groups is played once. Each plays with its own rules and its own kept
  worlds.
- **The eras:** 250, 500, 1,000, 1,500, 2,000, 2,500 and 3,000.
- **Two fixed eras:**
  - every later era plays 250: did the brains get better at all?
  - every other era plays 1,500: were they still getting better at the middle?

  Each pair plays on both eras' worlds, three coins each. An era's score is the
  mean of the six.
- **The noise:** 250, 1,500 and 3,000 each play themselves, four coins each.
  A contest of an era against itself should end at 0.5, so the spread of these
  twelve contests is the noise of that world. A score counts as a win only
  above 0.5 plus twice that spread.
- **Does the rank tell?** A world's **gain** is the median score of its eras
  against 250. Across the worlds, the gain is set against the evolution rank's
  score, and against health.

<!-- thesis E45 -->
> [!quote] The thesis of Experiment 45, written down before any of its runs existed
> **The claim.** Round 3's brains got better early and had stopped getting better by the middle of their runs; and how much a world's brains got better follows its place in the evolution rank, not its health. In most of the worlds the brains of every later era beat those of iteration 250 beyond a contest's own noise; against the brains of iteration 1,500, the eras after it do no better than that noise; and the worlds the rank puts first gained the most.
>
> **Why it would be so.** On the brain-copy tree, where conquests make 58% of the copies in the median world, round 3's worlds live through about 2,700 generations in 3,000 iterations, and their ancestry narrows far faster than chance allows: in 69 of the 82 worlds that lived, the living meet one ancestor sooner than in any of 100 shadow worlds of random parents (Candidates for open-ended evolution). Selection on what the brains do is beyond chance almost everywhere, yet the line of descent moves less than mutation alone would move it in 79 of the 82 worlds (Did evolution happen? Fourteen ways to find out): the brains are held in place, as they are once selection has found what it can. The notebook's two longest worlds, of B2 with brains mixed at birth, climbed for 20,000 to 25,000 iterations and then only wandered (Every era against three, Experiment 28); round 3's worlds copy brains several times as often, so they should reach their plateau within their 3,000 iterations. The evolution rank reads what an improvement needs: lines that spread beyond what luck spreads, behaviour new beyond what mutation alone makes, brains copied in proportion to what they do, and children that play like their parents. Health reads the shape of a world, and is unrelated to the rank (rank correlation -0.01). Against this: some lines never return to behaviour they had shown, so some worlds may still climb at the end; and eight of the rank's first ten stake a token at a time, for which two of its four readings are only approximations.
>
> **It holds if:** All three. (1) In at least two thirds of the worlds, the median score of the eras 500 to 3,000 against 250 lies above the world's noise band: above 0.5 plus twice the standard deviation of its twelve contests of an era against itself (250, 1,500 and 3,000, four coins each). An era's score is the share of the agents that hold the later era's brains at the end, the mean of its six contests, three coins on each era's world. (2) In at least two thirds of the worlds, the median score of the eras 2,000 to 3,000 against 1,500 lies within the band, between 0.5 minus and 0.5 plus twice that deviation. (3) A world's gain is the median score of its eras 500 to 3,000 against 250. Across the worlds, the gain rises with the evolution rank's score (research/screen/evolution.json of 12 November 2026): Spearman's rho above 0, with a one-sided permutation p below 0.05 (10,000 permutations); and the gain's rho with health's product is smaller than its rho with the evolution score.
>
> **It fails if:** Any of: (1) in fewer than half of the worlds do the eras 500 to 3,000 score above the band against 250, so most worlds did not get better at all; (2) in half of the worlds or more, the eras 2,000 to 3,000 score above the band against 1,500, so most were still getting better in the middle of their runs; (3) the gain does not rise with the evolution score (rho at 0 or below), or its rho with health's product is larger. Between these, the thesis is undecided. A world whose contests cannot be played decides nothing and is left out of every count.
<!-- /thesis -->

## What it costs

A contest plays 100 iterations of one world. At each world's pace in its last
thousand iterations, the 17 worlds' 78 contests a world take about 303
worker-hours in all. The rank's first worlds are among the slowest: eight of
its ten stake a [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") at a time, at 5 to 19 seconds an iteration. The [lab](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")
gives this experiment two of its three workers at most, so that the third is
free for Experiment 46 and the contests that read it
([Why does evolution level off?](why-does-evolution-level-off.md)): about six
days, a little less since contests write no [frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.").

## Experiment 44, withdrawn unplayed

<!-- experiment E44 -->

Experiment 44 was the same tournament on the 21 worlds of round 3's shortlist
by health. Each pair played once on each era's world, and every era also
played 3,000.

It was withdrawn unplayed on 12 November 2026, with the author's agreement,
after an independent review found two problems:
- In round 3, health runs against heredity, generations and fertility, yet
  health chose these worlds.
- The family trees behind that choice counted births only, though conquests
  copy brains more often
  ([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)).

Its thesis stays as it was written:

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

## Where to go next

- [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md):
  the other tests, and what together would say that evolution happened.
- [Every era against three](every-era-against-three.md): the same tournament on
  the notebook's two longest worlds of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.").
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the evolution rank, and the worlds' brain trees.
- [Why does evolution level off?](why-does-evolution-level-off.md):
  Experiment 46, which changes one thing at a time to find what stops it.

## How this was made

The plan is `book/experiments/E45.json`. `research/screen/evolution_plans.py`
wrote it on 12 November 2026 from two files:
- the evolution rank, `research/screen/evolution.json`, made by
  `research/screen/evolution.py` from what `research/screen/braintree.py`
  read;
- health of 9 November, `research/screen/health.json`.

Nothing was played before the plan was written. Experiment 44's plan,
`book/experiments/E44.json`, is kept as it was, marked withdrawn; before it
was written, one contest of 20 iterations was played as a check on two of its
worlds, one of them staking a token at a time.

<!-- runs E45 -->
> [!info]- The trials behind this page
> **1,326 trials**, contests: the world of one era, its agents given brains drawn by coin from two eras, every mutation off, and the later era's share of the agents counted at the end; each plays 100 iterations, from a world another run kept in its archive.
>
> - **against 250** — 612 trials: worlds `B1-30000-04d3eb-s4007`, `B1-30000-10c9f1-s3047`, `B1-30000-192733-s4002`, `B1-30000-1ad8a9-s3038`, `B1-30000-1cfaa4-s4008`, `B1-30000-26a341-s3093`, `B1-30000-511930-s3020`, `B1-30000-5379e2-s4002`, `B1-30000-625bc7-s3080`, `B1-30000-66c764-s3076`, `B1-30000-77ad6a-s3089`, `B1-30000-8526cf-s4006`, `B1-30000-9efb1d-s4003`, `B1-30000-a8a2d2-s4007`, `B1-30000-af1e28-s3018`, `B1-30000-b82ad0-s3042`, `B1-30000-fffc7d-s4003`; eras 250, 500, 1,000, 1,500, 2,000, 2,500, 3,000; coins 17, 18, 19.
> - **against 1,500** — 510 trials: worlds `B1-30000-04d3eb-s4007`, `B1-30000-10c9f1-s3047`, `B1-30000-192733-s4002`, `B1-30000-1ad8a9-s3038`, `B1-30000-1cfaa4-s4008`, `B1-30000-26a341-s3093`, `B1-30000-511930-s3020`, `B1-30000-5379e2-s4002`, `B1-30000-625bc7-s3080`, `B1-30000-66c764-s3076`, `B1-30000-77ad6a-s3089`, `B1-30000-8526cf-s4006`, `B1-30000-9efb1d-s4003`, `B1-30000-a8a2d2-s4007`, `B1-30000-af1e28-s3018`, `B1-30000-b82ad0-s3042`, `B1-30000-fffc7d-s4003`; eras 500 to 3,000, every 500; coins 17, 18, 19.
> - **each against itself** — 204 trials: worlds `B1-30000-04d3eb-s4007`, `B1-30000-10c9f1-s3047`, `B1-30000-192733-s4002`, `B1-30000-1ad8a9-s3038`, `B1-30000-1cfaa4-s4008`, `B1-30000-26a341-s3093`, `B1-30000-511930-s3020`, `B1-30000-5379e2-s4002`, `B1-30000-625bc7-s3080`, `B1-30000-66c764-s3076`, `B1-30000-77ad6a-s3089`, `B1-30000-8526cf-s4006`, `B1-30000-9efb1d-s4003`, `B1-30000-a8a2d2-s4007`, `B1-30000-af1e28-s3018`, `B1-30000-b82ad0-s3042`, `B1-30000-fffc7d-s4003`; eras 250, 1,500, 3,000; coins 17, 18, 19, 20.
>
> To play these trials again: `python3 gol_lab.py run E45`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- runs E44 -->
> [!info] The trials behind this page
> None: withdrawn unplayed on 12 November 2026, with the author's agreement, after an independent review (11 November): health, which chose its 21 worlds, runs against heredity, generations and fertility in round 3, and the family trees behind that choice counted births only, though conquests copy brains more often. Experiment 45 plays the same tournament with three coins on each era's world, on the worlds the evolution rank puts first and on the healthiest and least healthy.
<!-- /runs -->
