# A hundred thousand iterations

> [!question] Questions of this page
> - Is time what is missing? What do round 3's best worlds do over a hundred
>   thousand [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), about 33 times as long as round 3 played them?
> - Do their [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") keep getting better, or do they stop long before the
>   end, as round 3's did?
> - Does anything appear late that was not there early?

<!-- experiment E48 -->

## In short

Experiment 48 is planned and ready to run: nothing of it has been played. It
plays the rules of the three worlds the evolution rank puts first, s4003 of
Experiment 42 and s4002 and s4008 of Experiment 41, in fresh worlds of 10,000
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") for 100,000 iterations each. While a run goes on, it stops every 5,000
iterations, and its brains play their own past: the brains of 5,000
iterations before, and those of iterations 5,000 and 50,000. Its thesis,
written before any run: the brains still get better in the second half. It
takes about 650 worker-hours, about 13 and a half days on the lab's two
workers, and about 50 GB of disk.

## Why a long run, and why these worlds

Round 3's brains got better early and then stopped. Pooled over 17 worlds,
the later eras beat the brains of iteration 250 in 71% of contests, those of
iteration 500 already in 76%, and the eras after 1,500 beat those of 1,500 in
exactly half
([Did the brains get better?](round-3-eras-against-each-other.md)). The
notebook's two longest worlds, of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), climbed for 20,000 to 25,000 iterations
and then only wandered
([Every era against three](every-era-against-three.md)). Open-ended
evolution would keep finding better brains for as long as it runs. Whether
time is what is missing needs a run long enough to tell.

The worlds are the first three of the evolution rank
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md)):
the worlds whose lines spread furthest beyond chance, whose children play
most like their parents, and whose selection on what the brains do is
clearest. All three play the lottery and have no [revolutions](rules-in-detail.md#how-a-coalition-takes-a-node "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.").

| world | rank | generations in round 3's 3,000 iterations | copies made by conquest | seconds an iteration at 10,000 tokens | 100,000 iterations |
|---|---|---|---|---|---|
| s4003, strongest first (Experiment 42) | 1 | 3,317 | 58% | 6.0 | 167 worker-hours |
| s4002, weakest first (Experiment 41) | 2 | 4,014 | 67% | 6.4 | 177 worker-hours |
| s4008, weakest first (Experiment 41) | 3 | 4,014 | 64% | 4.6 | 128 worker-hours |

The seconds are the lab's cost model for each world's own kind of run
([What a simulation costs](runs-and-records.md#what-a-simulation-costs)).

- **Fresh worlds, not continued.** A world of round 3 played on would keep
  round 3's recording, 30,000 tokens and a whole world every 250 iterations:
  over 100 GB a world. Each run starts anew with [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 4801, at a third of
  round 3's size and with as many [founders](rules-in-detail.md#the-starting-ring "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") per token as round 3 gave it.
- **Can they live at 10,000 tokens?** Nobody knows: none of the three has
  been played at that size. s3058 lived at 30,000 tokens in round 3 and died
  at 10,000 in all three control runs of Experiment 46
  ([Why does evolution level off?](why-does-evolution-level-off.md)). So the
  plan names three reserves, the next worlds by the rank: s4007 weakest first
  (Experiment 41), s4003 weakest first (Experiment 41) and s4007 strongest
  first (Experiment 42). A world that dies out before iteration 10,000 is
  replaced by the next reserve, with the same seed.

## What is kept

- **Every frame.** The brain-copy tree, which the evolution rank reads for
  heredity, novelty and selection, is built from all of a run's frames in
  order. A run that keeps a frame only every few iterations (`frames_every`,
  [Runs and experiments](runs-and-records.md#runs-and-experiments)) could not
  be read that way. Every frame costs a world about 13 to 22 GB over 100,000
  iterations.
- **A whole world every 2,500 iterations**, brains included: 40 eras a run,
  for the readout's contests and for any played later.
- A checkpoint every 100 iterations, and the graph's costly statistics every
  25.

## The readout

At each check, every 5,000 iterations, the run stops and the world kept there
plays ([A readout](measuring-evolution.md#a-readout-contests-while-a-run-goes-on)):

- **the brains of 5,000 iterations before:** are they still getting better
  lately?
- **the brains of iterations 5,000 and 50,000:** how far have they come since
  the early gain, and since the middle of the run?

Each pairing has four coins, and each coin is played twice with the same
deal, once with the later brains in each place. Every coin has a deal of its
own. Odd coins play on the later
era's world, even ones on the earlier era's. A contest lasts 100 iterations,
or until one side has no [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") left
([When one side is gone](measuring-evolution.md#when-one-side-is-gone)).
That is 376 contests a run, 1,128 in all. The lab plays a check's contests
before anything else, so a run waits at its check until they have started.
The lab's panel above draws each run's scores as they come in: a dot for each
check and earlier era, with two standard errors either side, and 0.5 dashed.

A score above 0.5 against the recent past could also come from brains that go
round in circles, each beating the one before it and losing to older ones.
The fixed eras tell the two apart: against brains that only circle, the
scores against iterations 5,000 and 50,000 stay near 0.5.

## How it is read

<!-- thesis E48 -->
> [!quote] The thesis of Experiment 48, written down before any of its runs existed
> **The claim.** Round 3's best brains keep getting better for 100,000 iterations. Played fresh at 10,000 tokens under the rules of the three worlds the evolution rank puts first, in the second half of each run the brains of every check beat the brains of 5,000 iterations before.
>
> **Why it would be so.** In round 3's 3,000 iterations the brains got better early and then stopped: pooled over 17 worlds, the later eras beat the brains of iteration 250 in 71% of contests, those of iteration 500 already in 76%, and the eras after 1,500 beat those of 1,500 in exactly half (Did the brains get better? Round 3's eras against each other). The notebook's two longest worlds, of B2, climbed for 20,000 to 25,000 iterations and then only wandered (Every era against three). Open-ended evolution would keep finding better brains for as long as it runs, so the direct test is a long run of the worlds with the most evolution by the rank: lines that spread far beyond chance, faithful copies and selection on what the brains do (Candidates for open-ended evolution). These three lived through 3,300 to 4,000 generations of brain copies in round 3's 3,000 iterations; 100,000 iterations give them about 33 times as long.
>
> **It holds if:** Pooled over the runs and the ten checks from 55,000 to 100,000, the later brains' mean score against the brains of 5,000 iterations before is above 0.5, and so is the whole of its band, two standard errors either side; each pairing (one check of one run, the mean of its four coins) is one value. A world that dies out before iteration 10,000 is replaced by the next world by the rank, with the same seed, in the order of the reserves; a world that dies out later counts with the checks it reached.
>
> **It fails if:** The band includes 0.5 or lies below it: in the second half the brains no longer beat their recent past, and evolution has levelled off, as in round 3. If fewer than two runs reach iteration 100,000, the thesis is undecided. The scores against the brains of iterations 5,000 and 50,000, which tell steady progress from brains that only go round in circles, are reported beside, as are the brain-copy tree's readings over the whole run; they do not decide.
<!-- /thesis -->

## What it costs

By the lab's cost model, the three runs take about 472 worker-hours, and the
contests at most about 180 more, fewer where they end early: about 650 in
all. That is about 13 and a half days on the lab's two workers, or about 9 on
three. The frames take about 50 GB, and each run's kept worlds up to 2 GB
more.

## The plan of 9 November 2026

Before Experiment 48, this page proposed three worlds of health's shortlist,
seeds 3080, 3008 and 3026, played on from iteration 3,000 to 100,000 under
their own [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") and seeds: about 115 worker-hours and about 60 GB of disk,
with a draft thesis on health and on contests between iterations 50,000 and
100,000. It was never written as a plan. After the review of 11 November,
worlds are chosen by the evolution rank, not by health, and a long run starts
fresh, since a world played on keeps round 3's recording. The proposal and its
draft thesis are in the notebook's history (`git show
85be41c:book/pages/a-hundred-thousand-iterations.md`).

## Experiment 11, as planned on 9 October 2026

The first plan asked whether [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s agents fail to cooperate only because they
lack the time. One world of B1 at 204,800 tokens, seed 1, would be left for
100,000 iterations. It would advance in blocks of 500, with a sentinel that
stops it when something goes wrong and flags it when something stands out, and
keep frames only in short windows. It needed three things the lab could not do
on 9 October; of these, only kept checkpoints have been built since. Contests
then showed that B1's brains learn nothing after their first 50 to 100
iterations, and B1 is no longer measured against. So the plan is obsolete. Its plan file is `book/experiments/drafts/E11.json`. The
whole plan of 9 October, with its reasons, method, sentinel and costs, is in
the notebook's history: `git show
9ec755d:book/pages/a-hundred-thousand-iterations.md`. Its thesis stays here,
word for word.

> [!quote] The thesis of Experiment 11, written down before any of its runs existed
> **The claim.** Time is not what is missing. A world of 204,800 tokens left for 100,000 iterations, thirty times longer than any world of this book has lived, goes on wandering, but within bounds it reaches in its first few thousand iterations, and nothing appears late that was not there early. Its agents do not learn to cooperate: they stake on kin as they stake on strangers, the same two brains still rarely face each other twice, and the brains stay nearly blind.
>
> **Why it would be so.** Part III found what stands in the way of cooperation in the rules, not in a lack of time ([Chapter 40](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)): a node's brain is replaced almost every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), so nobody meets anybody twice ([Chapter 38](cooperation-help-and-gliders.md#do-agents-cooperate-help-kin-and-partners)); brains cannot see who is kin; and nothing an agent does creates anything. [Chapter 19](what-the-brains-are-like.md) found the brains nearly blind, with a gain of 0.0015 and their first-layer weights already, by iteration 3,000, at the balance between the jitters of [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") and its resets, so that selection can only set constants, and it has set them alike in every world. A hundred thousand iterations are a great deal of evolution: some 27,000 agents, each brain changed with probability 0.2 after every game, try some 500 million changed brains, and the living come to descend from a single founder every few hundred iterations ([Chapter 18](b1-brains-lineages-and-messages.md#genotypes-and-lineages)), so one line takes the world over again and again, dozens of times at the least. What is one step away will be found; what needs the rules changed will not. Against this: evolution raised the gain 2.5 times in 3,000 iterations ([Chapter 19](what-the-brains-are-like.md)), and the [baseline](b1-and-b2.md "The stated rules an experiment starts from: B1 or B2. Drawn worlds are stored as changes from B1, but judged against each other.") worlds are still moving at the longest scale yet measured: a mean over 500 iterations keeps between a third and two thirds of the spread of single iterations (a Hurst exponent of 0.8 to 0.9, in the 26 worlds of Experiment 2), so no world has yet been seen at rest.
>
> **It holds if:** From iteration 2,000 on, over at least 50,000 iterations, all three hold. (1) The wander levels off: for at least seven of the eight statistics (agents; [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") per agent; [Gini](measuring-tokens.md#the-gini-coefficient-and-the-lorenz-curve "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all."); births per agent; the share of stakes placed at home; the share of nodes kept; genotypes per agent; families), the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") difference between the means of 500-iteration blocks 20,000 to 50,000 iterations apart is at most twice the median difference between blocks 2,000 to 5,000 iterations apart. (2) In no three successive windows from iteration 5,000 on does the ratio of stakes on kin to stakes on others leave 0.85 to 1.15 (in the baseline worlds 0.89 to 1.10), or the share of connections whose two brains face each other again one game later rise above 10% (in the baseline worlds 3% to 9%). (3) At every kept checkpoint the median gain of the brains stays below 0.005 (in the baseline worlds 0.0011 to 0.0017).
>
> **It fails if:** Any one of: for two or more of the eight statistics, the difference at 20,000 to 50,000 iterations is more than twice that at 2,000 to 5,000, so the world goes on moving on scales the shorter runs could not see; or the kin ratio or the share of lasting partners crosses its line in three successive windows, so cooperation has appeared; or the median gain passes 0.005, so the brains have begun to see. A world that dies, or is stopped, before iteration 52,000 decides nothing.

*Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*

## Where to go next

- [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md):
  Experiment 45, where round 3's brains stopped getting better.
- [Why does evolution level off?](why-does-evolution-level-off.md):
  Experiment 46, one world's rules at 10,000 tokens with one thing changed at
  a time, and why its worlds died.
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the evolution rank that chose these worlds, and the brain-copy tree.
- [A readout: contests while a run goes on](measuring-evolution.md#a-readout-contests-while-a-run-goes-on):
  how the checks are played and scored.
- [Every era against three](every-era-against-three.md): the longest worlds
  of the notebook, and where their evolution stopped climbing.
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick, and why B1 stops learning within its first hundred
  iterations.

## How this was made

The plan is `book/experiments/E48.json`. `research/screen/evolution_plans.py`
wrote it on 22 November 2026 from the evolution rank,
`research/screen/evolution.json`, and its thesis was written into it by hand
before any run. Nothing was played before the plan was written. The obsolete
plan of 9 October is `book/experiments/drafts/E11.json`; its working numbers
are in `research/Plans.md`.

<!-- runs E48 -->
> [!info]- The runs behind this page
> **3 runs.** Every condition below is run once for every seed (4801), for 100,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **s4003 strongest first** — the settings of world 3 of [Experiment 42](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4003), as drawn but with `n_nodes` = 200; 10,000 tokens. Run `B1-10000-085a32-s4801`.
> - **s4002 weakest first** — the settings of world 2 of [Experiment 41](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4002), as drawn but with `n_nodes` = 50; 10,000 tokens. Run `B1-10000-f511a5-s4801`.
> - **s4008 weakest first** — the settings of world 8 of [Experiment 41](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4008), as drawn; 10,000 tokens. Run `B1-10000-1cfaa4-s4801`.
>
> To make these runs again: `python3 gol_lab.py run E48`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

## How our view changed

- **23 November 2026: stopped.** The author stopped Experiment 48 at 17,193
  (s4003), 27,509 (s4002) and 18,301 (s4008) iterations and ruled that no
  experiment may take longer than a day. Its thesis reads the checks from
  55,000 on, which no run reached, so it is not judged. What the readout
  said until then: all three worlds beat their founders by iteration 5,000
  (scores 1.0, 0.87 and 1.0). The eight later checks against the brains of
  5,000 iterations before gave 0.50, 0.25, 0.87, 0.51, 0.63, 0.50, 0.38 and
  0.64, a mean of 0.53, each with a band that includes 0.5. The runs and
  their kept worlds stay on the disk; shorter studies now play from them
  ([Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md)).
- **22 November 2026, before any run.** A reading of the contest code found
  three faults, mended before Experiment 48 started
  ([Measuring evolution](measuring-evolution.md#how-our-view-changed)): a
  contest whose world died out counted as a win, the coins of every check
  shared four deals, and the panel's band was too narrow. The brain-copy
  tree could not hold more than 16,383 iterations and now can. One limit
  stays: s4002 mixes two brains at every birth, so in its contests a child
  counts for its parent's era but carries half a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s brain, often of
  the other era. That pulls its scores towards 0.5. Its readout is the
  weakest of the three, and it is reported beside the other two as well as
  pooled with them.
- **22 November 2026.** From 9 November this page proposed three worlds of
  health's shortlist, played on from iteration 3,000. Since the review of
  11 November, worlds are chosen by the evolution rank, and a long run starts
  fresh rather than continuing a world of round 3. So the page now carries
  Experiment 48, the long run that plan asked for, with a readout built for
  it. The proposal of 9 November is cut to a paragraph; with its draft thesis
  it is in the notebook's history. The plan of 9 October and its thesis are
  kept as they were written.
- **9 November 2026.** From 9 October this page planned the run on B1, and
  from 14 October said it would be planned on B2. Since the author's
  decision of 7 November that only random worlds and candidates' worlds are
  played, it proposes a long run of three of round 3's candidates instead
  ([The long run, planned again](#the-plan-of-9-november-2026)).
  The plan of 9 October and its thesis are kept as they were written.
- **14 October 2026.** From 9 October, this page planned the run on B1,
  then the notebook's only baseline. A note added on 14 October said that if
  the run is made, it would be planned on B2. Contests between eras had
  shown that B1's brains learn nothing after their first 50 to 100
  iterations
  ([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).
  And B2 had been built, a baseline whose brains see
  ([Building B2: a world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break)).
- **24 October 2026.** From 9 October, this page said that the longest any
  world of the notebook had lived was 3,000 iterations. Its questions called
  a hundred thousand "thirty times longer than any world". That held until
  15 October 2026. That night the first of B2's worlds played on from
  Experiment 16 reached 10,000 iterations, and two of B1 followed as pilots
  ([Every simulation](every-simulation.md#the-pilot-batches)). By 17 October
  two worlds of Experiment 18 had lived 50,000
  ([Five mixed worlds, fifty thousand iterations](building-b2-and-the-search-for-a-healthy-rich-world.md#five-mixed-worlds-fifty-thousand-iterations)).
  This rewrite corrected the page; the thesis keeps the words of its day.
- **24 October 2026.** From 9 October, this page said that no world had
  been seen at rest. It read the Hurst exponent of 0.8 to 0.9 as a wander
  "at every scale that 3,000 iterations can show". Its reason for "twice"
  added that differences growing as that wander does (*H* ≈ 0.85) would
  grow about seven times over a tenfold lag. Both went too far. The exponent
  was measured over blocks of 10 to 500 iterations only. Over longer
  stretches, the spectrum of the same worlds' number of agents levels off,
  from periods of 800 iterations on. A world is pulled back to its level,
  as a page of the day before had found
  ([Power laws, real and apparent](b1-the-network-size-and-scaling.md#the-rhythm-of-the-wander)).
  And by the measure this page uses, a random walk has *H* = 1, so 0.8 to
  0.9 is a wander around a level. Its differences level off: about 1.1 times
  over a tenfold lag, not seven. The threshold of twice still lies between a
  world at rest (1) and a random walk (3.2), and the thesis keeps its words.
  This rewrite found both slips.
- **11 November 2026.** The plan of 9 October for B1 is obsolete, so this page
  now leads with the plan of 9 November on round 3's candidates and keeps the
  first plan as its thesis, word for word, and one paragraph. Its reasons,
  method, sentinel, costs and what had to be built are in the notebook's
  history (git show 9ec755d:book/pages/a-hundred-thousand-iterations.md).
- **12 November 2026.** The section Why "twice"?, which explained the
  threshold of Experiment 11's first test, is cut: block means 2,000 to 5,000
  iterations apart are about 3,500 apart, and 20,000 to 50,000 apart about
  35,000. Over that tenfold lag the differences grow 1 time for a world at rest,
  √10 ≈ 3.2 times for a random walk, and about 1.1 times for a wander like
  B1's, so twice lies between. It is in the notebook's history (git show
  c09615c:book/pages/a-hundred-thousand-iterations.md). The section Results,
  which said that none of the runs had been played, is cut too.
- **12 November 2026.** The table gave health's ranks (product, sum, worst) of
  an earlier order: 4, 19, 4 for seed 3080; 7, 28, 8 for seed 3008; and 27,
  41, 13 for seed 3026. The order of 9 November reads 4, 10, 4; 10, 20, 10;
  and 23, 36, 17 ([Which worlds are healthy?](which-worlds-are-healthy.md)).
  The time on the lab's workers is now given in worker-hours. On 9 November
  the external drive the three runs live on had 408 GB free.

<!-- turns -->
---

← [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md) · [Round 3 of the random screen](../README.md#round-3-of-the-random-screen)

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (9)
> [What this notebook is about](what-this-notebook-is-about.md) · [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md) · [Brains replaced by their average](brains-replaced-by-their-average.md) · [Every era against three](every-era-against-three.md) · [Why does evolution level off?](why-does-evolution-level-off.md) · [Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md)
<!-- /turns -->
