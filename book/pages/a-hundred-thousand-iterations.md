# A hundred thousand iterations

> [!question] Questions of this page
> - Is time what is missing? What does a world do over a hundred thousand
>   [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), many times longer than the worlds of round 3 have lived?
> - Does a world settle, or does it keep wandering? Does anything appear late
>   that was not there early?
> - Does a candidate world keep its health, and do its [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") keep getting
>   better, or do they stop long before the end?

## In short

This experiment has not been played: it is a plan, a draft for the author to
decide. Three of round 3's shortlisted worlds would be played on from iteration
3,000 to 100,000, keeping a whole world every 2,500 iterations. It asks whether
time is what is missing. Its draft thesis expects that it is not: each world
keeps its health, and its brains stop getting better long before the end, as
the notebook's longest worlds did. It needs nothing new built, and takes about
two and a half days.

## The long run, planned again on round 3's candidates

Since 7 November 2026 only two kinds of world are played: more worlds drawn at
random, and worlds that follow a promising candidate ([What could be played
next](candidates-for-open-ended-evolution.md#what-could-be-played-next)). [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") is
no longer a [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") anything is measured against. So a long run made now would
be one of round 3's candidates, played on. Three things make it far easier than
the plan of 9 October, below.

- **Nothing has to be built.** The [lab](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") can play a kept world on from its last
  state, iteration 3,000, under its own [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") and [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world."). It records every
  iteration as round 3 did, and it keeps whole worlds, brains included, as
  often as the plan asks ([Eras](measuring-evolution.md#eras)). [Frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.") for
  every iteration cost a candidate 0.15 to 0.22 MB an iteration, so no
  windows are needed.
- **The worlds are small and fast.** Round 3's worlds hold 30,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). The
  fastest of the shortlist play an iteration in under a second or two, where
  B1 at 204,800 tokens took 25.
- **The tests exist.** Contests between eras, a shadow world of pure luck and
  the rest ([Did evolution happen?](did-evolution-happen.md)) can read the run
  when it ends.

**The proposal.** Three worlds of the shortlist as it stood on 9 November
2026 (since health also counts [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") on loops per link, seed 3026 has left
it, but it still evolves the most), each played on from
iteration 3,000 to 100,000, keeping a whole world every 2,500 iterations: 39
more eras each.

| world | why this one | health's ranks (product, sum, worst) | generations in its first 3,000 iterations | seconds an iteration | 97,000 iterations more |
|---|---|---|---|---|---|
| seed 3080 | the healthiest of the fast worlds | 4, 19, 4 | 121 | 0.87 | about a day |
| seed 3008 | healthy, and its children act most like their parents (heredity index 0.62) | 7, 28, 8 | 947 | 1.32 | a day and a half |
| seed 3026 | evolves the most: 2,159 generations, and its line never comes back to where it was | 27, 41, 13 | 2,159 | 2.05 | two and a third days |

On three workers side by side that is about two and a half days, and about
60 GB of disk. All three runs already live on the external drive, with 408 GB
free, so they would grow there. The seconds are each world's own, over its iterations 2,000 to 3,000;
the ranks and generations are from
[Which worlds are healthy?](which-worlds-are-healthy.md) and
[Candidates for open-ended evolution](candidates-for-open-ended-evolution.md).

> [!quote] Draft thesis (to be fixed before any run)
> **Claim.** Time is not what is missing in round 3 either. A candidate world
> played on to 100,000 iterations keeps its health, and its brains stop
> getting better long before the end, as the notebook's longest worlds did.
>
> **Why.** The two worlds of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") played to 50,000 iterations climbed for 20,000
> to 25,000 and then only wandered
> ([Every era against three](every-era-against-three.md)). Round 3's worlds
> are smaller still: about fourteen ancestors share each world's future, so
> drift rules what spreads.
>
> **Confirm.** For each world: (1) its health by the product over iterations
> 50,000 to 100,000, scored with round 3's living worlds, lies within 0.1 of
> its health over iterations 100 to 3,000; (2) in lifetime contests played in
> both eras' worlds, the brains of iteration 100,000 hold less than 0.55 of a
> world contested with those of iteration 50,000; (3) the wander levels off,
> as the first part of Experiment 11's thesis asks.
>
> **Refute.** Health moves by 0.1 or more; or the brains of 100,000 hold 0.55
> or more of both worlds against those of 50,000, so evolution still climbs;
> or the wander keeps growing. A world that dies before iteration 52,000
> decides nothing.

**Choices for the author.** Which worlds: these three, others of the
shortlist, or none until Experiment 43 has shown which settings stay healthy
with new seeds. How long: 100,000 iterations, or 25,000 first. And the
lines of the thesis: 0.1 of health and 0.55 of a contest are proposals.

## Experiment 11, as planned on 9 October 2026

The first plan asked whether B1's agents fail to cooperate only because they
lack the time. One world of B1 at 204,800 tokens, seed 1, would be left for
100,000 iterations. It would advance in blocks of 500, with a sentinel that
stops it when something goes wrong and flags it when something stands out, and
keep frames only in short windows. It needed three things the lab could not do
on 9 October; of these, only kept checkpoints have been built since. Contests
then showed that B1's brains learn nothing after their first 50 to 100
iterations, and B1 is no longer a baseline anything is measured against. So the
plan is obsolete. Its plan file is `book/experiments/drafts/E11.json`. The
whole plan of 9 October, with its reasons, method, sentinel and costs, is in
the notebook's history: `git show
9ec755d:book/pages/a-hundred-thousand-iterations.md`. Its thesis stays here,
word for word, because the draft thesis above uses its first part.

> [!quote] The thesis of Experiment 11, written down before any of its runs existed
> **The claim.** Time is not what is missing. A world of 204,800 tokens left for 100,000 iterations, thirty times longer than any world of this book has lived, goes on wandering, but within bounds it reaches in its first few thousand iterations, and nothing appears late that was not there early. Its agents do not learn to cooperate: they stake on kin as they stake on strangers, the same two brains still rarely face each other twice, and the brains stay nearly blind.
>
> **Why it would be so.** Part III found what stands in the way of cooperation in the rules, not in a lack of time ([Chapter 40](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)): a node's brain is replaced almost every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), so nobody meets anybody twice ([Chapter 38](cooperation-help-and-gliders.md#do-agents-cooperate-help-kin-and-partners)); brains cannot see who is kin; and nothing an agent does creates anything. [Chapter 19](what-the-brains-are-like.md) found the brains nearly blind, with a gain of 0.0015 and their first-layer weights already, by iteration 3,000, at the balance between the jitters of [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") and its resets, so that selection can only set constants, and it has set them alike in every world. A hundred thousand iterations are a great deal of evolution: some 27,000 agents, each brain changed with probability 0.2 after every game, try some 500 million changed brains, and the living come to descend from a single [founder](rules-in-detail.md#the-starting-ring "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") every few hundred iterations ([Chapter 18](b1-brains-lineages-and-messages.md#genotypes-and-lineages)), so one line takes the world over again and again, dozens of times at the least. What is one step away will be found; what needs the rules changed will not. Against this: evolution raised the gain 2.5 times in 3,000 iterations ([Chapter 19](what-the-brains-are-like.md)), and the baseline worlds are still moving at the longest scale yet measured: a mean over 500 iterations keeps between a third and two thirds of the spread of single iterations (a Hurst exponent of 0.8 to 0.9, in the 26 worlds of Experiment 2), so no world has yet been seen at rest.
>
> **It holds if:** From iteration 2,000 on, over at least 50,000 iterations, all three hold. (1) The wander levels off: for at least seven of the eight statistics (agents; [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") per agent; [Gini](measuring-tokens.md#the-gini-coefficient-and-the-lorenz-curve "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all."); births per agent; the share of stakes placed at home; the share of nodes kept; genotypes per agent; families), the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") difference between the means of 500-iteration blocks 20,000 to 50,000 iterations apart is at most twice the median difference between blocks 2,000 to 5,000 iterations apart. (2) In no three successive windows from iteration 5,000 on does the ratio of stakes on kin to stakes on others leave 0.85 to 1.15 (in the baseline worlds 0.89 to 1.10), or the share of connections whose two brains face each other again one game later rise above 10% (in the baseline worlds 3% to 9%). (3) At every kept checkpoint the median gain of the brains stays below 0.005 (in the baseline worlds 0.0011 to 0.0017).
>
> **It fails if:** Any one of: for two or more of the eight statistics, the difference at 20,000 to 50,000 iterations is more than twice that at 2,000 to 5,000, so the world goes on moving on scales the shorter runs could not see; or the kin ratio or the share of lasting partners crosses its line in three successive windows, so cooperation has appeared; or the median gain passes 0.005, so the brains have begun to see. A world that dies, or is stopped, before iteration 52,000 decides nothing.

*Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*

## Where to go next

- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick, and why B1 stops learning within its first hundred
  iterations.
- [Eras, contests and invasions](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better): how a world's brains
  are played against their own past.
- [Every era against three](every-era-against-three.md): the longest worlds
  of the notebook, and where their evolution stopped climbing.
- [Do agents cooperate? Help, kin and partners](cooperation-help-and-gliders.md#do-agents-cooperate-help-kin-and-partners): the
  measures this run would take in every window.

## How this was made

No run of either plan exists, so this page has no box of runs. The obsolete
plan of 9 October is `book/experiments/drafts/E11.json`; its working numbers
are in `research/Plans.md`.

## How our view changed

- **9 November 2026.** From 9 October this page planned the run on B1, and
  from 14 October said it would be planned on B2. Since the author's
  decision of 7 November that only random worlds and candidates' worlds are
  played, it proposes a long run of three of round 3's candidates instead
  ([The long run, planned again](#the-long-run-planned-again-on-round-3s-candidates)).
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
  Experiment 16 reached 10,000 iterations, and two of B1 followed as [pilots](runs-and-records.md#pilot-worlds "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")
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

<!-- turns -->
---

← [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md) · [Round 3 of the random screen](../README.md#round-3-of-the-random-screen)

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (6)
> [What this notebook is about](what-this-notebook-is-about.md) · [Brains replaced by their average](brains-replaced-by-their-average.md) · [Every era against three](every-era-against-three.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md)
<!-- /turns -->
