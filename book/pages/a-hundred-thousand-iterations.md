# A hundred thousand iterations

> [!question] Questions of this page
> - Have the agents simply not had the time to learn to cooperate? What does
>   a world do over a hundred thousand iterations, thirty times longer than
>   any world of this notebook has lived?
> - Does a world settle, or does it keep wandering? Does anything appear late
>   that was not there early?
> - How can a run that lasts a month be watched, so that it stops when
>   something goes wrong and says so when something stands out?

> [!warning] Planned, not yet runnable
> The plan, the thesis and the method below were written on 9 October 2026,
> before any run of this experiment existed. The lab cannot run it yet: it
> needs three things it cannot yet do, listed at the end of the page.
> Until then the plan waits in `book/experiments/drafts/E11.json`.

## Why ask

[What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md) ended by saying that measuring the baseline world in every way had turned a vague
hope — that interesting things might emerge if we watch long enough — into
a list of rules that stand in the way. That is a claim about time, and it
has never been tested. The longest any world of this notebook has lived is
3,000 iterations; the largest worlds, those of
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md), lived 600. A
reader may fairly ask whether cooperation is impossible here, or only slow:
perhaps the brains need longer to learn it.

Three findings say the question deserves a month of computing.

- **The brains were still changing.** Evolution raised the brains' gain —
  how much of a change in what a brain sees reaches what it does — 2.5 times
  in 3,000 iterations ([What the brains are like](what-the-brains-are-like.md)). If it
  went on rising, the brains would one day see what is around them, and
  every road to cooperation that needs seeing — recognising kin, answering
  a partner ([Do agents cooperate?](do-agents-cooperate.md)) — would open.
- **No world has been seen at rest.** For this plan, the statistics of the
  26 baseline worlds that lived to the end
  ([A world's life](a-worlds-life.md)) were averaged over blocks of
  iterations, from iteration 500 on. Over blocks of 500 iterations the
  averages still keep between a third and two thirds of the spread of single
  iterations. A quantity that only jittered around a fixed level would keep
  about a twentieth (1/√500 ≈ 0.045). The worlds wander, slowly, at every scale
  that 3,000 iterations can show. (The **Hurst exponent** puts a number on
  this: how the spread of a mean over *B* iterations shrinks with *B*, as
  *B*^(*H* − 1). Noise has *H* = 0.5, a random walk *H* = 1. These worlds
  have 0.8 to 0.9.)
- **Births fall as a world ages**, long after it has settled — by 41% from
  iterations 100–200 to the end in the worlds of 10,000 tokens
  ([A world's life](a-worlds-life.md)) — the clearest sign of an adaptation
  that the study of the baseline saw. An adaptation that takes ten thousand iterations would
  not have been seen at all.

And two things say that nothing will change.

- **The obstacles are rules.** A node's brain is replaced almost every game,
  so the same two brains rarely meet twice; brains cannot see who is kin;
  and nothing an agent does creates anything that was not there before
  ([What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md)). Time does not change a rule.
- **The brains' rise has a ceiling.** By iteration 3,000 their first-layer
  weights already sit at the balance between mutation, which spreads them,
  and the rarer resets, which draw them back ([What the brains are like](what-the-brains-are-like.md)).
  The rise of the gain looks like the approach to that balance, not a
  climb.

> [!info] Added later: what the work on open-ended evolution found
> [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md), written after
> this page, plays the brains of different moments of one world against
> each other. In ten worlds of B1 at 5,000 tokens, nothing is learned after
> the first 50 to 100 iterations: every later era ties with the one before
> it, and the brains of iteration 1,000 hold 0.48 of a world contested with
> those of iteration 300. That speaks for the thesis below, before any run.
> [What the brains are like](what-the-brains-are-like.md), rewritten on the worlds of
> 100,000 tokens, now gives the rise of the gain as 2.4 times in 1,500
> iterations — to the 0.0015 that the worlds of 10,000 tokens reached in
> 3,000. And the work on open-ended evolution has made a second baseline, B2
> ([A world that cannot break](a-world-that-cannot-break.md)), whose brains see: if this
> run is made, it would now be planned on B2.

## How much evolution is a hundred thousand iterations?

A world of 204,800 tokens holds about 27,000 agents. After every game each
brain is changed with probability 0.2, so some 5,400 changed brains are
tried in every iteration, and over 100,000 iterations some 500 million. In
the worlds of 10,000 tokens one branch of the family tree replaced all
others about every 350 iterations
([Genotypes and lineages](genotypes-and-lineages.md)); a world twenty times larger
is slower, by a factor that [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) is
to measure. Even if it took ten times as long, one line would take the
world over again and again, dozens of times. Whatever is one change away
from what the brains do now would be found. What needs a rule changed
would not.

> [!info] Added later: the factor, measured
> [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) has measured it: the time a
> world needs to come down to a single founder's line grows as the tokens to
> the power 0.77. If that holds to 204,800 tokens, a world of that size needs
> about ten times as long as one of 10,000 — as much as this paragraph
> allowed for.

## The thesis

> [!quote] The thesis of Experiment 11, written down before any of its runs existed
> **The claim.** Time is not what is missing. A world of 204,800 tokens left for 100,000 iterations, thirty times longer than any world of this book has lived, goes on wandering, but within bounds it reaches in its first few thousand iterations, and nothing appears late that was not there early. Its agents do not learn to cooperate: they stake on kin as they stake on strangers, the same two brains still rarely face each other twice, and the brains stay nearly blind.
>
> **Why it would be so.** Part III found what stands in the way of cooperation in the rules, not in a lack of time ([Chapter 40](what-the-measurements-say-about-open-ended-evolution.md)): a node's brain is replaced almost every game, so nobody meets anybody twice ([Chapter 38](do-agents-cooperate.md)); brains cannot see who is kin; and nothing an agent does creates anything. [Chapter 19](what-the-brains-are-like.md) found the brains nearly blind, with a gain of 0.0015 and their first-layer weights already, by iteration 3,000, at the balance between the jitters of mutation and its resets, so that selection can only set constants, and it has set them alike in every world. A hundred thousand iterations are a great deal of evolution: some 27,000 agents, each brain changed with probability 0.2 after every game, try some 500 million changed brains, and the living come to descend from a single founder every few hundred iterations ([Chapter 18](genotypes-and-lineages.md)), so one line takes the world over again and again, dozens of times at the least. What is one step away will be found; what needs the rules changed will not. Against this: evolution raised the gain 2.5 times in 3,000 iterations ([Chapter 19](what-the-brains-are-like.md)), and the baseline worlds are still moving at the longest scale yet measured: a mean over 500 iterations keeps between a third and two thirds of the spread of single iterations (a Hurst exponent of 0.8 to 0.9, in the 26 worlds of Experiment 2), so no world has yet been seen at rest.
>
> **It holds if:** From iteration 2,000 on, over at least 50,000 iterations, all three hold. (1) The wander levels off: for at least seven of the eight statistics (agents; connections per agent; Gini; births per agent; the share of stakes placed at home; the share of nodes kept; genotypes per agent; families), the median difference between the means of 500-iteration blocks 20,000 to 50,000 iterations apart is at most twice the median difference between blocks 2,000 to 5,000 iterations apart. (2) In no three successive windows from iteration 5,000 on does the ratio of stakes on kin to stakes on others leave 0.85 to 1.15 (in the baseline worlds 0.89 to 1.10), or the share of connections whose two brains face each other again one game later rise above 10% (in the baseline worlds 3% to 9%). (3) At every kept checkpoint the median gain of the brains stays below 0.005 (in the baseline worlds 0.0011 to 0.0017).
>
> **It fails if:** Any one of: for two or more of the eight statistics, the difference at 20,000 to 50,000 iterations is more than twice that at 2,000 to 5,000, so the world goes on moving on scales the shorter runs could not see; or the kin ratio or the share of lasting partners crosses its line in three successive windows, so cooperation has appeared; or the median gain passes 0.005, so the brains have begun to see. A world that dies, or is stopped, before iteration 52,000 decides nothing.

*Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*

Why "twice"? Between a lag of about 3,500 iterations and one of about
35,000, ten times as far apart, a random walk's differences grow
√10 ≈ 3.2 times, and differences that grow as the baseline's wander does at
shorter scales (*H* ≈ 0.85) about seven times. A world at rest gives
differences that no longer grow at all. Twice lies between.

## Method

**One world.** The baseline B1 at 204,800 tokens, seed 1, for up to
100,000 iterations. One world, because the question is about time within a
world. If something stands out, seeds 2 and 3 are run to the same point,
to tell whether it belongs to this world or to the rules.

Why 204,800 tokens rather than an even 200,000: Experiment 8 already ran
this world, seed for seed, for 600 iterations (`B1-204800-s001`,
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)). The long
run's first frames must be those frames again — a free check that a run is
still the same run a month and several versions of the program later
([Is a run reproducible?](is-a-run-reproducible.md)).

**In blocks of 500.** The lab advances the world 500 iterations at a time.
After each block a **sentinel** reads the block's statistics and asks two
questions.

*Is the run healthy?* If not, the run stops. It can be taken up again from
its last checkpoint, once someone has looked.

| the sentinel stops the run when | because |
|---|---|
| the world dies (20 agents or fewer) | every run ends there ([When a world ends](extinction.md)) |
| the tokens no longer add up to 204,800, or a statistic is missing or not a number where it never was | that is a fault in the program, not a finding |
| an iteration takes more than twice as long as in the ten blocks before, two blocks in a row | something has gone wrong with the machine, or the world has fallen into a shape the program handles badly |
| memory or disk would run out within the next ten blocks | the run would fail half-way through a block |

*Does anything stand out?* That never stops the run: a world doing
something new is what this experiment is for. It is flagged instead.

| the sentinel flags | when |
|---|---|
| **outside the known** | a statistic per agent or a share lies outside the band of the thirty worlds of [Worlds of 100,000 tokens](worlds-of-100000-tokens.md), for three blocks in a row |
| **a new level** | a statistic's block mean lies beyond the range of all its earlier blocks by more than half that range, for five blocks in a row |
| **cooperation** | the kin ratio or the share of lasting partners leaves the baseline worlds' range, in a window (below) |
| **brains that see** | the median gain at a kept checkpoint passes 0.005 |

When something is flagged, the lab keeps the checkpoint of the block before
it, and the flag appears beside this page in the Research Notebook tab and in the
lab's log.

The rule for a new level asks for so much because the worlds wander. For
this plan, a simpler sentinel was tried on the 26 baseline worlds: each
block of 100 iterations from iteration 1,500 on, measured against the
blocks of iterations 500 to 1,499 of the same world, in units of how much
those blocks spread. Six such units sounds safe, yet 5% of all blocks went
beyond it, and runs of three such blocks in a row were common — in the
largest degree, the median degree and the number of leaves above all.
Before the run starts, the rule above is tried on the same worlds; it must
stay quiet there.

**What is kept.** At this size a frame is about 0.65 MB after reproduction
and 1.6 MB after the game, with the agents' decisions: 100,000 iterations
would need more than 200 GB, more than the disk holds. So the run keeps:

- the **statistics** of every frame, as every run does (`stats.jsonl`;
  [What a run records](frames-and-stats.md)): about 400 MB for the
  whole run;
- **frames**, with the decisions, only for the last 17 iterations of every
  block — enough for every measure of [Do agents cooperate?](do-agents-cooperate.md), which looks up to 16
  games ahead: about 8 GB;
- a **checkpoint** — every brain, every token and the state of the random
  stream — at the end of every block, and a copy kept every 10,000
  iterations and before anything that stands out: 0.8 GB each.

Nothing is lost by keeping so little. A run is reproducible
([Is a run reproducible?](is-a-run-reproducible.md)), so any stretch of the world can
be made again exactly, frames and all, from the checkpoint before it. A
frame that was not kept is a frame that has not been made again yet.

**What is measured.**

- **The wander**, as a **variogram**: for each of the eight statistics of
  the thesis, the median difference between the means of two blocks,
  against how far apart in time the blocks are — from 500 iterations to
  50,000. (The idea comes from geology, where it asks how different two
  samples of rock are, as a function of how far apart they were taken.) A
  world at rest gives a curve that levels off; a world still moving, one
  that keeps rising; a change of state, a step.
- **Cooperation**, in every window: the measures of
  [Do agents cooperate?](do-agents-cooperate.md) — how related neighbours are, the
  ratio of stakes on kin to stakes on others, whether a node taken stays in
  the family, how often a stake is answered with an equal one, and how long
  the same two brains face each other.
- **The brains**, at every kept checkpoint: the gain and the probes of
  [What the brains are like](what-the-brains-are-like.md) — what a brain responds to,
  and how much.
- **Everything of [Thirty worlds](thirty-worlds.md)**, as block means over
  the whole run, set beside the bands of the baseline worlds and of the
  worlds of [Worlds of 100,000 tokens](worlds-of-100000-tokens.md).
- **Later, from the kept checkpoints:** the tournament across time of
  [What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md) — the brains of iteration 10,000 against those of
  iteration 90,000, in one world. A later population that beats an earlier
  one is adaptation that accumulates (P2 on the ladder of
  [What this notebook is about](what-this-notebook-is-about.md)).

**What it costs.** The lab's cost model gives about 27,300 agents,
25 seconds an iteration, 5.9 GB of memory, and 29 days for 100,000
iterations on one worker. It keeps about 20 GB, where every frame would
have needed over 200. The thesis can be decided from iteration 52,000,
about two weeks in, and the run can be stopped after any block.

The run should start once [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) is
analysed, because the sentinel's bands of the known come from its thirty
worlds.

## What has to be built first

Three things the lab cannot do yet. Each is a change of its own, tested
before the run.

1. **Frames in windows, statistics always.** Today a run that records its
   statistics as it goes must keep every frame, because families can only
   be counted from every iteration in order ([Families](families.md)).
   The recorder will instead read every frame as it is made, and the store
   keep only the windows. A resumed run then takes its families up from the
   window before its checkpoint, which is why checkpoints fall at the ends
   of windows. This touches the engine (the run loop, the recorder and the
   settings), so it is only made while no experiment runs, and only with
   the proof that a run recorded whole is unchanged. Its test: a run
   recorded in windows has, row for row, the statistics of the same run
   recorded whole, and the frames it kept are that run's frames.
2. **Blocks and a sentinel.** The lab advances a run to the end of its next
   block, then judges it. The health rules stop it; the rules for what
   stands out flag it, keep a checkpoint, and write a line per block beside
   the run (`blocks.jsonl`) and into the Research Notebook tab.
3. **Kept checkpoints.** Copies of a run's checkpoint at chosen iterations,
   which the lab keeps and the analysis can open: for the brains' probes,
   for making a stretch again, and for the tournament.

With them comes an analysis of its own, which draws the variograms, the
windows of cooperation and the gains, and decides the thesis.

> [!info] Added later: what exists now
> Some of this has been built since, for B2's experiments. A run made with
> [`archive_every`](settings.md#archive_every) keeps whole worlds —
> the network, every token and every brain, as a checkpoint holds them — at
> fixed iterations, which is most of the third item; and the tournament
> across time that the kept checkpoints were to serve is the yardstick of
> [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
> (`gol_tournament.py`). [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) is
> analysed, so the sentinel's bands of the known exist. And the costs above
> may be too low: the same cost model gave the worlds of [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) 12
> seconds an iteration, and they took 17. They are also B1's costs; a run on
> B2, whose brains are smaller, would have to be costed again.

## Results

*Planned. Waiting for what has to be built, and then for the run.*

<!-- turns -->
---

← [Every era against three](every-era-against-three.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going) · [What would count as open-ended evolution](open-ended-test.md) →
<!-- /turns -->
