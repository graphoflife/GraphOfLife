# Meta VI · Every world side by side

> [!question] Questions of this chapter
> - What does every simulation made so far look like when all of them are
>   measured the same way?
> - Along which directions do the worlds differ most?
> - Which statistics do the settings decide, and which does the seed — the
>   chance a world is born with?
> - Which measures has the book applied to which kinds of world, and which
>   not yet?

Until now each chapter measured the worlds of one experiment, or of a few. That
is how experiments are read: one change against its control. But the book has
by now made **369 lab worlds** in fifteen experiments, and about a hundred
pilot worlds besides, and each was measured only for what its own chapter
asked. This chapter puts them all in one table and looks at them together. It
is a meta chapter: it builds nothing new into the worlds, it gathers what is
there.

## One table for every world

A lab run records about ninety-five statistics after every phase of every
iteration (the note [What a run records](../notes/frames-and-stats.md)). A
pilot world keeps only a few of them, and its checkpoints. To compare them,
`research/atlas/across.py` takes every world over the same stretches of
iterations — 400–499, 900–999, 1,900–1,999, and the last fifth of its life —
and makes one row of means for each:

- a **lab world's** row is the means of its own record over the stretch;
- a **pilot's** is made the same way from frames: the world is played on from
  its checkpoint at 1,000 and from its last, twenty games are recorded as the
  lab records its own, and each game is summarised by the same function the
  lab uses (`gol_series.frame_stats`). So every pilot that plays the engine's
  game gets the lab's statistics too. The older pilots that play the
  prototype's own game record no decisions and keep the statistics of their
  series.

The table is `book/results/worlds.json`. The Research tab can draw any two of
its statistics against each other: **Research → Correlations → Every world**,
each world a dot, coloured by its rule set and named when you point at it.

## Two clouds

Which directions do the worlds differ in most? Take the statistics that do
not grow with the size of a world (shares, means, ratios; the counts of
births, deaths and revolutions divided by the agents), standardise each, and
find the principal components: the directions along which the cloud of
worlds is longest. (The note [How many directions a cloud has](../notes/participation-ratio.md)
explains the idea.)

<!-- figure side_by_side/components -->
![Every lab world on its first two components](../figures/side_by_side/components.svg)

**Every lab world on its first two components.** Each dot is one of the 304 lab worlds that lived to iteration 500, placed by the two directions along which the 60 statistics that do not grow with a world differ most (principal components of their standardised means over iterations 400–499). B1 10k: the thirty B1 worlds of 10,000 tokens (Experiment 2); B1 fixed: the same with brains that never change (Experiment 7); B1 100k: B1 at 100,000 tokens (Experiment 10); B1 sizes: B1 at 800 to 409,600 tokens (Experiment 8) and at 5,000 with and without a hidden layer (Experiments 12 and 13); B2 10k: the ten B2 worlds of Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15 and 16); B2 5k: B2 at 5,000 tokens, with all its rules or with one taken away (Experiment 14) — except lapse: the condition in which any unused link lapses; B1 and B2 pilots: the pilot worlds that play the engine's game.

> [!example]- How to make this figure
> **Runs.** every simulation made so far, as `book/results/worlds.json` holds it (`python3 research/atlas/across.py lab`, `pilots`, `write`): a lab run's statistics are the means of its record over a stretch of iterations; a pilot's, of 20 games recorded from its checkpoint and summarised the same way; the lab worlds that lived to iteration 500.
>
> **Data.** From each world's means over iterations 400–499.
>
> 1. Keep the statistics that do not grow with the size of a world; divide the counts of agents (births, deaths, revolutions, leaves) by the agents.
> 2. Fill a missing value with the median of the worlds; standardise each statistic.
> 3. Take the principal components (a singular value decomposition) and draw every world on the first two.
>
> **To make it again:** `python3 book_figures.py side_by_side`.
<!-- /figure -->

The first component holds a third of all the differences, and it splits the
worlds into **two clouds**: B1 on the right, B2 on the left, with nothing in
between. It is made of how agents play:

- whether they spread their stake over several nodes or go all in
  (`spreadShare`);
- whether the flow cancels or runs one way (`flowImbalance`, `netFlowShare`);
- how much agents keep on their own node, and whether they hold it
  (`selfAllocationShare`, `heldHomeShare`);
- how many revolutions there are, and how much of the world sits in its core.

Every one of the many settings an experiment changed in B2 — mutation rates,
what a parent keeps, old age, random inputs, valence, mending, links at birth,
handing links over — moved its worlds only within the left cloud. Without
mending, and with handover, the worlds moved furthest, about a third of the way
towards the middle. One setting crossed over: the worlds of Experiment 14 in
which **any unused link lapses** (green) sit with B1. [Lightning in every world](72-lightning-in-every-world.md) found the same
in the flow: where a link must carry tokens to survive, agents stake on each
other and play the way B1's agents play.

The second component (14%) is how much a world breeds: births, the share of
tokens given to children, handovers, and the long tail of degrees that
breeding builds. It is spread wide only in one kind of world: the B1 worlds
whose brains never change (yellow), some of which breed every agent every
iteration and others not at all ([Chapter 41](41-do-the-brains-matter.md)).

## What the settings decide, and what the seed does

For each statistic, how much of its spread between worlds lies *between*
conditions rather than between seeds of one condition? That share is
**eta squared**, the same split of variance the note [Between worlds and
within them](../notes/between-and-within.md) explains; here the groups are the
35 conditions of the fifteen experiments.

<!-- figure side_by_side/decided -->
![What the settings decide, and what the seed does](../figures/side_by_side/decided.svg)

**What the settings decide, and what the seed does.** Left: the share of agents that spread their stake over several nodes rather than going all in — 99% of its differences between worlds lie between conditions. Right: births per agent — under a third do; worlds of one condition differ from each other as much as the conditions do. Each dot is a world, over iterations 400–499. B1 10k: the thirty B1 worlds of 10,000 tokens (Experiment 2); B1 fixed: the same with brains that never change (Experiment 7); B1 100k: B1 at 100,000 tokens (Experiment 10); B1 sizes: B1 at 800 to 409,600 tokens (Experiment 8) and at 5,000 with and without a hidden layer (Experiments 12 and 13); B2 10k: the ten B2 worlds of Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15 and 16); B2 5k: B2 at 5,000 tokens, with all its rules or with one taken away (Experiment 14) — except lapse: the condition in which any unused link lapses; B1 and B2 pilots: the pilot worlds that play the engine's game.

> [!example]- How to make this figure
> **Runs.** every simulation made so far, as `book/results/worlds.json` holds it (`python3 research/atlas/across.py lab`, `pilots`, `write`): a lab run's statistics are the means of its record over a stretch of iterations; a pilot's, of 20 games recorded from its checkpoint and summarised the same way.
>
> **Data.** From each world's `spreadShare` and `births` ÷ `nodes`.
>
> 1. Take each world's mean over iterations 400–499.
> 2. Group the worlds as in the other figures; draw a dot per world and the median.
>
> **To make it again:** `python3 book_figures.py side_by_side`.
<!-- /figure -->

| the settings decide | share | the seed decides | share |
|---|---|---|---|
| spread or all in (`spreadShare`) | 0.99 | tokens given to children (`reproTokenShare`) | 0.29 |
| the fewest links (`minDegree`) | 0.96 | agents cut off (`orphaned`) | 0.27 |
| kept at home (`selfAllocationShare`) | 0.95 | tokens per agent (`meanTokens`) | 0.24 |
| brains per agent (`brainDiversity`) | 0.94 | its median (`medianTokens`) | 0.22 |
| own node held (`heldHomeShare`) | 0.91 | curvature of the network (`ricciCurvature`) | 0.22 |
| flow that runs one way (`flowImbalance`) | 0.88 | flow per link (`meanEdgeFlow`) | 0.19 |
| net ÷ staked flow (`netFlowShare`) | 0.87 | net flow on loops (`netCyclingShare`) | 0.16 |
| revolutions per agent | 0.87 | the poorest agent (`minTokens`) | 0.15 |
| evenness of wealth (`tokenEntropy`) | 0.82 | starved per agent | 0.14 |
| agents in the core (`coreShare`) | 0.81 | fit of the degree law (`degreeGammaR2`) | 0.13 |

The settings decide **how agents play**: whether they spread or go all in,
what they keep at home, whether the flow runs one way. Those are nearly
settled by the rules, whatever the seed. The seed decides **how many agents
the tokens are spread over** (tokens per agent, the poorest, the cut-off and
the starved) and how much a world breeds. Two worlds of the same rules can
differ in these as much as two worlds of different rules, because they are
set by what happens early and by chance. The same was seen inside one
condition in [Appendix B](B-what-goes-with-what.md): half the differences
between thirty baseline worlds are the size of their largest hub.

For open-ended evolution this is worth keeping in mind. The things
evolution could change — how agents play — are pinned by the rules. The
things left free are decided by chance, early, and then stay.

## What has been measured where

The book's measures were each made for one question and applied to the worlds
that question was about. Gathered:

| measure | lab worlds | pilot worlds |
|---|---|---|
| the ~95 recorded statistics, lightning among them | every run | the engine's pilots, from recorded games (new); the prototype's, their series |
| flow modules (Chapter 71) | up to ten seeds of every condition (new) | the engine's pilots (new) |
| loops and the exact circulation, chains, help ([Chapter 58](58-gliders-and-who-helps-whom.md)) | ten B1 and ten B2 worlds | the alliance pilots, batches P and Q |
| lifetime contests between eras ([Chapter 50](50-why-evolution-stops.md)) | B2, through the long runs that go on from its worlds | every pilot batch |
| correlations among agents ([Appendix B](B-what-goes-with-what.md)) | ten B1 and ten B2 worlds; now any run, in the Research tab (new) | any run made on the server or in the browser (new) |
| how different the agents are ([Chapter 70](70-how-different-are-the-agents.md)) | B1 and B2, over a world's life (new) | the long runs (new) |
| exploitability ([Chapter 50](50-why-evolution-stops.md)) | Experiment 16 | — |

What is still missing: the lifetime contests for the lab's B1 worlds, whose
runs keep no archives of their brains, and every measure for the worlds of
the many experiments that ran at 5,000 tokens or fewer, which the book no
longer makes.

## What this says

- **There are two kinds of world, not a continuum.** Every lab world is B1-like
  or B2-like in how its agents play, and only one of all the changes tried —
  letting unused links lapse — moved a world from one kind to the other.
- **The rules pin the play; the seed sets the population.** How agents stake is
  decided by the settings; how many agents the tokens are spread over, how
  much a world breeds and how many are cut off are decided by chance, early.
- **Every measure the book has can now be taken of every world** that plays
  the engine's game, lab or pilot, and the table that holds them is in the
  book and in the Research tab.

**Related notes:** [What a run records](../notes/frames-and-stats.md) ·
[Between worlds and within them](../notes/between-and-within.md) ·
[How many directions a cloud has](../notes/participation-ratio.md) ·
[Spread between worlds](../notes/spread-between-worlds.md) ·
[Appendix B · What goes with what](B-what-goes-with-what.md) ·
[Lightning in every world](72-lightning-in-every-world.md)

<!-- turns -->
---

← [Chapter 69 · A hundred thousand iterations](69-a-hundred-thousand-iterations.md) · [Contents](../README.md) · [Chapter 76 · Meta VIII · Who helps whom](76-meta-8-who-helps-whom.md) →
<!-- /turns -->
