# The first hundred iterations

[Chapter 10](10-a-worlds-life.md) found that every world begins with a
**youth** unlike the rest of its life. This chapter looks at it closely.

> [!question] Questions of this chapter
> - How do a thousand founders, each with a random brain and a hundred tokens,
>   become a world of more than ten thousand agents?
> - Where do the agents come from in the youth, and how do they die?
> - Whose descendants are the agents of a settled world?

The worlds are the thirty of 100,000 tokens of [Chapter 9](09-thirty-worlds.md).
Their youth is the youth of the worlds of 10,000 tokens this chapter was first
written from, ten times over and a little faster; where it differs, it says so.

## A boom and a crash

<!-- figure youth/boom -->
![Agents in the first 150 iterations](../figures/youth/boom.svg)

**Agents in the first 150 iterations.** Each thin blue line is one of the 30 worlds: the number of agents alive after every game. The yellow line is their median. The axis is logarithmic, so equal heights are equal factors: the step from 100 to 1,000 is as tall as the one from 1,000 to 10,000.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. Take every run's rows with `phase` = 2 and 0 ≤ `iteration` ≤ 150.
> 2. Draw `nodes` against `iteration` for each run.
> 3. At each iteration, take the median of `nodes` over the runs that have a row there.
>
> **To make it again:** `python3 book_figures.py youth`.
<!-- /figure -->

At the start every founder holds a hundred tokens and has a brain nobody has
selected. In the first reproduction phase most of them give a large part of
their tokens to a child ([Chapter 4](04-the-brain.md): a median of half). The
children give in turn, and the median world grows from 1,000 founders to 1,370
agents after the first game (that of iteration 0), 4,050 after the game of
iteration 5, 12,480 after that of iteration 10 and 16,200 after that of
iteration 20. The median of the thirty worlds peaks at about 17,400 agents
around iteration 39, falls back, and is near 15,000 at iteration 100.

The axis is logarithmic. On it, steady growth by a constant factor per
iteration is a straight line, and the first ten or so iterations are close to
one: the population doubles about every three iterations — a little faster
than the worlds of 10,000 tokens, which doubled about every three and a half
and had only grown sevenfold by iteration 10, where these grew ninefold. Then
the lines break, and many fall steeply for a few iterations: the boom ends in
a crash.

## Where the agents come from, and where they go

<!-- figure youth/flows -->
![Births and deaths in the first 150 iterations](../figures/youth/flows.svg)

**Births and deaths in the first 150 iterations.** For each iteration, the median over the worlds of: the children born in the reproduction phase (yellow); the agents removed by the cleanup of that phase because they were joined to no one or to a piece smaller than the largest (orange — mostly newborns joined to no one); the agents removed by the cleanup of the game because nobody, not even themselves, staked a token on their node (red); and those removed in the game's cleanup because they were cut off (violet).

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. For iterations 0 to 150, take `births` and `orphaned` from the rows with `phase` = 1, and `starved` and `orphaned` from the rows with `phase` = 2.
> 2. At each iteration, take the median of each over the runs that have a row there.
>
> **To make it again:** `python3 book_figures.py youth`.
<!-- /figure -->

Births peak at a median of 10,360 in the reproduction phase of iteration 15.
Deaths follow: agents cut off in the game peak at 6,390 per iteration in the
same iteration, starvation at 1,670. So the crash is mostly the network
breaking apart: at the peak, nearly four agents are cut off for every one that
starves (five at 10,000 tokens). A newborn is joined to those of its parent's
candidates — the parent itself and its neighbours — that the parent's brain
picks ([Chapter 3](03-one-iteration.md)), and every one of those connections
is cut at the end of the game unless tokens cross it. In the boom, most agents
hold a few tokens and many new connections carry nothing; whole branches of
the network lose their link to the largest piece and are removed, with
everyone on them.

<!-- figure youth/children-share -->
![Share of all tokens given to children](../figures/youth/children-share.svg)

**Share of all tokens given to children.** In each reproduction phase, the tokens all parents together gave their children, as a share of all 100,000 tokens. The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th), here at every single iteration.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 1 and 0 ≤ `iteration` ≤ 150, and the statistic `reproTokenShare` — the sum of `invested` over the births of that phase, divided by the 100,000 tokens.
> 2. At each iteration take the median, quartiles and 5th and 95th percentiles over the runs.
>
> **To make it again:** `python3 book_figures.py youth`.
<!-- /figure -->

Giving so much to children does not last. In the first reproduction phase
parents together gave 50% of all tokens to their children; at iteration 10,
37%; at iteration 20, 24%; at iteration 50, 6%; at iteration 100, 2% — almost
exactly as in the worlds of 10,000 tokens. Two things lower it. Far fewer
agents have a child at all: about 74 births for every hundred agents in each
of the first ten iterations, about 3.5 in a settled world
([Chapter 12](12-births-deaths-and-ages.md)). And parents give smaller shares
([Chapter 4](04-the-brain.md)). Part of the first is plain rounding: a parent
gives ⌊*f* · τ⌋ tokens, and an agent holding 3 tokens that would give a fifth
gives ⌊0.6⌋ = 0 and has no child. The founders held 100 tokens each; the median
agent of a settled world holds about 4.5 ([Chapter 15](15-where-do-the-tokens-go.md)).

## A bottleneck of lines

<!-- figure youth/families -->
![Families: distinct ancestors eight iterations back](../figures/youth/families.svg)

**Families: distinct ancestors eight iterations back.** After every game, how many different genotypes of eight iterations earlier the living descend from (before iteration 8: how many founders). The line is the median of the worlds at each iteration, the darker band holds the middle half of them (from the 25th to the 75th percentile) and the paler band nine in ten (5th to 95th), at every iteration.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just after the game, and every statistic is a field of the row (see [What a run records](../notes/frames-and-stats.md)).
>
> 1. Take the rows with `phase` = 2 and 0 ≤ `iteration` ≤ 150, and the statistic `cladesInWindow` ([Families](../notes/families.md): for every living agent, follow its genotype's parents back to the newest one born at or before iteration t − 8, or to a founder's; count the distinct ones).
> 2. At each iteration take the median, quartiles and 5th and 95th percentiles over the runs.
>
> **To make it again:** `python3 book_figures.py youth`.
<!-- /figure -->

The number of **families** counts the different genotypes of eight
iterations earlier that the living descend from ([Families](../notes/families.md)).
Before iteration 8, "eight iterations earlier" is before the world began, so
it counts how many of the 1,000 founders still have descendants. It falls
fast: in the median world, around iteration 8, the living descend from about
390 of them. Later the count rises again, because by then there are many more
genotypes to descend from.

The bottleneck is narrower in a small world: the worlds of 10,000 tokens kept
about 19 of their 100 founders, a fifth, where these keep two fifths. A larger
world has more places where a line can hold out for a while.

Why do lines die so fast? In the game, a node that is won takes on the
**winner's brain**. A founder whose node is won, and whose children's nodes
are won, has no descendants left even though its node still exists.

Following the strongest founders' lines in one world shows how one of them
takes over:

<!-- figure youth/founder-lines -->
![Whose descendants? World of seed 1, first 300 iterations](../figures/youth/founder-lines.svg)

**Whose descendants? World of seed 1, first 300 iterations.** After every game, the share of the living agents that descend from each of the 1,000 founders, stacked. Each coloured layer is the line of one founder; the seven founders whose lines ever held the most are drawn in colour, the other 993 together in grey (most of them with no descendants alive at all). Where a layer reaches the full height, everyone alive descends from that founder.

> [!example]- How to make this figure
> **Runs.** `B1-100000-s001`, one of the 30 baseline runs.
>
> **Data.** From each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with `gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after reproduction and frame `2·t + 1` after the game.
>
> 1. Read every frame in order and remember each genotype's parent genotype (`brain_ids`, `parent_brain_ids`).
> 2. After each game up to iteration 300, follow every living agent's genotype back to its founder and count the agents per founder; divide by the number of agents.
> 3. Stack the shares, the founders with the largest share ever at the bottom.
>
> **To make it again:** `python3 book_figures.py youth`.
<!-- /figure -->

In the world with seed 1, no founder's descendants hold more than 8% of the
world at iteration 10; one line holds 38% at iteration 50, 74% at iteration
100, 86% at iteration 200 and 93% at iteration 300, with three other lines
still alive beside it. In the smaller world of the same seed one line held
52% after ten iterations, and every agent descended from one founder soon
after iteration 300; this one got there at iteration 825.

## One founder for everyone

How soon does that happen in each world? Every 25 iterations the family tree
of genotypes was climbed from the living ([The common ancestor](../notes/common-ancestor.md)
explains how), and the first check at which every living agent descends from
one and the same founder was noted:

<!-- figure youth/one-founder -->
![When everyone descends from one founder](../figures/youth/one-founder.svg)

**When everyone descends from one founder.** For each world, the first of the checks — made every 25 iterations — at which every living agent descends from one and the same founder. 27 of the 30 worlds got there; the world with seed 13 never did; the world with seed 18 never did; the world with seed 20 never did.

> [!example]- How to make this figure
> **Runs.** The 30 baseline runs `B1-100000-s001` … `B1-100000-s030`: the baseline B1 at 100,000 tokens, seeds 1 to 30, 1,500 iterations each (every setting is listed in [Chapter 9](09-thirty-worlds.md)). They are made with `python3 gol_lab.py run E10`.
>
> **Data.** From the runs' frames, through the lineage analysis of Experiment 10 (`python3 gol_lab.py analyse E10`), whose results file `book/results/E10.json` holds every run's `oneFounder`.
>
> 1. Read every frame of a run in order and remember, for every genotype (brain id), the genotype it was copied from (`parent_brain_ids`).
> 2. After the game of every 25th iteration, follow every living agent's genotype back through its parents to a founder's genotype (one with no parent).
> 3. The first iteration at which they all lead to the same founder is the world's value.
> 4. Count the worlds at each value.
>
> **To make it again:** `python3 book_figures.py youth`.
<!-- /figure -->

Of the thirty worlds, 27 got there within their 1,500 iterations. The fastest
did by iteration 125; a quarter by iteration 575; half of the 27 by iteration
1,000; the slowest at 1,475. Three worlds — seeds 13, 18 and 20 — still
descended from more than one founder at their last check. Not one world got
there within its first hundred iterations, where 13 of the thirty smaller
worlds did, and half of them by iteration 175. **A world ten times the size
takes about six times as long to come down to a single founder** —
[Chapter 59](59-worlds-of-100000-tokens.md) reads in that pace that a line
spreads through a world as a random walk spreads, not as a front. The other
999 founders' lines are gone by then: a line ends when the last agent carrying
one of its brains either dies or has its node won by an agent of another line,
whose brain then replaces its own.

## What this means

The youth of a world is a boom, a crash and a bottleneck, at either size. Its
numbers — about 74 births per hundred agents in each of the first ten
iterations, some twenty times the settled rate, and families falling from a
thousand to about four hundred — are unlike anything later in a world's life,
and that is why the rest of the book measures settled worlds from iteration
500 on. A large world comes through its bottleneck with more lines, and keeps
them longer: three of thirty had not come down to one founder after 1,500
iterations. Once it has, every agent of a settled world is, in its brains, a
descendant of one founder: whatever diversity of brains a world later holds,
it has made since.

> [!info] Added later: the youth is when the brains learn
> [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md) played the
> brains of different eras of ten worlds of 5,000 tokens against each other.
> By iteration 50 the brains of every world beat their founders' — they hold
> nine tenths of a world contested with them — and every later era ties with
> the one before it. Whatever the brains of a world learn, they learn in its
> youth.

To make every figure of this chapter: `python3 book_figures.py youth`.

<!-- turns -->
---

← [Chapter 10 · A world's life](10-a-worlds-life.md) · [Contents](../README.md) · [Chapter 12 · Births, deaths and ages](12-births-deaths-and-ages.md) →
<!-- /turns -->
