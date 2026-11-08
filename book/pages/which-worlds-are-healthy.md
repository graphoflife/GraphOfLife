# Which worlds are healthy? A proposed order

> [!question] Questions of this page
> - What could "healthy" mean for a world we hope will keep evolving?
> - How can criteria that say only yes or no put a hundred worlds in order?
> - Which worlds of round 3 are the most healthy, and which the least?

## In short

The author asked for the worlds drawn at random in order, from the most healthy
to the least, by a definition of health they can check. This page proposes
one. It keeps the author's five criteria (decentralised, resilient, almost
three-dimensional, diverse, not too even) and reads each of their lines as how
much of what it asks a world has: a world with 76% of its [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") in its [2-core](bridges.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.")
has 0.95 of the 80% the line asks. A world's health is the sum of its five
criteria, from 0 to 5. Of the hundred worlds of round 3, 82 lived to the end
of their run, and four meet every line: [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 4002 played weakest first, and
seeds 3092, 3023 and 3026. The [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") living world scores 4.2. What most worlds lack is
resilience: one agent's loss cuts off too much of them. Next comes agents that
act alike. The order is a proposal for the author to check against the worlds
themselves: every name in it opens its world in the viewer.

## Why an order, and not a verdict

The [criteria](what-makes-a-world-interesting.md) of the screen say yes or no.
A world meets a line or it does not, and it is healthy when it meets them all.
That is clear, but it throws away most of what we measured: a world with 79%
of its agents in its 2-core fails the line of 80% just as a world with 20%
does.

So the author asked for something else (7 November 2026): an **order** of the
worlds, from the most healthy to the least, made by a definition of health
they can check. They look at worlds near the top and near the bottom, in the
viewer, and say where the order is wrong. Then we change the definition, not
the order by hand, and look again. This page is the first proposal.

The order holds the hundred worlds of round 3 of
[A screen of random worlds](a-screen-of-random-worlds.md), and only those:
Experiment 40's 84, drawn from the [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") the author chose on 24 October
2026, and the eight draws of Experiments 41 and 42, each played twice, a [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")
at a time. Round 1 drew from an older list of settings, round 2 was never
played, and neither is in the order (the author, 8 November 2026). Every run's
name begins with "[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")" only because a drawn world's settings are kept as changes
from B1's rules; its seed says which world it is (3001 to 3100 for Experiment
40, 4001 to 4008 for the draws played a token at a time). There is
no [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") any more. B1 and [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") were where we learned to measure, and a world
is no longer judged against them.

## What healthy could mean

We want worlds in which evolution could go on for a very long time. We would
call such a world healthy if, beyond living,

1. **no one rules it:** no agent holds a large share of the links, or of the
   tokens;
2. **it holds together:** losing one agent or one link does not cut off a
   large part of it;
3. **it has room:** it is spread out like a space of a few [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), and not
   folded into a small world where everyone is a few steps from everyone;
4. **its agents differ:** in their tokens, their links, their ages, and in
   what they do in the same situation;
5. **it is not too even:** its tokens are not shared out more evenly than
   chance would share them.

These are the author's own criteria: decentralised, resilient, almost
three-dimensional, diverse and not too even. This page proposes a few changes
to their lines, each with its reason, [below](#what-the-proposal-changes).

A world should also *last*: one that keeps falling to the edge of [extinction](extinction.md "A world's end before its last iteration: it has dropped to its threshold of agents or fewer.")
is not well, even if it lives to the end. We looked for that, as the fewest
agents a world had in 5% of its [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). But no world that lived through
its run came below a few hundred agents, so a line for it would order nothing
yet, and the proposal leaves it out.

## A line, read as a share of what it asks

Each criterion is made of lines. Take the line "at least 80% of the agents in
the 2-core". A world with 76% of its agents in its 2-core misses it, but it
has most of what the line asks: 76 of the 80, or **0.95**. A world with 20% has
0.25. A world with 85% meets the line and scores **1**: more than enough counts
as enough.

A line that asks for *at most* some amount is read the other way round. The
line "no agent linked to more than 5% of the others" is met to 0.05 / 0.30 =
**0.17** by a world whose best-linked agent reaches 30% of the others: that
agent is six times over the line.

Some quantities do not count from zero. A world's width counts from its [random
twin](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.")'s, which is 1. Links per agent count from a tree's two, since a network in
one piece with two links an agent has no [loop](bridges.md#loops "A path through the network that comes back to where it started without using a connection twice.") at all. The largest holding
counts from one token. And how alike the agents are counts from "all alike",
1, so that what is measured is the share of agents unlike the commonest. In
each case the line is read as the share of the way from that floor to the
line.

A criterion is met as far as its weakest line, because it is met only when all
its lines are. A world's **health** is the sum of its five criteria's scores,
from 0 to 5: one point for each criterion met, part of a point for each one
nearly met. A world that died has no health. The dead come last in the order,
the longest-lived first.

**An example.** The world of seed 3077 lived through its 3,000 iterations
with a median of about 8,000 agents. No agent rules it: its best-linked agent
reaches 0.6% of the others, and its richest holds 0.2% of the tokens. It holds
together: 94% of its agents sit in the 2-core, and no single agent's loss cuts
off more than 0.13% of it. Its agents act differently (a same-position share of
0.96), and its tokens are a little less even than a random deal (a [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of
0.44, 1.04 times a random deal's). So four criteria are met in full: four
points.

But it is folded. Its typical distance is only 1.07 times its random twin's,
against the 1.3 the width line asks: it has (1.07 − 1) / (1.3 − 1) = 0.25 of
what that line asks. Its half-cut exponent is 1.03, the same as its twin's, and
beyond the 4/5 of five dimensions: nothing of what that line asks. Its
criterion *almost three-dimensional* scores 0, its weakest line's share, and
its health is **4.00** of 5. It is the 40th of the 82 living worlds. Were a world only as healthy
as its weakest criterion, it would be the 69th.

## What the proposal changes

The author's lines are kept, but for two, and a few are added:

<!-- table healthorder/changes -->
| what the proposal changes | why |
|---|---|
| adds: tokens of the richest agent, as a share of all tokens (at most 0.05) | no ruler: the hub line asks that no agent hold more than 5% of the links, this asks the same of tokens; a world unequal everywhere is healthy, one agent holding the world's fortune is not |
| adds: same-position share: of two agents in the same situation, how often they act measurably differently (at least 0.5) | the author's own number for very different behaviours surviving (2026-10-24), with its line at half: two agents more often act differently than alike |
| adds: Gini of the tokens, over the Gini of tokens dealt out at random at the same mean holding (at least 1) | with no baseline, chance draws the line: a world more even than a random deal has something flattening it; tokens dealt out at random, one at least, at a mean of m an agent have a Gini of (m - 1) / (2m - 1), so the line moves with the world's own riches (0.40 at 3 tokens an agent, 0.46 at 7, 0.47 at 10) |
| drops: genotypes_alike | a genotype is renewed by every mutation, so how many agents share one says how seldom brains mutate, not how differently agents act; the same-position share measures that (in round 3 no world came near the line: at most 26% of a world's agents shared a genotype) |
| drops: gini | drawn at B2's most even world of Experiment 16, a baseline the screen no longer compares with; the proposal's line for not too even is drawn by chance instead |
<!-- /table -->

**A line drawn by chance.** The author's line for *not too even* was the Gini
of B2's most even world, 0.46. With no baseline, chance can draw it instead.
Deal a world's tokens out to its agents completely at random, every agent
getting at least one, and see how unequal the result is. The most random way
to share whole tokens with a given mean has each agent's holding fall off
geometrically (one token is the most common holding, two the next, and so
on), and its Gini is (*m* − 1) / (2*m* − 1) for a mean of *m* tokens an agent:
0.40 at three tokens an agent, 0.46 at seven, 0.47 at ten. A world more even
than this has something flattening it. A world less even has agents that hold
on to what they win. So the line asks that a world's Gini be at least the Gini
of a random deal of its own tokens.

**The author's two new numbers.** On 24 October 2026 the author asked for two
numbers of health and chose to see them on every world of round 3 before
drawing their lines
([Two numbers reported, not judged](what-makes-a-world-interesting.md#two-numbers-reported-not-judged)).
Round 3 has run:
- the [same-position share](what-makes-a-world-interesting.md#the-same-position-share),
  for very different behaviours living side by side, gets a line at a half:
  two agents in the same situation more often act differently than alike. It
  is easy to meet for [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that read no random numbers, and hard for brains
  that read many
  ([How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md#what-decides-how-healthy-a-living-world-grows)).
- the [half-cut exponent](what-makes-a-world-interesting.md#the-half-cut-exponent),
  for a world that on the largest scale looks like a space, gets no line yet.
  We first drew one from what a lattice gives: halving a body of *d*
  dimensions cuts *N*^((*d* − 1)/*d*) of its links, so 1/2 is a plane's and
  2/3 three dimensions'. But a network as sparse as our worlds halves more
  cheaply than a lattice, by how sparse it is as much as by its dimension.
  Random networks of known shape show it, each of 4,000 agents put at random
  in a square or a cube and joined to those nearest:

<!-- table healthorder/calibration -->
| random network in | links an agent | agents in its largest piece | links cut to halve it | half-cut exponent | a lattice's |
|---|---|---|---|---|---|
| a plane | 2.9 | 92 | 5 | 0.36 | 0.50 |
| a plane | 4.8 | 3,622 | 23 | 0.38 | 0.50 |
| a plane | 7.7 | 3,986 | 68 | 0.51 | 0.50 |
| a cube | 2.9 | 1,785 | 9 | 0.29 | 0.67 |
| a cube | 4.7 | 3,833 | 105 | 0.56 | 0.67 |
| a cube | 7.4 | 3,991 | 251 | 0.67 | 0.67 |
<!-- /table -->

At about five links an agent, a random network in a cube reads 0.56 and one in
a plane 0.38; only at about eight links do they reach a lattice's values. At
three links both nearly fall apart. Round 3's living worlds have a median of
3.6 links an agent, and their exponents follow their links (a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of
0.67) and their dimension by balls (0.91). So a fixed line at 1/2 would call a
sparse world of three dimensions thin. The fair test compares each world with
a random network of its own size and density in a cube: a world as hard to
halve as that twin is three-dimensional on the largest scale. That is a step
for the author to approve
([For the author to check](#for-the-author-to-check)).

## The order

<!-- figure healthorder/order -->
![The living worlds, most healthy first](../figures/healthorder/order.svg)

**The living worlds, most healthy first.** Each of the 82 worlds that lived to the end of its run is a row, ranked by its health, the sum of its five criteria's scores; each criterion is a column, coloured by how much of what it asks the world has, from none (dark) to all of it (bright yellow): a criterion is met as far as its weakest line. A world named "w" or "s" is one of the eight draws played a token at a time, weakest first or strongest first. The 18 worlds that died are not shown: they come last in the order.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` (research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept world), scored by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. Average every number of a world over iterations 100 to the end of its run (the judge's averages, `research/screen/measures.py`).
> 2. For each line, the share of what it asks that the world has, counted from the line's floor (`progress` in `health.py`), at most 1.
> 3. Each criterion: its weakest line's share. Health: the sum over the five criteria.
> 4. Rank the living worlds by health, ties by their weakest criterion.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

In the heat map each row is a living world, the healthiest at the top, and
each column a criterion, bright where the world has all of what it asks. The
table gives every number. In it, a world named with a "w" or an "s" is one of
the eight draws played a token at a time, weakest first or strongest first. Each world's name opens it in the viewer,
on a computer that holds the runs. The column *rank by the author's lines*
says where the world would stand if its lines were the author's alone, and the
next, where it would stand if the weakest criterion decided.

Four worlds meet every line and eleven more meet four criteria in
full. A quarter of the living worlds score less than 3.75, and a quarter more
than 4.63. By the author's lines alone, six would meet every criterion: three
of the four above, and seeds 3051, 3062 and 3016.

<!-- table healthorder/order -->
| rank | world | round | health (of 5) | rank by the author's lines | rank if the weakest criterion decides | decentralised | resilient | almost 3-D | diverse | not too even | agents | first half → second | weakest line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [s4002 w](run:B1-30000-5379e2-s4002) | round 3, weakest first | 5.00 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 5,258 | 5.00 → 5.00 | none |
| 2 | [s3092](run:B1-30000-7fc03f-s3092) | round 3 | 5.00 | 1 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6,314 | 5.00 → 5.00 | none |
| 3 | [s3023](run:B1-30000-b0f6e3-s3023) | round 3 | 5.00 | 2 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1,609 | 5.00 → 5.00 | none |
| 4 | [s3026](run:B1-30000-d6955f-s3026) | round 3 | 5.00 | 5 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1,957 | 5.00 → 5.00 | none |
| 5 | [s4008 w](run:B1-30000-1cfaa4-s4008) | round 3, weakest first | 4.99 | 8 | 5 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 5,205 | 4.91 → 5.00 | agent_cut (0.99) |
| 6 | [s3008](run:B1-30000-e19f2d-s3008) | round 3 | 4.98 | 10 | 6 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 2,113 | 4.98 → 4.94 | core (0.98) |
| 7 | [s3042](run:B1-30000-b82ad0-s3042) | round 3 | 4.91 | 25 | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 11,458 | 4.91 → 4.90 | gini_random (0.91) |
| 8 | [s3045](run:B1-30000-f6d23f-s3045) | round 3 | 4.89 | 14 | 10 | 1.00 | 1.00 | 0.89 | 1.00 | 1.00 | 1,401 | 5.00 → 4.31 | width (0.89) |
| 9 | [s3022](run:B1-30000-298337-s3022) | round 3 | 4.85 | 15 | 7 | 1.00 | 0.92 | 1.00 | 0.95 | 0.97 | 5,443 | 4.84 → 4.86 | core (0.92) |
| 10 | [s3024](run:B1-30000-f05874-s3024) | round 3 | 4.83 | 17 | 15 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 4,580 | 4.81 → 4.84 | core (0.83) |
| 11 | [s3057](run:B1-30000-38aee9-s3057) | round 3 | 4.82 | 18 | 9 | 1.00 | 0.89 | 1.00 | 1.00 | 0.93 | 5,038 | 4.74 → 4.91 | agent_cut (0.89) |
| 12 | [s3080](run:B1-30000-625bc7-s3080) | round 3 | 4.79 | 16 | 14 | 1.00 | 0.83 | 1.00 | 1.00 | 0.96 | 2,002 | 4.81 → 4.77 | core (0.83) |
| 13 | [s3018](run:B1-30000-af1e28-s3018) | round 3 | 4.74 | 24 | 11 | 1.00 | 0.86 | 1.00 | 1.00 | 0.88 | 2,930 | 4.73 → 4.73 | core (0.86) |
| 14 | [s3046](run:B1-30000-27bd64-s3046) | round 3 | 4.74 | 23 | 22 | 1.00 | 1.00 | 1.00 | 1.00 | 0.74 | 2,492 | 4.65 → 4.73 | gini_random (0.74) |
| 15 | [s4008 s](run:B1-30000-645972-s4008) | round 3, strongest first | 4.73 | 27 | 24 | 1.00 | 0.73 | 1.00 | 1.00 | 1.00 | 5,380 | 4.73 → 4.74 | agent_cut (0.73) |
| 16 | [s3055](run:B1-30000-cca771-s3055) | round 3 | 4.71 | 26 | 13 | 1.00 | 0.83 | 1.00 | 1.00 | 0.87 | 2,552 | 4.65 → 4.78 | agent_cut (0.83) |
| 17 | [s4002 s](run:B1-30000-192733-s4002) | round 3, strongest first | 4.70 | 29 | 26 | 1.00 | 0.71 | 1.00 | 1.00 | 0.99 | 5,624 | 4.74 → 4.66 | agent_cut (0.71) |
| 18 | [s3058](run:B1-30000-4108dd-s3058) | round 3 | 4.68 | 9 | 19 | 0.92 | 1.00 | 1.00 | 0.77 | 1.00 | 1,197 | 4.40 → 4.78 | apart (0.77) |
| 19 | [s3021](run:B1-30000-4aff7d-s3021) | round 3 | 4.67 | 13 | 21 | 0.74 | 0.93 | 1.00 | 1.00 | 1.00 | 1,648 | 4.87 → 4.54 | richest (0.74) |
| 20 | [s3032](run:B1-30000-22865f-s3032) | round 3 | 4.64 | 31 | 23 | 1.00 | 0.91 | 1.00 | 1.00 | 0.74 | 7,294 | 4.66 → 4.63 | gini_random (0.74) |
| 21 | [s4003 w](run:B1-30000-9efb1d-s4003) | round 3, weakest first | 4.63 | 39 | 28 | 1.00 | 0.70 | 1.00 | 1.00 | 0.94 | 10,306 | 4.76 → 4.55 | agent_cut (0.70) |
| 22 | [s3006](run:B1-30000-023c0a-s3006) | round 3 | 4.62 | 37 | 18 | 1.00 | 0.85 | 1.00 | 1.00 | 0.77 | 6,907 | 4.62 → 4.63 | gini_random (0.77) |
| 23 | [s4003 s](run:B1-30000-fffc7d-s4003) | round 3, strongest first | 4.62 | 35 | 27 | 1.00 | 0.70 | 1.00 | 1.00 | 0.92 | 8,936 | 4.69 → 4.56 | agent_cut (0.70) |
| 24 | [s3019](run:B1-30000-b1209a-s3019) | round 3 | 4.61 | 21 | 12 | 1.00 | 0.92 | 0.94 | 0.84 | 0.91 | 3,040 | 4.55 → 4.58 | apart (0.84) |
| 25 | [s3049](run:B1-30000-c51eb8-s3049) | round 3 | 4.58 | 32 | 33 | 1.00 | 0.99 | 0.60 | 1.00 | 1.00 | 5,213 | 4.64 → 4.52 | width (0.60) |
| 26 | [s3054](run:B1-30000-d9254e-s3054) | round 3 | 4.55 | 19 | 20 | 1.00 | 0.80 | 1.00 | 0.75 | 1.00 | 3,323 | 4.56 → 4.54 | apart (0.75) |
| 27 | [s3064](run:B1-30000-a6a7ec-s3064) | round 3 | 4.55 | 34 | 35 | 1.00 | 0.97 | 0.58 | 1.00 | 0.99 | 5,346 | 4.50 → 4.59 | width (0.58) |
| 28 | [s3004](run:B1-30000-7d5208-s3004) | round 3 | 4.54 | 12 | 30 | 1.00 | 1.00 | 1.00 | 0.65 | 0.89 | 1,612 | 4.50 → 4.40 | apart (0.65) |
| 29 | [s3051](run:B1-30000-b5a76a-s3051) | round 3 | 4.53 | 4 | 42 | 1.00 | 1.00 | 1.00 | 0.53 | 1.00 | 976 | 4.25 → 4.52 | apart (0.53) |
| 30 | [s4006 s](run:B1-30000-8526cf-s4006) | round 3, strongest first | 4.51 | 33 | 16 | 1.00 | 0.88 | 1.00 | 0.83 | 0.80 | 7,166 | 4.51 → 4.48 | gini_random (0.80) |
| 31 | [s3087](run:B1-30000-363694-s3087) | round 3 | 4.44 | 20 | 40 | 1.00 | 0.98 | 1.00 | 0.53 | 0.94 | 10,456 | 4.42 → 4.33 | apart (0.53) |
| 32 | [s3062](run:B1-30000-def151-s3062) | round 3 | 4.44 | 6 | 51 | 0.44 | 1.00 | 1.00 | 1.00 | 1.00 | 2,364 | 4.45 → 4.33 | richest (0.44) |
| 33 | [s4004 s](run:B1-30000-2bac4d-s4004) | round 3, strongest first | 4.41 | 40 | 32 | 1.00 | 0.61 | 1.00 | 1.00 | 0.80 | 1,244 | 4.40 → 4.41 | agent_cut (0.61) |
| 34 | [s3052](run:B1-30000-0d058a-s3052) | round 3 | 4.41 | 43 | 17 | 0.89 | 0.81 | 1.00 | 0.91 | 0.80 | 7,558 | 4.16 → 4.34 | gini_random (0.80) |
| 35 | [s4004 w](run:B1-30000-17ac6e-s4004) | round 3, weakest first | 4.40 | 41 | 39 | 1.00 | 0.55 | 1.00 | 1.00 | 0.85 | 1,101 | 4.41 → 4.39 | agent_cut (0.55) |
| 36 | [s3016](run:B1-30000-b23381-s3016) | round 3 | 4.39 | 3 | 55 | 0.39 | 1.00 | 1.00 | 1.00 | 1.00 | 1,192 | 3.96 → 4.35 | richest (0.39) |
| 37 | [s3001](run:B1-30000-68cda7-s3001) | round 3 | 4.39 | 49 | 25 | 1.00 | 0.91 | 1.00 | 0.75 | 0.72 | 17,728 | 4.43 → 4.08 | gini_random (0.72) |
| 38 | [s4005 s](run:B1-30000-6f6c09-s4005) | round 3, strongest first | 4.39 | 47 | 29 | 1.00 | 0.70 | 1.00 | 1.00 | 0.69 | 3,930 | 4.45 → 4.32 | gini_random (0.69) |
| 39 | [s3047](run:B1-30000-10c9f1-s3047) | round 3 | 4.30 | 11 | 31 | 0.73 | 0.95 | 1.00 | 0.62 | 1.00 | 3,638 | 4.40 → 4.31 | apart (0.62) |
| 40 | [s3077](run:B1-30000-d6eabf-s3077) | round 3 | 4.25 | 53 | 69 | 1.00 | 1.00 | 0.25 | 1.00 | 1.00 | 7,999 | 4.22 → 4.27 | width (0.25) |
| 41 | [s3071](run:B1-30000-605c0d-s3071) | round 3 | 4.24 | 52 | 41 | 0.53 | 0.70 | 1.00 | 1.00 | 1.00 | 6,215 | 4.25 → 4.25 | hub (0.53) |
| 42 | [s3091](run:B1-30000-42d88d-s3091) | round 3 | 4.21 | 42 | 37 | 1.00 | 0.56 | 1.00 | 0.80 | 0.85 | 2,288 | 4.17 → 4.25 | agent_cut (0.56) |
| 43 | [s3050](run:B1-30000-ebe08c-s3050) | round 3 | 4.21 | 51 | 48 | 0.45 | 0.76 | 1.00 | 1.00 | 1.00 | 2,980 | 4.09 → 4.34 | richest (0.45) |
| 44 | [s4007 s](run:B1-30000-a8a2d2-s4007) | round 3, strongest first | 4.17 | 58 | 45 | 1.00 | 0.69 | 1.00 | 1.00 | 0.47 | 6,827 | 4.20 → 4.13 | gini_random (0.47) |
| 45 | [s4007 w](run:B1-30000-04d3eb-s4007) | round 3, weakest first | 4.16 | 57 | 50 | 1.00 | 0.72 | 1.00 | 1.00 | 0.45 | 6,536 | 4.09 → 4.24 | gini_random (0.45) |
| 46 | [s3063](run:B1-30000-924e70-s3063) | round 3 | 4.16 | 48 | 36 | 1.00 | 0.89 | 1.00 | 0.58 | 0.69 | 14,998 | 4.33 → 3.99 | apart (0.58) |
| 47 | [s3078](run:B1-30000-dd27b6-s3078) | round 3 | 4.12 | 28 | 49 | 1.00 | 0.67 | 1.00 | 0.45 | 1.00 | 1,646 | 4.09 → 4.10 | apart (0.45) |
| 48 | [s3005](run:B1-30000-c18a1f-s3005) | round 3 | 4.10 | 45 | 34 | 0.97 | 0.59 | 1.00 | 0.70 | 0.85 | 4,119 | 4.03 → 4.16 | agent_cut (0.59) |
| 49 | [s3082](run:B1-30000-1141ea-s3082) | round 3 | 4.06 | 61 | 43 | 0.59 | 0.50 | 1.00 | 1.00 | 0.97 | 3,294 | 4.20 → 3.92 | agent_cut (0.50) |
| 50 | [s3084](run:B1-30000-ad76aa-s3084) | round 3 | 4.01 | 63 | 76 | 0.89 | 1.00 | 0.12 | 1.00 | 1.00 | 1,424 | 3.84 → 4.15 | width (0.12) |
| 51 | [s3067](run:B1-30000-4bee8e-s3067) | round 3 | 4.01 | 44 | 54 | 0.92 | 0.70 | 1.00 | 0.39 | 1.00 | 15,925 | 4.22 → 3.73 | apart (0.39) |
| 52 | [s3059](run:B1-30000-2ec87a-s3059) | round 3 | 3.96 | 66 | 44 | 0.77 | 0.49 | 1.00 | 1.00 | 0.70 | 4,240 | 3.97 → 3.96 | agent_cut (0.49) |
| 53 | [s3061](run:B1-30000-00e3c8-s3061) | round 3 | 3.96 | 22 | 73 | 1.00 | 0.88 | 1.00 | 0.22 | 0.86 | 3,193 | 3.91 → 4.01 | apart (0.22) |
| 54 | [s3014](run:B1-30000-7d129f-s3014) | round 3 | 3.95 | 46 | 52 | 1.00 | 0.76 | 1.00 | 0.44 | 0.75 | 11,026 | 3.89 → 4.01 | apart (0.44) |
| 55 | [s3070](run:B1-30000-a4d3a0-s3070) | round 3 | 3.92 | 69 | 60 | 0.66 | 0.31 | 1.00 | 1.00 | 0.95 | 8,160 | 3.93 → 3.92 | agent_cut (0.31) |
| 56 | [s3043](run:B1-30000-02ed17-s3043) | round 3 | 3.85 | 30 | 64 | 1.00 | 0.93 | 0.81 | 0.28 | 0.84 | 2,515 | 3.81 → 3.90 | apart (0.28) |
| 57 | [s3072](run:B1-30000-f05562-s3072) | round 3 | 3.84 | 54 | 68 | 1.00 | 0.84 | 1.00 | 0.26 | 0.75 | 20,226 | 3.79 → 3.90 | apart (0.26) |
| 58 | [s3044](run:B1-30000-63827e-s3044) | round 3 | 3.84 | 64 | 47 | 1.00 | 0.52 | 1.00 | 0.85 | 0.47 | 3,543 | 3.86 → 3.83 | gini_random (0.47) |
| 59 | [s3013](run:B1-30000-7f7cd5-s3013) | round 3 | 3.81 | 55 | 53 | 1.00 | 0.41 | 1.00 | 0.49 | 0.91 | 11,011 | 3.88 → 3.74 | agent_cut (0.41) |
| 60 | [s3040](run:B1-30000-36d076-s3040) | round 3 | 3.78 | 71 | 59 | 0.31 | 0.70 | 1.00 | 0.98 | 0.79 | 4,090 | 3.85 → 3.63 | hub (0.31) |
| 61 | [s3065](run:B1-30000-ba3afe-s3065) | round 3 | 3.75 | 50 | 67 | 1.00 | 1.00 | 0.26 | 0.53 | 0.96 | 2,122 | 3.68 → 3.82 | width (0.26) |
| 62 | [s3093](run:B1-30000-26a341-s3093) | round 3 | 3.75 | 36 | 71 | 0.99 | 0.53 | 1.00 | 0.23 | 1.00 | 3,432 | 3.76 → 3.72 | apart (0.23) |
| 63 | [s3030](run:B1-30000-90fc91-s3030) | round 3 | 3.71 | 73 | 79 | 1.00 | 1.00 | 0.06 | 0.65 | 1.00 | 20,395 | 3.77 → 3.65 | width (0.06) |
| 64 | [s3012](run:B1-30000-02fa53-s3012) | round 3 | 3.68 | 65 | 58 | 0.95 | 0.32 | 0.70 | 0.72 | 0.99 | 2,522 | 3.68 → 3.68 | agent_cut (0.32) |
| 65 | [s3090](run:B1-30000-05ba40-s3090) | round 3 | 3.60 | 62 | 78 | 1.00 | 1.00 | 0.07 | 0.53 | 1.00 | 1,537 | 3.68 → 3.45 | width (0.07) |
| 66 | [s3033](run:B1-30000-631e49-s3033) | round 3 | 3.53 | 60 | 70 | 0.23 | 0.96 | 0.33 | 1.00 | 1.00 | 1,792 | 3.43 → 3.88 | richest (0.23) |
| 67 | [s3028](run:B1-30000-2eef58-s3028) | round 3 | 3.53 | 67 | 38 | 0.56 | 0.59 | 1.00 | 0.58 | 0.80 | 5,865 | 3.62 → 3.40 | links_min (0.56) |
| 68 | [s3066](run:B1-30000-dfd51b-s3066) | round 3 | 3.52 | 74 | 81 | 0.50 | 1.00 | 0.02 | 1.00 | 1.00 | 10,027 | 3.49 → 3.55 | width (0.02) |
| 69 | [s3041](run:B1-30000-965f1e-s3041) | round 3 | 3.49 | 72 | 56 | 0.37 | 0.46 | 1.00 | 0.79 | 0.87 | 4,434 | 3.23 → 3.70 | links_min (0.37) |
| 70 | [s3056](run:B1-30000-43985b-s3056) | round 3 | 3.47 | 56 | 57 | 0.71 | 0.53 | 1.00 | 0.33 | 0.90 | 3,560 | 3.52 → 3.43 | apart (0.33) |
| 71 | [s3095](run:B1-30000-3e5856-s3095) | round 3 | 3.47 | 59 | 77 | 0.35 | 1.00 | 0.12 | 1.00 | 1.00 | 3,517 | 3.29 → 4.14 | width (0.12) |
| 72 | [s3085](run:B1-30000-801b34-s3085) | round 3 | 3.47 | 70 | 46 | 0.85 | 0.47 | 1.00 | 0.61 | 0.53 | 4,259 | 3.50 → 3.44 | agent_cut (0.47) |
| 73 | [s3094](run:B1-30000-4ac946-s3094) | round 3 | 3.42 | 77 | 80 | 0.38 | 1.00 | 0.03 | 1.00 | 1.00 | 11,200 | 3.42 → 3.41 | width (0.03) |
| 74 | [s3083](run:B1-30000-0d7e0c-s3083) | round 3 | 3.29 | 75 | 62 | 0.30 | 0.29 | 1.00 | 0.79 | 0.91 | 4,148 | 3.19 → 3.39 | agent_cut (0.29) |
| 75 | [s3007](run:B1-30000-8119cd-s3007) | round 3 | 3.23 | 38 | 61 | 0.30 | 0.51 | 1.00 | 0.43 | 1.00 | 2,768 | 3.15 → 3.32 | richest (0.30) |
| 76 | [s3098](run:B1-30000-794ae8-s3098) | round 3 | 3.19 | 68 | 66 | 0.61 | 0.48 | 1.00 | 0.27 | 0.84 | 4,796 | 3.12 → 3.26 | apart (0.27) |
| 77 | [s3037](run:B1-30000-845393-s3037) | round 3 | 3.10 | 76 | 65 | 0.47 | 0.28 | 1.00 | 0.66 | 0.70 | 10,156 | 3.49 → 2.66 | agent_cut (0.28) |
| 78 | [s3076](run:B1-30000-66c764-s3076) | round 3 | 2.75 | 80 | 74 | 0.70 | 0.46 | 1.00 | 0.17 | 0.42 | 4,636 | 2.72 → 2.78 | children_alike (0.17) |
| 79 | [s3099](run:B1-30000-1ab8a0-s3099) | round 3 | 2.65 | 79 | 72 | 0.28 | 0.28 | 1.00 | 0.22 | 0.88 | 21,743 | 2.72 → 2.59 | apart (0.22) |
| 80 | [s3038](run:B1-30000-1ad8a9-s3038) | round 3 | 2.65 | 78 | 63 | 0.31 | 0.28 | 1.00 | 0.35 | 0.71 | 6,400 | 2.55 → 2.75 | agent_cut (0.28) |
| 81 | [s3089](run:B1-30000-77ad6a-s3089) | round 3 | 1.90 | 81 | 75 | 0.16 | 0.18 | 0.31 | 0.75 | 0.50 | 5,620 | 2.24 → 1.65 | hub (0.16) |
| 82 | [s3020](run:B1-30000-511930-s3020) | round 3 | 1.12 | 82 | 82 | 0.01 | 0.00 | 0.01 | 0.10 | 1.00 | 9,388 | 1.14 → 1.09 | core (0.00) |
<!-- /table -->

The worlds that died, longest-lived first:

<!-- table healthorder/dead -->
| world | round | died at iteration |
|---|---|---|
| [s3069](run:B1-30000-033fbf-s3069) | round 3 | 2,442 |
| [s3100](run:B1-30000-13bb83-s3100) | round 3 | 1,815 |
| [s4006 w](run:B1-30000-673ef1-s4006) | round 3, weakest first | 1,445 |
| [s3096](run:B1-30000-76ca8e-s3096) | round 3 | 1,030 |
| [s3009](run:B1-30000-f1b401-s3009) | round 3 | 920 |
| [s3086](run:B1-30000-1393f4-s3086) | round 3 | 719 |
| [s4001 s](run:B1-30000-ec8a95-s4001) | round 3, strongest first | 387 |
| [s4005 w](run:B1-30000-a6c23a-s4005) | round 3, weakest first | 230 |
| [s3015](run:B1-30000-f05b17-s3015) | round 3 | 214 |
| [s3079](run:B1-30000-afe35f-s3079) | round 3 | 210 |
| [s3010](run:B1-30000-30d3da-s3010) | round 3 | 202 |
| [s3081](run:B1-30000-378f0a-s3081) | round 3 | 167 |
| [s4001 w](run:B1-30000-1da684-s4001) | round 3, weakest first | 144 |
| [s3031](run:B1-30000-caf1c5-s3031) | round 3 | 142 |
| [s3036](run:B1-30000-1d8c54-s3036) | round 3 | 115 |
| [s3017](run:B1-30000-44f693-s3017) | round 3 | 94 |
| [s3027](run:B1-30000-be02ce-s3027) | round 3 | 26 |
| [s3074](run:B1-30000-165142-s3074) | round 3 | 12 |
<!-- /table -->

## What the order shows

**Four worlds meet every line, and they share no recipe.** Two play the
lottery and two let the largest stake win; two search far and two not at all;
two mend deaths and two do not; one mixes brains at birth. The settings that
clearly harm a world are missing from all four (a cap of four or eight links,
[estates](estate.md "The tokens a dead agent leaves, and who gets them.") given to the poorest). Beyond that they share only one thing: none
plays with [coalitions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake."), which three worlds in four lack anyway
([How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md)).

**Resilience is what most worlds lack.** Only 20 of the 82 living worlds are
resilient, against 48 decentralised, 65 almost three-dimensional, 41 diverse
and 32 not too even. The line a world falls furthest short of is most often
"no single agent's loss cuts off more than 15% of it" (in 20 worlds): a world
that hangs on a few agents. Next comes the same-position share (in 18):
agents that act alike.

**The proposal moves few worlds far** from where the author's own lines would
put them (a rank correlation of 0.86), and those for a reason that can be
named:
- down, where one agent holds a large share of the tokens: seed 3016 from 3rd
  to 36th (its richest agent holds 13% of all the tokens), 3062 from 6th to
  32nd (11%), 3007 from 38th to 75th (17%);
- down, where agents act alike: seeds 3051 (from 4th to 29th), 3047 (11th to
  39th), 3061, 3043 and 3093, whose same-position shares are 0.11 to 0.31;
- up, where a low Gini is still unequal for so poor a world: seed 3042 from
  25th to 7th (a Gini of 0.35, below B2's line, but 0.91 of a random deal at its
  mean holding).

**At the bottom are the [stars](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") and the worlds that one loss breaks.** In the
last two places are seeds 3020 and 3089, where one agent is linked to 99.5% and
31% of the others. Above them come worlds whose agents act alike and that single
losses cut apart.

## Healthier or sicker with time?

<!-- figure healthorder/halves -->
![Healthier or sicker with time?](../figures/healthorder/halves.svg)

**Healthier or sicker with time?** Each dot is a living world: its health scored over the first half of its run (iterations 100 to the middle) and over the second (the middle to the end). Above the dashed diagonal a world grew healthier; below it, sicker.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` (research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept world), scored by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. Score each half of every living world's run as the whole is scored.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

Health does not drift over a run. From the first half of their run to the
second, 29 worlds grew healthier by more than 0.05, 27 grew sicker, and 26
stayed as they were; the median change is nil. The largest falls are seed 3037
(from 3.49 to 2.66) and seed 3045 (5.00 to 4.31); the largest rises, seed 3095
(3.29 to 4.14) and seed 3016 (3.96 to 4.35).

## Every line against the worlds

<!-- figure healthorder/lines -->
![Every line of the proposal, against the living worlds](../figures/healthorder/lines.svg)

**Every line of the proposal, against the living worlds.** One chart a measure: each dot is a living world's average of it, from iteration 100 to the end of its run, and each dashed line is a line drawn on it; the short bar is the median. The title says which side passes and how many of the living worlds meet every line on the chart. The dots are spread sideways only so that they do not hide each other.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` (research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept world), scored by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. Read each living world's averages from `health.json`.
> 2. One dot per world and measure; each line's value as a guide.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

Four of the author's lines are met by every living world: how alike the
tokens are, the largest holding, how alike the 2-core's links are, and how
alike the ages are. They cost no world anything and order nothing here, but
they would catch a world of paupers, or one where everyone is the same age.
The lines that order the worlds most are the 2-core and the agent cut (met by
34 and 38 of the 82), the same-position share (42) and the random deal's Gini
(32).

## For the author to check

The order is a proposal, and the author corrects it. The quickest way:
1. Open the worlds near the top of the order and the living worlds near its
   bottom in the viewer (each name in the table opens its world), and look at
   them late in their run.
2. Note the worlds that are out of place: one you would call healthy that
   sits low, or a sick world that sits high.
3. The table says, for every world, which criterion and which line put it
   where it is. So a world out of place points at a line to move, or at a
   criterion that counts for too much or too little.

The tables sort by any column: click its head.

Three questions the order cannot answer by itself:
- **Should the criteria add up, or should the weakest decide?** Here they add
  up, so a world excellent in four criteria and poor in the fifth can rank
  high. If a world is only as healthy as its weakest criterion, as a body is
  only as healthy as its sickest organ, it would rank low.
- **Which of the proposed changes should stay?** Each is in the table of
  [What the proposal changes](#what-the-proposal-changes), with its reason, and
  the order by the author's own lines is in the table's column beside the
  proposal's rank.
- **Should the half-cut exponent be read against a twin in space?** Each world
  would be compared with a random network of its own size and density in a
  cube, and called three-dimensional on the largest scale if it is as hard to
  halve. That takes a few minutes of computing a world, and needs no world
  played again.

## What we learned

- **An order needs no new criteria.** The author's lines, each read as the
  share of what it asks, put round 3's 82 living worlds in an order that
  changes little when the weakest criterion decides instead (a rank
  correlation of 0.90).
- **Four of round 3's hundred worlds meet every line,** and they share no
  recipe.
- **Resilience is the rarest quality,** and agents that act alike the
  commonest weakness after it.
- **Health does not drift** over a run: as many worlds grow healthier as
  sicker.
- **The half-cut exponent reads how sparse a world is as much as its
  dimension.** A fair line for it needs a twin in space of the world's own
  density.

## Where to go next

- [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md):
  which settings make a world healthy, and which kill it.
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the same worlds read for what evolution needs: heredity, selection and
  change.
- [What makes a world interesting](what-makes-a-world-interesting.md): the
  author's criteria and lines this order starts from.
- [A screen of random worlds](a-screen-of-random-worlds.md): how the worlds
  were drawn, and what became of them.

## How this was made

The worlds are the runs of Experiments 40, 41 and 42, listed with every
setting on the page [A screen of random worlds](a-screen-of-random-worlds.md).
Nothing is played again: every number is read from what the worlds recorded,
by the screen's judge (`research/screen/measures.py`), and kept, one file a
world, in `GraphOfLifeRuns/.pilots/screen/worlds/`. The averages run from
iteration 100 to the end of each run, 3,000 iterations.

`research/screen/health.py` turns the averages into the order: `FLOOR` holds
where each line counts from, `PROPOSED` and `DROPPED` the changes with their
reasons, `progress` reads a line as a share and `score` a world as the sum of
its criteria. `python3 research/screen/health.py E41 E42 E40` writes
`research/screen/health.json`; with `--read 3` it first makes, on three
processes, any reading a world still lacks. The figures and tables of this
page are drawn from that file by `python3 book_figures.py healthorder`.
