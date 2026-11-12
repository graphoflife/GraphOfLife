# Which worlds are healthy? Three orders

> [!question] Questions of this page
> - What could "healthy" mean for a world we hope will keep evolving?
> - How can measures in different units be put together into one order, and in how many ways?
> - Which worlds of round 3 are the healthiest, and which the least?

## In short

The author asked for the worlds drawn at random in order, from the most healthy
to the least, by a definition of health they can check. Health here is read
from six criteria: the author's five (decentralised, resilient, almost
three-dimensional, diverse, not too even) and a sixth the author added, few
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") culled. No criterion says yes or no. Each measure of a world is scored
by how many of round 3's 82 living worlds it does better than, so every
difference counts, and the criteria are put together in three ways, each an
order of its own: multiplied, added, or the worst of them. Resilience counts,
besides the cuts, the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") on [loops](measuring-networks.md#loops "A path through the network that comes back to where it started without using a connection twice.") per link, which is low both for a tree
and for a world crowded with links. Five worlds are among the first 13 of all
three orders: [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 3042, first by the product, seed 4006 played strongest
first, and seeds 3018, 3080 and 3057. Seed 3077 shows how much the way of
putting them together matters: it is the third by the sum, the 24th by the
product and the 65th by its worst criterion, as it is folded into a small
world. The candidates
for [open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution "Evolution that keeps producing new things without end, not only up to a plateau.") are drawn from the first 13 of each order. Every name
in the orders opens its world in the viewer, for the author to check.

From 11 November 2026 this health is frozen at the version of 9 November and
used only as a filter: worlds that are dead, a [star](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub."), uniform or frozen are left
out. Worlds are ranked by evolution, against null models, and no longer by
these orders ([Health as a
filter](measuring-health.md#from-11-november-2026-health-as-a-filter)). The
orders stay as the record of how round 3's candidates were chosen.

## Why an order, and not a verdict

The [criteria](measuring-health.md#what-makes-a-world-interesting) of the screen say yes or no.
A world meets a line or it does not, and it is healthy when it meets them all.
That is clear, but it throws away most of what we measured: a world with 79%
of its agents in its [2-core](measuring-networks.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.") fails the line of 80% just as a world with 20%
does, and a world far past a line counts the same as one just past it.

So the author asked for something else (7 November 2026): an **order** of the
worlds, from the most healthy to the least, made by a definition of health
they can check. They look at worlds near the top and near the bottom, in the
viewer, and say where the order is wrong. Then we change the definition, not
the order by hand, and look again. The first definition read each line as the
share of what it asks, up to the line. The author found that the lines still
made it a yes or no past them (9 November 2026), and asked for something
else: "a continuous scoring ... always the bigger the better and always
positive", put together "by adding scores, multiplying scores, or just by the
worst score". This page is that definition.

The orders hold the hundred worlds of round 3 of
[A screen of random worlds](a-screen-of-random-worlds.md), and only those:
Experiment 40's 84, drawn from the [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") the author chose on 24 October
2026, and the eight draws of Experiments 41 and 42, each played twice, a token
at a time. Round 1 drew from an older list of settings, round 2 was never
played, and neither is in the orders (the author, 8 November 2026). Every run's
name begins with "[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")" only because a drawn world's settings are kept as changes
from B1's rules; its seed says which world it is (3001 to 3100 for Experiment
40, 4001 to 4008 for the draws played a token at a time). There is no [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")
any more. B1 and [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") were where we learned to measure, and a world is no longer
judged against them.

## What healthy could mean

We want worlds in which evolution could go on for a very long time. We would
call such a world healthy if, beyond living,

1. **no one rules it:** no agent holds a large share of the links, or of the
   tokens;
2. **it holds together:** losing one agent or one link does not cut off a
   large part of it;
3. **it has room:** it is spread out like a space of a few [dimensions](measuring-space.md#three-ways-to-measure-a-networks-dimension "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), and not
   folded into a small world where everyone is a few steps from everyone;
4. **its agents differ:** in their tokens, their links, their ages, and in
   what they do in the same situation;
5. **it is not too even:** its tokens are not shared out more evenly than
   chance would share them;
6. **it loses leaves, not branches:** when agents are cut off from the rest,
   they are few and poor, not whole branches with all they hold.

These are the author's own criteria: decentralised, resilient, almost
three-dimensional, diverse and not too even, and the sixth, which the author
added on 8 November 2026.

## Every measure, a score from 0 to 1

Each criterion is read from a few measures, each averaged over a world's run
from [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 100 to its end:

<!-- table healthorder/measures -->
| criterion | measure | better |
|---|---|---|
| decentralised | links of the best-linked agent, as a share of the others | less |
| decentralised | tokens of the richest agent, as a share of all tokens | less |
| resilient | share of the agents in the 2-core | more |
| resilient | most of the world one agent's loss cuts off | less |
| resilient | most of the world one link's loss cuts off | less |
| resilient | agents on loops per link: the share of the agents in the 2-core, over half the links an agent holds (a ring scores 1, a tree 0, a world of 28 links an agent 0.07) | more |
| almost 3-D | typical distance, over the random twin's | more |
| almost 3-D | dimension, by how balls grow | nearer three |
| diverse | same-position share: of two agents in the same situation, how often they act measurably differently | more |
| diverse | share of the agents holding the most common number of tokens | less |
| diverse | share of the 2-core's agents with its most common number of links | less |
| diverse | share of the agents of the most common age | less |
| diverse | share of the agents making the more common choice, to have a child or not | less |
| not too even | Gini of the tokens, over the Gini of tokens dealt out at random at the same mean holding | more |
| few tokens culled | share of all tokens culled in a game, with the agents cut off from the largest piece | less |
<!-- /table -->

The measures are in different units: a share of the agents, a number of
dimensions, a ratio of distances. To put them together, each is read the same
way: **a world's score in a measure is the share of the round's living worlds
it does better than.** The world with the smallest hub of all scores 1 in the
hub; a world whose hub is smaller than 70% of the others' scores 0.7; the world
with the largest hub scores 1/82, the least any world scores. Worlds that tie
share their place. So every score is positive, a bigger one is always better,
and every difference counts, not only one past a line.

A criterion is the average of its measures' scores, taken as a geometric mean:
the measures multiplied, and the root taken. One poor measure then pulls a
criterion down more than one good measure lifts it, as it should: a world
where one agent is linked to nearly everyone is not decentralised, however
evenly its tokens are spread.

For *almost three-dimensional*, a world's dimension is better the nearer it is
to three, and as far off at two as at four and a half. Its width is better the
wider: the longer the typical path between two agents, against a random network
of the same size and links, the more the world is a space and the less a small
world.

### Not a tree, and not crowded with links

How many links should an agent have? Fewer is not simply better: with too few,
the network becomes a tree, which a single loss can cut. More is not simply
better either. Four worlds of round 3 hold 10 to 28 links an agent, and when
two of them were played again, the author saw at once that they were not
healthy. A measure that is low at both ends, with no number in it, is **agents
on loops per link**: the share of the agents in the 2-core, divided by half
the links an agent holds.

- A ring, where every agent is on a loop with the fewest links possible, two,
  scores 1.
- A tree has no loops and scores 0. So does a star, whose agents hang off one
  hub: seed 3020 scores 0.002.
- A world of 28 links an agent, every agent on a loop, scores 2/28 = 0.07:
  seed 3094.
- The [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world of round 3 scores 0.40.

It is a measure of the resilient criterion, beside the two cuts: a thin tree
loses on it and on the cuts, a world crowded with links on it alone. Each
heavy [frame](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.")'s 2-core is set against the links of its own iteration
(`lean` in `research/screen/health.py`; the author, 9 November 2026: "fewer is
not exactly better, because at some point it becomes a tree").

**What is left out.** Four of the author's measures are not scored, each for
its reason:

<!-- table healthorder/dropped -->
| left out | why |
|---|---|
| links | links per agent asked for a range, 3 to 12, and neither more nor fewer is simply better: agents on loops per link (`lean`) reads it against the loops the links make |
| max_tokens | the largest holding is the richest agent's share of all tokens, counted the other way round, so the two would cancel; its line kept out a world of paupers, which no living world is |
| genotypes_alike | a genotype is renewed by every mutation, so how many agents share one says how seldom brains mutate, not how differently agents act; the same-position share measures that |
| gini | drawn at B2's most even world, a baseline the screen no longer compares with; the Gini over a random deal's at the same mean holding reads the same thing against chance |
<!-- /table -->

**The price.** A score says how a world stands among round 3's worlds, not
against a fixed mark. A world that loses 2% of its tokens in a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") scores
well here because most worlds of round 3 lose more; in a round of better worlds
it would score less. For an order of round 3, that is what is wanted. To
compare rounds, they are scored together.

## Few tokens culled

The author, on 8 November 2026: "Some simulations have huge parts culled all
the time. If everywhere a little bit is culled, like leaves, I think it is not
bad. But if every few iterations a whole branch, maybe 40% of the network, it
is quite bad." And on 9 November, how to measure it: "the average percentage
of tokens culled ... only in the game phase."

**What culled means.** At the end of every phase the [engine](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") removes the agents
that starved, those left with no token. Then it keeps the largest piece of the
network and removes every agent no longer joined to it: these are the
*culled* ("Culled" in the viewer's statistics). A culled agent may hold many
tokens; it is removed only because it lost its way to the rest. What it held is
shared out among the living.

**Why tokens.** A leaf, an agent cut off alone at the edge, is mostly poor,
while a whole branch holds its share of the world's riches. So the share of
all tokens culled in a game counts a leaf for little and a branch for what it
held, with no line for what makes a piece big. The starved hold nothing, so
every token a game's [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") shares out was a culled agent's, and each run
records it in its statistics. The measure is the share of all the world's
tokens culled in a game, averaged over the games from iteration 100 on. As an
average a game, it does not depend on how long a stretch is looked at.

<!-- figure healthorder/culls -->
![Leaves or branches: what the worlds lose in their culls](../figures/healthorder/culls.svg)

**Leaves or branches: what the worlds lose in their culls.** Each dot is a living world, averaged over the games from iteration 100 to the end of its run. Across, the share of its agents a game culls (cuts off from the largest piece of the network and removes); up, the share of all the world's tokens those agents held. On the dashed diagonal the culled hold as much as the average agent does. Below it they hold less: leaves, poor agents cut off at the edge. On it and above, they are whole branches, with all they held. Agents that starved are not counted: they hold nothing.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From each run's own statistics, `GraphOfLifeRuns/<run>/stats.jsonl` (the game phase's cleanup: `orphaned`, `redistributed`, `nodes`, `starved`), read by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. For every game: the agents culled over the agents just before the cleanup (those left, the starved and the culled), and the tokens shared out by the cleanup over all the world's tokens (the starved hold none, so every token shared out was a culled agent's).
> 2. Average both over the games from iteration 100 to the end of the run.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

**What the worlds lose.** The median world culls 8.9% of all its tokens in a
game, and 9% of its agents. Seed 3092 culls 2.3% of its tokens a game, seed
3042 5.6%, seed 3026 12%, and seed 3014 24%. Where little is culled it is
mostly poor agents: in seeds 3016 and 3090 the culled agents hold about 0.37
of an average agent's tokens. Where much is culled, the culled hold their
share or more: up to 1.17 times an average agent's, in seed 3087. Whole
branches, then, are what the worlds that cull most lose.

**Only the game.** As the author asked, the culls of the reproduction phase are
left out. They are many in some worlds: seed 3090 culls 14.5% of its agents in
an average reproduction phase, all as leaves (the largest piece it ever lost at
once was 0.8% of the network), and only 0.6% in a game.

**Mending is what spares a world.** Of Experiment 40's living worlds, those that
mend their dead (the setting *mend_on_death*: a dying agent's [neighbours](the-world.md#the-network "An agent joined to another by a connection.") are
joined up again, so that no death cuts the network apart) cull a median of 4%
of their tokens in a game, those that do not, 11% (a false discovery rate of
0.01; [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md)).

## Three ways to put the criteria together

A world has six criteria, each a score from 0 to 1. They can be put together in
three ways, and each makes an order of its own:

- **multiplied:** a strong criterion makes up for a weak one, but only in
  part. A world needs to be fair in all six;
- **added:** a strong criterion makes up for a weak one in full. A world
  excellent in five can be poor in the sixth;
- **the worst:** nothing makes up for a weak criterion. A world is as healthy
  as its weakest criterion, as a body is as healthy as its sickest organ.

So that the three can be read side by side, each is shown on the criteria's
own scale, from 0 to 1, without changing its order: the product as the six
criteria's geometric average (the sixth root of their product), the sum as
their average, and the worst as it is.

**An example.** The world of seed 3077 lived through its 3,000 iterations with
a median of about 8,000 agents. It holds together better than nearly every
world: no single agent's loss cuts off more than 0.13% of it, and 94% of its
agents sit in the 2-core. But it spends 8.2 links an agent on that, and its
agents on loops per link, 0.23, is among the lowest (resilient: 0.54). It culls very little, 0.5% of its
tokens in a game (0.90). Its agents act differently, and no agent rules it
(diverse 0.69, decentralised 0.70). But it is folded: its typical distance is
only 1.07 times its [random twin](measuring-space.md#the-random-twin-is-a-world-a-space "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.")'s, and its dimension, 5.2, is far from three.
Its width beats only 10% of the worlds, its dimension 6%, and its criterion
*almost three-dimensional* scores 0.08. Added up, its six criteria make it
the **third** of the 82 worlds (an average of 0.59). Multiplied, its folding
costs it more: the **24th** (0.48). By its worst criterion, it is the **65th**
(0.08).

The first 13 of each order:

<!-- table healthorder/first -->
| place | by the product | by the sum | by the worst criterion |
|---|---|---|---|
| 1 | [s3042](run:B1-30000-b82ad0-s3042) | [s4006 s](run:B1-30000-8526cf-s4006) | [s3024](run:B1-30000-f05874-s3024) |
| 2 | [s4006 s](run:B1-30000-8526cf-s4006) | [s3042](run:B1-30000-b82ad0-s3042) | [s3042](run:B1-30000-b82ad0-s3042) |
| 3 | [s3018](run:B1-30000-af1e28-s3018) | [s3077](run:B1-30000-d6eabf-s3077) | [s3018](run:B1-30000-af1e28-s3018) |
| 4 | [s3080](run:B1-30000-625bc7-s3080) | [s4002 w](run:B1-30000-5379e2-s4002) | [s3080](run:B1-30000-625bc7-s3080) |
| 5 | [s4002 w](run:B1-30000-5379e2-s4002) | [s3067](run:B1-30000-4bee8e-s3067) | [s3057](run:B1-30000-38aee9-s3057) |
| 6 | [s3057](run:B1-30000-38aee9-s3057) | [s3018](run:B1-30000-af1e28-s3018) | [s3023](run:B1-30000-b0f6e3-s3023) |
| 7 | [s3024](run:B1-30000-f05874-s3024) | [s4008 w](run:B1-30000-1cfaa4-s4008) | [s3022](run:B1-30000-298337-s3022) |
| 8 | [s4008 w](run:B1-30000-1cfaa4-s4008) | [s3006](run:B1-30000-023c0a-s3006) | [s3005](run:B1-30000-c18a1f-s3005) |
| 9 | [s3006](run:B1-30000-023c0a-s3006) | [s3087](run:B1-30000-363694-s3087) | [s3093](run:B1-30000-26a341-s3093) |
| 10 | [s3008](run:B1-30000-e19f2d-s3008) | [s3080](run:B1-30000-625bc7-s3080) | [s3008](run:B1-30000-e19f2d-s3008) |
| 11 | [s3022](run:B1-30000-298337-s3022) | [s3045](run:B1-30000-f6d23f-s3045) | [s4006 s](run:B1-30000-8526cf-s4006) |
| 12 | [s3023](run:B1-30000-b0f6e3-s3023) | [s3064](run:B1-30000-a6a7ec-s3064) | [s3019](run:B1-30000-b1209a-s3019) |
| 13 | [s3054](run:B1-30000-d9254e-s3054) | [s3057](run:B1-30000-38aee9-s3057) | [s4004 w](run:B1-30000-17ac6e-s4004) |
<!-- /table -->

The three orders agree only in part. The order by the product and the order
by the sum have a rank [correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes "Whether two quantities go up and down together, from −1 to 1.") of 0.87, the product and the worst 0.79,
the sum and the worst only 0.46. Five worlds are among the first 13 of all
three: seeds 3042, 4006 played strongest first, 3018, 3080 and 3057. The
candidates for open-ended evolution are the worlds among the first 13 of any
of the three orders that the candidates' own rules do not rule out: 21 worlds
([Candidates for open-ended evolution](candidates-for-open-ended-evolution.md#the-shortlist)).

## The orders

<!-- figure healthorder/order -->
![The living worlds, healthiest first](../figures/healthorder/order.svg)

**The living worlds, healthiest first.** Each of the 82 worlds that lived to the end of its run is a row, in the order of its health by the product of its six criteria; each criterion is a column, coloured by the world's score in it, from dark (worse than nearly every world) to bright yellow (better than every other). A world named "w" or "s" is one of the eight draws played a token at a time, weakest first or strongest first. The 18 worlds that died are not shown: they come last in every order.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` (research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept world), and each run's own statistics for its culls, scored by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. Average every measure of a world over iterations 100 to the end of its run (the judge's averages, `research/screen/measures.py`).
> 2. Score each measure by the share of the living worlds the world does better than (`shares` in `health.py`): the best 1, the worst 1/82, ties sharing their place.
> 3. Each criterion: the geometric mean of its measures' scores. Health: the six criteria multiplied (as their geometric average), added (as their average), or the worst of them.
> 4. Rank the living worlds by the product, ties by their weakest criterion.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

In the heat map each row is a living world, the healthiest by the product at
the top, and each column a criterion, bright where the world does better than
most. The table gives every number, sorted by the product; click a column's
head to sort it by the sum, the worst or any criterion. In it, a world named
with a "w" or an "s" is one of the eight draws played a token at a time,
weakest first or strongest first. Each world's name opens it in the viewer, on
a computer that holds the runs.

<!-- table healthorder/order -->
| rank by the product | world | round | by the product | by the sum | by the worst | rank by the sum | rank by the worst | decentralised | resilient | almost 3-D | diverse | not too even | few tokens culled | agents | first half → second | weakest criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [s3042](run:B1-30000-b82ad0-s3042) | round 3 | 0.606 | 0.624 | 0.427 | 2 | 2 | 0.90 | 0.68 | 0.51 | 0.68 | 0.43 | 0.55 | 11,458 | 0.603 → 0.595 | not too even (0.43) |
| 2 | [s4006 s](run:B1-30000-8526cf-s4006) | round 3, strongest first | 0.587 | 0.647 | 0.268 | 1 | 11 | 0.95 | 0.53 | 0.81 | 0.41 | 0.27 | 0.91 | 7,166 | 0.578 → 0.553 | not too even (0.27) |
| 3 | [s3018](run:B1-30000-af1e28-s3018) | round 3 | 0.565 | 0.574 | 0.390 | 6 | 3 | 0.69 | 0.62 | 0.53 | 0.57 | 0.39 | 0.65 | 2,930 | 0.547 → 0.559 | not too even (0.39) |
| 4 | [s3080](run:B1-30000-625bc7-s3080) | round 3 | 0.549 | 0.564 | 0.376 | 10 | 4 | 0.38 | 0.47 | 0.64 | 0.54 | 0.55 | 0.80 | 2,002 | 0.548 → 0.546 | decentralised (0.38) |
| 5 | [s4002 w](run:B1-30000-5379e2-s4002) | round 3, weakest first | 0.544 | 0.578 | 0.232 | 4 | 20 | 0.58 | 0.66 | 0.67 | 0.71 | 0.62 | 0.23 | 5,258 | 0.502 → 0.562 | few tokens culled (0.23) |
| 6 | [s3057](run:B1-30000-38aee9-s3057) | round 3 | 0.541 | 0.560 | 0.366 | 13 | 5 | 0.47 | 0.53 | 0.80 | 0.70 | 0.49 | 0.37 | 5,038 | 0.535 → 0.537 | few tokens culled (0.37) |
| 7 | [s3024](run:B1-30000-f05874-s3024) | round 3 | 0.536 | 0.540 | 0.456 | 27 | 1 | 0.52 | 0.46 | 0.53 | 0.51 | 0.63 | 0.60 | 4,580 | 0.558 → 0.515 | resilient (0.46) |
| 8 | [s4008 w](run:B1-30000-1cfaa4-s4008) | round 3, weakest first | 0.528 | 0.574 | 0.244 | 7 | 15 | 0.38 | 0.60 | 0.85 | 0.60 | 0.78 | 0.24 | 5,205 | 0.510 → 0.522 | few tokens culled (0.24) |
| 9 | [s3006](run:B1-30000-023c0a-s3006) | round 3 | 0.526 | 0.572 | 0.207 | 8 | 23 | 0.85 | 0.54 | 0.62 | 0.52 | 0.21 | 0.70 | 6,907 | 0.528 → 0.531 | not too even (0.21) |
| 10 | [s3008](run:B1-30000-e19f2d-s3008) | round 3 | 0.526 | 0.548 | 0.269 | 20 | 10 | 0.27 | 0.56 | 0.51 | 0.64 | 0.68 | 0.63 | 2,113 | 0.496 → 0.539 | decentralised (0.27) |
| 11 | [s3022](run:B1-30000-298337-s3022) | round 3 | 0.525 | 0.547 | 0.310 | 22 | 7 | 0.31 | 0.50 | 0.82 | 0.57 | 0.57 | 0.50 | 5,443 | 0.509 → 0.515 | decentralised (0.31) |
| 12 | [s3023](run:B1-30000-b0f6e3-s3023) | round 3 | 0.515 | 0.537 | 0.354 | 28 | 6 | 0.38 | 0.64 | 0.54 | 0.49 | 0.82 | 0.35 | 1,609 | 0.509 → 0.517 | few tokens culled (0.35) |
| 13 | [s3054](run:B1-30000-d9254e-s3054) | round 3 | 0.513 | 0.557 | 0.241 | 16 | 18 | 0.24 | 0.38 | 0.73 | 0.50 | 0.67 | 0.82 | 3,323 | 0.542 → 0.478 | decentralised (0.24) |
| 14 | [s3087](run:B1-30000-363694-s3087) | round 3 | 0.511 | 0.568 | 0.207 | 9 | 24 | 0.90 | 0.53 | 0.83 | 0.43 | 0.50 | 0.21 | 10,456 | 0.497 → 0.509 | few tokens culled (0.21) |
| 15 | [s3064](run:B1-30000-a6a7ec-s3064) | round 3 | 0.511 | 0.560 | 0.174 | 12 | 32 | 0.57 | 0.69 | 0.17 | 0.59 | 0.61 | 0.73 | 5,346 | 0.515 → 0.512 | almost 3-D (0.17) |
| 16 | [s3051](run:B1-30000-b5a76a-s3051) | round 3 | 0.504 | 0.553 | 0.198 | 19 | 26 | 0.20 | 0.49 | 0.54 | 0.49 | 0.85 | 0.74 | 976 | 0.502 → 0.478 | decentralised (0.20) |
| 17 | [s4008 s](run:B1-30000-645972-s4008) | round 3, strongest first | 0.501 | 0.554 | 0.171 | 18 | 34 | 0.52 | 0.51 | 0.79 | 0.63 | 0.71 | 0.17 | 5,380 | 0.534 → 0.457 | few tokens culled (0.17) |
| 18 | [s4003 s](run:B1-30000-fffc7d-s4003) | round 3, strongest first | 0.498 | 0.544 | 0.220 | 24 | 21 | 0.91 | 0.44 | 0.68 | 0.54 | 0.48 | 0.22 | 8,936 | 0.494 → 0.502 | few tokens culled (0.22) |
| 19 | [s3019](run:B1-30000-b1209a-s3019) | round 3 | 0.485 | 0.520 | 0.258 | 35 | 12 | 0.46 | 0.62 | 0.26 | 0.46 | 0.44 | 0.89 | 3,040 | 0.473 → 0.474 | almost 3-D (0.26) |
| 20 | [s3004](run:B1-30000-7d5208-s3004) | round 3 | 0.485 | 0.537 | 0.234 | 29 | 19 | 0.23 | 0.50 | 0.74 | 0.39 | 0.40 | 0.95 | 1,612 | 0.467 → 0.462 | decentralised (0.23) |
| 21 | [s3047](run:B1-30000-10c9f1-s3047) | round 3 | 0.482 | 0.548 | 0.154 | 21 | 38 | 0.15 | 0.49 | 0.72 | 0.49 | 0.91 | 0.52 | 3,638 | 0.492 → 0.461 | decentralised (0.15) |
| 22 | [s3005](run:B1-30000-c18a1f-s3005) | round 3 | 0.481 | 0.509 | 0.293 | 38 | 8 | 0.71 | 0.29 | 0.62 | 0.52 | 0.30 | 0.61 | 4,119 | 0.457 → 0.501 | resilient (0.29) |
| 23 | [s3026](run:B1-30000-d6955f-s3026) | round 3 | 0.477 | 0.513 | 0.243 | 36 | 17 | 0.42 | 0.75 | 0.53 | 0.24 | 0.74 | 0.39 | 1,957 | 0.461 → 0.467 | diverse (0.24) |
| 24 | [s3077](run:B1-30000-d6eabf-s3077) | round 3 | 0.476 | 0.592 | 0.077 | 3 | 65 | 0.70 | 0.54 | 0.08 | 0.69 | 0.65 | 0.90 | 7,999 | 0.484 → 0.487 | almost 3-D (0.08) |
| 25 | [s3062](run:B1-30000-def151-s3062) | round 3 | 0.476 | 0.545 | 0.146 | 23 | 41 | 0.15 | 0.53 | 0.68 | 0.50 | 0.95 | 0.46 | 2,364 | 0.471 → 0.466 | decentralised (0.15) |
| 26 | [s3045](run:B1-30000-f6d23f-s3045) | round 3 | 0.476 | 0.561 | 0.200 | 11 | 25 | 0.20 | 0.61 | 0.20 | 0.67 | 0.93 | 0.76 | 1,401 | 0.497 → 0.444 | almost 3-D (0.20) |
| 27 | [s3058](run:B1-30000-4108dd-s3058) | round 3 | 0.473 | 0.557 | 0.109 | 15 | 54 | 0.11 | 0.48 | 0.66 | 0.56 | 0.87 | 0.67 | 1,197 | 0.456 → 0.498 | decentralised (0.11) |
| 28 | [s3021](run:B1-30000-4aff7d-s3021) | round 3 | 0.470 | 0.536 | 0.137 | 30 | 44 | 0.14 | 0.65 | 0.57 | 0.47 | 0.88 | 0.51 | 1,648 | 0.467 → 0.444 | decentralised (0.14) |
| 29 | [s4003 w](run:B1-30000-9efb1d-s4003) | round 3, weakest first | 0.466 | 0.543 | 0.134 | 25 | 46 | 0.95 | 0.44 | 0.74 | 0.48 | 0.51 | 0.13 | 10,306 | 0.489 → 0.450 | few tokens culled (0.13) |
| 30 | [s3092](run:B1-30000-7fc03f-s3092) | round 3 | 0.461 | 0.532 | 0.179 | 31 | 31 | 0.24 | 0.62 | 0.18 | 0.65 | 0.73 | 0.77 | 6,314 | 0.433 → 0.478 | almost 3-D (0.18) |
| 31 | [s3093](run:B1-30000-26a341-s3093) | round 3 | 0.457 | 0.495 | 0.271 | 41 | 9 | 0.38 | 0.27 | 0.51 | 0.31 | 0.84 | 0.66 | 3,432 | 0.453 → 0.443 | resilient (0.27) |
| 32 | [s3049](run:B1-30000-c51eb8-s3049) | round 3 | 0.447 | 0.521 | 0.124 | 33 | 48 | 0.33 | 0.58 | 0.12 | 0.58 | 0.72 | 0.79 | 5,213 | 0.441 → 0.452 | almost 3-D (0.12) |
| 33 | [s4002 s](run:B1-30000-192733-s4002) | round 3, strongest first | 0.442 | 0.529 | 0.085 | 32 | 59 | 0.73 | 0.53 | 0.60 | 0.64 | 0.59 | 0.09 | 5,624 | 0.481 → 0.360 | few tokens culled (0.09) |
| 34 | [s3067](run:B1-30000-4bee8e-s3067) | round 3 | 0.440 | 0.575 | 0.069 | 5 | 69 | 0.79 | 0.31 | 0.82 | 0.07 | 0.83 | 0.62 | 15,925 | 0.485 → 0.402 | diverse (0.07) |
| 35 | [s3016](run:B1-30000-b23381-s3016) | round 3 | 0.432 | 0.554 | 0.094 | 17 | 57 | 0.09 | 0.52 | 0.26 | 0.58 | 1.00 | 0.87 | 1,192 | 0.398 → 0.424 | decentralised (0.09) |
| 36 | [s3055](run:B1-30000-cca771-s3055) | round 3 | 0.415 | 0.445 | 0.183 | 54 | 30 | 0.61 | 0.50 | 0.57 | 0.44 | 0.37 | 0.18 | 2,552 | 0.419 → 0.416 | few tokens culled (0.18) |
| 37 | [s3061](run:B1-30000-00e3c8-s3061) | round 3 | 0.414 | 0.452 | 0.195 | 50 | 27 | 0.58 | 0.54 | 0.71 | 0.34 | 0.34 | 0.20 | 3,193 | 0.417 → 0.424 | few tokens culled (0.20) |
| 38 | [s3032](run:B1-30000-22865f-s3032) | round 3 | 0.410 | 0.444 | 0.159 | 55 | 35 | 0.44 | 0.63 | 0.48 | 0.46 | 0.16 | 0.49 | 7,294 | 0.408 → 0.407 | not too even (0.16) |
| 39 | [s3056](run:B1-30000-43985b-s3056) | round 3 | 0.404 | 0.439 | 0.248 | 59 | 14 | 0.66 | 0.27 | 0.33 | 0.25 | 0.41 | 0.71 | 3,560 | 0.408 → 0.394 | diverse (0.25) |
| 40 | [s3065](run:B1-30000-ba3afe-s3065) | round 3 | 0.403 | 0.482 | 0.132 | 42 | 47 | 0.27 | 0.63 | 0.13 | 0.38 | 0.54 | 0.94 | 2,122 | 0.409 → 0.410 | almost 3-D (0.13) |
| 41 | [s3043](run:B1-30000-02ed17-s3043) | round 3 | 0.403 | 0.461 | 0.214 | 46 | 22 | 0.43 | 0.57 | 0.21 | 0.30 | 0.28 | 0.98 | 2,515 | 0.408 → 0.385 | almost 3-D (0.21) |
| 42 | [s3078](run:B1-30000-dd27b6-s3078) | round 3 | 0.399 | 0.459 | 0.151 | 48 | 39 | 0.15 | 0.37 | 0.61 | 0.41 | 0.89 | 0.33 | 1,646 | 0.384 → 0.401 | decentralised (0.15) |
| 43 | [s3046](run:B1-30000-27bd64-s3046) | round 3 | 0.397 | 0.434 | 0.171 | 60 | 33 | 0.47 | 0.60 | 0.63 | 0.44 | 0.17 | 0.29 | 2,492 | 0.387 → 0.408 | not too even (0.17) |
| 44 | [s3084](run:B1-30000-ad76aa-s3084) | round 3 | 0.396 | 0.558 | 0.085 | 14 | 61 | 0.15 | 0.45 | 0.09 | 0.72 | 0.98 | 0.96 | 1,424 | 0.381 → 0.398 | almost 3-D (0.09) |
| 45 | [s3059](run:B1-30000-2ec87a-s3059) | round 3 | 0.395 | 0.470 | 0.122 | 44 | 50 | 0.73 | 0.22 | 0.55 | 0.51 | 0.12 | 0.68 | 4,240 | 0.391 → 0.394 | not too even (0.12) |
| 46 | [s3041](run:B1-30000-965f1e-s3041) | round 3 | 0.393 | 0.460 | 0.136 | 47 | 45 | 0.77 | 0.14 | 0.41 | 0.31 | 0.35 | 0.78 | 4,434 | 0.357 → 0.430 | resilient (0.14) |
| 47 | [s3070](run:B1-30000-a4d3a0-s3070) | round 3 | 0.387 | 0.428 | 0.190 | 61 | 28 | 0.78 | 0.19 | 0.32 | 0.45 | 0.52 | 0.30 | 8,160 | 0.387 → 0.386 | resilient (0.19) |
| 48 | [s3090](run:B1-30000-05ba40-s3090) | round 3 | 0.381 | 0.511 | 0.082 | 37 | 63 | 0.18 | 0.51 | 0.08 | 0.55 | 0.76 | 0.99 | 1,537 | 0.405 → 0.318 | almost 3-D (0.08) |
| 49 | [s3028](run:B1-30000-2eef58-s3028) | round 3 | 0.380 | 0.414 | 0.244 | 62 | 16 | 0.63 | 0.26 | 0.55 | 0.25 | 0.24 | 0.56 | 5,865 | 0.363 → 0.376 | not too even (0.24) |
| 50 | [s3082](run:B1-30000-1141ea-s3082) | round 3 | 0.379 | 0.408 | 0.157 | 65 | 37 | 0.51 | 0.16 | 0.46 | 0.41 | 0.56 | 0.34 | 3,294 | 0.389 → 0.365 | resilient (0.16) |
| 51 | [s3013](run:B1-30000-7f7cd5-s3013) | round 3 | 0.378 | 0.446 | 0.122 | 52 | 51 | 0.91 | 0.28 | 0.44 | 0.46 | 0.46 | 0.12 | 11,011 | 0.378 → 0.375 | few tokens culled (0.12) |
| 52 | [s3071](run:B1-30000-605c0d-s3071) | round 3 | 0.377 | 0.445 | 0.119 | 53 | 52 | 0.12 | 0.31 | 0.40 | 0.36 | 0.77 | 0.72 | 6,215 | 0.356 → 0.375 | decentralised (0.12) |
| 53 | [s4005 s](run:B1-30000-6f6c09-s4005) | round 3, strongest first | 0.373 | 0.439 | 0.085 | 58 | 60 | 0.63 | 0.39 | 0.61 | 0.49 | 0.09 | 0.43 | 3,930 | 0.395 → 0.363 | not too even (0.09) |
| 54 | [s3083](run:B1-30000-0d7e0c-s3083) | round 3 | 0.371 | 0.440 | 0.090 | 57 | 58 | 0.71 | 0.09 | 0.33 | 0.47 | 0.45 | 0.59 | 4,148 | 0.365 → 0.369 | resilient (0.09) |
| 55 | [s3012](run:B1-30000-02fa53-s3012) | round 3 | 0.365 | 0.412 | 0.186 | 63 | 29 | 0.61 | 0.19 | 0.19 | 0.32 | 0.60 | 0.57 | 2,522 | 0.363 → 0.367 | almost 3-D (0.19) |
| 56 | [s3050](run:B1-30000-ebe08c-s3050) | round 3 | 0.362 | 0.451 | 0.065 | 51 | 70 | 0.06 | 0.50 | 0.52 | 0.37 | 0.80 | 0.44 | 2,980 | 0.335 → 0.358 | decentralised (0.06) |
| 57 | [s3033](run:B1-30000-631e49-s3033) | round 3 | 0.356 | 0.542 | 0.042 | 26 | 75 | 0.04 | 0.53 | 0.16 | 0.70 | 0.96 | 0.85 | 1,792 | 0.323 → 0.426 | decentralised (0.04) |
| 58 | [s3091](run:B1-30000-42d88d-s3091) | round 3 | 0.356 | 0.404 | 0.110 | 67 | 53 | 0.52 | 0.36 | 0.59 | 0.53 | 0.32 | 0.11 | 2,288 | 0.348 → 0.341 | few tokens culled (0.11) |
| 59 | [s3030](run:B1-30000-90fc91-s3030) | round 3 | 0.352 | 0.501 | 0.042 | 40 | 74 | 0.73 | 0.53 | 0.04 | 0.21 | 0.66 | 0.83 | 20,395 | 0.370 → 0.353 | almost 3-D (0.04) |
| 60 | [s3098](run:B1-30000-794ae8-s3098) | round 3 | 0.352 | 0.402 | 0.123 | 68 | 49 | 0.63 | 0.12 | 0.53 | 0.29 | 0.29 | 0.54 | 4,796 | 0.353 → 0.353 | resilient (0.12) |
| 61 | [s4004 w](run:B1-30000-17ac6e-s4004) | round 3, weakest first | 0.345 | 0.350 | 0.256 | 72 | 13 | 0.37 | 0.34 | 0.36 | 0.45 | 0.33 | 0.26 | 1,101 | 0.348 → 0.340 | few tokens culled (0.26) |
| 62 | [s3063](run:B1-30000-924e70-s3063) | round 3 | 0.345 | 0.470 | 0.098 | 45 | 56 | 0.92 | 0.54 | 0.84 | 0.26 | 0.10 | 0.16 | 14,998 | 0.342 → 0.351 | not too even (0.10) |
| 63 | [s3095](run:B1-30000-3e5856-s3095) | round 3 | 0.339 | 0.520 | 0.073 | 34 | 66 | 0.08 | 0.48 | 0.07 | 0.71 | 0.94 | 0.84 | 3,517 | 0.298 → 0.397 | almost 3-D (0.07) |
| 64 | [s3007](run:B1-30000-8119cd-s3007) | round 3 | 0.337 | 0.407 | 0.143 | 66 | 42 | 0.14 | 0.37 | 0.43 | 0.24 | 0.99 | 0.27 | 2,768 | 0.316 → 0.353 | decentralised (0.14) |
| 65 | [s3099](run:B1-30000-1ab8a0-s3099) | round 3 | 0.336 | 0.409 | 0.138 | 64 | 43 | 0.72 | 0.18 | 0.76 | 0.14 | 0.38 | 0.28 | 21,743 | 0.338 → 0.314 | diverse (0.14) |
| 66 | [s3001](run:B1-30000-68cda7-s3001) | round 3 | 0.334 | 0.457 | 0.098 | 49 | 55 | 0.80 | 0.51 | 0.94 | 0.25 | 0.15 | 0.10 | 17,728 | 0.345 → 0.288 | few tokens culled (0.10) |
| 67 | [s3052](run:B1-30000-0d058a-s3052) | round 3 | 0.327 | 0.354 | 0.158 | 71 | 36 | 0.16 | 0.38 | 0.60 | 0.33 | 0.26 | 0.40 | 7,558 | 0.294 → 0.367 | decentralised (0.16) |
| 68 | [s3094](run:B1-30000-4ac946-s3094) | round 3 | 0.318 | 0.474 | 0.030 | 43 | 77 | 0.43 | 0.40 | 0.03 | 0.30 | 0.70 | 1.00 | 11,200 | 0.318 → 0.315 | almost 3-D (0.03) |
| 69 | [s3066](run:B1-30000-dfd51b-s3066) | round 3 | 0.314 | 0.502 | 0.017 | 39 | 79 | 0.39 | 0.43 | 0.02 | 0.45 | 0.79 | 0.93 | 10,027 | 0.317 → 0.346 | almost 3-D (0.02) |
| 70 | [s3014](run:B1-30000-7d129f-s3014) | round 3 | 0.309 | 0.442 | 0.049 | 56 | 73 | 0.97 | 0.53 | 0.62 | 0.29 | 0.20 | 0.05 | 11,026 | 0.257 → 0.319 | few tokens culled (0.05) |
| 71 | [s4004 s](run:B1-30000-2bac4d-s4004) | round 3, strongest first | 0.303 | 0.323 | 0.146 | 78 | 40 | 0.44 | 0.39 | 0.34 | 0.39 | 0.23 | 0.15 | 1,244 | 0.315 → 0.312 | few tokens culled (0.15) |
| 72 | [s3085](run:B1-30000-801b34-s3085) | round 3 | 0.283 | 0.371 | 0.073 | 69 | 67 | 0.84 | 0.25 | 0.41 | 0.17 | 0.07 | 0.48 | 4,259 | 0.283 → 0.267 | not too even (0.07) |
| 73 | [s3040](run:B1-30000-36d076-s3040) | round 3 | 0.245 | 0.287 | 0.084 | 80 | 62 | 0.08 | 0.33 | 0.51 | 0.17 | 0.22 | 0.41 | 4,090 | 0.256 → 0.224 | decentralised (0.08) |
| 74 | [s3037](run:B1-30000-845393-s3037) | round 3 | 0.235 | 0.327 | 0.080 | 77 | 64 | 0.88 | 0.08 | 0.21 | 0.23 | 0.11 | 0.45 | 10,156 | 0.243 → 0.208 | resilient (0.08) |
| 75 | [s3072](run:B1-30000-f05562-s3072) | round 3 | 0.221 | 0.279 | 0.073 | 81 | 68 | 0.44 | 0.47 | 0.41 | 0.10 | 0.18 | 0.07 | 20,226 | 0.227 → 0.203 | few tokens culled (0.07) |
| 76 | [s3044](run:B1-30000-63827e-s3044) | round 3 | 0.203 | 0.342 | 0.037 | 74 | 76 | 0.69 | 0.35 | 0.55 | 0.40 | 0.04 | 0.04 | 3,543 | 0.223 → 0.201 | not too even (0.04) |
| 77 | [s4007 s](run:B1-30000-a8a2d2-s4007) | round 3, strongest first | 0.197 | 0.347 | 0.024 | 73 | 78 | 0.54 | 0.47 | 0.72 | 0.27 | 0.05 | 0.02 | 6,827 | 0.226 → 0.168 | few tokens culled (0.02) |
| 78 | [s3076](run:B1-30000-66c764-s3076) | round 3 | 0.192 | 0.336 | 0.012 | 75 | 82 | 0.84 | 0.21 | 0.44 | 0.14 | 0.01 | 0.38 | 4,636 | 0.193 → 0.208 | not too even (0.01) |
| 79 | [s3038](run:B1-30000-1ad8a9-s3038) | round 3 | 0.186 | 0.291 | 0.061 | 79 | 71 | 0.96 | 0.09 | 0.18 | 0.32 | 0.13 | 0.06 | 6,400 | 0.167 → 0.195 | few tokens culled (0.06) |
| 80 | [s4007 w](run:B1-30000-04d3eb-s4007) | round 3, weakest first | 0.164 | 0.362 | 0.012 | 70 | 80 | 0.69 | 0.52 | 0.63 | 0.29 | 0.02 | 0.01 | 6,536 | 0.116 → 0.178 | few tokens culled (0.01) |
| 81 | [s3089](run:B1-30000-77ad6a-s3089) | round 3 | 0.129 | 0.155 | 0.061 | 82 | 72 | 0.06 | 0.10 | 0.21 | 0.18 | 0.06 | 0.32 | 5,620 | 0.121 → 0.091 | not too even (0.06) |
| 82 | [s3020](run:B1-30000-511930-s3020) | round 3 | 0.105 | 0.328 | 0.012 | 76 | 81 | 0.01 | 0.04 | 0.10 | 0.04 | 0.90 | 0.88 | 9,388 | 0.111 → 0.065 | decentralised (0.01) |
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

## What the orders show

**The first by the product is seed 3042.** It is among the best in being
decentralised (0.90), holds together well (resilient 0.68), and is no worse
than middling in anything: its weakest criterion, *not too even*, scores
0.43. It mends its
dead, plays the lottery, searches for partners by jumps, mixes [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") at birth,
allows gifts and keeps a link only while it is paid for or used.

**The worst criterion differs from world to world.** It is *few tokens
culled* in 21 worlds, *decentralised* in 19, *almost three-dimensional* in 15,
*not too even* in 14, *resilient* in 9, and *diverse* in 4.

**The candidates mostly mend their dead.** Of the 21 candidates, 16 mend their
dead, against 44 of the 82 living worlds; 14 play the lottery, 12 allow gifts
and 13 put a token in every link.

**Seed 3092 meets every one of the author's lines, and stands near the middle.**
Its hub (2.7% of the others) and its richest agent (2.0% of the tokens) are
under the lines, but larger than in three worlds of four: decentralised 0.24.
Its width beats 22% of the worlds and its dimension, 4.3, only 15%: almost
three-dimensional 0.18. It is the 30th by the product, the 31st by the sum and
the 31st by its worst criterion.

**At the bottom are the stars and the worlds too even.** The last by the
product is seed 3020, where one agent is linked to 99.5% of the others, and
before it seed 3089, where one is linked to 31%. Before them come seeds 4007
played weakest first, 3038, 3076 and 4007 played strongest first: their tokens
are among the most evenly spread of the round, and three of them cull about a
quarter of all their tokens in every game.

## Healthier or sicker with time?

<!-- figure healthorder/halves -->
![Healthier or sicker with time?](../figures/healthorder/halves.svg)

**Healthier or sicker with time?** Each dot is a living world: its health by the product, scored over the first half of its run (iterations 100 to the middle) and over the second (the middle to the end), every half against every half of every world. Above the dashed diagonal a world grew healthier; below it, sicker.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` (research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept world), and each run's own statistics for its culls, scored by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. Average every measure of each half of every living world's run.
> 2. Score all the halves together, as the whole runs are scored.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

Health does not drift over a run. Scored by the product, every half of every
run against every other, 17 worlds grew healthier from the first half of their
run to the second by more than 0.02, 24 grew sicker, and 41 stayed as they
were; the median change is nil.

## Every measure against the worlds

<!-- figure healthorder/measures -->
![Every measure, against the living worlds](../figures/healthorder/measures.svg)

**Every measure, against the living worlds.** One chart a measure: each dot is a living world's average of it, from iteration 100 to the end of its run; the short bar is the median. The title says which criterion the measure belongs to and which way is better. A world's score in a measure is the share of the other worlds it does better than, so only the order of the dots counts. The dots are spread sideways only so that they do not hide each other.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 tokens, played for 3,000 iterations)
>
> **Data.** From the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` (research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept world), and each run's own statistics for its culls, scored by `research/screen/health.py` into `research/screen/health.json`.
>
> 1. Read each living world's averages from `health.json`.
> 2. One dot per world and measure.
>
> **To make it again:** `python3 book_figures.py healthorder`.
<!-- /figure -->

## The half-cut exponent

On 24 October 2026 the author asked for two numbers of health and chose to see
them on every world of round 3 first
([Two numbers reported, not judged](measuring-health.md#two-numbers-reported-not-judged)).
The [same-position share](measuring-health.md#the-same-position-share),
for very different behaviours living side by side, is one of the measures of
*diverse*. The [half-cut exponent](measuring-health.md#the-half-cut-exponent),
for a world that on the largest scale looks like a space, is not one of the
measures of *almost three-dimensional*. Halving a body of *d* dimensions cuts
*N*^((*d* − 1)/*d*) of its links, so 1/2 is a plane's and 2/3 three
dimensions'. But a network as sparse as our worlds halves more cheaply than a
lattice, by how sparse it is as much as by its dimension. Random networks of
known shape show it, each of 4,000 agents put at random in a square or a cube
and joined to those nearest:

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
3.6 links an agent, and their exponents follow their links (a correlation of
0.67) and their dimension by balls (0.91). So the exponent would score a world
for being dense as much as for having room. The fair test compares each world
with a random network of its own size and density in a cube: a world as hard
to halve as that twin is three-dimensional on the largest scale. That is a step
for the author to approve
([For the author to check](#for-the-author-to-check)).

## For the author to check

The orders are a proposal, and the author corrects them. The quickest way:
1. Open the first worlds of each order and the living worlds at the bottom in
   the viewer (each name in the table opens its world), and look at them late
   in their run.
2. Note the worlds that are out of place: one you would call healthy that
   sits low, or a sick world that sits high.
3. The table says, for every world, how it scores in each criterion and which
   is its weakest. So a world out of place points at a measure to change, or
   at a criterion that counts for too much or too little.

Two questions the orders cannot answer by themselves:
- **Should a world be scored against round 3, or against fixed marks?** Here a
  score is the share of the round's worlds a world beats. Fixed marks would
  keep a world's score when other rounds are added, but they need a mark for
  every measure, which is what the lines were.
- **Should the half-cut exponent be read against a twin in space?** Each world
  would be compared with a random network of its own size and density in a
  cube. That takes a few minutes of computing a world, and needs no world
  played again.

## What we learned

- **Measures in different units can be put together** by scoring each as the
  share of the worlds a world beats; then every difference counts, and the
  criteria can be added, multiplied or compared.
- **How the criteria are put together matters as much as the criteria.**
  Added, multiplied and the worst make three orders that agree only in part:
  a world folded into a small world can be the first by one and the 63rd by
  another.
- **Too many links is as unhealthy as too few.** Agents on loops per link is
  low for a tree and for a world crowded with links; counted in resilience, it
  sends the four worlds of 10 to 28 links an agent down the orders.
- **Five worlds are among the first of all three orders,** seeds 3042, 4006
  played strongest first, 3018, 3080 and 3057.
- **The share of tokens culled tells leaves from branches** without a line for
  what makes a piece big: the worlds that cull little lose poor agents, those
  that cull much lose their share of the riches.
- **Mending the dead spares a world its culls,** and 16 of the 21 candidates
  mend.
- **Health does not drift** over a run.

## Where to go next

- [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md):
  which settings make a world healthy, and which kill it.
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md):
  the same worlds read for what evolution needs: heredity, selection and
  change.
- [What makes a world interesting](measuring-health.md#what-makes-a-world-interesting): the
  author's criteria and lines this order starts from.
- [A screen of random worlds](a-screen-of-random-worlds.md): how the worlds
  were drawn, and what became of them.

## How this was made

The worlds are the runs of Experiments 40, 41 and 42, listed with every
setting on the page [A screen of random worlds](a-screen-of-random-worlds.md).
Nothing is played again: every number is read from what the worlds recorded,
by the screen's judge (`research/screen/measures.py`), and kept, one file a
world, in `GraphOfLifeRuns/.pilots/screen/worlds/`; the culls come from each
run's own statistics. The averages run from iteration 100 to the end of each
run, 3,000 iterations.

`research/screen/health.py` turns the averages into the orders: `MEASURES`
holds each criterion's measures and which way is better, `DROPPED` the measures
left out with their reasons, `shares` scores a measure by the share of the
worlds a world beats, and `score` makes the criteria and the three orders.
`python3 research/screen/health.py E41 E42 E40` writes
`research/screen/health.json`; with `--read 3` it first makes, on three
processes, any reading a world still lacks. The figures and tables of this
page are drawn from that file by `python3 book_figures.py healthorder`.

An earlier version of the sixth criterion counted the pieces culled at once,
read again from every phase of every world's frames, and a piece was big past
a quarter of the network. It is in the notebook's history
(`research/screen/culls.py`, at commit a9621d7); the share of tokens culled
replaced it as simpler, read straight from what each run recorded.

## How our view changed

- **9 November 2026, later.** Health came to count links again, as agents on
  loops per link in the resilient criterion, after two of the worlds the
  shortlist sent to Experiment 43 grew 10 to 32 links an agent (the author:
  "fewer is not exactly better, because at some point it becomes a tree"). The
  dense worlds fell down the order by the sum: seed 3094 from the 14th to the
  43rd, 3030 from the 13th to the 40th, 3095 from the 6th to the 34th, 3066
  from the 7th to the 39th, 3084 from the 2nd to the 14th. The orders changed
  little otherwise (rank correlations of 0.97, 0.93 and 0.98 with what they
  were), and agree more with each other. The shortlist went from 24 worlds to
  21.
- **11 November 2026.** Health is frozen at the version of 9 November, this
  page's, and used only as a filter from this day: worlds that are dead, a
  star, uniform or frozen are left out, and worlds are ranked by evolution. The
  orders are kept as they were.

