# Measuring health: criteria, the health bar and diversity

> [!question] Questions of this page
> - What do we look for in a world before we ask how it evolves?
> - How is health measured now, and what is it used for since 11 November 2026?
> - What were the seven marks of the health bar?
> - How much do a world's [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") differ, in one number?

## In short

Health says whether a world could hold long evolution at all: whether it lives,
holds together, has room, and has agents that differ. The definition in force
is the author's of 9 November 2026. It scores fifteen measures by how many of
round 3's living worlds a world does better than, and puts six criteria
together in three ways ([Which worlds are
healthy?](which-worlds-are-healthy.md)). From 11 November 2026 health is frozen
at that version and used only as a filter: worlds that are dead, a star,
uniform or frozen are left out, and worlds are ranked by evolution. A table
lists the earlier versions. They are kept in full below, because round 3's
pages were written with them and the open-ended test still asks for the health
bar and the diversity score.

It gathers what were 3 pages of their own, one section each:

- [What makes a world interesting](#what-makes-a-world-interesting): What do we look for in a world before we ask how it evolves?
- [The health bar](#the-health-bar): The seven marks a world's network must meet to count as healthy, why each is there, and how a world is read against them.
- [The diversity score](#the-diversity-score): How much the agents of a world differ, in one number, and how much their decisions differ.

## From 11 November 2026: health as a filter

From 11 November 2026, health is frozen at the version of 9 November and used
only as a filter. Worlds that are dead, a star, uniform or frozen are left out.
The worlds that are left are ranked by evolution, against null models, and not
by health ([What this notebook is about](what-this-notebook-is-about.md)). So a
world is chosen for what the notebook asks, whether it keeps evolving, and not
for how healthy it looks.

## The definition in force: health of 9 November 2026

Health, as the author defined it on 9 November 2026, reads six criteria from
fifteen measures of a world. Each measure is averaged over the world's run,
from iteration 100 to its end ([Every measure, a score from 0 to
1](which-worlds-are-healthy.md#every-measure-a-score-from-0-to-1) lists them):

- **decentralised:** the links of the best-linked agent, and the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") of the
  richest, each as a share;
- **resilient:** the share of the agents in the 2-core, the most of the world
  that one agent's or one link's loss cuts off, and agents on loops per link,
  which is low both for a tree and for a world crowded with links;
- **almost three-dimensional:** the typical distance over the random twin's,
  and a dimension near three;
- **diverse:** the [same-position share](#the-same-position-share), and how
  alike the agents are in tokens, in the links of the 2-core, in age and in the
  choice to have a child;
- **not too even:** the Gini of the tokens over the Gini of the same tokens
  dealt out at random;
- **few tokens culled:** the share of all the tokens culled in a game ([Few
  tokens culled](which-worlds-are-healthy.md#few-tokens-culled)).

No measure has a line. A world's score in a measure is the share of round 3's
living worlds it does better than, so every score is positive, and a bigger one
is always better. A criterion is the geometric mean of its measures' scores.
The six criteria are put together in three ways, each an order of its own:
multiplied, added, or the worst of them ([Three ways to put the criteria
together](which-worlds-are-healthy.md#three-ways-to-put-the-criteria-together)).
Four measures of the version before are left out, each for its reason: links
per agent, the largest holding, how many agents share a genotype, and [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s line
for the Gini.

## Earlier versions

| version | in use | what it asked | what it judged |
|---|---|---|---|
| [The health bar](#the-health-bar) | from 12 October 2026 | seven marks of a world's network at one moment, each met or missed | B2 and the worlds built from it; part 4 of the [open-ended test](measuring-evolution.md#what-would-count-as-open-ended-evolution) still asks for every mark |
| [The diversity score](#the-diversity-score) | from 16 October 2026 | four spreads of a world's agents in one number, ranked among frozen worlds | the search for a healthy, rich world; part 4 of the open-ended test |
| [The criteria of rounds 1 and 2](#the-criteria-of-rounds-1-and-2) | 23 October 2026 | six criteria read at iteration 2,000, each a yes or a no | round 1 of the screen (Experiment 38) |
| [The six criteria of round 3](#what-makes-a-world-interesting) | 24 October to 7 November 2026 | the same six, averaged over iterations 100 to 3,000, with looser lines | round 3, as first judged (Experiments 40 to 42) |
| a first order | 7 to 9 November 2026 | each line read as the share of what it asks, up to the line | round 3; the author found it still a yes or a no past the lines |
| [the definition in force](#the-definition-in-force-health-of-9-november-2026) | from 9 November 2026 | fifteen measures, each scored against round 3's living worlds; six criteria; three orders | round 3's orders and shortlist (Experiment 43); from 11 November, a filter only |

The earlier versions are kept below as they were. Round 3's pages and the
theses of their day were written with them, and the open-ended test still asks
for the health bar and the diversity score.

## What makes a world interesting

*This is how health was judged from 24 October to 7 November 2026, as round 3
was first judged. [The definition in
force](#the-definition-in-force-health-of-9-november-2026) replaced it on 9
November.*

We are looking for worlds in which evolution could go on for a very long
time. Such a world has to stay alive. It must not hang on a few agents, and it
must not fall apart when one agent dies. It should look like a space of about
three dimensions, as our own world does. And its agents should differ from
one another, in tokens, links, age, genes and decisions, without the tokens
being spread too evenly.

We turn this picture into six criteria. From round 3 of the screen on, every
number behind them is an **average over a world's whole run**, from iteration
100 to its end. The first hundred iterations are ignored: every world starts
the same way, and a world shows what it is only once it has settled.
- A world that meets all six criteria is **healthy**.
- What makes a world **interesting** is decided with the author once round 3
  has run. Until then a world counts as interesting, provisionally, when it is
  alive and misses at most two of the other five.

Each criterion is made of one or more **lines**: a number, and which side of
it passes. The author controls every line, sees every world's value beside
it, and sees how many worlds would pass if it moved. Rounds 1 and 2 were
judged by stricter criteria, at a single moment; they are kept at the
[end of this page](#the-criteria-of-rounds-1-and-2).

### The six criteria

<!-- table criteria/criteria -->
| criterion | what it asks |
|---|---|
| alive | played to the end of its run (3,000 iterations in round 3) and never down to 20 agents |
| decentralised | no agent linked to more than 5% of the others, and 3 to 12 links per agent |
| resilient | at least 80% of the agents in the 2-core, and no single agent's loss cuts off more than 15% of the world, nor a single link's more than 15% (the rules keep a world in one piece) |
| almost three-dimensional | a typical distance at least 1.3 times its random twin's, and a dimension of 2 to 5 |
| diverse | statistic by statistic: fewer than 80% of the agents hold the same number of tokens, and some agent holds more than 3; fewer than 80% of the 2-core's agents have the same number of links; fewer than 80% have the same age, and fewer than 80% the same genotype; and fewer than 95% make the same choice, to have a child or not |
| not too even | a Gini of the tokens of at least 0.46 (B2's most even world in Experiment 16) |
<!-- /table -->

Each criterion stands for one part of the picture.

1. **Alive.** A world that died tells us nothing about long evolution. A
   world dies when it falls to 20 agents. In round 3 a world must live through
   all 3,000 of its iterations.
2. **Decentralised.** If one agent holds links to many others, the world
   depends on it: when that hub dies, everything changes at once. We ask that
   no agent links to more than 5% of the world, and that agents have 3 to 12
   links each, as a network that is neither a tree nor a tangle.
3. **Resilient.** A healthy world does not break apart when one agent or one
   link disappears. We ask that most agents (80%) sit on loops (the
   *2-core*), and that no single loss cuts off more than 15% of the agents.
   The worlds are small, a few thousand agents, so the author allows a larger
   share than the [health bar](how-to-tell-a-world-that-keeps-evolving.md)'s 2%
   and 1%. A world is always in one piece: whatever is cut off from its
   largest piece dies at once.
4. **Almost three-dimensional.** The physical side of the project asks
   whether a space can grow out of the agents' links
   ([Physical inspiration](physical-inspiration.md)). We measure a world's
   dimension by how fast the number of agents within a given distance grows,
   and ask for 2 to 5. A tangled network can also read "three dimensions", so
   we ask a second thing: a typical distance between two agents at least 1.3
   times that in a *random twin*, a network with the same number of links per
   agent wired at random. A folded world fails this; a true space passes
   ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)).
5. **Diverse.** Evolution needs differences to choose from. Each statistic is
   judged on its own, not mixed into one score:
   - **tokens:** not diverse if 80% or more of the agents hold the same number
     of tokens, or if no agent holds more than 3;
   - **links:** counted in the 2-core only, since leaves do not make a network
     bad: not diverse if 80% or more of the core's agents have the same number
     of links;
   - **ages** and **genotypes:** not diverse if 80% or more of the agents share
     one age, or carry one genotype;
   - **decisions:** not diverse if 95% or more of the agents make the same
     choice, to have a child or not. A world where every agent has a child in
     every iteration, or none does, is sorted out.
6. **Not too even.** If almost every agent holds the same few tokens, there
   is little to win and little to select. We ask for a
   [Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-entropy-and-mobility) of at least 0.46, the Gini of B2's
   most even world in Experiment 16. Round 3 has no B2 worlds of its own, so
   this line is a fixed number.

### The lines

Every line, with the criterion it belongs to, where it is drawn, and the
values the sensitivity table moves it to:

<!-- table criteria/lines -->
| what is measured | criterion | passes | the sensitivity table tries |
|---|---|---|---|
| links of the best-linked agent, as a share of the others | decentralised | at most 0.05 | 0.02, 0.05, 0.1, 0.2 |
| links per agent | decentralised | at least 3 | 2, 3, 4 |
| links per agent | decentralised | at most 12 | 8, 12, 20 |
| share of the agents in the 2-core | resilient | at least 0.8 | 0.6, 0.7, 0.8, 0.9 |
| most of the world one agent's loss cuts off | resilient | at most 0.15 | 0.02, 0.05, 0.1, 0.15, 0.2, 0.3 |
| most of the world one link's loss cuts off | resilient | at most 0.15 | 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 |
| typical distance, over the random twin's | almost three-dimensional | at least 1.3 | 1, 1.1, 1.2, 1.3, 1.5, 2 |
| dimension, by how balls grow | almost three-dimensional | at least 2 | 1.5, 2, 2.3 |
| dimension, by how balls grow | almost three-dimensional | at most 5 | 4, 5, 6 |
| share of the agents holding the most common number of tokens | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| the largest holding | diverse | above 3 | 2, 3, 5, 10 |
| share of the 2-core's agents with its most common number of links | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| share of the agents of the most common age | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| share of the agents carrying the most common genotype | diverse | below 0.8 | 0.6, 0.7, 0.8, 0.9, 0.95 |
| share of the agents making the more common choice, to have a child or not | diverse | below 0.95 | 0.7, 0.8, 0.9, 0.95, 0.99, 1 |
| Gini of the tokens | not too even | at least 0.46 | 0.3, 0.4, 0.46, 0.5, 0.6 |
<!-- /table -->

**Where the numbers come from.** A world records a row of statistics every
iteration, a frame of every phase, and the whole world every 250 iterations.
The judge reads each number where it is kept, as often as it is kept, and
averages it from iteration 100 to the end:
- every iteration: the number of agents, their links, the hub, the largest
  holding, the Gini, and the share of agents that had a child;
- every 25th iteration: the dimension, and from the game's frame how alike
  the agents are, the 2-core, and the worst cuts;
- every world kept (every 250 iterations): the width against the random
  twin, and the two numbers of the next section.

**How alike.** For tokens, ages, genotypes and the core's links, "how alike"
is the share of agents that share the most common value: 1 when all are the
same. For the choice to have a child it is the share making the more common
choice, at each iteration: 0.5 when half the agents have a child, 1 when all
of them do or none does.

**The sensitivity table.** For each line, the judge moves that line alone
through the values in the last column. For each value it counts how many
living worlds pass the line, meet its criterion, are healthy and are
interesting. Where a count jumps between two values, the verdict depends on
exactly where we draw the line.

### Two numbers reported, not judged

The author asked for one more number for health: whether a world, on the
largest scale, looks like a space of three dimensions, really spread out,
while very different behaviours survive in it. Two numbers try to say this.
The judge reports them for every world of round 3, and their lines will be
set together afterwards.

<!-- table criteria/reported -->
| number | what it says |
|---|---|
| half-cut exponent | the links crossing the cheapest cut into halves found, as a power of the number of agents: 2/3 for a body of three dimensions, near 0 for a bottleneck, near 1 for a folded small world |
| half-cut exponent of the random twin | the same for the world's random twin, which is no space at all |
| same-position share | of two agents put in the same situation, how often they stake measurably differently |
| effective kinds of staking | how many kinds of staking the agents fall into, counted by their shares |
| doctrine alike | the share of agents making the more common choice between spreading their stake and staking it all on one place |
| having a child | the share of agents that have a child in an iteration |
| agents | how many agents the world holds |
<!-- /table -->

#### The half-cut exponent

Cut a world into two halves of equal size, as cheaply as you can, and count
the links that cross the cut.
- In a body of three dimensions the cheapest cut is a flat slice. A cube of
  N agents has halves that meet on a square of N^(2/3) agents.
- A world that hangs together at one bottleneck can be halved through a few
  links, whatever its size.
- A folded small world, where everyone is a few steps from everyone, needs a
  cut that grows like N itself.

So the exponent of the cut, ln C / ln N for C crossing links among N agents,
is about 2/3 for a space, near 0 for a bottleneck and near 1 for a folded
world.

Nobody can find the cheapest cut of a network of thousands of agents exactly.
The judge uses a standard shortcut. Think of the network as a drum: its
slowest ways to vibrate move one region up while another goes down, and the
line between them runs where few links join the two. Each of the ten slowest
ways (the eigenvectors of the network's Laplacian with the smallest
eigenvalues) splits the agents at its median, and the cheapest of the ten
splits counts. One alone is not enough: the slowest ways vibrate at nearly the
same rate, and in one world of round 3 the slowest cut 121 links where the
third cut 62. On a box of 16 by 8 by 8 agents the judge finds the slice of 64
links across its long side, and on real worlds the same ten cuts as an exact
calculation. The world's random twin is cut the same way, for comparison: a
twin is no space at all.

#### The same-position share

Put two agents in exactly the same situation: the same [neighbours](the-world.md#the-network "An agent joined to another by a connection."), the same
tokens, the same [messages](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read."). Do they stake their tokens differently? The judge
takes 400 of a world's agents and 200 situations from the world, plays every
one of those agents' [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") in each, and counts the pairs of agents whose
stakes differ by more than one brain differs from itself between two tries.
In a world where every agent would act alike in the same place the share is
near 0; in a world of many kinds of behaviour it is near 1. It is the measure
of [How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents), read at
every world kept.

### Healthy, and interesting

A world that meets all six criteria is healthy.

Worlds made with the same [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") but different [seeds](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") differ. A setting
that gives a healthy world in one seed may miss a line in another, and we do
not want to throw it away for that. So a world may miss some criteria and
still be **interesting**. How many, and which, will be decided with the
author once round 3 has run and we have looked at its worlds. Until then the
judge counts a world as interesting when it is alive and misses at most two of
criteria 2 to 6, the author's first answer.

Round 3 has run. Instead of a verdict, the author then asked (7 November
2026) for an order of every world drawn at random, from the most healthy to
the least, by a definition they can check
([Which worlds are healthy?](which-worlds-are-healthy.md)). Its definition of
9 November is now
[the definition in force](#the-definition-in-force-health-of-9-november-2026),
and the lines of this section are an earlier version.

### When a setting is marked bad

In the [screen of random worlds](a-screen-of-random-worlds.md),
every world draws its settings at random, fifty or a hundred worlds at a time.
After each round we look at every value a setting took. A value is **marked
bad**, and drawn no more, only when both of these hold:
- it was tried in **at least ten worlds** where it could make a difference;
- **every one of them died**, or **none of them was interesting**.

Why ten? One or two dull worlds can be bad luck. Ten worlds that all failed
are a pattern. Beside each value marked bad, the judge also prints how likely
such a run of failures would be by chance alone, if the value made no
difference at all. That number is a warning, not part of the rule: when
interesting worlds are rare overall, ten failures in a row come easily. While
interesting is only provisional, so is every value the judge marks bad.

The judge is `research/screen/judge.py`, and it reads the numbers with
`research/screen/measures.py`. The criteria and their lines live in one list,
`research/screen/criteria.py`, which this page's tables are made from, so the
page and the judge always agree.

### The criteria of rounds 1 and 2

Rounds 1 and 2 were judged at one moment, iteration 2,000, from the world kept
then, by these criteria (`research/screen/criteria_1.py`):

<!-- table criteria/criteria-1 -->
| criterion | what it asked |
|---|---|
| alive | more than 20 agents at iteration 2,000, and not stopped as too slow |
| decentralised | no agent linked to more than 5% of the others, and 3 to 12 links per agent |
| resilient | one piece, with at least 80% of the agents in the 2-core; no single agent's loss cuts off more than 2% of the world, and no single link's more than 1% |
| almost three-dimensional | a dimension of 2 to 4, and a typical distance at least 1.5 times its random twin's: a true space, not a folded small world |
| diverse | tokens, links, ages and genotypes spread more widely than in B2's median world, and its agents' decisions vary more (effective kinds of staking) |
| not too even | a Gini of the tokens no lower than B2's most even world |
<!-- /table -->

A world was healthy if it met all six, and interesting if it was alive and
missed at most one. Too slow meant it did not reach iteration 2,000 within
four hours of computing. The kinds of staking swing from moment to moment, so
the judge took their mean over iterations 1,500 and 2,000.

#### The lines we compare with

The lines for criteria 5 and 6 came from B2's ten worlds of Experiment 16,
read in the same way and at the same iterations.

<!-- table criteria/references -->
| the line | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score of B2's median world | 0.53 | 0.51 |
| effective kinds of staking of B2's median world | 19.7 | 16.5 |
| Gini of the tokens of B2's most even world | 0.49 | 0.46 |
<!-- /table -->

## The health bar

A world can only evolve for a long time if its network holds together. The
**health bar** says what that means, in seven marks that can be checked on a
world at one moment: one piece, made mostly of [loops](measuring-networks.md#loops), with no
agent or link whose loss cuts off much of it, no hub, a few links per agent,
and a dimension near three. A world that meets all seven has a healthy
network. The bar says nothing about the agents themselves: how much they
differ is the [diversity score](measuring-health.md#the-diversity-score).

### The seven marks

The bar was set in
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers).

| mark | what is measured | the line |
|---|---|---|
| one piece | how many pieces the network falls into | one |
| 2-core | the share of agents in the [2-core](measuring-networks.md#core-trees-and-leaves): on a loop, or between loops | at least 80% |
| agent cut | the largest share of the world that one agent's loss would cut off | at most 2% |
| link cut | the same for one link: the [cut risk](measuring-networks.md#cut-risk) | at most 1% |
| no hub | the share of the other agents that the best-connected agent is linked to | at most 5% |
| links per agent | the mean [degree](measuring-networks.md#an-agents-connections-degree-clustering-assortativity-attachment-and-scaling) | 3 to 12 |
| dimension | how fast balls grow ([Dimension and curvature from ball growth](measuring-space.md#three-ways-to-measure-a-networks-dimension)) | 2.3 to 4 |

### Why each mark is there

- **One piece.** The [cleanup](one-iteration.md#cleanup) after every phase
  keeps only the largest piece of the network; every agent cut off from it
  dies.
- **2-core.** Peel away every agent with fewer than two links, again and
  again; what is left is the 2-core. An agent outside it hangs on a tree, and
  a single cut can lose a tree whole.
- **The two cuts** count single points of failure. A [bridge](measuring-networks.md#bridges-cut-risk-loops-and-the-core-how-a-network-can-be-cut)
  with a tenth of the world behind it is a tenth of the world that dies the
  moment no token crosses it in a game, and a death can split a world in the
  same way.
- **No hub.** A world that hangs on one agent changes all at once when that
  agent dies.
- **Links per agent.** A tree has fewer than two links per agent; a complete
  network, where everyone is linked to everyone, has as many as there are
  other agents. Three to twelve is neither.
- **Dimension.** The notebook hopes for a world that grows into a space like
  ours ([Physical inspiration](physical-inspiration.md)). The author's "almost
  three" is the middle of the range.

### How it is read

`research/atlas/profile.py` reads a world from its saved state at one moment,
without its brains. One depth-first search finds the pieces and the worst cut
by one agent and by one link; peeling finds the 2-core; the dimension is the
lab's own measure: how the number of agents within 1 to 5 steps
grows, around 24 agents spread through the world. The notebook's tables give
the result as "marks (of 7)".
Earlier experiments read the same lines from a run's recorded statistics, as
medians over many iterations.

### Two examples

- **[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") fails it.** A [pilot](runs-and-records.md#pilot-worlds) of B1 at 5,000 tokens (seed 1,
  iterations 200 to 999) had 59% of its agents in the 2-core, a link whose
  loss could cut off 8% of the world, and a best-connected agent linked to 8%
  of the others. Its 3.3 links per agent and its dimension of 2.7 passed.
- **B2 nearly passes it.** In Experiment 14, ten worlds of B2 at 5,000
  tokens, every agent sat in the 2-core; read every tenth iteration, no
  agent's loss could cut off more than 1.2% of a world; the best-connected
  agent was linked to 2.1 to 3.1% of the others; agents had 5.0 to 5.8 links.
  Nine worlds met every line. The tenth missed only the dimension, 4.19
  against the bar's 4.

### Read it with care

- **Two marks follow a world's size.** In networks like B2's, the hub mark
  fails small worlds, where a best-connected agent with 12 to 47 links is
  already more than 5% of the world, and the dimension mark fails large ones.
  In the mixed worlds of Experiment 17, the hub mark failed at 96% of the
  moments below 300 agents, while worlds of 600 to 1,000 agents met every
  mark at 98% of their moments
  ([Mixed at birth: ten long worlds](building-b2-and-the-search-for-a-healthy-rich-world.md#the-marks-a-world-misses)).
- **The dimension mark cannot tell a space from a small world.** A B2 world
  of 600 to 1,500 agents passes it because it is small
  ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)). So the
  [screen of random worlds](a-screen-of-random-worlds.md) asks instead for a
  dimension of 2 to 4 and a typical distance at least 1.5 times its
  [random twin](measuring-space.md#the-random-twin-is-a-world-a-space)'s. Its criteria *decentralised* (no hub,
  links per agent) and *resilient* (one piece, 2-core, both cuts) are the
  other six marks
  ([What makes a world interesting](measuring-health.md#the-six-criteria)).

### Where it is used (the health bar)
- [A world that cannot break](building-b2-and-the-search-for-a-healthy-rich-world.md#building-b2-a-world-that-cannot-break): B2's rules were
  built to meet it.
- [Brains that see, and worlds that break](building-b2-and-the-search-for-a-healthy-rich-world.md#brains-that-see-and-worlds-that-break):
  no world there held.
- [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich): every world read against it; on
  17 October 2026, 99 of 624 met all seven marks.
- [Mixed at birth: ten long worlds](building-b2-and-the-search-for-a-healthy-rich-world.md#brains-mixed-at-birth-in-long-worlds),
  [The candidates, played on](building-b2-and-the-search-for-a-healthy-rich-world.md#the-candidates-played-on-and-played-again) and
  [Room for the steadiest world](building-b2-and-the-search-for-a-healthy-rich-world.md#room-and-riches-for-the-steadiest-world): theses
  that ask for every mark.
- [What makes a world interesting](measuring-health.md#what-makes-a-world-interesting) and
  [A screen of random worlds](a-screen-of-random-worlds.md): criteria built
  on six of the marks.
- [What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution): every mark,
  at every 10,000 iterations.

## The diversity score

Evolution needs differences to choose from. The **diversity score** says, in
one number from 0 to 1, how widely the agents of a world differ in their
tokens, their links, their ages and their [genotypes](measuring-evolution.md#genotypes-families-and-the-common-ancestor). Each of
the four is counted without drawing any line, then ranked among a fixed set
of 653 worlds; the score is the mean of the four ranks. How differently the
agents *decide* is a separate measure: the
[effective kinds of staking](b1-brains-lineages-and-messages.md#kinds-of-behaviour).

### Four spreads

**Tokens, links and ages.** Sort the agents into doubling classes: 1, 2–3,
4–7, 8–15, and so on (for ages, the age plus one, so that a newborn counts).
The **spread** is the [effective number](statistics.md#effective-numbers-how-many-kinds-and-how-many-directions) of classes the
agents fill: 2 to the power of the [entropy](measuring-tokens.md#entropy-and-evenness), in
bits, of the classes' shares. It is 1 when every agent is in one class, and
*k* when the agents are spread evenly over *k* classes. Agents holding one to
three tokens fill at most two classes.

**Genotypes.** Here the count is the effective number of genotypes,
1 / Σ *p*², where *p* is the share of the agents carrying each genotype,
divided by the number of agents. It is 1 when every agent has a genotype of
its own, and near 0 when one genotype is everywhere.

### From spreads to a score

The four spreads are on different scales, so each is turned into a
percentile: the share of the reference worlds whose agents spread less
widely. The score is the mean of the four percentiles, 1 for a world at the
top of all four and 0 for one at the bottom of all four.

The reference is frozen: the 653 worlds the notebook had read on 17 October
2026 (`book/results/diversity_reference.json`, made by
`research/atlas/profile.py --freeze`). A new world gets the percentile its
spreads would have among them, so no score moves as worlds are added.

So the score is relative: it places a world among the worlds this notebook
has made, not on an absolute scale. And it does not say how evenly the
tokens are shared; that is the [Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-entropy-and-mobility).

### Examples

From [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich):

- **B1's worlds** spread widely, their ages over 5.6 to 6.9 classes and their
  tokens over 3.5 to 4.1, and score 0.67 to 0.75.
- **B2's worlds** spread less: three agents in four hold three tokens or
  fewer, and the ages fill about 4 classes, since every agent dies at age 100.
- **B2's worlds without old age** (Experiment 15) score among the lowest of
  all: nearly every agent is of one age, the [founders](rules-in-detail.md#the-starting-ring "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' generation that never
  dies, and the ages fill 1.3 to 2.3 classes.
- **A score can fall.** A world that mixed brains at birth and left the
  [estates](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") of the dead to the poorest scored 0.57 at iteration 1,000 and,
  played on, 0.28 at iteration 10,000.

### The line of B2

[What makes a world interesting](measuring-health.md#the-lines-we-compare-with)
compares a world with the median of B2's ten worlds of Experiment 16, read
the same way at the same iteration. A world is *diverse* when it is above
both lines:

| B2's median world | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score | 0.53 | 0.51 |
| effective kinds of staking | 19.7 | 16.5 |

Other pages draw the line at other moments: 0.54 for B2's ten worlds played
on and read at iteration 3,000
([Room for the steadiest world](building-b2-and-the-search-for-a-healthy-rich-world.md#room-and-riches-for-the-steadiest-world)).

### Decisions: the effective kinds of staking

The score sees what agents hold and what they are, not what they do, and two
genotypes may act exactly alike: a B2 world of 3,400 agents holds about 950
effective genotypes but about 17 kinds of staking.
[How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents) puts the
brains of 400 agents into the same 200 situations, with the same random
inputs. Two brains act differently when their stakes are further apart than
95% of brains are from themselves between two halves of the random draws;
brains that do not are one kind. The kinds are counted by their carriers, as
an effective number: 3.2 in B1's median world at its end, 17.5 in B2's at
iteration 2,000.

For its lines, the screen of random worlds reads the kinds with no game
played, and averages each moment with the one 500 iterations before, since
the kinds swing from moment to moment.

### Where it is used (the diversity score)
- [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#two-halves): where the score was
  made.
- [What makes a world interesting](measuring-health.md#what-makes-a-world-interesting) and
  [A screen of random worlds](a-screen-of-random-worlds.md): the criterion
  *diverse*.
- [The candidates, played on](building-b2-and-the-search-for-a-healthy-rich-world.md#the-candidates-played-on-and-played-again) and
  [Five more seeds for the rules that held](building-b2-and-the-search-for-a-healthy-rich-world.md#five-more-seeds-for-the-rules-that-held):
  a world must score above B2's median world to be played on, and again to
  hold.
- [Room for the steadiest world](building-b2-and-the-search-for-a-healthy-rich-world.md#room-and-riches-for-the-steadiest-world): what each
  rule cost of the four parts.
- [Births as rich as the neighbours](building-b2-and-the-search-for-a-healthy-rich-world.md#births-as-rich-as-the-neighbours),
  [The search's best world, in new seeds](building-b2-and-the-search-for-a-healthy-rich-world.md#the-searchs-best-world-in-new-seeds),
  [Who inherits?](what-the-switches-did-in-pilots.md#who-inherits) and [Rules in company](what-the-switches-did-in-pilots.md#rules-in-company):
  worlds scored by it.

## Where to go next

- [A screen of random worlds](a-screen-of-random-worlds.md):
  the rounds, and what each found.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  where the health bar comes from.
- [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich): how the
  diversity score was made, and every world scored by it.
- [How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents): the
  measure behind the same-position share.
- [What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution): the test a
  healthy world's long run will face.
- [The random twin: is a world a space?](measuring-space.md#the-random-twin-is-a-world-a-space): the test the
  dimension mark needs beside it.
- [Effective numbers: how many kinds there really are](statistics.md#effective-numbers-how-many-kinds-and-how-many-directions):
  the count inside both measures.

## How our view changed

- *What makes a world interesting*:
  - **24 October 2026.** Since 19 October 2026, the notebook called the line of
    the sixth criterion "B2's least even world". The line the judge compares
    with is the lowest Gini among B2's worlds: 0.49 at iteration 1,000 and 0.46
    at 2,000 ([The lines we compare with](#the-lines-we-compare-with)). That is
    B2's **most** even world. The words are set right on this page and on every
    page that draws the line. The line itself, and every verdict judged by it,
    are unchanged. The theses of Experiments 31, 32, 38 and 39 keep the old
    words, as a thesis does, and mean this line.
  - **24 October 2026.** The author found the criteria too harsh, and revised
    them for round 3 (`research/screen/walkthrough.md`): averages over
    iterations 100 to 3,000, cuts up to 15%, a width of 1.3 and a dimension of 2
    to 5, and diversity statistic by statistic instead of one score against
    B2's median world. Even B2's own ten worlds met the old criteria only twice
    in ten as interesting, and never as healthy. Round 1 keeps its verdict by
    the criteria it was judged by.
  - **25 October 2026.** The judge for the revised criteria was built. This page
    now states them, with every line and its sensitivity table, and keeps the
    first criteria at its end. Two choices were made in building it. The choice
    between spreading a stake and staking it all on one place is reported, not
    judged: in the first world of round 3, 98.5% of the agents spread, which says
    nothing about how alike their stakes are. And the half cut is the cheapest of
    ten cuts, since the slowest way to vibrate alone varied by a factor of two
    between ways that vibrate at nearly the same rate.
  - **7 November 2026.** Round 3 has run, and the author asked for an order of
    the worlds rather than a verdict. The proposal
    ([Which worlds are healthy?](which-worlds-are-healthy.md)) keeps these lines
    but reads them as shares, draws lines for the half-cut exponent and the
    same-position share, and would replace B2's Gini line by one drawn by chance.
    This page's lines stand until the author decides.
- **11 November 2026.** This page now opens with the definition in force, of 9
  November 2026, until then only on Which worlds are healthy?, and a table of
  the earlier versions. From this day health is frozen at that version and used
  only as a filter: worlds that are dead, a star, uniform or frozen are left
  out, and worlds are ranked by evolution. The earlier versions are kept as
  they were.

<!-- turns -->
---

← [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) · [Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [How runs are made, recorded and costed](runs-and-records.md) →

**Also about:** [Brains: what they are, and whether they matter](../README.md#brains-what-they-are-and-whether-they-matter).

> [!info]- Pages that link here (16)
> [What this notebook is about](what-this-notebook-is-about.md) · [How this notebook is written](how-this-notebook-is-written.md) · [A screen of random worlds](a-screen-of-random-worlds.md) · [Which worlds are healthy? Three orders](which-worlds-are-healthy.md) · [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md) · [Glossary](glossary.md) · [Measuring lives: births, deaths, ages and estates](measuring-lives.md) · [Measuring tokens: inequality and flow](measuring-tokens.md) · [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Measuring space: dimension and the random twin](measuring-space.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Statistics: the tools of the notebook](statistics.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md)
<!-- /turns -->
