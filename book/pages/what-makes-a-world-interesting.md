# What makes a world interesting

> [!question] Questions of this page
> - What do we look for in a world before we ask how it evolves?
> - When is a world healthy, and when is it still interesting?
> - When do we give up on a [setting](settings.md "One number or switch of a run's rules, fixed before it starts.")?

## In short

We are looking for worlds in which evolution could go on for a very long
time. Such a world has to stay alive. It must not hang on a few [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), and it
must not fall apart when one agent dies. It should look like a space of about
three [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), as our own world does. And its agents should differ from
one another, in [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), links, age, genes and decisions, without the tokens
being spread too evenly.

We turn this picture into six criteria. From round 3 of the screen on, every
number behind them is an **average over a world's whole run**, from [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")
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

## The six criteria

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
   depends on it: when that [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") dies, everything changes at once. We ask that
   no agent links to more than 5% of the world, and that agents have 3 to 12
   links each, as a network that is neither a tree nor a tangle.
3. **Resilient.** A healthy world does not break apart when one agent or one
   link disappears. We ask that most agents (80%) sit on [loops](bridges.md#loops "A path through the network that comes back to where it started without using a connection twice.") (the
   *[2-core](bridges.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.")*), and that no single loss cuts off more than 15% of the agents.
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
   times that in a *[random twin](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.")*, a network with the same number of links per
   agent wired at random. A folded world fails this; a true space passes
   ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)).
5. **Diverse.** Evolution needs differences to choose from. Each statistic is
   judged on its own, not mixed into one score:
   - **tokens:** not diverse if 80% or more of the agents hold the same number
     of tokens, or if no agent holds more than 3;
   - **links:** counted in the 2-core only, since leaves do not make a network
     bad: not diverse if 80% or more of the core's agents have the same number
     of links;
   - **ages** and **[genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."):** not diverse if 80% or more of the agents share
     one age, or carry one genotype;
   - **decisions:** not diverse if 95% or more of the agents make the same
     choice, to have a child or not. A world where every agent has a child in
     every iteration, or none does, is sorted out.
6. **Not too even.** If almost every agent holds the same few tokens, there
   is little to win and little to select. We ask for a
   [Gini coefficient](gini-coefficient.md) of at least 0.46, the Gini of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s
   most even world in Experiment 16. Round 3 has no B2 worlds of its own, so
   this line is a fixed number.

## The lines

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
iteration, a [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") of every phase, and the whole world every 250 iterations.
The judge reads each number where it is kept, as often as it is kept, and
averages it from iteration 100 to the end:
- every iteration: the number of agents, their links, the hub, the largest
  holding, the Gini, and the share of agents that had a child;
- every 25th iteration: the dimension, and from the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")'s frame how alike
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

## Two numbers reported, not judged

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

### The half-cut exponent

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
eigenvalues) splits the agents at its [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above."), and the cheapest of the ten
splits counts. One alone is not enough: the slowest ways vibrate at nearly the
same rate, and in one world of round 3 the slowest cut 121 links where the
third cut 62. On a box of 16 by 8 by 8 agents the judge finds the slice of 64
links across its long side, and on real worlds the same ten cuts as an exact
calculation. The world's random twin is cut the same way, for comparison: a
twin is no space at all.

### The same-position share

Put two agents in exactly the same situation: the same [neighbours](the-world.md#the-network "An agent joined to another by a connection."), the same
tokens, the same [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read."). Do they stake their tokens differently? The judge
takes 400 of a world's agents and 200 situations from the world, plays every
one of those agents' [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") in each, and counts the pairs of agents whose
stakes differ by more than one brain differs from itself between two tries.
In a world where every agent would act alike in the same place the share is
near 0; in a world of many kinds of behaviour it is near 1. It is the measure
of [How different are the agents?](how-different-are-the-agents.md), read at
every world kept.

## Healthy, and interesting

A world that meets all six criteria is healthy.

Worlds made with the same settings but different [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") differ. A setting
that gives a healthy world in one seed may miss a line in another, and we do
not want to throw it away for that. So a world may miss some criteria and
still be **interesting**. How many, and which, will be decided with the
author once round 3 has run and we have looked at its worlds. Until then the
judge counts a world as interesting when it is alive and misses at most two of
criteria 2 to 6, the author's first answer.

## When a setting is marked bad

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

## The criteria of rounds 1 and 2

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

### The lines we compare with

The lines for criteria 5 and 6 came from B2's ten worlds of Experiment 16,
read in the same way and at the same iterations.

<!-- table criteria/references -->
| the line | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score of B2's median world | 0.53 | 0.51 |
| effective kinds of staking of B2's median world | 19.7 | 16.5 |
| Gini of the tokens of B2's most even world | 0.49 | 0.46 |
<!-- /table -->

## Where to go next

- [A screen of random worlds](a-screen-of-random-worlds.md):
  the rounds, and what each found.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  where the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") comes from.
- [Healthy and rich](healthy-and-rich.md): how the
  [diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.") was made, and every world scored by it.
- [How different are the agents?](how-different-are-the-agents.md): the
  measure behind the same-position share.
- [What would count as open-ended evolution](open-ended-test.md): the test a
  healthy world's long run will face.

## How our view changed

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

<!-- turns -->
---

← [Healthy and rich](healthy-and-rich.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [The health bar](health-bar.md) →

> [!info]- Pages that link here (14)
> [Glossary](glossary.md) · [Measuring inequality: the Gini coefficient, the richest tenth, entropy and mobility](gini-coefficient.md) · [Stars and hubs](stars-and-hubs.md) · [The random twin: is a world a space?](random-twin.md) · [How alike the agents are](how-alike-the-agents-are.md) · [How different are the agents?](how-different-are-the-agents.md) · [Rules in company](rules-in-company.md) · [Healthy and rich](healthy-and-rich.md) · [The health bar](health-bar.md) · [The diversity score](diversity-score.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [The candidates, played on and played again](the-candidates-played-on.md) · [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [A screen of random worlds](a-screen-of-random-worlds.md)
<!-- /turns -->
