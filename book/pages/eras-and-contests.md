# Eras, contests and invasions: does a world keep getting better?

## In short

A [world](the-world.md) whose numbers have settled may have stopped
evolving, or may still be improving in ways no single moment shows. To
tell, the notebook plays a world's [brains](the-brain.md) of two moments
against each other. The brains of one [iteration](one-iteration.md), kept
in the world's archive, are an **era**; a match between two eras is a
**contest**. Its score is the share of the world the later era's
brains hold at the end: 0.5 means no better. Contests compare brains only
with their past; **exploitability** measures what one agent would gain by
its best reply to all the others. The **invasion test** plants a strategy
written by hand in an evolved world, and asks whether it spreads further
than the world's own brains would.

## Why not watch a number rise?

Many numbers rise where nothing is selected. A record grows even in a world
that stands still, and a count only adds. Everything climbs while a young
world settles
([Numbers that rise and mean nothing](how-to-tell-a-world-that-keeps-evolving.md#numbers-that-rise-and-mean-nothing)).
A contest counts nothing, and plays the settled world against its own past.

## Eras

If its plan asks ([`archive_every`](settings.md#archive_every)), a run keeps
a whole world every so many iterations: its network, tokens, ages and every
brain. The brains of one such moment are an era, named by its iteration;
era 0 is the [founders](starting-ring.md).

## A contest, step by step

1. Take the world of one archived moment: its network, tokens and ages.
2. Give every agent a brain drawn from era A or era B, a coin each, and
   clear the [messages](messages.md) in flight.
3. Switch [mutation](mutation.md) off, so every copy of a brain keeps its
   era, and let the world play *T* iterations under its own rules.
4. B's **score** is the share of agents carrying a B brain at the end. A
   score of 0.5 is a tie; above it, B's brains spread at A's expense.

Each pair is played twice, once in each era's world, so neither plays only
at home.

## Scores and ratings

Playing every pair of a set of eras is a **cross-time tournament**. A score
*p* can be read as odds. One **rating** *r* per era is fitted to every
contest at once by [least squares](correlation.md#fitting-a-straight-line), with one era, usually
the founders, at 0:

$$
\log \frac{p}{1-p} \approx r_B - r_A .
$$

This is the model of Bradley and Terry, which chess calls Elo. Multiplied by
400 / ln 10, the ratings are Elo points. An era rated +200 beats the
founders about three times in four, and one rated +400 ten times in eleven.

## A contest a lifetime long

Until 15 October 2026 a contest lasted *T* = 20 iterations. That is long
enough for most of a [B1](b1-and-b2.md) world to change hands, but in B2
every agent lives 100. A **lifetime contest** lasts *T* = 100. In the
[median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of Experiment 16's ten B2 worlds, the brains of iteration 2,000
played those of 500. In contests of 20 iterations they held 0.482 of a
world, a tie. In contests of 100 they held 0.73, and more than half in all
ten worlds.

## Plateau, climb or circle

Contests are noisy, so an era is also played against itself. A score within
0.5 plus or minus twice the standard deviation of those self-contests could
be noise.

Contests between neighbouring eras cannot tell three stories apart:
- a **plateau**, where evolution has stopped;
- a slow climb;
- a circle, where each era beats the one before and loses to one long gone,
  as rock, paper and scissors do.

Playing every era against a few fixed eras can tell them apart. On a
plateau the scores against a fixed era stop rising, on a climb they keep
rising, and in a circle they rise and fall.

Experiment 18's two worlds were played to 50,000 iterations. In [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.")
contests, their eras after 25,000 scored medians of 0.55 and 0.46 against
the brains of 25,000. That is inside their noise bands of 0.40 to 0.60 and
0.36 to 0.64: evolution had stopped climbing
([Every era against three](every-era-against-three.md)). A plateau is not a
top: a best reply would still have gained about a node a game (see
[Exploitability](#exploitability), below).

## The yardstick

In this notebook, "evolution keeps going" means that later eras keep
beating earlier ones, beyond the noise. B1 fails it. In the median of
Experiment 12's ten worlds, the brains of iteration 50 held 0.90 against the
founders. Every later era tied with the one before (0.47 to 0.52), and
contests a lifetime long agree.
[What would count as open-ended evolution](open-ended-test.md) asks more: to
beat the best mixture of all earlier eras. Wins over the last era alone can
go round in a circle.

## Exploitability

Contests compare a world's brains with their own past. They cannot say how
good the brains are in any absolute sense. Brains that tie with their
ancestors may be at the top, or stuck far below it: a plateau looks the
same either way. **Exploitability** is a measure from game theory that can
tell the two apart.

### The idea of a best reply

Picture one agent while every other agent keeps its play. Whatever the
others do, some way of staking does best against them: the agent's **best
reply**. How much better the best reply does than the agent's own play is
the agent's exploitability.

In a **Nash equilibrium**, named after the mathematician John Nash, no
player gains by changing its own play while the others keep theirs. So at
an equilibrium, exploitability is zero for everyone. A population that
evolution drives towards an equilibrium becomes less exploitable as it
goes; one that is stuck short of it stays exploitable.

### How exploitability is measured

The agents' game is too large to solve, but a single agent's best reply to
one game is a small problem (`research/oee/checks/exploitability.py`):

1. Take a world as its archive keeps it. Play it on for one iteration, a
   phase of reproduction and a game, so that its messages are fresh. Then
   draw, eight times, the random numbers the brains read. For each draw,
   run every agent's brain and turn its outputs into stakes exactly as the
   engine does. This is the others' play, as a distribution over eight
   draws.
2. Score a way of staking for one agent by what it **holds** after the
   game. It scores one for keeping its own node, and one for every
   neighbour's node it takes. Everyone else stakes as drawn, and every node
   is decided by the engine's rule. Under B2 the owner's stake counts one
   and a half times, and a [revolution](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") can take a node. Ties are drawn with
   a fixed [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") for each node and draw. The score is averaged over the
   eight draws.
3. The agent's own play scores what its drawn stakes hold, averaged over
   the eight draws. The best reply is searched among a few natural plays,
   none of which marks a stake revolutionary:
   - everything at home;
   - for each draw, the cheapest way to keep its node and then take the
     cheapest neighbours. Taking a neighbour costs one token more than the
     largest rival offer on it;
   - the same without defending beyond what B2 keeps at home anyway;
   - the agent's own stakes of each draw.

   Each is scored over all eight draws, so a play that only fits one draw
   is not rewarded for it. If none does better than the agent's own play,
   the agent's exploitability is zero.
4. Exploitability is the best reply's score minus the agent's own, averaged
   over up to 300 agents with at least three tokens.

**An example.** In the median of Experiment 16's ten B2 worlds, the
founders could have gained 0.87 nodes a game by their best replies. The
brains of iterations 1,000 to 2,000 could have gained about 1.2. Evolution
moved them away from an equilibrium, not towards it
([Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on)).

### What exploitability can and cannot say

The best reply is only searched among some plays, so the exploitability
found is a lower bound. The true distance from an equilibrium is at least
this large.

It is a distance in **one game**. Holding a node for one game is not the
same as spreading over a lifetime. A play that takes many cheap nodes now
may cost an agent later. So a population can be exploitable in one game and
still hard to beat in the long run.

The measure is a probe, not a verdict. What it shows most clearly is a
*trend*: whether evolution moves a population towards an equilibrium, away
from it, or nowhere.

## The invasion test

A contest asks whether a world's later brains beat its earlier ones. The
**invasion test** asks something else: is there play *better* than what a
world has evolved? It plants a strategy in the evolved world and watches
whether it spreads. A **strategy** takes the place of an agent's brain: it
is a rule written by hand, or a brain whose weights were set by hand.

### How an invasion is played

1. Take an evolved world as its archive keeps it: its graph, its tokens
   and its brains (`GraphOfLife.from_checkpoint`).
2. Choose a **patch**: an agent drawn at random, then the agents nearest to
   it, step by step through the network. The patch grows until it holds the
   share of the world asked for (5% in
   [Strategies written by hand](strategies-written-by-hand.md#strategies-written-by-hand)).
   A patch can also be a given number of agents, down to a single one.
3. Give every agent of the patch the strategy in place of its brain
   (`research/oee/pilots/manual.py`; the rules now live in the engine's
   `gol_strategies.py`). It spreads as a brain does, into children and into
   nodes its agents take, and it never mutates.
4. Play on under the world's own rules and count, after every iteration, the
   agents that carry the strategy, as a share of all agents.

### What a strategy can and cannot compute

A strategy reads exactly the inputs a brain reads, and writes exactly the
outputs a brain writes. So it sees nothing a brain could not see. It acts
only through the outputs a brain has, which the engine turns into a child,
its links, the stakes and the messages.

What it makes of what it reads is another matter. A rule written in Python
can combine its inputs in ways a brain cannot. A B2 brain reads one column
of inputs for each candidate, the agent itself and each neighbour. It
scores each column on its own, with the same weights, and no unit sees two
columns. Two of the
[three strategies](strategies-written-by-hand.md#three-strategies) first
planted show the difference:
- **The raider** stakes all it can on the poorest neighbour outside its
  team, if it is sure to take it. Twelve of a B2 brain's 16 units, with
  weights set by hand, play it
  ([A raider in a brain's weights](what-a-brain-can-compute.md#a-raider-in-a-brains-weights)).
- **The factory** is a team whose members write into their messages how far
  they are from the team's frontier, where strangers begin. A member with a
  stranger next door writes 0 to its teammates: news of a third agent, the
  stranger. No B2 brain of any size can put such news into a message
  ([What no brain of B2 can compute](what-a-brain-can-compute.md#what-no-brain-of-b2-can-compute)).

So a strategy that wins shows that the game holds better play than
evolution found. Whether a brain of that world could play it is a separate
question, answered for the raider by building the brain.

### The control

A patch can grow or shrink because of where it sits. It may lie next to
rich agents or poor ones, in the middle of the world or at an edge. So the
share a strategy reaches means little on its own. Every patch is also
played as a **control**. The same agents each get a frozen copy of one
resident brain: the brain of an agent of the patch, drawn at random. The
copies are marked, so they can be counted. A strategy beats the residents
if it ends with more of the world than this control, planted in the same
place on the same seed.

The lab's tests, from Experiment 29 on
([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)),
ask a little more. A planting **takes hold** when, after 300 iterations,
its agents are alive, more than the patch began with, and more than the
control holds there.

### Reading an invasion

- **Growth from a patch** says that the strategy beats the residents *when
  it has a crowd*: better play exists than the play evolved.
- **Growth from a single agent** says more: the strategy needs no crowd. A
  mutation changes one brain at a time, so if a mutation ever made it, it
  could spread from there. A strategy that grows from a patch but not from
  one agent has an **Allee effect**. Ecologists use the name for a species
  that does well only in numbers. Evolution, which tries one agent at a
  time, can miss such a strategy although it is better.
- **A strategy's parts** can be tested one at a time by dropping them. The
  raider without its password
  ([The password](strategies-written-by-hand.md#the-password)) shows which
  part wins.

**An example.** In world 2 of Experiment 16 at iteration 2,000, a patch of
5% of raiders was planted on two seeds. In 300 iterations it grew to 40%
and 50% of the world. The control, the residents' own brains planted in the
same places, ended with 7% and 1%
([What the strategies did](strategies-written-by-hand.md#what-the-strategies-did)).

**One planting says little.** The raider's rule and the brain that plays it
choose differently in only one or two situations in a hundred. Yet planted
on the same seed in world 3, the rule died out while the brain took 59% of
the world. Their worlds part ways within a few iterations, and from there
chance decides whether a patch takes hold
([A raider in a brain's weights](what-a-brain-can-compute.md#a-raider-in-a-brains-weights)).
So the lab's tests plant a strategy in several places in each world.

A strategy that dies out says at most that *this* strategy is worse, never
that nothing better exists. An invasion test can show that a game is not
solved, but not that it is.

## Where it is used

- **Contests.**
  [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#how-to-tell-a-world-that-keeps-evolving)
  builds the yardstick on B1, and
  [Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)
  applies it to B2. [Every era against three](every-era-against-three.md)
  plays fixed eras. [Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)
  compares rules by their contests. [Pilot worlds](the-lab.md#pilot-worlds) plays
  the eras of pilots.
- **Exploitability.**
  [Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on)
  measures it era by era in Experiment 16's B2 worlds.
  [How much room is left](every-era-against-three.md#how-much-room-is-left)
  measures it in every long world.
  [Why the raiders stopped winning](does-evolution-close-the-door.md#what-we-expected-of-older-worlds)
  builds Experiment 30's expectation on it.
- **The invasion test.**
  [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md)
  plants the first strategies, and measures how large a crowd the raider
  needs (Experiment 29). [What a brain can compute](what-a-brain-can-compute.md)
  plants a brain that plays the raider.
  [Why the raiders stopped winning](does-evolution-close-the-door.md) plants
  raiders in older worlds, crews of raiders and raiders that breed
  (Experiments 30, 33 and 34). It also plants them in B1 (Experiment 35).
  [The lab, the engine and pilots](the-lab.md) explains how the lab plays
  invasions, like contests, as the trials of an experiment's plan.

## Where to go next

- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick, held to both baselines.
- [Every era against three](every-era-against-three.md): every era of the
  longest worlds against fixed eras, and the room a best reply finds there.
- [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md):
  the first invasion tests, and what a raider needs to win.
- [What would count as open-ended evolution](open-ended-test.md): what later
  eras would have to beat.

## How our view changed

- **24 October 2026.** From 15 October, the note on the invasion test said
  that a strategy written by hand "can do nothing an evolved brain could
  not". The reason it gave was that a strategy reads and writes exactly what
  a brain does. On 19 October,
  [What no brain of B2 can compute](what-a-brain-can-compute.md#what-no-brain-of-b2-can-compute)
  showed that this holds for the raider but not for the factory. Twelve of a
  B2 brain's 16 units play the raider. But a brain scores each candidate's
  column on its own. So no B2 brain of any size can write news of a third
  agent into a message. The factory's distance to its frontier is such news.
  A strategy sees only what a brain sees and acts only through what a brain
  decides, but it may compute what no brain can.
- **24 October 2026.** From 15 October, the note on exploitability described
  three details of the measure differently from the program that makes it
  (`research/oee/checks/exploitability.py`). We read the program when the
  notes became sections of this page. Before the measure, the world plays
  one whole iteration, a phase of reproduction and then a game; the note
  said one game. The gain is averaged over up to 300 agents with three
  tokens or more, where the note said 300. In 7 of the 145 measurements
  behind [How much room is left](every-era-against-three.md#how-much-room-is-left),
  only 154 to 274 agents held that many. And the plays searched for the best
  reply include the agent's own stakes of each draw, which the note did not
  list.

<!-- turns -->
---

[Does evolution keep going?](../README.md#does-evolution-keep-going) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics), [Strategies written by hand](../README.md#strategies-written-by-hand).

> [!info]- Pages that link here (18)
> [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [Every era against three](every-era-against-three.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [What would count as open-ended evolution](open-ended-test.md) · [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) · [What a brain can compute](what-a-brain-can-compute.md) · [Why the raiders stopped winning](does-evolution-close-the-door.md) · [Gliders, and who helps whom](gliders-and-who-helps-whom.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) · [Links to strangers](links-to-strangers.md) · [Eight switches at once, and how sixteen worlds can test them](eight-switches-at-once.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
