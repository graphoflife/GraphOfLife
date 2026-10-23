# Eras and contests: does a world keep getting better?

## In short

A [world](the-world.md) whose numbers have settled may have stopped
evolving, or may still be improving in ways no single moment shows. To
tell, the notebook plays a world's [brains](the-brain.md) of two moments
against each other. The brains of one [iteration](one-iteration.md), kept
in the world's archive, are an **era**; a match between two eras is a
**contest**. Its score is the share of the world the later era's
brains hold at the end: 0.5 means no better.

## Why not watch a number rise?

Many numbers rise where nothing is selected: a record grows even in a world
that stands still, a count only adds, and everything climbs while a young
world settles
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#numbers-that-rise-and-mean-nothing)).
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
4. B's **score** is the share of agents carrying a B brain at the end. 0.5
   is a tie; above it, B's brains spread at A's expense.

Each pair is played twice, once in each era's world, so neither plays only
at home.

## Scores and ratings

Playing every pair of a set of eras is a **cross-time tournament**. A score
*p* can be read as odds, and one **rating** *r* per era is fitted to every
contest at once by [least squares](least-squares.md), with one era, usually
the founders, at 0:

$$
\log \frac{p}{1-p} \approx r_B - r_A .
$$

This is the model of Bradley and Terry, which chess calls Elo. Multiplied by
400 / ln 10, the ratings are Elo points: an era rated +200 beats the
founders about three times in four, +400 ten times in eleven.

## A contest a lifetime long

Until 15 October 2026 a contest lasted *T* = 20 iterations: long enough for
most of a [B1](b1-and-b2.md) world to change hands, but in B2 every agent
lives 100. A **lifetime contest** lasts *T* = 100. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of Experiment 16's ten B2 worlds, the brains of
iteration 2,000 held 0.482 of a world against those of 500 in contests of
20 iterations, a tie, but 0.73 in contests of 100, and more than half in all
ten worlds.

## Plateau, climb or circle

Contests are noisy, so an era is also played against itself: a score within
0.5 plus or minus twice the standard deviation of those self-contests could
be noise.

Contests between neighbouring eras cannot tell three stories apart: a
**plateau**, where evolution has stopped; a slow climb; and a circle, where
each era beats the one before and loses to one long gone, as rock, paper and
scissors do. Playing every era
against a few fixed eras can: on a plateau the scores against a fixed era
stop rising, on a climb they keep rising, in a circle they rise and fall.

In lifetime contests in Experiment 18's two worlds of 50,000 iterations,
the eras after 25,000 scored medians of 0.55 and 0.46 against the brains of
25,000, inside their
noise bands of 0.40 to 0.60 and 0.36 to 0.64: evolution had stopped
climbing ([Every era against three](every-era-against-three.md)). A plateau
is not a top: a best reply would still have gained about a node a game
([Exploitability](exploitability.md)).

## The yardstick

In this notebook, "evolution keeps going" means that later eras keep
beating earlier ones, beyond the noise. B1 fails it. In the median of
Experiment 12's ten worlds, the brains of iteration 50 held 0.90 against the
founders, and every later era tied with the one before (0.47 to 0.52);
contests a [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.") long agree.
[What would count as open-ended evolution](open-ended-test.md) asks more: to
beat the best mixture of all earlier eras, since wins over the last era
alone can go round in a circle.

## Where it is used

[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
builds the yardstick on B1; [Why evolution stops](how-to-tell-a-world-that-keeps-evolving.md#why-evolution-stops)
applies it to B2; [Every era against three](every-era-against-three.md)
plays fixed eras; [Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why) compares rules by
their contests; [Pilot worlds](pilot-worlds.md) plays the eras of pilots.

## Where to go next

- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
- [The invasion test](invasion-test.md), which asks whether better play
  exists at all
- [Exploitability](exploitability.md)

<!-- turns -->
---

[Does evolution keep going?](../README.md#does-evolution-keep-going) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (6)
> [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
