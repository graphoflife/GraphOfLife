# The settled life of a world

To compare worlds — thirty [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world."), or two conditions — each world has to be
summed up in one number per statistic: *where does it settle?* This note
gives the rule the notebook uses and why.

## The rule

> A world's level for a statistic is its **mean over [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 500 to the
> end of the run** (2,999 for the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") runs), over the rows of the phase
> the statistic belongs to. Only worlds that lived to the end are measured
> this way; worlds that died are counted apart.

For the baseline that is the mean over 2,500 [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") rows — or, for a graph
statistic measured every 25 iterations, over 100.

## Why from iteration 500?

Because a world's first few hundred iterations are unlike the rest of its
life. It starts with a boom and a crash
([The first hundred iterations](the-first-hundred-iterations.md)), and its
births, deaths and structure take hundreds of iterations to settle into the
ranges they keep ([Births, deaths and ages](births-deaths-and-ages.md)).
Averaging the youth in would mix two different things.

## Why a long mean, and not the end?

Because a single world does not settle at one level: it **wanders**, up and
down over hundreds of iterations ([A world's life](a-worlds-life.md)).
Where it stands at iteration 2,999, or over its last few hundred iterations,
is wherever the wander happens to have taken it. A mean over 2,500
iterations averages much of the wander away, so it says more about the world
and less about the moment. [How much does the seed decide?](how-much-does-the-seed-decide.md)
measures how much: the worlds' sizes differ by a third of the mean when each
is measured over its last 100 iterations, and by a sixth when measured over
its settled life.

Pages written before this rule (the first analyses of Experiments 2 to 6)
used the last fifth of a run, iterations 2,400 to 2,999. Where this notebook
checks a thesis that was written with the last fifth in mind, it says so.

## Why not the worlds that died?

A world that dies did not end up anywhere: the last part of its life is its
dying. Its mean would mix a settled world with a collapse. So the dead are
reported beside the living, as an outcome of their own — how many died, and
when ([When a world ends](extinction.md)).

<!-- turns -->
---

← [Bands: many worlds in one figure](bands.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Spread between worlds](spread-between-worlds.md) →

> [!info]- Pages that link here (10)
> [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [When a world ends](extinction.md) · [What the baseline world is](what-the-baseline-world-is.md) · [A world's life](a-worlds-life.md) · [Stars and hubs](stars-and-hubs.md) · [How worlds are measured](how-worlds-are-measured.md) · [Spread between worlds](spread-between-worlds.md) · [Autocorrelation: how long a world remembers](autocorrelation.md) · [How many seeds an experiment needs](seeds-needed.md)
<!-- /turns -->
