# The settled life of a world

To compare worlds — thirty seeds, or two conditions — each world has to be
summed up in one number per statistic: *where does it settle?* This note
gives the rule the book uses and why.

## The rule

> A world's level for a statistic is its **mean over iterations 500 to the
> end of the run** (2,999 for the baseline runs), over the rows of the phase
> the statistic belongs to. Only worlds that lived to the end are measured
> this way; worlds that died are counted apart.

For the baseline that is the mean over 2,500 game rows — or, for a graph
statistic measured every 25 iterations, over 100.

## Why from iteration 500?

Because a world's first few hundred iterations are unlike the rest of its
life. It starts with a boom and a crash
([Chapter 11](../chapters/11-the-first-hundred-iterations.md)), and its
births, deaths and structure take hundreds of iterations to settle into the
ranges they keep ([Chapter 12](../chapters/12-births-deaths-and-ages.md)).
Averaging the youth in would mix two different things.

## Why a long mean, and not the end?

Because a single world does not settle at one level: it **wanders**, up and
down over hundreds of iterations ([Chapter 10](../chapters/10-a-worlds-life.md)).
Where it stands at iteration 2,999, or over its last few hundred iterations,
is wherever the wander happens to have taken it. A mean over 2,500
iterations averages much of the wander away, so it says more about the world
and less about the moment. [Chapter 13](../chapters/13-how-much-does-the-seed-decide.md)
measures how much: the worlds' sizes differ by a third of the mean when each
is measured over its last 100 iterations, and by a sixth when measured over
its settled life.

Chapters written before this rule (the first analyses of Experiments 2 to 6)
used the last fifth of a run, iterations 2,400 to 2,999. Where this book
checks a thesis that was written with the last fifth in mind, it says so.

## Why not the worlds that died?

A world that dies did not end up anywhere: the last part of its life is its
dying. Its mean would mix a settled world with a collapse. So the dead are
reported beside the living, as an outcome of their own — how many died, and
when ([When a world ends](extinction.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
