# When a world ends

A run goes for a fixed number of iterations — 3,000 for the baseline runs —
unless its world **dies out** first.

## The rule

After every iteration — that is, after its game and the cleanup that follows
— the number of agents is counted. If it is at most
[`extinction_threshold`](settings.md#extinction_threshold), 20 in the
baseline, the world counts as **extinct** and the run stops.

The check is made once per iteration, after the game; not after the
reproduction phase. A world that falls to 20 or fewer during reproduction and
recovers in the game is not extinct.

## Why 20 and not 0?

A world of a handful of agents is not a population any more, and its
statistics — inequality, clustering, families — mean little. Stopping at 20
also saves running a dying world to its end.

But a world at 20 agents is not always dying.
[Chapter 31](../chapters/31-how-does-a-worlds-size-follow-its-tokens.md) ran
small worlds with the threshold at 0: all six worlds of 800 and 1,600 tokens
lived to iteration 600, and five of them were worlds the threshold had stopped
after their first game; one fell to 4 agents and came back. So "extinct" in
this book means exactly what the rule says — **fell to 20 agents or fewer** —
which is not always the same as dying. An experiment that cares about
extinction should run a condition with the threshold at 0 beside it.

## An empty world

If a cleanup ever removes every agent, the program puts one new agent with a
fresh random brain and all *T* tokens on a node of its own. With the
threshold at 20, that world is extinct at once.

## How the book counts the dead

A world that dies out did not end up anywhere: the last stretch of its life
is its dying. So where the worlds of a setting end up is measured over the
worlds that lived to the end, and the dead are reported beside them, as an
outcome of their own ([The settled life of a world](settled-life.md)). Of the
30 baseline worlds, four died: after 4, 313, 1,547 and 2,184 iterations
([Chapter 9](../chapters/09-a-worlds-life.md)).

"Died after *N* iterations" means: the iteration numbered *N* − 1 was the
last to run, and it ended with 20 agents or fewer.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
