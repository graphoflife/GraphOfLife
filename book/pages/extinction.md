# When a world ends

## In short

A run goes for a fixed number of iterations, unless its world dies out first.
In this notebook that has an exact meaning: if an iteration's game leaves 20
[agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") or fewer, the world is **extinct** and its run stops. A world that
small is no longer a population worth measuring. But it is not always dying:
small worlds run without the rule fell below 20 agents and grew back. So the
notebook reports the worlds that die out apart from those that live to the
end.

## The rule

A run goes for a fixed number of iterations — 3,000 for the thirty [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")
worlds of 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), 1,500 for the thirty of 100,000 — unless its world
**dies out** first.

After every iteration — that is, after its game and the cleanup that follows
— the agents are counted. If there are at most
[`extinction_threshold`](settings.md#extinction_threshold) of them, 20 in both
baselines, the world counts as **extinct** and the run stops.

The count is made once per iteration, after the game, and not after the
reproduction phase. That misses no world. Only the reproduction phase adds
agents; a game and its cleanup can only take them away. So a world left with
20 agents or fewer by its reproduction phase has 20 or fewer after the game
too.

What the timing does decide is when a small world is first judged. A world of
800 tokens starts with 8 [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts."), one per hundred tokens, already below the
line. It still plays one whole iteration before its agents are counted.

## Why 20 and not 0?

A world of a handful of agents is not a population any more, and its
statistics — inequality, [clustering](degree.md#clustering "How often an agent's neighbours are also each other's neighbours."), families — mean little. Stopping at 20
also saves running a dying world to its end.

But a world at 20 agents is not always dying.
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
ran small worlds with the threshold at 0. All six worlds of 800 and 1,600
tokens lived to iteration 600, and five of them were worlds the threshold had
stopped after their first game. One of them fell to 4 agents and came back.

So "extinct" in this notebook means exactly what the rule says — **fell to 20
agents or fewer** — which is not always the same as dying. An experiment that
cares about extinction should run a condition with the threshold at 0 beside
it.

## An empty world

If a cleanup ever removes every agent, the program puts one new agent on a
node of its own, with a fresh random [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") and all *T* tokens of the world.
With the threshold at 20, that world is extinct at once. With the threshold
at 0, a run can never stop as extinct: when its agents are counted, the world
always holds at least this one.

## How the notebook counts the dead

A world that dies out did not end up anywhere: the last stretch of its life is
its dying. So where the worlds of a setting end up is measured over the worlds
that lived to the end. The dead are reported beside them, as an outcome of
their own ([The settled life of a world](median-and-quantiles.md#the-settled-life-of-a-world)).

Of the 30 baseline worlds of 10,000 tokens, four died: after 4, 313, 1,547
and 2,184 iterations. Of the 30 of 100,000 tokens, none did
([A world's life](a-worlds-life.md)).

Iterations are numbered from 0. So "died after *N* iterations" means that the
iteration numbered *N* − 1 was the last to run, and it ended with 20 agents or
fewer.

## Where it is used

- [A world's life](a-worlds-life.md): the four [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") worlds of 10,000 tokens that
  died out, one of them in its fourth iteration.
- [Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break):
  four of ten worlds of 5,000 tokens died out when their brains could see, and
  eight of ten when those brains also changed only when copied.
- [Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why): all ten [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") worlds that kept
  nothing at home died out, within 26 to 47 iterations.
- [How this notebook is written](how-this-notebook-is-written.md): the rule
  that the dead are counted apart.

## Where to go next

- [One iteration, step by step](one-iteration.md): the cleanup after which the
  agents are counted.
- [The settled life of a world](median-and-quantiles.md#the-settled-life-of-a-world): why the dead are measured
  apart from the living.
- [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md):
  small worlds near the line, and below it.

## How our view changed

- **9 October 2026.** The page first gave a single reason for stopping at 20:
  it saves running a dying world to its end. Small worlds run with the
  threshold at 0 then showed that a world at 20 agents is not always dying:
  all six of 800 and 1,600 tokens lived on
  ([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)).
- **23 October 2026.** Until now the page said that a world which falls to 20
  agents or fewer in the reproduction phase, and recovers in the game, is not
  extinct. No world can do that, because a game adds no agents: only the
  reproduction phase makes children (`GraphOfLifeSimple.py`, read when this
  page was rewritten). The count after the game misses no world.

<!-- turns -->
---

← [How a brain changes](mutation.md) · [How the simulation works](../README.md#how-the-simulation-works)

> [!info]- Pages that link here (12)
> [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [One iteration, step by step](one-iteration.md) · [Every setting](settings.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [The life of a B1 world](a-worlds-life.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [The lab, the engine and pilots](the-lab.md) · [Many worlds at once: the median, bands, the settled life and the spread](median-and-quantiles.md) · [How sure can we be? The bootstrap, permutation tests, the Wilson interval and how many seeds](bootstrap.md)
<!-- /turns -->
