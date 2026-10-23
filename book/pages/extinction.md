# When a world ends

A run goes for a fixed number of [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") — 3,000 for the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") runs —
unless its world **dies out** first.

## The rule

After every iteration — that is, after its [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") and the [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") that follows
— the number of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") is counted. If it is at most
[`extinction_threshold`](settings.md#extinction_threshold), 20 in the
baseline, the world counts as **extinct** and the run stops.

The check is made once per iteration, after the game; not after the
reproduction phase. A world that falls to 20 or fewer during reproduction and
recovers in the game is not extinct.

## Why 20 and not 0?

A world of a handful of agents is not a population any more, and its
statistics — inequality, [clustering](clustering.md "How often an agent's neighbours are also each other's neighbours."), families — mean little. Stopping at 20
also saves running a dying world to its end.

But a world at 20 agents is not always dying.
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) ran
small worlds with the threshold at 0: all six worlds of 800 and 1,600 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")
lived to iteration 600, and five of them were worlds the threshold had stopped
after their first game; one fell to 4 agents and came back. So "extinct" in
this notebook means exactly what the rule says — **fell to 20 agents or fewer** —
which is not always the same as dying. An experiment that cares about
extinction should run a condition with the threshold at 0 beside it.

## An empty world

If a cleanup ever removes every agent, the program puts one new agent with a
fresh random [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") and all *T* tokens on a node of its own. With the
threshold at 20, that world is extinct at once.

## How the notebook counts the dead

A world that dies out did not end up anywhere: the last stretch of its life
is its dying. So where the worlds of a setting end up is measured over the
worlds that lived to the end, and the dead are reported beside them, as an
outcome of their own ([The settled life of a world](settled-life.md)). Of the
30 baseline worlds of 10,000 tokens, four died: after 4, 313, 1,547 and 2,184
iterations; of the 30 of 100,000 tokens, none
([A world's life](a-worlds-life.md)).

"Died after *N* iterations" means: the iteration numbered *N* − 1 was the
last to run, and it ended with 20 agents or fewer.

<!-- turns -->
---

← [How a brain changes](mutation.md) · [How the simulation works](../README.md#how-the-simulation-works)

> [!info]- Pages that link here (11)
> [How to read this notebook](../README.md) · [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [One iteration, step by step](one-iteration.md) · [Every setting](settings.md) · [What the baseline world is](what-the-baseline-world-is.md) · [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md) · [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [The lab, the engine and pilots](the-lab.md) · [The settled life of a world](settled-life.md)
<!-- /turns -->
