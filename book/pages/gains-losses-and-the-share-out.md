# Gains, losses and the share-out

Every [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") records, for every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") alive at its end, its change of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")
over the phase: `delta` = tokens now − tokens when the phase began (a newborn
counts its whole endowment as gained). Five statistics summarise it.

## The five

| field | definition |
|---|---|
| `gainers` | agents alive at the end with `delta` > 0 |
| `losers` | agents alive at the end with `delta` < 0 |
| `maxTokenAdded` | the largest `delta` |
| `maxTokenLost` | minus the smallest `delta`, as a positive number |
| `redistributed` | the tokens the [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") dealt out among the survivors |

The agents with `delta` = 0 are not counted by either: they are
`nodes` − `gainers` − `losers`.

## Why gains and losses do not balance

Tokens are conserved, so one might expect the gains to equal the losses.
They do not, because a frame only describes agents **still alive**. An agent
that died in the phase lost everything, but it is not in the frame, so its
loss is never counted — while the tokens it let go of turn up as gains of the
survivors (the cut-off agents' tokens through the share-out, others through
stakes). Summed over the frame,

$$
\sum_{\text{survivors}} \text{delta} = \text{tokens held by the agents that died, at the start of the phase} .
$$

## The share-out

The cleanup removes the agents that hold nothing and those cut off from the
largest piece of the network, pools the tokens of the dead — only the cut-off
ones hold any — and deals them out by one multinomial draw: each token goes
to a survivor chosen uniformly at random ([Seeds and the random
stream](random-numbers.md)). `redistributed` is the size of that pool.

**An example.** 100 tokens are dealt out among 1,000 survivors. Each survivor
gets a binomial number of them, with mean 0.1: about 90% get none, 9% get
one, and fewer than 1% get two or more. For an agent holding 2 tokens, one
extra token is a 50% gain — the share-out is a small lottery that matters most
to the poor ([Gains and losses](gains-and-losses.md)).

## In the code

`delta` is computed by the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") (`GraphOfLifeSimple.py`, `_frame`); the
statistics by `gol_series.frame_stats`; `redistributed` comes from the
cleanup report of each frame.

<!-- turns -->
---

← [Entropy and evenness](entropy-and-evenness.md) · [Wealth: who holds the tokens](../README.md#wealth-who-holds-the-tokens)

**Also about:** [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens).

> [!info]- Pages that link here (3)
> [Glossary](glossary.md) · [Gains and losses](gains-and-losses.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
