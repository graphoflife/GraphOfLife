# Gains, losses and the share-out

Every frame records, for every agent alive at its end, its change of tokens
over the phase: `delta` = tokens now − tokens when the phase began (a newborn
counts its whole endowment as gained). Five statistics summarise it.

## The five

| field | definition |
|---|---|
| `gainers` | agents alive at the end with `delta` > 0 |
| `losers` | agents alive at the end with `delta` < 0 |
| `maxTokenAdded` | the largest `delta` |
| `maxTokenLost` | minus the smallest `delta`, as a positive number |
| `redistributed` | the tokens the cleanup dealt out among the survivors |

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
to the poor ([Chapter 20](../chapters/20-gains-and-losses.md)).

## In the code

`delta` is computed by the engine (`GraphOfLifeSimple.py`, `_frame`); the
statistics by `gol_series.frame_stats`; `redistributed` comes from the
cleanup report of each frame.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
