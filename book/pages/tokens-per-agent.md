# Tokens per agent: mean, median, extremes

Four simple numbers describe how the tokens of a world are held, after every
phase. The viewer shows them in its *General* section.

## The four

For the *n* agents alive, holding τ₁, …, τₙ tokens, with *T* = Σ τᵢ:

| statistic | field | definition |
|---|---|---|
| mean tokens | `meanTokens` | *T* / *n* |
| median tokens | `medianTokens` | the middle value of the sorted τᵢ ([Median](median-and-quantiles.md)) |
| richest | `maxTokens` | max τᵢ |
| poorest | `minTokens` | min τᵢ |

## What each says

- **The mean is not about wealth at all.** Tokens are conserved, so *T* is
  the same after every phase — 100,000 in the baseline worlds the notebook
  studies, 10,000 in those it teaches with — and the mean is *T* divided by the number
  of agents. It rises when agents die and falls when they are born. A settled
  baseline world has a mean of about 7 at 100,000 tokens, 9 at 10,000.
- **The median is the typical agent.** About 4.5 in a settled baseline world
  (5 at 10,000 tokens).
  The gap between mean and median is itself a reading of inequality: a few
  rich agents pull the mean up and leave the median where it is.
- **The richest** is a reading of how far concentration goes. In a settled
  baseline world the richest agent holds about 2% of all tokens at 100,000
  tokens — some 2,100 — and about 8%, some 800, at 10,000
  ([Where do the tokens go?](where-do-the-tokens-go.md)).
- **The poorest is almost always 1.** An agent holding 0 tokens after a phase
  has already been removed by the cleanup, so the poorest survivor holds at
  least 1 ([Births and the four ways to die](births-and-deaths.md)).

## An example

Five agents with 1, 2, 2, 5 and 40 tokens: the mean is 50/5 = 10, the median
2, the richest 40, the poorest 1. The richest holds 80% of everything; the
mean is five times the median.

<!-- turns -->
---

← [The richest tenth's share](richest-tenth.md) · [Wealth: who holds the tokens](../README.md#wealth-who-holds-the-tokens) · [Entropy and evenness](entropy-and-evenness.md) →
<!-- /turns -->
