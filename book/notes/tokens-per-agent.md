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

- **The mean is not about wealth at all.** Tokens are conserved, so
  *T* = 10,000 in every baseline world after every phase, and the mean is
  10,000 divided by the number of agents. It rises when agents die and falls
  when they are born. A settled baseline world has a mean of about 9.
- **The median is the typical agent.** About 5 in a settled baseline world.
  The gap between mean and median is itself a reading of inequality: a few
  rich agents pull the mean up and leave the median where it is.
- **The richest** is a reading of how far concentration goes. In a settled
  baseline world the richest agent holds about 8% of all tokens — some 800
  ([Chapter 14](../chapters/14-where-do-the-tokens-go.md)).
- **The poorest is almost always 1.** An agent holding 0 tokens after a phase
  has already been removed by the cleanup, so the poorest survivor holds at
  least 1 ([Births and the four ways to die](births-and-deaths.md)).

## An example

Five agents with 1, 2, 2, 5 and 40 tokens: the mean is 50/5 = 10, the median
2, the richest 40, the poorest 1. The richest holds 80% of everything; the
mean is five times the median.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
