# Splitting variation: where, who and chance, and between worlds and within them

> [!question] Questions of this page
> - [Agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") decide differently. Is that because of where they are, who they
>   are, how they answer their place, or chance?
> - Worlds of different [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") differ, and each world wanders. How much of the
>   variation is the seed, and how much is time?

## In short

A spread of numbers can often be split exactly into parts with different
causes. The notebook does it twice. For a decision, put every [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") into
every situation with the same random draws. The variance then splits into
**where** an agent is, **who** it is, **who it is, there**, and **chance**.
For a statistic of many worlds, cut each world's life into stretches. The
variance then splits into what lies **between** the worlds, fixed by the
seed, and what lies **within** each: its wander over time. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") most of the variation
of a world is time, not seed.

## Where, who and chance: splitting a decision's variance

Agents decide differently from one another. There are four possible reasons:
- they are in different situations (**where**);
- their brains differ (**who**);
- each brain reacts differently to the situation it is in (**who, there**);
- or the random inputs every brain reads differ (**chance**).

This section shows how the variance of a decision splits into those four
parts.

### The design

Take *B* brains and *S* situations. Put every brain into every situation *K*
times, each time with fresh random inputs. But use the *same* *K* draws for
every brain, so that the brains meet exactly the same inputs. Write *d*_{bsk}
for what brain *b* decides in situation *s* on draw *k*. It may be a share of
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), or a yes or no as 1 or 0. Then take the averages:

- *d̄*_b: brain *b* over all situations and draws;
- *d̄*_s: situation *s* over all brains and draws;
- *d̄*_{bs}: brain *b* in situation *s*, over its draws;
- *d̄*: everything.

They split the sum of squared deviations from the grand mean exactly into
four parts. Statisticians call this a two-way analysis of variance with
replicates:

$$
\begin{aligned}
\text{situation} &= B K \sum_s (\bar d_s - \bar d)^2 \\
\text{brain} &= S K \sum_b (\bar d_b - \bar d)^2 \\
\text{brain in situation} &= K \sum_{b,s} (\bar d_{bs} - \bar d_b - \bar d_s + \bar d)^2 \\
\text{chance} &= \sum_{b,s,k} (d_{bsk} - \bar d_{bs})^2
\end{aligned}
$$

In words: the first two parts say how far each situation's average lies
from the grand mean, and each brain's. The third says how far each brain in
each situation lies from what those two would predict. The last says how far
each draw lies from its brain's average in its situation. Each part divided by their sum is a **share** of the variance.

### How to read the shares

- **Situation** large: every brain does the same thing in the same place,
  and agents differ because their places do.
- **Brain** large: some brains always do more of it than others, wherever
  they are. It is a trait, like a temperament.
- **Brain in situation** large: brains differ in *how they respond*. One
  gives more when rich, another when poor.
- **Chance** large: the random inputs decide, and a brain does not do the
  same thing twice.

"Do agents behave the same in the same circumstance?" is answered by the last
three together, against chance alone. If brain and brain in situation are
small, agents in the same place act alike, and their differences are the
dice.

### Units as well as shares

Shares hide sizes. A decision every agent makes the same way has a tiny
variance, and its shares can be anything. So the decision's spread is also
given in its own units. That is how far apart two agents in the same
situation and draw are, as the standard deviation over agents. It is set
against how far one brain is from itself over the draws.

[How different are the agents?](how-different-are-the-agents.md) uses this
split, with a small worked example.

## Between worlds and within them

The worlds of thirty seeds differ, and each world also changes over time.
How much of all the variation lies **between** the worlds, something each
seed fixes? And how much lies **within** each world, its wander over time?
This section splits it.

### The split

As for the [autocorrelation](autocorrelation.md), cut each of *N* worlds'
lives, from [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 100 on, into stretches of 100 iterations. Take the mean
of the statistic in each. A world of 10,000 tokens, run to iteration
2,999, gives *m* = 29 levels; a world of 100,000 tokens, run to 1,499, gives
14. Write *x*ᵢⱼ for the level of world *i* in stretch *j*, and *x̄*ᵢ for
world *i*'s own average of its levels.

- **Between:** how much the worlds' own averages differ. It is the variance
  of *x̄*₁, …, *x̄*_N, with *N* − 1 in the denominator:

  $$
  B = \frac{1}{N-1}\sum_{i=1}^{N} \bigl(\bar x_i - \bar{\bar x}\bigr)^2 .
  $$

- **Within:** how much a world moves around its own average. It is the
  variance of each world's *m* levels, with *m* − 1 in the denominator,
  averaged over the worlds:

  $$
  W = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{m-1}\sum_{j=1}^{m} \bigl(x_{ij} - \bar x_i\bigr)^2 .
  $$

The **share between** is *B* / (*B* + *W*).

### How to read the share between

- Close to 1: the worlds differ, but each stays where it is. Knowing the seed
  tells you the level.
- Close to 0: every world covers much the same range over time. The seed
  tells you little.

In the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds of 100,000 tokens the share between is a fifth to a
quarter for most statistics: 21% for the number of agents. It is half for the
share of agents in the core. At 10,000 tokens it is a tenth to a quarter, and
17% for the number of agents
([How much does the seed decide?](how-much-does-the-seed-decide.md)). So most
of the variation is time, not seed.

### A caution

Even *B* is partly wander. The average of levels that are correlated over
several stretches still carries some of the noise of the wander. So a share
between of 17% is an upper bound on what the seed itself decides. The sharper
test asks whether a world's first half predicts its second half, across
worlds ([Is a correlation real?](correlation.md#is-a-correlation-real)).

## Where to go next

- [How different are the agents?](how-different-are-the-agents.md): the split
  of where, who and chance, measured in B1 and [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.").
- [How much does the seed decide?](how-much-does-the-seed-decide.md): the
  share between, statistic by statistic.
- [How long a world remembers](autocorrelation.md): the wander that the share
  within measures.
- [Many worlds at once](median-and-quantiles.md): the spread between worlds,
  as one number.

<!-- turns -->
---

← [Fitting a power law properly](power-law-fit.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Effective numbers: how many kinds, and how many directions](effective-number.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (6)
> [How much does the seed decide?](how-much-does-the-seed-decide.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Do the brains matter?](do-the-brains-matter.md) · [How different are the agents?](how-different-are-the-agents.md) · [Every world side by side](every-world-side-by-side.md) · [How worlds are measured](how-worlds-are-measured.md)
<!-- /turns -->
