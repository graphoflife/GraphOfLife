# Age, lifetime and survival curves

## In short

An [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s **age** is how many [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") ago it was born; its **lifetime** is
how many [games](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") it lives through. They answer two different questions: how old
are the agents alive now, and how long does an agent live? The answers differ,
because the living are a sample weighted towards long lives. In the [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") worlds
of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), about half the lives end within ten iterations, yet the
[median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") age of the living is 54.

## Age

An agent's **age** is the number of iterations since it was born. It is 0 in
the iteration of its birth, 1 in the next, and so on. [Founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") are born at
iteration 0. Every frame records the age of every living agent, as `ages`
([What a run records](frames-and-stats.md)).

Age belongs to the agent, not to its [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."). When an agent's node is won in the
game, the agent keeps its id and its age; only its brain changes. So age is
the age of the agent, not of its genotype. The brain on an old node may have
arrived in the last game
([Questioning the mechanics](questioning-the-mechanics.md#two-roads-of-inheritance)).

## Lifetime

An agent's **lifetime** is how many games it lived through. Say it was born in
iteration *b*, and the last game after which it was still alive was in
iteration *d*. Then its lifetime counts the games from *b* to *d*, both
included:

$$
L = d - b + 1 .
$$

An agent that survives exactly one game has lifetime 1. It is alive after the
game of its birth iteration, and dies in the next iteration, at age 1. In
general, an agent of lifetime *L* dies in iteration *b* + *L*, at age *L*, in
one of that iteration's two [cleanups](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.").

An agent that never outlives a game has lifetime 0, and is not counted. That
is a newborn cut off in the cleanup after its birth, or one that dies in the
cleanup after its first game. There are many: in the settled B1 worlds of
100,000 tokens, nearly a third of all newborns
([Births, deaths and ages](births-deaths-and-ages.md)).

## Lives that have not ended

When a run stops, some agents are still alive. Their lifetime is not known,
only that it is **at least** what they have lived so far. Leaving them out
would make lives look shorter than they are, because the long lives are the
ones most likely to be unfinished. Counting them as if they had ended would do
the same. The right way is the **Kaplan–Meier estimate**
([Survival curves](#survival-curves-kaplanmeier)). It counts an unfinished
life as "survived at least this long", and no further.

## Survival curves (Kaplan–Meier)

*What share of agents live at least L iterations?* If every life had ended by
the end of the run, the answer would be a simple count. But some are still
going, and for them we know only that they lasted **at least** so long. The
**Kaplan–Meier estimate** (Kaplan and Meier 1958) uses exactly that much and
no more.

### The idea: survive one step at a time

To live at least *L* + 1 iterations, a life must first live at least *L*, and
then not end at *L*. So the share surviving to *L* + 1 is the share surviving
to *L*, times one more factor. That factor is the chance of not ending at *L*,
among the lives that got that far. Chaining the steps gives

$$
S(L) = \prod_{\ell < L} \left( 1 - \frac{d_\ell}{n_\ell} \right),
$$

where, for each length ℓ:

- *n*ℓ is the number of lives **at risk** at ℓ: the lives that lasted at
  least ℓ. They ended at ℓ or later, or are still going after lasting ℓ or
  more;
- *d*ℓ is the number of lives that **ended** at exactly ℓ.

In words: multiply, step by step, the shares of the lives still at risk that
did not end there. A life still going at the end of the run, having lasted
*L*₀, counts in *n*ℓ for every ℓ ≤ *L*₀. It is then dropped, without ever
counting as an end. That is how "at least *L*₀" is used: no more, no less.

### A worked example of survival

Take six agents' lives, in iterations; a + means still alive at the end of
the run: 1, 1, 2, 3+, 4, 6+.

| ℓ | at risk *n*ℓ | ended *d*ℓ | 1 − *d*/*n* | share living longer than ℓ |
|---|---|---|---|---|
| 1 | 6 | 2 | 4/6 | 0.667 |
| 2 | 4 | 1 | 3/4 | 0.500 |
| 3 | 3 | 0 (the 3+ leaves here) | 1 | 0.500 |
| 4 | 2 | 1 | 1/2 | 0.250 |
| 6 | 1 | 0 (the 6+ leaves here) | 1 | 0.250 |

So half of all lives last at least 3 iterations, and a quarter at least 5.
Ignoring the unfinished lives would have given a quarter for "at least 3":
only the 4, of the four finished lives. Counting them as if they ended would
understate the long lives too.

### Drawing a survival curve

The curve *S*(*L*) starts at 1 and falls in steps. The notebook draws it on
[logarithmic axes](correlation.md#logarithmic-axes-and-power-laws). Then both
the many short lives and the few long ones are visible
([Births, deaths and ages](births-deaths-and-ages.md),
[Genotypes and lineages](genotypes-and-lineages.md)).

The **median lifetime** is the smallest *L* with *S*(*L* + 1) ≤ ½: half of all
lives end by then. In the code, this is `kaplan_meier(lives)` in
`book_figures.py`, where a life is (first iteration, last iteration, still
going).

## Why the living are older than the dead

Picture a hospital where most patients go home after a day, and a few stay a
month. On any day you visit, the long stays fill far more beds than their
number suggests. A patient who stays thirty days is there on thirty days; one
who stays a day, on one.

Agents are the same. Suppose half of all agents live 1 iteration and half live
99. The **average lifetime** is 50. But look at the world at any moment. An
agent of the second kind is alive at 99 times as many moments as one of the
first. So 99 in 100 of the agents you see are long-lived ones: the **living
are a sample weighted by lifetime**.

That is why the living look so old. In the 30 B1 worlds of 100,000 tokens
(Experiment 10), about half of the agents that live through a game are gone
within ten iterations. Two thirds are, if the newborns that never do are
counted too. Yet the median age of the agents alive at the end is 54
([Births, deaths and ages](births-deaths-and-ages.md)).

In general, take a world in a steady state, where lifetimes *L* occur with
frequency *p*(*L*) and have mean μ. The agents alive at a random moment have
lifetimes with frequency

$$
\frac{L \, p(L)}{\mu} :
$$

each lifetime's frequency multiplied by its length, and divided by the mean so
that the frequencies add up to one. So long lives are over-represented in
proportion to their length, and each such agent is equally likely to be at any
point of its life. This is the *inspection paradox* of renewal theory.

The B1 worlds of 10,000 tokens are not exactly in a steady state: their births
slow down for over a thousand iterations. But the effect is the same in kind.
Those of 100,000 tokens are in one from iteration 500 on
([How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life)).

## The ages of a steady world

The same idea, turned into a formula for the ages of the living. Suppose that
in every iteration, always, *B* newborns live through their first game.
Suppose too that every agent's life follows the same survival curve *S*(*L*),
the share of lives lasting at least *L* iterations. The agents of age *a* alive
after a game were born *a* iterations earlier, and their lives lasted at least
*a* + 1 games. There are *B* · *S*(*a* + 1) of them. Summed over all ages, the
living number *B* · Σ *S*(*L*), and Σ *S*(*L*) over *L* ≥ 1 is the mean
lifetime μ. So in such a world

$$
\text{share of the living of age } a = \frac{S(a+1)}{\mu} ,
$$

and in particular the newborns, of age 0, are a share 1/μ of the living. This
is the **stationary population** of demography. Its ages follow from the
lifetimes alone. A world whose living are older than its lifetimes say has had
more births in the past than now.

Two examples. In the B1 worlds of 100,000 tokens, the lives begun from
iteration 500 to 1,399 last 41.3 iterations on average. So the newborns should
be 1/41.3 = 2.4% of the living, and the median age of the living 55. Over
iterations 500 to 1,499 the worlds had 2.3 to 2.6% newborns, and a median age
of 55 ([How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life)).

In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") every agent lives exactly 100 iterations. So in a steady B2 world, each
age from 0 to 99 holds one agent in a hundred. Such a world keeps its size only
if one agent in a hundred has a child in every iteration. B2's worlds in
Experiment 14 had 1.00 to 1.05 births per hundred agents from iteration 200 to
1,000 ([Building B2: a world that cannot break](a-world-that-cannot-break.md#old-age)).

## The risk of dying at each age

The survival curve *S*(*L*) is built from one number per age. An agent
**reaches** age *a* when its lifetime is at least *a*, and its life **ends
there** when its lifetime is exactly *a*. Of the agents whose lives reach age
*a*, the share whose lives end there is

$$
h(a) = \frac{d(a)}{n(a)} ,
$$

with *n*(*a*) the lives lasting at least *a* (finished or not) and *d*(*a*)
the finished lives of exactly *a*. This is the **hazard**, the chance of dying
at age *a* for an agent that got there. To last at least *L* iterations, a
life must get past every age before *L*, so

$$
S(L) = \prod_{a < L} \bigl(1 - h(a)\bigr) .
$$

If the hazard is the same at every age, an agent's past says nothing about its
future: lives are **memoryless**. Then *S*(*L*) falls along a straight line
when *S* is drawn on a logarithmic axis against a plain axis of *L*.

A hazard that falls with age means that those who survive are safer than the
young. The young die most, as in a population with high infant mortality.
Either each agent grows safer as it lives, or the fragile die first and leave
the robust. A hazard that rises with age is ageing in the usual sense.

The worlds show both ends. In the settled B1 worlds of 100,000 tokens the
hazard falls at every age. It is 21% at age 1, 2.7% at 10 and 1.1% at 100
([How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life)). In B2 it is 0 at
every age below 100, and every agent that reaches 100 dies at it: ageing at
its sharpest.

## Cohorts

Agents born in the same stretch of iterations are a **cohort**. Following each
cohort's lives says how life expectancy changes over a world's life. Cohorts
born late can only be followed to the end of the run. The Kaplan–Meier
estimate leaves their unfinished lives open, rather than cutting them short.

To compare cohorts on equal terms, each is measured by how many of its first
100 iterations an agent lives on average. That is the **restricted mean
lifetime**,

$$
\sum_{L=1}^{100} S(L) ,
$$

the area under the survival curve from *L* = 1 to 100. Every cohort born at
least 100 iterations before the end can be measured by it in full. In the
median B1 world of 100,000 tokens, an agent born in iterations 1 to 24 lived 5
of its first 100 iterations. One born from iteration 500 on lived 28 to 31
([How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life)).

## Where it is used

- [Births, deaths and ages](births-deaths-and-ages.md): how long agents live,
  and how old the living are, in the B1 worlds.
- [How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life): the ages of the
  living, the cohorts and the hazard, iteration by iteration.
- [Building B2: a world that cannot break](a-world-that-cannot-break.md#old-age):
  a world in which every agent dies at 100.

## Where to go next

- [Births and the four ways to die](births-and-deaths.md): how a life ends.
- [Genotypes and the family tree](genotype.md#lifetimes-of-genotypes): the
  lifetimes of brains, rather than of agents.

## References

- Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from
  incomplete observations. *Journal of the American Statistical Association*
  53, 457–481.

<!-- turns -->
---

← [The estate of the dead](estate.md) · [Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages) · [Reproduction statistics](reproduction-statistics.md) →

**Also about:** [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (11)
> [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [Births and the four ways to die](births-and-deaths.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [Genotypes, families and the common ancestor](genotype.md) · [How worlds are measured](how-worlds-are-measured.md) · [What the viewer can colour by](viewer-colours.md) · [Two numbers together: correlation, a fitted line and logarithmic axes](correlation.md)
<!-- /turns -->
