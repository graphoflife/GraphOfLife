# Age and lifetime

Two different questions: *how old are the agents alive now?* and *how long
does an agent live?* They have different answers, and the difference is
worth understanding.

## Age

An agent's **age** is the number of iterations since it was born: 0 in the
iteration of its birth, 1 in the next, and so on. Founders are born at
iteration 0. Every frame records the age of every living agent
([What a run records](frames-and-stats.md)).

Note that an agent keeps its id and its age when its node is won in the
game; only its brain changes. Age is the age of the agent, not of its
genotype.

## Lifetime

An agent's **lifetime** is how many games it lived through: if it was born
in iteration *b* and the last game after which it was alive was in iteration
*d*, its lifetime is

$$
L = d - b + 1 .
$$

An agent born in a reproduction phase that is gone before the game of the
same iteration has lifetime 0 and is not counted; one that survives exactly
one game has lifetime 1.

## Lives that have not ended

At the end of a run, some agents are still alive. Their lifetime is not
known — only that it is **at least** what they have lived so far. Leaving
them out would make lives look shorter than they are (the long lives are the
ones most likely to be unfinished); counting them as if they ended would do
the same. The right way is the **Kaplan–Meier estimate**
([Survival curves](kaplan-meier.md)), which counts an unfinished life as
"survived at least this long" and no further.

## Why the living are older than the dead

Suppose half of all agents live 1 iteration and half live 99. The **average
lifetime** is 50. But look at the world at any moment: an agent of the
second kind is alive at 99 times as many moments as one of the first, so 99 in
100 of the agents you see are long-lived ones. The **living are a sample
weighted by lifetime**. This is why, in
[Chapter 12](../chapters/12-births-deaths-and-ages.md), half of all agents
are gone within ten iterations, while the median age of the agents alive at
the end is 72.

In general: in a world in a steady state, where lifetimes *L* occur with
frequency *p*(*L*) and have mean μ, the agents alive at a random moment have
lifetimes with frequency

$$
\frac{L \, p(L)}{\mu} ,
$$

so long lives are over-represented in proportion to their length, and each
such agent is equally likely to be at any point of its life. This is the
*inspection paradox* of renewal theory. (The baseline world is not exactly in
a steady state — its births slow down over the run — but the effect is the
same in kind.)

## The risk of dying at each age

The survival curve *S*(*L*), the share of lives lasting at least *L*
iterations, is built from one number per age: of the agents whose lives
reach age *a*, the share whose lives end there,

$$
h(a) = \frac{d(a)}{n(a)} ,
$$

with *n*(*a*) the lives lasting at least *a* (finished or not) and *d*(*a*)
the finished lives of exactly *a*. This is the **hazard**, the chance of
dying at age *a* for an agent that got there, and

$$
S(L) = \prod_{a < L} \bigl(1 - h(a)\bigr) .
$$

If the hazard is the same at every age, an agent's past says nothing about
its future: lives are **memoryless**, and *S*(*L*) falls along a straight
line when it is drawn on a logarithmic axis against *L*. A hazard that falls
with age means that those who survive become safer — the young die most,
as in a population with high infant mortality. A hazard that rises with age
is ageing in the usual sense.

## Cohorts

Agents born in the same stretch of iterations are a **cohort**. Following
each cohort's lives says how life expectancy changes over a world's life.
Cohorts born late can only be followed to the end of the run, and the
Kaplan–Meier estimate leaves their unfinished lives open rather than
cutting them short. To compare cohorts on equal terms, each is measured by
how many of its first 100 iterations an agent lives on average, the
**restricted mean lifetime** Σ *S*(*L*) for *L* = 1 to 100: every cohort born
at least 100 iterations before the end can be measured by it in full
([Chapter 13](../chapters/13-how-ages-change.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
