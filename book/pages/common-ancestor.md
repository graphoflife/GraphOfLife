# The common ancestor of the living

Every living [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s genotype descends, through the family tree, from one of
the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") ([Genotypes](genotype.md)). Go back far enough and some
genotype is an ancestor of **all** of them. How far back is that?

## Definition

At [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") *t*, let *c*(*g*) be the number of living agents whose genotype
is *g* or descends from *g*. The **common ancestor** of a share *q* of the
living is the **newest** genotype *g* with

$$
c(g) \ge q \times (\text{number of living agents}).
$$

The notebook uses *q* = 1 (all of the living), 0.9 and 0.5. Its **depth** is
how many iterations back that genotype was born: *t* minus the iteration of
the first [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") it appears in.

While the living still descend from more than one founder, no genotype is an
ancestor of all of them, and the depth for *q* = 1 is undefined.

## How it is computed

Genotype numbers are handed out in order, and a parent always has a smaller
number than its child. So take the living genotypes with their counts and
work from the **largest number down**: pop the largest, check whether its
count reaches each share, and add its count to its parent's (putting the
parent in the queue if it is not there). A genotype is reached only after all
its descendants have been added into it, so the first genotype found to hold
a share is the newest that does. In the code: `shared_ancestors` in
`gol_analysis.py`, checked every 25 iterations.

## What the depth says

A shallow common ancestor means the living descend from a recent single
line: one genotype, a few iterations ago, took over — a **sweep**. A deep one
means several lines have coexisted for a long time. In the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") the
depth for all of the living rises and falls in a saw-tooth: it grows by one
per iteration while no line takes over, and drops when one does
([Genotypes and lineages](genotypes-and-lineages.md)).

<!-- turns -->
---

← [Families](families.md) · [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are)

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (7)
> [Glossary](glossary.md) · [The first hundred iterations](the-first-hundred-iterations.md) · [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [Do the brains matter?](do-the-brains-matter.md) · [Genotypes and the family tree](genotype.md) · [Brain diversity](brain-diversity.md)
<!-- /turns -->
