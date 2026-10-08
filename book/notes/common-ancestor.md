# The common ancestor of the living

Every living agent's genotype descends, through the family tree, from one of
the founders ([Genotypes](genotype.md)). Go back far enough and some
genotype is an ancestor of **all** of them. How far back is that?

## Definition

At iteration *t*, let *c*(*g*) be the number of living agents whose genotype
is *g* or descends from *g*. The **common ancestor** of a share *q* of the
living is the **newest** genotype *g* with

$$
c(g) \ge q \times (\text{number of living agents}).
$$

The book uses *q* = 1 (all of the living), 0.9 and 0.5. Its **depth** is
how many iterations back that genotype was born: *t* minus the iteration of
the first frame it appears in.

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
means several lines have coexisted for a long time. In the baseline the
depth for all of the living rises and falls in a saw-tooth: it grows by one
per iteration while no line takes over, and drops when one does
([Chapter 16](../chapters/16-genotypes-and-lineages.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
