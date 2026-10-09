# Mobility

A world can be unequal in two very different ways. In one, the same agents
are rich year after year. In the other, wealth is just as unequal at every
moment, but who holds it keeps changing. The Gini coefficient
([The Gini coefficient](gini-coefficient.md)) cannot tell the two apart,
because it looks at one moment. **Mobility** compares two moments. Economists
measure it for incomes across years and generations (Shorrocks 1978; Chetty
et al. 2014); this note gives the three measures the book uses.

## Rank correlation over time

Take the agents alive at a moment *t* and again *k* games later. Rank them
by their tokens at each moment, the poorest first; agents with equal tokens
share the mean of their ranks. **Spearman's rank correlation** ρ is the
ordinary correlation ([Correlation](correlation.md)) of the two lists of
ranks:

- ρ = 1: every agent kept its place in the order of wealth;
- ρ = 0: the later order has nothing to do with the earlier;
- ρ < 0: the order turned round, the poor became the rich.

Ranks rather than tokens make ρ deaf to how unequal the world is, and to a
few very rich agents, who would dominate an ordinary correlation of the
tokens themselves. Drawn against *k*, ρ shows how long wealth is remembered.

## A table of moves

Sort the agents at *t* into fifths by their rank of wealth, from the poorest
fifth to the richest. Do the same *k* games later. For every pair of fifths,
count the agents that went from the one to the other, and add a column for
those no longer alive. Divide each row by its total, and each row says where
the agents of one fifth went. This is a **transition matrix** *P*: its entry
*Pᵢⱼ* is the share of the agents of fifth *i* found in fifth *j* later. A
world where nobody moves has 1 on the diagonal and 0 elsewhere.

## One number: the Shorrocks index

For *n* classes, with the diagonal entries *Pᵢᵢ* computed among the agents
still alive,

$$
M = \frac{n - \sum_i P_{ii}}{n - 1}
$$

(Shorrocks 1978). If nobody moves, every *Pᵢᵢ* is 1 and *M* = 0. If where an
agent ends up has nothing to do with where it started, every row is (⅕, ⅕,
⅕, ⅕, ⅕), the diagonal adds up to 1, and *M* = 1. So *M* runs from 0 (a rigid
world) to 1 (a world without memory of wealth). It can exceed 1, when the
order tends to reverse.

## Where it is used

[Chapter 20](../chapters/20-do-the-rich-stay-rich.md) measures all three in
the baseline worlds.

## References

- Chetty, R., Hendren, N., Kline, P. and Saez, E. (2014). Where is the land
  of opportunity? The geography of intergenerational mobility in the United
  States. *Quarterly Journal of Economics* 129, 1553–1623.
- Shorrocks, A. F. (1978). The measurement of mobility. *Econometrica* 46,
  1013–1024.
