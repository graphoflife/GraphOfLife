# Effective numbers: how many kinds there really are

A world has 1,600 genotypes. Does it have 1,600 kinds of agents? Not if one
genotype is carried by half the agents and the rest by one agent each. The
**effective number** of kinds counts kinds by how much of the population
they hold.

## The count

Let p_i be the share of agents of kind i (a genotype, a kind of behaviour, a
family). The effective number is

$$
D = \frac{1}{\sum_i p_i^2}
$$

the inverse of the chance that two agents drawn at random are of the same
kind (the Simpson index). Two properties make it the right count:

- if there are n kinds, equally common, D = n;
- if one kind holds nearly everyone, D is close to 1, however many rare kinds
  there are.

So D is "the number of equally common kinds that would be as diverse". It is
one of the Hill numbers, the family of diversity counts ecologists use (this
one, of order 2, weighs the common kinds most).

## Kinds that are similar

Two genotypes one mutation apart may act exactly alike. Counting them as two
kinds overstates diversity. That is why
[How different are the agents?](how-different-are-the-agents.md) groups brains into kinds **by what they do**, not by their weights: brains
whose decisions, in the same situations, differ no more than a brain differs
from itself between two sets of random inputs, are one kind. The effective
number of kinds of behaviour is then set beside the effective number of
genotypes: the first can be far smaller, if most genetic differences change
nothing an agent does.

## Two effective numbers

[Brain diversity](brain-diversity.md) uses a sibling of this count, 2^H with H
the entropy of the shares — the Hill number of order 1, which weighs rare kinds
a little more than the one above (order 2) does. Both are 1 for a single kind
and n for n equally common ones; on a population with many rare kinds, 2^H is
the larger. Where the notebook gives an effective number it says which.

<!-- turns -->
---

← [Where, who and chance: splitting a decision's variance](variance-split.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Many switches in few worlds: fractional factorial designs](fractional-factorial.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (3)
> [How alike the agents are](how-alike-the-agents-are.md) · [How different are the agents?](how-different-are-the-agents.md) · [Healthy and rich](healthy-and-rich.md)
<!-- /turns -->
