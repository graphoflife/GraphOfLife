# Brain diversity

How many different brains does a world hold? Three statistics count
genotypes ([Genotypes and the family tree](genotype.md)).

## The three

| field | definition |
|---|---|
| `distinctBrains` | the number of different genotypes among the living |
| `brainDiversity` | `distinctBrains` / `nodes`: distinct genotypes per agent |
| `distinctParents` | the number of different **parent** genotypes among the living |

- `brainDiversity` = 1 means no two agents share a genotype; a small value
  means a few genotypes have been copied over much of the world. A settled
  baseline world has about 0.63: on average a genotype is carried by about 1.6
  agents ([Chapter 17](../chapters/17-genotypes-and-lineages.md)).
- `distinctParents` looks one step up the family tree. It is deliberately
  shallow: it says nothing about how many separate lines there are, which is
  what [Families](families.md) and [the common ancestor](common-ancestor.md)
  measure.

## Counting is not the whole story

Two worlds with 1,000 genotypes each can be very different: in one, every
genotype is carried by about one agent; in the other, one genotype is carried
by half the world and the rest by one agent each. The **effective number of
genotypes**, 2^H with *H* the entropy of the genotypes' shares, weighs the
common ones more ([Entropy and evenness](entropy-and-evenness.md)). In a
settled baseline world it is about 0.49 per agent, against 0.63 distinct
genotypes per agent ([Chapter 25](../chapters/25-how-even-is-a-world.md)).

## A caution about names

A genotype number changes whenever a brain is offered a change and takes it
([How a brain changes](mutation.md)). With `mutation_sparsity` = 0 the change
changes nothing, yet the number still changes. Counting genotypes counts
names; whether different names are different brains has to be checked
separately ([Chapter 37](../chapters/37-do-the-brains-matter.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
