# Brain diversity

How many different [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") does a world hold? Three statistics count
genotypes ([Genotypes and the family tree](genotype.md)).

## The three

| field | definition |
|---|---|
| `distinctBrains` | the number of different genotypes among the living |
| `brainDiversity` | `distinctBrains` / `nodes`: distinct genotypes per [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") |
| `distinctParents` | the number of different **parent** genotypes among the living |

- `brainDiversity` = 1 means no two agents share a genotype; a small value
  means a few genotypes have been copied over much of the world. A settled
  [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world has about 0.63: on average a genotype is carried by about 1.6
  agents ([Genotypes and lineages](genotypes-and-lineages.md)).
- `distinctParents` looks one step up the family tree. It is deliberately
  shallow: it says nothing about how many separate lines there are, which is
  what [Families](families.md) and [the common ancestor](common-ancestor.md)
  measure.

## Counting is not the whole story

Two worlds with 1,000 genotypes each can be very different: in one, every
genotype is carried by about one agent; in the other, one genotype is carried
by half the world and the rest by one agent each. The **[effective number](effective-number.md "How many kinds there really are, when kinds are counted by how much of the population they hold.") of
genotypes**, 2^H with *H* the entropy of the genotypes' shares, weighs the
common ones more ([Entropy and evenness](entropy-and-evenness.md)). In a
settled baseline world it is about 0.45 per agent at 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") (0.49 at
10,000), against 0.63 distinct genotypes per agent at both sizes
([How even is a world?](where-do-the-tokens-go.md#how-even-is-a-world)).

## A caution about names

A genotype number changes whenever a brain is offered a change and takes it
([How a brain changes](mutation.md)). With `mutation_sparsity` = 0 the change
changes nothing, yet the number still changes. Counting genotypes counts
names; whether different names are different brains has to be checked
separately ([Do the brains matter?](do-the-brains-matter.md)).

<!-- turns -->
---

← [How a signal fades through layers](vanishing-signal.md) · [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are) · [Families](families.md) →

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [How alike the agents are](how-alike-the-agents-are.md) · [How different are the agents?](how-different-are-the-agents.md) · [What a run records](frames-and-stats.md) · [Effective numbers: how many kinds there really are](effective-number.md)
<!-- /turns -->
