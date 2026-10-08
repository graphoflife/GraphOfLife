# Genotypes and the family tree

Every brain carries a whole number, its **genotype**. Two agents with the
same genotype have brains with exactly the same weights, because they hold
copies of one brain. This note says how genotypes are numbered, how they form
a family tree, and what the book counts with them.

## How genotypes are numbered

- The founders' brains get the genotypes 1, 2, …, *n*.
- **Copying keeps the number.** A child's brain before it changes, and the
  brain a node takes on from the winner of the game, are copies: same
  weights, same genotype.
- **Changing gives a new number.** When a brain changes
  ([How a brain changes](mutation.md)), it gets the next unused number, and
  remembers the number it had as its **parent genotype**. Numbers are never
  reused, and a parent always has a smaller number than its child.

## Genotypes are not agents

An agent keeps its id for life, but its genotype can change in two ways:
its brain can mutate, or its node can be won in the game, and then it holds a
copy of the winner's brain. Many agents can share a genotype, and a genotype
can live on in many places after the agent it first appeared in has died.

Two different counts follow:

- the number of **agents** alive — `nodes` in the statistics;
- the number of **distinct genotypes** among them — `distinctBrains`.

## The family tree

Following parent genotypes back gives a **tree**: every genotype has one
parent, back to a founder's genotype, which has none. The frames record, for
every living agent, its genotype and that genotype's parent, so the tree can
be rebuilt from a run. Every genotype first appears in some frame; the
iteration of that frame is the genotype's **birth**.

The book uses the tree in three ways:

- **Families** — how many different genotypes of eight iterations earlier the
  living descend from ([Families](families.md)).
- **The common ancestor** — the newest genotype from which all (or nine in
  ten, or half) of the living descend ([The common ancestor](common-ancestor.md)).
- **Founder lines** — how many of the living descend from each founder,
  drawn as a [Muller plot](muller-plot.md).

## Lifetimes of genotypes

A genotype is *alive* after a game if at least one living agent carries it.
Its lifetime is from the first game after which it is alive to the last
([Age and lifetime](age-and-lifetime.md); [Chapter 16](../chapters/16-genotypes-and-lineages.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
