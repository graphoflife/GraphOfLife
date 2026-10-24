# Muller plots

How do lines of descent grow, shrink and replace each other over time? A
**Muller plot** — named after the geneticist H. J. Muller, who drew such
pictures in 1932 — shows it as stacked layers.

## How the notebook draws one

1. **Choose the lines.** Fix a moment *t*₀ and a set of ancestors: the 100
   [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 0, or every genotype alive after the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") of some
   iteration *t*₀. Each ancestor starts a **line**: itself and every genotype
   descended from it ([Genotypes and the family tree](genotype.md)).
2. **Count.** After every later game, follow each living [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s genotype
   up the family tree until one of the chosen ancestors is reached, and count
   the agents in each line. Divide by the number of living agents: each
   line's **share** of the living. The shares add up to 1.
3. **Stack.** Draw the shares as layers stacked one on another, so that the
   top of the stack is always at 1. The lines that ever held the largest
   share are drawn in colour, at the bottom; all the others together in grey
   on top.

## How to read one

- The **height** of a layer at a moment is that line's share of the world.
- A layer that **swells** is a line spreading; one that **thins out** and
  vanishes is a line dying out.
- A layer that fills the **whole height** means everyone alive descends from
  that one ancestor: the line has **swept** the world.

A line spreads in two ways in Graph of Life: by births, and — far more often
— by winning nodes in the game, which copies its [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") onto the conquered
node ([How the game is played](how-the-game-is-played.md)). A Muller plot
does not tell them apart.

## In this notebook

- [The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations): the lines of
  the 100 founders in the first 300 iterations of the world with [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1.
- [Genotypes and lineages](genotypes-and-lineages.md): the families of
  the genotypes alive at a later moment, in two worlds.

<!-- turns -->
---

← [Logarithmic axes and power laws](logarithmic-axes.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Fitting a straight line, and R²](least-squares.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (4)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [Genotypes, families and the common ancestor](genotype.md)
<!-- /turns -->
