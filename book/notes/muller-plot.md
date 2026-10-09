# Muller plots

How do lines of descent grow, shrink and replace each other over time? A
**Muller plot** — named after the geneticist H. J. Muller, who drew such
pictures in 1932 — shows it as stacked layers.

## How the book draws one

1. **Choose the lines.** Fix a moment *t*₀ and a set of ancestors: the 100
   founders at iteration 0, or every genotype alive after the game of some
   iteration *t*₀. Each ancestor starts a **line**: itself and every genotype
   descended from it ([Genotypes and the family tree](genotype.md)).
2. **Count.** After every later game, follow each living agent's genotype
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
— by winning nodes in the game, which copies its brain onto the conquered
node ([Chapter 15](../chapters/15-how-the-game-is-played.md)). A Muller plot
does not tell them apart.

## In this book

- [Chapter 11](../chapters/11-the-first-hundred-iterations.md): the lines of
  the 100 founders in the first 300 iterations of the world with seed 1.
- [Chapter 17](../chapters/17-genotypes-and-lineages.md): the families of
  the genotypes alive at a later moment, in two worlds.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
