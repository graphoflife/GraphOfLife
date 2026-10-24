# Bands: many worlds in one figure

Thirty worlds, each with a value at each of 3,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), drawn as
thirty lines, are a tangle nobody can read. The notebook draws them as a **band**
instead: a line for the typical world, and shaded regions for how far the
others spread around it.

![Thirty worlds as a band](../figures/measure/bands.svg)

## How a band is made

Given a statistic and a set of runs (say, the number of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") after every
[game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") in the 30 [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") runs):

1. **Stretches.** Cut the iterations into consecutive stretches. For a
   statistic measured every iteration over 3,000 iterations: 600 stretches of
   5 iterations (0–4, 5–9, …). For a graph statistic, measured only every 25
   iterations: 120 stretches of 25.
2. **One value per world and stretch.** For each world, the mean of the
   statistic over the iterations of the stretch at which it was measured.
3. **Across the worlds.** In each stretch, take the
   [quantiles](median-and-quantiles.md) of those values over the worlds that
   have one — a world that died has none after its death:
   - the **line** is the median, *Q*(0.5);
   - the **darker band** runs from *Q*(0.25) to *Q*(0.75): the middle half
     of the worlds;
   - the **paler band** runs from *Q*(0.05) to *Q*(0.95): nine worlds in ten.
4. **Draw** the line and the bands at the middle of each stretch.

In the code: `bands` in `gol_analysis.py`.

## How to read one

- The line is the **typical world**, not any single world. No world need
  follow it.
- A **narrow band** means the worlds agree at that moment; a wide one that
  they differ.
- A band says nothing about how a single world moves through time. A world
  near the top of the band at one moment can be near the bottom later. To see
  single worlds, the notebook draws them separately, as in
  [A world's life](a-worlds-life.md).
- After a world dies, the band is made of fewer worlds. In the baseline of
  10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") four of thirty died, and its last thousand iterations are bands
  of 26; in the baseline of 100,000 tokens none died.

<!-- turns -->
---

← [Median and quantiles](median-and-quantiles.md) · [Methods and statistics](../README.md#methods-and-statistics) · [The settled life of a world](settled-life.md) →

> [!info]- Pages that link here (5)
> [The life of a B1 world](a-worlds-life.md) · [How the game is played: what agents stake, and who wins the nodes](how-the-game-is-played.md) · [The shape of the network](what-shape-does-the-network-take.md) · [How worlds are measured](how-worlds-are-measured.md) · [Median and quantiles](median-and-quantiles.md)
<!-- /turns -->
