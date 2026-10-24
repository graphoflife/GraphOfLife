# What would count as open-ended evolution

No run can show that evolution goes on for ever. A run is finite, and the
budget is finite: a world of 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") runs about 100,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")
in a day and a half, and a world of 100,000 tokens a tenth as far in two
days. What a run can show is that, as far as it went, nothing stopped:
- later [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") beat their whole past;
- they keep getting better;
- they keep doing new things;
- the world stays one in which much can happen.

This note writes the test down before any of the long runs it will judge
exists. A thesis written after its results can be bent to fit them; one
written before cannot.

## The test

A configuration of rules shows open-ended evolution, as far as a run can,
when **at least two of three new worlds** pass all four parts below.
- New means [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") that played no part in choosing the configuration: a
  configuration picked as the best of many is lucky as well as good, and
  its own seed says nothing about that.
- Every world played counts, the ones that fail included.

1. **It beats its whole past.** At every 10,000 iterations, the brains of
   that moment beat the best mixture of all earlier [eras](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") in [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.")
   contests: 100 iterations, played on both worlds, from both sides.
   - The best mixture is the Nash mixture of the eras' contest table: the
     blend of past brains that does best against everything.
   - Beating the last era is not enough. In a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") like this one every
     strategy has an answer, so a chain of "beats the one before" can go
     round in a circle for ever without getting anywhere, as rock, paper
     and scissors do.
   - The margin must be larger than the spread of an era played against
     itself.
2. **It keeps rising.** The eras' ratings, read against the best mixture
   of the others (Nash averaging, Balduzzi et al., 2018), rise over the
   second half of the run. The 95% interval of the slope must lie above
   zero, bootstrapped over contests.
3. **It keeps doing new things.** New kinds of behaviour appear faster than
   they would by [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") alone, and the old ones do not simply come back.
   - Kinds of behaviour means two things: kinds of staking
     ([How different are the agents?](how-different-are-the-agents.md)), and
     the signatures of the strategies written by hand. A signature is how
     a strategy answers a fixed set of situations
     ([What a brain can compute](what-a-brain-can-compute.md)).
   - "By mutation alone" is the same brains changed only by the random
     mutations their [lineages](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor.") collected over the same time, with no
     selection.
   - A world whose kinds come and go in a cycle does not pass.
4. **It stays a world in which much can happen**
   ([Healthy and rich](healthy-and-rich.md)):
   - it never falls to 20 [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.");
   - it meets every mark of the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") at every 10,000 iterations;
   - its agents are more diverse than those of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world;
   - its tokens are no more even than B2's: a
     [Gini coefficient](gini-coefficient.md) no lower than the least of
     B2's worlds read at the same iteration.

## What it does not ask

- **Speed.** Evolution may take as long as it likes, so long as within the
  run it does not stop.
- **Typical worlds.** "Two of three new worlds" asks whether a
  configuration *can* keep evolving, not whether it usually does. A
  configuration that does so in a few worlds is a find.
- **A particular kind of novelty.** Larger groups, deeper organisation and
  new signals all count, so long as they beat the past and were not seen
  before.

## Why these four parts

Each part closes a way to be fooled:
- **Part 1** closes the circle. Later brains beat earlier ones in every
  kind of world the notebook has played this long, but by less and less. A
  game that only goes round would pass a test of the last step.
- **Part 2** closes the plateau that still wins by a hair.
- **Part 3** closes the arms race that only swaps two answers back and
  forth.
- **Part 4** closes worlds that evolve only because they are poor in
  everything else, such as a few hundred agents holding everything evenly.

**Related notes:** [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) ·
[What a brain can compute](what-a-brain-can-compute.md) ·
[Healthy and rich](healthy-and-rich.md)

<!-- turns -->
---

← [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going) · [Exploitability](exploitability.md) →

**Also about:** [Start here](../README.md#start-here).

> [!info]- Pages that link here (7)
> [Glossary](glossary.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [Every era against three](every-era-against-three.md) · [What a brain can compute](what-a-brain-can-compute.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The health bar](health-bar.md)
<!-- /turns -->
