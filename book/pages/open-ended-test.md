# What would count as open-ended evolution

## In short

No run can show that evolution goes on for ever. So the notebook writes
down, before the long runs exist, what a run would have to show. A configuration of
rules passes when at least two of three new worlds pass four parts. Their
[brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") beat the best mixture of their whole past, and their ratings keep
rising. They keep doing new things, faster than [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") alone would. And
they stay worlds in which much can happen. Each part closes one way of being
fooled. They are a circle, a plateau and an arms race that swaps two
answers. The fourth is a world that evolves only because it is poor in all
else.

## Why write the test first

No run can show that evolution goes on for ever. A run is finite, and so is
the budget. A world of 10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") runs about 100,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") in a day
and a half. A world of 100,000 tokens runs a tenth as far in two days
([What a simulation costs](what-a-simulation-costs.md)). What a run can show
is that, as far as it went, nothing stopped:
- later brains beat their whole past;
- they keep getting better;
- they keep doing new things;
- the world stays one in which much can happen.

This page writes the test down before any of the long runs it will judge
exists. A thesis written after its results can be bent to fit them; one
written before cannot.

## The test

A configuration of rules passes the test when **at least two of three new
worlds** pass all four parts below. That is open-ended evolution, as far as a
run can show it.
- New means [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") that played no part in choosing the configuration. A
  configuration picked as the best of many is lucky as well as good, and its
  own seed says nothing about that.
- Every world played counts, the ones that fail included.

The four parts use the contests of
[Eras, contests and invasions](eras-and-contests.md). An **era** is the
brains of one moment of a world. A **contest** plays two eras against each
other in the same world, and scores the share of it the later one holds.

1. **It beats its whole past.** At every 10,000 iterations, the brains of
   that moment beat the best mixture of all earlier eras. They do so in
   contests a [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.") long: 100 iterations, played on both worlds, from both
   sides.
   - The best mixture is the Nash mixture of the eras' table of contests: the
     blend of past brains that does best against everything.
   - Beating the last era is not enough. In a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") like this one, every
     strategy has an answer. So a chain of "beats the one before" can go round
     in a circle for ever without getting anywhere, as rock, paper and
     scissors do.
   - The margin must be larger than the spread of an era played against
     itself.
2. **It keeps rising.** Read each era's rating against the best mixture of
   the others (Nash averaging; Balduzzi and others, 2018). The ratings rise
   over the second half of the run: the 95% interval of the slope, found by
   the [bootstrap](bootstrap.md) over contests, lies above zero.
3. **It keeps doing new things.** New kinds of behaviour appear faster than
   they would by mutation alone, and the old ones do not simply come back.
   - Kinds of behaviour means two things. One is kinds of staking
     ([How different are the agents?](how-different-are-the-agents.md)). The
     other is the signatures of the strategies written by hand: how a
     strategy answers a fixed set of situations
     ([What a brain can compute](what-a-brain-can-compute.md)).
   - "By mutation alone" means the same brains, changed only by the random
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

- **Speed.** Evolution may take as long as it likes, so long as it does not
  stop within the run.
- **Typical worlds.** "Two of three new worlds" asks whether a configuration
  *can* keep evolving, not whether it usually does. A configuration that does
  so in a few worlds is a find.
- **A particular kind of novelty.** Larger groups, deeper organisation and
  new signals all count, so long as they beat the past and were not seen
  before.

## Why these four parts

Each part closes a way to be fooled:
- **Beating the whole past** closes the circle. Later brains beat earlier
  ones in every kind of world the notebook has played this long, but by less
  and less. A game that only goes round would pass a test of the last step.
- **Rising** closes the plateau that still wins by a hair.
- **New things** closes the arms race that only swaps two answers back and
  forth.
- **Room** closes worlds that evolve only because they are poor in
  everything else, such as a few hundred agents holding everything evenly.

## Where to go next

- [Eras, contests and invasions](eras-and-contests.md): the contests and
  ratings the test is built from.
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick, held to both [baselines](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.").
- [What a brain can compute](what-a-brain-can-compute.md): the strategies
  whose signatures count as new things.
- [Healthy and rich](healthy-and-rich.md): what a world in which much can
  happen looks like.

## References

- Balduzzi, D., Tuyls, K., Pérolat, J. and Graepel, T. (2018). Re-evaluating
  evaluation. *Advances in Neural Information Processing Systems* 31.

<!-- turns -->
---

← [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Does evolution keep going?](../README.md#does-evolution-keep-going)

**Also about:** [Start here](../README.md#start-here).

> [!info]- Pages that link here (8)
> [Glossary](glossary.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Eras, contests and invasions: does a world keep getting better?](eras-and-contests.md) · [Every era against three](every-era-against-three.md) · [What a brain can compute](what-a-brain-can-compute.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The health bar](health-bar.md)
<!-- /turns -->
