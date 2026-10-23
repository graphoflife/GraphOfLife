# What makes a world interesting

> [!question] Questions of this page
> - What do we look for in a world before we ask how it evolves?
> - When is a world healthy, and when is it still interesting?
> - When do we give up on a [setting](settings.md "One number or switch of a run's rules, fixed before it starts.")?

## In short

We are looking for worlds in which evolution could go on for a very long
time. Such a world has to stay alive. It must not hang on a few [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), and it
must not fall apart when one agent dies. It should look like a space of about
three [dimensions](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."), as our own world does. And its agents should differ from
one another, in [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), links, age, genes and decisions, without the tokens
being spread too evenly.

We turn this picture into six criteria and check them at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 2,000.
- A world that meets all six is **healthy**.
- A world that is alive and misses at most one of the other five is still
  **interesting**: worlds vary, and one miss should not rule a world out.

## The six criteria

<!-- table criteria/criteria -->
| criterion | what it asks |
|---|---|
| alive | more than 20 agents at iteration 2,000, and not stopped as too slow |
| decentralised | no agent linked to more than 5% of the others, and 3 to 12 links per agent |
| resilient | one piece, with at least 80% of the agents in the 2-core; no single agent's loss cuts off more than 2% of the world, and no single link's more than 1% |
| almost three-dimensional | a dimension of 2 to 4, and a typical distance at least 1.5 times its random twin's: a true space, not a folded small world |
| diverse | tokens, links, ages and genotypes spread more widely than in B2's median world, and its agents' decisions vary more (effective kinds of staking) |
| not too even | a Gini of the tokens no lower than B2's least even world |
<!-- /table -->

Each criterion stands for one part of the picture.

1. **Alive.** A world that has died, or that runs so slowly it can never be
   played for long, tells us nothing about long evolution. Too slow means it
   did not reach iteration 2,000 within four hours of computing.
2. **Decentralised.** If one agent holds links to many others, the world
   depends on it: when that [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") dies, everything changes at once. We ask that
   no agent links to more than 5% of the world, and that agents have 3 to 12
   links each, as a network that is neither a tree nor a tangle.
3. **Resilient.** A healthy world does not break apart when one agent or one
   link disappears. We ask that the world is one piece, that most agents (80%)
   sit on [loops](loops.md "A path through the network that comes back to where it started without using a connection twice.") (the *[2-core](core-trees-leaves.md "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.")*), and that no single loss cuts off more than 2%
   of the agents, or 1% for a single link. These are the marks of the
   [health bar](how-to-tell-a-world-that-keeps-evolving.md)
   other than its dimension.
4. **Almost three-dimensional.** The physical side of the project asks
   whether a space can grow out of the agents' links
   ([Physical inspiration](physical-inspiration.md)). We
   measure a world's dimension by how fast the number of agents within a
   given distance grows. A tangled network can also read "three dimensions",
   so we ask a second thing: a typical distance between two agents at least
   1.5 times that in a *[random twin](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.")*, a network with the same number of links
   per agent wired at random. A folded world fails this; a true space passes
   ([What folds a world](what-folds-a-world.md)).
5. **Diverse.** Evolution needs differences to choose from. We ask that the
   agents' tokens, links, ages and [genotypes](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") spread more widely than in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world of [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), our standard world, by the *[diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.")*
   ([Healthy and rich](healthy-and-rich.md)). And we
   ask that their decisions differ more: the [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."), put in the same
   situations, fall into more kinds of staking than in B2's median
   world ([How different are the agents?](how-different-are-the-agents.md)).
   The kinds swing from moment to moment, so we take their mean over
   iterations 1,500 and 2,000.
6. **Not too even.** If almost every agent holds the same few tokens, there
   is little to win and little to select. We ask that the tokens are spread at
   least as unevenly as in B2's least even world, measured by the
   [Gini coefficient](gini-coefficient.md).

## Healthy, and interesting

A world that meets all six criteria is healthy. That is a high bar: in the
worlds of the notebook's first experiments, few met it.

A world that is alive and misses at most one of criteria 2 to 6 is
**interesting**. Worlds made with the same settings but different [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.")
differ. A setting that gives a healthy world in one seed may miss one line in
another, and we do not want to throw it away for that.

## The lines we compare with

The lines for criteria 5 and 6 come from B2's ten worlds of Experiment 16,
read in the same way and at the same iterations.

<!-- table criteria/references -->
| the line | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score of B2's median world | 0.53 | 0.51 |
| effective kinds of staking of B2's median world | 19.7 | 16.5 |
| Gini of the tokens of B2's least even world | 0.49 | 0.46 |
<!-- /table -->

## When a setting is marked bad

In the [screen of random worlds](a-screen-of-random-worlds.md),
every world draws its settings at random, fifty worlds at a time. After each
round we look at every value a setting took. A value is **marked bad**, and
drawn no more, only when both of these hold:
- it was tried in **at least ten worlds** where it could make a difference;
- **every one of them died**, or **none of them was interesting**.

Why ten? One or two dull worlds can be bad luck. Ten worlds that all failed
are a pattern. Beside each value marked bad, the judge also prints how likely
such a run of failures would be by chance alone, if the value made no
difference at all. That number is a warning, not part of the rule: when
interesting worlds are rare overall, ten failures in a row come easily.

The judge is `research/screen/judge.py`. The criteria live in one list,
`research/screen/criteria.py`, which this page's tables are made from, so the
page and the judge always agree.

## Where to go next

- [A screen of random worlds](a-screen-of-random-worlds.md):
  the rounds, and what each found.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  where the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") comes from.
- [Healthy and rich](healthy-and-rich.md): how the
  diversity score was made, and every world scored by it.
- [What would count as open-ended evolution](open-ended-test.md): the test a
  healthy world's long run will face.

<!-- turns -->
---

← [Healthy and rich](healthy-and-rich.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [The health bar](health-bar.md) →

> [!info]- Pages that link here (6)
> [Glossary](glossary.md) · [Stars and hubs](stars-and-hubs.md) · [The random twin: is a world a space?](random-twin.md) · [The health bar](health-bar.md) · [The diversity score](diversity-score.md) · [A screen of random worlds](a-screen-of-random-worlds.md)
<!-- /turns -->
