# The health bar

## In short

A world can only evolve for a long time if its network holds together. The
**health bar** says what that means, in seven marks that can be checked on a
world at one moment: one piece, made mostly of [loops](loops.md), with no
[agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") or link whose loss cuts off much of it, no [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."), a few links per agent,
and a dimension near three. A world that meets all seven has a healthy
network. The bar says nothing about the agents themselves: how much they
differ is the [diversity score](diversity-score.md).

## The seven marks

The bar was set in
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers).

| mark | what is measured | the line |
|---|---|---|
| one piece | how many pieces the network falls into | one |
| 2-core | the share of agents in the [2-core](core-trees-leaves.md): on a loop, or between loops | at least 80% |
| agent cut | the largest share of the world that one agent's loss would cut off | at most 2% |
| link cut | the same for one link: the [cut risk](cut-risk.md) | at most 1% |
| no hub | the share of the other agents that the best-connected agent is linked to | at most 5% |
| links per agent | the mean [degree](degree.md) | 3 to 12 |
| dimension | how fast balls grow ([Dimension and curvature from ball growth](ball-dimension-and-curvature.md)) | 2.3 to 4 |

## Why each mark is there

- **One piece.** The [cleanup](one-iteration.md#cleanup) after every phase
  keeps only the largest piece of the network; every agent cut off from it
  dies.
- **2-core.** Peel away every agent with fewer than two links, again and
  again; what is left is the 2-core. An agent outside it hangs on a tree, and
  a single cut can lose a tree whole.
- **The two cuts** count single points of failure. A [bridge](bridges.md)
  with a tenth of the world behind it is a tenth of the world that dies the
  moment no [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") crosses it in a game, and a death can split a world in the
  same way.
- **No hub.** A world that hangs on one agent changes all at once when that
  agent dies.
- **Links per agent.** A tree has fewer than two links per agent; a complete
  network, where everyone is linked to everyone, has as many as there are
  other agents. Three to twelve is neither.
- **Dimension.** The notebook hopes for a world that grows into a space like
  ours ([Physical inspiration](physical-inspiration.md)). The author's "almost
  three" is the middle of the range.

## How it is read

`research/atlas/profile.py` reads a world from its saved state at one moment,
without its [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."). One depth-first search finds the pieces and the worst cut
by one agent and by one link; peeling finds the 2-core; the dimension is the
[lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s own measure: how the number of agents within 1 to 5 steps
grows, around 24 agents spread through the world. The notebook's tables give
the result as "marks (of 7)".
Earlier experiments read the same lines from a run's recorded statistics, as
[medians](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") over many iterations.

## Two examples

- **[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") fails it.** A [pilot](pilot-worlds.md) of B1 at 5,000 tokens ([seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1,
  iterations 200 to 999) had 59% of its agents in the 2-core, a link whose
  loss could cut off 8% of the world, and a best-connected agent linked to 8%
  of the others. Its 3.3 links per agent and its dimension of 2.7 passed.
- **[B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") nearly passes it.** In Experiment 14, ten worlds of B2 at 5,000
  tokens, every agent sat in the 2-core; read every tenth iteration, no
  agent's loss could cut off more than 1.2% of a world; the best-connected
  agent was linked to 2.1 to 3.1% of the others; agents had 5.0 to 5.8 links.
  Nine worlds met every line. The tenth missed only the dimension, 4.19
  against the bar's 4.

## Read it with care

- **Two marks follow a world's size.** In networks like B2's, the hub mark
  fails small worlds, where a best-connected agent with 12 to 47 links is
  already more than 5% of the world, and the dimension mark fails large ones.
  In the mixed worlds of Experiment 17, the hub mark failed at 96% of the
  moments below 300 agents, while worlds of 600 to 1,000 agents met every
  mark at 98% of their moments
  ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md#the-marks-a-world-misses)).
- **The dimension mark cannot tell a space from a small world.** A B2 world
  of 600 to 1,500 agents passes it because it is small
  ([What folds a world](what-folds-a-world.md)). So the
  [screen of random worlds](a-screen-of-random-worlds.md) asks instead for a
  dimension of 2 to 4 and a typical distance at least 1.5 times its
  [random twin](random-twin.md)'s. Its criteria *decentralised* (no hub,
  links per agent) and *resilient* (one piece, 2-core, both cuts) are the
  other six marks
  ([What makes a world interesting](what-makes-a-world-interesting.md#the-six-criteria)).

## Where it is used

- [A world that cannot break](a-world-that-cannot-break.md): B2's rules were
  built to meet it.
- [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md):
  no world there held.
- [Healthy and rich](healthy-and-rich.md): every world read against it; on
  17 October 2026, 99 of 624 met all seven marks.
- [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md),
  [The candidates, played on](the-candidates-played-on.md) and
  [Room for the steadiest world](room-for-the-steadiest-world.md): theses
  that ask for every mark.
- [What makes a world interesting](what-makes-a-world-interesting.md) and
  [A screen of random worlds](a-screen-of-random-worlds.md): criteria built
  on six of the marks.
- [What would count as open-ended evolution](open-ended-test.md): every mark,
  at every 10,000 iterations.

## Where to go next

- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers):
  where the bar was set.
- [The random twin: is a world a space?](random-twin.md): the test the
  dimension mark needs beside it.
- [The diversity score](diversity-score.md): the other half of a healthy,
  rich world.

<!-- turns -->
---

← [What makes a world interesting](what-makes-a-world-interesting.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [The diversity score](diversity-score.md) →

**Also about:** [The shape and health of the network](../README.md#the-shape-and-health-of-the-network).

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [The estate of the dead](estate.md) · [Stars and hubs](stars-and-hubs.md) · [The random twin: is a world a space?](random-twin.md) · [The diversity score](diversity-score.md)
<!-- /turns -->
