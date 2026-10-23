# B1 and B2: the two baselines

## In short

A **baseline** is a stated set of rules: every [setting](settings.md) of a
[world](the-world.md), written down in one file. An
[experiment](the-lab.md#runs-and-experiments) changes one thing of its
baseline, so that whatever differs comes from that change. **B1** is the
algorithm exactly as a new run is offered it: the original rules. **B2** is
B1 with rules that keep the network from breaking and agents from staking or
breeding themselves to death, and with smaller [brains](the-brain.md) that
see.

## What a baseline is

To learn what one rule does, compare worlds that differ in that rule and in
nothing else. So every experiment names a baseline and keeps all of it
except what a condition changes. A run's name
starts with its baseline: `B2-5000-d6b607-s001` is B2 at 5,000 tokens with
one change (deaths not mended), [seed](random-numbers.md) 1. The baselines
are kept as plans, `book/experiments/B1.json` and `B2.json`.

## B1: the rules as they were built

B1 is what the app has offered for a new simulation since 2 October 2026.
Its brain has five hidden layers of 50, 45, 40, 35 and 30 units. A
[message](messages.md) holds 30 numbers, and each phase begins with an extra
look in which agents only write messages.
[Coalitions can take nodes](revolution.md), and a parent may hand
connections to its child. A brain [changes](mutation.md) with probability
0.2 at birth and again after every [game](one-iteration.md#phase-2-the-game),
wherever it sits. Every rule added
later for open-ended evolution is off. A plan that names no size gets 10,000
tokens: 100 founders on [the starting ring](starting-ring.md).

B1's brains barely see. Each of the five layers passes on about a quarter of
a change, so a change in what a founder sees reaches what it does at a
[gain](vanishing-signal.md) of 0.0006.

## B2: a world that cannot break

B2 was made on 12 October 2026 from [pilot worlds](pilot-worlds.md). It
differs from B1 in these settings and in no other
([Every setting](settings.md) explains each):

<!-- table catalogue/baselines -->
| setting | B1 | B2 | what it is |
|---|---|---|---|
| [`hidden_layers`](settings.md#hidden_layers) | 50, 45, 40, 35, 30 | 16 | The widths of the brain's hidden layers, in order from input to output. |
| [`message_amount`](settings.md#message_amount) | 30 | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
| [`message_prepass`](settings.md#message_prepass) | on | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
| [`allow_handover`](settings.md#allow_handover) | on | off | Whether a parent may move some of its own connections to its newborn child. |
| [`mutate_on_replication`](settings.md#mutate_on_replication) | off | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
| [`incumbency`](settings.md#incumbency) | 1.0 | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
| [`birth_links`](settings.md#birth_links) | 0 | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
| [`mend_on_death`](settings.md#mend_on_death) | off | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
| [`prune_needs`](settings.md#prune_needs) | 0 | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
| [`valence`](settings.md#valence) | 0 | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
| [`keep_home`](settings.md#keep_home) | 0.0 | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
| [`max_age`](settings.md#max_age) | 0 | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
| [`min_child`](settings.md#min_child) | 1 | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
<!-- /table -->

In words, the changes fall into four groups:

- **Brains that see.** One hidden layer of 16 units: a founder's gain is
  about 0.2. A brain changes only when it is copied, into a newborn or into
  a node its owner has won, as genomes do.
- **Shorter messages.** 8 numbers and no extra look: half the work of B1's,
  for worlds that run thousands of iterations.
- **A network that holds.** A newborn is joined to two of its parent's
  neighbours, drawn at random; a death is
  [mended in a ring](a-world-that-cannot-break.md#mending-a-death); an
  unused link lapses only where its ends share at least two neighbours; a
  brain may choose at most 8 links; no handover.
- **Death by age, and a home advantage.** 30% of an agent's tokens stay at
  home, so nobody stakes itself to death; a child gets at least 10 tokens;
  every agent dies at age 100; a stake on one's own node counts 1.5 times
  ([Who dies, and why](who-dies-and-why.md)).

## An example: ten worlds of each at 5,000 tokens

In B1 (the baseline of Experiment 13), the [cull](one-iteration.md#cleanup)
of the [median](median-and-quantiles.md) world removed 1.9% of its agents in
every iteration from
iteration 500 on, and only 51% of its agents sat in the
[2-core](core-trees-leaves.md).

In B2 (Experiment 14), not one agent of the ten worlds was cut off from
iteration 200 to 1,000, and every agent sat in the 2-core. Every agent lives
exactly 100 iterations, so a world keeps its size only if one agent in a
hundred has a child each iteration; births ran at 1.00 to 1.05 per hundred
agents and iteration.

B2 pays in wealth: in the median of Experiment 15's B2 worlds, at 10,000
tokens, 52% of the agents held a single token. In the words of
[Healthy and rich](healthy-and-rich.md), B1's worlds are rich but fragile,
B2's healthy but poor.

## Where it is used

- **B1**, the world the notebook first set out to understand, is the start
  of Experiments 1 to 8, 10, 12, 13 and 37; [Thirty worlds](thirty-worlds.md)
  introduces its study.
- **B2**, the ground for open-ended evolution, is the start of Experiments 9
  and 14 to 21 and most later ones;
  [Why evolution stops](why-evolution-stops.md) asks the question it was
  built for. [What would count as open-ended evolution](open-ended-test.md)
  asks a new world to be as diverse as B2's median world.
- [Every simulation: what was run, and what was measured](every-simulation.md)
  gives the baseline of every experiment and pilot batch.

## Where to go next

- [A world that cannot break](a-world-that-cannot-break.md)
- [The lab, the engine and pilots](the-lab.md)
- [Eras and contests: does a world keep getting better?](eras-and-contests.md)

<!-- turns -->
---

[The baselines B1 and B2](../README.md#the-baselines-b1-and-b2) · [What the baseline world is](what-the-baseline-world-is.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works).

> [!info]- Pages that link here (13)
> [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [The world: tokens, agents and a network](the-world.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Every setting](settings.md) · [The share function f(a, b)](share-function.md) · [What a brain sees](brain-inputs.md) · [What a brain says](brain-outputs.md) · [The estate of the dead](estate.md) · [The even split](even-split.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
