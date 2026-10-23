# The diversity score

## In short

Evolution needs differences to choose from. The **diversity score** says, in
one number from 0 to 1, how widely the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") of a world differ in their
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), their links, their ages and their [genotypes](genotype.md). Each of
the four is counted without drawing any line, then ranked among a fixed set
of 653 worlds; the score is the mean of the four ranks. How differently the
agents *decide* is a separate measure: the
[effective kinds of staking](how-different-are-the-agents.md#kinds-of-behaviour).

## Four spreads

**Tokens, links and ages.** Sort the agents into doubling classes: 1, 2–3,
4–7, 8–15, and so on (for ages, the age plus one, so that a newborn counts).
The **spread** is the [effective number](effective-number.md) of classes the
agents fill: 2 to the power of the [entropy](entropy-and-evenness.md), in
bits, of the classes' shares. It is 1 when every agent is in one class, and
*k* when the agents are spread evenly over *k* classes. Agents holding one to
three tokens fill at most two classes.

**Genotypes.** Here the count is the effective number of genotypes,
1 / Σ *p*², where *p* is the share of the agents carrying each genotype,
divided by the number of agents. It is 1 when every agent has a genotype of
its own, and near 0 when one genotype is everywhere.

## From spreads to a score

The four spreads are on different scales, so each is turned into a
percentile: the share of the reference worlds whose agents spread less
widely. The score is the mean of the four percentiles, 1 for a world at the
top of all four and 0 for one at the bottom of all four.

The reference is frozen: the 653 worlds the notebook had read on 17 October
2026 (`book/results/diversity_reference.json`, made by
`research/atlas/profile.py --freeze`). A new world gets the percentile its
spreads would have among them, so no score moves as worlds are added.

So the score is relative: it places a world among the worlds this notebook
has made, not on an absolute scale. And it does not say how evenly the
tokens are shared; that is the [Gini coefficient](gini-coefficient.md).

## Examples

From [Healthy and rich](healthy-and-rich.md):

- **[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds** spread widely, their ages over 5.6 to 6.9 classes and their
  tokens over 3.5 to 4.1, and score 0.67 to 0.75.
- **[B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s worlds** spread less: three agents in four hold three tokens or
  fewer, and the ages fill about 4 classes, since every agent dies at age 100.
- **B2's worlds without old age** (Experiment 15) score among the lowest of
  all: nearly every agent is of one age, the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' generation that never
  dies, and the ages fill 1.3 to 2.3 classes.
- **A score can fall.** A world that mixed [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") at birth and left the
  [estates](estate.md "The tokens a dead agent leaves, and who gets them.") of the dead to the poorest scored 0.57 at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") 1,000 and,
  played on, 0.28 at iteration 10,000.

## The line of B2

[What makes a world interesting](what-makes-a-world-interesting.md#the-lines-we-compare-with)
compares a world with the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of B2's ten worlds of Experiment 16, read
the same way at the same iteration. A world is *diverse* when it is above
both lines:

| B2's median world | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score | 0.53 | 0.51 |
| effective kinds of staking | 19.7 | 16.5 |

Other pages draw the line at other moments: 0.54 for B2's ten worlds played
on and read at iteration 3,000
([Room for the steadiest world](room-for-the-steadiest-world.md)).

## Decisions: the effective kinds of staking

The score sees what agents hold and what they are, not what they do, and two
genotypes may act exactly alike: a B2 world of 3,400 agents holds about 950
effective genotypes but about 17 kinds of staking.
[How different are the agents?](how-different-are-the-agents.md) puts the
brains of 400 agents into the same 200 situations, with the same random
inputs. Two brains act differently when their stakes are further apart than
95% of brains are from themselves between two halves of the random draws;
brains that do not are one kind. The kinds are counted by their carriers, as
an effective number: 3.2 in B1's median world at its end, 17.5 in B2's at
iteration 2,000.

For its lines, the screen of random worlds reads the kinds with no [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")
played, and averages each moment with the one 500 iterations before, since
the kinds swing from moment to moment.

## Where it is used

- [Healthy and rich](healthy-and-rich.md#two-halves): where the score was
  made.
- [What makes a world interesting](what-makes-a-world-interesting.md) and
  [A screen of random worlds](a-screen-of-random-worlds.md): the criterion
  *diverse*.
- [The candidates, played on](the-candidates-played-on.md) and
  [Five more seeds for the rules that held](the-candidates-played-on.md#five-more-seeds-for-the-rules-that-held):
  a world must score above B2's median world to be played on, and again to
  hold.
- [Room for the steadiest world](room-for-the-steadiest-world.md): what each
  rule cost of the four parts.
- [Births as rich as the neighbours](room-for-the-steadiest-world.md#births-as-rich-as-the-neighbours),
  [The search's best world, in new seeds](the-candidates-played-on.md#the-searchs-best-world-in-new-seeds),
  [Who inherits?](who-inherits.md) and [Rules in company](rules-in-company.md):
  worlds scored by it.

## Where to go next

- [How different are the agents?](how-different-are-the-agents.md): the kinds
  of staking, measured.
- [Effective numbers: how many kinds there really are](effective-number.md):
  the count inside both measures.
- [The health bar](health-bar.md): the other half of the picture.

<!-- turns -->
---

← [The health bar](health-bar.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (2)
> [Glossary](glossary.md) · [The health bar](health-bar.md)
<!-- /turns -->
