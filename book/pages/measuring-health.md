# Measuring health: criteria, the health bar and diversity

> [!question] Questions of this page
> - What do we look for in a world before we ask how it evolves?
> - How is health measured now, and what is it used for since 11 November 2026?
> - What were the seven marks of the health bar?
> - How much do a world's [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") differ, in one number?

## In short

Health says whether a world could hold long evolution at all: whether it lives,
holds together, has room, and has agents that differ. The definition in force
is the author's of 9 November 2026; since 11 November it is only a filter. Its
earlier versions are kept below.

Its sections:

- [From 11 November 2026: health as a filter](#from-11-november-2026-health-as-a-filter)
- [The definition in force: health of 9 November 2026](#the-definition-in-force-health-of-9-november-2026)
- [Earlier versions](#earlier-versions)
- [The health bar](#the-health-bar)
- [The diversity score](#the-diversity-score)

## From 11 November 2026: health as a filter

From 11 November 2026, health is frozen at the version of 9 November and used
only as a filter. The filter is four rules of the funnel
(`research/screen/funnel.py`), and a world is left out at the first it meets:
- **died:** it did not live to the end of its run;
- **a star:** one agent is linked to more than half the world, on average.
  That is a far stricter line than the star of
  [Stars and hubs](measuring-networks.md#stars-and-hubs), a hub joined to a
  tenth of the world (`STAR` in `book_chapters/common.py`);
- **one holding:** 80% or more of the agents hold the same number of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.");
- **standing still:** its line of descent hardly moves from era to era, and
  two agents of one era hardly act differently, both less than 0.05 above the
  noise.

The funnel adds a fifth rule, of evolution rather than of health: **no
heredity**, children acting no more like their parents than strangers do (a
heredity index below 0.1 in both kept worlds read). A world that none of the
five rules out is a candidate world
([The worlds ruled out](candidates-for-open-ended-evolution.md#the-worlds-ruled-out)).
The worlds left are ranked by evolution, against null models, and not by
health ([What this notebook is about](what-this-notebook-is-about.md)). So a
world is chosen for what the notebook asks, whether it keeps evolving, and not
for how healthy it looks.

## The definition in force: health of 9 November 2026

Health of 9 November 2026 scores fifteen measures by the share of round 3's
living worlds a world beats, and combines six criteria in three orders
([Every measure, a score from 0 to 1](which-worlds-are-healthy.md#every-measure-a-score-from-0-to-1)).

### Two numbers reported, not judged

The author asked for one more number for health: whether a world, on the
largest scale, looks like a space of three dimensions, really spread out,
while very different behaviours survive in it. Two numbers try to say this.
The judge reports them for every world of round 3, and their lines will be
set together afterwards.

<!-- table criteria/reported -->
| number | what it says |
|---|---|
| half-cut exponent | the links crossing the cheapest cut into halves found, as a power of the number of agents: 2/3 for a body of three dimensions, near 0 for a bottleneck, near 1 for a folded small world |
| half-cut exponent of the random twin | the same for the world's random twin, which is no space at all |
| same-position share | of two agents put in the same situation, how often they stake measurably differently |
| effective kinds of staking | how many kinds of staking the agents fall into, counted by their shares |
| doctrine alike | the share of agents making the more common choice between spreading their stake and staking it all on one place |
| having a child | the share of agents that have a child in an iteration |
| agents | how many agents the world holds |
<!-- /table -->

#### The half-cut exponent

Cut a world into two halves of equal size, as cheaply as you can, and count
the links that cross the cut.
- In a body of three dimensions the cheapest cut is a flat slice. A cube of
  N agents has halves that meet on a square of N^(2/3) agents.
- A world that hangs together at one bottleneck can be halved through a few
  links, whatever its size.
- A folded small world, where everyone is a few steps from everyone, needs a
  cut that grows like N itself.

So the exponent of the cut, ln C / ln N for C crossing links among N agents,
is about 2/3 for a space, near 0 for a bottleneck and near 1 for a folded
world.

Nobody can find the cheapest cut of a network of thousands of agents exactly.
The judge uses a standard shortcut. Think of the network as a drum: its
slowest ways to vibrate move one region up while another goes down, and the
line between them runs where few links join the two. Each of the ten slowest
ways (the eigenvectors of the network's Laplacian with the smallest
eigenvalues) splits the agents at its median, and the cheapest of the ten
splits counts. One alone is not enough: the slowest ways vibrate at nearly the
same rate, and in one world of round 3 the slowest cut 121 links where the
third cut 62. On a box of 16 by 8 by 8 agents the judge finds the slice of 64
links across its long side, and on real worlds the same ten cuts as an exact
calculation. The world's random twin is cut the same way, for comparison: a
twin is no space at all.

#### The same-position share

Put two agents in exactly the same situation: the same [neighbours](the-world.md#the-network "An agent joined to another by a connection."), the same
tokens, the same [messages](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read."). Do they stake their tokens differently? The judge
takes 400 of a world's agents and 200 situations from the world, plays every
one of those agents' [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") in each, and counts the pairs of agents whose
stakes differ by more than one brain differs from itself between two tries.
In a world where every agent would act alike in the same place the share is
near 0; in a world of many kinds of behaviour it is near 1. It is the measure
of the [effective kinds of staking](#decisions-the-effective-kinds-of-staking),
read at every world kept.

## Earlier versions

| version | in use | what it asked | what it judged |
|---|---|---|---|
| [The health bar](#the-health-bar) | from 12 October 2026 | seven marks of a world's network at one moment, each met or missed | [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") and the worlds built from it; part 4 of the [open-ended test](measuring-evolution.md#what-would-count-as-open-ended-evolution) still asks for every mark |
| [The diversity score](#the-diversity-score) | from 16 October 2026 | four spreads of a world's agents in one number, ranked among frozen worlds | the search for a healthy, rich world; part 4 of the open-ended test |
| [The criteria of rounds 1 and 2](rounds-1-and-2-of-the-screen.md#the-criteria-of-rounds-1-and-2) | 23 October 2026 | six criteria read at iteration 2,000, each a yes or a no | round 1 of the screen (Experiment 38) |
| [The six criteria of round 3](rounds-1-and-2-of-the-screen.md#what-makes-a-world-interesting) | 24 October to 7 November 2026 | the same six, averaged over iterations 100 to 3,000, with looser lines | round 3, as first judged (Experiments 40 to 42) |
| a first order | 7 to 9 November 2026 | each line read as the share of what it asks, up to the line | round 3; the author found it still a yes or a no past the lines |
| [the definition in force](#the-definition-in-force-health-of-9-november-2026) | from 9 November 2026 | fifteen measures, each scored against round 3's living worlds; six criteria; three orders | round 3's orders and shortlist (Experiment 43); from 11 November, a filter only |

## The health bar

A world can only evolve for a long time if its network holds together. The
**health bar** says what that means, in seven marks that can be checked on a
world at one moment: one piece, made mostly of [loops](measuring-networks.md#loops), with no
agent or link whose loss cuts off much of it, no hub, a few links per agent,
and a dimension near three. A world that meets all seven has a healthy
network. The bar says nothing about the agents themselves: how much they
differ is the [diversity score](measuring-health.md#the-diversity-score).

### The seven marks

The bar was set in
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers).

| mark | what is measured | the line |
|---|---|---|
| one piece | how many pieces the network falls into | one |
| 2-core | the share of agents in the [2-core](measuring-networks.md#core-trees-and-leaves): on a loop, or between loops | at least 80% |
| agent cut | the largest share of the world that one agent's loss would cut off | at most 2% |
| link cut | the same for one link: the [cut risk](measuring-networks.md#cut-risk) | at most 1% |
| no hub | the share of the other agents that the best-connected agent is linked to | at most 5% |
| links per agent | the mean [degree](measuring-networks.md#an-agents-connections-degree-and-clustering) | 3 to 12 |
| dimension | how fast balls grow ([Dimension and curvature from ball growth](measuring-space.md#three-ways-to-measure-a-networks-dimension)) | 2.3 to 4 |

### Why each mark is there

- **One piece.** The [cleanup](one-iteration.md#cleanup) after every phase
  keeps only the largest piece of the network; every agent cut off from it
  dies.
- **2-core.** Peel away every agent with fewer than two links, again and
  again; what is left is the 2-core. An agent outside it hangs on a tree, and
  a single cut can lose a tree whole.
- **The two cuts** count single points of failure. A [bridge](measuring-networks.md#bridges-cut-risk-loops-and-the-core-how-a-network-can-be-cut)
  with a tenth of the world behind it is a tenth of the world that dies the
  moment no token crosses it in a game, and a death can split a world in the
  same way.
- **No hub.** A world that hangs on one agent changes all at once when that
  agent dies.
- **Links per agent.** A tree has fewer than two links per agent; a complete
  network, where everyone is linked to everyone, has as many as there are
  other agents. Three to twelve is neither.
- **Dimension.** The notebook hopes for a world that grows into a space like
  ours ([Physical inspiration](physical-inspiration.md)). The author's "almost
  three" is the middle of the range.

### How it is read

`research/atlas/profile.py` reads a world from its saved state at one moment,
without its brains. One depth-first search finds the pieces and the worst cut
by one agent and by one link; peeling finds the 2-core; the dimension is the
lab's own measure: how the number of agents within 1 to 5 steps
grows, around 24 agents spread through the world. The notebook's tables give
the result as "marks (of 7)".
Earlier experiments read the same lines from a run's recorded statistics, as
medians over many iterations.

### Two examples

- **[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") fails it.** A [pilot](runs-and-records.md#pilot-worlds) of B1 at 5,000 tokens ([seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1,
  iterations 200 to 999) had 59% of its agents in the 2-core, a link whose
  loss could cut off 8% of the world, and a best-connected agent linked to 8%
  of the others. Its 3.3 links per agent and its dimension of 2.7 passed.
- **B2 nearly passes it.** In Experiment 14, ten worlds of B2 at 5,000
  tokens, every agent sat in the 2-core; read every tenth iteration, no
  agent's loss could cut off more than 1.2% of a world; the best-connected
  agent was linked to 2.1 to 3.1% of the others; agents had 5.0 to 5.8 links.
  Nine worlds met every line. The tenth missed only the dimension, 4.19
  against the bar's 4.

### Read it with care

- **Two marks follow a world's size.** In networks like B2's, the hub mark
  fails small worlds, where a best-connected agent with 12 to 47 links is
  already more than 5% of the world, and the dimension mark fails large ones.
  In the mixed worlds of Experiment 17, the hub mark failed at 96% of the
  moments below 300 agents, while worlds of 600 to 1,000 agents met every
  mark at 98% of their moments
  ([Mixed at birth: ten long worlds](building-b2-and-the-search-for-a-healthy-rich-world.md#the-marks-a-world-misses)).
- **The dimension mark cannot tell a space from a small world.** A B2 world
  of 600 to 1,500 agents passes it because it is small
  ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)). So the
  [screen of random worlds](a-screen-of-random-worlds.md) asks instead for a
  dimension of 2 to 4 and a typical distance at least 1.5 times its
  [random twin](measuring-space.md#the-random-twin-is-a-world-a-space)'s. Its criteria *decentralised* (no hub,
  links per agent) and *resilient* (one piece, 2-core, both cuts) are the
  other six marks
  ([What makes a world interesting](rounds-1-and-2-of-the-screen.md#the-six-criteria)).

### Where it is used (the health bar)

<!-- used the-health-bar -->
- [What this notebook is about](what-this-notebook-is-about.md#a-ladder-of-four-steps)
- [Measuring lives: births, deaths, ages and estates](measuring-lives.md#an-example-paying-for-births)
- [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md#where-degree-is-used)
- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md#what-folds-a-world)
- [Measuring space: dimension and the random twin](measuring-space.md#folded)
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers)
- [Other contests: a lottery, and alliances](another-game.md#networks-that-hold)
- [Rules without numbers](rules-without-numbers.md#in-short)

> [!info]- And in the Archive (5)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#stars) · [B1: lives and wealth](b1-lives-and-wealth.md#where-to-go-next) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md#worlds-that-cannot-break) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#two-halves) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md#three-kinds-of-rules)
<!-- /used -->

## The diversity score

Evolution needs differences to choose from. The **diversity score** says, in
one number from 0 to 1, how widely the agents of a world differ in their
tokens, their links, their ages and their [genotypes](measuring-evolution.md#genotypes-families-and-the-common-ancestor). Each of
the four is counted without drawing any line, then ranked among a fixed set
of 653 worlds; the score is the mean of the four ranks. How differently the
agents *decide* is a separate measure: the
[effective kinds of staking](b1-brains-lineages-and-messages.md#kinds-of-behaviour).

### Four spreads

**Tokens, links and ages.** Sort the agents into doubling classes: 1, 2–3,
4–7, 8–15, and so on (for ages, the age plus one, so that a newborn counts).
The **spread** is the [effective number](statistics.md#effective-numbers-how-many-kinds) of classes the
agents fill: 2 to the power of the [entropy](measuring-tokens.md#entropy-and-evenness), in
bits, of the classes' shares. It is 1 when every agent is in one class, and
*k* when the agents are spread evenly over *k* classes. Agents holding one to
three tokens fill at most two classes.

**Genotypes.** Here the count is the effective number of genotypes,
1 / Σ *p*², where *p* is the share of the agents carrying each genotype,
divided by the number of agents. It is 1 when every agent has a genotype of
its own, and near 0 when one genotype is everywhere.

### From spreads to a score

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
tokens are shared; that is the [Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-and-entropy).

### Examples

From [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich):

- **B1's worlds** spread widely, their ages over 5.6 to 6.9 classes and their
  tokens over 3.5 to 4.1, and score 0.67 to 0.75.
- **B2's worlds** spread less: three agents in four hold three tokens or
  fewer, and the ages fill about 4 classes, since every agent dies at age 100.
- **B2's worlds without old age** (Experiment 15) score among the lowest of
  all: nearly every agent is of one age, the [founders](rules-in-detail.md#the-starting-ring "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' generation that never
  dies, and the ages fill 1.3 to 2.3 classes.
- **A score can fall.** A world that mixed brains at birth and left the
  [estates](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") of the dead to the poorest scored 0.57 at iteration 1,000 and,
  played on, 0.28 at iteration 10,000.

### The line of B2

[What makes a world interesting](rounds-1-and-2-of-the-screen.md#the-lines-we-compare-with)
compares a world with the median of B2's ten worlds of Experiment 16, read
the same way at the same iteration. A world is *diverse* when it is above
both lines:

| B2's median world | at iteration 1,000 | at iteration 2,000 |
|---|---|---|
| diversity score | 0.53 | 0.51 |
| effective kinds of staking | 19.7 | 16.5 |

Other pages draw the line at other moments: 0.54 for B2's ten worlds played
on and read at iteration 3,000
([Room for the steadiest world](building-b2-and-the-search-for-a-healthy-rich-world.md#room-and-riches-for-the-steadiest-world)).

### Decisions: the effective kinds of staking

The score sees what agents hold and what they are, not what they do, and two
genotypes may act exactly alike: a B2 world of 3,400 agents holds about 950
effective genotypes but about 17 kinds of staking.
The method puts the
brains of 400 agents into the same 200 situations, with the same random
inputs, as [How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents) first did. Two brains act differently when their stakes are further apart than
95% of brains are from themselves between two halves of the random draws;
brains that do not are one kind. The kinds are counted by their carriers, as
an effective number: 3.2 in B1's median world at its end, 17.5 in B2's at
iteration 2,000.

For its lines, the screen of random worlds reads the kinds with no game
played, and averages each moment with the one 500 iterations before, since
the kinds swing from moment to moment.

### Where it is used (the diversity score)

<!-- used the-diversity-score -->
- [Measuring tokens: inequality and flow](measuring-tokens.md#the-effective-number)
- [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md#a-caution-about-names)
- [Every simulation: what was run, and what was measured](every-simulation.md#what-was-measured-on-what)
- [Statistics: the tools of the notebook](statistics.md#where-to-go-next)

> [!info]- And in the Archive (3)
> [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md#three-ways-to-count) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#two-halves) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md#nearer-the-authors-picture)
<!-- /used -->

## Where to go next

- [A screen of random worlds](a-screen-of-random-worlds.md):
  the rounds, and what each found.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  where the health bar comes from.
- [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich): how the
  diversity score was made, and every world scored by it.
- [What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution): the test a
  healthy world's long run will face.
- [The random twin: is a world a space?](measuring-space.md#the-random-twin-is-a-world-a-space): the test the
  dimension mark needs beside it.
- [Effective numbers: how many kinds there really are](statistics.md#effective-numbers-how-many-kinds):
  the count inside both measures.

## How our view changed

- **11 November 2026.** This page now opens with the definition in force, of 9
  November 2026, until then only on Which worlds are healthy?, and a table of
  the earlier versions. From this day health is frozen at that version and used
  only as a filter: worlds that are dead, a star, uniform or frozen are left
  out, and worlds are ranked by evolution. The earlier versions are kept as
  they were.
- **12 November 2026.** What makes a world interesting, the criteria of 24
  October to 7 November 2026 and those of rounds 1 and 2, moved to the Archive
  with their entries of this list
  ([Rounds 1 and 2 of the screen](rounds-1-and-2-of-the-screen.md#what-makes-a-world-interesting)).
  The two numbers reported, not judged, stay here, under the definition in
  force.

<!-- turns -->
---

← [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) · [Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [How runs are made, recorded and costed](runs-and-records.md) →

**Also about:** [Brains: what they are, and whether they matter](../README.md#brains-what-they-are-and-whether-they-matter).

> [!info]- Pages that link here (17)
> [What this notebook is about](what-this-notebook-is-about.md) · [How this notebook is written](how-this-notebook-is-written.md) · [Which worlds are healthy? Three orders](which-worlds-are-healthy.md) · [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md) · [Glossary](glossary.md) · [Measuring lives: births, deaths, ages and estates](measuring-lives.md) · [Measuring tokens: inequality and flow](measuring-tokens.md) · [Measuring a network: connections, hubs, distances and cuts](measuring-networks.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Measuring space: dimension and the random twin](measuring-space.md) · [Measuring evolution: genotypes, eras and the open-ended test](measuring-evolution.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [Statistics: the tools of the notebook](statistics.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md) · [Rounds 1 and 2 of the screen, and the criteria they were judged by](rounds-1-and-2-of-the-screen.md)
<!-- /turns -->
