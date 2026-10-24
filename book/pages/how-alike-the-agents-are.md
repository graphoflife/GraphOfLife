# How alike the agents are

> [!question] Questions of this page
> - How many kinds of [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") does a world hold — counted by name, by the numbers
>   in their [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."), or by what they do?
> - What makes the agents of a world alike, and what keeps them apart?
> - Does the variety last, or is it churn?
> - What would it take for many kinds to evolve, and to stay?

## In short

Evolution can only choose among differences. The notebook has counted a
world's agents in three ways. It counts the names of their brains, the numbers
in them, and what they do. The counts fall by orders of magnitude, because
most of what makes two brains two changes nothing they do. [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s agents act
nearly alike. [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s differ as widely as random brains, in nearly every variant
of B2. The one exception: where every unused link lapses, they become as alike
as B1's. And B2's kinds are renewed, but they replace one another instead of
piling up.

## Why variety matters

A breeder can only choose among the plants in the field. If every plant is
the same, there is nothing to choose, and nothing changes. Evolution is in the
same position: selection picks among differences, and without differences it
has nothing to pick.

[Open-ended evolution](open-ended-test.md) asks for more than differences at
one moment. It asks for new kinds that keep appearing, and stay beside the old
ones ([What this notebook is about](what-this-notebook-is-about.md)). So two
questions matter: how many kinds a world holds now, and whether they pile up.

Several pages of this notebook have counted the variety of a world, each in
its own way. This page puts their counts side by side. It adds one measure
taken in every kind of world the notebook has made. And it asks what the
counts say about open-ended evolution.

## Three ways to count

Two agents can differ in three ways, and each gives a different count:

- **by name.** Every brain carries a [genotype](genotype.md): a number it gets
  when it is made, and a new one at every change
  ([How a brain changes](mutation.md)). Two agents with one genotype have the
  same brain;
- **by the numbers.** A brain is a set of weights. How far apart are the
  weights of two brains?
- **by what they do.** Put two agents in the same situation, and see whether
  they stake differently. Agents that act alike are one kind of behaviour.
  The kinds are counted by how many agents carry each
  ([Effective numbers](effective-number.md)).

| what is counted | how | B1 (10,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")) | B2 (10,000 tokens) | pages |
|---|---|---|---|---|
| **names** | genotypes: a brain gets a new one at every change | 0.65 per agent | 0.46 per agent | [Genotypes and lineages](genotypes-and-lineages.md), [How even is a world?](where-do-the-tokens-go.md#how-even-is-a-world) |
| **numbers** | how far apart two brains' weights are | two agents of one world as far apart as two random brains | — | [What the brains are like](what-the-brains-are-like.md) |
| **behaviour** | kinds of agent that act measurably differently in the same situations, counted by their carriers | 3.2 kinds of staking, 9 over every decision | 17.5 kinds of staking, 54 over every decision | [How different are the agents?](how-different-are-the-agents.md) |

The counts fall by orders of magnitude from the first to the last. A B1 world
of 1,500 agents holds about a thousand genotypes, and a handful of kinds of
behaviour. A B2 world of 3,400 agents holds about 1,600 genotypes, and a few
dozen kinds. Most of what makes two genotypes two changes nothing they do.

A fourth count, the [diversity score](diversity-score.md), asks how widely a
world's agents spread in what they hold: their tokens, links, ages and
genotypes. The screen of random worlds uses it, beside the kinds of staking,
to judge whether a world is diverse
([What makes a world interesting](what-makes-a-world-interesting.md)).

## What the pages found

**In B1, the agents have many names, but act alike.**

- A genotype lasts a couple of [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). What lasts long enough for
  selection is a line, a family of genotypes. A line can take a world over in
  a few hundred iterations
  ([Genotypes and lineages](genotypes-and-lineages.md),
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)).
- B1 keeps five genotypes for every eight agents, in every world and at every
  size. That is a balance of making and copying. It moves with the rate of
  change and with nothing else
  ([How even is a world?](where-do-the-tokens-go.md#how-even-is-a-world),
  [How fast should brains change?](how-fast-should-brains-change.md)).
- Two brains of one world are as far apart in their numbers as two brains
  drawn at random. Nearly all of a brain is neutral: free to drift, because
  nothing depends on it
  ([What the brains are like](what-the-brains-are-like.md)).
- Selection set the same constants in every world
  ([How the game is played](how-the-game-is-played.md),
  [What the brains are like](what-the-brains-are-like.md)). They are: no
  children, a spread stake, and every token revolutionary.
- Without evolution, worlds started from random brains end in very different
  places. With it, they all end alike
  ([Do the brains matter?](do-the-brains-matter.md)).
- In the same situation, the agents act nearly alike: seven times more alike
  than random brains. Their largest kind of staking is the [even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly."), which
  the rules make of scores at or below zero
  ([How different are the agents?](how-different-are-the-agents.md)).

**In B2, the agents differ as widely as random brains, but steadily.** Nine
pairs in ten stake measurably differently. They share one habit, going all
in, and differ in when and on whom
([How different are the agents?](how-different-are-the-agents.md)).

**In both:**

- Where agents differ, they differ in how they answer a place, hardly ever in
  a fixed leaning
  ([How different are the agents?](how-different-are-the-agents.md)). Much of
  the place reaches them through the rules: a single token kept at home, or an
  even split over the [neighbours](the-world.md#the-network "An agent joined to another by a connection.").
- The groups tokens stay in are the network's regions. Only B2's agents keep
  a partner longer than chance, for a few [games](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")
  ([Flow modules: the groups tokens stay in](flow-modules.md)).

## Why B1's agents are alike and B2's are not

B1 and B2 differ in many rules at once, and
[Every setting](settings.md) lists them. So the difference between their
agents has more than one candidate cause:

- **Brains.** B1's brains have five hidden layers, and pass on a thousandth
  of a change in what they see
  ([What the brains are like](what-the-brains-are-like.md)). B2's have one,
  and see
  ([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)).
  A blind brain can differ from another only in its constants.
- **How the stake is made.** B1's brains spread their stake, and a spread
  stake changes little when the scores change. B2's go all in. An all-in
  stake goes to whichever [candidate](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.") scores highest, so a small difference
  between two brains flips the choice.
- **How brains change.** B1 changes every brain after every game. B2 changes
  a brain only when it is copied, so a brain keeps its quirks for as long as
  it lives.
- **How links lapse.** In B1, every link not used in a game lapses. In B2,
  only one whose two ends have at least two neighbours in common, so that
  they stay joined twice over
  ([Every setting](settings.md#prune_needs)).

The measures below, taken in every kind of world, tell the four apart. The
last is enough on its own.

## In every kind of world

[How different are the agents?](how-different-are-the-agents.md) measured B1
and B2. Its measures were taken again in three [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") of every condition the
[lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") has run, at the end of each run (58 worlds). And they were taken in both
seeds of every [pilot](the-lab.md#pilot-worlds "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") condition (69).

<!-- figure how_alike/lab -->
![How alike the agents are, in every lab condition](../figures/how_alike/lab.svg)

**How alike the agents are, in every lab condition.** Each dot is a world. Top: how many kinds of staking its agents fall into, counted by the agents in each (an effective number; logarithmic axis). Bottom: how far apart two agents' stakes are in the same situation and with the same random inputs, divided by how far apart two fresh founder brains are in the same situations: 1 means as different as brains drawn at random, less means more alike. Orange: B1 and its variants; blue: B2 and its variants; green: B2 with lapsing links. B2 without random inputs is left out of both: without noise every difference counts, and its kinds cannot be compared. B1: B1 at 10,000 tokens (Experiment 2); fixed: brains that never change (Experiment 7); B1 5k, B1 100k, B1 sizes: B1 at 5,000 (Experiment 12), 100,000 (Experiment 10) and 3,200 to 204,800 tokens (Experiment 8). B2: B2 at 10,000 tokens (Experiment 16); m .05 to m .8: the chance that a copied brain changes, 0.05 to 0.8 (B2's own is 0.2; Experiment 9); keep 1, keep ½: the last token kept, half the tokens kept; child: children of any size; no age: no old age (Experiment 15); B2 5k: B2 at 5,000 tokens, and with one of its rules taken away — no link: no links at birth, no val.: no valence, hand: handover allowed, no mend: no mending, lapse: any unused link lapses (Experiment 14).

> [!example]- How to make this figure
> **Runs.** Three seeds of every condition of the lab's experiments at the end of their runs (Experiments 7 to 16; B2 itself from Experiment 16 at 2,000 iterations and B1 from Experiment 2 at 3,000), and both seeds of every pilot condition at its checkpoint of 1,000 or 2,000 iterations.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/across_lab.json`, `B1.json`, `B2.json` (`research/oee/checks/diversity.py`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every situation and draw, and turn their outputs into decisions by the engine's rules.
> 4. Count the kinds of staking as in [How different are the agents?](how-different-are-the-agents.md); divide the agents' total variation by the founders'.
>
> **To make it again:** `python3 book_figures.py how_alike`.
<!-- /figure -->

**In nearly every variant of B2, the agents stay as different as random
brains.** The variants change a great deal:

- how often brains change;
- the last token, or half the tokens, kept at home;
- children of any size, half the tokens, no links at birth;
- no cap on links, and handover.

In each, two agents stake as differently as two fresh [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") brains in the
same situations: 0.88 to 1.07 times as far apart. Only without mending are
they somewhat more alike (0.69). Without old age they are as different too
(1.06), but there it says nothing. Most of the agents are the founders
themselves, who never die ([Healthy and rich](healthy-and-rich.md)).

What moves is how many kinds there are, and the rate at which brains change
moves it most.

<!-- figure how_alike/mutation -->
![The more brains change, the more kinds](../figures/how_alike/mutation.svg)

**The more brains change, the more kinds.** B2's worlds of Experiment 9 at 1,500 iterations, at five chances that a brain changes when it is copied (B2's own is 0.2). Each dot is a world; the line joins the medians. Left: kinds of staking, counted by their carriers; right: distinct genotypes per agent.

> [!example]- How to make this figure
> **Runs.** Three seeds of each of Experiment 9's five mutation rates, at iteration 1,500.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/across_lab.json` and `B2_1500.json` (`research/oee/checks/diversity.py`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every situation and draw, and turn their outputs into decisions by the engine's rules.
>
> **To make it again:** `python3 book_figures.py how_alike`.
<!-- /figure -->

Take the chance that a copied brain changes. At 0.05, B2's agents fall into
about 4.5 kinds of staking; at 0.8, into about 60. Meanwhile the genotypes per
agent go from 0.21 to 0.90. The kinds do not rise evenly with the rate: three
worlds per rate are few, and the count of a world swings with its seed. But
the two ends are an order of magnitude apart.

**One rule makes B2's agents as alike as B1's: every unused link lapses.**
Take B1's rule for links, and everything else of B2's. Then the agents fall
into one or two kinds of staking. In six of the seven worlds, nine agents in
ten are of the largest kind. Two agents of such a world stake a fifth as
differently as two random brains, or less. That is 0.22 in the lab's worlds,
and 0.12 and 0.13 in the pilots with and without gifts. It is as alike as the agents
of B1 and its variants (0.10 to 0.26).

And in six of these seven worlds, the agents do what B1's do. They spread
their stake over their candidates, in 95 to 100% of cases, where B2's agents
spread in at most 4%. Why? A link that no stake crosses lapses. So an agent
that goes all in on one neighbour keeps its other links only where they stake
on it, and spreading keeps them all. The brains, and the way brains change,
are B2's in these worlds. So neither is needed for the likeness: the rule for
links is enough, through the stake it selects.

Three other rules make the agents more alike, though not nearly as much:

- links that hold a token: 0.66 and 0.72 of random brains' distance;
- brains that grow: 0.67;
- brains whose every unit holds a token: 0.47.

<!-- figure how_alike/pilots -->
![Kinds of staking in the pilot worlds](../figures/how_alike/pilots.svg)

**Kinds of staking in the pilot worlds.** Each dot is a pilot world at its checkpoint of 1,000 iterations (batches A, N and S) or 2,000 (batches P and Q), the bar the median; orange: B1's rules underneath, blue: B2's. The pages that ran each batch say what its conditions change. Batch S: lottery, genotype, kin, family, fam. pool — contests by lottery, sides pooled by genotype, an eye for kin, an eye for a family tag, sides pooled by family (all without revolutions); B2, norev, gifts, g.+lapse, lapse, by tok. — B2 itself, without revolutions, with gifts, with gifts and every unused link lapsing, with every unused link lapsing, estates shared by wealth; B1, B1norev — B1, and B1 without revolutions.

> [!example]- How to make this figure
> **Runs.** Both seeds of every pilot condition that has a checkpoint there.
>
> **Data.** From `GraphOfLifeRuns/.pilots/diversity/across_pilots.json` (`research/oee/checks/diversity.py`).
>
> 1. Load the world and play one game, so its messages are fresh (`research/oee/checks/diversity.py`).
> 2. Take what 200 living agents read, and draw their random input rows 8 times, the same draws for every brain.
> 3. Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every situation and draw, and turn their outputs into decisions by the engine's rules.
>
> **To make it again:** `python3 book_figures.py how_alike`.
<!-- /figure -->

## Does the variety last?

Variety at one moment is not open-ended evolution. Think of a shop whose
shelves always hold a hundred products, because every new one pushes an old
one out. Its range never grows. A world that keeps making new kinds while it
loses old ones is like that shop. It holds as many kinds at every moment, and
no more.

What the notebook has measured over time:

- **Names churn fastest.** A genotype lasts a couple of iterations
  ([Genotypes and lineages](genotypes-and-lineages.md)).
- **B2's kinds of behaviour turn over in hundreds of iterations.** Of the
  kinds a sample can find, two thirds are gone 500 iterations later, and new
  ones have taken their place. A core of about a quarter lasts 5,000
  iterations ([How different are the agents?](how-different-are-the-agents.md)).
- **B1's few kinds last.** Of the agents of one moment, 0.82 have a kind still
  found 5,000 iterations later, against 0.90 for the same moment sampled
  twice. One of the two long worlds settled on a single kind of staking for
  good.
- **Better brains do keep coming in B2.** The brains of 2,000 iterations beat
  those of 500 over an agent's [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.")
  ([Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on)).
  But they come more and more slowly, towards a plateau from about 5,000
  iterations on. Whether the plateau is the end is still open for B2. Two
  worlds with brains mixed at birth, played to 50,000 iterations, climbed for
  20,000 to 25,000, and then wandered
  ([Every era against three](every-era-against-three.md)).

So B2's variety is renewed, and some of the renewal is improvement. What is
missing is accumulation: new kinds replace old ones instead of joining them.

## What it would take

For many kinds to live side by side, a kind needs a reason to stay: somewhere
it does better than the others. Then a new kind is added to the old ones,
rather than taking their place. The notebook's evidence points four ways, and
none of them needs a tuned number:

1. **Places that differ in what pays.** B2's kinds already sit at different
   [degrees](degree.md "The number of an agent's connections: how many neighbours it has.") in the network
   ([How different are the agents?](how-different-are-the-agents.md)). That
   fits niches, but it fits families that live together too. To tell them
   apart, move a kind to another place, and see whether it does worse there.
2. **Payoffs that depend on how common a kind is**, so that a rare kind has an
   edge. Nothing in the game does this now. Where payoffs do depend on
   numbers, they favour the common. Alliances, where a convention pays most
   when shared, made one kind win ([Alliances](another-game.md#alliances)).
   And a raider written by hand takes hold only in a crowd
   ([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)).
3. **Heredity of whole parts.** Mixing brains at birth gave some of the
   largest steps of evolution of any pilot, in one of two worlds
   ([Brains mixed at birth](mixed-brains-costly-units.md#brains-mixed-at-birth)).
   A child's brain is mixed from a parent and a mate. So a unit that works can
   spread without the rest of the brain it arose in.
4. **Groups that hold together.** None of these worlds makes any beyond the
   regions of its network
   ([Flow modules: the groups tokens stay in](flow-modules.md),
   [Who helps whom](who-helps-whom.md)).

And one thing to avoid: a rule that leaves a single way of acting as the way
to keep going. Lapsing links do that to the stake: spread it, or lose your
links. The kinds of staking then collapse to one or two, as above. A rule that
open-ended evolution can use leaves more than one way to live.

And one measure to take before any of them: play each kind against the kinds
that replaced it. If new kinds beat old ones, the turnover is selection. If
they only differ, it is drift: variety without direction.

## What we learned

- **Kinds of behaviour are far fewer than names.** A B1 world holds about a
  thousand genotypes but a handful of kinds of staking. Most of what makes two
  brains two changes nothing they do.
- **B1's agents act alike; B2's do not.** B2's agents differ as widely as
  random brains, in nearly every variant of its rules.
- **The rule for links decides it.** Where every unused link lapses, B2's
  agents spread their stakes and become as alike as B1's, with B2's brains.
- **The rate of change sets how many kinds there are:** about 4.5 kinds of
  staking at 0.05, and about 60 at 0.8.
- **Variety is renewed, not accumulated.** B2's kinds turn over in hundreds of
  iterations, and the new replace the old. For kinds to pile up, a kind needs
  somewhere it does better than the others.

## Where to go next

- [How different are the agents?](how-different-are-the-agents.md): the
  measure of behaviour, in B1 and B2.
- [What the brains are like](what-the-brains-are-like.md): what evolved brains
  compute, and why B1's barely see.
- [Genotypes and lineages](genotypes-and-lineages.md): how names and lines are
  born, spread and die.
- [The diversity score](diversity-score.md): how widely a world's agents
  spread in what they hold.

## How this was made

The three figures of this page are drawn with
`python3 book_figures.py how_alike`, from the measures of
`research/oee/checks/diversity.py`. Each says, folded beneath it, which
worlds it reads and how. The counts use
[Effective numbers](effective-number.md) and
[Splitting variation](variance-split.md).
[Genotypes, families and the common ancestor](genotype.md#brain-diversity)
says how genotypes are counted.

<!-- turns -->
---

[Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are) · [Genotypes and lineages](genotypes-and-lineages.md) →

> [!info]- Pages that link here (1)
> [How different are the agents?](how-different-are-the-agents.md)
<!-- /turns -->
