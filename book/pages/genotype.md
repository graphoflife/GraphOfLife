# Genotypes, families and the common ancestor

## In short

Every [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") carries a whole number, its **genotype**, which a copy keeps and
a change replaces. The new number remembers the old one as its parent, so the
genotypes of a world form one family tree. On it the notebook counts
**families**, or lines of descent, and finds the **common ancestor** of the
living. A **Muller plot** draws the lines of descent as they grow and replace each
other. The notebook also counts how many different genotypes a world holds, its
**brain diversity**: in a settled [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world, five for every eight [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.").

## Genotypes and the family tree

Think of a song passed on by ear, and give every version of it a number. A
singer who sings it exactly passes on the same version, with the same number.
A singer who changes a note makes a new version: it gets a new number, and a
note saying which version it came from. Follow the notes back, and every
version leads to one of the first songs.

A brain's genotype works the same way. Two agents with the same genotype have
brains with exactly the same weights, because they hold copies of one brain.

### How genotypes are numbered

- The [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' brains get the genotypes 1, 2, …, *n*. In a B1 world of
  100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), that is 1 to 1,000.
- **Copying keeps the number.** A child's brain before it changes, and the
  brain a node takes on from the winner of the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), are copies: same
  weights, same genotype.
- **Changing gives a new number.** When a brain changes, it gets the next
  unused number, and remembers the number it had as its **parent genotype**.
  Numbers are never reused, and a parent always has a smaller number than its
  child.

A brain is offered a change only at set moments, and changes with
probability 0.2 each time ([How a brain changes](mutation.md)):

- **in both [baselines](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")**, a newborn's brain, just copied from its parent;
- **in B1**, also every surviving brain after every game, wherever it sits;
- **in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")**, instead, the copy a node takes on when another agent wins it
  ([`mutate_on_replication`](settings.md#mutate_on_replication)). A brain
  that stays where it is never changes.

B2 has that rule because in B1, 97% of new genotypes are made in brains that
are not being copied
([Variation in the living](questioning-the-mechanics.md#variation-in-the-living)).

An example. Agent A carries genotype 7. It has a child, B, whose brain is a
copy: genotype 7. The copy changes, and becomes genotype 1,204, the next
unused number, with parent 7. In the game, B wins the node of its [neighbour](the-world.md#the-network "An agent joined to another by a connection.")
C, and C's node takes on a copy of B's brain: genotype 1,204 again. Now two
agents carry 1,204. In B1, each copy may change after the game, into a new
number with parent 1,204. In B2, C's copy may change the moment it is made,
and A keeps genotype 7 for as long as it keeps its node.

### Genotypes are not agents

An agent keeps its id and its age for life, but its genotype can change. In
B1 it changes when its brain changes after a game. In both baselines it
changes when its node is won in the game, for then it holds a copy of the
winner's brain. In B2 only the second can happen: an agent that keeps its
node keeps its genotype.

Many agents can share a genotype, and a genotype can live on in many places
after the agent it first appeared in has died. Brains spread mostly by
winning nodes. In a settled B1 world of 100,000 tokens, about 8,100 nodes are
taken by a neighbour in each game. Only about 480 children are born in an
[iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")
([Two roads of inheritance](questioning-the-mechanics.md#two-roads-of-inheritance)).

Two different counts follow:

- the number of **agents** alive — `nodes` in the statistics;
- the number of **distinct genotypes** among them — `distinctBrains`
  ([Brain diversity](#brain-diversity)).

A settled B1 world of some 14,400 agents holds about 9,000 different
genotypes at once ([Genotypes and lineages](genotypes-and-lineages.md)).

### The family tree

Following parent genotypes back gives a **tree**: every genotype has one
parent, back to a founder's genotype, which has none. The frames record, for
every living agent, its genotype and that genotype's parent (`brain_ids` and
`parent_brain_ids`), so the tree can be rebuilt from a run. A genotype's
**birth** is the iteration of the first frame it appears in.

A few genotypes never appear in a frame. A frame is taken after the [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.")
that ends a phase, and a newborn whose brain changed may be cut off in that
cleanup. In B1 and B2 this breaks no line of the tree. A genotype never seen
either has no descendants, or is a founder's, which is a root anyway.

Some experiments mix a newborn's brain from its parent's and a mate's
([`mix_at_birth`](settings.md#mix_at_birth)). The mixture gets a new number
of its own, and the tree records only the parent's side. A mixture that
changes again at once never reaches a frame, so in those worlds the tree
rebuilt from the frames has gaps.

The notebook uses the tree in three ways:

- **[Families](#families)** — how many different genotypes of eight
  iterations earlier the living descend from.
- **[The common ancestor of the living](#the-common-ancestor-of-the-living)**
  — the newest genotype that all of them descend from, or nine in ten, or
  half.
- **Founder lines** — how many of the living descend from each founder,
  drawn as a [Muller plot](#muller-plots).

### Lifetimes of genotypes

A genotype is *alive* after a game if at least one living agent carries it.
Its lifetime is from the first game after which it is alive to the last,
counted as an agent's lifetime is ([Age and lifetime](age-and-lifetime.md)).
Genotypes are short-lived. In the B1 worlds of 100,000 tokens the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.")
genotype lasts two iterations. Of the agents that live through their first
game, 48% last at least ten
([Genotypes and lineages](genotypes-and-lineages.md)).

## Families

How many separate lines of descent does a world hold? A **family**, or line
of descent (a lineage), is one genotype together with every genotype
descended from it. Its members are the agents that carry them.

Counting genotypes gives too many lines: one mutation makes a new genotype,
but parent and child are still one family. The **families** count, recorded
as `cladesInWindow`, asks instead how many different **ancestors of eight
iterations ago** the living descend from.

### Climbing back eight iterations

A genotype is **born** in the iteration of the first frame it appears in. At
iteration *t*, let the **anchor** be *t* − 8. For every living agent, take its
genotype *g* and climb the family tree
([Genotypes and the family tree](#genotypes-and-the-family-tree)):

1. If *g* was born at or before the anchor, stop: *g* is this agent's family.
2. If *g* has no parent (a founder's genotype), stop: *g* is the family.
3. Otherwise move to *g*'s parent and go back to step 1.

The **number of families** is the number of different genotypes the climbs
stop at.

### An example of families

At iteration 100, the living carry genotypes 5,021, 5,022, 5,100 and 4,870.
Suppose 5,021 and 5,022 were born at iteration 96 and 97, both from 4,900,
born at iteration 95, whose parent 4,610 was born at 88. Genotype 5,100 was
born at 99 from 4,870, and 4,870 itself at 90. The anchor is 100 − 8 = 92.

- 5,021 → 4,900 (95, after the anchor 92) → 4,610 (88 ≤ 92): family **4,610**.
- 5,022 → 4,900 → 4,610: family **4,610**.
- 5,100 → 4,870 (born at 90 ≤ 92): family **4,870**.
- 4,870: born at 90 ≤ 92: family **4,870**.

Two families.

### Why eight iterations?

Short enough to be a count of *current* lines, long enough that a single
change does not count as a family of its own. In B1 every brain is offered a
change after every game, and takes it with probability 0.2. So in eight
iterations a line gathers on average 8 × 0.2 = 1.6 changes, and more where it
passes through a birth. The climb usually goes a few steps.

Other pages fix the ancestors at another moment, with the same idea:

- the genotypes alive 100 iterations earlier
  ([One world, many colours](one-world-many-colours.md));
- those alive at one chosen iteration
  ([Genotypes and lineages](genotypes-and-lineages.md));
- the founders ([Muller plots](#muller-plots)).

### Early in a run

Before iteration 8, "eight iterations ago" is before the world began. So the
climb ends at the founders, and the count is how many of them still have
descendants. In the median B1 world of 100,000 tokens, around iteration
8, the living descend from about 390 of the 1,000 founders. The worlds of
10,000 tokens kept about 19 of their 100
([The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations)).

Later the count rises again, as the surviving lines branch. From iteration
100 on, the B1 worlds hold a median of 3,300 families at 100,000 tokens, and
315 at 10,000. Experiment 6 tested its thesis on this count
([Genotypes and lineages](genotypes-and-lineages.md)).

### Families in the code

`CladeWindow` in `gol_series.py`. It needs every frame in order, because the
family tree is a chain. So it is computed while a run records, for the rows of
both phases. A run that keeps only some of its frames has no families count.
It keeps a genotype's place in the tree for sixteen iterations after it first
appears. A genotype still alive keeps its place for as long as it lives.

## Muller plots

How do lines of descent grow, shrink and replace each other over time? A
**Muller plot** shows it as stacked layers. It is named after the geneticist
H. J. Muller, who drew such pictures in 1932.

### How the notebook draws one

1. **Choose the lines.** Fix a moment *t*₀ and a set of ancestors. They may
   be the 100 founders at iteration 0, or every genotype alive after the game
   of some iteration *t*₀. Each ancestor starts a **line**: itself and every genotype
   descended from it ([The family tree](#the-family-tree)).
2. **Count.** After every later game, follow each living agent's genotype up
   the family tree until one of the chosen ancestors is reached. Count the
   agents in each line, and divide by the number of living agents. That is
   each line's **share** of the living, and the shares add up to 1.
3. **Stack.** Draw the shares as layers stacked one on another, so that the
   top of the stack is always at 1. The lines that ever held the largest share
   are drawn in colour, at the bottom; all the others together in grey on top.

### How to read a Muller plot

- The **height** of a layer at a moment is that line's share of the world.
- A layer that **swells** is a line spreading. One that **thins out** and
  vanishes is a line dying out.
- A layer that fills the **whole height** means that everyone alive descends
  from that one ancestor: the line has **swept** the world.

A line spreads in two ways in Graph of Life. One is births. The other, far
more often, is winning nodes in the game, which copies its brain onto the
conquered node ([How the game is played](how-the-game-is-played.md)). A
Muller plot does not tell the two apart.

### Where Muller plots are used

- [The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations):
  the lines of the 100 founders in the first 300 iterations of the world with
  [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1.
- [Genotypes and lineages](genotypes-and-lineages.md): the families of the
  genotypes alive at a later moment, in two worlds.

## The common ancestor of the living

Every living agent's genotype descends, through the family tree, from one of
the founders. Go back far enough and some genotype is an ancestor of **all**
of them. How far back is that?

### The newest genotype they share

At iteration *t*, let *c*(*g*) be the number of living agents whose genotype
is *g* or descends from *g*. The **common ancestor** of a share *q* of the
living is the **newest** genotype *g* from which at least that share
descends:

$$
c(g) \ge q \times (\text{number of living agents}).
$$

The notebook uses *q* = 1 (all of the living), 0.9 and 0.5. Its **depth** is
how many iterations back that genotype was born: *t* minus the iteration of
the first frame it appears in.

While the living still descend from more than one founder, no genotype is an
ancestor of all of them. Then the depth for *q* = 1 is undefined. Where a
depth is undefined, the notebook's figures and medians count it as the age of
the world.

### An example of a common ancestor

At iteration 100, ten agents are alive. Here are their genotypes, how many
agents carry each, and the family tree above them, with the iteration in
which each genotype was born:

```
4,002 (born at 80)
├── 4,610 (88)
│   └── 4,900 (95)
│       ├── 5,021 (96): 3 agents
│       └── 5,022 (97): 4 agents
└── 4,870 (90): 2 agents
    └── 5,100 (99): 1 agent
```

So *c*(4,900) = 3 + 4 = 7, and *c*(4,610) = 7 too. *c*(4,870) = 2 + 1 = 3,
and *c*(4,002) = 10.

- **All of the living**, 10 agents: only 4,002 holds them all. Its depth is
  100 − 80 = 20 iterations.
- **Nine in ten**, 9 agents: again 4,002, at depth 20.
- **Half**, 5 agents: 4,900, 4,610 and 4,002 each hold at least 5. The
  newest is 4,900, at depth 100 − 95 = 5.

### How it is computed

Genotype numbers are handed out in order, and a parent always has a smaller
number than its child. So take the living genotypes with their counts, and
work from the **largest number down**. Take the largest, and check whether its
count reaches each share. Then add its count to its parent's, putting the
parent in the queue if it is not there. A genotype is reached only after all
its descendants have been added into it. So the first genotype found to hold
a share is the newest that does.

In the example, 5,100, 5,022 and 5,021 hold too few. Then 4,900 holds 7, the
first to reach half. Then come 4,870 and 4,610, and last 4,002, with all 10.
In the code: `shared_ancestors` in `gol_analysis.py`, checked every 25
iterations.

### What the depth says

A shallow common ancestor means the living descend from a recent single
line: one genotype, a few iterations ago, took over — a **sweep**. A deep one
means several lines have coexisted for a long time. In B1 the depth for all
of the living rises and falls in a saw-tooth. It grows by one per iteration
while no line takes over, and drops when one does
([Genotypes and lineages](genotypes-and-lineages.md)).

From iteration 500 on, all the living of the B1 worlds of 100,000 tokens
shared one ancestor a median of 946 iterations back. For many worlds that is
the age of the world itself, since their living did not yet all descend from
one founder. Nine in ten shared one from 745 iterations back, and half from
468 (at 10,000 tokens: 730, 590 and 300). The first drop is the moment
everyone alive descends from one founder
([One founder for everyone](a-worlds-life.md#one-founder-for-everyone)).
[Do the brains matter?](do-the-brains-matter.md) counts the drops in worlds
whose brains never change. Once one founder's line fills such a world, every
sweep is chance.

## Brain diversity

How many different brains does a world hold? Three statistics count
genotypes, in every frame.

### The three counts

| field | definition |
|---|---|
| `distinctBrains` | the number of different genotypes among the living |
| `brainDiversity` | `distinctBrains` / `nodes`: distinct genotypes per agent |
| `distinctParents` | the number of different **parent** genotypes among the living; while a founder's genotype lives, its missing parent counts as one more |

- `brainDiversity` = 1 means no two agents share a genotype; a small value
  means a few genotypes have been copied over much of the world. A settled
  B1 world has about 0.63: on average a genotype is carried by about 1.6
  agents ([Genotypes and lineages](genotypes-and-lineages.md)). In B2, where
  a brain changes only when it is copied, ten worlds of 10,000 tokens held
  0.46 ([How fast should brains change?](how-fast-should-brains-change.md)).
- `distinctParents` looks one step up the family tree. It is deliberately
  shallow: it says nothing about how many separate lines there are, which is
  what [Families](#families) and
  [the common ancestor](#the-common-ancestor-of-the-living) measure.

An example. Eight agents are alive: four carry genotype 12, and the other four
carry 13, 14, 15 and 16, one each. Then `distinctBrains` = 5, and
`brainDiversity` = 5/8 = 0.625, close to a settled B1 world's 0.63. If 12
and 13 come from genotype 5, and 14, 15 and 16 from 12, then
`distinctParents` = 2.

### Counting is not the whole story

Two worlds with 1,000 genotypes each can be very different. In one, every
genotype is carried by about one agent. In the other, one genotype is carried
by half the world, and the rest by one agent each. The **effective number of
genotypes**, 2^*H* with *H* the entropy of the genotypes' shares, weighs the
common ones more
([Entropy and evenness](gini-coefficient.md#entropy-and-evenness)). It is
how many equally common genotypes would be as diverse as the ones there are
([Effective numbers](effective-number.md)).

In the example of eight agents, the shares are one half and four eighths.
Each share *p* adds *p* · log₂(1/*p*) to *H*: the half adds ½ · 1, each
eighth adds ⅛ · 3. So *H* = ½ + 4 · ⅜ = 2 bits, and 2^*H* = 4. The five
genotypes are as diverse as four equally common ones would be.

In a settled B1 world the effective number is about 0.45 per agent at 100,000
tokens, and 0.49 at 10,000. The distinct genotypes number 0.63 per agent at
both sizes ([How even is a world?](where-do-the-tokens-go.md#how-even-is-a-world)).

### A caution about names

A genotype number changes whenever a brain is offered a change and takes it
([How a brain changes](mutation.md)). With `mutation_sparsity` = 0 the change
changes nothing, yet the number still changes. Counting genotypes counts
names; whether different names are different brains has to be checked
separately ([Do the brains matter?](do-the-brains-matter.md)).
[How different are the agents?](how-different-are-the-agents.md) counts kinds
of behaviour instead, and compares them with the genotypes.

## Where to go next

- [Genotypes and lineages](genotypes-and-lineages.md): all of these at work in
  the B1 worlds — how long genotypes last, and how lines take worlds over.
- [How a brain changes](mutation.md): when a brain changes in B1 and in B2,
  and what a change does to it.
- [One world, many colours](one-world-many-colours.md): families drawn in
  colour on the network itself.
- [What a run records](frames-and-stats.md): the frames and statistics that
  all these counts are read from.

## How our view changed

- **9 October 2026.** Until that day, the families count let go of every
  genotype sixteen iterations after it first appeared, even one still alive.
  Seen again, such a genotype was taken for a newborn, and for eight
  iterations was counted into its parent's family. So runs recorded before
  2026-10-09 carry counts that are slightly low. Over the 3,000 iterations of
  `B1-10000-s001`, they were low by 0.2% on average, and by 2% at most. That
  is small beside the differences between seeds, but it also made the count
  depend on when a run had last been paused. Since then, a living genotype
  keeps its place for as long as it lives.
- **24 October 2026.** These notes were written on 8 October 2026, when B1 was
  the only baseline. Two of them called it "the baseline", and the note on
  families took B1's rule for changes as the only one. B2, made on
  12 October, changes a brain only when it is copied. So this page now says
  which baseline each statement is about. The 0.63 genotypes per agent, and
  the 1.6 changes a line gathers in eight iterations, are B1's. B2's worlds
  of 10,000 tokens hold 0.46 genotypes per agent, and in B2 an agent that
  keeps its node keeps its genotype.
- **24 October 2026.** Three statements made exact against the code. The
  note on genotypes said that every genotype first appears in some frame; a
  few never do. It said the tree can be rebuilt from a run; in worlds that mix
  newborns' brains, it has gaps
  ([The family tree](#the-family-tree)). And the note on brain diversity
  defined `distinctParents` as the number of parent genotypes among the
  living. While a founder's genotype lives, the code also counts its missing
  parent as one.

## References

- Muller, H. J. (1932). Some genetic aspects of sex. *The American
  Naturalist* 66, 118–138.

<!-- turns -->
---

← [How different are the agents?](how-different-are-the-agents.md) · [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are)

**Also about:** [How the simulation works](../README.md#how-the-simulation-works), [Does evolution keep going?](../README.md#does-evolution-keep-going), [Methods and statistics](../README.md#methods-and-statistics).

> [!info]- Pages that link here (19)
> [Glossary](glossary.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [The world: tokens, agents and a network](the-world.md) · [The brain](the-brain.md) · [How a brain changes](mutation.md) · [The life of a B1 world](a-worlds-life.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [Age, lifetime and survival curves](age-and-lifetime.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [How alike the agents are](how-alike-the-agents-are.md) · [Genotypes and lineages](genotypes-and-lineages.md) · [Do the brains matter?](do-the-brains-matter.md) · [How different are the agents?](how-different-are-the-agents.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Who helps whom](who-helps-whom.md) · [The diversity score](diversity-score.md) · [One world, many colours: what the viewer can colour by](one-world-many-colours.md) · [What a run records](frames-and-stats.md) · [Effective numbers: how many kinds, and how many directions](effective-number.md)
<!-- /turns -->
