# Measuring evolution: genotypes, eras and the open-ended test

> [!question] Questions of this page
> - What is a genotype, a family, and the common ancestor of the living?
> - How do we tell whether a world keeps getting better?
> - What would count as open-ended evolution?
> - What do assembly theory's index and assembly count?

## In short

How the notebook follows evolution: genotypes, families and the common ancestor; eras of a world played against each other; and the test, written down before the long runs, of what would count as open-ended evolution.

Its sections:

- [Genotypes, families and the common ancestor](#genotypes-families-and-the-common-ancestor)
- [Eras, contests and invasions: does a world keep getting better?](#eras-contests-and-invasions-does-a-world-keep-getting-better)
- [What would count as open-ended evolution](#what-would-count-as-open-ended-evolution)

## Genotypes, families and the common ancestor

Every brain carries a whole number, its **genotype**, which a copy keeps and
a change replaces. The new number remembers the old one as its parent, so the
genotypes of a world form one family tree. On it the notebook counts
**families**, or lines of descent, and finds the **common ancestor** of the
living. A **Muller plot** draws the lines of descent as they grow and replace each
other. The notebook also counts how many different genotypes a world holds, its
**brain diversity**: in a settled B1 world, five for every eight agents.

### Genotypes and the family tree

Think of a song passed on by ear, and give every version of it a number. A
singer who sings it exactly passes on the same version, with the same number.
A singer who changes a note makes a new version: it gets a new number, and a
note saying which version it came from. Follow the notes back, and every
version leads to one of the first songs.

A brain's genotype works the same way. Two agents with the same genotype have
brains with exactly the same weights, because they hold copies of one brain.

#### How genotypes are numbered

- The founders' brains get the genotypes 1, 2, …, *n*. In a B1 world of
  100,000 tokens, that is 1 to 1,000.
- **Copying keeps the number.** A child's brain before it changes, and the
  brain a node takes on from the winner of the game, are copies: same
  weights, same genotype.
- **Changing gives a new number.** When a brain changes, it gets the next
  unused number, and remembers the number it had as its **parent genotype**.
  Numbers are never reused, and a parent always has a smaller number than its
  child.

A brain is offered a change only at set moments, and changes with
probability 0.2 each time ([How a brain changes](rules-in-detail.md#how-a-brain-changes)):

- **in both baselines**, a newborn's brain, just copied from its parent;
- **in B1**, also every surviving brain after every game, wherever it sits;
- **in B2**, instead, the copy a node takes on when another agent wins it
  ([`mutate_on_replication`](settings.md#mutate_on_replication)). A brain
  that stays where it is never changes.

B2 has that rule because in B1, 97% of new genotypes are made in brains that
are not being copied
([Variation in the living](questioning-the-mechanics.md#variation-in-the-living)).

An example. Agent A carries genotype 7. It has a child, B, whose brain is a
copy: genotype 7. The copy changes, and becomes genotype 1,204, the next
unused number, with parent 7. In the game, B wins the node of its neighbour
C, and C's node takes on a copy of B's brain: genotype 1,204 again. Now two
agents carry 1,204. In B1, each copy may change after the game, into a new
number with parent 1,204. In B2, C's copy may change the moment it is made,
and A keeps genotype 7 for as long as it keeps its node.

#### Genotypes are not agents

An agent keeps its id and its age for life, but its genotype can change. In
B1 it changes when its brain changes after a game. In both baselines it
changes when its node is won in the game, for then it holds a copy of the
winner's brain. In B2 only the second can happen: an agent that keeps its
node keeps its genotype.

Many agents can share a genotype, and a genotype can live on in many places
after the agent it first appeared in has died. Brains spread mostly by
winning nodes. In a settled B1 world of 100,000 tokens, about 8,100 nodes are
taken by a neighbour in each game. Only about 480 children are born in an
iteration
([Two roads of inheritance](questioning-the-mechanics.md#two-roads-of-inheritance)).

Two different counts follow:

- the number of **agents** alive — `nodes` in the statistics;
- the number of **distinct genotypes** among them — `distinctBrains`
  ([Brain diversity](#brain-diversity)).

A settled B1 world of some 14,400 agents holds about 9,000 different
genotypes at once ([Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages)).

#### The family tree

Following parent genotypes back gives a **tree**: every genotype has one
parent, back to a founder's genotype, which has none. The frames record, for
every living agent, its genotype and that genotype's parent (`brain_ids` and
`parent_brain_ids`), so the tree can be rebuilt from a run. A genotype's
**birth** is the iteration of the first frame it appears in.

A few genotypes never appear in a frame. A frame is taken after the cleanup
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

#### Lifetimes of genotypes

A genotype is *alive* after a game if at least one living agent carries it.
Its lifetime is from the first game after which it is alive to the last,
counted as an agent's lifetime is ([Age and lifetime](measuring-lives.md#age-lifetime-and-survival-curves)).
Genotypes are short-lived. In the B1 worlds of 100,000 tokens the median
genotype lasts two iterations. Of the agents that live through their first
game, 48% last at least ten
([Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages)).

### Families

How many separate lines of descent does a world hold? A **family**, or line
of descent (a lineage), is one genotype together with every genotype
descended from it. Its members are the agents that carry them.

Counting genotypes gives too many lines: one mutation makes a new genotype,
but parent and child are still one family. The **families** count, recorded
as `cladesInWindow`, asks instead how many different **ancestors of eight
iterations ago** the living descend from.

#### Climbing back eight iterations

A genotype is **born** in the iteration of the first frame it appears in. At
iteration *t*, let the **anchor** be *t* − 8. For every living agent, take its
genotype *g* and climb the family tree
([Genotypes and the family tree](#genotypes-and-the-family-tree)):

1. If *g* was born at or before the anchor, stop: *g* is this agent's family.
2. If *g* has no parent (a founder's genotype), stop: *g* is the family.
3. Otherwise move to *g*'s parent and go back to step 1.

The **number of families** is the number of different genotypes the climbs
stop at.

#### An example of families

At iteration 100, the living carry genotypes 5,021, 5,022, 5,100 and 4,870.
Suppose 5,021 and 5,022 were born at iteration 96 and 97, both from 4,900,
born at iteration 95, whose parent 4,610 was born at 88. Genotype 5,100 was
born at 99 from 4,870, and 4,870 itself at 90. The anchor is 100 − 8 = 92.

- 5,021 → 4,900 (95, after the anchor 92) → 4,610 (88 ≤ 92): family **4,610**.
- 5,022 → 4,900 → 4,610: family **4,610**.
- 5,100 → 4,870 (born at 90 ≤ 92): family **4,870**.
- 4,870: born at 90 ≤ 92: family **4,870**.

Two families.

#### Why eight iterations?

Short enough to be a count of *current* lines, long enough that a single
change does not count as a family of its own. In B1 every brain is offered a
change after every game, and takes it with probability 0.2. So in eight
iterations a line gathers on average 8 × 0.2 = 1.6 changes, and more where it
passes through a birth. The climb usually goes a few steps.

Other pages fix the ancestors at another moment, with the same idea:

- the genotypes alive 100 iterations earlier
  ([One world, many colours](measures-atlases-and-assembly.md#one-world-many-colours-what-the-viewer-can-colour-by));
- those alive at one chosen iteration
  ([Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages));
- the founders ([Muller plots](#muller-plots)).

#### Early in a run

Before iteration 8, "eight iterations ago" is before the world began. So the
climb ends at the founders, and the count is how many of them still have
descendants. In the median B1 world of 100,000 tokens, around iteration
8, the living descend from about 390 of the 1,000 founders. The worlds of
10,000 tokens kept about 19 of their 100
([The first hundred iterations](b1-lives-and-wealth.md#the-first-hundred-iterations)).

Later the count rises again, as the surviving lines branch. From iteration
100 on, the B1 worlds hold a median of 3,300 families at 100,000 tokens, and
315 at 10,000. Experiment 6 tested its thesis on this count
([Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages)).

#### Families in the code

`CladeWindow` in `gol_series.py`. It needs every frame in order, because the
family tree is a chain. So it is computed while a run records, for the rows of
both phases. A run that keeps only some of its frames has no families count.
It keeps a genotype's place in the tree for sixteen iterations after it first
appears. A genotype still alive keeps its place for as long as it lives.

### Muller plots

How do lines of descent grow, shrink and replace each other over time? A
**Muller plot** shows it as stacked layers. It is named after the geneticist
H. J. Muller, who drew such pictures in 1932.

#### How the notebook draws one

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

#### How to read a Muller plot

- The **height** of a layer at a moment is that line's share of the world.
- A layer that **swells** is a line spreading. One that **thins out** and
  vanishes is a line dying out.
- A layer that fills the **whole height** means that everyone alive descends
  from that one ancestor: the line has **swept** the world.

A line spreads in two ways in Graph of Life. One is births. The other, far
more often, is winning nodes in the game, which copies its brain onto the
conquered node ([How the game is played](b1-the-game-and-its-flows.md#how-the-game-is-played-what-agents-stake-and-who-wins-the-nodes)). A
Muller plot does not tell the two apart.

#### Where Muller plots are used

<!-- used muller-plots -->
- [The ideas this builds on](the-ideas-this-builds-on.md#genealogy-and-chance)

> [!info]- And in the Archive (1)
> [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md#lines-take-over)
<!-- /used -->

### The common ancestor of the living

Every living agent's genotype descends, through the family tree, from one of
the founders. Go back far enough and some genotype is an ancestor of **all**
of them. How far back is that?

#### The newest genotype they share

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

#### An example of a common ancestor

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

#### How it is computed

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

#### What the depth says

A shallow common ancestor means the living descend from a recent single
line: one genotype, a few iterations ago, took over — a **sweep**. A deep one
means several lines have coexisted for a long time. In B1 the depth for all
of the living rises and falls in a saw-tooth. It grows by one per iteration
while no line takes over, and drops when one does
([Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages)).

From iteration 500 on, all the living of the B1 worlds of 100,000 tokens
shared one ancestor a median of 946 iterations back. For many worlds that is
the age of the world itself, since their living did not yet all descend from
one founder. Nine in ten shared one from 745 iterations back, and half from
468 (at 10,000 tokens: 730, 590 and 300). The first drop is the moment
everyone alive descends from one founder
([One founder for everyone](b1-lives-and-wealth.md#one-founder-for-everyone)).
[Do the brains matter?](do-the-brains-matter.md) counts the drops in worlds
whose brains never change. Once one founder's line fills such a world, every
sweep is chance.

### Brain diversity

How many different brains does a world hold? Three statistics count
genotypes, in every frame.

#### The three counts

| field | definition |
|---|---|
| `distinctBrains` | the number of different genotypes among the living |
| `brainDiversity` | `distinctBrains` / `nodes`: distinct genotypes per agent |
| `distinctParents` | the number of different **parent** genotypes among the living; while a founder's genotype lives, its missing parent counts as one more |

- `brainDiversity` = 1 means no two agents share a genotype; a small value
  means a few genotypes have been copied over much of the world. A settled
  B1 world has about 0.63: on average a genotype is carried by about 1.6
  agents ([Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages)). In B2, where
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

#### Counting is not the whole story

Two worlds with 1,000 genotypes each can be very different. In one, every
genotype is carried by about one agent. In the other, one genotype is carried
by half the world, and the rest by one agent each. The **effective number of
genotypes**, 2^*H* with *H* the entropy of the genotypes' shares, weighs the
common ones more
([Entropy and evenness](measuring-tokens.md#entropy-and-evenness)). It is
how many equally common genotypes would be as diverse as the ones there are
([Effective numbers](statistics.md#effective-numbers-how-many-kinds)).

In the example of eight agents, the shares are one half and four eighths.
Each share *p* adds *p* · log₂(1/*p*) to *H*: the half adds ½ · 1, each
eighth adds ⅛ · 3. So *H* = ½ + 4 · ⅜ = 2 bits, and 2^*H* = 4. The five
genotypes are as diverse as four equally common ones would be.

In a settled B1 world the effective number is about 0.45 per agent at 100,000
tokens, and 0.49 at 10,000. The distinct genotypes number 0.63 per agent at
both sizes ([How even is a world?](b1-lives-and-wealth.md#how-even-is-a-world)).

#### A caution about names

A genotype number changes whenever a brain is offered a change and takes it
([How a brain changes](rules-in-detail.md#how-a-brain-changes)). With `mutation_sparsity` = 0 the change
changes nothing, yet the number still changes. Counting genotypes counts
names; whether different names are different brains has to be checked
separately ([Do the brains matter?](do-the-brains-matter.md)).
[How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents) counts kinds
of behaviour instead, and compares them with the genotypes.

## Eras, contests and invasions: does a world keep getting better?

A [world](the-world.md) whose numbers have settled may have stopped
evolving, or may still be improving in ways no single moment shows. To
tell, the notebook plays a world's [brains](the-brain.md) of two moments
against each other. The brains of one [iteration](one-iteration.md), kept
in the world's archive, are an **era**; a match between two eras is a
**contest**. Its score is the share of the world the later era's
brains hold at the end: 0.5 means no better. Contests compare brains only
with their past; **exploitability** measures what one agent would gain by
its best reply to all the others. The **invasion test** plants a strategy
written by hand in an evolved world, and asks whether it spreads further
than the world's own brains would.

### Why not watch a number rise?

Many numbers rise where nothing is selected. A record grows even in a world
that stands still, and a count only adds. Everything climbs while a young
world settles
([Numbers that rise and mean nothing](how-to-tell-a-world-that-keeps-evolving.md#numbers-that-rise-and-mean-nothing)).
A contest counts nothing, and plays the settled world against its own past.

### Eras

If its plan asks ([`archive_every`](settings.md#archive_every)), a run keeps
a whole world every so many iterations: its network, tokens, ages and every
brain. The brains of one such moment are an era, named by its iteration;
era 0 is the [founders](rules-in-detail.md#the-starting-ring).

### A contest, step by step

1. Take the world of one archived moment: its network, tokens and ages.
2. Give every agent a brain drawn from era A or era B, a coin each, and
   clear the [messages](rules-in-detail.md#messages) in flight.
3. Switch [mutation](rules-in-detail.md#how-a-brain-changes) off, so every copy of a brain keeps its
   era, and let the world play *T* iterations under its own rules.
4. B's **score** is the share of agents carrying a B brain at the end. A
   score of 0.5 is a tie; above it, B's brains spread at A's expense.

Each pair is played twice, once in each era's world, so neither plays only
at home.

### Scores and ratings

Playing every pair of a set of eras is a **cross-time tournament**. A score
*p* can be read as odds. One **rating** *r* per era is fitted to every
contest at once by [least squares](statistics.md#fitting-a-straight-line), with one era, usually
the founders, at 0:

$$
\log \frac{p}{1-p} \approx r_B - r_A .
$$

This is the model of Bradley and Terry, which chess calls Elo. Multiplied by
400 / ln 10, the ratings are Elo points. An era rated +200 beats the
founders about three times in four, and one rated +400 ten times in eleven.

### A contest a lifetime long

Until 15 October 2026 a contest lasted *T* = 20 iterations. That is long
enough for most of a [B1](b1-and-b2.md) world to change hands, but in B2
every agent lives 100. A **lifetime contest** lasts *T* = 100. In the
median of Experiment 16's ten B2 worlds, the brains of iteration 2,000
played those of 500. In contests of 20 iterations they held 0.482 of a
world, a tie. In contests of 100 they held 0.73, and more than half in all
ten worlds.

### Plateau, climb or circle

Contests are noisy, so an era is also played against itself. A score within
0.5 plus or minus twice the standard deviation of those self-contests could
be noise.

Contests between neighbouring eras cannot tell three stories apart:
- a **plateau**, where evolution has stopped;
- a slow climb;
- a circle, where each era beats the one before and loses to one long gone,
  as rock, paper and scissors do.

Playing every era against a few fixed eras can tell them apart. On a
plateau the scores against a fixed era stop rising, on a climb they keep
rising, and in a circle they rise and fall.

Experiment 18's two worlds were played to 50,000 iterations. In lifetime
contests, their eras after 25,000 scored medians of 0.55 and 0.46 against
the brains of 25,000. That is inside their noise bands of 0.40 to 0.60 and
0.36 to 0.64: evolution had stopped climbing
([Every era against three](every-era-against-three.md)). A plateau is not a
top: a best reply would still have gained about a node a game (see
[Exploitability](#exploitability), below).

### The yardstick

In this notebook, "evolution keeps going" means that later eras keep
beating earlier ones, beyond the noise. B1 fails it. In the median of
Experiment 12's ten worlds, the brains of iteration 50 held 0.90 against the
founders. Every later era tied with the one before (0.47 to 0.52), and
contests a lifetime long agree.
[What would count as open-ended evolution](measuring-evolution.md#what-would-count-as-open-ended-evolution) asks more: to
beat the best mixture of all earlier eras. Wins over the last era alone can
go round in a circle.

### Exploitability

Contests compare a world's brains with their own past. They cannot say how
good the brains are in any absolute sense. Brains that tie with their
ancestors may be at the top, or stuck far below it: a plateau looks the
same either way. **Exploitability** is a measure from game theory that can
tell the two apart.

#### The idea of a best reply

Picture one agent while every other agent keeps its play. Whatever the
others do, some way of staking does best against them: the agent's **best
reply**. How much better the best reply does than the agent's own play is
the agent's exploitability.

In a **Nash equilibrium**, named after the mathematician John Nash, no
player gains by changing its own play while the others keep theirs. So at
an equilibrium, exploitability is zero for everyone. A population that
evolution drives towards an equilibrium becomes less exploitable as it
goes; one that is stuck short of it stays exploitable.

#### How exploitability is measured

The agents' game is too large to solve, but a single agent's best reply to
one game is a small problem (`research/oee/checks/exploitability.py`):

1. Take a world as its archive keeps it. Play it on for one iteration, a
   phase of reproduction and a game, so that its messages are fresh. Then
   draw, eight times, the random numbers the brains read. For each draw,
   run every agent's brain and turn its outputs into stakes exactly as the
   engine does. This is the others' play, as a distribution over eight
   draws.
2. Score a way of staking for one agent by what it **holds** after the
   game. It scores one for keeping its own node, and one for every
   neighbour's node it takes. Everyone else stakes as drawn, and every node
   is decided by the engine's rule. Under B2 the owner's stake counts one
   and a half times, and a revolution can take a node. Ties are drawn with
   a fixed seed for each node and draw. The score is averaged over the
   eight draws.
3. The agent's own play scores what its drawn stakes hold, averaged over
   the eight draws. The best reply is searched among a few natural plays,
   none of which marks a stake revolutionary:
   - everything at home;
   - for each draw, the cheapest way to keep its node and then take the
     cheapest neighbours. Taking a neighbour costs one token more than the
     largest rival offer on it;
   - the same without defending beyond what B2 keeps at home anyway;
   - the agent's own stakes of each draw.

   Each is scored over all eight draws, so a play that only fits one draw
   is not rewarded for it. If none does better than the agent's own play,
   the agent's exploitability is zero.
4. Exploitability is the best reply's score minus the agent's own, averaged
   over up to 300 agents with at least three tokens.

**An example.** In the median of Experiment 16's ten B2 worlds, the
founders could have gained 0.87 nodes a game by their best replies. The
brains of iterations 1,000 to 2,000 could have gained about 1.2. Evolution
moved them away from an equilibrium, not towards it
([Would we notice if it went on?](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on)).

#### What exploitability can and cannot say

The best reply is only searched among some plays, so the exploitability
found is a lower bound. The true distance from an equilibrium is at least
this large.

It is a distance in **one game**. Holding a node for one game is not the
same as spreading over a lifetime. A play that takes many cheap nodes now
may cost an agent later. So a population can be exploitable in one game and
still hard to beat in the long run.

The measure is a probe, not a verdict. What it shows most clearly is a
*trend*: whether evolution moves a population towards an equilibrium, away
from it, or nowhere.

### The invasion test

A contest asks whether a world's later brains beat its earlier ones. The
**invasion test** asks something else: is there play *better* than what a
world has evolved? It plants a strategy in the evolved world and watches
whether it spreads. A **strategy** takes the place of an agent's brain: it
is a rule written by hand, or a brain whose weights were set by hand.

#### How an invasion is played

1. Take an evolved world as its archive keeps it: its graph, its tokens
   and its brains (`GraphOfLife.from_checkpoint`).
2. Choose a **patch**: an agent drawn at random, then the agents nearest to
   it, step by step through the network. The patch grows until it holds the
   share of the world asked for (5% in
   [Strategies written by hand](strategies-written-by-hand.md#strategies-written-by-hand)).
   A patch can also be a given number of agents, down to a single one.
3. Give every agent of the patch the strategy in place of its brain
   (`research/oee/pilots/manual.py`; the rules now live in the engine's
   `gol_strategies.py`). It spreads as a brain does, into children and into
   nodes its agents take, and it never mutates.
4. Play on under the world's own rules and count, after every iteration, the
   agents that carry the strategy, as a share of all agents.

#### What a strategy can and cannot compute

A strategy reads exactly the inputs a brain reads, and writes exactly the
outputs a brain writes. So it sees nothing a brain could not see. It acts
only through the outputs a brain has, which the engine turns into a child,
its links, the stakes and the messages.

What it makes of what it reads is another matter. A rule written in Python
can combine its inputs in ways a brain cannot. A B2 brain reads one column
of inputs for each candidate, the agent itself and each neighbour. It
scores each column on its own, with the same weights, and no unit sees two
columns. Two of the
[three strategies](strategies-written-by-hand.md#three-strategies) first
planted show the difference:
- **The raider** stakes all it can on the poorest neighbour outside its
  team, if it is sure to take it. Twelve of a B2 brain's 16 units, with
  weights set by hand, play it
  ([A raider in a brain's weights](what-a-brain-can-compute.md#a-raider-in-a-brains-weights)).
- **The factory** is a team whose members write into their messages how far
  they are from the team's frontier, where strangers begin. A member with a
  stranger next door writes 0 to its teammates: news of a third agent, the
  stranger. No B2 brain of any size can put such news into a message
  ([What no brain of B2 can compute](what-a-brain-can-compute.md#what-no-brain-of-b2-can-compute)).

So a strategy that wins shows that the game holds better play than
evolution found. Whether a brain of that world could play it is a separate
question, answered for the raider by building the brain.

#### The control

A patch can grow or shrink because of where it sits. It may lie next to
rich agents or poor ones, in the middle of the world or at an edge. So the
share a strategy reaches means little on its own. Every patch is also
played as a **control**. The same agents each get a frozen copy of one
resident brain: the brain of an agent of the patch, drawn at random. The
copies are marked, so they can be counted. A strategy beats the residents
if it ends with more of the world than this control, planted in the same
place on the same seed.

The lab's tests, from Experiment 29 on
([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)),
ask a little more. A planting **takes hold** when, after 300 iterations,
its agents are alive, more than the patch began with, and more than the
control holds there.

#### Reading an invasion

- **Growth from a patch** says that the strategy beats the residents *when
  it has a crowd*: better play exists than the play evolved.
- **Growth from a single agent** says more: the strategy needs no crowd. A
  mutation changes one brain at a time, so if a mutation ever made it, it
  could spread from there. A strategy that grows from a patch but not from
  one agent has an **Allee effect**. Ecologists use the name for a species
  that does well only in numbers. Evolution, which tries one agent at a
  time, can miss such a strategy although it is better.
- **A strategy's parts** can be tested one at a time by dropping them. The
  raider without its password
  ([The password](strategies-written-by-hand.md#the-password)) shows which
  part wins.

**An example.** In world 2 of Experiment 16 at iteration 2,000, a patch of
5% of raiders was planted on two seeds. In 300 iterations it grew to 40%
and 50% of the world. The control, the residents' own brains planted in the
same places, ended with 7% and 1%
([What the strategies did](strategies-written-by-hand.md#what-the-strategies-did)).

**One planting says little.** The raider's rule and the brain that plays it
choose differently in only one or two situations in a hundred. Yet planted
on the same seed in world 3, the rule died out while the brain took 59% of
the world. Their worlds part ways within a few iterations, and from there
chance decides whether a patch takes hold
([A raider in a brain's weights](what-a-brain-can-compute.md#a-raider-in-a-brains-weights)).
So the lab's tests plant a strategy in several places in each world.

A strategy that dies out says at most that *this* strategy is worse, never
that nothing better exists. An invasion test can show that a game is not
solved, but not that it is.

### Where it is used

<!-- used eras-contests-and-invasions-does-a-world-keep-getting-better -->
- [What this notebook is about](what-this-notebook-is-about.md#a-ladder-of-four-steps)
- [The ideas this builds on](the-ideas-this-builds-on.md#where-to-go-next)
- [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md#what-decides-how-a-world-evolves)
- [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md#what-could-be-played-next)
- [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md#where-to-go-next)
- [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md#why-play-eras-against-each-other)
- [A hundred thousand iterations](a-hundred-thousand-iterations.md#where-to-go-next)
- [Every setting](settings.md#archive_every)
- [B1 and B2: the two baselines](b1-and-b2.md#where-to-go-next)
- [What the brains are like](what-the-brains-are-like.md#when-the-brains-were-given-sight)
- [Brains replaced by their average](brains-replaced-by-their-average.md#five-worlds-from-iteration-1500)
- [How big should a brain be?](how-big-should-a-brain-be.md#a-planned-experiment-answered-from-round-3)
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md#playing-against-the-past)
- [How fast should brains change?](how-fast-should-brains-change.md#how-the-rates-are-compared)
- [Every era against three](every-era-against-three.md#where-to-go-next)
- [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md#the-invasion-test)
- [What a brain can compute](what-a-brain-can-compute.md#planted-in-its-place)
- [One agent in a sea of average agents](one-agent-in-a-sea-of-average-agents.md#where-to-go-next)
- [Other contests: a lottery, and alliances](another-game.md#what-the-contests-did)
- [Rules without numbers](rules-without-numbers.md#did-the-worlds-that-lived-keep-evolving)
- [How runs are made, recorded and costed](runs-and-records.md#runs-and-experiments)
- [Every simulation: what was run, and what was measured](every-simulation.md#the-labs-experiments)

> [!info]- And in the Archive (6)
> [What B1 is, and what stood in its way](what-the-baseline-world-is.md#where-the-world-stands-on-the-ladder) · [B1: the game and its flows](b1-the-game-and-its-flows.md#what-we-learned-how-the-game-is-played) · [Building B2, and the search for a healthy, rich world](building-b2-and-the-search-for-a-healthy-rich-world.md#what-kills-a-world) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md#does-evolution-go-on-eight-switches-at-once) · [Cooperation, help and gliders (side study)](cooperation-help-and-gliders.md#both-inputs-tested) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md#brains-as-assembled-objects)
<!-- /used -->

## What would count as open-ended evolution

No run can show that evolution goes on for ever. So the notebook writes
down, before the long runs exist, what a run would have to show. A configuration of
rules passes when at least two of three new worlds pass four parts. Their
brains beat the best mixture of their whole past, and their ratings keep
rising. They keep doing new things, faster than mutation alone would. And
they stay worlds in which much can happen. Each part closes one way of being
fooled. They are a circle, a plateau and an arms race that swaps two
answers. The fourth is a world that evolves only because it is poor in all
else.

### Why write the test first

No run can show that evolution goes on for ever. A run is finite, and so is
the budget. A world of 10,000 tokens runs about 100,000 iterations in a day
and a half. A world of 100,000 tokens runs a tenth as far in two days
([What a simulation costs](runs-and-records.md#what-a-simulation-costs)). What a run can show
is that, as far as it went, nothing stopped:
- later brains beat their whole past;
- they keep getting better;
- they keep doing new things;
- the world stays one in which much can happen.

This page writes the test down before any of the long runs it will judge
exists. A thesis written after its results can be bent to fit them; one
written before cannot.

### The test

A configuration of rules passes the test when **at least two of three new
worlds** pass all four parts below. That is open-ended evolution, as far as a
run can show it.
- New means seeds that played no part in choosing the configuration. A
  configuration picked as the best of many is lucky as well as good, and its
  own seed says nothing about that.
- Every world played counts, the ones that fail included.

The four parts use the contests of
[Eras, contests and invasions](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better). An **era** is the
brains of one moment of a world. A **contest** plays two eras against each
other in the same world, and scores the share of it the later one holds.

1. **It beats its whole past.** At every 10,000 iterations, the brains of
   that moment beat the best mixture of all earlier eras. They do so in
   contests a lifetime long: 100 iterations, played on both worlds, from both
   sides.
   - The best mixture is the Nash mixture of the eras' table of contests: the
     blend of past brains that does best against everything.
   - Beating the last era is not enough. In a game like this one, every
     strategy has an answer. So a chain of "beats the one before" can go round
     in a circle for ever without getting anywhere, as rock, paper and
     scissors do.
   - The margin must be larger than the spread of an era played against
     itself.
2. **It keeps rising.** Read each era's rating against the best mixture of
   the others (Nash averaging; Balduzzi and others, 2018). The ratings rise
   over the second half of the run: the 95% interval of the slope, found by
   the [bootstrap](statistics.md#how-sure-can-we-be-the-bootstrap-permutation-tests-the-wilson-interval-and-how-many-seeds) over contests, lies above zero.
3. **It keeps doing new things.** New kinds of behaviour appear faster than
   they would by mutation alone, and the old ones do not simply come back.
   - Kinds of behaviour means two things. One is kinds of staking
     ([How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents)). The
     other is the signatures of the strategies written by hand: how a
     strategy answers a fixed set of situations
     ([What a brain can compute](what-a-brain-can-compute.md)).
   - "By mutation alone" means the same brains, changed only by the random
     mutations their lineages collected over the same time, with no
     selection.
   - A world whose kinds come and go in a cycle does not pass.
4. **It stays a world in which much can happen**
   ([Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich)):
   - it never falls to 20 agents;
   - it meets every mark of the [health bar](measuring-health.md#the-health-bar "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") at every 10,000 iterations;
   - its agents are more diverse than those of B2's median world;
   - its tokens are no more even than B2's: a
     [Gini coefficient](measuring-tokens.md#measuring-inequality-the-gini-coefficient-the-richest-tenth-and-entropy) no lower than the least of
     B2's worlds read at the same iteration.

### What it does not ask

- **Speed.** Evolution may take as long as it likes, so long as it does not
  stop within the run.
- **Typical worlds.** "Two of three new worlds" asks whether a configuration
  *can* keep evolving, not whether it usually does. A configuration that does
  so in a few worlds is a find.
- **A particular kind of novelty.** Larger groups, deeper organisation and
  new signals all count, so long as they beat the past and were not seen
  before.

### Why these four parts

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

- [Genotypes and lineages](b1-brains-lineages-and-messages.md#genotypes-and-lineages): all of these at work in
  the B1 worlds — how long genotypes last, and how lines take worlds over.
- [How a brain changes](rules-in-detail.md#how-a-brain-changes): when a brain changes in B1 and in B2,
  and what a change does to it.
- [One world, many colours](measures-atlases-and-assembly.md#one-world-many-colours-what-the-viewer-can-colour-by): families drawn in
  colour on the network itself.
- [What a run records](runs-and-records.md#what-a-run-records): the frames and statistics that
  all these counts are read from.
- [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick, held to both baselines.
- [Every era against three](every-era-against-three.md): every era of the
  longest worlds against fixed eras, and the room a best reply finds there.
- [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md):
  the first invasion tests, and what a raider needs to win.
- [What a brain can compute](what-a-brain-can-compute.md): the strategies
  whose signatures count as new things.
- [Healthy and rich](building-b2-and-the-search-for-a-healthy-rich-world.md#healthy-and-rich): what a world in which much can
  happen looks like.
- [Assembly theory](measures-atlases-and-assembly.md#assembly-theory-how-many-steps-to-make-a-thing-and-how-many-copies): how many steps an agent's
  neighbourhood takes to make, and how many agents share its shape. What the
  index and assembly are is in
  [Measures of the archived studies](measures-of-the-archived-studies.md#assembly-index-and-assembly).

## How our view changed

- *Genotypes, families and the common ancestor*:
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
- *Eras, contests and invasions: does a world keep getting better?*:
  - **24 October 2026.** From 15 October, the note on the invasion test said
    that a strategy written by hand "can do nothing an evolved brain could
    not". The reason it gave was that a strategy reads and writes exactly what
    a brain does. On 19 October,
    [What no brain of B2 can compute](what-a-brain-can-compute.md#what-no-brain-of-b2-can-compute)
    showed that this holds for the raider but not for the factory. Twelve of a
    B2 brain's 16 units play the raider. But a brain scores each candidate's
    column on its own. So no B2 brain of any size can write news of a third
    agent into a message. The factory's distance to its frontier is such news.
    A strategy sees only what a brain sees and acts only through what a brain
    decides, but it may compute what no brain can.
  - **24 October 2026.** From 15 October, the note on exploitability described
    three details of the measure differently from the program that makes it
    (`research/oee/checks/exploitability.py`). We read the program when the
    notes became sections of this page. Before the measure, the world plays
    one whole iteration, a phase of reproduction and then a game; the note
    said one game. The gain is averaged over up to 300 agents with three
    tokens or more, where the note said 300. In 7 of the 145 measurements
    behind [How much room is left](every-era-against-three.md#how-much-room-is-left),
    only 154 to 274 agents held that many. And the plays searched for the best
    reply include the agent's own stakes of each draw, which the note did not
    list.
- **12 November 2026.** Moved to [Measures of the archived
  studies](measures-of-the-archived-studies.md): the section on the assembly
  index and assembly, which only pages of the Archive use.

## References

- Muller, H. J. (1932). Some genetic aspects of sex. *The American
  Naturalist* 66, 118–138.

- Balduzzi, D., Tuyls, K., Pérolat, J. and Graepel, T. (2018). Re-evaluating
  evaluation. *Advances in Neural Information Processing Systems* 31.

<!-- turns -->
---

← [Measuring space: dimension and the random twin](measuring-space.md) · [Reference: the rules and measures in detail](../README.md#reference-the-rules-and-measures-in-detail) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) →

**Also about:** [Brains: what they are, and whether they matter](../README.md#brains-what-they-are-and-whether-they-matter), [How the simulation works](../README.md#how-the-simulation-works), [Does evolution keep going?](../README.md#does-evolution-keep-going), [Methods and statistics](../README.md#methods-and-statistics), [Strategies written by hand](../README.md#strategies-written-by-hand), [Start here](../README.md#start-here).

> [!info]- Pages that link here (35)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [How this notebook is written](how-this-notebook-is-written.md) · [The ideas this builds on](the-ideas-this-builds-on.md) · [Candidates for open-ended evolution](candidates-for-open-ended-evolution.md) · [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md) · [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [The world: tokens, agents and a network](the-world.md) · [The brain](the-brain.md) · [Glossary](glossary.md) · [The rules in detail](rules-in-detail.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Measuring lives: births, deaths, ages and estates](measuring-lives.md) · [What keeps wealth spread?](what-keeps-wealth-spread.md) · [Do the brains matter?](do-the-brains-matter.md) · [Brains replaced by their average](brains-replaced-by-their-average.md) · [How big should a brain be?](how-big-should-a-brain-be.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [How fast should brains change?](how-fast-should-brains-change.md) · [Every era against three](every-era-against-three.md) · [Can a strategy written by hand beat evolution?](strategies-written-by-hand.md) · [What a brain can compute](what-a-brain-can-compute.md) · [One agent in a sea of average agents](one-agent-in-a-sea-of-average-agents.md) · [Rules without numbers](rules-without-numbers.md) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) · [How runs are made, recorded and costed](runs-and-records.md) · [Statistics: the tools of the notebook](statistics.md) · [B1: lives and wealth](b1-lives-and-wealth.md) · [B1: the network, size and scaling](b1-the-network-size-and-scaling.md) · [B1: brains, lineages and messages](b1-brains-lineages-and-messages.md) · [What the switches did in pilots (indicative)](what-the-switches-did-in-pilots.md) · [Cooperation, help and gliders (side study)](cooperation-help-and-gliders.md) · [Space at scale (side study)](space-at-scale.md) · [Measures, atlases and assembly (side studies)](measures-atlases-and-assembly.md)
<!-- /turns -->
