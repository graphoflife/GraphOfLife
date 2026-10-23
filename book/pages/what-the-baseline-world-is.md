# What B1 is, and what stood in its way

> [!question] Questions of this page
> - What is a settled B1 world, in numbers, and what do the pages that
>   measured it suggest about how it works?
> - What changes when a world is ten times larger, and what does not?
> - Where does B1 stand on the ladder towards [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.")?
> - What stood in its way, which changes to the rules were proposed, and what
>   came of them?
> - How would we know that a change of the rules worked?

## In short

The first baseline, B1, has been measured from every side, in thirty worlds
of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). All six theses written before its runs failed, and each
failure taught something new about the world. Together, the measurements show
a world of places, not individuals: the rules shape its network and its
wealth, while its [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") barely see and change almost every game. Read for
open-ended evolution, B1 learns what it learns in its first fifty to a
hundred iterations, builds no organisations, and has seven obstacles in its
way, its blind brains first. Giving the brains sight broke the worlds, so the
notebook built a second baseline, B2, that cannot break. There evolution goes
on for thousands of iterations before it, too, slows to a plateau.

## What the baseline world is

Thirty worlds of the first baseline, [B1](b1-and-b2.md), have been looked at
from every side ([Thirty worlds](a-worlds-life.md#thirty-worlds)). This section puts the
pieces together: what the theses predicted, what was found instead, what
seems to hold across the pages, and what that changed in how the notebook
measures.

It was first written after Experiments 1 to 5, on worlds of 10,000 tokens,
and kept up to date as more pages were written. Its second edition reads
thirty worlds of 100,000 tokens. Two pages are new with it:
[How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life) and
[The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world).

### The record

Each study of B1 began with a thesis: a prediction written down before any of
its runs existed, together with what would confirm it and what would refute
it. Each thesis was tested on the worlds of 10,000 tokens. The second edition
tested it again on the worlds of 100,000.

- [Is a run reproducible?](is-a-run-reproducible.md) Thesis **refuted**, but
  only in the last bits of a number, by how many threads the matrix library
  used. Every run now uses one thread, and is then exactly reproducible.
- [A world's life](a-worlds-life.md). Thesis **refuted** at both sizes. At
  10,000 tokens births fell by 41%, and 4 of 30 worlds died. At 100,000 none
  died and the typical world held its size; only the fall of births (28%)
  broke the thesis.
- [How much does the seed decide?](how-much-does-the-seed-decide.md) Thesis
  **refuted, narrowly**, at both sizes. The [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") decides little: worlds differ
  mostly because each one wanders, not because of their seeds.
- [Where do the tokens go?](where-do-the-tokens-go.md) Thesis **refuted** at
  both sizes. Inequality is moderate and steady, with a [Gini coefficient](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") just
  under 0.5, and [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") stake 70% of their tokens on their [neighbours](the-world.md#the-network "An agent joined to another by a connection.").
- [What shape does the network take?](what-shape-does-the-network-take.md)
  Thesis **refuted**, narrowly at 10,000 tokens (a core of 53%) and more
  clearly at 100,000 (57%). The network is part tree, part web.
- [Genotypes and lineages](genotypes-and-lineages.md). Thesis **refuted** at
  both sizes. A [genotype](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") lasts two iterations. Lines of descent take the world
  over: again and again in a small world, about once a run in a large one.

Six of six theses failed, and no failure changed with the size of the world.
That is what a thesis written before its runs is for: each failure is
something about the world that was not known before.

### The baseline world in numbers

Here is a settled B1 world of 100,000 tokens. Each number is the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") over
the thirty worlds of each world's mean over iterations 500 to 1,499. The
worlds of 10,000 tokens are given in brackets where they differ.

- **About 14,400 agents** (1,300), with **3.5 [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.")** each (3.3)
  ([A world's life](a-worlds-life.md),
  [What shape does the network take?](what-shape-does-the-network-take.md)).
- **3.5 births** per hundred agents per iteration (3.1), and as many deaths.
  Three deaths in four are agents cut off from the network
  ([Births, deaths and ages](births-deaths-and-ages.md)).
- **Ages that settle by iteration 500.** 2.4% of the living are newborns,
  and half are younger than 55 iterations. The risk of dying falls with age,
  to about one game in a hundred: there is no ageing. The ages of the living
  follow from the [lifetimes](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death."), as in a stationary population, one whose make-up
  holds still while its members come and go
  ([How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life)).
- **A Gini coefficient of wealth of 0.47** (0.49), with the [richest tenth](richest-tenth.md "The share of all tokens held by the richest tenth of the agents.")
  holding **42%** of the tokens (44%)
  ([Where do the tokens go?](where-do-the-tokens-go.md)).
- **44%** of agents keep their node in a game (45%), and [coalitions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") win
  **half** of all nodes ([How the game is played](how-the-game-is-played.md)).
- **26%** of connections are [bridges](bridges.md "A connection that lies on no loop: cut it and the network falls into two pieces.") (31%), **35%** of agents are leaves
  (36%), **57%** sit in a core of [loops](loops.md "A path through the network that comes back to where it started without using a connection twice.") (54%), and two agents are **18
  steps** apart (10)
  ([What shape does the network take?](what-shape-does-the-network-take.md)).
- **A genotype for every 1.6 agents**, each lasting a couple of iterations,
  and lines that bring a whole world down to one [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s descendants by
  about iteration 1,000 (175 at 10,000 tokens)
  ([Genotypes and lineages](genotypes-and-lineages.md)).
- **Brains with a [gain](vanishing-signal.md) of 0.0015**: a change in what a
  brain sees reaches what it does at a seven-hundredth of its size. Selection
  has set their constants alike in every world: a child in 3.6% of cases, the
  stake spread in 98%. In the median world 42% of agents score every
  candidate at zero or below, so the rules split their tokens evenly
  ([What the brains are like](what-the-brains-are-like.md)).
- **[Messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") that are names.** One agent's messages to two readers differ by
  0.0005, two agents' messages by 0.6, and agents of one genotype write the
  same ([What agents say to each other](what-agents-say-to-each-other.md)).
- **An order of wealth that is half forgotten in fifty to a hundred games**,
  and rich agents with one connection that die three times as often as other
  leaves ([Do the rich stay rich?](where-do-the-tokens-go.md#do-the-rich-stay-rich)).
- **Connections gained in proportion to the connections an agent has**
  (as *k*^1.02) and lost faster (as *k*^1.20), and **no new connection ever
  bringing two agents closer**
  ([How the network grows](how-agents-have-children.md#how-the-network-grows)).
- **Likeness that fades within two or three steps**: tokens, age,
  connections and kin alike ([Like next to like](like-next-to-like.md)).
- **A finite [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.")**: 2.8 by how the room within *r* steps grows (2.3 at
  10,000 tokens, 3.3 in worlds of 60,000 agents), and about 1.9 by how a
  walker spreads. Random networks with the same connections have none
  ([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)).
- **A core** of three agents in five at the start of a game (57% after it,
  as above). It holds three quarters of the tokens and every loop. Its
  dimension by balls is a little lower than the whole world's (2.5), and the
  same by walks ([The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world)).

### What seems to hold across the pages

Six patterns run through the pages on B1.

#### One slow wander, in everything

Watch one world's number of agents over its life: it drifts up and down,
slowly, like the level of a lake.
[How much does the seed decide?](how-much-does-the-seed-decide.md) measured
how long a world remembers its level, statistic by statistic. They all forget
at much the same pace. A hundred iterations later a world's level is still
clearly alike: [correlations](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of 0.52 to 0.69 for the number of agents,
connections, inequality, families and the core, and 0.36 for births. Three
hundred iterations later nothing is left (−0.06 to 0.08). So size, wealth and
shape do not seem to wander each on its own schedule. They look like one slow
process, seen through many windows.

What could that process be? In the worlds of 10,000 tokens,
[Genotypes and lineages](genotypes-and-lineages.md) found one with about that
rhythm. Every 350 iterations or so, one branch of the family tree outlived
all the others, and the whole world descended from it. Was the wander made by
these sweeps? The large worlds answer no. They forget their level within
three hundred iterations, as the small ones do, but a line needs about a
thousand to take one over ([Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)).
The wander is faster than the sweeps, and so not made by them.
[Do the brains matter?](do-the-brains-matter.md) finds the other half of the
answer: a world whose brains never change hardly wanders at all.

#### A world of attack

Agents put only 29% of their staked tokens on their own node
([How the game is played](how-the-game-is-played.md)). More than half of all
nodes change hands in every game, and coalitions of smaller stakers take half
of all nodes.

Yet defence got better as a world settled. Over its first five hundred
iterations, the share of agents that keep their node rose from 18% in the
first game to 44%. Staking more at home is not what changed. At 100,000
tokens the share of staked tokens put at home did rise, from 13% to 29%. But
at 10,000 tokens it barely moved (25% to 30%), and defence got better all the
same.

What changed is how evenly agents stake
([Where the tokens flow](where-the-tokens-flow.md)). Picture a node with *d*
connections. Under the [even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly."), the agent living there is one of *d* + 1
stakers on it, all staking about the same. So it keeps its node about one
game in *d* + 1: in 74% of games with one connection, and in 2% with fifty or
more, as measured. The founders went all in half the time, and in the first
game 43% of them put nothing on their own node. By iteration 500, 98% of
agents spread their stake. Defence got better because the stakes became
even, not because anyone chose to defend.

#### A turnover that settles late

Births fall long after a world's size has settled: by 28% from iterations
100–200 to the end in a large world, and by 41% in a small one. With them,
the ages of the living rise. In a large world the change is over by
iteration 500. From then on its births per agent, its lifetimes and its ages
hold still, as a stationary population's do
([How ages change over a world's life](births-deaths-and-ages.md#how-ages-change-over-a-worlds-life)). The small worlds
went on slowing until about iteration 1,250, and their living grew older than
their lifetimes alone would make them.

Is this slow change adaptation? The second step of the ladder towards
open-ended evolution looks for adaptation that keeps accumulating
([What this notebook is about](what-this-notebook-is-about.md#a-ladder-of-four-steps)).
Neither trend is that: the demography of a world is finished early. Playing a
world's brains against its own past later showed the same of the brains: B1
stops improving within its first hundred iterations
([Where the world stands on the ladder](#where-the-world-stands-on-the-ladder)).

#### Brains that cannot see

[What the brains are like](what-the-brains-are-like.md) opened the brains and
found them nearly blind. A B1 brain has five hidden layers, and each squashes
what it receives through a sigmoid: an S-shaped curve whose slope is never
more than a quarter. So each layer passes a change in its inputs on at about
a quarter of its size, and five layers in a row leave about a thousandth,
since (1/4)^5 = 1/1,024. Much of what the pages on B1 found, and the
measurements after them, follows from that one number:

- The stakes are nearly even, so the game moves tokens almost as a random
  walk would: as if each token stepped, game after game, to its agent or to a
  neighbour picked at random ([Where the tokens flow](where-the-tokens-flow.md)).
  It does so down to the two-game rhythm of wealth that a random walk shows on
  [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") and leaves ([Do the rich stay rich?](where-do-the-tokens-go.md#do-the-rich-stay-rich)).
- An agent's tokens are set by its place, its connections plus one, and not
  by how it plays ([How properties scale together](how-properties-scale-together.md)).
  In the long run such a walk visits each agent in proportion to its
  connections plus one, its resting state; the rich are those it visits most.
- Messages carry a name and nothing else
  ([What agents say to each other](what-agents-say-to-each-other.md)). Kin
  could tell each other by it, but cannot use it
  ([Do agents cooperate?](do-agents-cooperate.md)).
- Selection has acted, the same way in every world, but on constants: have
  few children, spread the stake, and mark every staked token revolutionary,
  so that it can join a coalition.

So evolution in B1 is the evolution of a few constants. That is the first
obstacle to open-ended evolution, and in B1 it sits in the brains'
architecture, not in the world. Giving the brains sight was the first thing
the notebook changed. It was not enough: it broke the worlds, as
[What stands in the way](#what-stands-in-the-way) tells.

#### A space, not a small world

New connections are only ever made inside a neighbourhood, and none brings
two agents closer ([How the network grows](how-agents-have-children.md#how-the-network-grows)). So a
world grows as a body does. It does not become a "small world", in which
everyone is a few steps from everyone. It has a finite dimension
([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)),
and its core is a space of the same kind
([The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world)). Its width grows with its
size ([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)),
where a random network with the same connections would be a small world.

Yet nothing in this space keeps regions apart. Likeness fades within two or
three steps ([Like next to like](like-next-to-like.md)). Lines of descent,
which last much longer than genotypes, do hold patches for a while. Two
neighbours share a family of 25 iterations back 99 times as often as chance,
and the excess reaches about 7 steps; for families of 200 iterations back,
about 12 ([Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)). But a patch
widens as a random walk on these worlds would, until one line holds the whole
world. A world has room for regions that differ, and lines that make them for
a while, but nothing that keeps them apart.

#### Stars

Seven of the thirty large worlds condensed, for long stretches, around a
single agent joined to a tenth of the world or more; in one, to more than
half ([How much does the seed decide?](how-much-does-the-seed-decide.md)).
Such a world plays its game differently
([How the game is played](how-the-game-is-played.md)). It also has a different
shape: fewer agents in the core, and far more leaves and bridges
([What shape does the network take?](what-shape-does-the-network-take.md)).
[The largest and the smallest](the-largest-and-the-smallest.md) shows that
the centre of a star is a place, handed down from parent to newborn child.

Stars are not new with size: ten of the 26 worlds of 10,000 tokens had one by
the same measure. But in a large world a star decides the world's whole
shape. Across the thirty, the share of agents in the core follows the share
joined to the largest hub almost exactly
([How much does the seed decide?](how-much-does-the-seed-decide.md)).

B2 counts a star as a fault of the network. The [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") of
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
allows no agent to be joined to more than 5% of the others.
[A world that cannot break](a-world-that-cannot-break.md) keeps hubs down: a
newborn's links to its parent's neighbours are drawn at random, and a brain
may choose at most 8 links.

### What this changed in the method

- **The youth is left out.** Every thesis from here on is checked from
  iteration 500 on, the settled life of a run
  ([The settled life of a world](settled-life.md)).
- **Long stretches, not the last fifth.** A run's mean over its settled life
  varies across seeds by much less than its last fifth does. For the number
  of agents it varies by 0.10 of the mean instead of 0.17 at 100,000 tokens,
  and 0.17 instead of 0.25 at 10,000. So it needs fewer than half the seeds
  ([How many seeds an experiment needs](seeds-needed.md)).
- **Thirty seeds see changes of about a tenth.** Over the settled life,
  thirty seeds per condition see a change of 8% in the number of agents, 9%
  in connections and 5% in inequality at 100,000 tokens, four times in five
  (12%, 14% and 5% at 10,000). Smaller effects are reported as not seen, not
  as absent.
- **The dead are counted apart.** About one world in eight dies at 10,000
  tokens: it falls to 20 agents, which
  [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
  later found is not always the end. None of thirty did at 100,000.
  Extinction is reported for every condition, but only a large change in it
  can be seen: from 13% to 30% would take about ninety seeds per condition
  ([When a world ends](extinction.md)).
- **Pairing by seed helps little.** Two conditions with the same seed share
  their starting ring, and little else.
- **Lines, not genotypes.** A genotype lasts two iterations. What lasts,
  spreads and takes over is a line, and the pages after these follow lines.
- **Size is a setting like any other.** A finding at one size is checked at
  another before it is trusted. What a larger world changes, and what it does
  not, is told in [What the second edition changed](#what-the-second-edition-changed).

### What came next

The measurements of B1 then read the same thirty worlds through every
statistic the Graph of Life viewer offers: [entropy](entropy-and-evenness.md "How evenly something is shared out among the agents, from all in one place to spread perfectly evenly."), gains and losses, the
flow of tokens, what crosses a connection, children, [power laws](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones."), the largest
and the smallest, scaling, geometry, how worlds break, pictures, cooperation.
They closed with [Questioning the mechanics](questioning-the-mechanics.md),
which questions the rules themselves, and with what all of it means for
open-ended evolution: the
[next section](#what-the-measurements-say-about-open-ended-evolution).

Every number of these pages is a number about B1, with nothing to hold it
against. The notebook asks for a null beside every number, and the most
important one was missing: a world in which the brains make no difference.
So the control came first. [Do the brains matter?](do-the-brains-matter.md)
runs worlds whose brains never change, and worlds in which every decision is
drawn at random.
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
then asked, at the author's request, how a world's size follows its tokens.
[Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens) made the thirty worlds
this edition is written from, and asked of them the question they were made
for: how a line takes over a world ten times the size.

Three pages were planned next: on the rate of [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype."), on coalitions
switched off, and on the size of a brain. After
[What the brains are like](what-the-brains-are-like.md) the brain moved up,
since a brain that cannot see was the first thing to change. What happened
when it changed is told in [The road from here](#the-road-from-here). The
rate of mutation became [How fast should brains change?](how-fast-should-brains-change.md),
coalitions switched off are among
[The switches the world was built with](the-switches-the-world-was-built-with.md),
and *How big should a brain be?* is still to come.

## What the measurements say about open-ended evolution

The pages on B1 measured its world in every way the viewer can, and a few
more. In the second edition they read thirty worlds of 100,000 tokens, with
two new pages: [What crosses a connection](where-the-tokens-flow.md#what-crosses-a-connection) and
[The largest and the smallest](the-largest-and-the-smallest.md). This section
steps back and asks what all of it means for the question the notebook
exists for: **can open-ended evolution happen in Graph of Life, and what
would it take?**

It is written the way a researcher keeps a notebook: what we know, what we
think it means, what we proposed to do about it, and how we would know
whether it worked. Most of the proposals have been tried since; each says
what came of it.

### What we are looking for

**Open-ended evolution** is evolution that does not run out of new things to
make. Life on Earth has it: four billion years in, it still produces new
kinds of organisms, new ways of living, new levels of organisation. Almost
every artificial system stops. It finds a way of doing well, fills the world
with it, and from then on only reshuffles
([The ideas this builds on](the-ideas-this-builds-on.md)).

[What this notebook is about](what-this-notebook-is-about.md#a-ladder-of-four-steps)
sets out a ladder of four claims, each needed before the next:

1. **P1, there is evolution:** [lineages](families.md "A line of descent: the agents whose genotypes go back to one ancestor.") last long enough for selection to act
   on them, and selection changes the outcome.
2. **P2, adaptation accumulates:** a late population would beat its own
   ancestors.
3. **P3, there are organisations:** groups that persist while their members
   come and go.
4. **P4, the space of organisations keeps growing**, without a ceiling.

The project's research plan (`research/Research.md`) aims at the top of the
ladder. It defines open-endedness as unbounded growth in the space of
**multi-agent organisations**: structures made of agents, which persist and
are reproduced. The reason is a counting argument. Picture a brain as a panel
of dials. It has a fixed number of weights, so however long evolution turns
them, it searches one fixed box. Agents are different: the number of ways to
arrange *n* agents grows faster than any exponential in *n*. Just to wire *n*
agents together there are 2^(*n*(*n* − 1)/2) networks, one for every choice of
which pairs are joined. If anything here can grow without bound, it is how
agents are arranged and what they do to each other.

### The baseline world, in one paragraph

Here is what the measurements found, put together. **Tokens** move almost as
a random walk would, for the reason
[Brains that cannot see](#brains-that-cannot-see) gives; they run downhill
from rich neighbourhoods to poor ([Gains and losses](gains-and-losses.md)),
and never circulate. **Connections** have a price. Two connections in three
carry one or two tokens a game, most of them one each way: an equal exchange
that keeps the connection alive at its price of a token a game. Every
connection that carries nothing is cut
([What crosses a connection](where-the-tokens-flow.md#what-crosses-a-connection),
[Do agents cooperate?](do-agents-cooperate.md)). **The network** is a
stretched web of stars, grown locally by children joining their parents'
neighbourhoods. It is four and a half times as long as a random network with
the same connections, with a thirty-third of its [spectral gap](spectral-gap.md "How hard a network is to cut into two halves, and how fast a walker on it forgets where it began.")
([The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world)). It breaks along its
poor regions, whose borders go quiet. Then the [cull](one-iteration.md#cleanup),
the rule that only the largest piece of the network lives on, kills whatever
fell off ([How a world breaks](how-a-world-breaks.md)). **Brains** spread by
conquest seventeen times more than by birth, and change mostly in place, not
when copied ([Questioning the mechanics](questioning-the-mechanics.md#two-roads-of-inheritance)).
The brain on a hub is replaced almost every game, and the same two brains
rarely face each other twice ([Where the tokens flow](where-the-tokens-flow.md),
[Do agents cooperate?](do-agents-cooperate.md)). **Relatives** live together,
but cannot recognise each other, and do not treat each other differently
([Do agents cooperate?](do-agents-cooperate.md)). Almost every structural
statistic is the same in every world: it follows from the rules, not from
what brains compute.

The short version: **a world of places, not individuals.** The stage, the
network and the landscape of wealth, is stable and made by the rules. The
actors, the brains, change almost every game.

### Where the world stands on the ladder

**P1, evolution: partly.** There are lineages, and they matter. Lines take
whole worlds over
([One slow wander, in everything](#one-slow-wander-in-everything)).
Families occupy regions that widen as their line ages, about as a random walk
spreads ([One world, many colours](one-world-many-colours.md),
[A space, not a small world](#a-space-not-a-small-world)). And worlds whose
brains never change end up in very different places, while worlds that
evolve all end up alike ([Do the brains matter?](do-the-brains-matter.md)).

But heredity is weak. In B1 a brain is changed with probability 0.2 after
every game, whether or not it is copied, and a brain is overwritten whenever
its node is won. The project's [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") measurements (`research/pilot_heredity.py`,
at a higher mutation rate) found a lineage 45% of the way to a stranger after
one iteration, and 92% after ten. Selection can only work on what is
inherited, and here little is inherited for long.

**P2, accumulation: not in B1.** The test is a tournament across time. Bring
back the brains of an early iteration, say 500, and set them against those
of a late one, say 2,500, in the same world (Cliff and Miller 1995). If
adaptation accumulates, the late brains win. When this section was first
written, nothing could answer it.
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
then ran the tournament on ten B1 worlds of 5,000 tokens (Experiment 12).

Every world learned fast. By iteration 50 its brains beat its founders', by
+361 Elo points in the median world, and by +400 at iteration 100. (A rating
of +400 means winning about ten contests in eleven.) Then nothing more. The
brains of iteration 1,000 held 0.483 of a world contested with those of
iteration 300. Every [era](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") tied with the one before it (0.47 to 0.52). A
dominance tournament, which crowns an era only if it beats every earlier
champion, crowned no more champions than chance. Contests five times as
long, 100 iterations, agree: the brains of iteration 1,000 held a median
0.49 against those of 500. In B1, adaptation accumulates for fifty to a
hundred iterations, and stops.

In B2 it goes on longer. Judged by the same contests of 20 iterations, every
pilot for B2 seemed to stop within a few hundred iterations too, whether its
brains had a memory or could grow ([Why evolution stops](why-evolution-stops.md)).
But a contest must last as long as what it judges, and a B2 agent lives 100
iterations. Played a lifetime long, the brains of iteration 2,000 beat those
of 500 in all ten worlds of Experiment 16. In the three worlds played on to
10,000 iterations, the gains went on to about iteration 5,000, and then all
but stopped
([Would we notice if it went on?](why-evolution-stops.md#would-we-notice-if-it-went-on)).

**P3, organisations: not found.** The candidates exist: family regions, and
hubs with their stars. But nothing shows a group that persists while its
members change. The nearest thing is a star's **dynasty**: a centre joined to
thousands, passed from parent to newborn child for hundreds of iterations,
while the brain on it changes every game
([The largest and the smallest](the-largest-and-the-smallest.md)). It
persists while its members come and go. But it is a place, kept by the rules
of the even split, not a group that does anything together. The same two
brains face each other one more game in only one case in twenty-five; no
brain holds a position.

None has been found since. In the pilots for B2, connected groups of one
genotype and of close kin shrank over time in every world. In B1 the groups
that tokens keep to, the [flow modules](map-equation.md "A group of agents that a token, following the flow, tends to stay inside."), are the network's own regions. They
last nearly as long when every agent's partners are dealt at random among its
neighbours, and they outlast their first members only as a region outlasts
its occupants. B2's agents add one thing of their own: pairs of partners
that last a few games longer than chance would keep them
([Flow modules: the groups tokens stay in](flow-modules.md)).

**P4: out of reach** until P1 to P3 stand.

### What stands in the way

The measurements, read together, point at seven obstacles. The first is the
brains themselves; the other six are rules of B1. Four of the six are rules
that [Questioning the mechanics](questioning-the-mechanics.md) examines one at
a time. Here each obstacle is said in a few lines, with what B2 did about it.

**The first: brains that cannot see.** This obstacle was found last, by
[What the brains are like](what-the-brains-are-like.md), after the other six
were listed. But it comes before them. B1's brains barely respond to what
they see ([Brains that cannot see](#brains-that-cannot-see)), and every remedy
for the six needs brains that do: incumbency worth defending, tags that tell
kin, cooperation that pays. [What agents say to each other](what-agents-say-to-each-other.md)
adds that the tags already exist. A message is the name of its writer's
genotype, and kin could tell each other by it, if a brain could compare what
it reads. The remedy is in the architecture: fewer layers, a steeper
squashing, larger starting weights
([How a signal fades through layers](vanishing-signal.md)).

[Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)
tried fewer layers, as the first step towards B2. One hidden layer of 16
gives a founder a gain of about 0.2, against 0.0006 through B1's five layers.
The worlds broke. At 5,000 tokens, four of ten seeing worlds died out, and
eight of ten when their brains also changed only when copied, against none of
the ten B1 worlds. A brain that sees can stop having children: in six of the
twelve worlds that died, the last brains had no child at all. So sight needed
a world that cannot break, and B2 is that world.

The six rules:

1. **Variation in the living.** 97% of new genotypes are made in brains that
   are not being copied, so successful brains are worn away in place
   ([Variation in the living](questioning-the-mechanics.md#variation-in-the-living)).
   *In B2* a brain changes only when it is copied.
2. **Conquest erases.** A won node loses its brain entirely. An individual
   ends whenever its node is won, and on a hub that is almost every game
   ([Two roads of inheritance](questioning-the-mechanics.md#two-roads-of-inheritance)).
   *In B2* conquest still erases.
3. **Nothing can be held.** There is no incumbency, and there are no lasting
   partners. The "shadow of the future" that reciprocity needs, the chance of
   meeting the same partner again, is about 0.04 (0.05 at 10,000 tokens)
   ([Hubs are places](questioning-the-mechanics.md#hubs-are-places)).
   *In B2* a [home stake](home-stake.md "The tokens an agent stakes on its own node, to keep it.") counts 1.5 times
   ([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)). How often the same two brains
   meet again has not been measured there.
4. **Nothing is positive-sum.** Tokens are conserved, and every stake is a
   transfer: no interaction makes anything that was not there before. So an
   organisation can never be worth more than its parts, and there is no
   reason for one to form (`research/Research.md`, §3.3). *For B2* a yield on
   stakes returned, and support stakes, were tried and gave nothing that
   lasted ([Why evolution stops](why-evolution-stops.md)). Alliances, in which
   two neighbours that both want it count their stakes together, were taken
   up at once ([Alliances](alliances.md)).
5. **Nobody can tell kin from stranger.** Kin selection, help given to
   relatives who carry the same genes, has its precondition: relatives live
   together. But brains cannot see relatedness
   ([Do agents cooperate?](do-agents-cooperate.md)). *In B2* brains that see
   could tell kin by the names in their messages
   ([What agents say to each other](what-agents-say-to-each-other.md),
   [Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break));
   whether they do has not been measured. Given a sense of kin as an input,
   brains formed groups only where a rule pooled the stakes of kin
   ([Who helps whom](who-helps-whom.md)).
6. **The world can neither reach across nor split.** Growth is local only, so
   the network stretches, and its shortcuts are never renewed. The cull kills
   any part that separates, so populations can never diverge in isolation
   ([One rule reaches across the world](questioning-the-mechanics.md#one-rule-reaches-across-the-world)).
   [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
   adds that a bigger world is no escape: distances grow like a power of its
   size, and the regions the cull removes grow with it. *In B2* worlds cannot
   split at all ([A world that cannot break](a-world-that-cannot-break.md)),
   rather than living on apart, and long-range links were not adopted.

### What we propose to change

Each proposal is a rule that can be switched on, run against the baseline
with thirty seeds, and measured, as [Do the brains matter?](do-the-brains-matter.md)
does. Those marked with a § come from the research plan
(`research/Research.md`, §4); the others come from what the measurements
found. They are ordered by what has to come first, not by how interesting
they are. Under each, **What came of it** says what has happened since.

#### First: make heredity real

- **Mutation at replication, and nowhere else** (§4.1 of the plan). A brain
  changes only when it is copied, into a child or into a won node. A brain
  that is not copied stays itself. Its measure was to be how long a lineage
  stays like its ancestor, and whether selection then separates good brains
  from bad.
- **A germline** (§4.2), if that is not enough: every agent carries an
  unchanged copy of the brain it was born with, and passes that on.

**What came of it.** [Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)
tried mutation at replication with brains that see (Experiment 13, ten
worlds of 5,000 tokens). It made things worse: eight worlds died out instead
of four ([What stands in the way](#what-stands-in-the-way)). A stricter
variant, which changed a brain only when it was copied into a newborn, froze
a pilot world into one genotype with no births at all. B2 keeps the rule,
inside rules that stop worlds breaking and starving
([A world that cannot break](a-world-that-cannot-break.md),
[Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)). How long a lineage now stays like
its ancestor has not been measured. A germline has not been tried.

#### Second: let individuals last

- **Incumbency.** The agent on a node gets an advantage in defending it: its
  home stake counts double, or a coalition must outweigh the home stake too.
  Measure: how long brains hold hubs, and how long partners last.
- **Recombination at conquest.** A won node takes a mix of the winner's brain
  and its own, not a replacement. It is the closest thing to sex this world
  could have, and a way for an agent to survive in part when it loses.

**What came of it.** [`incumbency`](settings.md#incumbency) is now a setting.
In pilots with brains that see, a home stake counting double cut conquest
from 40–90% of the nodes a game to 15–25%, and births came back as the way
brains spread. But one agent gathered 57% of all the tokens, staked them all
at home where nobody could outbid them, and had child after child: a queen.
B2 uses 1.5, with a share of every agent's tokens kept at home, and in the
pilots no queen formed ([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)). How long
brains hold their nodes, and how long partners last, has not been measured.
Under B2's cap on links no agent is joined to more than a few per cent of the
others ([A world that cannot break](a-world-that-cannot-break.md)), so there
are no hubs to hold. Recombination at conquest has not been tried; brains
have been mixed only at birth, a newborn's brain mixing its parent's with a
neighbour's ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md)).

#### Third: let cooperation pay, and let kin find each other

- **Mutual flow yield** (§4.4). A connection across which tokens flow in both
  directions in a game yields a small number of new tokens, split between its
  ends. A small decay of every holding keeps the world bounded. Now two
  agents together can be worth more than two apart, and there is a defection
  problem to solve, which is what cooperation theory is about.
- **Heritable tags.** Every brain carries a few numbers, a tag, copied with it
  and changed rarely; every agent sees its candidates' tags among its inputs.
  Tags are how cooperation can evolve without memory or reputation (Riolo,
  Cohen and Axelrod 2001), and they give relatives a way to recognise each
  other.

**What came of it.** A yield on stakes returned, paid for by a small tax on
holdings, was tried in the pilots for B2
([Why evolution stops](why-evolution-stops.md)). It went to whoever had the
most partners, and built stars rather than partnerships. Evolution stopped as
in every other pilot judged by short contests. Cooperation needs a surplus
that pairs, not hubs, can earn. The messages already serve as tags
([What agents say to each other](what-agents-say-to-each-other.md)). A family
tag, inherited and changed now and then, was later tried in pilots
([Another game](another-game.md)). With a sense of family alone, the fighting
stayed as it was; only where a rule pooled the stakes of a family did
families of hundreds form, rising and falling like empires.

#### Fourth: open the world

- **Proposed connections** (§4.5). Two agents anywhere in each other's
  neighbourhood of neighbourhoods may form a connection if both want it, at a
  cost; and, rarely, a child is joined to an agent anywhere in the world. The
  world's shortcuts would be renewed instead of spent.
- **Let separate pieces live.** Drop the cull: a piece that separates becomes
  its own world, with its own tokens, and may meet the main world again
  through the long-range connections above. Isolation, divergence and
  reunion, the oldest road to new species, would become possible.
- **Several kinds of tokens** (§4.3): tokens of different colours, which
  agents value differently by inheritance. Niches, and a reason to trade.

**What came of it.** The world closed instead. In the pilots for B2 an agent
could pay tokens for a short walk through the network, and join an agent it
met if both agreed. Few links were made, one to ten an iteration, and little
changed. Later, [Links to strangers](links-to-strangers.md) let agents search
further. When a search cost tokens, hardly anyone paid for one; when it was
free, agents reached all the time. Either way the median proposal went to an
agent only two or three steps away. Yet every searching world evolved faster
than B2 did. B2 keeps every rule local
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md),
[Physical inspiration](physical-inspiration.md)), so no child is joined to an
agent anywhere. And it took the other road from letting pieces live:
[A world that cannot break](a-world-that-cannot-break.md) adds local rules
under which worlds cannot split at all. A piece that lives on as a world of
its own is planned for *Local rules only*. Several kinds of tokens have not
been tried.

#### Later: let the possible grow

Brains that can grow (§4.6), agents that set some of their own rules (§4.7),
a second heritable structure that belongs to a relationship rather than an
agent (§4.8), and, at the very end, groups that copy themselves (§4.9): each
only once the earlier steps hold.

**What came of it.** In the pilots for B2 a brain could gain or lose hidden
units by mutation, beside a "shadow" size that nothing reads
([Why evolution stops](why-evolution-stops.md)). Where parents could not hand
links to their children, as in B2, the real size followed its shadow. What
rose, rose because a size cannot fall below its floor, the left wall, and not
because selection favoured larger brains.
Evolution stopped as in every other pilot judged by short contests. §4.7 to
§4.9 have not been tried.

Three smaller repairs from [Questioning the mechanics](questioning-the-mechanics.md)
can go along with any of these: a smooth [share function](share-function.md "How a brain's two numbers become one share between 0 and 1: of tokens for a child, or of a stake marked revolutionary.")
([The share function jumps](questioning-the-mechanics.md#the-share-function-jumps)),
a separate output for whole-agent decisions
([Decisions averaged over neighbours](questioning-the-mechanics.md#decisions-averaged-over-neighbours)),
and the option to hold tokens back from the game
([Every token at stake, every game](questioning-the-mechanics.md#every-token-at-stake-every-game)).
**What came of them.** Only something like the last has been tried.
[`keep_home`](settings.md#keep_home) keeps a share of every agent's tokens on
its own node, so that no agent can stake itself to death
([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)): held at home, not out of the
game. The share function and the averaging of whole-agent decisions are
unchanged.

### How we will know

A rule change is only progress if a measurement says so, against a null. The
measures, by rung of the ladder:

- **For P1.** The heredity half-life: how far a lineage's brains drift from
  their ancestor's in a given time, in weights and in behaviour, against
  unrelated pairs. And selection ablation: the same world with the winner of
  every node drawn by chance among its stakers. If the real rule does no
  better, selection is not doing the work.
- **For P2.** The cross-time tournament: earlier against later brains, in
  the same world. A clean gradient, later beating earlier, is accumulation. A
  pattern of bands is cycling, as rock–paper–scissors would give.
- **For cooperation**, the measures of
  [Do agents cooperate?](do-agents-cooperate.md): kin discrimination;
  Hamilton's accounting Σ(*rB* − *C*), which adds up, over every helping act,
  the benefit *B* to the one helped, weighted by how related *r* the two are,
  minus the cost *C* to the helper; reciprocity beyond the even split; the
  lifetimes of partnerships; coalitions that repeat; and whether regions held
  by one family outlast mixed ones.
- **For P3.** Groups that persist while their members turn over, found by
  methods that do not assume them: flow modules (Rosvall and Bergstrom 2008),
  or groups that carry information about their own past into their future
  better than their surroundings do (Krakauer et al. 2020). Always against a
  network rewired at random with the same connections, which the measurements
  showed is very different from the real one.
- **For P4.** Bedau and Packard's evolutionary activity (1992): the
  cumulative presence of components that persist, set against a "neutral
  shadow", a run in which the same events happen but no component has any
  advantage. And the MODES measures of change, novelty, complexity and
  ecology (Dolson, Vostinar, Wiser and Ofria 2019), each with a filter that
  only counts what persists. Open-endedness is the curves that never level
  off, above their shadows.

Some of these have been used since. The cross-time tournament is built and is
B2's yardstick, with a dominance tournament held against its chance null
([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).
It taught one lesson on the way: a contest must last as long as what it
judges, in B2 an agent's lifetime. A neutral shadow judged brains that can
grow ([Why evolution stops](why-evolution-stops.md)). Flow modules have been
read in worlds of every kind the notebook has made
([Flow modules: the groups tokens stay in](flow-modules.md)). The heredity
half-life, selection ablation, the cooperation measures of
[Do agents cooperate?](do-agents-cooperate.md) and evolutionary activity have
not yet been used in the notebook. Help has been counted instead, exactly by
the rules ([Who helps whom](who-helps-whom.md)).

### What the measurements changed in what this notebook teaches

Findings change the teaching, and the earlier pages have been brought up to
date where they did:

- **Wealth follows connections because of a law**, not a mystery: the even
  split's resting state. [Where do the tokens go?](where-do-the-tokens-go.md)
  now says so, and [Where the tokens flow](where-the-tokens-flow.md) derives
  it.
- **"Keeping one's node" is about places**: hubs are nearly always taken
  ([How the game is played](how-the-game-is-played.md)).
- **The network's shape has a mechanism**: copying neighbourhoods at birth
  gives the stars, the triangles and the hierarchy
  ([What shape does the network take?](what-shape-does-the-network-take.md),
  [How properties scale together](how-properties-scale-together.md)).
- **"An agent" has two meanings**, the node and the brain, and the notebook
  now says which it means where it matters
  ([Questioning the mechanics](questioning-the-mechanics.md#two-roads-of-inheritance)).

### Questions for the next analyses

Some questions needed no new runs, only new looks at the old ones. Two have
been answered since, and one in part.

- **Do family regions persist as organisations?** A region held by one
  lineage for hundreds of iterations, while its nodes turn over, would be a
  first candidate for P3, measurable against a null that shuffles lineages.
  [Like next to like](like-next-to-like.md) found genotypes clustered within
  two or three steps, and no territories. Lines, which last longer, sit in
  patches that widen as they age
  ([A space, not a small world](#a-space-not-a-small-world)). Whether a patch
  is ever more than the sum of its agents is still open.
- **What does a brain respond to?** Feed a stored brain the same inputs with
  one changed, a neighbour's tokens or a message, and see what its stakes do.
  If brains mostly tell "me" from "others" and little else, the even split is
  no surprise. [What the brains are like](what-the-brains-are-like.md) did
  this. Doubling every neighbour's tokens changes the stakes of 1.5% of
  agents; setting every message to 0 changes those of 9%. Brains respond to
  almost nothing.
- **Where are the old?** At 10,000 tokens a picture suggested the edges. At
  100,000 the old sit in clumps of their own
  ([One world, many colours](one-world-many-colours.md)), as cohorts
  ([Like next to like](like-next-to-like.md)). Age against distance from the
  hubs would say; it has not been measured.
- **Why do worlds reproduce less as they age?** Births per agent fall over a
  world's first five hundred iterations and then, in a large world, hold
  still ([A turnover that settles late](#a-turnover-that-settles-late)).
  Founders reproduce twenty times as often as settled agents
  ([How agents have children](how-agents-have-children.md)). That is a
  candidate for an adaptation, the clearest one the measurements saw, but one
  that is finished early. The cross-time tournament dates it: B1's brains
  beat their founders by iteration 50 and improve no further. And the births
  probe of [Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)
  puts the change in the brains. Put back into their founders' world, the
  founders' own brains have a child in about three agent-iterations in four;
  B1's later brains, in 3 to 9 in a hundred.

### What the second edition changed

Redoing the study of B1 on worlds ten times larger left most of it as it
was, and that is itself a finding. Per agent, a world of 100,000 tokens does
what a world of 10,000 does: the same births and deaths within a tenth, the
same inequality, the same game, the same brains
([Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)). What changed is how
the agents are arranged, and how fast anything crosses a world:

- **The shape changes.** Half the [clustering](clustering.md "How often an agent's neighbours are also each other's neighbours."), more loops per agent, fewer
  bridges, distances twice as long, and a ball dimension that rises with size
  ([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md),
  [The core of a world](what-shape-does-the-network-take.md#the-core-of-a-world)).
- **Stars can take a world.** One world in four condensed around a single
  centre, and then everything about it, its core, its game, its cuts, follows
  from the star ([Stars](#stars),
  [The largest and the smallest](the-largest-and-the-smallest.md)).
- **Lines take longer.** A takeover by one founder's line takes six times as
  long in a world ten times larger. So lines live side by side for most of a
  run, and the messages of a world speak more dialects
  ([What agents say to each other](what-agents-say-to-each-other.md),
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)).
- **Demography settles.** In a large world, ages, lifetimes and births per
  agent hold still from iteration 500 on
  ([A turnover that settles late](#a-turnover-that-settles-late)).

For open-ended evolution the larger world is the better stage. It has more
room and more time for lines to differ, and the first lasting inherited
structure, the dynasty. But it does not remove a single obstacle: every one
of them is a rule, and the rules are the same at both sizes. So the proposals
were planned at 10,000 tokens, where a run is eight times cheaper. The first
experiments on the way to B2, Experiments 12 to 14, ran smaller still, at
5,000. Those that bear on space and organisation, P3 and the physics of
space, are to be checked at 100,000.

### The road from here

When this section was written, the plan was this. First, experiments with
single settings: the rate of mutation, coalitions switched off, the size of
the brain, the size of the world. They are cheap, and each teaches how the
world works. Then rule changes, in the order above: first heredity, then
lasting individuals, then cooperation that pays and kin that know each
other, then an open world. Each was to be measured with the instruments
above.

The experiments did not keep this order. The notebook built its yardstick
first ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).
Then it gave the brains sight, with change only when copied
([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)),
and the worlds broke. So it built rules that keep a world whole
([A world that cannot break](a-world-that-cannot-break.md)) and keep agents
from staking or breeding themselves to death, with an incumbent's advantage
of 1.5 ([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)). Together these rules are
the second baseline, B2. In B2 evolution goes on for thousands of
iterations, more slowly as it goes, and then levels off
([Why evolution stops](why-evolution-stops.md)). What stops it is not that
the game has no better play left: a strategy written by hand took half of a
B2 world ([Strategies written by hand](strategies-written-by-hand.md)). The
search runs out before the game does. What moved since is told in
[Healthy and rich](healthy-and-rich.md) and [Rules in company](rules-in-company.md).

It is a long way. But the measurements have made it shorter in one important
sense. They turned a vague hope, that interesting things might emerge if we
watch long enough, into a list of specific rules that stand in the way, each
of which can be changed and tested on its own.

The notebook's second aim points at the same rules.
[Physical inspiration](physical-inspiration.md), written after this section,
sets out that aim: a world that grows a space, and stable things in it. The
cull and the [share-out](gains-losses-and-the-share-out.md "How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random."), which this page questions for evolution's sake, also
break locality, the speed limit physics insists on. And the organisations of
P3 are what a physicist would call particles.
[Space and dimension](../README.md#space-and-dimension) follows that aim, and
the rule changes of both aims will be tested together where they coincide.

## What we learned

- **B1 is a world of places, not individuals.** Its network and its landscape
  of wealth follow from the rules and are alike in every world. The brains on
  its nodes change almost every game.
- **Its brains are nearly blind**, so selection sets only a few constants,
  the same in every world, within the first fifty to a hundred iterations.
  After that, B1 does not improve.
- **Six of six theses failed**, at both sizes. Per agent, a world ten times
  larger does what a small one does; what changes is its shape, its stars and
  the pace of its lines.
- **On the ladder**, B1 has evolution of a weak kind (P1, partly), no
  accumulation after its first boom (P2), and no organisations (P3).
- **Seven obstacles stood in its way**: blind brains first, then six rules.
  Sight alone broke the worlds. B2 gave the brains sight inside a world that
  cannot break, and changed three of the six rules; there evolution goes on
  for thousands of iterations before it slows to a plateau.

## Where to go next

- [Questioning the mechanics](questioning-the-mechanics.md): B1's rules one
  at a time, and what B2 did about each.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  the yardstick for evolution, and how B1 fared by it.
- [Why evolution stops](why-evolution-stops.md): whether a world that cannot
  break keeps evolving.
- [Thirty worlds](a-worlds-life.md#thirty-worlds): the thirty B1 worlds behind every number
  here.

## How our view changed

- **9 October 2026.** The list of what stands in the way named six obstacles,
  all of them rules. [What the brains are like](what-the-brains-are-like.md)
  then found B1's brains nearly blind, with a gain of 0.0015. That put a
  seventh obstacle before the six, since every remedy for them needs brains
  that respond to what is around them.
- **11 October 2026.** The study of B1 said that a world has room for regions
  that differ, "and nothing to make them".
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens) found that lines of
  descent do make patches, for a while. What is missing is something to keep
  them apart.
- **14 October 2026.** The study said that defence got better without agents
  staking more at home, and that what changed instead was not known. In the
  worlds of 100,000 tokens agents did stake more at home, but that is not the
  cause: defence got better because the stakes became even
  ([Where the tokens flow](where-the-tokens-flow.md)).
- **14 October 2026.** The plan was to try single settings first, then the
  rules in the order of the proposals. The notebook built its yardstick
  first, then gave the brains sight, and the worlds broke. So it had to build
  a world that cannot break, B2, before anything else could be tried
  ([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break),
  [A world that cannot break](a-world-that-cannot-break.md)).
- **14 October 2026.** Mutation only when copied was proposed as the first
  step to real heredity. Tried with brains that see, it did harm: eight worlds
  in ten died out instead of four. B2 keeps it, but only beside rules that
  stop worlds breaking and starving.
- **14 October 2026.** Incumbency was proposed as a home stake that counts
  double. At double, one agent gathered 57% of all the tokens: a queen. B2
  uses 1.5, with 30% of every agent's tokens kept at home
  ([Who dies, and why](a-world-that-cannot-break.md#who-dies-and-why)).
- **14 October 2026.** A yield on mutual flow was proposed to make
  cooperation pay. A yield on stakes returned went to whoever had the most
  partners, and built stars, not partnerships
  ([Why evolution stops](why-evolution-stops.md)).
- **14 October 2026.** The proposals would have opened the world: links to
  agents anywhere, and separated pieces that live on. B2 kept every rule local
  and made worlds that cannot split instead
  ([A world that cannot break](a-world-that-cannot-break.md)).
- **14 October 2026.** Blind brains were called the first obstacle, sitting
  "in the architecture, not in the world". But sight was not enough: in pilot
  worlds that could not break, seeing brains stopped improving within a few
  hundred iterations too. So [Why evolution stops](why-evolution-stops.md)
  proposed that the obstacle lay in the game, not in the brain: a contest
  best played unpredictably, which evolved brains play by using their random
  inputs as dice.
- **15 October 2026.** Both halves of that reading were corrected. Contests
  an agent's lifetime long show that in B2 evolution did not stop: the brains
  of iteration 2,000 beat those of 500 in all ten worlds of Experiment 16,
  and the gains went on to about iteration 5,000. And the dice were
  inherited: random founder brains already roll them, and evolved brains roll
  them as much in a contest where dice are worthless. A strategy written by
  hand took half of one world, so the game has better play left; the search
  runs out before the game does
  ([Why evolution stops](why-evolution-stops.md),
  [Strategies written by hand](strategies-written-by-hand.md)).

<!-- turns -->
---

← [B1 and B2: the two baselines](b1-and-b2.md) · [The baselines B1 and B2](../README.md#the-baselines-b1-and-b2) · [The life of a B1 world](a-worlds-life.md) →

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (21)
> [What this notebook is about](what-this-notebook-is-about.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [How a brain changes](mutation.md) · [The life of a B1 world](a-worlds-life.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [How births build the network](how-agents-have-children.md) · [The estate of the dead](estate.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses](gains-and-losses.md) · [Where tokens flow](where-the-tokens-flow.md) · [Power laws, real and apparent](power-laws-real-and-apparent.md) · [How properties scale together](how-properties-scale-together.md) · [How a world breaks](how-a-world-breaks.md) · [Stars and hubs](stars-and-hubs.md) · [Physical inspiration](physical-inspiration.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [What the brains are like](what-the-brains-are-like.md) · [What agents say to each other](what-agents-say-to-each-other.md) · [Do the brains matter?](do-the-brains-matter.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Do agents cooperate?](do-agents-cooperate.md)
<!-- /turns -->
