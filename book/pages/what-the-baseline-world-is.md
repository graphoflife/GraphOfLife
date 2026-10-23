# What the baseline world is

The pages on the [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") look at thirty baseline worlds from every side. This page puts
the pieces together: what the theses predicted, what was found instead, what
seems to hold across the pages, and what that changed in how the notebook
measures. It was first written after Experiments 1 to 5, on worlds of 10,000
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), and brought up to date as more were written. In this second edition they
read thirty worlds of 100,000 tokens ([Thirty worlds](thirty-worlds.md)), and
two of the pages are new: how ages change over a world's life
([How ages change over a world's life](how-ages-change.md)) and the core of a world
([The core of a world](the-core-of-a-world.md)).

> [!question] Questions of this page
> - What is a settled baseline world, in numbers?
> - What do the pages on the baseline, taken together, suggest about how it
>   works?
> - What changes when a world is ten times larger, and what does not?
> - What do the experiments still need — and in what order?

## The record

Each thesis was written before the runs of 10,000 tokens and tested on them;
the second edition tests it again on the worlds of 100,000.

- [Is a run reproducible?](is-a-run-reproducible.md) —
  thesis **refuted**, but only by the number of threads of the matrix
  library, in the last bits. Every run now uses one thread, and is then
  exactly reproducible.
- [A world's life](a-worlds-life.md) — thesis **refuted**
  at both sizes. At 10,000 tokens births fell by 41% and 4 of 30 worlds died;
  at 100,000 none died, the typical world held its size, and only the fall of
  births (28%) broke the thesis.
- [How much does the seed decide?](how-much-does-the-seed-decide.md) — thesis **refuted, narrowly**, at both sizes. Little: worlds
  differ mostly because each one wanders, not because of their [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.").
- [Where do the tokens go?](where-do-the-tokens-go.md) —
  thesis **refuted** at both sizes. Inequality is moderate (a [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") just under
  0.5) and steady, and [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") stake 70% of their tokens on their [neighbours](the-world.md#the-network "An agent joined to another by a connection.").
- [What shape does the network take?](what-shape-does-the-network-take.md) — thesis **refuted**, narrowly at 10,000 tokens (a core of
  53%), more clearly at 100,000 (57%). Part tree, part web.
- [Genotypes and lineages](genotypes-and-lineages.md) —
  thesis **refuted** at both sizes. [Genotypes](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") last two [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."); lines take
  the world over — again and again in a small world, about once a run in a
  large one.

Six of six theses failed, and none of the failures changed with the size of
the world. That is what a thesis written before its runs is for: each failure
is something about the world that was not known before.

## The baseline world in numbers

A settled baseline world of 100,000 tokens — each number the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") over the
thirty worlds of each world's mean over iterations 500 to 1,499, with the
worlds of 10,000 tokens in brackets where they differ:

- **about 14,400 agents** (1,300) with **3.5 [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.")** each (3.3)
  ([A world's life](a-worlds-life.md), [What shape does the network take?](what-shape-does-the-network-take.md));
- **3.5 births** per hundred agents per iteration (3.1), and as many deaths,
  three in four of them agents cut off from the network
  ([Births, deaths and ages](births-deaths-and-ages.md));
- ages that **settle by iteration 500**: 2.4% of the living are newborns, half
  are younger than 55 iterations, and the risk of dying falls with age to
  about one [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") in a hundred, with no ageing; the ages of the living follow
  from the [lifetimes](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death.") as in a stationary population
  ([How ages change over a world's life](how-ages-change.md));
- a **Gini coefficient** of wealth of **0.47** (0.49), the [richest tenth](richest-tenth.md "The share of all tokens held by the richest tenth of the agents.")
  holding **42%** (44%) ([Where do the tokens go?](where-do-the-tokens-go.md));
- **44%** of agents keep their node in a game (45%); **half** of all nodes are
  won by [coalitions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") ([How the game is played](how-the-game-is-played.md));
- **26%** of connections [bridges](bridges.md "A connection that lies on no loop: cut it and the network falls into two pieces.") (31%), **35%** of agents leaves (36%), **57%**
  in a core of [loops](loops.md "A path through the network that comes back to where it started without using a connection twice.") (54%), **18 steps** from one agent to another (10)
  ([What shape does the network take?](what-shape-does-the-network-take.md));
- a genotype for every **1.6 agents**, each lasting a couple of iterations;
  lines that bring a whole world down to one [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s descendants by about
  iteration 1,000 (175 at 10,000 tokens)
  ([Genotypes and lineages](genotypes-and-lineages.md));
- [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") with a **gain of 0.0015**: a change in what a brain sees reaches
  what it does at a seven-hundredth of its size. Selection has set their
  constants alike in every world — a child in 3.6% of cases, the stake
  spread in 98% — and in the median world 42% of agents score every
  [candidate](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.") at zero or below, so the rules split their tokens evenly
  ([What the brains are like](what-the-brains-are-like.md));
- [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read.") that are **names**: one agent's messages to two readers differ
  by 0.0005, two agents' by 0.6, and agents of one genotype write the same
  ([What agents say to each other](what-agents-say-to-each-other.md));
- an order of wealth that is **half forgotten in fifty to a hundred games**,
  and rich agents with one connection that die three times as often as other
  leaves ([Do the rich stay rich?](do-the-rich-stay-rich.md));
- connections gained in proportion to the connections an agent has
  (k^1.02) and lost faster (k^1.20), and **no new connection ever
  bringing two agents closer** ([How the network grows](how-the-network-grows.md));
- likeness that fades within **two or three steps** — tokens, age,
  connections and kin alike ([Like next to like](like-next-to-like.md));
- a **finite [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.")**: 2.8 by the room within *r* steps (2.3 at 10,000
  tokens, 3.3 in worlds of 60,000 agents), about 1.9 by how a walker spreads,
  where random networks with the same connections have none
  ([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md));
- a **core** of three agents in five at the start of a game (57% after it,
  as above), holding three quarters of the tokens and every loop, a little
  lower in dimension by balls than the whole world (2.5) and the same by
  walks ([The core of a world](the-core-of-a-world.md)).

## What seems to hold across the pages

### One slow wander, in everything

[How much does the seed decide?](how-much-does-the-seed-decide.md) measured how long a world
remembers its level, statistic by statistic. They forget at much the same
pace: a hundred iterations later, a world's level is still clearly alike
([correlations](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of 0.52 to 0.69 for the number of agents, connections,
inequality, families and the core; 0.36 for births); three hundred later,
nothing is left (−0.06 to 0.08). Size, wealth and shape do not seem to wander
each on its own schedule; they look like one slow process seen through many
windows.

In the worlds of 10,000 tokens, [Genotypes and lineages](genotypes-and-lineages.md)
found a process with about that rhythm: every 350 iterations or so, one
branch of the family tree outlived all the others and the whole world
descended from it, and the notebook asked whether the wander was the sweeps. The
large worlds answer no. They forget their level within three hundred
iterations, as the small ones do, but a line needs about a thousand to take
one over ([Worlds of 100,000 tokens](worlds-of-100000-tokens.md)): the wander is faster
than the sweeps, and so not made by them.
[Do the brains matter?](do-the-brains-matter.md) finds the other half of
the answer: a world whose brains never change hardly wanders at all.

### A world of attack

Agents put only 29% of their staked tokens on their own node
([How the game is played](how-the-game-is-played.md)), more than half of all nodes
change hands in every game, and coalitions of smaller stakers take half of
all nodes. Over the first five hundred iterations, the share of agents that
keep their node rose, from 18% in the first game to 44%. The share of staked
tokens put at home rose too, from 13% to 29% — but at 10,000 tokens it
barely moved (25% to 30%) while defence got better all the same, so staking
more at home is not what changed.

> [!info] Added later: why defence got better
> [Where the tokens flow](where-the-tokens-flow.md) answers it. Under an [even split](even-split.md "What the rules do when a brain scores no candidate above zero: the stake is spread evenly.")
> the agent on a node with *d* connections is one of *d* + 1 nearly equal
> stakers there, and it keeps its node about one game in *d* + 1 — 74% of
> games with one connection, 2% with fifty or more, as measured. The
> founders went all in half the time, and in the first game 43% of them put
> nothing on their own node; by iteration 500, 98% of agents spread their
> stake ([How the game is played](how-the-game-is-played.md)). Defence got better
> because the stakes became even, not because anyone chose to defend.

### A turnover that settles late

Births fall long after a world's size has settled — by 28% from iterations
100–200 to the end in a large world, by 41% in a small one — and with them
the ages of the living rise. But in a large world the change is over by
iteration 500: from then on its births per agent, its lifetimes and its ages
hold still, as a stationary population's do
([How ages change over a world's life](how-ages-change.md)). The small worlds went on slowing until
about iteration 1,250, and their living grew older than their lifetimes
alone would make them. Neither is the lasting trend that the second rung of
the ladder in [What this notebook is about](what-this-notebook-is-about.md) looks for —
adaptation that accumulates. The demography of a world is finished early.

> [!info] Added later: the second rung, measured
> [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) played the
> brains of different moments of a world against each other, in ten baseline
> worlds of 5,000 tokens. Every world learned fast — by iteration 50 its
> brains beat the founders' (+361 Elo in the median world, +400 by iteration
> 100) — and then nothing: the brains of iteration 1,000 held 0.483 of a
> world contested with those of iteration 300, every [era](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") tied with the one
> before (0.47 to 0.52), and a dominance tournament crowned no more champions
> than chance. The baseline does not climb the second rung.

### Brains that cannot see

[What the brains are like](what-the-brains-are-like.md) opened the brains and found
them nearly blind. Five sigmoid layers pass a change in the inputs on at a
quarter of its size each, about a thousandth in all. Much of what these pages, and the measurements
after them, found follows from that one number:

- the stakes are nearly even, and the game moves tokens almost as a random
  walk would ([Where the tokens flow](where-the-tokens-flow.md)), down to the
  two-game rhythm of wealth that a random walk shows on [hubs](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") and leaves
  ([Do the rich stay rich?](do-the-rich-stay-rich.md));
- an agent's tokens are set by its place, connections plus one, and not by
  how it plays ([How properties scale together](how-properties-scale-together.md)); the rich
  are those the walk's resting state favours;
- messages carry a name and nothing else
  ([What agents say to each other](what-agents-say-to-each-other.md)), and kin, who could
  tell each other by it, cannot use it
  ([Do agents cooperate?](do-agents-cooperate.md));
- selection has acted, the same way in every world, but on constants: have
  few children, spread the stake, make every token revolutionary.

Evolution so far is the evolution of a few constants. That is the first
obstacle to [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau."), and it sits in the architecture, not in
the world.

> [!info] Added later: sight was not enough
> [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md) removed this
> obstacle: with one hidden layer of 16 a founder's gain is about 0.2 instead
> of 0.0006. The worlds broke instead. At 5,000 tokens four of ten seeing
> worlds died out, and eight of ten when brains change only when copied,
> against none of the baseline's: a brain that sees can stop having children,
> and in six of the twelve dead worlds the last brains had none at all. In
> [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") worlds that cannot break, seeing brains — with memory, with brains
> that can grow — stopped improving within a few hundred iterations too.
> [Why evolution stops](why-evolution-stops.md) proposes why: a contest like this
> game is best played unpredictably, the evolved brains use their random
> inputs as dice, and once a population plays that way there is no better
> strategy left to select. The obstacle would then lie in the game, not in
> the brain — a reading Experiment 16 is to test.

### A space, not a small world

New connections are only ever made inside a neighbourhood, and none brings
two agents closer ([How the network grows](how-the-network-grows.md)). So a world
grows as a body does. It has a finite dimension
([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)), its core is a
space of the same kind ([The core of a world](the-core-of-a-world.md)), and its
width grows with its size ([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)),
where a random network of the same connections would be a small world. Yet
nothing in the space keeps regions apart: likeness fades within two or three
steps ([Like next to like](like-next-to-like.md)). A world has the room for
regions that differ, and nothing to make them.

> [!info] Added later: lines hold patches
> [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) finds that lines of descent do
> hold regions: two neighbours share a family of 25 iterations back 99 times
> as often as chance, and the excess reaches about 7 steps; for families of
> 200 iterations back, about 12. But a patch widens as a random walk on these
> worlds would, until one line holds the whole world: nothing keeps patches
> apart.

### Stars

Seven of the thirty large worlds condensed, for long stretches, around a
single agent joined to a tenth of the world or more — in one, to more than
half ([How much does the seed decide?](how-much-does-the-seed-decide.md)). Such a world plays
its game differently ([How the game is played](how-the-game-is-played.md)) and has a
different shape: fewer agents in the core, far more leaves and bridges
([What shape does the network take?](what-shape-does-the-network-take.md)).
[The largest and the smallest](the-largest-and-the-smallest.md) shows that the centre of a
star is a place, handed down from parent to newborn child. Stars are not new
with size — ten of the 26 worlds of 10,000 tokens had one by the same
measure — but in a large world a star decides the world's whole shape: across
the thirty, the share of agents in the core follows the share joined to the
largest hub almost exactly ([How much does the seed decide?](how-much-does-the-seed-decide.md)).

> [!info] Added later: stars as a fault
> [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") counts a star as a fault of the network: the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") of
> [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) allows no agent
> to be joined to more than 5% of the others, and
> [A world that cannot break](a-world-that-cannot-break.md) keeps hubs down by drawing a
> newborn's links at random and letting a brain choose at most 8.

## What this changed in the method

- **The youth is left out.** Every thesis from here on is checked from
  iteration 500 on, the settled life of a run ([The settled life](settled-life.md)).
- **Long stretches, not the last fifth.** A run's mean over its settled life
  varies across seeds by much less than its last fifth does — for the number
  of agents 0.10 of the mean instead of 0.17 at 100,000 tokens, 0.17 instead
  of 0.25 at 10,000 — so it needs fewer than half the seeds
  ([How many seeds](seeds-needed.md)).
- **Thirty seeds see changes of about a tenth.** Over the settled life,
  thirty seeds per condition can see a change of 8% in the number of agents,
  9% in connections and 5% in inequality at 100,000 tokens, four times in
  five (12%, 14% and 5% at 10,000). Smaller effects are reported as not seen,
  not as absent.
- **The dead are counted apart.** About one world in eight dies at 10,000
  tokens — falls, that is, to 20 agents, which [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)
  later found is not always the end; none of thirty did at 100,000.
  Extinction is reported for every condition; only a large change in it —
  from 13% to 30% would take about ninety seeds per condition — can be seen
  ([When a world ends](extinction.md)).
- **Pairing by seed helps little.** Two conditions with the same seed share
  their starting ring and little else.
- **Lines, not genotypes.** A genotype lasts two iterations; what lasts,
  spreads and takes over is a line, and the pages to come follow lines.
- **Size is a [setting](settings.md "One number or switch of a run's rules, fixed before it starts.") like any other.** The per-agent numbers of a world
  hardly change from 10,000 tokens to 100,000; its shape, its distances, its
  extremes and the pace of its lines do ([Worlds of 100,000 tokens](worlds-of-100000-tokens.md)).
  A finding at one size is checked at another before it is trusted.

## What came next

The measurements of the baseline then read the same thirty worlds through every statistic the
viewer offers — [entropy](entropy-and-evenness.md "How evenly something is shared out among the agents, from all in one place to spread perfectly evenly."), gains and losses, the flow of tokens, what crosses
a connection, children, [power laws](power-law-fit.md "A quantity whose frequency falls as a power of its size: few huge values, many small ones."), the largest and the smallest, scaling,
geometry, how worlds break, pictures, cooperation — and closed with a page
questioning the rules themselves and a second meta page,
[What the measurements say about open-ended evolution](what-the-measurements-say-about-open-ended-evolution.md), on what all of it means for open-ended evolution.

Every number of these pages is a number about the baseline, with nothing to hold
it against. The rules of this notebook ask for a null beside every number, and
the most important one was missing: a world in which the brains make no
difference. So the control came first:
[Do the brains matter?](do-the-brains-matter.md) runs worlds whose brains never
change, and worlds in which every decision is drawn at random.
[How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md) then asks, at
the author's request, how a world's size follows its tokens. The pages
planned after them — the rate of [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype."), coalitions switched off, the size
of a brain — are listed in [the contents](../README.md). After
[What the brains are like](what-the-brains-are-like.md), the size and shape of a brain
moves up: a brain that cannot see is the first thing to change.
[Worlds of 100,000 tokens](worlds-of-100000-tokens.md) made the thirty worlds of 100,000
tokens this edition of the baseline's study is written from, and asked of them
the question they were made for: how a line takes over a world ten times the
size.

> [!info] Added later: the brain was changed in B2
> It came after [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
> had built a yardstick to judge it.
> [Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md) gave the brains
> one hidden layer of 16, and the worlds broke — which is why
> [A world that cannot break](a-world-that-cannot-break.md) and
> [Who dies, and why](who-dies-and-why.md) first build a world that cannot break,
> and [Why evolution stops](why-evolution-stops.md) asks whether evolution then
> keeps going. *How big should a brain be?*, on the size of a brain, is still to come.

<!-- turns -->
---

← [B1 and B2: the two baselines](b1-and-b2.md) · [The baselines B1 and B2](../README.md#the-baselines-b1-and-b2) · [Thirty worlds](thirty-worlds.md) →

> [!info]- Pages that link here (3)
> [Thirty worlds](thirty-worlds.md) · [The estate of the dead](estate.md) · [Do the brains matter?](do-the-brains-matter.md)
<!-- /turns -->
