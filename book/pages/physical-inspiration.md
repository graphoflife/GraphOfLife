# Physical inspiration

> [!question] Questions of this page
> - Which ideas from physics shaped the rules of Graph of Life?
> - Where does the algorithm follow them, and where does it break them?
> - What would it mean for a space, a particle or life to emerge here — and
>   how would one measure it?

## In short

The rules of Graph of Life were chosen with physics in mind: they conserve
the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), are the same everywhere, and are almost all local. The notebook's
second aim asks whether such rules grow, by themselves, a space of a few
dimensions and things that last in it. The worlds of the first baseline are
spaces: about three-dimensional by the room they hold, but two-dimensional to
a random walker. The second baseline's rules keep a world whole but fold it
into a near small world; nothing that lasts like a particle has been found.
Three rules still reach across a whole world at once; making them local is
what physics asks, and what [open-ended evolution](open-ended-test.md "Evolution that keeps producing new things without end, not only up to a plateau.") asked for too.

## The second aim

Graph of Life was not built to simulate our universe. But its rules were
chosen with physics in mind. The hope was that very simple rules, local and
conserving, might grow by themselves the kinds of thing physics describes:

- a space with a few dimensions;
- stable things that persist in it;
- and, in rare places, something that lives.

This is the notebook's second aim, beside open-ended evolution
([What this notebook is about](what-this-notebook-is-about.md)). As this page
will show, the two keep meeting.

The page teaches the physical ideas behind the rules, one at a time. For each,
it says where the algorithm follows the idea, and where it breaks it. Then it
asks the question this raises: a question the notebook can measure.

> [!warning] Analogies are questions, not answers
> An analogy says "this might work like that". It does not say that it does.
> Tokens are not energy, and [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") are not particles. A [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that "decides"
> is a function: in the first baseline it turns 154 numbers into 45 others
> ([What a brain sees](brain-inputs.md)). Every analogy below is offered as a
> way to ask a sharper question, and the page says where each one stops.

Two sets of rules recur on this page
([B1 and B2: the two baselines](b1-and-b2.md)). **B1** is the world as it was
first built: the original rules. **B2** is B1 with rules that keep its network
from breaking, and with smaller brains that see.

## Space made of relations

In everyday physics, space is the stage: things happen *in* it. Several
attempts to unite gravity with quantum theory turn this around. They build
space itself out of discrete pieces and the relations between them:

- causal sets (Bombelli, Lee, Meyer and Sorkin 1987);
- causal dynamical triangulations (Ambjørn, Jurkiewicz and Loll 2005);
- the hypergraph models of Wolfram (2020).

Space, in these pictures, is what a very large network looks like from far
away.

Graph of Life takes the same view. There is no space outside the network.
Agents are not *somewhere*; they are joined. The distance between two agents
is the number of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") on the shortest path between them
([Path length](path-length.md)), and nothing else.

A network suits this well, because it can take any shape: a line, a sheet, a
lattice of cubes, a tree, a tangle. Nobody chooses which shape a world takes:
it follows from the rules. So the first question is: **which rules make a
network that, seen from far away, looks like a space of three dimensions?**

### How to read a dimension off a network

A network has no coordinates, but it has distances, and distances are enough.
Here are four ways to measure a dimension. On a lattice of *d* dimensions,
each gives exactly *d*
([Three ways to measure a network's dimension](ball-dimension-and-curvature.md)).

- **How balls grow.** Count the agents within *r* steps of an agent: its
  *ball*. In *d* dimensions, their number grows as *r*^*d*.
- **How wide a lump is.** *N* agents packed into *d* dimensions span about
  *N*^(1/*d*) steps. A cube of a thousand points is ten points wide.
- **How many boxes cover it.** In *d* dimensions, boxes half as wide take
  2^*d* times as many ([Box dimension](ball-dimension-and-curvature.md#box-dimension)).
- **How a random walker returns.** A walker that wanders at random is back
  where it started after *t* steps with a probability that falls as
  *t*^(−*d*/2). This is the **spectral dimension**, the one quantum-gravity
  models measure. In causal dynamical triangulations it is about 4 at large
  scales, and about 2 at the smallest (Ambjørn, Jurkiewicz and Loll 2005).

On a tree in which every agent has two children or more, the first two give
no finite answer at all. The number of agents within *r* steps grows faster
than any power of *r*. A tree drawn at random among all trees is different. By
theory it reads 2 by balls, and 4/3 by walks
([The rulers, tried on known spaces](how-many-dimensions-does-a-world-have.md#the-rulers-tried-on-known-spaces)).
On a regular grid, all four measures agree. On irregular networks they need
not, and how they differ says what kind of space a network is.

### What the worlds turn out to be

The worlds of B1 are about **three-dimensional** by the first two measures.
Over their first few steps, their balls grow as *r*^3
([The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world)).
They do so at every size, from a few hundred agents to sixty thousand
([Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md)). That
is the reading of the Graph of Life viewer, which averages the balls of many
agents. The few agents next to a hub, an agent with very many connections,
pull that average up. A typical agent's ball grows more slowly in small
worlds
([The rulers, on the worlds](how-many-dimensions-does-a-world-have.md#the-rulers-on-the-worlds)).

And the distance across a world, its diameter, grows as *N*^0.30 for *N*
agents. That is how a body of 3.3 dimensions would grow, since 1/0.30 ≈ 3.3
([Distances grow like a power](how-does-a-worlds-size-follow-its-tokens.md#distances-grow-like-a-power)).

[Is a world a space?](how-many-dimensions-does-a-world-have.md) measured the
dimension with two rulers, tried first on spaces whose dimension is known. By
the room within *r* steps, worlds of 60,000 agents read 3.3, steady from 3
steps to 14. By how a random walker spreads, they read about 2. Networks with
the same connections at every agent, wired at random, read neither. These
[random twins](random-twin.md) are *small worlds*, in which everyone is a few
steps from everyone.

That the worlds are spaces at all is a surprise worth taking seriously, and
also with care. By box covering, the worlds measure about 1.9. They are full
of stars, hubs with many agents of one connection around them, and of trees.
And about a quarter of their connections (a third at 10,000 tokens) are
**bridges**: cut one, and the network falls in two ([Bridges](bridges.md)).
Some of them, at times, hold half a world to the rest
([How a world breaks](how-a-world-breaks.md)).

A three-dimensional lattice has no bridges at all. To cut a cube of *N* points
in two, you must cut about *N*^(2/3) connections: a whole cross-section.

So the worlds are spaces, with room like a three-dimensional body. But they
are full of dead ends, which slow anything that spreads through them. They are
a fractal more than a solid.

### A well-knit world

So the aim can be put precisely. Find rules under which a world grows into a
network that is three-dimensional by every measure at large scales, and **well
knit**. Bridges will always exist, but they should hold twigs, not half the
world.

The tools are ready. There are the dimensions above. There is the cut risk:
the largest share of a world that one bridge holds to the rest
([Cut risk](bridges.md#cut-risk)). And there is the spectral gap, which says
how hard a network is to cut in two
([The spectral gap](path-length.md#the-spectral-gap)). For a *d*-dimensional
body of *N* agents, it shrinks as *N*^(−2/*d*); for a network with cheap
cuts, it is tiny. One hint came early: bigger worlds have more loops per
connection, and fewer leaves
([Is a big world the same kind of world?](how-does-a-worlds-size-follow-its-tokens.md#is-a-big-world-the-same-kind-of-world)).
A leaf is an agent with a single connection.

B2 was built with rules of this kind, all of them local
([A world that cannot break](a-world-that-cannot-break.md#a-world-that-cannot-break)):

- a newborn is joined to its parent and to two of its parent's [neighbours](the-world.md#the-network "An agent joined to another by a connection.");
- when an agent dies, the groups its former neighbours fall into are joined
  up in a ring: the death is **mended**;
- an unused connection may lapse only where its two ends share two
  neighbours;
- a brain may choose at most eight connections.

A mended death can never split the world, and such a lapse cannot either. So
the cull, the rule that kills every piece cut off from the rest (below), is
left with nothing to do. In a model without brains, mending brought the
largest share of a world one agent's loss could cut off from 37% to 1.4%. The
largest share one connection's loss could cut off fell from 12% to 1.2%, at a
ball dimension of 3.7. With brains, in Experiment 14, not one agent of ten B2
worlds was cut off from iteration 200 to 1,000.

But a world that cannot break is not yet a space.
[What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)
read B2's worlds with the full rulers. They are nearly small worlds: a typical
agent is about as near to everyone as in the world's random twin. Every B2
world starts as a space, and folds once its first generation dies of age. Each
mended death joins agents that were two steps apart, through the one who died.
A ball dimension like the 3.7 above is read off the first few steps. Such a
reading cannot tell a space from a small world.

So room and resilience pull apart. Mended in a ring, a world folds. Mended
only where it would split, the model without brains stretches into a line.
The nearest to this aim so far are B2 worlds in which the tokens of the dead
go to the newborns, or to their parents. They cannot break, and they are
spaces of about three dimensions. But nine agents in ten hold three tokens or
fewer, and their evolution stops ([Who inherits?](who-inherits.md)). A world
that is a well-knit space, rich, and still evolving is yet to be found
([Healthy and rich](healthy-and-rich.md)).

## A speed limit

Nothing in physics travels faster than light. Whatever happens at a point can
only affect what lies inside its **light cone**. That is the region light
could reach from it: a sphere that grows at the speed of light. This is
**locality**, and it is one of the deepest facts we know about the world.

Graph of Life is built to be local. Every iteration has two phases, one of
reproduction and one of the game
([One iteration, step by step](one-iteration.md)). In one phase, an agent
reads only a few things ([What a brain sees](brain-inputs.md)):

- itself and its neighbours;
- a summary of each neighbour's own neighbourhood;
- messages written by itself and its neighbours ([Messages](messages.md)). In
  B1 they are written in a first look of the same phase, before anyone acts;
  in B2, in the phase before.

And it acts only on its own node and its neighbours' nodes. Follow a
difference at one agent, *x*, through one phase:

![A light cone on a network](../diagrams/light-cone.svg)

So news of *x* can travel at most **four connections per phase**, eight per
iteration. Tokens and brains are slower. A token moves at most one connection
per game, because a stake goes only to a neighbour. A brain moves at most one
connection per game too, because a node is won only by a neighbour. This
mirrors physics as well: light is the fastest thing there is, and matter
moves more slowly.

Lines of descent move far more slowly still. In a B1 world, the patch a line
holds advances about a step every thirty-five iterations. It widens as a
random walk spreads
([Where the lines live](how-does-a-worlds-size-follow-its-tokens.md#where-the-lines-live)).

### Where the speed limit is broken

Three rules reach across the whole world at once.

1. **The cull.** After every phase, only the largest connected piece of the
   network survives, and every agent outside it dies
   ([One iteration, step by step](one-iteration.md#cleanup)). Whether a
   region lives depends on the connections of the entire world, at that
   instant: action at a distance. In B1, three deaths in four are of this
   kind, and the big losses come this way, whole regions at once
   ([How a world breaks](how-a-world-breaks.md)). The regions lost grow with
   the world
   ([Big worlds do not average out](how-does-a-worlds-size-follow-its-tokens.md#big-worlds-do-not-average-out)).
   In B2 nobody can be cut off, so the cull has nothing left to do.
2. **The share-out.** The tokens of the dead are dealt out to survivors drawn
   at random from the whole world
   ([The share-out](gains-and-losses.md#the-share-out)). In B1 these are the
   tokens of agents cut off; in B2, those of agents that die of age. The
   tokens are conserved, but they jump: a token can leave one end of a world
   and arrive at the other in one step.
3. **The one random stream.** Every random number in a world is drawn from a
   single sequence, in a fixed order ([Seeds and the random stream](random-numbers.md)).
   That holds for every tie, every probabilistic decision and every [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.").
   This is what makes a run exactly repeatable. But it also couples
   everything. If one agent anywhere takes one more random number, every
   later draw everywhere shifts. So two worlds that differ in one detail
   cannot keep their difference inside a light cone.

Each of the three could be made local:

- a piece that separates could live on, as a world of its own;
- the tokens of the dead could go to their former neighbours;
- every agent could draw its random numbers from a stream of its own, fixed
  by the seed, its id and the iteration. Generators called "counter-based" do
  exactly this, and keep a run exactly repeatable (Salmon, Moraes, Dror and
  Shaw 2011).

The first two changes were also proposed for quite another reason. The cull
and the share-out stand in the way of open-ended evolution, as
[Questioning the mechanics](questioning-the-mechanics.md#one-rule-reaches-across-the-world)
and
[What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)
found. Physics and evolution, asked separately, point at the same rules.

None of the three has been made local yet. B2 took another road: its worlds
cannot split at all, rather than living on apart. The tokens of the dead have
been given to other heirs, such as the newborns, their parents or the poorest
([The estate of the dead](estate.md)). They have not gone to the dead agent's
former neighbours. All three changes are planned, in *Local rules only*.

### A word of care about long-range connections

[What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution)
also proposed letting distant agents connect, to renew the world's shortcuts.
Locality argues for care. A connection between two agents that were a hundred
steps apart is a message that travelled a hundred steps in an instant.

Births already respect the speed limit. A child joins its parent's
neighbourhood, so any two agents it joins were at most two steps apart. New
connections could do the same. They could form only between agents a few
steps apart, or with a reach that costs tokens to extend.

That is what was tried. In [Links to strangers](links-to-strangers.md), the
agents of B2 worlds could send a search through the network and propose a
link where it ended. Its steps cost tokens, or were free. When a search cost
tokens, hardly anyone paid for one; when it was free, agents reached all the
time. Either way, the strangers they found were two or three steps away. The
network grew denser where it already was, and its dimension did not change. A
child joined to an agent anywhere in the world was never adopted.

### A test: how far does a difference travel?

Locality makes a prediction that can be checked. Run two worlds that are
identical except for one token at one agent. Then watch where they differ,
game after game. Under local rules, the difference must stay inside a cone
that grows by at most four connections per phase. How fast it actually
spreads, and whether it grows or dies away, says how chaotic the world is.

This is **damage spreading**, a standard experiment on cellular automata and
random networks (Derrida and Weisbuch 1986). It needs local random numbers
first: with one shared stream, the difference is everywhere at once. It is
planned, as *How far does a difference travel?*

## Conservation

Energy is conserved: it changes form, but the total never changes. Emmy
Noether (1918) showed why. Conservation of energy follows from the laws being
the same at every moment.

Tokens are conserved in the same sense. Every game moves them, every birth
splits them, and every death passes them on. The total stays exactly what it
was at the start ([One iteration, step by step](one-iteration.md)). Tokens
also come in whole units, like quanta.

Physics asks more than a constant total: energy is conserved **locally**. The
energy in a region changes only by what flows across its boundary: a
*continuity equation*. Stakes respect this, since a token moves only along a
connection. The share-out does not, as above.

## Agents as little calculators

The laws of physics are written as differential equations. Such an equation
says how a quantity at a point changes, given its values nearby. To solve one,
a computer cuts space into a grid. Each point of the grid then updates itself
from its own value and its neighbours'. This is a **finite-difference**
scheme. Every grid point is a little calculator that looks only at itself and
its neighbours, and information moves one cell per step. Physicists, in this
sense, already simulate the world with local agents.

Graph of Life's agents are such calculators, with one difference. Their
calculator, the brain, is not given by any equation: it evolves. And yet, in
B1, what the agents do comes to something an equation would give. They spread
their stakes almost evenly over themselves and their neighbours
([Where tokens flow](where-the-tokens-flow.md)). And an
[even split](even-split.md) is exactly one step of the **heat equation**:

![The even split as a step of the heat equation](../diagrams/finite-difference.svg)

On a network, the same holds with the discrete Laplacian, the
[token curvature](gains-and-losses.md#token-curvature). It adds up, over an
agent's neighbours, how many more tokens each holds than the agent. For
agents with *d* neighbours each, an even split changes an agent's tokens by
its curvature divided by *d* + 1
([Tokens run downhill](gains-and-losses.md#tokens-run-downhill)). So a settled
B1 world, in its tokens, behaves almost like a computer solving the heat
equation on its own network.

### Down to a numerical artefact

The likeness even reaches a known artefact of such computations. An explicit
step of the heat equation that moves more than half of a cell's content to its
neighbours overshoots. The solution then flips about its resting value at
every step. A hub with *d* neighbours keeps only 1/(*d* + 1) of its tokens.
With two neighbours or more, it passes more than half of them on.
[A two-game rhythm](where-do-the-tokens-go.md#a-two-game-rhythm) found exactly
that flip: wealth swings between hubs and their leaves with a period of two
games.

### What the brains do, and what the rules do

Do the brains, then, compute the heat equation? No.
[What the brains are like](what-the-brains-are-like.md) found the evolved
brains of B1 nearly blind. A change in what they see moves their outputs by
about a thousandth of its size. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world, 42% of agents (58% at
10,000 tokens) give every candidate a score of zero or below. The rules then
split their tokens exactly evenly. The others give every candidate nearly the
same score.

Selection set one constant: spread the stake. The heat equation is what the
rules make of it.

The brains of B2, which see, are different. They go all in on one node in
nearly every case, so the even split hardly arises there
([How often it happens](even-split.md#how-often-it-happens)).

### Heat, not waves

That is also a limitation. The heat equation **forgets**. It smooths every
difference away, and its world ends uniform: "heat death" in miniature. The
fundamental equations of physics are of another kind. They are wave
equations, which carry signals and energy without losing them, and run as
well backwards as forwards. Diffusion appears in physics only when many
particles are averaged.

So a question: **can rules make tokens move like waves**, sloshing back and
forth with a characteristic frequency, rather than spread like heat? A
frequency would show as a peak in a power spectrum
([The power spectrum](autocorrelation.md#the-power-spectrum-of-a-time-series)).
The spectrum of the number of agents shows none
([Power laws, real and apparent](power-laws-real-and-apparent.md)). In B1's
worlds of 100,000 tokens it falls as *f*^−1.80, close to the smooth slope of a
random walk. It flattens only at the slowest rhythms, where something pulls a
world back to its level.

The two-game swing of the hubs is no such wave either. It is the overshoot of
a step that is too large, damped in every game, and it carries nothing from
place to place. In physics a frequency is an energy (*E* = *hν*). Whether the
rate of an exchange could play that part here is an open, speculative
question.

## A beginning

The universe began small, dense and hot. It expanded and cooled, and formed
structure out of a nearly uniform start: atoms, stars, galaxies.

A world of Graph of Life begins the same way, at least in outline
([The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations)).
In a world of 10,000 tokens, a hundred [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") hold a hundred tokens each, in
a ring: small, dense in tokens and uniform. Then the network, which is space
here, grows explosively. There are about 75 births per hundred agents in each
of the first ten iterations.

In the median world, the founders' 100 become 1,132 agents by iteration 10,
and about 1,850 by iteration 20. Meanwhile the tokens per agent fall from 100
to under 9, and then to about 5. That is expansion, and cooling, if tokens per
agent are read as a temperature.

Then comes a crash and a bottleneck. The families fall from a hundred to about
twenty, and every later brain descends from a single founder. Then comes
structure: inequality, hubs, and after about a hundred iterations a world that
has settled down. (The notebook measures a world from iteration 500 on, its
[settled life](median-and-quantiles.md#the-settled-life-of-a-world "A world from iteration 500 to its end, after its youth, where it is measured.").)

The worlds of 100,000 tokens do the same ten times over, a little faster.
Their bottleneck keeps about 400 of their 1,000 founders' lines. And three of
thirty still descended from more than one founder after 1,500 iterations.

The differences matter as much. Space here does not stretch everywhere at
once: new places are born one at a time, next to their parents. Birth makes
space. Every B2 world starts as one. It folds only from iteration 100 on,
when its first generation begins to die of age
([A world through its life](how-many-dimensions-does-a-world-have.md#a-world-through-its-life)).
And nothing here plays the part of gravity, which in the real universe gathers
matter into clumps.

## Everywhere the same

On large scales the universe looks the same everywhere and in every
direction: the *cosmological principle*. Look closely at any part, and you see
only elementary particles, all of a few kinds, all alike. Structure appears
only when very many are taken together.

Graph of Life's agents are its elementary particles. Every one follows the
same rules, and differs only in its state: its tokens, its connections, its
brain. And its worlds are statistically homogeneous, nearly the same at every
size. Inequality, [home stakes](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node "The tokens an agent stakes on its own node, to keep it.") and [genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.") per agent are the same in worlds
of five hundred agents and of sixty thousand
([Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md#is-a-big-world-the-same-kind-of-world)).
Births and deaths run a little faster in larger worlds: 13 to 21% more per
agent at 100,000 tokens than at 10,000.

Structure, where there is any, shows only in collective patterns: families
side by side ([One world, many colours](one-world-many-colours.md)), hubs with
their stars. And most of it does not reach far.
[Like next to like](like-next-to-like.md#how-alike-at-a-distance) measured how
alike agents are at a distance. Tokens, ages and connections are alike only
within one or two steps, and agents of one genotype within two or three.
Beyond that, what one agent holds is like what any other holds.

Lines of descent reach further. Give every agent its *family*: the genotype it
descends from, a number of iterations back. Two agents belong to the same
family of 25 iterations back more often than chance would have them out to
about 7 steps. For a family of 200 iterations back, it is out to about 12
([Where the lines live](how-does-a-worlds-size-follow-its-tokens.md#where-the-lines-live)).
A family is a patch of the world that widens as its line ages, as a random
walk spreads. So a world has territories after all: of lines, not of
genotypes.

### No scale at all

Many natural systems have **no characteristic scale**. Think of water at a
critical point, 374 °C and 218 atmospheres, or of a magnet at its Curie
temperature. There, fluctuations of every size appear at once. Correlations
span the whole system, and quantities follow power laws.

Bak, Tang and Wiesenfeld (1987) proposed that some systems tune themselves to
such a point: **self-organised criticality**. That, they argued, is why
sandpiles, earthquakes and perhaps evolution show events of every size (Bak
and Sneppen 1993). Some have argued that complex structure and computation
live best near such a point, "at the edge of chaos" (Langton 1990).

What the notebook has found is mixed
([Power laws, real and apparent](power-laws-real-and-apparent.md)):

- the tokens of the richest agents follow the most convincing power law, over
  one to two decades;
- the connections of the best-connected agents plausibly follow one;
- deaths do not come in avalanches of every size: the agents one game removes
  have a typical size;
- and a world's wander is a random walk's, pulled back to its level within a
  few hundred iterations. It is not the 1/*f* noise of a critical system,
  which has memory on every time scale.

But [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md#big-worlds-do-not-average-out)
found a property that does not change with size. The mean game cuts off
about 2% of a world, at five hundred agents and at sixty thousand, with no
trend. The worst cuts off a third to a half, at every size. Whether the whole
distribution is the same at every size has not been compared yet.

Finding such scaleless properties systematically is now an aim of the
notebook. Measure the same thing at many sizes, and ask whether, rescaled, it
is the same (*Which properties have no scale?*).

Two measurements since sharpen the picture. What agents hold is correlated
over one or two steps only. That is a short correlation length, the distance
over which likeness fades, so the worlds are far from critical
([Like next to like](like-next-to-like.md)). But the space itself has no
scale. Within *r* steps lie about *r*^3.3 agents, the same law from 3 steps to
14 ([The rulers, on the worlds](how-many-dimensions-does-a-world-have.md#the-rulers-on-the-worlds)).

## Why life is possible: a flow of low entropy

Energy is conserved. And disorder always grows: that is the second law of
thermodynamics. How, then, is anything as ordered as life possible? The
answer was given by Boltzmann (1886) and made famous by Schrödinger (1944).
The Earth is not a closed system.

It takes in energy as **sunlight**: a few energetic photons, from the Sun's
surface at about 5,800 kelvin. It sends the same energy back to space as
**infrared**: many feeble photons, from a planet at about 255 kelvin. Energy
in equals energy out. But the radiation it sends out carries about twenty
times the entropy of the radiation it takes in. That is roughly the ratio of
the two temperatures, 5,800/255 ≈ 23.

That export of entropy pays for every ordered thing on Earth, from a cloud to
a cell. The Sun, in turn, is a reservoir. Gravity gathered its hydrogen, and
fusion releases the energy slowly, over billions of years: a gradient that
lasts.

Graph of Life has the conservation, and not the gradient. A token is a token,
wherever it came from: there are no fresh tokens to feed on, and no spent
tokens to shed. And in B1 the even split drives a world towards its resting
state. There every agent holds tokens in proportion to its connections plus
one ([The game as a random walk](where-the-tokens-flow.md#the-game-as-a-random-walk)).
That is the miniature heat death above.

There are two ways a gradient could be built in.

- **Tokens with a quality.** Fresh tokens would become spent when used, and
  only slowly fresh again: the count conserved, the quality not. The token
  colours of the research plan (`research/Research.md`, §4.3) are one version.
  They have not been tried yet.
- **Sources and sinks.** A few "stars" would release tokens slowly, and there
  would be a way for tokens to leave: a driven, open world. Physics knows what
  can form in such a flow: **dissipative structures** (Nicolis and Prigogine
  1977). Examples are the hexagonal convection cells that appear in a pan of
  oil heated from below.

Either would make a world in which order can feed on a flow: the condition, on
Earth, for life. Both are planned, as *A source and a sink*.

## Particles: stable things in a restless world

Conway's Game of Life (Gardner 1970) is a grid of cells, each alive or dead,
with three local rules. Out of them come **gliders**: small patterns that move
across the grid and persist, although every cell under them keeps changing.
They behave like particles: they travel, collide, annihilate and make new
patterns. And they are made of nothing but the rule.

In Graph of Life, a particle would be the same kind of thing. It would be a
pattern that persists while the agents, tokens and brains that carry it
change. That is exactly the third rung of the ladder towards open-ended
evolution: P3, an organisation
([A ladder of four steps](what-this-notebook-is-about.md#a-ladder-of-four-steps)).
Something as humble as an electron, emerging here, would already be a win for
both aims.

What to look for:

- **Bound tokens**: tokens that keep going round a closed loop of agents,
  instead of spreading.
- **Groups that outlast their members**: a set of agents whose membership
  turns over while the set persists.
- **A beat**: a pattern with a characteristic frequency, local in the
  network.

### What has been found

Nothing yet lasts in the way a glider does.

- **Bound tokens.** Within one game, almost no tokens go round once the stakes
  that come back have cancelled
  ([Do tokens go round?](where-the-tokens-flow.md#do-tokens-go-round)). Over
  fifty games, more goes round than chance would make. But no ring of tokens
  outlasts those fifty games
  ([Where the tokens go](gliders-and-who-helps-whom.md#where-the-tokens-go)).
- **Patterns that travel.** When an agent takes a neighbour's node with at
  least half of its tokens, its wealth and a copy of its brain move together.
  Such moves form chains far more often than chance would. In B1 they are
  *gliders*, rich agents moving on; in B2, *puffers*, a [lineage](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor.") copying itself
  outward. But no chain keeps a direction; each wanders like a random walk
  ([Gliders, and who helps whom](gliders-and-who-helps-whom.md)).
- **A beat.** The nodes that launch such chains fire without a rhythm. And the
  two-game swing of the hubs is an overshoot, damped in every game.
- **Groups.** The nearest thing is a star's **dynasty**, in B1
  ([How a star is handed down](the-largest-and-the-smallest.md#how-a-star-is-handed-down)).
  A centre joined to thousands of agents is handed down from parent to child,
  in the longest dynasties for hundreds of iterations. Meanwhile the brain on
  it changes nearly every game. It is a place that lasts, kept by the rules,
  not a group that does anything together
  ([Where the world stands on the ladder](what-the-baseline-world-is.md#where-the-world-stands-on-the-ladder)).
  And families hold patches of a world that widen as their lines age.

### Vacuum and matter

Most of the universe is nearly empty; matter is a tiny fraction of it. Most of
a B1 world is a restless sea of ordinary agents playing a near-even split: a
kind of vacuum. Could something stable form in it, rarely and locally, as
matter did?

One way to ask is an experiment planned for later, *One agent in a sea of
average agents*. Make a sea of agents that all decide as the average evolved
agent decides. Place in it a single agent, or a small group, with a brain of
its own. Can it draw tokens from the sea without exhausting it, and without
being cut off? Can a group hold together in it?

Biologists know this as the question whether a rare mutant can **invade** a
resident population (Maynard Smith and Price 1973; Metz, Nisbet and Geritz
1992). Physicists know it as a small system in contact with a **reservoir**.

A first answer, of another kind, has come from evolved B2 worlds. Strategies
written by hand were planted there, as patches of agents
([How large a crowd a raider needs](strategies-written-by-hand.md#how-large-a-crowd-a-raider-needs)).
A single raider never took hold. About a hundred raiders had to start
together, and the world they started in decided as much as their number.

## Clocks, cones and relativity

Special relativity says there is no absolute "now". Two observers moving past
each other disagree about which distant events happen at the same time. Yet
the laws of physics are the same for both. Graph of Life has one clock for
everyone: its iterations and phases. So it is not relativistic. And a speed
limit alone does not make it so: a grid with a speed limit still has a
preferred frame, its own.

But one idea points the way. To compute the future of one agent, you do not
need the whole world, only its past cone. That is the agents near it, then the
agents near those, back to the start. Different regions could then be computed
to different "times", as long as no region gets ahead of what can reach it.
Computer scientists use exactly this to run large simulations in parallel:
every part keeps its own clock, "virtual time" (Jefferson 1985).

Wolfram (2020) argues for more. Suppose the outcome of a system of local
updates does not depend on the order in which they are applied: **causal
invariance**. Then observers who order them differently must agree on what
happens, and special relativity follows in the large.

Graph of Life is not there. Its global rules make every agent's future depend
on the whole world, so no cone can be computed on its own. In B2 the cull has
nothing left to do, but the share-out and the one random stream still reach
everywhere. Making them local is the first step towards even asking the
question.

There is a second, looser parallel, with general relativity. There, space is
not fixed but shaped by what is in it. Wheeler put it in one line:
"Spacetime tells matter how to move; matter tells spacetime how to curve"
(Wheeler and Ford 1998).

In Graph of Life, the network tells the tokens where they can flow: only along
connections. And the tokens tell the network which connections remain. In B1,
a connection that no token crosses in a game is cut
([How a world breaks](how-a-world-breaks.md)). In B2 an unused connection
lapses only where its two ends share two neighbours, and deaths are mended
with new connections. There, the tokens have less say over the network.

## Where the analogy stops

- There is nothing quantum here: no superposition, no interference.
- The rules have no continuous symmetries, no rotations and no boosts, and
  the network is no smooth space.
- Agents carry brains, functions that evolve. Particles carry no such thing,
  and the laws of physics, as far as anyone knows, do not evolve.
- Three rules are global, as above.
- And the "beginning" is a choice of the program, not an event.

## What this asks of the notebook

The second aim adds its own questions, and its own planned pages, under
[Space and dimension](../README.md#space-and-dimension) in the contents. They
are not written yet:

- *Dimensions at scale.*
  [Is a world a space?](how-many-dimensions-does-a-world-have.md) has read
  balls and walks on worlds of 60,000 agents. Next comes every measure of
  dimension on the largest worlds the notebook can afford: ball growth, the
  width of a lump, boxes, the spectral dimension.
- *How far does a difference travel?* Damage spreading: the light cone,
  measured, once random numbers are local.
- *Which properties have no scale?* The same quantities across many sizes,
  rescaled.
- *Local rules only.* The cull, the share-out and the random stream, each made
  local, and what changes.
- *A source and a sink.* Tokens with a quality, or stars: a world with a flow
  running through it.
- *Particles.* Bound tokens, persistent groups, beats.
- *One agent in a sea of average agents.* The invasion experiment above.

## What we learned

- **The rules were made local, conserving and uniform on purpose**, so that
  space, stable things and life could emerge rather than be built in.
- **Three rules break locality**: the cull, the share-out and the single
  random stream. Making them local is what physics asks, and what open-ended
  evolution asked for independently. B2 has left the cull with nothing to do;
  the other two remain.
- **B1's worlds are spaces of a kind.** They look three-dimensional in how
  their balls grow and how their diameter grows. But a walker reads only two
  dimensions in them: they are full of dead ends, held together by bridges
  ([Is a world a space?](how-many-dimensions-does-a-world-have.md)).
- **A world that cannot break folds.** B2's local rules keep every world
  whole, but deaths mended in a ring fold it into a near small world. Room and
  resilience pull apart.
- **B1's evolved brains spread their stakes evenly, and the rules make that a
  step of the heat equation.** A world that only diffuses ends uniform. A
  world in which something lasting forms will need more: a flow, a gradient,
  something that holds tokens together.
- **Particles and organisations are the same question.** A pattern that
  persists while what carries it changes is what both aims are looking for.
  The nearest so far are a star's dynasty, a place handed down, and chains of
  takeovers that wander.

## Where to go next

- [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md):
  the dimension of the worlds, measured, and the rules that fold them.
- [Where tokens flow](where-the-tokens-flow.md): the game as a random walk of
  tokens, the other face of the heat equation.
- [Gliders, and who helps whom](gliders-and-who-helps-whom.md): patterns that
  outlast their parts, looked for in the record of play.
- [What this notebook is about](what-this-notebook-is-about.md): the first
  aim, open-ended evolution, which keeps meeting this one.

## How our view changed

- **14 October 2026.** From 9 October, this page said that the evolved brains
  compute, in effect, the heat equation.
  [What the brains are like](what-the-brains-are-like.md) found them nearly
  blind. Selection made them spread their stakes. The rules turn a spread
  stake into a step of the heat equation
  ([What the brains do, and what the rules do](#what-the-brains-do-and-what-the-rules-do)).
- **14 October 2026.** From 9 October, this page said that kin, like tokens,
  ages and connections, are alike only within one or two steps. Beyond three
  steps, it said, every part of a world is like every other. Agents of one
  genotype cluster within two or three steps
  ([Like next to like](like-next-to-like.md)). And
  [Worlds of 100,000 tokens](how-does-a-worlds-size-follow-its-tokens.md#worlds-of-100000-tokens)
  found that lines of descent hold patches. They reach about 7 steps for
  families of 25 iterations back, and about 12 for 200. So a world has
  territories, of lines ([Everywhere the same](#everywhere-the-same)).
- **17 October 2026.** From 14 October, this page said that
  [A world that cannot break](a-world-that-cannot-break.md) builds rules for a
  well-knit world. It gave them a ball dimension of 3.7, in a model without
  brains.
  [What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)
  then read B2's worlds with the full rulers. They are not spaces but nearly
  small worlds, folded by deaths mended in a ring. A dimension read off a
  world's first few steps cannot tell the two apart
  ([A well-knit world](#a-well-knit-world)).
- **24 October 2026.** When the page was rewritten in simple English, four
  statements were checked against the pages that measure them, and
  corrected:
  - From 9 October, the page said that on a tree, the agents within *r*
    steps grow faster than any power of *r*. That holds for a tree in which
    every agent has two children or more. A tree drawn at random among all
    trees has a ball dimension of 2
    ([The rulers, tried on known spaces](how-many-dimensions-does-a-world-have.md#the-rulers-tried-on-known-spaces)).
  - From 9 October, it said that births, like inequality, home stakes and
    genotypes, are the same in worlds of every size. But since 11 October,
    [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md#a-large-world-beside-ten-small-ones)
    has found them a little faster in larger worlds. Births and deaths per
    agent are 13 to 21% higher at 100,000 tokens than at 10,000.
  - From 9 October, it said that the richest agents' tokens follow a clean
    power law. On this day,
    [Power laws, real and apparent](power-laws-real-and-apparent.md) read its
    figure again, point by point. At 100,000 tokens the law holds for one
    decade, to about 500 tokens; at 10,000, for two.
  - From 9 October, it said that the share of a world one game cuts off has
    the same distribution at every size.
    [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md#big-worlds-do-not-average-out)
    shows less. The mean, about 2%, and the worst game, 35% to 48%, do not
    change with size. The whole distributions have not been compared.

## References

- Ambjørn, J., Jurkiewicz, J. and Loll, R. (2005). Spectral dimension of the universe is scale dependent. *Physical Review Letters* 95, 171301.
- Bak, P., Tang, C. and Wiesenfeld, K. (1987). Self-organized criticality: an explanation of 1/*f* noise. *Physical Review Letters* 59, 381–384.
- Bak, P. and Sneppen, K. (1993). Punctuated equilibrium and criticality in a simple model of evolution. *Physical Review Letters* 71, 4083–4086.
- Boltzmann, L. (1886). *Der zweite Hauptsatz der mechanischen Wärmetheorie*. Lecture to the Imperial Academy of Sciences, Vienna.
- Bombelli, L., Lee, J., Meyer, D. and Sorkin, R. D. (1987). Space-time as a causal set. *Physical Review Letters* 59, 521–524.
- Derrida, B. and Weisbuch, G. (1986). Evolution of overlaps between configurations in random Boolean networks. *Journal de Physique* 47, 1297–1303.
- Gardner, M. (1970). Mathematical games: the fantastic combinations of John Conway's new solitaire game "life". *Scientific American* 223(4), 120–123.
- Jefferson, D. R. (1985). Virtual time. *ACM Transactions on Programming Languages and Systems* 7, 404–425.
- Langton, C. G. (1990). Computation at the edge of chaos: phase transitions and emergent computation. *Physica D* 42, 12–37.
- Metz, J. A. J., Nisbet, R. M. and Geritz, S. A. H. (1992). How should we define 'fitness' for general ecological scenarios? *Trends in Ecology & Evolution* 7, 198–202.
- Nicolis, G. and Prigogine, I. (1977). *Self-Organization in Nonequilibrium Systems*. New York: Wiley.
- Noether, E. (1918). Invariante Variationsprobleme. *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse*, 235–257.
- Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011). Parallel random numbers: as easy as 1, 2, 3. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC11)*.
- Schrödinger, E. (1944). *What Is Life?* Cambridge: Cambridge University Press.
- Wheeler, J. A. and Ford, K. (1998). *Geons, Black Holes, and Quantum Foam: A Life in Physics*. New York: W. W. Norton.
- Wolfram, S. (2020). A class of models with the potential to represent fundamental physics. *Complex Systems* 29(2), 107–536.

<!-- turns -->
---

[Space and dimension](../README.md#space-and-dimension) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) →

**Also about:** [Start here](../README.md#start-here).

> [!info]- Pages that link here (17)
> [What this notebook is about](what-this-notebook-is-about.md) · [Questioning the mechanics](questioning-the-mechanics.md) · [Seeds and the random stream](random-numbers.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [The even split](even-split.md) · [Where tokens flow](where-the-tokens-flow.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) · [Like next to like: how alike agents are at a distance](like-next-to-like.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [Three ways to measure a network's dimension](ball-dimension-and-curvature.md) · [Worlds of many sizes](how-does-a-worlds-size-follow-its-tokens.md) · [What a brain can compute](what-a-brain-can-compute.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The health bar](health-bar.md)
<!-- /turns -->
