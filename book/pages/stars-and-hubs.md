# Stars and hubs

## In short

Most [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") have one or two [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."); a few, the **hubs**, have hundreds or
thousands. A world whose best-connected agent is joined to a tenth of it or
more is a **star**, and seven of the thirty [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")
were. A star's centre is a place, not an individual: the [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") on it changes
nearly every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), while its connections pass from parent to newborn child. The
[health bar](health-bar.md) counts any agent joined to more than 5% of the others as a fault.

## Hubs

An agent's [degree](degree.md) is its number of connections. In a settled
baseline world of 100,000 tokens the median agent has 2 and the mean is 3.5, but
the best-connected agent of the median world has 640 (from 130 to 11,000
between worlds; [The largest and the smallest](the-largest-and-the-smallest.md)).
An agent that far above the rest is a **hub**. There is no exact line for a hub;
there is one for a star.

Hubs are places. Under the near-[even split](even-split.md), the agent on a node
with *d* connections is one of *d* + 1 stakers of about equal size, and keeps
its node about as often as one of them would: with fifty connections or more,
in 2.0% of games
([Where the tokens flow](where-the-tokens-flow.md#who-keeps-their-node)).

## Stars

A world is a **star** when its best-connected agent is joined, on average over
the world's [settled life](median-and-quantiles.md#the-settled-life-of-a-world), to at least a tenth of the agents
alive (in the code, `STAR = 0.1` in `book_chapters/common.py`). By this measure
7 of the 30 worlds of Experiment 10, at 100,000 tokens, were stars, and 10 of
the 26 surviving worlds of 10,000 tokens of Experiment 2. In the large worlds a star sets the
shape: across the thirty, the share of agents in the
[2-core](bridges.md#core-trees-and-leaves) falls as the share joined to the largest hub
rises, with a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of −0.96
([How much does the seed decide?](how-much-does-the-seed-decide.md)).

The largest star is the world with [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 28, whose centre was joined to 58% of
all agents on average, for a thousand [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). At iteration 1,122 it was
joined to 20,666 of the 25,958 agents alive and held 20,286 tokens, a fifth of
all; 94% of its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") were leaves, holding 2.8 tokens each on average. A
[coalition](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") of neighbours won its node in each of the hundred games from then
on, and the node carried 78 different [genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.") in them
([The largest and the smallest](the-largest-and-the-smallest.md#how-a-star-works)).

## How a star is handed down

An agent gains connections as its neighbours have children, about in
proportion to those it has, but loses them a little faster, so the well
connected drift back down
([How the network grows](how-agents-have-children.md#who-gains-and-who-loses)).

A star's centre is kept another way. Now and then it has a child and hands it
nearly all its connections ([How agents have children](how-agents-have-children.md)):
at iteration 1,153 the centre of seed 28 went from 18,913 connections to 219,
and its child, of age 0, took 18,865. Over the settled lives of the thirty
worlds, first place in connections changed hands 1,299 times, 21% of them to a
child of the agent before and 16% to a newborn one. A line of first agents,
each the child of the one before, is a **dynasty**; dynasties lasted a median
of 7 iterations, and the longest 942.

## Why the health bar counts a hub as a fault

The author's picture of a world that might keep evolving asks for a network
that is resilient, decentralised and close to three-dimensional, with few
single points of failure. So the
[health bar](how-to-tell-a-world-that-keeps-evolving.md#the-network-that-holds-in-numbers)
allows no agent to be joined to more than 5% of the others. If one agent holds
links to many others, the world depends on it: when that hub dies, everything
changes at once ([What makes a world interesting](what-makes-a-world-interesting.md#the-six-criteria)).
And a star is mostly leaves, and short: in seed 28 only 19% of the agents were
in the core and 72% were leaves, and two agents were 4.2 steps apart on
average, against 12 to 26 steps in the worlds without a star
([What shape does the network take?](what-shape-does-the-network-take.md)).

[B2](a-world-that-cannot-break.md) keeps hubs down: a newborn's links are drawn
at random, a brain may choose at most 8 links
([`valence`](settings.md#valence)), and no parent hands its links to a child.

## Where it is used

- [How the game is played](how-the-game-is-played.md#do-the-worlds-agree): the
  stars play the game apart from the other worlds.
- [Where the tokens flow](where-the-tokens-flow.md#the-game-as-a-random-walk):
  the even split fits the game with a slope of 0.94 without the stars, 0.65
  with them.
- [What goes with what](what-goes-with-what.md#between-worlds): the one
  component that orders the thirty worlds by their largest hub holds 47% of
  their differences.
- [What the measurements say about open-ended evolution](what-the-baseline-world-is.md#what-the-measurements-say-about-open-ended-evolution):
  a dynasty is the nearest thing yet to a group that lasts while its members
  change.

## Where to go next

- [The largest and the smallest](the-largest-and-the-smallest.md): stars and
  dynasties in full.
- [Degree](degree.md): connections, hubs and leaves.
- [A world that cannot break](a-world-that-cannot-break.md): the rules that
  keep hubs down.

<!-- turns -->
---

← [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Distances in a network: path length, radius and diameter, the spectral gap](path-length.md) →

**Also about:** [Wealth: who holds the tokens](../README.md#wealth-who-holds-the-tokens).

> [!info]- Pages that link here (7)
> [Glossary](glossary.md) · [How much does the seed decide?](how-much-does-the-seed-decide.md) · [The largest and the smallest](the-largest-and-the-smallest.md) · [The even split](even-split.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)
<!-- /turns -->
