# What the viewer can colour by

The viewer draws every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") as a dot and every [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") as a line, and can
colour and size both by a measured quantity. This note says what each
quantity is, and how a number becomes a colour — so that a picture can be
read as precisely as a chart. [One world, many colours](one-world-many-colours.md)
shows one world coloured six ways.

## From a number to a colour

For the [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") on screen, and for the quantity chosen:

1. **Values.** Every agent (or connection) gets its value *v*.
2. **Logarithm, if chosen.** *v* becomes ln(1 + *v*) — or, for a quantity
   that can be negative, sign(*v*)·ln(1 + |*v*|), which squeezes large
   magnitudes but keeps the sign.
3. **Range.** For an amount, the range runs from the frame's smallest value
   to its largest. For a **signed** quantity (a change, a curvature), it runs
   from −*m* to +*m*, *m* the largest magnitude, so that the middle of the
   colour map always means zero.
4. **Position.** Each value is placed in the range as a number between 0
   and 1, and the **colour map** gives its colour. An unknown value (the age
   in a run recorded before ages were kept) is put in the middle.

The same colour therefore means different values in different frames: the
range follows the frame. The legend in the corner gives the range.

**Colour maps.** *viridis*, *plasma*, *magma*, *inferno*, *cividis* and
*ember* run from dark to light and suit amounts; *coolwarm* and *spectral*
diverge from a neutral middle and suit signed quantities; *turbo* is a
rainbow; *grayscale* is black to white.

## Agents

| choice | the value of agent *u* |
|---|---|
| [Tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") | its tokens |
| Degree | its number of connections ([Degree](degree.md)) |
| Token change (signed) | its tokens now minus its tokens before the phase; a newborn counts from 0 |
| Token change (magnitude) | the size of that change |
| Token curvature (signed) | Σ over its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") of (their tokens − its tokens) ([Token curvature](gains-and-losses.md#token-curvature)) |
| Token curvature (before phase) | the same, on the network as it stood before the phase; newborns have none |
| Loops through it | fundamental cycles of a breadth-first spanning tree through it ([Loops](bridges.md#loops)) |
| Triangles | the triangles it is a corner of ([Clustering](degree.md#clustering)) |
| Share of total tokens | its tokens divided by all tokens |
| [Brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") id | its genotype ([Genotypes](genotype.md)) |
| Parent brain id | its genotype's parent genotype |
| Age | [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") it has lived ([Age and lifetime](age-and-lifetime.md)) |
| Node id | its id, which is its birth order |

A word on **brain id**: a genotype is a number, and the colour map turns
numbers that are close into colours that are close. Close genotype numbers
were *born* at about the same time — they need not be related. To see kinship,
colour by family instead, as [One world, many colours](one-world-many-colours.md) does.

## Connections

| choice | the value of the connection between *u* and *v* |
|---|---|
| Average / weaker / stronger endpoint tokens | the mean, smaller, larger of their tokens |
| Endpoint token gap | \|tokens of *u* − tokens of *v*\| |
| Average / weaker / stronger endpoint degree | the mean, smaller, larger of their degrees |
| Average endpoint curvature (signed) | the mean of their token curvatures |
| Token flow | tokens staked across it in the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), both ways together ([Token flow](token-flow.md)) |
| Loops through it | fundamental cycles through it, as for agents |
| Triangles | triangles it is a side of |
| Bridge | 1 if it lies on no loop — cutting it splits the network ([Bridges](bridges.md)) — else 0 |

## The built-in looks

| look | dots coloured by | dots sized by | lines coloured by |
|---|---|---|---|
| default | tokens (log), *coolwarm* | tokens (log) | one colour |
| wealth | tokens (log), *inferno* | tokens | one colour |
| lineage | brain id, *turbo* | tokens (log) | the colour of their dots |
| structure | degree (log), *cividis* | degree | average endpoint degree (log) |
| flow | token curvature (log), *coolwarm* | size of the token change (log) | token flow (log) |
| minimal | one colour | one size | one colour |

<!-- turns -->
---

← [What a run records](frames-and-stats.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Pilot worlds](pilot-worlds.md) →

> [!info]- Pages that link here (4)
> [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [Measuring the flow: token flow, lightning and circulation](token-flow.md) · [Bridges, cut risk, loops and the core: how a network can be cut](bridges.md) · [One world, many colours](one-world-many-colours.md)
<!-- /turns -->
