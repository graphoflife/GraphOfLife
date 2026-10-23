# Token curvature

The viewer can colour every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") by its **[token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") curvature**: how much richer
its neighbourhood is than it is. The name comes from mathematics: it is the
discrete Laplacian of the token field, the quantity that drives heat flow.

## Definition

For agent *u* with tokens τ(*u*) and [neighbours](the-world.md#the-network "An agent joined to another by a connection.") *N*(*u*),

$$
\kappa(u) = \sum_{v \in N(u)} \bigl( \tau(v) - \tau(u) \bigr)
         = \sum_{v \in N(u)} \tau(v) \;-\; \deg(u)\, \tau(u) .
$$

- κ > 0: the agent is in a **valley** — its neighbours hold more than it does.
- κ < 0: the agent is on a **peak** — it holds more than its neighbours.
- κ = 0: as much above some neighbours as below others.

**Example.** An agent with 5 tokens and neighbours holding 2, 8 and 20:
κ = (2 − 5) + (8 − 5) + (20 − 5) = −3 + 3 + 15 = 15, a valley. A [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.") with 200
tokens and 50 neighbours holding 4 each: κ = 50 · (4 − 200) = −9,800, a deep peak.

## Why it is called curvature: heat

If heat spreads along the [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") of a network, each node's temperature
changes in proportion to the sum of its differences with its neighbours —
exactly κ. A valley warms up, a peak cools down, and the field smooths out.
In matrix language κ = −*L*τ, with *L* the graph's Laplacian matrix
([degrees](degree.md "The number of an agent's connections: how many neighbours it has.") on the diagonal, −1 for every connection).

## What it is for

If tokens behaved like heat, an agent's change of tokens in a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") would
follow its curvature at the start of the game. [Gains and losses](gains-and-losses.md)
tests that, and [Where the tokens flow](where-the-tokens-flow.md) explains
why something like it should happen: an agent that splits its stake evenly
over itself and its neighbours sends tokens **down** the slope.

## In the viewer

`token_curvature` is measured on the [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") itself; `token_curvature_pre` on
the graph and tokens as they were when the phase began, so that it can be read
against the change the phase then produced. On a signed colour scale, blue is
a peak and red a valley.

<!-- turns -->
---

← [Lightning: tokens that go round](lightning.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [Flows that go round](circulation.md) →

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [Gains and losses](gains-and-losses.md) · [Physical inspiration](physical-inspiration.md) · [What goes with what](what-goes-with-what.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
