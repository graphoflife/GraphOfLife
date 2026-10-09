# Token curvature

The viewer can colour every agent by its **token curvature**: how much richer
its neighbourhood is than it is. The name comes from mathematics: it is the
discrete Laplacian of the token field, the quantity that drives heat flow.

## Definition

For agent *u* with tokens τ(*u*) and neighbours *N*(*u*),

$$
\kappa(u) = \sum_{v \in N(u)} \bigl( \tau(v) - \tau(u) \bigr)
         = \sum_{v \in N(u)} \tau(v) \;-\; \deg(u)\, \tau(u) .
$$

- κ > 0: the agent is in a **valley** — its neighbours hold more than it does.
- κ < 0: the agent is on a **peak** — it holds more than its neighbours.
- κ = 0: as much above some neighbours as below others.

**Example.** An agent with 5 tokens and neighbours holding 2, 8 and 20:
κ = (2 − 5) + (8 − 5) + (20 − 5) = −3 + 3 + 15 = 15, a valley. A hub with 200
tokens and 50 neighbours holding 4 each: κ = 50 · (4 − 200) = −9,800, a deep peak.

## Why it is called curvature: heat

If heat spreads along the connections of a network, each node's temperature
changes in proportion to the sum of its differences with its neighbours —
exactly κ. A valley warms up, a peak cools down, and the field smooths out.
In matrix language κ = −*L*τ, with *L* the graph's Laplacian matrix
(degrees on the diagonal, −1 for every connection).

## What it is for

If tokens behaved like heat, an agent's change of tokens in a game would
follow its curvature at the start of the game. [Chapter 20](../chapters/20-gains-and-losses.md)
tests that, and [Chapter 21](../chapters/21-where-the-tokens-flow.md) explains
why something like it should happen: an agent that splits its stake evenly
over itself and its neighbours sends tokens **down** the slope.

## In the viewer

`token_curvature` is measured on the frame itself; `token_curvature_pre` on
the graph and tokens as they were when the phase began, so that it can be read
against the change the phase then produced. On a signed colour scale, blue is
a peak and red a valley.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
