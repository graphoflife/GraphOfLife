# Cut risk

Counting [bridges](bridges.md) says how many [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") lie on no [loop](loops.md "A path through the network that comes back to where it started without using a connection twice."). It
does not say how badly the network would break if one were cut: a bridge to a
single leaf costs one [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), a bridge between two halves costs half the world.
The **cut risk** measures the worst case.

## Definition

Removing a bridge splits the network into two pieces. Call the smaller one the
part **behind** the bridge. The cut risk is the largest such part, as a share
of all agents:

$$
\texttt{cutRisk} = \max_{\text{bridges } e} \frac{\min\bigl(|A_e|, |B_e|\bigr)}{|V|} ,
$$

where *A*ₑ and *B*ₑ are the two pieces left by removing *e*. It is 0 when there
is no bridge, and at most ½ — a network split down the middle by one
connection.

## Before and after

- `cutRisk` is measured on the [frame](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections."), after the phase.
- `cutRiskBefore` is measured on the network as it stood **before** the
  phase's cleanup removed anybody. A frame shows the world after the cull, so
  a cut that removes a whole side changes the network and the death toll at
  once; measured beforehand, the cut risk can be read as a *predictor* of the
  cull rather than a consequence of it.

## Why it matters here

The cleanup keeps only the largest piece of the network
([One iteration, step by step](one-iteration.md)). A bridge with a tenth of the
world behind it is a tenth of the world that dies the moment no [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") crosses
that one connection in a game. [How a world breaks](how-a-world-breaks.md)
asks how often that happens.

## How it is computed

Tarjan's depth-first walk finds every bridge ([Bridges](bridges.md)), and the
same walk counts the agents below each one, so the size of the part behind
every bridge costs nothing extra (`bridge_splits` and `worst_cut_share` in
`GraphOfLifeSimple.py`).

<!-- turns -->
---

← [Loops](loops.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Radius and diameter](radius-and-diameter.md) →

> [!info]- Pages that link here (9)
> [Glossary](glossary.md) · [The life of a B1 world](a-worlds-life.md) · [The shape of the network](what-shape-does-the-network-take.md) · [How a world breaks](how-a-world-breaks.md) · [Bridges](bridges.md) · [Physical inspiration](physical-inspiration.md) · [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) · [The health bar](health-bar.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
