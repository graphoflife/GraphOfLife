# Cut risk

Counting [bridges](bridges.md) says how many connections lie on no loop. It
does not say how badly the network would break if one were cut: a bridge to a
single leaf costs one agent, a bridge between two halves costs half the world.
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

- `cutRisk` is measured on the frame, after the phase.
- `cutRiskBefore` is measured on the network as it stood **before** the
  phase's cleanup removed anybody. A frame shows the world after the cull, so
  a cut that removes a whole side changes the network and the death toll at
  once; measured beforehand, the cut risk can be read as a *predictor* of the
  cull rather than a consequence of it.

## Why it matters here

The cleanup keeps only the largest piece of the network
([Chapter 3](../chapters/03-one-iteration.md)). A bridge with a tenth of the
world behind it is a tenth of the world that dies the moment no token crosses
that one connection in a game. [Chapter 25](../chapters/25-how-a-world-breaks.md)
asks how often that happens.

## How it is computed

Tarjan's depth-first walk finds every bridge ([Bridges](bridges.md)), and the
same walk counts the agents below each one, so the size of the part behind
every bridge costs nothing extra (`bridge_splits` and `worst_cut_share` in
`GraphOfLifeSimple.py`).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
