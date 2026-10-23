# The spectral gap

How hard is a network to cut into two halves? Counting [bridges](bridges.md "A connection that lies on no loop: cut it and the network falls into two pieces.") answers for
cuts of a single [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."); the **spectral gap** λ₂ answers for every way of
dividing the network at once. It is also the speed at which anything that
spreads along the connections — a random walker, heat, [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") — forgets where
it started.

## Step by step

1. **The random walk.** A walker at [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") *u* moves to a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") chosen
   uniformly at random. After many steps, where is it? On a connected network
   it ends up at agent *u* with probability proportional to deg(*u*), whatever
   it started from.
2. **How fast?** The difference from that resting distribution shrinks, step
   by step, roughly by a fixed factor (1 − λ₂). After about 1/λ₂ steps the
   start is forgotten. λ₂ near 1 means a few steps suffice; λ₂ near 0 means
   the walker stays in its own region for a long time.
3. **Why a region traps it.** A walker leaves a region only through the
   connections that lead out of it. If a big region has few such connections —
   a bottleneck — the walk lingers. So a small λ₂ means there is a cheap cut
   somewhere.

## The definition

λ₂ is the second-smallest eigenvalue of the **normalised Laplacian**

$$
\mathcal{L} = I - D^{-1/2} A D^{-1/2} ,
$$

where *A* is the adjacency matrix (*A*ᵤᵥ = 1 if *u* and *v* are joined) and *D*
the diagonal matrix of [degrees](degree.md "The number of an agent's connections: how many neighbours it has."). Its eigenvalues lie between 0 and 2; the
smallest is always 0, and λ₂ > 0 exactly when the network is connected.

**Cheeger's inequality** ties λ₂ to the network's worst bottleneck *h*, the
smallest ratio of connections leaving a set to the connections inside it
(over every set holding at most half the network):

$$
\frac{h^2}{2} \;\le\; \lambda_2 \;\le\; 2h .
$$

A small λ₂ promises that a good cut exists; a large one proves that none does.

## Examples

- A complete network of *n* agents: λ₂ = *n*/(*n* − 1), about 1.
- A ring of *n* agents: λ₂ = 1 − cos(2π/*n*) ≈ 2π²/*n*², tiny for large *n* —
  a ring is easy to cut (twice) into two arcs.
- A settled [baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") world: about 0.0012 at 100,000 tokens, 0.0026 at
  10,000 ([The geometry of a world](the-geometry-of-a-world.md)) — a random
  walk needs on the order of 1/0.0012 ≈ 800 steps (or 1/0.0026 ≈ 400) to
  forget its start.

## How it is computed

By the Lanczos method on the largest connected piece (`gol_spectral.py`):
48 steps of an [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") that builds the part of the spectrum near the
bottom, then the smallest eigenvalue of the small tridiagonal matrix that
results, by bisection. It is an upper bound on λ₂, exact on well-connected
networks and, on nearly divided ones, a small number whose smallness matters
more than its last digit.

<!-- turns -->
---

← [Radius and diameter](radius-and-diameter.md) · [The shape and health of the network](../README.md#the-shape-and-health-of-the-network) · [Assortativity](assortativity.md) →

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [The geometry of a world](the-geometry-of-a-world.md) · [Physical inspiration](physical-inspiration.md) · [Worlds of 100,000 tokens](worlds-of-100000-tokens.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
