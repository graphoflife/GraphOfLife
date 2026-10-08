# The spectral gap

How hard is a network to cut into two halves? Counting bridges answers for
cuts of a single connection; the **spectral gap** λ₂ answers for every way of
dividing the network at once. It is also the speed at which anything that
spreads along the connections — a random walker, heat, tokens — forgets where
it started.

## Step by step

1. **The random walk.** A walker at agent *u* moves to a neighbour chosen
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
the diagonal matrix of degrees. Its eigenvalues lie between 0 and 2; the
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
- A settled baseline world: about 0.0026 ([Chapter 24](../chapters/24-the-geometry-of-a-world.md)) —
  a random walk needs on the order of 1/0.0026 ≈ 400 steps to forget its start.

## How it is computed

By the Lanczos method on the largest connected piece (`gol_spectral.py`):
48 steps of an iteration that builds the part of the spectrum near the
bottom, then the smallest eigenvalue of the small tridiagonal matrix that
results, by bisection. It is an upper bound on λ₂, exact on well-connected
networks and, on nearly divided ones, a small number whose smallness matters
more than its last digit.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
