# Box dimension

A second way to give a network a [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space."): cover it with boxes and count
them (Song, Havlin and Makse 2005). It is the network version of the
box-counting dimension of fractals.

## The idea

Cover a shape with boxes of side ℓ and count how many are needed, *N*(ℓ). For
a line of length 1, *N* = 1/ℓ; for a square, (1/ℓ)²; for a cube, (1/ℓ)³. In
general

$$
N(\ell) \propto \ell^{-d_B} ,
$$

and *d*_B is the **box dimension**. A fractal can have a *d*_B that is not a
whole number.

## On a network

A **box of size ℓ** is a set of [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") no two of which are more than ℓ − 1
steps apart. The viewer builds boxes as balls: a centre and every agent within
(ℓ − 1)/2 steps of it — two agents in such a ball are at most ℓ − 1 apart.
Covering a network with the fewest boxes is a hard problem, so it is done
greedily:

1. Order the agents by number of [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."), most first.
2. Go down the list; every agent not yet in a box becomes the centre of a new
   box, which takes every agent within (ℓ − 1)/2 steps of it that no box holds
   yet.
3. Count the boxes, for ℓ = 1, 3, 5, 9, 17, 33 (stopping once one box holds
   everything).

Then fit ln *N*(ℓ) against ln ℓ by [least squares](least-squares.md):

$$
\texttt{boxDimension} = -\text{slope}, \qquad \texttt{boxDimensionR2} = R^2 .
$$

**Example.** In the world of [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 after its last [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."): 1,336 boxes of size 1
(every agent alone), 460 of size 3, 122 of size 5, 41 of size 9, 5 of size 17,
2 of size 33 — a slope of about −2, a box dimension of about 2
([The geometry of a world](what-shape-does-the-network-take.md#the-geometry-of-a-world)).

## Reading the R²

Small-world networks are covered in a number of boxes that falls
**exponentially** in ℓ rather than as a power, and over a short range even an
exponential looks respectable on log–log axes. A high R² is necessary for the
dimension to mean something, but not sufficient.

<!-- turns -->
---

← [The spectral dimension](spectral-dimension.md) · [Space and dimension](../README.md#space-and-dimension)

> [!info]- Pages that link here (5)
> [Glossary](glossary.md) · [The shape of the network](what-shape-does-the-network-take.md) · [Physical inspiration](physical-inspiration.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
