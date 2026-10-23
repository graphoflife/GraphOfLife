# Flows that go round

[Tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") move over the network every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."): an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s stake on a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s
node goes to that node, whoever takes it. Add up what each [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") carried
each way over a stretch of games and every connection has a **[net flow](token-flow.md "The tokens a game moves along each connection, from the agent who stakes to the node staked on.")**: the
tokens that went one way minus those that came back. This note shows how such
a flow splits into a part that runs downhill and a part that goes round, and
how to tell whether a world's flow goes round more than chance would make it.

## Two kinds of flow

Write the net flow on the connection from agent $u$ to agent $v$ as
$F_{uv}$, with $F_{vu} = -F_{uv}$. Two kinds of flow cannot be mistaken for
each other.

A **gradient** flow runs downhill along a *pressure* $\varphi$, one number per
agent:

$$F_{uv} = \varphi_u - \varphi_v .$$

Tokens leave the high agents and arrive at the low ones. Round any [loop](loops.md "A path through the network that comes back to where it started without using a connection twice.") of the
network a gradient flow adds up to nothing, since the differences cancel:
$(\varphi_a - \varphi_b) + (\varphi_b - \varphi_c) + (\varphi_c - \varphi_a) = 0$.

A **circulation** is the opposite: at every agent as much arrives as leaves,
so it has no sources and no sinks. It can only go round loops, and a network
without loops (a tree) carries none.

Every flow on a network is the sum of exactly one gradient flow and exactly
one circulation, and the two are perpendicular: their energies, the sums of
their squared values over the connections, add up to the flow's. This is the
discrete **Helmholtz–Hodge decomposition** (Jiang, Lim, Yao and Ye, 2011, use it
to rank from pairwise comparisons; Candogan, Menache, Ozdaglar and Parrilo,
2011, to split games into their potential and cyclic parts). The gradient part
is found by [least squares](least-squares.md "The straight line that fits a cloud of points best, and how much of their spread it explains."), the pressure that explains as much of the flow as
possible, which means solving one linear system with the network's Laplacian.
The circulation is what is left over. The **circulation share** is the
circulation's energy as a fraction of the flow's.

## A worked example

Four agents: a triangle $A, B, C$ and a fourth, $D$, hanging off $C$. Over some
games, net 3 tokens went from $A$ to $B$, 3 from $B$ to $C$, 1 from $C$ to $A$,
and 2 from $C$ to $D$.

The connection $C$–$D$ lies on no loop, so its flow can only be gradient: all 2
tokens of it. Round the triangle the flow adds up to $3 + 3 + 1 = 7$ in the
direction $A \to B \to C \to A$. The circulation takes a third of that loop
sum to each of the loop's three connections, $7/3 \approx 2.33$ round the loop. The
gradient keeps the rest: $3 - 7/3 = 2/3$ from $A$ to $B$, $2/3$ from $B$ to $C$
and $1 - 7/3 = -4/3$ from $C$ to $A$, that is $4/3$ from $A$ to $C$. That is
the pressure $\varphi_A > \varphi_B > \varphi_C > \varphi_D$, with steps of $2/3$,
$2/3$ and $2$.

The energies: the flow's is $3^2 + 3^2 + 1^2 + 2^2 = 23$; the circulation's is
$3 \cdot (7/3)^2 = 49/3$; the gradient's is $(2/3)^2 + (2/3)^2 + (4/3)^2 + 2^2 = 20/3$.
They add up, $49/3 + 20/3 = 23$, and the circulation share is
$49/69 \approx 0.71$.

## Against what?

A share of 0.71 means nothing without a level to compare it with, and two
are used.

**The random level.** If every connection carried a flow of the same size in
a random direction, the expected circulation share would be the share of the
network's independent loops among its connections,

$$\frac{E - N + C}{E},$$

with $E$ connections, $N$ agents and $C$ separate pieces. In the example
$E = 4$, $N = 4$, $C = 1$: a quarter. On a tree it is zero; the more loops,
the higher.

**The sign-flip level.** Real connections carry very different amounts, and a
connection that carries much counts for much. The fairer level keeps each
connection's amount and throws only its direction at random. Its expectation
is

$$\frac{\sum_e P_{ee} F_e^2}{\sum_e F_e^2},$$

where $P_{ee}$ is how much of a unit flow on connection $e$ alone is
circulation: one third for each side of the triangle, zero for the [bridge](bridges.md "A connection that lies on no loop: cut it and the network falls into two pieces.")
$C$–$D$. In the example it is $(9 + 9 + 1)/3/23 = 19/69 \approx 0.28$, and averaging
over all sixteen ways of flipping the four directions gives the same. In a
world it is estimated by flipping the directions at random a few times.

The example's 0.71 is far above both levels: its flow goes round. A share
*below* the sign-flip level means the opposite, a flow whose directions line up
downhill more than chance: tokens running from some agents to others.

## Does a ring last?

A ring that is there in one stretch of games and gone in the next is an
accident of those games. A ring that **persists** is a structure. Split the
flow of two consecutive stretches (50 games each in
[Gliders, and who helps whom](gliders-and-who-helps-whom.md)), and over the connections both used
take the [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of their circulations, and of their gradient parts. A
correlation near zero means each stretch's rings are new; a positive one,
rings or slopes that come back on the same connections.

## In this notebook

`research/oee/checks/circulation.py` does all of the above for a [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") run's
recorded games, for the whole flow and separately for **gifts** (stakes that
did not take their target, so their tokens went to whoever held it) and
**takeovers** (stakes that took it). Gifts going round would be generalised
reciprocity: every agent gives to one and receives from another, like the
armshells and necklaces of the Kula ring that Bronisław Malinowski described
in *Argonauts of the Western Pacific* (1922). What the worlds of this notebook do
instead is in [Gliders, and who helps whom](gliders-and-who-helps-whom.md).

## What it can and cannot say

A flow added up over a stretch of games forgets *when* tokens moved, and the
network it lives on is every connection that carried anything in that
stretch, though not all of them existed at once. Two-way traffic on one
connection, $u$ to $v$ and $v$ to $u$, cancels in the net flow and is invisible
here: direct give-and-take between two agents has to be looked for in the
gross flows. And a long route through the network counts largely as
circulation even if it never closes. One token carried along a route of $L$
connections from $a$ to $b$ has energy $L$. Its gradient part is the electrical
flow from $a$ to $b$, spread over every way between them, and its energy is
the effective resistance between $a$ and $b$, which on a network with many
loops is far less than $L$. The rest is circulation. So coherent routes, not
only closed rings, raise the share.

<!-- turns -->
---

← [Token curvature](token-curvature.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [Flow modules and the map equation](map-equation.md) →

> [!info]- Pages that link here (6)
> [Glossary](glossary.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [Lightning in every world](lightning-in-every-world.md) · [Who helps whom](who-helps-whom.md) · [Gliders, and who helps whom](gliders-and-who-helps-whom.md) · [Help](help.md)
<!-- /turns -->
