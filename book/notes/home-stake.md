# Staking at home, and keeping one's node

In the game every agent stakes all its tokens on its **candidates**: its own
node and its neighbours' nodes ([Chapter 3](../chapters/03-one-iteration.md)).
Two questions about this: how much do agents stake **at home**, on their own
node? And how often does an agent **keep** its node — win it itself? This
note defines the statistics that answer them.

## How much is staked at home

`selfAllocationShare`, in every game row: of all the tokens staked in the
game, the share staked by agents on their own node,

$$
\text{selfAllocationShare} = \frac{\sum_u a_{u \to u}}{\sum_u \tau(u)} ,
$$

where *a*ᵤ→ᵤ is what agent *u* staked on its own node and τ(*u*) everything
it staked (all its tokens). It is a share of **tokens**, so a rich agent
counts more than a poor one.

The book also looks at single agents: for an agent *u*, the share of its own
stake it put at home, *a*ᵤ→ᵤ / τ(*u*). This is read from the decisions in the
frames ([What a run records](frames-and-stats.md)).

## How often an agent keeps its node

`heldHomeShare`, in every game row: of all nodes on which anyone staked
anything, the share won by **their own agent**. A node that is not kept was
won by a neighbour (alone or with a coalition), and its agent now carries a
copy of that neighbour's brain.

## How often a coalition wins

`revolutions`, in every game row: the number of nodes won by a coalition
([How a coalition takes a node](revolution.md)). The book divides it by the
number of agents at the start of the game, `revolutions/nodes_before`.

## Why they matter

Together they say what kind of contest the game is. If agents staked
everything at home and always kept their nodes, the game would change
nothing; brains would be passed on only to children. In fact, in a settled
baseline world agents stake about 30% of their tokens at home, keep their
node in about 45% of games, and about half of all nodes are won by a
coalition rather than by their largest staker ([Chapter 15](../chapters/15-how-the-game-is-played.md)).

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
