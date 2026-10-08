# Token flow

In every game every agent stakes all its tokens on itself and its neighbours
([Chapter 3](../chapters/03-one-iteration.md)). A stake of agent *u* on a
neighbour *v*'s node is a **flow** of tokens from *u* to *v*: whatever happens
to the node, those tokens end the game at *v*'s node. These statistics, in the
viewer's *Game (Blotto)* section, describe that flow. A stake on one's own
node moves nothing and is left out of all of them.

## The flow of a game

Write *a*(*u* → *v*) for the tokens *u* staked on neighbour *v*'s node.

| field | definition |
|---|---|
| `totalFlow` | Σ over all *u* ≠ *v* of *a*(*u* → *v*): every token staked on a neighbour |
| `meanEdgeFlow` | the mean, over the connections that carried anything, of *a*(*u* → *v*) + *a*(*v* → *u*) |
| `maxEdgeFlow` | the largest such sum on one connection |
| `netFlowShare` | the share of `totalFlow` left after cancelling each pair's two directions against each other (below) |
| `flowImbalance` | the share of `totalFlow` that must flow one way ([Lightning](lightning.md)) |
| `spreadShare` | the share of the agents that staked who spread their tokens rather than going all in |
| `revoltShare` | the share of all staked tokens marked revolutionary |
| `prunedEdges` | the connections removed at the end of the game because nothing crossed them |

## Cancelling: the net flow

If *u* puts 5 tokens on *v*'s node and *v* puts 3 on *u*'s, 3 tokens went
each way and cancel: the **net** flow is 2 from *u* to *v*. Doing this for
every pair gives a flow in which every connection carries tokens in at most
one direction:

$$
\text{net}(u, v) = \max\bigl(0,\ a(u \to v) - a(v \to u)\bigr) .
$$

`netFlowShare` = Σ net / `totalFlow`. It is 1 when nothing is returned and 0
when every stake is matched exactly by one coming back. Cancelling changes no
agent's balance: it removes equal amounts going both ways.

## Why flow matters

Two rules make flow the life of a connection:

- a connection across which **no** tokens flow in a game is cut at its end
  ([`prune_after`](settings.md#prune_after), [`inactive_window`](settings.md#inactive_window));
- a node's tokens after the game are **exactly** the tokens staked on it.

So the flow decides which connections survive and how the tokens are
reshuffled. [Chapter 20](../chapters/20-where-the-tokens-flow.md) shows where
it goes.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
