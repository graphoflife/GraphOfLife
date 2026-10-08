# Lightning: tokens that go round

Does the flow of a game ([Token flow](token-flow.md)) go round in circles —
tokens leaving an agent, crossing several connections, and arriving back
where they started? The viewer calls such a closed loop of flow a
**lightning**. Ecologists measure the same thing in food webs as a *cycling
index* (Finn 1976). This note explains what is measured and why only bounds
can be given.

## Loops of flow

Think of the flow as a directed network: an arrow from *u* to *v* carrying
*a*(*u* → *v*) tokens. A **loop** of length *L* is a closed path
*u*₁ → *u*₂ → … → *u*_L → *u*₁ along arrows that carry at least one token.
Taking one token off every arrow of the loop removes *L* tokens of flow that
went round. A loop of length *L* **scores** *L*², so that one long circuit
counts for more than many short ones.

## The greedy peel

Finding the loops that use the most flow is as hard as finding the longest
cycle in a network — a problem with no efficient exact solution. So the
program peels loops greedily:

1. Start at an agent (in increasing order of id) and walk along arrows that
   still carry tokens, preferring an agent not yet visited in this walk.
2. As soon as the walk reaches an agent already on it, a loop is closed:
   remove one token from each arrow of the loop, and add *L*² to the score.
3. Repeat until no loop can be found from any start.

| field | definition |
|---|---|
| `lightningScore` | Σ *L*² over the loops peeled |
| `cyclingShare` | the tokens of flow the loops used, ÷ `totalFlow` |
| `lightningLongest` | the longest loop found |

Being greedy, these are **lower bounds**: the true maximum is at least this.

## The ceiling: what conservation forbids

Every token that goes round a loop leaves an agent and comes back to it, so
loops change no agent's balance. Whatever net imbalance there is — agents
that received more than they sent, or the reverse — cannot be in loops:

$$
\texttt{flowImbalance} = \frac{\tfrac12 \sum_u \bigl| \text{in}(u) - \text{out}(u) \bigr|}{\texttt{totalFlow}} .
$$

(Halved because every stranded token appears twice: once as a surplus, once
as a deficit.) This is exact, and an **upper bound**: at most
1 − `flowImbalance` of the flow can go round. The truth lies between
`cyclingShare` and that ceiling.

## The net version

Two neighbours staking on each other form a loop of length 2 — circulation of
the most trivial kind. The `net…` statistics repeat everything on the net flow
(each pair's two directions cancelled), where loops of length 2 cannot exist:
`netLightningScore`, `netCyclingShare` (÷ the net total), `netLightningLongest`.
A loop that survives cancellation is tokens genuinely going *round*.

## An example

Agents A, B, C. A stakes 3 on B, B 3 on C, C 3 on A, and B 2 on A. Total flow
11. The loop A → B → C → A can be peeled three times: 9 tokens, score 3·9 =
27. What remains is B → A (2), with no loop: `cyclingShare` = 9/11. The
imbalance: A received 3 + 2 = 5 and sent 3 (+2); B received 3, sent 5 (−2);
C received 3, sent 3 (0); half of 2 + 2 is 2, so `flowImbalance` = 2/11 and the
ceiling 9/11 — the greedy peel found everything there was.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
