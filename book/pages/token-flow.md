# Measuring the flow: token flow, lightning and circulation

## In short

In every game, the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") stakes on a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s node flow along
the [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") between them: that is the **token flow**. Once each pair's two
directions cancel, what is left is the **net flow**. A **lightning** is a
closed loop of the flow, and only a floor and a ceiling can be given for how
much goes round. Over a stretch of games, the net flow splits exactly into a
part that runs downhill and a **circulation** that goes round.

## Token flow

Picture each token an agent stakes on a neighbour walking across the
connection between them. In every game every agent stakes all its tokens on itself and its neighbours
([One iteration, step by step](one-iteration.md)). A stake of agent *u* on a
neighbour *v*'s node is a **flow** of tokens from *u* to *v*. Whatever happens
to the node, those tokens end the game at *v*'s node. A stake on one's own
node moves nothing, and the measures of flow leave it out. The viewer shows
them in its *Game (Blotto)* section ([What a run records](frames-and-stats.md)).

### The flow of a game

Write *a*(*u* → *v*) for the tokens *u* staked on neighbour *v*'s node. The
first five fields below measure the flow itself. The last three say how the
agents staked, and which connections the game cut.

| field | definition |
|---|---|
| `totalFlow` | Σ over all *u* ≠ *v* of *a*(*u* → *v*): every token staked on a neighbour |
| `meanEdgeFlow` | the mean, over the connections that carried anything, of *a*(*u* → *v*) + *a*(*v* → *u*) |
| `maxEdgeFlow` | the largest such sum on one connection |
| `netFlowShare` | the share of `totalFlow` left after cancelling each pair's two directions against each other ([below](#cancelling-the-net-flow)) |
| `flowImbalance` | the share of `totalFlow` that must flow one way ([The ceiling: what conservation forbids](#the-ceiling-what-conservation-forbids)) |
| `spreadShare` | the share of the agents that staked who spread their tokens rather than going all in ([What a brain says](brain-outputs.md)) |
| `revoltShare` | the share of all staked tokens, [home stakes](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node "The tokens an agent stakes on its own node, to keep it.") included, marked revolutionary ([How a coalition takes a node](revolution.md)) |
| `prunedEdges` | the connections cut at the end of the game because nothing crossed them ([Why flow matters](#why-flow-matters)) |

A run records the first three and the last three after every game. The
other two, `netFlowShare` and `flowImbalance`, are worked out with the loops
of [Lightning](#lightning-tokens-that-go-round), after the games of every
25th iteration.

**Example.** Three agents, A, B and C, are joined in a triangle. In one game,
A stakes 3 tokens on B's node, B stakes 3 on C's and 2 on A's, and C stakes 3
on A's. Everything else they hold, they stake on their own nodes. Then
`totalFlow` is 3 + 3 + 2 + 3 = 11 tokens. The connection A–B carried
3 + 2 = 5 tokens, and B–C and C–A carried 3 each. So `meanEdgeFlow` is 11/3,
about 3.7, and `maxEdgeFlow` is 5.

### Cancelling: the net flow

Say *u* puts 5 tokens on *v*'s node, and *v* puts 3 on *u*'s. Then 3 tokens
went each way and cancel: the **net** flow is 2 from *u* to *v*. Doing this for
every pair gives a flow in which every connection carries tokens in at most
one direction:

$$
\text{net}(u, v) = \max\bigl(0,\ a(u \to v) - a(v \to u)\bigr) .
$$

In words: the net flow from *u* to *v* is what *u* sent *v* minus what *v*
sent back. It is nothing if *v* sent back as much or more. Then
`netFlowShare` = Σ net / `totalFlow`. It is 1 when nothing is returned and 0
when every stake is matched exactly by one coming back. Cancelling changes no
agent's balance: it removes equal amounts going both ways.

In the example, the 3 and the 2 on A–B cancel to 1 from A to B. The net flow
is 1 + 3 + 3 = 7 tokens, and `netFlowShare` is 7/11, about 0.64.

The same cancelling over a stretch of games gives the net flow of the
stretch. [Flows that go round](#flows-that-go-round) splits it in two.

### Why flow matters

Two rules make the flow matter:

- In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it."), a connection across which **no** tokens flow in a game is cut at
  its end ([`prune_after`](settings.md#prune_after),
  [`inactive_window`](settings.md#inactive_window)). In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") it is cut only if
  its two ends share at least two neighbours
  ([`prune_needs`](settings.md#prune_needs)), so no cut can split a world.
- When the game is decided, a node holds **exactly** the tokens staked on it,
  whoever wins it.

So the flow decides which connections survive, and how the tokens are
reshuffled.

### Where token flow is used

- **The budget of a game.** A settled B1 world of 100,000 tokens keeps 30% of
  its tokens at home in a game. Another 51% are matched by a stake coming
  back, and 20% go one way (49% and 21% at 10,000 tokens). So `totalFlow` is
  about 71% of the tokens, and `netFlowShare` about 0.28
  ([The budget of a game](where-the-tokens-flow.md#the-budget-of-a-game)).
- **The connections.** Two connections in three carry one or two tokens a
  game, most of them one each way. In B1, every connection that carried
  nothing is cut
  ([What crosses a connection](where-the-tokens-flow.md#what-crosses-a-connection)).
- **B2.** In B2's worlds almost nothing cancels: `netFlowShare` is about
  0.95. A connection there can survive unused, and agents stake mostly on
  others who do not stake back
  ([Every world](where-the-tokens-flow.md#every-world)).
- **The viewer** can colour every connection by its token flow, both
  directions together ([What the viewer can colour by](viewer-colours.md)).

## Lightning: tokens that go round

Does the flow of a game go round in circles? Picture tokens leaving an
agent, crossing several connections and arriving back where they started,
like current round a circuit. The viewer calls such a closed loop of flow a
**lightning**. Ecologists measure the same thing in food webs as a *cycling
index* (Finn 1976). This section explains what is measured, and why only
bounds can be given.

### Loops of flow

Think of the flow of a game ([Token flow](#token-flow)) as a directed
network: an arrow from *u* to *v* carrying *a*(*u* → *v*) tokens. A **loop**
of length *L* is a closed path along *L* arrows,
*u*₁ → *u*₂ → … → *u*_L → *u*₁. Each of its arrows carries at least one
token. Taking one token off every arrow of the loop removes *L* tokens of
flow that went round. A token taken off cannot be counted in a second loop.
A loop of length *L* **scores** *L*², so that one long circuit counts for
more than many short ones.

### The greedy peel

Finding the loops that use the most flow is as hard as finding the longest
loop in a network. No fast exact method is known for that. So the program
(`gol_lightning.py`) peels loops greedily:

1. Start at an agent, in increasing order of id. Walk along arrows that
   still carry tokens, preferring an agent not yet visited in this walk.
2. As soon as the walk reaches an agent already on it, a loop is closed.
   Remove one token from each arrow of the loop, and add *L*² to the score.
3. Walk again from the same agent while its walks keep closing loops. As
   soon as a walk ends at an agent with no arrow left to follow, move on to
   the next agent. The walk from an agent always takes the same path: the
   first unvisited target in order of id. So an agent is given up even when
   another path from it would have closed a loop.

Two limits keep the search short. A walk is given up after 4,096 steps, and
the peel of one flow stops after 20,000 loops.

| field | definition |
|---|---|
| `lightningScore` | Σ *L*² over the loops peeled |
| `cyclingShare` | the tokens of flow the loops used, ÷ `totalFlow` |
| `lightningLongest` | the longest loop found |

Being greedy, these are **lower bounds**: the true maximum is at least this.

### The ceiling: what conservation forbids

Every token that goes round a loop leaves an agent and comes back to it, so
loops change no agent's balance. Whatever net imbalance there is — agents
that received more than they sent, or the reverse — cannot be in loops.
Write in(*u*) for the tokens *u*'s neighbours staked on its node, and
out(*u*) for the tokens *u* staked on theirs. Then

$$
\texttt{flowImbalance} = \frac{\tfrac12 \sum_u \bigl| \text{in}(u) - \text{out}(u) \bigr|}{\texttt{totalFlow}} .
$$

In words: take the gap between what each agent received and what it sent,
counted positive either way. Add up the gaps, halve the sum, and divide by
the whole flow. It is halved because every stranded token appears twice:
once as a surplus, once as a deficit. This is exact, and an **upper bound**:
at most 1 − `flowImbalance` of the flow can go round. The truth lies between
`cyclingShare` and that ceiling.

### The net version

Two neighbours staking on each other form a loop of length 2:
[circulation](#flows-that-go-round) of the most trivial kind. The `net…`
statistics repeat everything on the net flow, in which each pair's two
directions have cancelled ([Cancelling: the net flow](#cancelling-the-net-flow)).
There, loops of length 2 cannot exist. The statistics are
`netLightningScore`, `netCyclingShare` (÷ the net total) and
`netLightningLongest`. A loop that survives cancellation is tokens genuinely
going *round*.

Cancelling changes no agent's balance, so the same stranded tokens bound the
net flow too. A share `flowImbalance` ÷ `netFlowShare` of the net flow must
run one way, and at most the rest can go round.

### An example: three agents

Take agents A, B and C, as in [Token flow](#the-flow-of-a-game). A stakes 3
on B, B 3 on C, C 3 on A, and B 2 on A. The total flow is 11. The loop
A → B → C → A can be peeled three times. That uses 9 tokens, and each peel
scores 3² = 9, so the score is 3 · 9 = 27. What remains is B → A (2), with no
loop, so `cyclingShare` = 9/11.

The imbalance: A received 3 + 2 = 5 and sent 3 (+2). B received 3 and sent
5 (−2). C received 3 and sent 3 (0). Half of 2 + 2 is 2, so
`flowImbalance` = 2/11, and the ceiling is 9/11. The greedy peel found
everything there was.

After cancelling, the net flow is A → B (1), B → C (3) and C → A (3): 7
tokens. The loop can now be peeled only once, so `netCyclingShare` = 3/7
and `netLightningLongest` = 3. The same 2 tokens are stranded, so the
ceiling allows 5 of the 7. But this time the floor, 3/7, is the truth. The 2
stranded tokens run B → C → A, two steps each, so they use up 4 tokens of the
net flow, not 2.

### Where lightning is used

A run records these statistics after the games of every 25th iteration
([What a run records](frames-and-stats.md)).

- **Do tokens go round?** In B1's worlds of 100,000 tokens, every matched
  stake is a loop of two. So between 72% and 77% of the flow as staked goes
  round, yet the greedy peel finds only 13%: a weak floor. After cancelling,
  it finds 0.3% of the net flow on loops, where the ceiling allows 16%
  ([Do tokens go round?](where-the-tokens-flow.md#do-tokens-go-round)).
- **Every world.** [Lightning in every world](where-the-tokens-flow.md#lightning-in-every-world)
  reads the floor in every world the notebook has made. One rule decides how
  much goes round: whether a connection must carry tokens to survive. And the
  longest loop found is as long as a world is wide.
- **Over many games.** The greedy count of single games is set beside the
  exact split of fifty games in
  [A greedy count and an exact one](where-the-tokens-flow.md#a-greedy-count-and-an-exact-one).
  The split is explained under [Flows that go round](#flows-that-go-round).
- **A yardstick for evolution.** How much of the flow goes round is one
  statistic that might keep rising in a world that keeps evolving
  ([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).

## Flows that go round

Tokens move over the network every game: an agent's stake on a neighbour's
node goes to that node, whoever takes it. Add up what each connection carried
each way over a stretch of games. Then every connection has a
[net flow](#cancelling-the-net-flow): the tokens that went one way minus
those that came back. Picture water on a hillside. Some of it runs downhill,
from high ground to low; some turns in eddies, which have no top and no
bottom. This section shows how a flow splits into those two parts. It also
shows how to tell whether a world's flow goes round more than chance would
make it.

### Two kinds of flow

Write the net flow on the connection from agent $u$ to agent $v$ as
$F_{uv}$, with $F_{vu} = -F_{uv}$. Two kinds of flow cannot be mistaken for
each other.

A **gradient** flow runs downhill along a *pressure* $\varphi$, one number per
agent:

$$F_{uv} = \varphi_u - \varphi_v .$$

In words: the flow from $u$ to $v$ is how much higher $u$'s pressure is than
$v$'s. Tokens leave the high agents and arrive at the low ones. Round any loop
of the network a gradient flow adds up to nothing, since the differences
cancel:
$(\varphi_a - \varphi_b) + (\varphi_b - \varphi_c) + (\varphi_c - \varphi_a) = 0$.

A **circulation** is the opposite: at every agent as much arrives as leaves,
so it has no sources and no sinks. It can only go round loops, and a network
without loops (a tree) carries none.

Every flow on a network is the sum of exactly one gradient flow and exactly
one circulation. The two are perpendicular: their **energies** add up to the
flow's. A flow's energy is the sum of its squared values over the
connections. This is the discrete **Helmholtz–Hodge decomposition**. Jiang,
Lim, Yao and Ye (2011) use it to rank from pairwise comparisons. Candogan,
Menache, Ozdaglar and Parrilo (2011) use it to split games into their
potential and cyclic parts.

The gradient part is found by [least squares](least-squares.md "The straight line that fits a cloud of points best, and how much of their spread it explains."): the pressure that explains as
much of the flow as possible. That means solving one linear system with the
network's Laplacian. That matrix holds each agent's number of connections on
its diagonal, and −1 wherever two agents are joined. The circulation is what
is left over. The **circulation share** is the circulation's energy as a
fraction of the flow's.

### A worked example

Four agents: a triangle $A, B, C$ and a fourth, $D$, hanging off $C$. Over some
games, net 3 tokens went from $A$ to $B$, 3 from $B$ to $C$, 1 from $C$ to $A$,
and 2 from $C$ to $D$.

The connection $C$–$D$ lies on no loop, so its flow can only be gradient: all 2
tokens of it. Round the triangle the flow adds up to $3 + 3 + 1 = 7$ in the
direction $A \to B \to C \to A$. The circulation takes a third of that loop
sum to each of the loop's three connections, $7/3 \approx 2.33$ round the loop. The
gradient keeps the rest: $3 - 7/3 = 2/3$ from $A$ to $B$, $2/3$ from $B$ to $C$
and $1 - 7/3 = -4/3$ from $C$ to $A$, that is $4/3$ from $A$ to $C$. Round the
triangle the gradient adds up to $2/3 + 2/3 - 4/3 = 0$, as it must. It comes
from the pressure $\varphi_A > \varphi_B > \varphi_C > \varphi_D$, with steps
of $2/3$, $2/3$ and $2$.

The energies: the flow's is $3^2 + 3^2 + 1^2 + 2^2 = 23$; the circulation's is
$3 \cdot (7/3)^2 = 49/3$; the gradient's is $(2/3)^2 + (2/3)^2 + (4/3)^2 + 2^2 = 20/3$.
They add up, $49/3 + 20/3 = 23$, and the circulation share is
$49/69 \approx 0.71$.

### Against what?

A share of 0.71 means nothing without a level to compare it with, and two
are used.

**The random level.** Suppose every connection carried a flow of the same
size, in a random direction. Then the expected circulation share is the
network's number of independent loops, its cycle rank
([Loops](bridges.md#loops)), as a share of its connections:

$$\frac{E - N + C}{E},$$

with $E$ connections, $N$ agents and $C$ separate pieces. In the example
$E = 4$, $N = 4$, $C = 1$: a quarter. On a tree it is zero; the more loops,
the higher.

**The sign-flip level.** Real connections carry very different amounts, and a
connection that carries much counts for much. The fairer level keeps each
connection's amount and throws only its direction at random. Its expectation
is

$$\frac{\sum_e P_{ee} F_e^2}{\sum_e F_e^2}.$$

Here $P_{ee}$ is how much of a unit flow on connection $e$ alone is
circulation. It is one third for each side of the triangle, and zero for the
bridge $C$–$D$. In words: each connection's energy, weighted by how much of a flow
on it alone would go round, as a share of the whole energy. In the example it
is $(9 + 9 + 1)/3/23 = 19/69 \approx 0.28$. Averaging over all sixteen ways of
flipping the four directions gives the same. In a world it is estimated by
flipping the directions at random a few times: four, in
[Gliders, and who helps whom](gliders-and-who-helps-whom.md#where-the-tokens-go).

The example's 0.71 is far above both levels: its flow goes round. A share
*below* the sign-flip level means the opposite. The flow's directions line up
downhill more than chance: tokens run from some agents to others.

### Does a ring last?

A ring that is there in one stretch of games and gone in the next is an
accident of those games. A ring that **persists** is a structure. Split the
flow of two consecutive stretches, 50 games each in
[Gliders, and who helps whom](gliders-and-who-helps-whom.md). Then, over the
connections both stretches used, take the [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of their circulations,
and of their gradient parts. A correlation near zero means each stretch's
rings are new. A positive one means rings or slopes that come back on the
same connections.

### Where circulation is used

`research/oee/checks/circulation.py` does all of the above for a [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") run's
recorded games. It splits the whole flow, and two parts of it separately.
**Gifts** are stakes that did not take their target, so their tokens went to
whoever held it. **Takeovers** are stakes that took it. Gifts going round
would be generalised reciprocity: every agent gives to one and receives from
another. Bronisław Malinowski described the classic case in *Argonauts of the
Western Pacific* (1922): the Kula ring, whose armshells and necklaces pass
from island to island.

- [Where the tokens go](gliders-and-who-helps-whom.md#where-the-tokens-go)
  splits the flow of 50 games in the twenty worlds of B1 and B2. Gifts run
  downhill; takeovers go round, but never twice in the same place.
- [A greedy count and an exact one](where-the-tokens-flow.md#a-greedy-count-and-an-exact-one)
  sets this split beside the greedy count of
  [Lightning](#lightning-tokens-that-go-round). Over fifty games, 80% of
  B1's net flow circulates, by energy, where random flow on the same network
  would circulate about 60%.
- [Help](help.md) splits the net help between agents the same way, to look
  for help that goes round.

### What it can and cannot say

A flow added up over a stretch of games forgets *when* tokens moved. The
network it lives on is every connection that carried anything in that
stretch, though not all of them existed at once.

Two-way traffic on one connection, $u$ to $v$ and $v$ to $u$, cancels in the
net flow and is invisible here. Direct give-and-take between two agents has
to be looked for in the gross flows ([Token flow](#token-flow)).

And a long route through the network counts largely as circulation, even if
it never closes. One token carried along a route of $L$ connections from $a$
to $b$ has energy $L$. Think of every connection as a wire of resistance 1.
Then the token's gradient part is the electrical current from $a$ to $b$,
spread over every way between them. Its energy is the effective resistance
between $a$ and $b$, which on a network with many loops is far less than $L$.
The rest is circulation. So coherent routes, not only closed rings, raise the share.

## Where to go next

- [Where tokens flow](where-the-tokens-flow.md): all three measures on the
  worlds: the budget of a game, the connections, and the loops.
- [Gliders, and who helps whom](gliders-and-who-helps-whom.md#where-the-tokens-go):
  the flow of fifty games split exactly, gifts and takeovers apart.
- [Flow modules: the groups tokens stay in](flow-modules.md): who trades with
  whom, and for how long.
- [Where tokens go](where-tokens-go.md): every measure of the flow in the
  notebook, side by side.

## How our view changed

- **14 October 2026.** From 8 October, the note on lightning described the
  greedy peel's last step as "Repeat until no loop can be found from any
  start". A pass through the whole notebook corrected it from the code
  (`gol_lightning.py`). The peel gives an agent up as soon as one walk from
  it ends without closing a loop. So loops can be left in the flow. Only the
  description changed: the code, and every number it gave, did not.
- **24 October 2026.** When the three notes became one page, their statements
  were checked against the code. The note on token flow said that a stake on
  one's own node is left out of every statistic in its table. The statistic
  `revoltShare` counts it: it is the share of all staked tokens, home stakes
  included, that are marked revolutionary (`gol_series.py`).

<!-- turns -->
---

← [Flow modules: the groups tokens stay in](flow-modules.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [Token curvature](token-curvature.md) →

> [!info]- Pages that link here (9)
> [Glossary](glossary.md) · [Where tokens flow](where-the-tokens-flow.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Who helps whom](who-helps-whom.md) · [Gliders, and who helps whom](gliders-and-who-helps-whom.md) · [Help](help.md) · [What a run records](frames-and-stats.md) · [What the viewer can colour by](viewer-colours.md)
<!-- /turns -->
