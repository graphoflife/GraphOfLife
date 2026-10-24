# Where tokens go

> [!question] Questions of this page
> - Every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), every [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") is staked somewhere. Where does it go, and does
>   anything come back?
> - What do the notebook's measures of flow say together: the budget of a
>   game, the [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."), the [loops](bridges.md#loops "A path through the network that comes back to where it started without using a connection twice."), the [circulation](token-flow.md#flows-that-go-round "The part of a flow that goes round in loops, as against the part that runs from sources to sinks."), the groups, the estate
>   of the dead?

## In short

In every game each [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") stakes all its tokens, on its own node and on its
[neighbours](the-world.md#the-network "An agent joined to another by a connection.")'. Seven pages measured where the tokens go, each with its own tool,
and this page puts them side by side. Most of what they found is the
arithmetic of one rule, every agent splitting its tokens over its node and its
neighbours', played out on a network. That rule makes wealth follow
connections, makes tokens run downhill, and leaves almost nothing going round
in one game. The [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") add a small margin on top. Two rules decide the rest:
whether a connection must carry tokens to survive, and who inherits the dead.

## The picture

Picture every agent cutting its tokens into equal heaps: one for its own node,
and one for each neighbour's. That is the **even split**
([The even split](even-split.md)). If every agent did only that, the tokens of
a world would wander like a drop of ink in water: each token, game after game,
stays or steps to a neighbour, each with the same chance. Such a wandering is
a **random walk**, and it has a resting state that can be worked out exactly.

The brains of a [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") world come close to that picture. So the measures of flow
mostly measure the walk, and the interesting part is where a world departs
from it.

## The budget of a game

Every token is staked in every game, in one of three ways: on the agent's own
node; on a neighbour that stakes back, so the two stakes cancel (*matched*);
or one way. In a settled world of B1 at 100,000 tokens, **30% of the tokens
stay home, 51% are matched and 20% go one way**; at 10,000 tokens it is 30%,
49% and 21% ([The budget of a game](where-the-tokens-flow.md#the-budget-of-a-game)).

Nine stakes in ten are returned. That sounds like reciprocity, but it is what
breadth alone gives: agents stake on 81% to 96% of the agents they could stake
on, so a stake usually meets one coming back
([Stakes that are returned](where-the-tokens-flow.md#stakes-that-are-returned)).

The brains come close to the even split. Agents with one connection keep 58%
of their tokens at home, where an even split gives 50%. Agents with ten to
forty-nine connections keep 10%, where it gives 7%. Nearly all of the
difference is the rounding of tokens into whole ones. And 42% of the agents
score every target zero or less, so the rules split their tokens evenly for
them ([How widely agents stake](where-the-tokens-flow.md#how-widely-agents-stake)).

On a single connection, a game is small change. 52% of the connections carry
exactly two tokens, and on 70% the two directions cancel exactly: a stake for
a stake. In B1 a connection that carries nothing is cut, every one of the
1,650,222 such connections counted
([What crosses a connection](where-the-tokens-flow.md#how-much-crosses-a-connection)).

## Where tokens come to rest

A random walk on a network comes to rest when every agent holds tokens in
proportion to its connections plus one: **an agent's share settles near
(connections + 1) / Σ(connections + 1)**. B1's games follow that walk with a
slope of 0.94, in the worlds without a giant [star](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub.")
([The game as a random walk](where-the-tokens-flow.md#the-game-as-a-random-walk)).

So in B1, wealth is a matter of place. Tokens and connections go together (a
rank [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of +0.56). A rich agent is still rich ten games later
(+0.78). The [richest tenth](gini-coefficient.md#the-richest-tenths-share "The share of all tokens held by the richest tenth of the agents.") hold about two fifths of the tokens, 39% to 44%
([Rich agents are connected agents](where-do-the-tokens-go.md#rich-agents-are-connected-agents),
[What goes with what](what-goes-with-what.md#the-agents)).

But the place is rich, not whoever sits on it. The more connections a node
has, the more often its agent loses it. An agent with one connection keeps its
node in 74% of games, one with fifty or more in 2%. A hub of a hundred
connections changes hands in 98 games of 100
([Who keeps their node](where-the-tokens-flow.md#who-keeps-their-node)).

In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") wealth has come loose from place. Tokens and connections go *against*
each other (−0.18), and wealth sticks half as well (+0.41 over ten games).
B2's agents keep their nodes, 88% to 98% of the time, and tokens move through
stakes that fail instead
([What goes with what](what-goes-with-what.md#token-curvature-and-token-change)).

## Downhill

Tokens run from peaks to valleys. An agent that holds more than its neighbours
loses, on average, and one that holds less gains. They do so at about a fifth
of the pace a plain even split would give: a slope of 0.20 in B2 and 0.21 in
B1, where the even split would give 1
([What goes with what](what-goes-with-what.md#token-curvature-and-token-change)).

The two [baselines](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.") run downhill in different ways
([Chains of takeovers](gliders-and-who-helps-whom.md#chains-of-takeovers)):
- **B1 through gliders.** A rich agent takes a neighbour's node and moves on,
  carrying its packet of tokens. Chains of three or more takeovers are eleven
  times as common as chance would make them. The mover keeps its old node
  after only 9% of moves. And the packet grows on the way, from 18 tokens to
  82; one grew from 380 to 1,358, a seventh of its world.
- **B2 through tribute.** Takeovers in chains are 23 times chance, but they
  are *puffers*: the mover keeps its node after 93% of moves, and the packet
  shrinks from three tokens to two. Gifts drain weakly towards the richer.

## Round and round?

Within one game almost nothing goes round. Cancel every stake against the one
coming back, and look for loops in what is left. A greedy search finds 0.3%
of the [net flow](token-flow.md "The tokens a game moves along each connection, from the agent who stakes to the node staked on.") in loops, and the conservation of tokens allows at most 16%:
the net flow of a game runs downhill
([Do tokens go round?](where-the-tokens-flow.md#do-tokens-go-round),
[The flow, in round figures](where-the-tokens-flow.md#the-flow-in-round-figures)).
Before cancelling, the longest loops are as long as the world is wide: the
longest grows as the tokens to the power 0.31. After cancelling they are five
or six agents long at 100,000 tokens
([The longest loop grows with the world](where-the-tokens-flow.md#the-longest-loop-grows-with-the-world)).

Over fifty games the picture changes. The flow of fifty games can be split
exactly into a part that runs downhill and a part that circulates: 80% of
B1's net flow circulates, and 59% of B2's. **But random flow on the same
networks circulates 59% to 60% and 50% to 53%.** The excess the agents add is
about 0.2 in B1, and under 0.1 in B2
([A greedy count and an exact one](where-the-tokens-flow.md#a-greedy-count-and-an-exact-one)).

One rule decides how much goes round in the first place: **whether a
connection must carry tokens to survive.** Where it must (B1, and B2 with
every unused link lapsing), the greedy search finds about a sixth of the
staked flow on loops: 13% in B1 at 100,000 tokens, 17% at 10,000, and 11% to
28% in the B2 [pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") whose links lapse. In B2, whose connections survive
unused if their ends share two neighbours, it finds 4.5%, and the share falls
over a world's life
([Every world](where-the-tokens-flow.md#every-world)).

## Gifts, takeovers and groups

- **Gifts circulate less than chance**, in B1 and B2 alike, and run downhill:
  towards the well connected in B1, towards the richer in B2. Only alliances
  make gifts go round (+0.04 to +0.14 over chance), and not for long
  ([Where the tokens go](gliders-and-who-helps-whom.md#where-the-tokens-go)).
- **Takeovers circulate more than chance** (+0.07 in B2, +0.24 in B1), but
  never twice in the same place: their persistence is 0.00.
- **The groups tokens stay in** are, in B1, the network's own regions. A null
  model that keeps the network and redeals the stakes compresses the flow as
  well (0.388 against 0.384). In B2 they are short-lived pairs: 9% last five
  games, where the null gives 4%. These pairs are the one structure B2's
  agents add themselves
  ([Flow modules](flow-modules.md#what-the-network-makes-and-what-the-agents-make)).
  A fifth of B1's long-lived groups end with none of their first members, but
  so do the null's.

## The estate of the dead

The dead leave their tokens to the living ([The estate of the dead](estate.md)).
In B1 that is 1.3% of the tokens in every game, handed out one token at a
time, and it levels the world: over a world's life, the estate goes with
falling inequality
([The tokens of the dead](where-do-the-tokens-go.md#the-tokens-of-the-dead),
[What goes with what](what-goes-with-what.md#a-whole-world-over-time)). In B2
the pool is small, about seventy tokens an [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") among 3,600 agents. There
where it goes matters more than how much it is: **an estate that pays for
births fills a world**, to half as many agents again, and poorer ones
([Who inherits?](who-inherits.md),
[What holds a world's size](what-holds-a-worlds-size.md)).

## What we learned

- **Flow here is mostly arithmetic.** An even split on a network, rounded to
  whole tokens, explains where tokens rest, why wealth follows connections in
  B1, how fast tokens run downhill, and most of what circulates. The brains'
  part is a margin: about 0.2 of circulation in B1, pairs in B2, brief rings
  in alliances.
- **Nothing that flows comes back to the same place.** Gliders move on and
  grow, takeovers never repeat, gifts drain. A lasting cycle of exchange, the
  kind a structure could be built on, has not appeared in any world.
- **Two rules matter most for flow.** Whether an unused connection lapses
  decides how much goes round. Who inherits the dead decides how many agents
  share the tokens.
- **Still open:** how one game's 0.3% to 16% of circulation grows to fifty
  games' 80% and 59%; what makes B1's excess over chance; whether alliances'
  groups outlast the network's regions; how the flow of brains that see
  differs; and how an estate shared by wealth (`by_tokens`) moves tokens.

## Where to go next

- [Where tokens flow](where-the-tokens-flow.md): the budget of a game, the
  random walk, the connections and the loops, in full.
- [How unequal is a world, and who stays rich](where-do-the-tokens-go.md):
  who holds the tokens the flow leaves behind.
- [Gliders, and who helps whom](gliders-and-who-helps-whom.md): chains of
  takeovers, gifts, and the flow of fifty games.
- [Flow modules: the groups tokens stay in](flow-modules.md): who trades with
  whom, and for how long.

## How our view changed

- **23 October 2026.** This page said that in a settled B1 world 30% of the
  tokens stay home, 51% are matched and 20% go one way "at 10,000 tokens and
  at 100,000 alike". At 10,000 tokens the shares are 49% and 21%
  ([The budget of a game](where-the-tokens-flow.md#the-budget-of-a-game)). It
  also said that where a connection must carry tokens, 17% to 28% of the
  staked flow is on loops. B1 at 100,000 tokens has 13% there, and one of the
  four B2 pilots whose unused links lapse has 11%
  ([Every world](where-the-tokens-flow.md#every-world)). And it said the
  longest loops after cancelling are five or six agents long, which holds at
  100,000 tokens; in the largest worlds they are 6.5 to 6.75. It also said
  that batch H was testing the estate left to a dead agent's own children:
  that was done, and is in [Who inherits?](who-inherits.md).

<!-- turns -->
---

[The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [How the game is played](how-the-game-is-played.md) →

> [!info]- Pages that link here (3)
> [The estate of the dead](estate.md) · [Where tokens flow](where-the-tokens-flow.md) · [Measuring the flow: token flow, lightning and circulation](token-flow.md)
<!-- /turns -->
