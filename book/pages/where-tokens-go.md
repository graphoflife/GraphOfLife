# Where tokens go

> [!question] Questions of this page
> - Every game, every token is staked somewhere. Where does it go, and does
>   anything come back?
> - What do the notebook's measures of flow — the budget of a game, the links,
>   the loops, the circulation, the groups, the estate of the dead — say
>   together?

Seven pages measured how tokens move, each with its own tool. This page
puts them side by side. Most of what they found is the arithmetic of one rule
— every agent splits its tokens over its node and its neighbours' — played out
on a network; what the brains add on top is small, and the notebook can say where.

## The budget of a game

Every token is staked in every game: on the agent's own node, on a neighbour
that stakes back (matched), or one way. In a settled world of B1, **30% of the
tokens stay home, 51% are matched and 20% go one way**, at 10,000 tokens and
at 100,000 alike ([Where the tokens flow](where-the-tokens-flow.md#the-budget-of-a-game)).
Nine stakes in ten are returned — but that is what breadth alone gives, since
agents stake on 81% to 96% of the agents they could stake on
([Where the tokens flow](where-the-tokens-flow.md#stakes-that-are-returned)).

The brains come close to the **even split**: each agent's tokens shared
equally over its node and its links. Agents with one link keep 58% at home
where an even split gives 50%, agents with ten to forty-nine links 10% where
it gives 7%, and nearly all the difference is the rounding of tokens into
whole ones; 42% of the agents score every target zero or less, and the rules
then split their tokens evenly for them
([Where the tokens flow](where-the-tokens-flow.md#how-widely-agents-stake)).

On a link, a game is small change: 52% of links carry exactly two tokens,
and on 70% the two directions cancel exactly — a stake for a stake. A link
that carries nothing is cut, every one of 1,650,222 of them in B1's worlds
([What crosses a connection](what-crosses-a-connection.md#how-much-crosses-a-connection)).

## Where tokens come to rest

An even split on a network is a random walk, and a random walk on a network
comes to rest in proportion to each node's links: **an agent's share of the
tokens settles near (links + 1) / Σ(links + 1)**. B1's games follow it with a
slope of 0.94 in the worlds without a giant star
([Where the tokens flow](where-the-tokens-flow.md#the-game-as-a-random-walk)). So in
B1 wealth is a matter of place: tokens and links go together (a rank
correlation of +0.56), a rich agent is still rich ten games later (+0.78), and
the richest tenth holds about two fifths of the tokens (39% to 44%)
([Where do the tokens go?](where-do-the-tokens-go.md#rich-agents-are-connected-agents),
[What goes with what](what-goes-with-what.md#the-agents)). The place is rich, not its
holder: the more links a node has, the more often its holder loses it — an
agent with one link keeps its node in 74% of games, one with fifty or more in
2%, and a hub of a hundred links changes hands in 98 games of 100
([Where the tokens flow](where-the-tokens-flow.md#who-keeps-their-node)).

In B2 wealth has come loose from place: tokens and links go *against* each
other (−0.18), and wealth sticks half as well (+0.41 over ten games), because
B2's agents keep their nodes — 88% to 98% of the time — and tokens move
through stakes that fail instead
([What goes with what](what-goes-with-what.md#token-curvature-and-token-change)).

## Downhill

Tokens run from peaks to valleys: an agent that holds more than its
neighbours loses, on average, and one that holds less gains — at about a
fifth of the pace a plain even split would give (a slope of 0.20 in B2 and
0.21 in B1, where the even split would be 1)
([What goes with what](what-goes-with-what.md#token-curvature-and-token-change)). The
two baselines run downhill differently
([Gliders, and who helps whom](gliders-and-who-helps-whom.md#chains-of-takeovers)):

- **B1 through gliders.** A rich agent takes a neighbour's node and moves on,
  carrying its packet: chains of three or more takeovers are eleven times as
  common as chance, the mover keeps its old node after only 9% of moves, and
  the packet grows on the way, from 18 tokens to 82 — one grew from 380 to
  1,358, a seventh of its world.
- **B2 through tribute.** Takeovers in chains are 23 times chance, but they are
  puffers: the mover keeps its node after 93% of moves, the packet shrinks from
  three tokens to two, and gifts drain weakly towards the richer.

## Round and round?

Within one game almost nothing goes round. After the returned stakes are
cancelled, a greedy search finds 0.3% of the net flow in loops, and the
conservation of tokens allows at most 16%: the net flow of a game runs downhill
([Where the tokens flow](where-the-tokens-flow.md#do-tokens-go-round),
[What crosses a connection](what-crosses-a-connection.md#the-flow-in-round-figures)).
The gross loops are as long as the world is wide — the longest grows as the
tokens to the power 0.31 — and after cancelling they are five or six agents
long ([Lightning in every world](lightning-in-every-world.md#the-longest-loop-grows-with-the-world)).

Over fifty games, the exact decomposition of the flow into a part that runs
downhill and a part that circulates finds 80% of B1's net flow circulating
and 59% of B2's. **But random flow on the same networks circulates 59% to 60%
and 50% to 53%**: the excess the agents add is about 0.2 in B1 and under 0.1
in B2 ([Lightning in every world](lightning-in-every-world.md#a-greedy-count-and-an-exact-one)).
How the one-game count and the fifty-game count fit together is still open.

One rule decides how much goes round in the first place: **whether a link must
carry tokens to survive.** Where it must — B1, and B2 with every unused link
lapsing — 17% to 28% of the staked flow is on loops; in B2, whose links survive
unused if their ends share two neighbours, 4.5%, and the share falls over a
world's life ([Lightning in every world](lightning-in-every-world.md#every-world)).

## Gifts, takeovers and groups

- **Gifts circulate less than chance**, in B1 and B2 alike, and run downhill
  — towards the well connected in B1, towards the richer in B2. Only
  alliances make gifts go round (+0.04 to +0.14 over chance), and not for long
  ([Gliders, and who helps whom](gliders-and-who-helps-whom.md#where-the-tokens-go)).
- **Takeovers circulate more than chance** (+0.07 in B2, +0.24 in B1), but
  never twice in the same place: their persistence is 0.00.
- **The groups tokens stay in** are, in B1, the network's own regions: a null
  that keeps the network and redeals the stakes compresses the flow as well
  (0.388 against 0.384). In B2 they are short-lived pairs — 9% last five games
  where the null gives 4% — the one structure B2's agents add themselves
  ([Flow modules: the groups tokens stay in](flow-modules.md#what-the-network-makes-and-what-the-agents-make)).
  A fifth of B1's long-lived groups end with none of their first members — but
  so do the null's.

## The estate of the dead

The dead leave their tokens to the living. In B1 that is 1.3% of the tokens
in every game, handed out one token at a time, and it levels the world: over
a world's life, the estate goes with falling inequality
([Where do the tokens go?](where-do-the-tokens-go.md#the-tokens-of-the-dead),
[What goes with what](what-goes-with-what.md#a-whole-world-over-time)). In B2 the
pool is small — about seventy tokens an iteration among 3,600 agents — and
where it goes matters more than how much it is: **an estate that pays for
births fills a world**, to half as many agents again and poorer
([Who inherits?](who-inherits.md),
[What holds a world's size](what-holds-a-worlds-size.md)).

## What this says

- **Flow here is mostly arithmetic.** An even split on a network, rounded to
  whole tokens, explains where tokens rest, why wealth follows links in B1,
  how fast tokens run downhill, and most of what circulates. The brains'
  part is a margin: about 0.2 of circulation in B1, pairs in B2, brief rings
  in alliances.
- **Nothing that flows comes back to the same place.** Gliders move on and
  grow, takeovers never repeat, gifts drain. A lasting cycle of exchange — the
  kind a structure could be built on — has not appeared in any world.
- **Two rules matter most for flow**: whether an unused link lapses (it decides
  how much goes round), and who inherits the dead (it decides how many agents
  share the tokens).
- **Open:** how one game's 0.3–16% of circulation grows to fifty games' 80%
  and 59%; what makes B1's excess over chance; whether alliances' groups
  outlast the network's regions; how the flow of brains that see differs; and
  where the estate goes under the rules not yet compared — `by_tokens`, and
  `children`, which batch H is testing.

**Related notes:** [Where do the tokens go?](where-do-the-tokens-go.md) ·
[Where the tokens flow](where-the-tokens-flow.md) ·
[What crosses a connection](what-crosses-a-connection.md) ·
[Gliders, and who helps whom](gliders-and-who-helps-whom.md) ·
[Flow modules: the groups tokens stay in](flow-modules.md) ·
[Lightning in every world](lightning-in-every-world.md) ·
[What goes with what](what-goes-with-what.md) ·
[Who inherits?](who-inherits.md) ·
[What holds a world's size](what-holds-a-worlds-size.md)

<!-- turns -->
---

[The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [How the game is played](how-the-game-is-played.md) →

> [!info]- Pages that link here (1)
> [Lightning in every world](lightning-in-every-world.md)
<!-- /turns -->
