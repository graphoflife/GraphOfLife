# The even split

## In short

In every [game](one-iteration.md#phase-2-the-game) an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") stakes all its
[tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") on its own node and its [neighbours](the-world.md#the-network "An agent joined to another by a connection.")' nodes, guided by its
[brain](the-brain.md)'s score for each. When a brain that spreads its stake
scores every node at zero or below, the rules split the tokens evenly: the
**even split**. It is common in the
[baseline B1](b1-and-b2.md#b1-the-rules-as-they-were-built). A
game played this way moves tokens like a lazy random walk, which comes to rest
with every agent holding tokens in proportion to its [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") plus one.

## The rule

An agent with τ tokens and *d* neighbours has *d* + 1
[candidates](one-iteration.md#phase-1-reproduction): its own node first, then
its neighbours' nodes. Its brain gives each a stake score, and either goes
**all in**, every token on the candidate it scores highest, or **spreads**,
splitting the tokens in proportion to the scores, negative ones counted as
zero.

A spread needs at least one score above zero. If every score is zero or below,
the τ tokens are split evenly
([Splitting tokens into whole numbers](largest-remainder.md)):

1. every candidate gets ⌊τ / (*d* + 1)⌋ tokens;
2. the tokens left over, fewer than *d* + 1, go one each down the list of
   candidates. Their remainders are all equal, so the order of the list
   decides, and the agent's own node is first
   ([Where the tokens flow](where-the-tokens-flow.md#how-widely-agents-stake)).

With 3 tokens and one neighbour, an agent stakes 2 at home and 1 on the
neighbour, so whole tokens tilt the split towards home. The share staked at
home, by [degree](degree.md), in the thirty baseline worlds of 100,000 tokens of
Experiment 10 ([Thirty worlds](a-worlds-life.md#thirty-worlds)):

| connections | 1 | 2 | 3–4 | 5–9 | 10–49 | 50+ |
|---|---|---|---|---|---|---|
| even split in fractions, 1/(*d* + 1) | 50% | 33% | 23% | 14% | 7% | 1.2% |
| even split in whole tokens | 60% | 44% | 34% | 23% | 13% | 3.0% |
| what the agents really staked | 58% | 40% | 28% | 19% | 10% | 2.8% |

## How often it happens

Evolved brains in B1 spread in 98% of cases. In the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world of the thirty,
**42%** of the agents score every candidate at zero or below and so get the even
split (the worlds range from 13% to 98%; at 10,000 tokens the median world has
58%). Fresh [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' brains, given the same inputs, do so 49% of the time:
selection did not make it. The other agents split nearly evenly too, because
their nearly blind brains give all candidates almost the same score
([What the brains are like](what-the-brains-are-like.md#what-evolution-changed)).

In [B2](b1-and-b2.md#b2-a-world-that-cannot-break), where nearly every agent goes all in
(spreading in 0.1–4% of cases), the even split hardly arises
([How different are the agents?](how-different-are-the-agents.md)).

## A lazy random walk

Think of one token of an agent that splits evenly. It goes to each of the
*d* + 1 candidate nodes with the same chance, 1/(*d* + 1), and stays with the
node it lands on, whoever wins it. Game after game, that is a **random walk**:
at each step a token stays where it is or moves to a neighbour. The walk is
*lazy* because staying is one of the choices.

How close is the real game?
[Where the tokens flow](where-the-tokens-flow.md#the-game-as-a-random-walk)
compares, agent by agent, the balance an exact even split would give with the
real one. In the 23 worlds without a
[star](stars-and-hubs.md), the real balances follow the even split's with a
slope of 0.94, and it accounts for 45% of their variance
([Fitting a straight line, and R²](correlation.md#fitting-a-straight-line)); at 10,000 tokens, 0.94
and 46%. With the seven stars, 0.65 and 34%: the rounding of whole tokens,
small at one leaf, piles up at a star's centre.

## The resting state

A random walk on a fixed network is at rest when, across every connection, as
many tokens go one way as the other. With π(*u*) the share of tokens at *u*:

$$
\frac{\pi(u)}{d_u + 1} = \frac{\pi(v)}{d_v + 1} \quad \text{for every connection } \{u, v\} .
$$

That holds when π(*u*) is proportional to *d*ᵤ + 1. With *T* tokens in all,

$$
\tau_u = T \cdot \frac{d_u + 1}{\sum_v (d_v + 1)} ,
$$

so every agent holds the same number of tokens for each of its candidates. The
worlds come close. Among the 394,056 agents alive after the last game of the
thirty, rest would mean 1.70 tokens per candidate; the mean of each class runs
from 2.28 for agents with one connection to 1.43–1.51 for those with ten or
more ([How properties scale together](how-properties-scale-together.md#tokens-and-connections)).
The network keeps changing under the tokens, so no world is ever quite at rest.

## Where it is used

- [Where the tokens flow](where-the-tokens-flow.md#who-keeps-their-node): an
  agent keeps its node about as often as one of *d* + 1 equal stakers would:
  in 74% of games with one connection, 2.0% with fifty or more.
- [Do the rich stay rich?](where-do-the-tokens-go.md#a-two-game-rhythm): on a hub
  and its leaves the even split overshoots, and wealth swings with a period of
  two games.
- [Physical inspiration](physical-inspiration.md): an even split is one step of
  the heat equation; [What goes with what](what-goes-with-what.md#token-curvature-and-token-change)
  finds tokens running downhill at about a fifth of its pace.

## Where to go next

- [Splitting tokens into whole numbers](largest-remainder.md): the rounding.
- [Where the tokens flow](where-the-tokens-flow.md): the random walk, measured.
- [What the brains are like](what-the-brains-are-like.md): why brains score
  alike.
- [Stars and hubs](stars-and-hubs.md): where the walk fits worst.

<!-- turns -->
---

← [How the game is played: what agents stake, and who wins the nodes](how-the-game-is-played.md) · [The game and the flow of tokens](../README.md#the-game-and-the-flow-of-tokens) · [Where tokens flow](where-the-tokens-flow.md) →

**Also about:** [Brains, and how alike agents are](../README.md#brains-and-how-alike-agents-are).

> [!info]- Pages that link here (7)
> [Glossary](glossary.md) · [Splitting tokens into whole numbers](largest-remainder.md) · [Where tokens go](where-tokens-go.md) · [How the game is played: what agents stake, and who wins the nodes](how-the-game-is-played.md) · [Flow modules: the groups tokens stay in](flow-modules.md) · [An agent's connections: degree, clustering, assortativity, attachment and scaling](degree.md) · [Stars and hubs](stars-and-hubs.md)
<!-- /turns -->
