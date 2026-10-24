# Births and the four ways to die

## In short

After a world's founders, every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") enters it the same way: a parent gives
it [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") in a reproduction phase, and it is born. In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") it leaves in one of
four ways: it **starves** or it is **cut off**, after reproduction or after
the game. Births and deaths are counted per hundred agents alive when their
phase began. In [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.") none of the four can happen, and every agent dies of old
age, at 100 iterations.

## Birth

A birth is a split. A parent gives some of its own tokens to a child. The
child takes a new place nearby, with a copy of the parent's [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.").

Here is the exact rule
([One iteration, step by step](one-iteration.md#phase-1-reproduction)). In
the reproduction phase, every agent holding at least one token decides how
many of them to give a child. If that number is at least 1, a child is born.
An agent has at most one child in a phase. Every agent but the founders a
world starts with ([The starting ring](starting-ring.md)) was born this way.

The run writes a row of statistics after every phase
([What a run records](frames-and-stats.md)). The phase-1 row counts the
children as `births`: every child born, also those the cleanup removes at
once. B2 asks more of a birth ([In B2](#in-b2)).

## The four ways to die

After each phase the world is tidied up. Whoever holds nothing leaves it, and
so does every part of the network that has come loose from the main piece.

Exactly: every death happens in a **cleanup**, and a cleanup runs after each
of the two phases ([One iteration, step by step](one-iteration.md#cleanup)).
In B1 it removes an agent for one of two reasons:

- **starved**: it holds no tokens;
- **cut off**: it holds tokens, but is not in the largest connected piece of
  what is left once the starved are gone. The statistics call it `orphaned`.

Two reasons and two phases make four ways to die:

| | after reproduction (phase 1) | after the game (phase 2) |
|---|---|---|
| **starved** | a parent that gave its child every token it had | an agent on whose node nobody staked anything, not even itself |
| **cut off** | mostly a newborn joined to nobody; also any piece cut loose by a handover, or by the death of a parent that gave everything | anyone whose part of the network came loose from the largest piece: when unused [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") were cut, or when a starved agent that joined it to the rest was removed |

A handover is a parent moving one of its own connections to its newborn
child. In B1, only parents can starve in reproduction. Every agent alive after
a game holds at least one token. And in B1's reproduction phase, an agent
loses tokens only by giving them to its child.

Each row of the statistics counts its phase's deaths as `starved` and
`orphaned`. So the four counts are `starved` and `orphaned` in the phase-1
rows and in the phase-2 rows.

The starved hold no tokens. The cut off do. Their tokens are pooled and dealt
out among the survivors, each token to a survivor drawn at random.
[The estate of the dead](estate.md) says more, and
[Seeds and the random stream](random-numbers.md#how-a-draw-becomes-a-decision)
how such a draw is made.

## Rates per hundred

A count alone says little: a world of 2,000 agents has more births than one
of 1,000. To compare, a count is divided by the number of agents there were
**when the phase began**, the row's `nodes_before`, and multiplied by 100:

$$
\text{births per hundred agents} = 100 \cdot \frac{\texttt{births}}{\texttt{nodes\_before}}
\quad\text{(phase-1 rows)},
$$

and the same for each way of dying, in the rows of its phase. For example,
"3.1 births per hundred agents" means: of every hundred agents alive at the
start of a reproduction phase, 3.1 have a child.

Here are the five rates of a settled B1 world. Each is a world's mean over
iterations 500 to 1,499. The table gives the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of the 30 worlds of
100,000 tokens (Experiment 10;
[Births, deaths and ages](births-deaths-and-ages.md)):

| per hundred agents | after reproduction | after the game |
|---|---|---|
| born | 3.5 | — |
| starved | 0.15 | 0.7 |
| cut off | 0.4 | 2.1 |

About as many agents die as are born: the typical settled world hardly
changes in size, while its members come and go
([A world's life](a-worlds-life.md#a-worlds-life)). Three deaths in four are
cuts. The worlds of 10,000 tokens (Experiment 2) had a little less of
everything: 3.1 births per hundred agents.

## In B2

B2's rules were made so that none of the four can happen
([Building B2: a world that cannot break](a-world-that-cannot-break.md#who-dies-and-why)):

- **Nobody starves.** In the game a share of every agent's tokens stays on
  its own node, at least one token ([`keep_home`](settings.md#keep_home)).
  And a parent always keeps at least one token.
- **Nobody is cut off.** A newborn is always joined to its parent, and to two
  of the parent's [neighbours](the-world.md#the-network "An agent joined to another by a connection.") ([`birth_links`](settings.md#birth_links)). The
  neighbours of an agent that dies are joined up again
  ([`mend_on_death`](settings.md#mend_on_death)). An unused connection is cut
  only if its two ends stay joined through neighbours they share
  ([`prune_needs`](settings.md#prune_needs)). And no parent hands over a
  connection.

Instead, an agent dies of **old age** ([`max_age`](settings.md#max_age)). It
dies in the cleanup after the reproduction phase of the iteration in which it
turns 100. So every agent of B2 lives exactly 100 iterations. Its tokens are
shared out like any other estate. The statistics file has no field for these
deaths: a frame's `cleanup` record counts them as `aged`.

B2 also asks more of a birth. A child needs at least 10 tokens
([`min_child`](settings.md#min_child)), and the parent keeps its last token.
A parent with 8 connections or more has no child
([`valence`](settings.md#valence)).

Since every agent lives 100 iterations, a B2 world keeps its size only if one
agent in a hundred has a child in each iteration. In the ten B2 worlds of
5,000 tokens of Experiment 14, from iteration 200 to 1,000, births ran at
1.00 to 1.05 per hundred agents. Not one agent starved or was cut off.

## Where it is used

- [One iteration, step by step](one-iteration.md#what-an-iteration-does-on-average):
  what an average iteration of a settled world does.
- [The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations):
  births and deaths in a world's youth.
- [Births, deaths and ages](births-deaths-and-ages.md): the rates over a
  world's life.
- [Building B2: a world that cannot break](a-world-that-cannot-break.md#who-dies-and-why):
  which deaths a world needs.

## Where to go next

- [The estate of the dead](estate.md): what becomes of the tokens the dead
  leave.
- [Age and lifetime](age-and-lifetime.md): how long a life lasts, and how the
  notebook measures it.
- [Reproduction statistics](reproduction-statistics.md): what a run records
  of every birth.

## How our view changed

- **8 October 2026.** This page said that an agent leaves a world in one of
  four ways. It also said that a parent that gives a child at least one
  token has one.
- **12 October 2026.** B2 was made
  ([Building B2: a world that cannot break](a-world-that-cannot-break.md)).
  Under its rules nobody starves or is cut off, and every agent dies of age,
  at 100. A birth needs at least ten tokens, from a parent with fewer than
  eight connections. Since 23 October 2026 the page says which rules each
  statement holds for.

<!-- turns -->
---

← [How births build the network](how-agents-have-children.md) · [Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages) · [The estate of the dead](estate.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works).

> [!info]- Pages that link here (9)
> [One iteration, step by step](one-iteration.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [The estate of the dead](estate.md) · [Age and lifetime](age-and-lifetime.md) · [Reproduction statistics](reproduction-statistics.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Measuring inequality: the Gini coefficient, the richest tenth, entropy and mobility](gini-coefficient.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
