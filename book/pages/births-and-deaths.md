# Births and the four ways to die

An [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") comes into a world in one way and leaves it in one of four. This
note defines each, says how it is counted, and how counts are turned into
**rates**.

## Birth

In the reproduction phase, every agent with at least one [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") decides how
many tokens to give a child; if that number is at least 1, a child is born
([One iteration, step by step](one-iteration.md)). The phase-1 row of the
statistics counts them as `births`.

## The four ways to die

Every death happens in a **cleanup**, and the cleanup runs after both
phases. It removes an agent for one of two reasons
([One iteration, step by step](one-iteration.md)):

- **starved** — it holds no tokens;
- **cut off** (the statistics call it `orphaned`) — it holds tokens, but it
  is not part of the largest connected piece of the network.

With two phases, that makes four ways:

| | after reproduction (phase 1) | after the game (phase 2) |
|---|---|---|
| **starved** | a parent that gave its child every token it had | an agent on whose node nobody staked anything, not even itself |
| **cut off** | mostly a newborn joined to nobody; also any piece the handovers cut loose | anyone whose part of the network lost its last [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") to the largest piece when unused connections were cut |

They are counted in each row as `starved` and `orphaned`, so the four counts
are `starved` and `orphaned` in the phase-1 rows and in the phase-2 rows.

The starved hold no tokens. The cut off do, and their tokens are pooled and
dealt out among the survivors ([Seeds and the random stream](random-numbers.md)).

## Rates per hundred

A count depends on the size of the world: a world of 2,000 agents has more
births than one of 1,000. To compare, a count is divided by the number of
agents there were **when the phase began** — the row's `nodes_before` — and
multiplied by 100:

$$
\text{births per hundred agents} = 100 \cdot \frac{\texttt{births}}{\texttt{nodes\_before}}
\quad\text{(phase-1 rows)},
$$

and the same for each way of dying, in the rows of its phase. "3.1 births per
hundred agents" means: of every hundred agents alive at the start of a
reproduction phase, 3.1 have a child.

## Where the notebook uses it

[One iteration, step by step](one-iteration.md) shows what an average
iteration of a settled world does;
[The first hundred iterations](a-worlds-life.md#the-first-hundred-iterations) the youth;
[Births, deaths and ages](births-deaths-and-ages.md) the rates over a
world's life.

<!-- turns -->
---

← [How births build the network](how-agents-have-children.md) · [Agents' lives: births, deaths, ages](../README.md#agents-lives-births-deaths-ages) · [The estate of the dead](estate.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works).

> [!info]- Pages that link here (8)
> [One iteration, step by step](one-iteration.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [Births, deaths and ages](births-deaths-and-ages.md) · [The estate of the dead](estate.md) · [Reproduction statistics](reproduction-statistics.md) · [How unequal is a world, and who stays rich](where-do-the-tokens-go.md) · [Tokens per agent: mean, median, extremes](tokens-per-agent.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
