# Help

What would cooperation be in a game whose only act is to stake [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") on
[nodes](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")? This note defines **help** from the rules of the game alone, without
asking what an agent meant, and the three signs by which help becomes
**cooperation**. [Gliders, and who helps whom](gliders-and-who-helps-whom.md) measures both.

## A stake that saves a node

Every stake an agent $u$ puts on a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") $v$'s node changes the chance that
$v$ keeps it. Compare the game as it was with the same game without $u$'s
stake, everyone else's stakes unchanged:

$$\text{effect}(u \to v) = P(v \text{ keeps its node} \mid \text{all stakes}) - P(v \text{ keeps its node} \mid \text{all stakes but } u\text{'s}).$$

A positive effect is **help**: without it, $v$ would have been more likely to
lose its node. A negative effect is an **attack**. Most stakes have no effect
at all, because the node would have gone the same way without them.

Both probabilities come from the world's own rule for deciding a node, counted
exactly, nothing sampled (`research/oee/checks/decisive.py`). Under the
[engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s rule ([One iteration, step by step](one-iteration.md)) the largest stake
wins, the owner's own stake counting one and a half times in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), ties drawn at
random, unless a crowd of revolutionary stakes outweighs the largest staker and
everyone above the crowd, in which case the winner is drawn from the step of the
crowd that tipped it. Under the [pilots](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.")' alliances
([Alliances](another-game.md#alliances)) the side with the largest total
wins. A tie between $k$ stakers gives each a probability $1/k$. In both cases
the probability is what the rule would give if the game were played many
times, so the effect is a fraction of a node: a stake that turns a lost node
into a tie has the effect $1/2$.

Only two rules let one agent's stake raise another's chance to keep its node.
[Revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") do, because a stake on $v$'s node can join $v$'s own in the crowd
that overthrows an attacker. Pooled alliances do, because an ally's stake
counts for $v$'s side. Without either, a stake on someone else's node is a bid
against its holder and can only lower that chance: help is then impossible
by rule.

## A worked example

$U$ stakes 6 tokens on $V$'s node and marks none of them revolutionary. $V$
stakes 4 on its own node, all revolutionary. $V$'s neighbour $W$ stakes 3 on
$V$'s node, all revolutionary.

**Under [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s rule** (no extra weight for the owner): the largest stake is
$U$'s, so $U$ is the one the crowd must beat. The crowd is $W$ with 3 and $V$
with 4, together 7, more than 6. Counted up from the smallest, $W$'s 3 is not
yet more than $V$'s 4 plus $U$'s 6, but $W$'s and $V$'s 7 is more than $U$'s 6
alone, so $V$'s step tips it and $V$ keeps its node: $P = 1$. Without $W$'s
stake the crowd is $V$ alone, and 4 is not more than 6: $U$ takes the node,
$P = 0$. The effect of $W$'s stake is $1 - 0 = +1$: it saved $V$'s node.

**Under B2's rule** $V$'s 4 counts as 6 and ties with $U$'s 6. With $W$, $V$
keeps the node whichever of the two is drawn as the largest: $P = 1$. Without
$W$, it keeps it only when it is drawn: $P = 1/2$. The effect of $W$'s stake is
$+1/2$.

The effect of $U$'s stake, given $W$'s, is zero under both rules: $V$ keeps its
node with or without it. And $W$'s 3 tokens go to $V$'s node, so $W$ has paid
3 tokens for $V$'s keeping it, and $V$ gained them.

## From help to cooperation

Help alone is not yet cooperation. It can be a by-product, help that happens
because an agent staked a little on every neighbour, with no more reason to
help this one than any other. Evolution can favour helping only by one of a
few routes, each of which leaves its own mark in who helps whom (Martin Nowak,
*Five rules for the evolution of cooperation*, Science, 2006). Three can be
read from the record of the games, each against the level chance would give:

- **Kin.** Help goes to clones, agents carrying the helper's own [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."), more
  often than clones are among the helper's neighbours.
- **Direct reciprocity.** Help is returned: of the pairs in which $u$ helped
  $v$ within a stretch of 50 games, the share in which $v$ also helped $u$. Its
  chance level gives every pair the same chance per game they spent as
  neighbours, $1 - (1 - q)^n$ for $n$ such games, $q$ being help per
  neighbour-game, so that pairs that were simply neighbours for longer do not
  pass for pairs that return help.
- **Indirect reciprocity.** Help goes round: $u$ helps $v$, $v$ helps $w$, $w$
  helps $u$. The net help between two agents in a stretch is a flow on the
  network, and the share of it that circulates is compared with the share
  that would if each pair's net help had a random direction (the note
  [*Flows that go round*](token-flow.md#flows-that-go-round)).

Help that shows none of the three is help without cooperation.

## What it can and cannot say

The effect is that of one game, with everyone else's stakes as they were: had
$W$ not staked, $V$ might have staked differently. It is measured one stake at a
time, so when two helpers each would have been enough, neither is counted as
saving the node; help that is redundant is undercounted. The effect is about
keeping a node. The tokens a helper hands over count too, but later, and they
are a transfer that the helper loses. And a stake with a positive effect may
also have been a bid that could have won the node for its owner; help is what
it did, not what it was for.

<!-- turns -->
---

← [Gliders, and who helps whom](gliders-and-who-helps-whom.md) · [Groups and cooperation](../README.md#groups-and-cooperation)

> [!info]- Pages that link here (3)
> [Measuring the flow: token flow, lightning and circulation](token-flow.md) · [Who helps whom](who-helps-whom.md) · [Gliders, and who helps whom](gliders-and-who-helps-whom.md)
<!-- /turns -->
