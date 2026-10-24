# Messages

## In short

[Agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") can say things to each other: every time an agent looks at its
[neighbours](the-world.md#the-network "An agent joined to another by a connection."), it writes a **message** to each of them, and one to itself. A
message is a list of numbers between −1 and 1, 30 in [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.") and 8 in [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."). Looking
at a neighbour, an agent reads four messages: its own to itself and to that
neighbour, and the neighbour's to it and to itself. No rule says what the
numbers mean: a message means whatever the [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that write and read it make
of it, and that can evolve. In B1, a message turned out to say little more
than who wrote it.

## Notes between neighbours

Picture neighbours who leave each other notes. Every agent writes one note to
each neighbour, and one to itself. When it later considers a neighbour, it
reads four notes: its note to itself, its note to that neighbour, the
neighbour's note to it, and the neighbour's note to itself. The notes are
lists of numbers, and nothing gives them a meaning but the brains that write
and read them.

## Who writes what

Whenever an agent *u* looks at its candidates — itself and its neighbours —
its brain gives one column of outputs for every candidate *v*. In B1, rows
15–44 of that column are the 30 numbers of a message
([What a brain says](brain-outputs.md)). Passed through tanh, a function that
squeezes any number into the range from −1 to 1, they are the message from
*u* **to** *v*. So in every look an agent writes one message to itself and
one to each neighbour.

How many numbers a message holds is a setting,
*m* ([`message_amount`](settings.md#message_amount)): *m* = 30 in B1, and
*m* = 8 in B2, whose message fills output rows 11–18.

## Who reads what

When *u* looks at a candidate *v*, it reads four messages
([What a brain sees](brain-inputs.md)):

1. what *u* wrote to itself — a memory of sorts;
2. what *u* wrote to *v*;
3. what *v* wrote to *u*;
4. what *v* wrote to itself.

Only the last message written from each writer to each reader is kept. A
message that does not exist yet — between two agents who have just become
neighbours — reads as 30 zeros, or *m* zeros in general.

Some experiments change what an agent writes to itself. Under
[`self_message`](settings.md#self_message) = largest, it is the largest,
number by number, of everything the agent writes in that look.

## When messages are delivered

Messages are not delivered the moment they are written. Every look of a
phase writes into an **outbox**, and the outbox is delivered when the look is
over. So every agent in a look reads the same generation of messages,
whatever the order in which the agents are visited.

With [`message_prepass`](settings.md#message_prepass) on, as in B1, each
phase has **two looks**:

1. **The prepass.** Every agent looks at its candidates and writes messages,
   and does nothing else. The messages are delivered.
2. **The look that acts.** Every agent looks again, reads the messages of
   the prepass — written in this phase, about the world as it now is — and
   decides. It writes messages again, which are delivered at the end of the
   phase and read in the next phase's prepass.

B2 has no prepass, so each of its phases has one look. An agent there acts
on messages written a phase earlier, before the births, deaths and conquests
since. The messages it writes as it acts are read in the next phase.

## Forgetting

At the end of every game, messages to or from anyone who is no longer a
neighbour (or no longer alive) are deleted. A [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") that is cut and
later made again starts with zeros.

## What it costs a brain

In B1, 120 of a brain's 154 inputs are messages, and 30 of its 45 outputs.
Its first layer has 154 × 50 = 7,700 weights, and 120 × 50 = 6,000 of them
read messages: more than a third of all the brain's 15,550 weights. In B2,
32 of 66 inputs and 8 of 19 outputs are messages.

## What the messages say

Do the messages carry anything useful? For B1,
[What agents say to each other](what-agents-say-to-each-other.md) answered:
hardly. It read every message of the 30 B1 worlds of 100,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") at their
end. Two messages written by one agent to two readers differ by 0.0005, as
the root mean square of the differences of their 30 numbers. Messages written
by two different agents differ by 0.60. So an agent says the same thing to
everyone, and all agents of one [genotype](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from.") write alike: a message is a name.
What a message could be about — the writer's and the reader's tokens and
connections, and whether it is written to oneself — accounts for only 1.2%
of the messages' variety.

## Where it is used

- [What agents say to each other](what-agents-say-to-each-other.md): what
  the messages of B1's worlds carry, number by number.
- [What a brain can compute](what-a-brain-can-compute.md): why no message
  can carry news of a third agent, and how `self_message` = largest would let
  one.
- [Do agents cooperate?](do-agents-cooperate.md): messages are the only way
  a brain could tell kin from strangers.

## Where to go next

- [What a brain sees](brain-inputs.md): where the four messages sit among
  the inputs.
- [What a brain says](brain-outputs.md): the rows that write a message.
- [One iteration, step by step](one-iteration.md): when the looks happen in
  an iteration.

<!-- turns -->
---

← [What a brain says](brain-outputs.md) · [How the simulation works](../README.md#how-the-simulation-works) · [How a coalition takes a node](revolution.md) →

> [!info]- Pages that link here (13)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Every setting](settings.md) · [What a brain sees](brain-inputs.md) · [What a brain says](brain-outputs.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [What agents say to each other](what-agents-say-to-each-other.md) · [Eras, contests and invasions: does a world keep getting better?](eras-and-contests.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [What a brain can compute](what-a-brain-can-compute.md) · [Do agents cooperate? Help, kin and partners](do-agents-cooperate.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
