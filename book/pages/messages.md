# Messages

[Agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") can say things to each other. A **message** is a list of *m* = 30
numbers between −1 and 1 ([`message_amount`](settings.md#message_amount)).
Nothing in the rules says what the numbers mean: a message means whatever
the [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") that write and read it make of it, and that can evolve.

## Who writes what

Whenever an agent *u* looks at its [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides.") — itself and its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") —
its brain gives, for every candidate *v*, 30 message outputs (rows 15–44,
[What a brain says](brain-outputs.md)). Passed through tanh, they are the
message from *u* **to** *v*. So in every look an agent writes one message to
itself and one to each neighbour.

## Who reads what

When *u* looks at a candidate *v*, it reads four messages
([What a brain sees](brain-inputs.md)):

1. what *u* wrote to itself — a memory of sorts;
2. what *u* wrote to *v*;
3. what *v* wrote to *u*;
4. what *v* wrote to itself.

Only the last message written from each writer to each reader is kept. A
message that does not exist yet — between two agents who have just become
neighbours — reads as 30 zeros.

## When messages are delivered

Messages are not delivered the moment they are written. Every look of a
phase writes into an **outbox**, and the outbox is delivered when the look
is over. So every agent in a look reads the same generation of messages,
whatever the order in which agents are visited.

With [`message_prepass`](settings.md#message_prepass) on, as in the
[baseline](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2."), each phase has **two looks**:

1. **The prepass.** Every agent looks at its candidates and writes messages,
   and does nothing else. The messages are delivered.
2. **The look that acts.** Every agent looks again, reads the messages of
   the prepass — written in this phase, about the world as it now is — and
   decides. It writes messages again, which are delivered at the end of the
   phase and read in the next phase's prepass.

## Forgetting

At the end of every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), messages to or from anyone who is no longer a
neighbour (or no longer alive) are deleted. A [connection](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes.") that is cut and
later made again starts with zeros.

## What it costs a brain

Of the 154 inputs of a baseline brain, 120 are messages, and of its 45
outputs, 30 are. Most of a brain's first layer, 120 × 50 = 6,000 of its
15,550 weights, reads messages. Whether the messages carry anything useful is
an open question for this notebook.

<!-- turns -->
---

← [What a brain says](brain-outputs.md) · [How the simulation works](../README.md#how-the-simulation-works) · [How a coalition takes a node](revolution.md) →

> [!info]- Pages that link here (10)
> [Glossary](glossary.md) · [One iteration, step by step](one-iteration.md) · [The brain](the-brain.md) · [Every setting](settings.md) · [What a brain sees](brain-inputs.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [What agents say to each other](what-agents-say-to-each-other.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [Do agents cooperate?](do-agents-cooperate.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
