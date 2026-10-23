# Messages

Agents can say things to each other. A **message** is a list of *m* = 30
numbers between −1 and 1 ([`message_amount`](settings.md#message_amount)).
Nothing in the rules says what the numbers mean: a message means whatever
the brains that write and read it make of it, and that can evolve.

## Who writes what

Whenever an agent *u* looks at its candidates — itself and its neighbours —
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
baseline, each phase has **two looks**:

1. **The prepass.** Every agent looks at its candidates and writes messages,
   and does nothing else. The messages are delivered.
2. **The look that acts.** Every agent looks again, reads the messages of
   the prepass — written in this phase, about the world as it now is — and
   decides. It writes messages again, which are delivered at the end of the
   phase and read in the next phase's prepass.

## Forgetting

At the end of every game, messages to or from anyone who is no longer a
neighbour (or no longer alive) are deleted. A connection that is cut and
later made again starts with zeros.

## What it costs a brain

Of the 154 inputs of a baseline brain, 120 are messages, and of its 45
outputs, 30 are. Most of a brain's first layer, 120 × 50 = 6,000 of its
15,550 weights, reads messages. Whether the messages carry anything useful is
an open question for this book.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
