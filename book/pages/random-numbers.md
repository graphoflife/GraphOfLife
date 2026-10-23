# Seeds and the random stream

Graph of Life is full of chance: random starting [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."), random tie-breaks,
probabilistic decisions, random inputs, random sharing out of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). Yet a
run can be made again exactly. This note says how.

## Pseudo-random numbers

A computer does not throw dice. It computes a long sequence of numbers that
*look* random — pass the usual statistical tests — from a starting value
called the **seed**. The same seed gives the same sequence, every time, on
every computer that uses the same algorithm.

## The two streams of a run

Every run has one seed, a whole number (the experiments use 1, 2, 3, …).
From it two streams are started:

1. **The ring's stream.** networkx builds the starting ring
   ([The starting ring](starting-ring.md)) from a generator of its own,
   seeded with the run's seed.
2. **The world's stream.** Everything else — the founders' brains, every
   tie broken at random, every probabilistic yes-or-no decision, every
   random input of a brain, every [share-out](gains-losses-and-the-share-out.md "How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.") of tokens, every [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") — is
   drawn from **one** stream: numpy's `RandomState` (the Mersenne Twister
   algorithm), seeded with the run's seed.

Because there is one stream and the program always takes its draws in the
same order — [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") are visited in increasing order of id, [neighbours](the-world.md#the-network "An agent joined to another by a connection.") in
increasing order of id — the whole history of a world is a function of its
[settings](settings.md "One number or switch of a run's rules, fixed before it starts.") and its seed. Nothing else enters: not the time, not the computer's
load, not how many runs go on at once.

## What "the same" needs

The same Python, numpy and networkx versions, and — this was found in
[Is a run reproducible?](is-a-run-reproducible.md) — the matrix library
held to **one thread**. With several threads, a sum of many products may be
added up in a different order, and floating-point addition is not exactly
associative: (*a* + *b*) + *c* can differ from *a* + (*b* + *c*) in the last
bit. A brain's output can then differ in its last digit, a tie can tip the
other way, and the two runs part. The [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") runs every simulation on one
thread.

## How a draw becomes a decision

Three kinds of draw appear in the rules:

- **a uniform number** *U* between 0 and 1: a yes-or-no decision read as a
  probability *q* says yes when *U* < *q* ([A yes-or-no decision](binary-decision.md));
- **a uniform choice** among several equal [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."): a tie for the largest
  stake, or a rung of a coalition ([How a coalition takes a node](revolution.md));
- **a multinomial draw**: *P* tokens dealt out among *N* agents, each token
  independently to agent *j* with probability *q*ⱼ. The result is *N* whole
  numbers that add up to exactly *P*. The [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") deals out the tokens of the
  dead this way, with *q*ⱼ = 1/*N*.

<!-- turns -->
---

← [The starting ring](starting-ring.md) · [How the simulation works](../README.md#how-the-simulation-works) · [The share function f(a, b)](share-function.md) →

> [!info]- Pages that link here (12)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [Is a run reproducible?](is-a-run-reproducible.md) · [Every setting](settings.md) · [The starting ring](starting-ring.md) · [A yes-or-no decision](binary-decision.md) · [How a brain changes](mutation.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [Births and the four ways to die](births-and-deaths.md) · [Gains, losses and the share-out](gains-losses-and-the-share-out.md) · [Physical inspiration](physical-inspiration.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
