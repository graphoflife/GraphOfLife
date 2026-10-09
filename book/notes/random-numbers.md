# Seeds and the random stream

Graph of Life is full of chance: random starting brains, random tie-breaks,
probabilistic decisions, random inputs, random sharing out of tokens. Yet a
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
   random input of a brain, every share-out of tokens, every mutation — is
   drawn from **one** stream: numpy's `RandomState` (the Mersenne Twister
   algorithm), seeded with the run's seed.

Because there is one stream and the program always takes its draws in the
same order — agents are visited in increasing order of id, neighbours in
increasing order of id — the whole history of a world is a function of its
settings and its seed. Nothing else enters: not the time, not the computer's
load, not how many runs go on at once.

## What "the same" needs

The same Python, numpy and networkx versions, and — this was found in
[Chapter 8](../chapters/08-is-a-run-reproducible.md) — the matrix library
held to **one thread**. With several threads, a sum of many products may be
added up in a different order, and floating-point addition is not exactly
associative: (*a* + *b*) + *c* can differ from *a* + (*b* + *c*) in the last
bit. A brain's output can then differ in its last digit, a tie can tip the
other way, and the two runs part. The lab runs every simulation on one
thread.

## How a draw becomes a decision

Three kinds of draw appear in the rules:

- **a uniform number** *U* between 0 and 1: a yes-or-no decision read as a
  probability *q* says yes when *U* < *q* ([A yes-or-no decision](binary-decision.md));
- **a uniform choice** among several equal candidates: a tie for the largest
  stake, or a rung of a coalition ([How a coalition takes a node](revolution.md));
- **a multinomial draw**: *P* tokens dealt out among *N* agents, each token
  independently to agent *j* with probability *q*ⱼ. The result is *N* whole
  numbers that add up to exactly *P*. The cleanup deals out the tokens of the
  dead this way, with *q*ⱼ = 1/*N*.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
