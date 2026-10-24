# Seeds and the random stream

> [!question] Questions of this page
> - A computer cannot throw dice: where does the chance in a world come from?
> - What exactly does a run's seed decide?
> - What has to stay the same for a run to come out the same again?
> - How does a random number become a decision?

## In short

Graph of Life is full of chance: random starting [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."), random tie-breaks,
probabilistic decisions, random inputs, random sharing out of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). Yet a
run can be made again exactly. A computer makes its random numbers by a fixed
calculation from a starting number, the **seed**, so the same seed gives the
same numbers. A run's seed starts two streams: one builds the starting ring,
and one gives the world every other random choice, always in the same order.
So a run is decided by its [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") and its seed alone, given the same
libraries and a matrix library held to one thread.

## Dice from a formula

A computer does not throw dice. It computes a long sequence of numbers that
*look* random — they pass the usual statistical tests — from a starting value
called the **seed**. Think of a vast book of random digits: the seed says on
which page to start reading. The same seed gives the same sequence, every
time, on every computer that uses the same algorithm.

For example, start numpy's `RandomState`, the generator every world uses, from
seed 1, and ask it for three numbers between 0 and 1. It gives 0.417, 0.720
and 0.000114. From seed 2 it gives 0.436, 0.026 and 0.550. Ask again
tomorrow, on another computer, and the answers are the same.

## The two streams of a run

Every run has one seed, a whole number; most experiments use 1, 2, 3, ….
A run started without a seed is given one drawn at random, and that seed is
written down with the run. From the seed, two streams are started:

1. **The ring's stream.** The networkx library builds the starting ring
   ([The starting ring](starting-ring.md)) from a generator of its own,
   seeded with the run's seed.
2. **The world's stream.** Everything else is drawn from **one** stream:
   numpy's `RandomState` (the Mersenne Twister algorithm), seeded with the
   run's seed. It gives the founders' brains, every tie broken at random and
   every probabilistic yes-or-no decision. It also gives every random input of
   a brain, every [share-out](gains-and-losses.md#gains-losses-and-the-share-out "How a cleanup deals the tokens of the dead out among the living: in the baseline, each token to a survivor drawn at random.") of tokens and every mutation.

The program always takes its draws in the same order: [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") are visited in
increasing order of id, and so are [neighbours](the-world.md#the-network "An agent joined to another by a connection."). So the whole history of a world
is a function of its settings and its seed. Nothing else enters: not the
time, not the computer's load, not how many runs go on at once.

Two more things keep it so. Every world has a stream of its own, so two
worlds played side by side in one program cannot disturb each other. And a
checkpoint saves the stream's place, so a run stopped and resumed draws
exactly the numbers it would have drawn
([Is a run reproducible?](is-a-run-reproducible.md)).

One stream for a whole world has a price. If one agent anywhere takes one
more random number, every later draw everywhere shifts.
[Physical inspiration](physical-inspiration.md) counts this among the three
rules that break locality.

## What "the same" needs

A run comes out the same only with the same Python, numpy and networkx
versions. It also needs the matrix library held to **one thread**, as
[Is a run reproducible?](is-a-run-reproducible.md) found. Here is why. A brain
computes sums of many products. With several threads, a sum may be added up
in a different order. And on a computer, adding floating-point numbers is not
exactly associative. The sum (*a* + *b*) + *c* can differ from
*a* + (*b* + *c*) in the last bit. For example, (0.1 + 0.2) + 0.3 gives 0.6000000000000001, but
0.1 + (0.2 + 0.3) gives 0.6. A brain's output can then differ in its last
digit, a tie can tip the other way, and the two runs part.

So the lab runs every simulation on one thread. A different processor or
library version can round a sum differently too. So every run records its
versions and its processor. The lab refuses to continue a run under different
ones ([The lab, the engine and pilots](the-lab.md)).

## How a draw becomes a decision

Three kinds of draw appear in the rules:

- **a uniform number** *U* between 0 and 1. A yes-or-no decision read as a
  probability *q* says yes when *U* < *q*, which happens with probability
  exactly *q* ([A yes-or-no decision](binary-decision.md)).
- **a uniform choice** among several equal [candidates](one-iteration.md#phase-1-reproduction "An agent itself and each of its neighbours, the ones its brain looks at when it decides."). It breaks a tie for
  the largest stake. It also picks which of the equal stakers that tipped a
  coalition takes the node ([How a coalition takes a node](revolution.md)).
- **a multinomial draw**: *P* tokens dealt out among *N* agents, each token
  independently to agent *j* with probability *q*ⱼ. The result is *N* whole
  numbers that add up to exactly *P*. The [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") deals out the tokens of the
  dead this way. Under the [baselines](b1-and-b2.md "The stated rules an experiment starts from and changes one thing of: B1 or B2.")' rule, `uniform`, every *q*ⱼ is 1/*N*.

A small multinomial draw: deal 3 tokens to 2 agents, each token to either one
with probability ½. The outcomes are (3, 0), (2, 1), (1, 2) and (0, 3), with
chances 1/8, 3/8, 3/8 and 1/8. Every outcome adds up to 3: no token is lost or
made.

Other draws feed a brain rather than decide for it. Each random input is a
uniform number between −2 and 2 ([What a brain sees](brain-inputs.md)). A
mutation adds normal draws to a brain's numbers
([How a brain changes](mutation.md)).

## Where to go next

- [Is a run reproducible?](is-a-run-reproducible.md): the experiment that
  tested all this, and found the threads.
- [How much does the seed decide?](how-much-does-the-seed-decide.md): how
  differently thirty worlds of the same settings turn out.
- [How many seeds an experiment needs](bootstrap.md#how-many-seeds-an-experiment-needs): why an experiment
  runs many seeds.
- [The starting ring](starting-ring.md): what the ring's stream builds.

<!-- turns -->
---

← [The starting ring](starting-ring.md) · [How the simulation works](../README.md#how-the-simulation-works) · [The share function f(a, b)](share-function.md) →

> [!info]- Pages that link here (14)
> [Glossary](glossary.md) · [The world: tokens, agents and a network](the-world.md) · [Is a run reproducible?](is-a-run-reproducible.md) · [Every setting](settings.md) · [The starting ring](starting-ring.md) · [A yes-or-no decision](binary-decision.md) · [What a brain sees](brain-inputs.md) · [How a brain changes](mutation.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [The life of a B1 world](a-worlds-life.md) · [Births and the four ways to die](births-and-deaths.md) · [Gains and losses: who gains, who loses, and the share-out](gains-and-losses.md) · [Physical inspiration](physical-inspiration.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
