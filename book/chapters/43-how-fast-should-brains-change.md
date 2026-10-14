# How fast should brains change?

> [!question] Questions of this chapter
> - Does the rate at which brains change decide where evolution stops, or
>   only how soon it gets there?
> - Is there a rate past which selection can no longer keep what it has found?
> - What does the rate do to a world — to how many kinds of brain it holds, and
>   how often its nodes change hands?

<!-- experiment E09 -->

## Why ask

Evolution needs **variation** ([Chapter 1](01-what-this-book-is-about.md)).
In B2 a brain changes only when it is copied — into a newborn, or into a node
its owner has just won — and then with probability *p* = 0.2
([How a brain changes](../notes/mutation.md);
[`mutate_on_replication`](../notes/settings.md#mutate_on_replication)). One
copy in five is a new genotype. Is that a lot or a little?

[Chapter 53](53-why-evolution-stops.md) proposed that evolution stops because
the game has an answer: a contest for nodes is best played at random, and
once a population plays it well, nothing better exists to be selected. That
reading makes a prediction about *p*. Theory offers three pictures of what the
rate of change can do:

1. **Selection finds the answer.** If one way of playing is best, any world
   that varies enough to find it ends up there, however fast it varies: the
   rate of change decides how soon a world arrives, not where.
2. **Mutation–selection balance** (Haldane 1927). Changes are blind, so they
   keep breaking what selection has built, and selection keeps repairing it.
   How much is broken at any moment rises with how often things break: the
   brains of a faster-changing world are a little worse, the more the faster.
3. **An error threshold** (Eigen 1971; Eigen and Schuster 1977). Past a
   critical rate, a well-adapted type melts into a cloud of variants faster
   than selection can favour it, and what was learned is lost all at once.
   Below the threshold little changes; above it, everything does.

If the game has an answer, the first picture should hold over a wide range of
rates. If evolution stops for another reason — because damage keeps pace with
repair — the second or third should.

## Why the question moved

This experiment was first planned on the baseline B1, to ask whether three
births per hundred agents is a rate selection finds or the churn of
mutation. Two pilots at 10,000 tokens (seed 1, 600 iterations) pointed to
churn: births per hundred agents were 2.0 at *p* = 0.05, 2.9 at the baseline's
0.2 and 8.0 at 0.8, and genotypes per agent 0.32, 0.60 and 0.98 — a
mutation–selection balance, in a world whose brains are nearly blind.

In B2 that test is gone. Every agent dies at age 100, so births are held at
one in a hundred agents per iteration whatever the brains want
([Chapter 52](52-who-dies-and-why.md)). What the rate of change can still
move is how good the brains become — and that is the question Part V needs
answered.

## How the rates are compared

Five rates, each double the one before: *p* = 0.05, 0.1, 0.2 (B2), 0.4 and
0.8 — from one copy in twenty changed to four in five. Each is run with ten
seeds at 10,000 tokens for 1,500 iterations, archiving the whole world every
100. Worlds with the same seed start from the same founders, the same brains
and the same ring; they part at the first copy one of them changes and the
other does not.

Two kinds of contest are played, both as in
[Chapter 49](49-how-to-tell-a-world-that-keeps-evolving.md) — a world whose
agents each get a brain from one side or the other by a coin, mutation off,
twenty iterations, and the share each side holds at the end:

- **Across rates.** For each seed and each pair of rates, the brains of
  iteration 1,500 of one rate play those of the other, once in each rate's
  world at iteration 1,500. Ten pairs, ten seeds, two worlds: 200 contests.
  Under the first picture every pair ties; under the second the slower rate
  wins, the more the further apart; under the third the fastest rate loses to
  all the others.
- **Within a rate.** Each world plays its own eras, as in Chapter 49 — to see
  when each rate stops, and whether the slowest has stopped at all by
  iteration 1,500.

Beside them, the world statistics of every run: agents, births, genotypes per
agent, and the share of nodes kept by their own agent in a game.

## The thesis

<!-- thesis E09 -->
<!-- /thesis -->

## The runs

<!-- runs E09 -->
<!-- /runs -->

> [!warning] Waiting for the runs
> The plan, the thesis and the method above were written before any run of
> Experiment 9 existed; the results follow when they are done. The B1 plan
> this chapter had before was never run.

<!-- turns -->
<!-- /turns -->
