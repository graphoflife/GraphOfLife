# Does a smaller step restart evolution?

> [!question] Questions of this page
> - If mutations are made small, does a world that has stopped getting
>   better get better again?
> - Are [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") made with small mutations better than brains made with the
>   engine's own, after the same time in the same world?

<!-- experiment E50 -->
<!-- experiment E51 -->

## In short

Every world measured so far gets better early and then stops
([Why does evolution level off?](why-does-evolution-level-off.md)). One
mutation of the engine changes so much of a brain that the mutant is close to
a new brain
([Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md)).
If that is what stops evolution, a smaller step should start it again.

Experiment 50 takes one kept world and plays it on six times for 2,500
[iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."): twice as it is, twice with mutations of exactly 10 of a brain's
numbers, twice with 100. Experiment 51 then plays the brains against each
other. Neither has been played yet.

## The world that is played on

The world is one of round 3's s3058 at 40,000 [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."), from Experiment 46
([Why does evolution level off?](why-does-evolution-level-off.md)), as it
stood at iteration 20,000: about 5,000 [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."). The largest stake wins a
node, the [estate](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") of the dead goes to their children, and an agent lives
about six iterations, longer than in most of round 3's worlds. Whether this
world had stopped getting better by iteration 20,000 was never measured.
The two runs played on as it is, and two contests of its own earlier brains,
say so.

A run played on starts from that world: the same agents, brains, links and
messages. From there it has its own settings and its own
[seed](rules-in-detail.md#seeds-and-the-random-stream), so the six runs
differ by their mutations and their luck, and by nothing else
([Runs and experiments](runs-and-records.md#runs-and-experiments)).

## The three arms

| arm | a mutation changes | setting |
|---|---|---|
| as it is | about a tenth of the brain's 12,800 numbers, and sometimes redraws part of a layer | none |
| 10 numbers | exactly 10 numbers, nothing redrawn | [`mutation_numbers`](settings.md#mutation_numbers) = 10 |
| 100 numbers | exactly 100 numbers, nothing redrawn | `mutation_numbers` = 100 |

How often a brain mutates is the same in all three: one copy in five.

<!-- thesis E50 -->
> [!quote] The thesis of Experiment 50, written down before any of its runs existed
> **The claim.** A world that has stopped getting better gets better again when its mutations are made small. Played on for 2,500 iterations from the same kept world, with mutations of 10 or of 100 numbers, its brains beat the brains it started from, and they beat the brains of the same world played on with the engine's own mutation.
>
> **Why it would be so.** One mutation of the engine leaves a brain deciding differently from its parent in a third to two thirds of the situations it meets, and in one world as differently as an unrelated brain; with ten numbers changed, in one or two in a hundred (card B1 of research/efficient/plan.md). Selection can sort such mutants but cannot improve a brain step by step, which would explain why every world measured gets better early and then stops. A mutation small enough to keep most of what a brain does lets a small gain be found and kept. The world played on is one of round 3's s3058 at 40,000 tokens (Experiment 46), kept at iteration 20,000: the largest stake wins a node, estates go to the children of the dead, and an agent lives about six iterations.
>
> **It holds if:** Both, for at least one of the two small arms (10 numbers, 100 numbers). A coin is two contests with one deal, the two sides in swapped places, and scores the mean of the two; a score is the mean of its coins, and its band two standard errors of that mean either side. (1) The arm's brains at iteration 22,500 against the brains of iteration 20,000, on both worlds, pooled over its two runs: the score and its whole band are above 0.5. (2) The arm's brains at 22,500 against the brains at 22,500 of the arm played on as it is, on both arms' worlds, pooled over the two seeds: the score and its whole band are above 0.5.
>
> **It fails if:** For both small arms the score of (2) is 0.5 or less: brains made with small mutations are no better than brains made with the engine's. Anything else is undecided. Reported beside it: (1) for the arm played on as it is, which says whether the world had stopped; and the world's own earlier brains, iteration 20,000 against 17,500 and against 15,000.
<!-- /thesis -->

## The contests

A contest gives the agents of a kept world brains drawn from two sets and
plays 100 iterations with every mutation off
([Eras, contests and invasions](measuring-evolution.md#eras-contests-and-invasions-does-a-world-keep-getting-better)).
Here every coin is two contests with one deal, the two sets in swapped
places, and every coin has a deal of its own. Three kinds are played:

- **each run's end against its start**: the brains of iteration 22,500
  against those of 20,000, on both worlds;
- **the small arms against the arm as it is**, both at 22,500, on both arms'
  worlds, for each seed;
- **the world's own past**: iteration 20,000 against 17,500 and against
  15,000, to see whether it had stopped.

## What it can and cannot show

Two seeds an arm and one world are little. A gain that shows here says a
smaller step can restart evolution in this world; it does not say that it
goes on, or that it holds elsewhere. No gain could mean that 2,500
iterations are too few for small steps to add up, or that this world's [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")
has nothing more to find.

<!-- runs E50 -->
> [!info]- The runs behind this page
> **6 runs.** Every condition below is run once for every seed (5001 to 5002), played on from the world `B1-40000-bbbccd-s4602` kept at iteration 20,000, to iteration 22,500 — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-bbbccd-s5001-on45a3d3` … `B1-40000-bbbccd-s5002-on45a3d3`.
> - **10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-b7eb00-s5001-on45a3d3` … `B1-40000-b7eb00-s5002-on45a3d3`.
> - **100 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 100, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-766af3-s5001-on45a3d3` … `B1-40000-766af3-s5002-on45a3d3`.
>
> To make these runs again: `python3 gol_lab.py run E50`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

<!-- runs E51 -->
> [!info]- The trials behind this page
> **192 trials**, contests: the world of one era, its agents given brains drawn by coin from two eras, every mutation off, and the later era's share of the agents counted at the end; each plays 100 iterations, from a world another run kept in its archive.
>
> - **as it is, seed 5001: 22,500 against 20,000** — 16 trials: worlds `B1-40000-bbbccd-s5001-on45a3d3`; eras 20,000, 22,500; coins 1 to 8, every 1.
> - **as it is, seed 5002: 22,500 against 20,000** — 16 trials: worlds `B1-40000-bbbccd-s5002-on45a3d3`; eras 20,000, 22,500; coins 9 to 16, every 1.
> - **10 numbers, seed 5001: 22,500 against 20,000** — 16 trials: worlds `B1-40000-b7eb00-s5001-on45a3d3`; eras 20,000, 22,500; coins 17 to 24, every 1.
> - **10 numbers, seed 5002: 22,500 against 20,000** — 16 trials: worlds `B1-40000-b7eb00-s5002-on45a3d3`; eras 20,000, 22,500; coins 25 to 32, every 1.
> - **100 numbers, seed 5001: 22,500 against 20,000** — 16 trials: worlds `B1-40000-766af3-s5001-on45a3d3`; eras 20,000, 22,500; coins 33 to 40, every 1.
> - **100 numbers, seed 5002: 22,500 against 20,000** — 16 trials: worlds `B1-40000-766af3-s5002-on45a3d3`; eras 20,000, 22,500; coins 41 to 48, every 1.
> - **10 numbers against as it is, seed 5001, at 22,500** — 16 trials: worlds `B1-40000-b7eb00-s5001-on45a3d3`, `B1-40000-bbbccd-s5001-on45a3d3`; eras 22,500; coins 49 to 56, every 1.
> - **10 numbers against as it is, seed 5002, at 22,500** — 16 trials: worlds `B1-40000-b7eb00-s5002-on45a3d3`, `B1-40000-bbbccd-s5002-on45a3d3`; eras 22,500; coins 57 to 64, every 1.
> - **100 numbers against as it is, seed 5001, at 22,500** — 16 trials: worlds `B1-40000-766af3-s5001-on45a3d3`, `B1-40000-bbbccd-s5001-on45a3d3`; eras 22,500; coins 65 to 72, every 1.
> - **100 numbers against as it is, seed 5002, at 22,500** — 16 trials: worlds `B1-40000-766af3-s5002-on45a3d3`, `B1-40000-bbbccd-s5002-on45a3d3`; eras 22,500; coins 73 to 80, every 1.
> - **the world before: 20,000 against 17,500** — 16 trials: worlds `B1-40000-bbbccd-s4602`; eras 17,500, 20,000; coins 81 to 88, every 1.
> - **the world before: 20,000 against 15,000** — 16 trials: worlds `B1-40000-bbbccd-s4602`; eras 15,000, 20,000; coins 89 to 96, every 1.
>
> To play these trials again: `python3 gol_lab.py run E51`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

## Where to go next

- [Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md):
  what one mutation does to a brain, and Experiment 49.
- [Why does evolution level off?](why-does-evolution-level-off.md):
  Experiment 46, whose world is played on here.

## How this was made

The plans are `book/experiments/E50.json` and `E51.json`; the second is
written by `python3 research/efficient/E50/make_e51.py`. They belong to
`research/efficient/plan.md`, the plan of 23 November 2026 to study
evolution with experiments of a day at most.
