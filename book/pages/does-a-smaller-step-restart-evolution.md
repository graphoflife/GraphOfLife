# Does a smaller step restart evolution?

> [!question] Questions of this page
> - If mutations are made small, does a world that has stopped getting
>   better get better again?
> - Are [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") made with small mutations better than brains made with the
>   engine's own, after the same time in the same world?
> - Does a structure of many [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") that formed under small mutations last,
>   and does it need them?

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
other. Its thesis is undecided: with mutations of 10 numbers the brains beat
the ones they started from (0.72 of the agents), but against the brains of
the unchanged runs the two seeds disagree
([What the contests said](#what-the-contests-said)).

The six runs are done, and they became four different kinds of world. Both
runs with mutations of 10 numbers left the old world behind within 600 to
1,700 iterations for one with a few hundred agents, many links and no
births. In one of them a core of about 160 rich agents formed, nearly all
linked to one another, whose agents lived on for over 2,000 iterations
with almost no births ([Does the core last?](#does-the-core-last)).

Experiment 53 played that world on for 5,000 iterations. With mutations of
10 numbers the core spread to the whole world: 476 agents with 50 to 64
links and almost the same [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") each, who all stake one token on every
[neighbour](the-world.md#the-network "An agent joined to another by a connection."), all marked for a coalition. Nobody is born and nobody dies. With
the engine's mutation given back, the world was thousands of short-lived
agents again within 500 iterations. The thesis, that the core would last,
is refuted: it did not last, it turned into something else.

## The world that is played on

The world is one of round 3's s3058 at 40,000 tokens, from Experiment 46
([Why does evolution level off?](why-does-evolution-level-off.md)), as it
stood at iteration 20,000: about 5,000 agents. The largest stake wins a
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
| as it is | about a tenth of the brain's 13,600 numbers, and sometimes redraws part of a layer | none |
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

## What the contests said

Experiment 51's contests were played on 24 November 2026: 191 of the 192,
one still to come. A score is the share of the agents a side holds after a
contest, the mean of its coins, with two standard errors either side.

**The thesis is undecided.**

| what was played | score | band | coins |
|---|---|---|---|
| **(1) the end against the start**: brains of 22,500 against those of 20,000 | | | |
| 10 numbers, both runs | 0.72 | 0.60 to 0.84 | 16 |
| 100 numbers, both runs | 0.22 | 0.14 to 0.30 | 16 |
| as it is, both runs | 0.30 | 0.10 to 0.49 | 16 |
| **(2) against the arm as it is**, both at 22,500 | | | |
| 10 numbers, both seeds | 0.63 | 0.42 to 0.83 | 16 |
| 100 numbers, both seeds | 0.65 | 0.47 to 0.84 | 16 |
| **the world before**: brains of 20,000 | | | |
| against those of 17,500 | 0.91 | 0.86 to 0.96 | 8 |
| against those of 15,000 | 0.90 | 0.88 to 0.92 | 7 |

- **With mutations of 10 numbers the brains got better than their start**:
  0.72, the whole band above a half, in both runs (0.68 and 0.76). That is
  the first part of the claim.
- **Against the arm as it is, nothing is settled.** The small arms hold
  0.63 and 0.65 of the agents, but the bands include a half, because the
  two seeds disagree. With seed 5002 the brains of 10 numbers hold 0.97;
  with seed 5001 they hold 0.28. That second pair is the core world against
  the thin world the unchanged run had become: brains made for one kind of
  world against brains made for another.
- **With 100 numbers the brains got worse than their start** (0.22), and
  so did the arm as it is (0.30; in one of its runs 0.10).
- **The world had not stopped.** Its brains of iteration 20,000 beat those
  of 17,500 and of 15,000 nine times in ten. The page's first sentence does
  not hold for this world: it was not at rest when it was played on.

Read by the tokens a side holds, the picture is the same but for one line:
the brains of 10 numbers against their start come to 0.55 (0.48 to 0.62).
They end with more agents than their start's brains, and about as many
tokens.

So this world does not climb and stop. It swings: brains that beat their
past nine times in ten at one moment lose to it 2,500 iterations later.
With mutations of 10 numbers it went somewhere new and its brains held
their gain. Whether that is "better" depends on the world the contest is
played in, and the six runs had become four kinds of world. A contest
between two moments says less than it seems when the world between them
has changed its kind.

The readings are made by `python3 research/efficient/E50/e51.py` (with
`tokens` after it for the share of the tokens).

## What it can and cannot show

Two seeds an arm and one world are little. A gain that shows here says a
smaller step can restart evolution in this world; it does not say that it
goes on, or that it holds elsewhere. No gain could mean that 2,500
iterations are too few for small steps to add up, or that this world's [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.")
has nothing more to find.

## Does the core last?

<!-- experiment E53 -->

The six runs of Experiment 50 were finished on 24 November 2026, before
their contests. They did not stay one kind of world. The world they started
from had about 5,000 agents with one or two links each, and one agent linked
to a third of all the others. After 2,500 iterations:

| run | agents | what the world is |
|---|---|---|
| as it is, seed 5001 | 1,036 | thin: no agent with more than 35 links, 36 tokens an agent |
| as it is, seed 5002 | 3,682 | as before: one agent with 1,224 links |
| 10 numbers, seed 5001 | 484 | a **core**: 166 agents with 84% of the links possible among them made, 145 tokens each, nine tenths of what agents hold; 318 poor agents round it |
| 10 numbers, seed 5002 | 687 | dense all over: an agent has 36 links (the middle one), 51 have more than 80 |
| 100 numbers, seed 5001 | 1,994 | had a core of 157 at iteration 21,250, and has lost it |
| 100 numbers, seed 5002 | 6,769 | a [star](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub."): one agent linked to all the others, holding 6,033 tokens |

In the first run with 10 numbers the core formed within 600 iterations and
then stood still in one respect: from iteration 20,600 on almost nobody was
born and few died (564 agents then, 484 at the end). The middle agent of the
core at 22,500 is 2,410 iterations old. Brains still spread, by winning nodes: seven or eight nodes in ten
change hands in every game.

This is the first lasting structure of several agents the notebook has
seen form by itself. Experiment 53 asks what it depends on. It plays that
world on for 5,000 iterations, twice with mutations of 10 numbers as
before, and twice with the engine's own mutation given back.

<!-- thesis E53 -->
> [!quote] The thesis of Experiment 53, written down before any of its runs existed
> **The claim.** The core of rich agents that formed in Experiment 50 needs small mutations. Played on for 5,000 iterations from the same kept world, it is still there in both runs whose mutations change 10 numbers, and it has fallen apart in both runs given back the engine's own mutation.
>
> **Why it would be so.** In Experiment 50 both branches with mutations of 10 numbers went from a world of 5,000 poor agents round one hub to a core: about 160 agents nearly all linked to one another, holding nine tenths of the agents' tokens, with almost no births and few deaths for over 2,000 iterations. A branch with mutations of 100 numbers formed such a core and lost it again, and neither branch with the engine's mutation formed one. Where the largest stake wins a node, a team of poor agents loses to one rich agent (card D2 of research/efficient/plan.md), so having no children and keeping every link is what holds a core together; a mutant that decides differently in a third of what it meets does neither. If the core lasts only while copies are faithful, the way brains vary decides which structures a world can keep.
>
> **It holds if:** A world has a core if at least 100 agents have more than half as many links as its best-linked agent, at least half of the links possible among them are made, and they hold at least half of the tokens agents hold. All four: both runs with 10 numbers have a core in the world kept at iteration 27,500; neither run with the engine's mutation has one there.
>
> **It fails if:** A run with 10 numbers has no core at 27,500, or a run with the engine's mutation has one. Reported beside it: each run's core in every world it kept, every 500 iterations, and its births per agent.
<!-- /thesis -->

<!-- runs E53 -->
> [!info]- The runs behind this page
> **4 runs.** Every condition below is run once for every seed (5301 to 5302), played on from the world `B1-40000-b7eb00-s5001-on45a3d3` kept at iteration 22,500, to iteration 27,500 — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-b7eb00-s5301-one41f90` … `B1-40000-b7eb00-s5302-one41f90`.
> - **the engine's mutation** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-bbbccd-s5301-one41f90` … `B1-40000-bbbccd-s5302-one41f90`.
>
> To make these runs again: `python3 gol_lab.py run E53`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

### What Experiment 53 found

The four runs were played on 24 November 2026, in about two and a half
hours.

**The thesis is refuted.** Neither run with mutations of 10 numbers has a
core at iteration 27,500, as the thesis defines one: in both, fewer than a
fifth of the links possible among the well-linked agents are made. The
second half of the claim held: both runs given back the engine's mutation
lost the core, within 500 iterations.

But the core did not fall apart under small mutations. It spread to the
whole world. What stands at iteration 27,500 is a world of equals:

| | 22,500 to 23,000 | 24,000 to 24,500 | 25,500 to 26,000 | 27,000 to 27,500 |
|---|---|---|---|---|
| agents | 478 | 476 | 476 | 476 |
| links an agent, least to most | 2 to 163 | 7 to 161 | 50 to 63 | 50 to 64 |
| tokens an agent, most | 162 | 158 | 63 | 64 |
| [Gini](measuring-tokens.md#the-gini-coefficient-and-the-lorenz-curve "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of tokens | 0.43 | 0.11 | 0.03 | 0.03 |
| births, deaths in 500 iterations | 37, 45 | 0, 0 | 2, 2 | 0, 0 |
| nodes won by a coalition | 100% | 100% | 100% | 100% |

(10 numbers, seed 5301. With seed 5302 the same from iteration 25,000 on:
472 agents, no births, no deaths, a Gini of 0.11, and no link cut at all.)

Every agent does the same thing. It puts one token on each of its 56
neighbours and one on itself, and marks every one of them for a
[coalition](rules-in-detail.md#how-a-coalition-takes-a-node). So on every
node the coalition of all its neighbours outweighs any single stake, takes
the node, and one of them, drawn by lot, puts its brain there. Every node
gets about as many tokens as its agent gave away. Nobody starves, every link
carries a stake and is kept, and nobody has a child. The same 476 agents
live on for 4,500 iterations while their brains change places every game:
only one node in fifty keeps its brain.

Under the engine's mutation the same world, from the same start, is back
within 500 iterations to what it was before Experiment 50: thousands of
agents that live a few iterations (50,000 to 180,000 births and as many
deaths in every 500 iterations), most tokens kept at home, and in one run a
single agent with over 2,000 links.

What this suggests, to be tested:

- **A way of acting that only works if nearly everyone follows it can last
  only where copies are faithful.** With one copy in five changed in a
  third of what it decides, such a way is broken faster than it spreads.
  With ten numbers changed, it holds for thousands of iterations.
- **The rule that carries it is the coalition.** Where the largest stake
  wins, one rich agent beats many poor ones; the coalition turns that
  round, and many small stakes beat any large one.
- **It is order, not yet evolution that goes on.** The even world is the
  most mixed state its tokens can be in, and it changed less and less from
  iteration 25,500 on. Whether its brains still get better is not known.

The readings are made by `python3 research/efficient/E50/e53.py`.

<!-- runs E50 -->
> [!info]- The runs behind this page
> **6 runs.** Every condition below is run once for every seed (5001 to 5002), played on from the world `B1-40000-bbbccd-s4602` kept at iteration 20,000, to iteration 22,500 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-bbbccd-s5001-on45a3d3` … `B1-40000-bbbccd-s5002-on45a3d3`.
> - **10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-b7eb00-s5001-on45a3d3` … `B1-40000-b7eb00-s5002-on45a3d3`.
> - **100 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 100, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-766af3-s5001-on45a3d3` … `B1-40000-766af3-s5002-on45a3d3`.
>
> To make these runs again: `python3 gol_lab.py run E50`.
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

- [A world of equals](a-world-of-equals.md): what the new state is, and
  Experiments 54 and 56, which ask whether it is an end and whether it
  forms from founders.
- [Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md):
  what one mutation does to a brain, and Experiment 49.
- [Why does evolution level off?](why-does-evolution-level-off.md):
  Experiment 46, whose world is played on here.

## How this was made

The plans are `book/experiments/E50.json` and `E51.json`; the second is
written by `python3 research/efficient/E50/make_e51.py`. They belong to
`research/efficient/plan.md`, the plan of 23 November 2026 to study
evolution with experiments of a day at most.
