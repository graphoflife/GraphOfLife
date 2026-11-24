# Are mutants as fit as their parents?

> [!question] Questions of this page
> - How much does one [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") change what a [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") decides?
> - Does a mutant do as well as the [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") it came from?
> - Is the size of a mutation what stops evolution?
> - Is a line's success in its brain, in its place, or luck?

<!-- experiment E49 -->

## In short

The worlds measured so far get better for a while and then stop
([Why does evolution level off?](why-does-evolution-level-off.md)). One cause
could be the way brains vary. When the engine changes a brain, it changes a
share of all its numbers at once, and often draws part of a layer afresh.

Asked the same 300 real situations, a brain changed once in that way decides
differently from its parent in 34% to 67% of them. In one of three worlds
that is as often as two unrelated brains of the world differ. A brain with
one number changed decides differently in under 1% of them.

Experiment 49 asked whether that also shows in how well a mutant does. It
marked mutants and unchanged agents in a kept world and counted their copies
30 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") later. That count turned out to say nothing: by iteration 30 a
single line has taken over, and which one is chance. By the thesis's own
words it is refuted, but the reading it names could not have decided either
way. The first few iterations, which the thesis does not read, do show a
pattern. In s4003 a line whose brain was changed by the engine's mutation
has 0.69 times the copies of an unchanged line after three iterations and
0.41 to 0.58 times after six. Mutations of 1, 10 and 100 numbers cost
nothing that can be measured. In s4008 no mutation costs anything.

Experiment 52 then asked how much of a line's success is its brain's at
all. It replayed 400 marked agents with other luck and with their brains
swapped. The place a line starts in counts for far more than its brain. In
a world where the largest stake wins, a brain that did well in one place
does better in another too (a [correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes "Whether two quantities go up and down together, from −1 to 1.") of 0.25). In a world that plays
the lottery it barely does (0.09).

## What one mutation does to a brain's decisions

A **probe** is one real situation: what an agent of a kept world saw of
itself and its [neighbours](the-world.md#the-network "An agent joined to another by a connection."), and the [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") it held. A brain is asked what it
would do there: how many tokens it would give a child, and where its first
token would go in the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."). Two brains are compared by the share of 300
probes on which they decide differently. Nothing is played.

For each world, 200 brains were drawn, and 20 mutants made of each in five
ways: by the engine's own mutation, and by changing exactly 1, 10, 100 or
1,000 of the brain's numbers by the engine's own noise, with nothing drawn
afresh.

| world, iteration | numbers in a brain | two unrelated brains differ on | engine's mutation | 1 number | 10 | 100 | 1,000 |
|---|---|---|---|---|---|---|---|
| s4003, 15,000 | 9,459 | 76% | 67% | 0.4% | 2.0% | 6.5% | 20% |
| s4003, 5,000 | 9,459 | 62% | 54% | 0.2% | 0.9% | 3.3% | 10% |
| s4008, 15,000 | 7,123 | 83% | 49% | 1.1% | 5.6% | 18% | 43% |
| s4002, 15,000 | 12,835 | 83% | 34% | 0.4% | 2.3% | 7.7% | 21% |

Each cell is the mean share of probes on which a mutant decides differently
from its parent. In s4003 at 15,000, 47% of the engine's mutants were at
least as far from their parent as an unrelated brain is. With one number
changed, 43% to 72% of mutants decided exactly as their parent on every
probe.

Two more things show in the same reading. The brains of one world do not
share one way of deciding: two of them drawn at random differ on 62% to 83%
of the probes. And the worlds' populations change what they do between any
two kept worlds about as much after 2,500 iterations as after 25,000, far
more than two halves of one population differ. Behaviour is neither still
nor moving in a direction.

## The assay

An **assay** is a short play of a kept world
([Runs and experiments](runs-and-records.md#runs-and-experiments)). 200
agents are drawn at random, each a **site**. At 100 of them the brain is
changed once; at the other 100, the controls, it is left as it is. Every
site's brain gets a mark of its own. The world then plays 30 iterations with
every mutation off, so every copy keeps its mark, and the agents carrying
each mark are counted. A count of 1 is a line that held its place, 0 one
that died out, more one that spread.

Mutants and controls play in the same world at the same time, so their
counts compare directly. A mark changes nothing in the play: the engine
never reads it. Worlds that mix two brains at every birth cannot be marked
this way, which leaves out s4002.

<!-- thesis E49 -->
> [!quote] The thesis of Experiment 49, written down before any of its runs existed
> **The claim.** The engine's own mutation makes a brain less fit, and a small change does not. In three kept worlds of Experiment 48, an agent whose brain was changed once by the engine's mutation leaves fewer copies over 30 iterations than an agent left as it was. An agent with 1 or 10 of its brain's numbers changed leaves as many.
>
> **Why it would be so.** Asked the same 300 real situations, a brain changed once by the engine's mutation decides differently from its parent in 34% to 67% of them, and in one world as often as an unrelated brain of the same world does; with one number changed it decides differently in under 1% of them (card B1 of research/efficient/plan.md, 23 November 2026). Experiment 48's worlds beat their founders by iteration 5,000 and then stopped getting better. If a mutant is nearly a new brain, selection can sort brains but cannot improve one step by step. This experiment asks whether that difference in what a mutant decides is also a difference in how well it does.
>
> **It holds if:** For each world, pooled over its ten assays, R is the mean count at iteration 30 of the 1,000 mutant sites over that of the 1,000 control sites. Both: R of the engine's mutation is below 0.8 in all three worlds, each with p below 0.05 when the labels mutant and control are shuffled within each assay (2,000 shuffles); and R is above 0.9 for one number changed and for ten, in all three worlds.
>
> **It fails if:** Either: R of the engine's mutation is 0.9 or more in two of the three worlds, so a mutant is as fit as its parent and the size of a mutation is not what stops evolution; or R is below 0.8 for one number changed or for ten in two of the three worlds, so even a small change breaks a brain. Anything else is undecided. R for 100 and for 1,000 numbers changed, and the share of sites that died out, are reported beside it.
<!-- /thesis -->

What the assay cannot show is a small gain. Over 30 iterations most single
lines die out by chance, so one mutant that is a little better than its
parent cannot be told from a lucky one. The assay measures what a kind of
mutation costs on average.

## What the assays said

All 150 assays were played on 23 and 24 November 2026, in 2 hours 23
minutes on three workers.

### The reading the thesis names

R is the mean count of the mutant sites at iteration 30 over that of the
control sites, pooled over a world's ten assays.

| world | engine's mutation | 1 number | 10 | 100 | 1,000 |
|---|---|---|---|---|---|
| s4003 at 15,000 | 0.00 | 3,261 | 0.22 | 0.01 | 0.98 |
| s4003 at 5,000 | 0.78 | 0.23 | 1.03 | no control left | 0.00 |
| s4008 at 15,000 | 8.83 | 0.66 | 2.29 | 4.38 | 0.01 |

**By its own words the thesis is refuted**: R for one number changed is
below 0.8 in two of the three worlds. But these numbers say nothing. With the
labels mutant and control shuffled, R ranges just as widely, and none of the
fifteen differs from what shuffling gives (the smallest *p* is 0.07, for one
number changed in s4003 at 15,000, where R is 3,261). The reading could not
have confirmed the thesis either.

The reason is how fast lines die in these worlds. Of 100 marked lines, a
third are left after three iterations, about three after nine, and after
30 a few lines in a thousand, each of which by then holds hundreds or
thousands of agents: up to 3,247. The mean count at iteration
30 is the count of that one line, and whether it started as a mutant or a
control is a coin toss. The plan chose iteration 30 without knowing this.
That is a fault of the plan, not of the worlds.

### The first iterations (not what the thesis reads)

Each site's count was kept every three iterations. The same ratio R after
three and after six iterations, with the range it takes in 95% of 2,000
shufflings of the labels:

| world | mutation | R after 3 | shuffled | R after 6 | shuffled |
|---|---|---|---|---|---|
| s4003 at 15,000 | engine's | **0.69** | 0.84 to 1.19 | **0.41** | 0.62 to 1.63 |
| | 1 number | 1.02 | 0.85 to 1.18 | 0.96 | 0.66 to 1.50 |
| | 10 | 0.98 | 0.85 to 1.19 | 1.04 | 0.70 to 1.46 |
| | 100 | 1.08 | 0.86 to 1.18 | 1.05 | 0.62 to 1.60 |
| | 1,000 | 0.91 | 0.85 to 1.19 | 0.72 | 0.63 to 1.58 |
| s4003 at 5,000 | engine's | **0.69** | 0.87 to 1.16 | **0.58** | 0.74 to 1.36 |
| | 1 number | 1.01 | 0.86 to 1.16 | 1.02 | 0.79 to 1.28 |
| | 10 | 1.03 | 0.88 to 1.15 | 1.23 | 0.77 to 1.32 |
| | 100 | 0.94 | 0.88 to 1.15 | 1.03 | 0.78 to 1.28 |
| | 1,000 | 1.02 | 0.87 to 1.15 | 1.09 | 0.77 to 1.28 |
| s4008 at 15,000 | engine's | 1.08 | 0.80 to 1.26 | 1.08 | 0.66 to 1.50 |
| | 1 number | 1.18 | 0.81 to 1.23 | 0.94 | 0.67 to 1.51 |
| | 10 | 1.01 | 0.78 to 1.28 | 1.17 | 0.67 to 1.51 |
| | 100 | 1.03 | 0.81 to 1.23 | 1.02 | 0.71 to 1.42 |
| | 1,000 | 1.03 | 0.80 to 1.24 | 1.13 | 0.64 to 1.53 |

In s4003, at both iterations of its history, a line changed by the engine's
mutation falls behind at once: 0.69 times the copies after three iterations
(*p* = 0.0005 at both), 0.41 and 0.58 times after six. No smaller mutation
leaves its range. In s4008 nothing does, the engine's mutation included,
though it changes half of what a brain there decides.

This was not planned before the assays, so it is a pattern to test, not a
result. What it suggests:

- In s4003, a copy that mutates is mostly a copy lost. One copy in five
  mutates, so the world pays for its mutations with roughly a tenth of all
  its copies.
- A mutation of up to a hundred numbers keeps what the brain can do. That
  is the step a search for small gains needs
  ([Does a smaller step restart evolution?](does-a-smaller-step-restart-evolution.md)).
- In s4008 what a brain decides hardly matters for how its line does over a
  few iterations. That world plays the lottery, in which every way of
  staking returns the same tokens on average, and its brains mutate where
  they sit, one in five of them every iteration.

<!-- runs E49 -->
> [!info]- The trials behind this page
> **150 trials**, assays: in the world of one era some agents' brains are changed once and marked, as many more only marked, every mutation off, and the agents carrying each mark counted at the end; each plays 30 iterations, from a world another run kept in its archive.
>
> - **s4003 at 15,000** — 50 trials: worlds `B1-10000-085a32-s4801`; at iteration 15,000; mutations of 1, 10, 100, 1,000, engine; mutants 100; controls 100; seeds 1 to 10, every 1.
> - **s4003 at 5,000** — 50 trials: worlds `B1-10000-085a32-s4801`; at iteration 5,000; mutations of 1, 10, 100, 1,000, engine; mutants 100; controls 100; seeds 1 to 10, every 1.
> - **s4008 at 15,000** — 50 trials: worlds `B1-10000-1cfaa4-s4801`; at iteration 15,000; mutations of 1, 10, 100, 1,000, engine; mutants 100; controls 100; seeds 1 to 10, every 1.
>
> To play these trials again: `python3 gol_lab.py run E49`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

## Brain, place or luck?

<!-- experiment E52 -->

The first assays of Experiment 49 showed something the plan had not reckoned
with. Marked lines die out fast: after three iterations a third are left,
after nine about one in thirty. And a single line among 200 can hold a
thousand agents or more after 30 iterations. Chance alone does not spread a
line like that, so lines differ in how well they do.

A line can do well for three reasons: its brain, the place it started in
(its tokens, links and neighbours), or luck. Only the brain is inherited, so
only its share can be selected. Experiment 52 takes the three apart. It
marks 400 agents and plays them again and again from the same kept world:
six times as they are, with different luck, and six times with the marked
brains moved among the marked agents, each brain in another's place.

<!-- thesis E52 -->
> [!quote] The thesis of Experiment 52, written down before any of its runs existed
> **The claim.** How many copies an agent's line has after six iterations is partly its brain's doing: a brain that did well in one place does better than others in another place too. The place counts for more than the brain, and luck for more than both.
>
> **Why it would be so.** In the assays of Experiment 49 one line among 200 marked agents held up to half the world after 30 iterations, far more than chance spreads a line, so lines differ in how well they do. A line could do well because of its brain, or because of where it started: the tokens it held, its links, its neighbours. Only what is in the brain can be selected. If the brain's share is nil, there is no selection on brains at all in these worlds, and no way of varying brains could help.
>
> **It holds if:** In each of the two worlds, 400 agents are marked and played 9 iterations: six times as they are, with different luck, and six times with the marked brains moved among the marked agents. A line's score is the logarithm of one plus its copies after six iterations. r_brain is the correlation, over the 400 brains, between a brain's mean score in its own place and its mean score in the places it was moved to; r_place the correlation, over the 400 places, between a place's mean score with its own brain and with other brains. Bands are from 2,000 resamplings of the 400. Both, in both worlds: the band of r_brain lies above 0; and r_place is larger than r_brain.
>
> **It fails if:** In both worlds the band of r_brain includes 0 or lies below it: over six iterations a brain's success does not travel with it. Anything else is undecided. Reported beside it: the same with the share of lines alive after three iterations, and r_same, the correlation between two plays of the same brain in the same place.
<!-- /thesis -->

<!-- runs E52 -->
> [!info]- The trials behind this page
> **24 trials**, assays: in the world of one era some agents' brains are changed once and marked, as many more only marked, every mutation off, and the agents carrying each mark counted at the end; each plays 9 iterations, from a world another run kept in its archive.
>
> - **s4003 at 15,000: as they are** — 6 trials: worlds `B1-10000-085a32-s4801`; at iteration 15,000; mutations of 1; mutants 0; controls 400; seeds 1; played with the luck of seeds 11 to 16, every 1.
> - **s4003 at 15,000: brains swapped** — 6 trials: worlds `B1-10000-085a32-s4801`; at iteration 15,000; mutations of 1; mutants 0; controls 400; seeds 1; brains moved by seeds 1, 2, 3; played with the luck of seeds 21, 22.
> - **s3058 at 20,000: as they are** — 6 trials: worlds `B1-40000-bbbccd-s4602`; at iteration 20,000; mutations of 1; mutants 0; controls 400; seeds 1; played with the luck of seeds 11 to 16, every 1.
> - **s3058 at 20,000: brains swapped** — 6 trials: worlds `B1-40000-bbbccd-s4602`; at iteration 20,000; mutations of 1; mutants 0; controls 400; seeds 1; brains moved by seeds 1, 2, 3; played with the luck of seeds 21, 22.
>
> To play these trials again: `python3 gol_lab.py run E52`.
<!-- /runs -->

### What the assays said (Experiment 52)

The 24 assays were played on 24 November 2026. A line's score is the
logarithm of one plus its copies after six iterations. Each number is a
correlation over the 400 marked agents, with the range of 95% of 2,000
resamplings.

| world | the brain, in another place (r_brain) | the place, with another brain (r_place) | the same brain in the same place, played again (r_same) |
|---|---|---|---|
| s3058 at 20,000 (40,000 tokens, the largest stake wins) | 0.25 (0.15 to 0.35) | 0.58 (0.47 to 0.68) | 0.52 (0.39 to 0.63) |
| s4003 at 15,000 (10,000 tokens, the lottery) | 0.09 (−0.02 to 0.19) | 0.40 (0.31 to 0.50) | 0.36 (0.22 to 0.49) |

**The thesis is undecided.** In s3058 the brain's share is above nought and
smaller than the place's, as claimed. In s4003 its range includes nought, so
the claim does not hold in both worlds; nor is it refuted, which needed both
ranges to include nought.

- **In both worlds the place counts for far more than the brain.** A line
  does well mostly because of where it starts: its tokens, its links, its
  neighbours. The place's correlation is as large as that of the same brain
  in the same place played again, so most of what repeats is the place.
- **In s3058 a brain that does well in one place does better in another
  too.** A correlation of 0.25 means about 6% of the differences between
  lines travel with the brain. That is what selection can work on.
- **In s4003 almost nothing travels with the brain** over six iterations
  (0.09, about 1% of the differences). Read by which lines are still alive
  after three iterations it is 0.15 (0.05 to 0.24), just above nought.
  Selection on brains is weak there. This fits a world that plays the
  lottery, where every way of staking returns the same tokens on average.

So how much selection has to work with depends on the rules of the world,
and a way of varying brains can only help where the brain counts. The
numbers are made by `python3 research/efficient/C6/c6.py`.

## Where to go next

- [Why does evolution level off?](why-does-evolution-level-off.md): four
  causes that could stop evolution, of which unfaithful copies is one.
- [Does a smaller step restart evolution?](does-a-smaller-step-restart-evolution.md):
  Experiments 50 and 51, which play a world on with small mutations.
- [A hundred thousand iterations](a-hundred-thousand-iterations.md):
  Experiment 48, whose kept worlds are played here.
- [How fast should brains change?](how-fast-should-brains-change.md):
  Experiment 9 changed how often a brain mutates, not how much.

## How this was made

The assays are read by `python3 research/efficient/C2/c2.py` (its numbers in
`research/efficient/C2/c2.json`). The table of what a mutation does to a
brain's decisions is made by `python3 research/efficient/B1/b1.py`, which reads the
worlds Experiment 48 kept at iterations 5,000 and 15,000
(`research/efficient/probe.py` asks the brains). The two readings beside it
are `research/efficient/B2/b2.py` and `research/efficient/A1/a1.py`. The
plan they belong to is `research/efficient/plan.md`, written on 23 November
2026 when the author ruled that no experiment may take longer than a day.
