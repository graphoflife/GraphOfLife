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

Experiment 49 asks whether that also shows in how well a mutant does. It
marks mutants and unchanged agents in a kept world and counts their copies
30 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") later. It has not been played yet.

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

## Where to go next

- [Why does evolution level off?](why-does-evolution-level-off.md): four
  causes that could stop evolution, of which unfaithful copies is one.
- [A hundred thousand iterations](a-hundred-thousand-iterations.md):
  Experiment 48, whose kept worlds are played here.
- [How fast should brains change?](how-fast-should-brains-change.md):
  Experiment 9 changed how often a brain mutates, not how much.

## How this was made

The table is made by `python3 research/efficient/B1/b1.py`, which reads the
worlds Experiment 48 kept at iterations 5,000 and 15,000
(`research/efficient/probe.py` asks the brains). The two readings beside it
are `research/efficient/B2/b2.py` and `research/efficient/A1/a1.py`. The
plan they belong to is `research/efficient/plan.md`, written on 23 November
2026 when the author ruled that no experiment may take longer than a day.
