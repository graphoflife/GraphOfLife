# The lab, the engine and pilots

## In short

The **engine** is the program that plays a [world](the-world.md). The
**lab** carries out an
experiment's plan: it has the engine play every run the plan asks for,
several at a time, and records how each was made, so that it can be made
again exactly. An **experiment** is that plan, with a **thesis** written
down before any of its runs exist. **Pilots** are worlds made by hand, to
see quickly whether an idea is worth an experiment.

## The engine

The engine is the simulation code, `GraphOfLifeSimple.py`. Given the
[settings](settings.md) and a [seed](random-numbers.md), it builds
[the starting ring](starting-ring.md) and plays
[one iteration](one-iteration.md) after another, and the run writes the
whole world down after every phase ([What a run records](frames-and-stats.md)).
With the same settings, seed and libraries, and one thread for the matrix
library, it plays the same world, frame for frame.

## Runs and experiments

A **run** is one world played from its [founders](starting-ring.md) to its
end: as many
iterations as asked for, or until it falls to 20 agents or fewer
([When a world ends](extinction.md)). It is named for what it is —
baseline, world size, what differs, seed — so `B1-5000-s001` is
[B1](b1-and-b2.md) at 5,000 tokens with seed 1, and two experiments that
need the same run share it, as Experiments 12 and 13 share theirs.

An **experiment** is a plan, a file `book/experiments/E…json`. It names a
baseline, the conditions that change it, the world's size, the seeds, how
far to run and what to keep. It also holds the **thesis**: a claim, why it
would be so, and what would confirm or refute it. The thesis is fixed before
the runs start, and the page quotes it from the plan, so it cannot be
reworded once the results are in. Some plans ask for **trials** instead:
[contests](eras-and-contests.md) or [invasions](invasion-test.md) played
from worlds that runs kept.

## The lab

The lab is `gol_lab.py`. It works out which runs a plan means, makes those
that do not exist yet, and advances them, each in a process of its own. It
can be paused and continued at any point, and it estimates a plan's time
from measured costs ([What a simulation costs](what-a-simulation-costs.md)).

Every run is made and advanced by a frozen copy of the engine, a
**snapshot**, taken from committed code when it was first needed, so a later
change to the code cannot mix into a run that has started. Beside its
frames, each run writes `meta.json` and `provenance.json`: its settings and
seed, its snapshot and the commit it came from, and the versions of Python,
numpy and networkx. The lab keeps the matrix library to one thread, and
refuses to continue a run under other library versions or on another
processor.

## An example: Experiment 1

Experiment 1 tested the lab itself. Three seeds of B1 were each run four
ways for 30 iterations: straight through; stopped at iteration 10 and
continued in a new process; killed without saving at 15 and resumed from
its save at 10; and with the matrix library on every core. The first three
ways recorded the same frames, bit for bit. On every core the frames were
the same too, but in two seeds the [messages](messages.md) in flight at the
end differed in their last bits ([Is a run reproducible?](is-a-run-reproducible.md)).
All twelve runs were made on the snapshot `f39288e90b003df0`, from commit
`ccba58f`.

## Pilots

A pilot is one world run by hand with `research/oee/pilots/run.py`: quick to
set up, usually two seeds of each condition, no thesis in advance
([Pilot worlds](pilot-worlds.md)). It can try rules the engine does not have
yet, or, with `--proto engine=true`, play the engine's own game.

A pilot shows what *can* happen, or gives a number worth testing; a
difference between pilots of two seeds each is a lead, not a finding. Pilots
of two seeds suggested that [brains](the-brain.md) that see break worlds; Experiment 13 then
ran ten seeds of each condition, and eight of the ten worlds whose brains
see and change only when copied died out, against none of B1's
([Brains that see, and worlds that break](brains-that-see-and-worlds-that-break.md)).

## Starting an experiment

On a computer running `gol_server.py`, press ▶ on the experiment's page in
the Research Notebook tab; the lab says how long is left. Or type
`python3 gol_lab.py run E14` in a terminal, and
`python3 gol_lab.py analyse E14` once the runs are done.
`python3 gol_lab.py verify E14` plays a few of the runs again and checks them
against what was recorded, and `python3 book_figures.py` draws every figure
again and writes it, with its caption and recipe, into the pages.

## Where it is used

Every page with a box *The runs behind this page* reads the lab's runs;
[Every simulation: what was run, and what was measured](every-simulation.md)
lists every experiment and pilot batch. Pages built on pilots say so, as
[The switches the world was built with](the-switches-the-world-was-built-with.md)
does.

## Where to go next

- [Is a run reproducible?](is-a-run-reproducible.md)
- [Pilot worlds](pilot-worlds.md)
- [B1 and B2: the two baselines](b1-and-b2.md)

<!-- turns -->
---

← [Every world side by side](every-world-side-by-side.md) · [Methods and statistics](../README.md#methods-and-statistics) · [How worlds are measured](how-worlds-are-measured.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works).

> [!info]- Pages that link here (5)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [B1 and B2: the two baselines](b1-and-b2.md)
<!-- /turns -->
