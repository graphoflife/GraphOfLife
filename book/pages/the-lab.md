# The lab, the engine and pilots

> [!question] Questions of this page
> - What plays a world, and what makes sure it can be played again exactly?
> - What is an experiment, and why is its thesis written before its runs?
> - What is a pilot, and how far can it be trusted?

## In short

The **engine** is the program that plays a [world](the-world.md). The
**lab** carries out an experiment's plan. It has the engine play every run
the plan asks for, several at a time. And it records how each was made, so
that it can be made again exactly. An **experiment** is that plan, with a
**thesis** written down before any of its runs exist. **Pilots** are worlds
made by hand, to see quickly whether an idea is worth an experiment.

## The engine

The engine is the simulation code, `GraphOfLifeSimple.py`. Given the
[settings](settings.md) and a [seed](random-numbers.md), it builds
[the starting ring](starting-ring.md). Then it plays
[one iteration](one-iteration.md) after another. The run writes the whole
world down after every phase ([What a run records](frames-and-stats.md)).
With the same settings, seed and libraries, and one thread for the matrix
library, it plays the same world, frame for frame.

## Runs and experiments

A **run** is one world played from its [founders](starting-ring.md) to its
end. That is as many iterations as asked for, or until it falls to 20 agents
or fewer ([When a world ends](extinction.md)). A run is named for what it is:
baseline, world size, what differs, seed. So `B1-5000-s001` is
[B1](b1-and-b2.md) at 5,000 tokens with seed 1. Two experiments that need
the same run share it, as Experiments 12 and 13 share theirs.

An **experiment** is a plan, a file `book/experiments/E…json`. It names a
baseline, the conditions that change it, the world's size, the seeds, how
far to run and what to keep. It also holds the **thesis**: a claim, why it
would be so, and what would confirm or refute it. The thesis is fixed before
the runs start. The page quotes it from the plan, so it cannot be reworded
once the results are in. Some plans ask for **trials** instead:
[contests](eras-and-contests.md) or
[invasions](eras-and-contests.md#the-invasion-test) played from worlds that
runs kept.

## The lab

The lab is `gol_lab.py`. It works out which runs a plan means, makes those
that do not exist yet, and advances them, each in a process of its own. It
can be paused and continued at any point, and it estimates a plan's time
from measured costs ([What a simulation costs](what-a-simulation-costs.md)).

Every run is made and advanced by a frozen copy of the engine, a
**snapshot**. The snapshot is taken from committed code when it is first
needed. So a later change to the code cannot mix into a run that has
started. Beside its frames, each run writes two files, `meta.json` and
`provenance.json`. They hold its settings and seed, its snapshot and the
commit it came from, and the versions of Python, numpy and networkx. The lab
keeps the matrix library to one thread. It refuses to continue a run under
other library versions, or on another processor.

## An example: Experiment 1

Experiment 1 tested the lab itself. Three seeds of B1 were each run four
ways for 30 iterations:
- straight through;
- stopped at iteration 10, and continued in a new process;
- killed without saving at 15, and resumed from its save at 10;
- and with the matrix library on every core.

The first three ways recorded the same frames, bit for bit. On every core the
frames were the same too. But in two seeds the [messages](messages.md) in
flight at the end differed in their last bits
([Is a run reproducible?](is-a-run-reproducible.md)). All twelve runs were
made on the snapshot `f39288e90b003df0`, from commit `ccba58f`.

## Pilot worlds

Most pages read **runs**: worlds the lab made for an experiment. There are
ten or thirty seeds of every condition, and a thesis written before the first
of them existed. Some pages, most of them about B2, read **pilots** instead. A pilot
is a world made by hand, to find out whether an idea is worth an experiment
at all. It is quick to set up, usually two seeds of each condition, and has
no thesis in advance.

A pilot shows what *can* happen, or gives a number worth testing. A
difference between pilots of two seeds each is a lead, not a finding. Pilots
of two seeds suggested that [brains](the-brain.md) that see break worlds.
Experiment 13 then ran ten seeds of each condition. Of the worlds whose
brains see and change only when copied, eight in ten died out. None of B1's
did
([Brains that see, and worlds that break](a-world-that-cannot-break.md#brains-that-see-and-worlds-that-break)).

### What makes one

A pilot is one process running one world (`research/oee/pilots/run.py`):

    python3 run.py OUTDIR --iters 1000 --ckpt 100 --set total_tokens=10000 seed=1 ... --proto ...

`--set` takes the engine's settings by name, exactly as a run's would
([Every setting](settings.md)). `--proto` switches on mechanics that exist
only in the pilots' prototype (`research/oee/pilots/proto.py`). That is a
copy of the engine with hooks: alliances, kin pooling, memory inputs, support
stakes, other contests. Each is described at the top of that file. A rule
that proves itself there can later be written into the engine, as the rules
of B2 were.

The prototype states some of B2's rules in its own words: `mor=all` for
mutate_on_replication, `keep_frac` for keep_home, `repair` for mend_on_death,
`safe_prune` for prune_needs. The two versions differ in small details, such
as which links a crowded newborn may still make. The switch
`--proto engine=true` makes the prototype play the engine's own phases
unchanged, and only keep its books. A check, `research/oee/pilots/equiv.py`,
makes sure that such a world stays identical to the engine's, iteration for
iteration. The pilots of
[The switches the world was built with](the-switches-the-world-was-built-with.md)
and of [Other contests](another-game.md) run this way.

### What a pilot records

- `config.json`: every setting and prototype switch.
- `series.jsonl`: one row per iteration. It holds the agents, the
  connections, ages, the number of founder [lineages](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor.") and [genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), and the
  [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of tokens. It also counts what happened in the iteration: births,
  nodes taken, and how many by kin. And it counts deaths, how many of them
  starved and how many were cut off, and [mutations](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.").

  Every tenth row adds the network's health
  ([Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md)).
  That is the [2-core](bridges.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them."), the worst cut by one agent and by one connection, the
  best-connected agent's share, links per agent and the [ball dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.").
  It also adds the sizes of groups: clones (neighbours with the same brain),
  kin (brains within four mutations), and sides where the rules have them.
- `ckpt_<iteration>.npz`: the whole world every `--ckpt` iterations, with
  every agent's founder lineage. The tournaments are played from these.

### Tournaments between its eras

`research/oee/pilots/tourney.py` plays the cross-time tournament
([Eras, contests and invasions](eras-and-contests.md)) on a pilot's
checkpoints. Each contest takes one era's world, its graph and tokens, and
gives every agent a brain drawn from one era or the other. It switches
mutation off and plays `--T` iterations. The later era's score is the share
of agents carrying its brains at the end. Each pair is played in both eras'
worlds, `--reps` times each, so with the usual `--reps 2` a pair is four
contests. `research/oee/pilots/elo.py` fits one rating per era.

The brains can be dealt in three ways:

- **coins**, the default: a coin per agent, as in the lab's contests, so the
  two eras are mixed agent by agent;
- **halves** (`--layout halves`): era B gets the half of the agents nearest,
  in steps, to an agent drawn at random, and era A the rest. Each era then
  starts as one territory. This is the fairer contest for anything that
  depends on neighbours of one kind standing together. An alliance is such a
  thing, and a coin per agent cuts through it
  ([Alliances](another-game.md#alliances));
- **turf** (`--layout turf`): halves, and the era whose world the contest is
  played in keeps its own brains where they are on its half. So its clones
  still sit next to clones, and its allies next to allies. Each era plays once
  at home in this way and once away, where its brains are drawn at random. It
  is the contest for what brains are as a group
  ([Other contests](another-game.md)).

Within an era a brain keeps its genotype's number in a contest, so clones are
still clones. That matters where stakes are pooled by genotype.

### How far to trust one

A pilot has two seeds where an experiment has ten, and it was not announced
with a thesis before it ran. So a pilot can show that something *can*
happen, or give a number worth testing. But a difference between two pilots
of two seeds each is a lead, not a finding
([How many seeds an experiment needs](bootstrap.md#how-many-seeds-an-experiment-needs)).
The pages built on pilots say so where it matters. Whatever they find that
the notebook relies on later is meant to be tested again as an experiment of
the lab.

Pilot worlds are kept in `GraphOfLifeRuns/.pilots`, outside the repository
like every run. The command lines that made them are in each page's recipes
and in `research/oee/pilots/LOG.md`.

## Starting an experiment

On a computer running `gol_server.py`, press ▶ on the experiment's page in
the Research Notebook tab; the lab says how long is left. In a terminal, the
same is `python3 gol_lab.py run E14`, and `python3 gol_lab.py analyse E14`
once the runs are done. The command `python3 gol_lab.py verify E14` plays a
few of the runs again, and checks them against what was recorded. And
`python3 book_figures.py` draws every figure again. It writes each, with its
caption and recipe, into the pages.

## Where it is used

Every page with a box *The runs behind this page* reads the lab's runs.
[Every simulation: what was run, and what was measured](every-simulation.md)
lists every experiment and pilot batch. Pages built on pilots say so, as
[The switches the world was built with](the-switches-the-world-was-built-with.md)
does.

## Where to go next

- [Is a run reproducible?](is-a-run-reproducible.md)
- [Every simulation: what was run, and what was measured](every-simulation.md)
- [B1 and B2: the two baselines](b1-and-b2.md)

<!-- turns -->
---

← [Every world side by side](every-world-side-by-side.md) · [Methods and statistics](../README.md#methods-and-statistics) · [How worlds are measured](how-worlds-are-measured.md) →

**Also about:** [How the simulation works](../README.md#how-the-simulation-works).

> [!info]- Pages that link here (22)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [Glossary](glossary.md) · [How this notebook is written](how-this-notebook-is-written.md) · [Is a run reproducible?](is-a-run-reproducible.md) · [Every setting](settings.md) · [Seeds and the random stream](random-numbers.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [The estate of the dead](estate.md) · [How different are the agents?](how-different-are-the-agents.md) · [Eras, contests and invasions: does a world keep getting better?](eras-and-contests.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Other contests: a lottery, and alliances](another-game.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [Links to strangers](links-to-strangers.md) · [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Eight switches at once, and how sixteen worlds can test them](eight-switches-at-once.md) · [The health bar](health-bar.md) · [What a simulation costs](what-a-simulation-costs.md) · [Every simulation: what was run, and what was measured](every-simulation.md) · [What a run records](frames-and-stats.md)
<!-- /turns -->
