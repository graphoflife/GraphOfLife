# Pilot worlds

Most pages read **runs**: worlds the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") made for an experiment, ten or
thirty [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") of every condition, with a thesis written before the first of
them existed. Some pages, most of them about [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death."), read **pilots** instead, worlds made by
hand to find out whether an idea is worth an experiment at all. This note says
what a pilot is, what it records and how far it can be trusted.

## What makes one

A pilot is one process running one world (`research/oee/pilots/run.py`):

    python3 run.py OUTDIR --iters 1000 --ckpt 100 --set total_tokens=10000 seed=1 ... --proto ...

`--set` takes the engine's settings by name, exactly as a run's would
([Every setting](settings.md)). `--proto` switches on mechanics that exist
only in the pilots' prototype (`research/oee/pilots/proto.py`), a copy of the
engine with hooks: alliances, kin pooling, memory inputs, support stakes, other
contests. Each is described at the top of that file. A rule that proves itself
there can later be written into the engine, as the rules of B2 were.

The prototype states some of B2's rules in its own words: `mor=all` for
mutate_on_replication, `keep_frac` for keep_home, `repair` for mend_on_death,
`safe_prune` for prune_needs. The two versions differ in small details, such
as which links a crowded newborn may still make. With `--proto engine=true`
the prototype instead plays the engine's own phases unchanged and only keeps
its books; `research/oee/pilots/equiv.py` checks that such a world stays
identical to the engine's, [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.") for iteration. The pilots of
[The switches the world was built with](the-switches-the-world-was-built-with.md) and
[Another game](another-game.md) run this way.

## What it records

- `config.json`: every setting and prototype switch.
- `series.jsonl`: one row per iteration. It holds the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), the [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."),
  ages, the number of [founder](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.") [lineages](genotype.md#families "A line of descent: the agents whose genotypes go back to one ancestor.") and [genotypes](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), and the [Gini](gini-coefficient.md "How unequally tokens are held: 0 if every agent holds as many, near 1 if one holds nearly all.") of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.").
  It also counts what happened in the iteration: births, nodes taken (and how
  many by kin), deaths (and of them, how many starved and how many were cut
  off), [mutations](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.").
  Every tenth row adds the network's health: the [2-core](bridges.md#core-trees-and-leaves "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them."), the worst cut by one
  agent and by one connection, the best-connected agent's share, links per
  agent and the [ball dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.")
  ([How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)).
  It also adds the sizes of groups: clones ([neighbours](the-world.md#the-network "An agent joined to another by a connection.") with the same [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.")),
  kin (brains within four mutations), and sides where the rules have them.
- `ckpt_<iteration>.npz`: the whole world every `--ckpt` iterations, with
  every agent's founder lineage. The tournaments are played from these.

## Tournaments between its eras

`research/oee/pilots/tourney.py` plays the [cross-time tournament](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") of
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) on a
pilot's checkpoints. Each contest takes one era's world, its graph and tokens,
and gives every agent a brain drawn from one era or the other. It switches
mutation off and plays `--T` iterations; the later era's score is the share of
agents carrying its brains at the end. Each pair is played in both eras'
worlds, `--reps` times each, so with the usual `--reps 2` a pair is four
contests. `research/oee/pilots/elo.py` fits one rating per era.

The brains can be dealt in two ways:

- **coins** (the default): a coin per agent, as in [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md), so the two eras
  are mixed agent by agent;
- **halves** (`--layout halves`): era B gets the half of the agents nearest,
  in steps, to an agent drawn at random, and era A the rest. Each era then
  starts as one territory. This is the fairer contest for anything that
  depends on neighbours of one kind standing together, such as an alliance,
  which a coin per agent cuts through
  ([Alliances](another-game.md#alliances));
- **turf** (`--layout turf`): halves, and the era whose world the contest is
  played in keeps its own brains where they are on its half, so its clones
  still sit next to clones and its allies next to allies. Each era plays once
  at home in this way and once away, where its brains are drawn at random.
  It is the contest for what brains are as a group
  ([Another game](another-game.md)).

Within an era a brain keeps its genotype's number in a contest, so clones
are still clones, which matters where stakes are pooled by genotype.

## How far to trust one

A pilot has two seeds where an experiment has ten, and it was not announced
with a thesis before it ran. So a pilot can show that something *can* happen,
or give a number worth testing, but a difference between two pilots of two
seeds each is a lead, not a finding
([How many seeds an experiment needs](seeds-needed.md)). The pages built
on pilots say so where it matters. Whatever they find that the notebook relies on
later is meant to be tested again as an experiment of the lab.

Pilot worlds are kept in `GraphOfLifeRuns/.pilots`, outside the repository
like every run. The command lines that made them are in each page's
recipes and in `research/oee/pilots/LOG.md`.

<!-- turns -->
---

← [What the viewer can colour by](viewer-colours.md) · [Methods and statistics](../README.md#methods-and-statistics) · [Many worlds at once: the median, bands, the settled life and the spread](median-and-quantiles.md) →

> [!info]- Pages that link here (11)
> [Glossary](glossary.md) · [Every setting](settings.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [The estate of the dead](estate.md) · [How different are the agents?](how-different-are-the-agents.md) · [Eras and contests: does a world keep getting better?](eras-and-contests.md) · [Does evolution keep going?](how-to-tell-a-world-that-keeps-evolving.md) · [Other contests: a lottery, and alliances](another-game.md) · [Revolutions, handover and gifts: the switches the world was built with](the-switches-the-world-was-built-with.md) · [The health bar](health-bar.md) · [The lab, the engine and pilots](the-lab.md)
<!-- /turns -->
