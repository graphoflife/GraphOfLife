# How large a crowd a raider needs

> [!question] Questions of this chapter
> - How many raiders must start together for the raid to take hold?
> - Is it the world, or the place in it, that decides?

<!-- experiment E29 -->

## Why ask

The raider of [Chapter 51](51-strategies-written-by-hand.md) took half of
two evolved worlds when 5% of a world started as raiders. Planted alone it
died out every time, in all twenty tries. Its play pays only when three
things come at once:
- the raid itself;
- a password, so that raiders never raid each other;
- enough teammates who know the password.

Evolution makes new play one agent at a time, and the raider is a brain B2
can grow ([Chapter 94](94-what-a-brain-can-compute.md)). So how large a
crowd it needs says how deep the valley is between the play evolution has
and the raider's.

Twenty failures of a single raider do not prove it never takes hold. With
none in twenty, it could still take hold up to one time in seven. And
Chapter 51 planted every size in one place per world, which mixes up what
the world allows with where the patch happened to land.

This experiment plants patches of 1, 3, 10, 30 and 100 raiders, and of 5%
of the world, in ten places each in the two worlds where the raiders took
hold. Every planting has a control: the same patch given frozen copies of
one resident's brain. A patch of 5% is also planted in five places in
each of the other eight worlds.

## The thesis

<!-- thesis E29 -->
> [!quote] The thesis of Experiment 29, written down before any of its runs existed
> **The claim.** A raider needs a crowd. In worlds 2 and 3 of Experiment 16 at iteration 2,000, the share of plantings in which a patch of raiders takes hold rises with the patch: none of the 20 single raiders takes hold, and at least half of the 20 patches of 5% do. A planting takes hold when, after 300 iterations, its raiders are alive, more than the patch began with, and more than the control holds there: the same patch given frozen copies of one resident's brain.
>
> **Why it would be so.** In Chapter 51 a patch of 5% of raiders took 40% and 50% of world 2 and 62% of world 3, while a single raider died out in all 20 plantings in world 2, with peaks of up to 28 agents. Its play pays only with three things at once: the raid, the password and enough teammates (the green beard). Chapter 94's replay of the same patches lost one of four in worlds 2 and 3 to chance. Twenty plantings of each size, ten places in each world, bound how often a size could take hold and be missed: none in 20 still allows up to 14% at 95% confidence.
>
> **It holds if:** No single raider of 20 takes hold, at least 10 of the 20 patches of 5% do, and no smaller patch takes hold in three or more plantings in twenty more than a larger one.
>
> **It fails if:** A single raider takes hold, fewer than 10 of the 20 patches of 5% do, or a smaller patch takes hold in three or more plantings in twenty more than a larger one.
<!-- /thesis -->

## The trials

Each trial is [Chapter 51's invasion test](../notes/invasion-test.md),
played by the lab:
- The world as it stood at iteration 2,000 is loaded.
- A patch is chosen on the trial's place: an agent drawn at random, then
  its nearest neighbours, step by step, until the patch has its size.
- The patch is given the strategy, which never mutates.
- The world plays 300 iterations under its own rules, and the strategy's
  agents and tokens are counted after every iteration.

<!-- runs E29 -->
> [!info] The trials behind this chapter
> **320 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **raiders, worlds 2 and 3** — 120 trials: worlds `B2-10000-s002`, `B2-10000-s003`; at iteration 2,000; strategies raider; patches of 1, 3, 10, 30, 100 agents; 5% of the world; places 1 to 10, every 1.
> - **controls, worlds 2 and 3** — 120 trials: worlds `B2-10000-s002`, `B2-10000-s003`; at iteration 2,000; strategies resident; patches of 1, 3, 10, 30, 100 agents; 5% of the world; places 1 to 10, every 1.
> - **raiders, the other eight worlds** — 40 trials: worlds `B2-10000-s001`, `B2-10000-s004`, `B2-10000-s005`, `B2-10000-s006`, `B2-10000-s007`, `B2-10000-s008`, `B2-10000-s009`, `B2-10000-s010`; at iteration 2,000; strategies raider; patches of 5% of the world; places 1 to 5, every 1.
> - **controls, the other eight worlds** — 40 trials: worlds `B2-10000-s001`, `B2-10000-s004`, `B2-10000-s005`, `B2-10000-s006`, `B2-10000-s007`, `B2-10000-s008`, `B2-10000-s009`, `B2-10000-s010`; at iteration 2,000; strategies resident; patches of 5% of the world; places 1 to 5, every 1.
>
> To play these trials again: `python3 gol_lab.py run E29`, or ▶ in the Book tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

## What the raiders did

<!-- figure crowd/crowd -->
![How many plantings took hold, by the size of the patch](../figures/crowd/crowd.svg)

**How many plantings took hold, by the size of the patch.** Bars: of the 20 plantings of each size (ten places in each of worlds 2 and 3), how many raider patches took hold: alive after 300 iterations, more than they began with, and more than the control planted in the same place. Grey dots: how many of the controls, frozen copies of one resident's brain, were larger after 300 iterations than they began.

> [!example]- How to make this figure
> **Runs.** Experiment 29's 320 invasions (`python3 gol_lab.py run E29`): the ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, as their archives keep them at iteration 2,000. In worlds 2 and 3, patches of 1, 3, 10, 30 and 100 agents and of 5% of the world on places 1 to 10; in the other eight, 5% on places 1 to 5. Every patch once with the raider's rule and once as a control, frozen copies of one patch member's brain (`gol_strategies.plant`).
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of iteration 2,000; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch has its size.
> 2. Give the patch the raider ([Chapter 51](51-strategies-written-by-hand.md)'s rule), or the control; neither ever mutates.
> 3. Play 300 iterations under the world's own rules, counting the patch's agents after every iteration.
> 4. A planting takes hold when, after 300 iterations, its raiders are alive, more than the patch began with, and more than the control holds there.
>
> **To make it again:** `python3 book_figures.py crowd`.
<!-- /figure -->

**About a hundred raiders are the least that can start a raid.**
- **Up to 30 raiders:** none took hold in 20 plantings. The patches died
  out faster the smaller they were. Half had died out by iteration 51 for a
  single raider, 72 for three, 100 for ten and 148 for thirty.
- **A hundred raiders** took hold 4 times in 20.
- **5% of the world,** about 175 agents, took hold 12 times in 20.

Chapter 51's twenty single raiders in world 2 are ten of these plantings
again plus ten more, so 30 different single raiders have now died. If a
single raider could take hold, it would do so less than one time in ten,
or these 30 would very likely have shown it.

The controls, frozen copies of the residents' own brains, show no such
threshold:
- **Alive after 300 iterations:** a larger patch of residents lived more
  often, 0, 1, 5, 8, 13 and 16 times in 20, from one agent to 5%.
- **Larger than they began:** rarely, 0 to 7 times in 20.

A raid is either joined by enough raiders or it dies.

<!-- figure crowd/worlds -->
![A patch of 5% in every world](../figures/crowd/worlds.svg)

**A patch of 5% in every world.** Each dot is one planting of a patch of 5% of the world at iteration 2,000: left of each world, the raiders (filled where they took hold), right of it, the control planted in the same place. Worlds 2 and 3 have ten places, the others five.

> [!example]- How to make this figure
> **Runs.** Experiment 29's 320 invasions (`python3 gol_lab.py run E29`): the ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, as their archives keep them at iteration 2,000. In worlds 2 and 3, patches of 1, 3, 10, 30 and 100 agents and of 5% of the world on places 1 to 10; in the other eight, 5% on places 1 to 5. Every patch once with the raider's rule and once as a control, frozen copies of one patch member's brain (`gol_strategies.plant`).
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of iteration 2,000; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch has its size.
> 2. Give the patch the raider ([Chapter 51](51-strategies-written-by-hand.md)'s rule), or the control; neither ever mutates.
> 3. Play 300 iterations under the world's own rules, counting the patch's agents after every iteration.
> 4. A planting takes hold when, after 300 iterations, its raiders are alive, more than the patch began with, and more than the control holds there.
>
> **To make it again:** `python3 book_figures.py crowd`.
<!-- /figure -->

**The world decides as much as the crowd.**
- **World 2:** every one of ten patches of 5% took hold, each holding 40 to
  50% of the world after 300 iterations.
- **World 3:** two of ten took hold, with 58% and 62%. In the other eight
  places the raiders died out or kept a handful.
- **The other eight worlds:** four let a raid take hold in some places
  (world 6 in three of five, worlds 8, 9 and 10 in one each). Four let it
  take hold in none (worlds 1, 4, 5 and 7).
- **Over all ten worlds,** at the same five places, the raid took hold 13
  times in 50.

## The verdict

**The thesis holds.**
- No single raider took hold, 0 of 20.
- 12 of the 20 patches of 5% did, where at least 10 were asked for.
- No smaller patch took hold more often than a larger one.

## What it teaches

- **The valley is about a hundred agents wide.** The raider pays only in
  company: it needs about a hundred neighbours playing it at once before it
  takes hold. Mutation changes one brain at a time, and a hundred
  neighbours changing the same way at once does not happen. So evolution
  cannot reach the raider by mutation alone, though a B2 brain can play it
  ([Chapter 94](94-what-a-brain-can-compute.md)). Chapter 51 named this the
  green beard's valley; here it is measured.
- **A crowd can grow from one.** A brain's children carry it. A raider that
  bred before it died would grow its own crowd, if its first children lived
  long enough. The single raiders' lines grew to at most 28 agents
  ([Chapter 51](51-strategies-written-by-hand.md)) before dying out, about a
  quarter of the crowd they needed.
- **Where a raid lands matters.** In most worlds most places beat it. What
  makes a place raidable is the next question for the ladder of strategies:
  a defence evolution finds, or how wealth lies around the patch.


<!-- turns -->
---

← [Chapter 95 · Every era against three](95-every-era-against-three.md) · [Contents](../README.md) · [Chapter 97 · Does evolution close the door?](97-does-evolution-close-the-door.md) →
<!-- /turns -->
