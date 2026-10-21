# A crew of raiders

> [!question] Questions of this chapter
> - Where the world is too poor for any lone raider, can raiders that pool
>   their stakes take hold?
> - Does the game itself reward groups, and the poorer the world, the more?

<!-- experiment E33 -->

## Why ask

The raider of [Chapter 51](51-strategies-written-by-hand.md) took half of
some evolved worlds. In B2's worlds played on, its door then closed
([Chapter 97](97-does-evolution-close-the-door.md)).
- **Early:** at iteration 500 every patch of 5% took hold.
- **Late:** at 5,000 and 10,000 none did, in 30 plantings.

No defence closed the door. The raider stakes only where it is sure to win,
and against a neighbour of one token that needs three tokens. By 5,000 the
patches it started from held two tokens an agent.

Stakes from several agents do not simply add up. A node goes to the largest
single stake, the holder's own counting one and a half times, unless
revolutionaries together outweigh it
([How a coalition takes a node](../notes/revolution.md)).

The **crew** raider uses that:
- **Its rule:** it stakes all it can on the weakest stranger next door that
  is no richer than itself, and marks every token a revolution token.
- **What it does:** two crew raiders of two tokens, each staking one, take a
  stranger of one token that neither could take alone.
- **How they join:** raiders that each see the same stranger as the weakest
  next door become one mob without a word passing.

If a world of many small agents shuts out lone raiders but not crews, the
game itself rewards groups, and the poorer the world, the more. That would
be a reason for larger and larger groups that comes from the rules, not
from anything added to them.

This experiment plants crews of 5% where the lone raiders were planted in
Chapter 97, in the same places, beside the lone raiders' and the controls'
plantings, which it shares with that experiment.

## The thesis

<!-- thesis E33 -->
> [!quote] The thesis of Experiment 33, written down before any of its runs existed
> **The claim.** Raiders that pool their stakes take hold where lone raiders cannot. In B2's worlds played on, at iterations 5,000 and 10,000, where patches of 5% of lone raiders took hold in none of 30 plantings, patches of 5% of crew raiders — raiders that stake on the weakest stranger next door no richer than themselves, every token of it a revolution token, so that raiders bordering the same stranger become one mob — take hold in at least 5 of the same 30 plantings.
>
> **Why it would be so.** In Experiment 30 the lone raider's door closed because the patches it started from grew poor: it needs three tokens to take even a neighbour of one, and by iteration 5,000 a patch held two tokens an agent. In that game a node goes to the largest single stake, the holder's own counting one and a half times, unless revolutionaries together outweigh it. Two raiders of two tokens, each staking one as a revolution token, take a stranger of one token that neither could take alone. Raiders that each see the same stranger as the weakest next door join without any message. A world of many small agents shuts out lone raiders, not crews: if that is so, the poorer the world, the more a group is worth.
>
> **It holds if:** At least 5 of the 30 crew plantings at 5,000 and 10,000 take hold: alive after 300 iterations, more than the patch began with, and more than the control planted in the same place.
>
> **It fails if:** Four or fewer take hold: pooling stakes does not open the door, and the chapter says what the crews met instead.
<!-- /thesis -->

## The trials

<!-- runs E33 -->
> [!info] The trials behind this chapter
> **135 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **crews** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies crew; patches of 5% of the world; places 1, 2, 3.
> - **lone raiders** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E33`, or ▶ in the Book tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

## What the crews did

<!-- figure crews/hold -->
![Crews and lone raiders, planted in the same places](../figures/crews/hold.svg)

**Crews and lone raiders, planted in the same places.** Left: of the fifteen plantings at each era (five worlds, three places), how many took hold, lone raiders (violet) and crews (orange). Right: each place once, by the iteration the lone raiders' last agent died in and the crew's (300: still alive at the end). Below the dashed line the crew died first.

> [!example]- How to make this figure
> **Runs.** Experiment 33's 135 invasions (`python3 gol_lab.py run E33`): B2's worlds of Experiment 16 played on to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the crew's rule, the lone raider's ([Chapter 51](51-strategies-written-by-hand.md)'s) or frozen copies of one patch member's brain (the control); the lone raiders' and the controls' plantings are Experiment 30's own.
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the patch's strategy and its tokens after every iteration.
> 3. A planting takes hold when its strategy is then alive, holds more agents than the patch began with, and more than the control planted in the same place.
>
> **To make it again:** `python3 book_figures.py crews`.
<!-- /figure -->

**No crew took hold, at any era.**
- **At 2,000,** where four of the fifteen lone raiders' plantings took hold
  (worlds 2 and 3, as in
  [Chapter 97](97-does-evolution-close-the-door.md)), no crew did, not even
  in those places.
- **At 5,000 and 10,000,** none of thirty did, as for the lone raiders.

**And the crews died first.** In 42 of the 45 places the crew was gone
before the lone raiders planted there: in 15 of 15 at 2,000, 14 at 5,000,
13 at 10,000.

| planted at | the crew's last agent died, median | the lone raiders' | the control's |
|---|---|---|---|
| 2,000 | 107 iterations later | 206 | still alive at 300 |
| 5,000 | 100 | 151 | still alive at 300 |
| 10,000 | 108 | 155 | 297 |

<!-- figure crews/wealth -->
![What became of the patches planted at 5,000 and 10,000](../figures/crews/wealth.svg)

**What became of the patches planted at 5,000 and 10,000.** The median over the thirty plantings at 5,000 and 10,000 of each kind: crews (orange), lone raiders (violet), controls (grey). Left: the patch's strategy's agents as a share of the patch planted (0 once it has died out). Right: their tokens per agent, over the plantings still alive.

> [!example]- How to make this figure
> **Runs.** Experiment 33's 135 invasions (`python3 gol_lab.py run E33`): B2's worlds of Experiment 16 played on to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the crew's rule, the lone raider's ([Chapter 51](51-strategies-written-by-hand.md)'s) or frozen copies of one patch member's brain (the control); the lone raiders' and the controls' plantings are Experiment 30's own.
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the patch's strategy and its tokens after every iteration.
>
> **To make it again:** `python3 book_figures.py crews`.
<!-- /figure -->

**The crews paid their neighbours.** Ten iterations after planting, at
5,000 and 10,000:

| | tokens per agent, median |
|---|---|
| crews | 1.33 |
| lone raiders | 1.65 |
| controls | 1.85 |

Every patch started at about 2.0.

**Neither kind of raider had a child.** Both rules wait for 30 tokens. Of
the 7,587 agents the 45 patches were planted on, 4 held that many: 3 at
2,000, 1 at 5,000, none at 10,000. So a patch lived only as long as the
nodes it held and the nodes it took. The crews took few, and lasted about
as long as the youngest agent they were planted on: B2's agents die at
age 100.

## What a crew meets

<!-- figure crews/mobs -->
![A crew's first game, as planted](../figures/crews/mobs.svg)

**A crew's first game, as planted.** Every stranger a crew raider would stake on in its first game, over the fifteen plantings of each era. Left: how many raiders of the crew chose it. Right: whether the raiders that chose it outweigh it even if it keeps everything at home (sure to win), only if it keeps no more than it must (might win), or not even then (sure to lose: their tokens go to the stranger).

> [!example]- How to make this figure
> **Runs.** No world is played: the worlds and places of Experiment 33's crews, read as they were planted — B2's worlds played on (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`) at iterations 2,000, 5,000 and 10,000, places 1 to 3.
>
> **Data.** From the worlds' checkpoints at each era, read by `research/oee/strategies/mobs.py`.
>
> 1. Draw each patch as the planting draws it.
> 2. For every patch agent with a stranger next door: the weakest such stranger (the first, where several are as weak); if the agent holds at least as much and has more than its home share, it stakes all but that share there.
> 3. Raiders that chose the same stranger are one mob. Weigh the mob against the stranger's stake at home, counted one and a half times: everything it holds (sure to win), or only what B2 makes it keep, 30% rounded up and at least one token (might win). What other neighbours stake on it is left out.
>
> **To make it again:** `python3 book_figures.py crews`.
<!-- /figure -->

The worlds were read as the crews were planted, before any game was
played. Three things stood in the crews' way.

- **Few could stake at all.** B2 keeps at least one token of every agent at
  home, and seven agents in ten of a patch planted at 5,000 or 10,000 held
  a single token: 1,844 of 2,568, and 1,848 of 2,520.
- **Mobs hardly formed.** A stranger next to a patch borders 1.6 of its
  agents on average.

  | planted at | strangers chosen by one raider | by two | by three or more |
  |---|---|---|---|
  | 5,000 | 454 | 37 | 6 |
  | 10,000 | 361 | 32 | 9 |

- **A stake that is not pooled is a gift.** At 5,000, 151 of the 497
  strangers chosen could not be taken by the raiders that chose them, even
  if the stranger kept at home only what it must. Their tokens go to the
  stranger.
  - At 5,000 and 10,000, 266 of the 280 raiders behind such stakes were
    agents of two tokens staking one on a stranger of one.
  - That stranger's single token, at home, counts one and a half.
  - Two such raiders would take it. One alone hands it a token, and is left
    with one, too few to stake.

The crew's rule sent out more raiders than the lone raider's: 549 against
353 at 5,000. Nearly all of the difference (310 of the 358 extra raiders at
5,000 and 10,000) was such a raider of two tokens against a stranger of
one.

## The verdict

**The thesis fails.** No crew took hold in the thirty plantings at 5,000
and 10,000; the thesis asked for five. Pooling stakes did not open the door
that closed on the lone raiders. The crews fared worse than the lone
raiders, and the poorer the world, the less a mob was worth, not the more:
- the fewer raiders could stake at all;
- the fewer bordered the same stranger;
- and a stake that found no partner went to the stranger.

## What it teaches

- **The game rewards a group only where a group can gather.** On B2's
  network a stranger next to a patch touches one or two of its agents. With
  each raider choosing alone, about one stranger in ten drew a mob. A crew
  that pools its stakes would have to choose its target together, and that
  needs news about a third agent, which no message of a B2 brain carries
  ([Chapter 94](94-what-a-brain-can-compute.md)).
- **A raider written by hand is barren in a poor world.** Its rule asks for
  30 tokens before a child, and almost no agent of a late world holds that
  many. [Chapter 97](97-does-evolution-close-the-door.md) saw the same in
  its replays. A planted strategy that cannot breed lives only by taking
  nodes, each of which dies of age within a hundred iterations. So the
  closed door of Chapter 97 says as much about the raider's own rule as
  about the world.
- **The next rungs must breed as the world's own agents do.** That means a
  child as soon as an agent can give it the least child the world allows,
  keeping its home share. Without it, no strategy planted in a late world
  can take hold, however well it raids. With it, the lone raider and the
  crew can be planted again where they failed.

<!-- turns -->
---

← [Chapter 97 · Does evolution close the door?](97-does-evolution-close-the-door.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
