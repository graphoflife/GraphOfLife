# Does evolution close the door?

> [!question] Questions of this chapter
> - When a world has evolved for longer, can a raid that once took half of
>   it still take hold?
> - If evolution closes one door, does another open?

<!-- experiment E30 -->

## Why ask

A raid written by hand took half of some evolved B2 worlds at iteration
2,000 ([Chapter 51](51-strategies-written-by-hand.md)). Those worlds had
never met a raider. Do worlds that evolve longer get harder to raid?

Three answers are possible, and each says something different about
open-ended evolution:
- **The door stays open.** Later worlds are raided as easily as early ones.
  Evolution does not find the defence any more than it found the raid, so
  better play lies across a valley it does not cross. The next step is to
  ask what would let it cross: a crowd, or a rule.
- **The door closes, and others open.** Later worlds resist this raid but
  lose to others. That is an arms race, which may go round in circles or
  climb.
- **The door closes, and nothing opens.** Later worlds resist every raid
  tried: a top, at least for these brains.

The way evolution's own contests fade
([Chapter 95](95-every-era-against-three.md)) says which of these to
expect. So does how much a single agent could gain by its best reply,
which stays about one node a game at every era from 2,500 to 50,000 in
every long world.

This experiment plants raiders, 5% of a world in three places each, in
B2's five worlds played on to 10,000 iterations, at iterations 500,
2,000, 5,000 and 10,000. Each planting has its control.

## The thesis

<!-- thesis E30 -->
> [!quote] The thesis of Experiment 30, written down before any of its runs existed
> **The claim.** Evolution in B2 does not learn to resist the raider. In B2's worlds played on to 10,000 iterations, patches of 5% of raiders take hold as often at iterations 5,000 and 10,000 as at 500 and 2,000: a one-sided Fisher exact test finds no fewer late than early at the 5% level. A planting takes hold as in Experiment 29: after 300 iterations its raiders are alive, more than the patch began with, and more than the control patch of residents holds there.
>
> **Why it would be so.** In every long world the book has, an agent's best reply to the rest would gain about one node a game, as much at 50,000 iterations as at 2,500 (Chapter 95): evolution does not close in on any equilibrium. The raider exploits what B2's brains do not do, and nothing in a world that has never met a raider selects against one. Later brains beat earlier ones by less and less, which a world learning to defend itself would not show.
>
> **It holds if:** The plantings that take hold at 5,000 and 10,000 together (30, in five worlds at three places each) are not fewer than at 500 and 2,000 together by a one-sided Fisher exact test at the 5% level.
>
> **It fails if:** Fewer take hold late than early by that test: evolution closed the door. The chapter then asks what opened instead (other raids that later brains lose to) or whether nothing did.
<!-- /thesis -->

## The trials

The worlds are B2's own: Experiment 16's worlds, played on to 10,000
iterations as pilots of the engine's own rules. Each trial plays from a
pilot's checkpoint, which holds a whole world, as a run's archive does.
The test is the invasion test of [Chapter 96](96-how-large-a-crowd-a-raider-needs.md).

<!-- runs E30 -->
> [!info] The trials behind this chapter
> **120 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **raiders** — 60 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 500, 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 60 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 500, 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E30`, or ▶ in the Book tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- turns -->
---

← [Chapter 96 · How large a crowd a raider needs](96-how-large-a-crowd-a-raider-needs.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
