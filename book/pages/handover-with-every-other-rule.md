# Handover, with every other rule

> [!question] Questions of this chapter
> - A parent may pass some of its own links to its newborn — B1's rule,
>   switched off in B2. What does it do to a world, whatever its other rules?
> - Does it cost the same health with every rule, and does it buy anything?

With **handover**
([`allow_handover`](settings.md#allow_handover)) a parent moves some
of its links to its newborn, and a parent that hands over all but one hangs
on that one. B1 has it; B2 was built without it, because in
[Chapter 48](a-world-that-cannot-break.md) handover left 40% of a world's
agents in trees outside its core, and one link's loss could cut off 4% of it.
The author asked for handover in combination (16 October 2026): batch X plays
B2 with handover together with each of 23 other rules — the switches of
[Chapter 88](revolutions-with-every-other-rule.md) and batch S's other ways
of deciding a node — two seeds each, 1,000 iterations, every world the fourth
corner of a two-by-two whose other corners are B2, B2 with handover alone,
and B2 with the other rule alone.

## What handover changes

<!-- figure handovercombos/changes -->
![Each rule, with handover and without](../figures/handovercombos/changes.svg)

**Each rule, with handover and without.** Each row is one other rule. First: what adding handover to a world of B2 with that rule changes (batch X), at iteration 1,000 — in orange, the share of the agents holding three tokens or fewer; in violet, the diversity score; a dot for each seed, 1 and 2. Then: what it changes in the marks of the health bar the world meets. Last: the share of a contested world the brains of 1,000 of the world with both rules hold after 100 iterations against those of 500, the mean of two contests, a dot for each seed; the dashed line marked B2 is the median of the lab's ten B2 worlds, played the same way.

> [!example]- How to make this figure
> **Runs.** Batch X's pilot worlds (B2 at 10,000 tokens, handover and one other rule, seeds 1 and 2, `GraphOfLifeRuns/.pilots/X/`) and their controls: the same rule alone, from the batch or lab experiment that made it.
>
> **Data.** From each world's checkpoint at iteration 1,000 (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`) and the contests in `GraphOfLifeRuns/.pilots/batch_TUV_T100.json` and `rejudge_T100.json`.
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. Subtract the world with the rule alone from the world with both, seed by seed.
> 3. Seat the brains of the two eras on alternate agents of each era's world, play 100 iterations, and count the agents the later brains hold.
>
> **To make it again:** `python3 book_figures.py handovercombos`.
<!-- /figure -->

**It costs health with nearly every rule** — two marks of the health bar at
the median, up to four; in five of the 46 worlds, none. The marks it costs are
the network's: with handover, 38 of the 46 worlds have less than 0.8 of their
agents in the 2-core, 42 have a link whose loss would cut off more than 1% of
the world, and 36 an agent whose loss would cut off more than 2%; without it,
none, two and three do. A parent that gives its links away thins the network
into trees — 1.8 links fewer per agent at the median — and trees hang on
single links.

**It makes worlds a little poorer, and a little more diverse.** The share
of agents holding three tokens or fewer rises by 0.04 at the median, and the
diversity score by 0.08.

**It cancels the riches of mixing.** B2 with brains mixed at birth holds 28%
and 40% of its agents at three tokens or fewer at iteration 1,000; with
handover as well, 48 and 32 more of every hundred are that poor — handover is one of the rules
[Chapter 82](mixing-with-every-other-rule.md) found to cancel mixing's
riches. It costs riches too where links or brain units hold a token (14 to 22
more of every hundred poor).

**It unfolds a world a little.** A parent that passes links on does not
gather them till it dies, and the dying leave rings of 3.7 groups fewer
([Chapter 87](what-folds-a-world.md)): a typical agent is further from
everyone than in B2 — the typical distance over the random twin's rises by
0.24 at the median, in 41 of 46 worlds. The room is bought with the trees and
bridges above: room and resilience pull apart, as Chapter 87 found.

**Evolution goes on, more slowly.** In lifetime contests the brains of 1,000
take more than half of a contested world from those of 500 in 35 of the 46
worlds, with a median of 0.57 — below B2's 0.68 and B2 without revolutions'
0.64, and as with handover alone (0.60 and 0.60). It is slowest where every
unused link lapses (0.47 and 0.47), with the estate to the newborns (0.48 and
0.49), and in one seed each with a faster change of brains (0.40), units that
hold a token (0.45) and mixing (0.47); fastest with half kept at home (0.65
and 0.75) and brains that grow (0.60 and 0.72).

## What it teaches

- **Handover trades resilience for room.** Whatever else the rules are, it
  thins the network into trees hanging on single links, and with fewer links
  to gather the dying fold the world less.
- **It is no friend of riches**: it cancels mixing's, and those of links and
  units that hold a token.
- **Evolution slows with it**, to a median of 0.57 against B2's 0.68.

**Related notes:** [Every setting](settings.md#allow_handover) ·
[Chapter 48 · A world that cannot break](a-world-that-cannot-break.md) ·
[Chapter 82 · Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) ·
[Chapter 86 · Eight switches at once](eight-switches-at-once.md) ·
[Chapter 87 · What folds a world](what-folds-a-world.md) ·
[Chapter 88 · Revolutions, with every other rule](revolutions-with-every-other-rule.md)

<!-- turns -->
---

← [Chapter 88 · Revolutions, with every other rule](88-revolutions-with-every-other-rule.md) · [Contents](../README.md) · [Chapter 90 · Gifts, with every other rule](90-gifts-with-every-other-rule.md) →
<!-- /turns -->
