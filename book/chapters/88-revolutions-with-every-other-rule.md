# Revolutions, with every other rule

> [!question] Questions of this chapter
> - In a game, smaller stakers may band together and take a node from its
>   largest staker — a revolution. What becomes of a world without them,
>   whatever its other rules?
> - Are there rules with which revolutions matter, where in B2 itself they do
>   not?

A **revolution** lets a coalition of smaller stakers take a node that the
largest single staker would otherwise win
([How a coalition takes a node](../notes/revolution.md),
[`allow_revolutions`](../notes/settings.md#allow_revolutions)). In B1
revolutions take most of the nodes that change hands, and without them B1
becomes a world of mayflies; in B2 they won about one node in three hundred a
game, and B2 with and without them were the same world
([Chapter 54](54-the-switches-the-world-was-built-with.md)). But B2 itself is
one world among many the book now plays. The author asked for revolutions in
combination (16 October 2026): batch W plays B2 without revolutions together
with each of the other switches, two seeds each, 1,000 iterations — every
world the fourth corner of a two-by-two whose other corners are B2, B2
without revolutions, and B2 with the other switch alone. Batch S's other ways
of deciding a node — a lottery, sides pooled by genotype or by family, a
sense of kin or of family — had only ever been played without revolutions;
for them batch W plays the missing corner, the same rule *with* revolutions.

## What turning revolutions off changes

<!-- figure norevcombos/changes -->
![Each rule, with no revolutions and without](../figures/norevcombos/changes.svg)

**Each rule, with no revolutions and without.** Each row is one other rule. Above: what adding no revolutions to a world of B2 with that rule changes (batch W), at iteration 1,000 — in orange, the share of the agents holding three tokens or fewer; in violet, the diversity score; a dot for each seed, 1 and 2. Below: the share of a contested world the brains of 1,000 of the world with both rules hold after 100 iterations against those of 500, the mean of two contests, a dot for each seed; the dashed line marked B2 is the median of the lab's ten B2 worlds, played the same way.

> [!example]- How to make this figure
> **Runs.** Batch W's pilot worlds (B2 at 10,000 tokens, no revolutions and one other rule, seeds 1 and 2, `GraphOfLifeRuns/.pilots/W/`) and their controls: the same rule alone, from the batch or lab experiment that made it.
>
> **Data.** From each world's checkpoint at iteration 1,000 (`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`) and the contests in `GraphOfLifeRuns/.pilots/batch_TUV_T100.json` and `rejudge_T100.json`.
>
> 1. Read each world as `research/atlas/profile.py` reads every world.
> 2. Subtract the world with the rule alone from the world with both, seed by seed.
> 3. Seat the brains of the two eras on alternate agents of each era's world, play 100 iterations, and count the agents the later brains hold.
>
> **To make it again:** `python3 book_figures.py norevcombos`.
<!-- /figure -->

**For most rules, nothing.** Over the 24 rules and two seeds, the median
change is +0.002 in the share of agents holding three tokens or fewer and
+0.008 in the diversity score, and the marks of the health bar hardly move.
Revolutions are rare in most of these worlds: a median of 0.24 per hundred
agents an iteration in the worlds that have them, and 3.7 in a hundred of the
nodes that change hands; nine worlds in ten stay below 0.8 and 7.4.

**With brains mixed at birth, revolutions keep a world small and rich.**
Mixed worlds are small, and in them revolutions are five to ten times as
common as in B2 — 1.0 and 1.4 per hundred agents an iteration, seven or eight
in a hundred of the nodes that change hands. Without them, the two mixed
worlds held 1,410 and 360 more agents at iteration 1,000, and 39 and 14 more
of every hundred held three tokens or fewer. Revolutions are as common in
other small worlds — where links or brain units hold a token, 0.75 to 1.35
per hundred agents — but there, without them, the worlds did not change one
way: with units that hold a token one world grew much richer (44 fewer of
every hundred at three tokens or fewer) and the other stayed as it was, and
with links that hold a token one grew a little richer and the other
poorer.

**Without random inputs, a little poorer and less diverse.** Six and seven
more of every hundred agents at three tokens or fewer, and diversity scores
lower by 0.11 and 0.12, in both seeds.

**Evolution goes on without revolutions.** In lifetime contests the brains
of 1,000 take more than half of a contested world from those of 500 in 42 of
the 48 worlds, with a median of 0.64 (B2's ten worlds: 0.68). The six at a
half or below are the worlds with the last token kept at home (0.45 and
0.49), and one seed each with no old age (0.48), with the estate to the
parents (0.48), with nodes by lottery (0.47) and with sides pooled by family
(0.46). The fastest: units that hold a token (0.79 and 0.89), slow brain
change and brains that grow (0.72 and 0.85 each), half kept at home (0.72 and
0.84).

## What it teaches

- **In B2's kind of world, revolutions are a detail** — with one exception:
  they are what keeps brains mixed at birth from filling their world. That
  fits [Chapter 86](86-eight-switches-at-once.md), where no revolutions moved
  the share of poor agents by only 0.03 on average over sixteen worlds.
- **Evolution does not need them**: without revolutions the later brains
  win in 42 of 48 worlds, about as often as in B2.
- **How often revolutions happen is not how much they matter.** In one world
  where every unused link lapses they won 96 of every hundred nodes that
  changed hands, and without them the world changed hardly at all; in the
  mixed worlds they won seven or eight in a hundred, and without them the
  worlds grew poor.

**Related notes:** [How a coalition takes a node](../notes/revolution.md) ·
[Every setting](../notes/settings.md#allow_revolutions) ·
[Chapter 54 · The switches the world was built with](54-the-switches-the-world-was-built-with.md) ·
[Chapter 82 · Brains mixed at birth, with every other rule](82-mixing-with-every-other-rule.md) ·
[Chapter 86 · Eight switches at once](86-eight-switches-at-once.md)

<!-- turns -->
---

← [Chapter 86 · Eight switches at once](86-eight-switches-at-once.md) · [Contents](../README.md) · [Chapter 60 · Worlds of 100,000 tokens](60-worlds-of-100000-tokens.md) →
<!-- /turns -->
