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

<!-- turns -->
---

← [Chapter 97 · Does evolution close the door?](97-does-evolution-close-the-door.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
