# Raiders that breed

> [!question] Questions of this chapter
> - The raiders written by hand wait for 30 tokens before a child, and in a
>   late world almost nobody holds that many. Does a raider that breeds as
>   the world's own agents do take hold where the barren one could not?
> - Is it the raid that does it, or the breeding alone?

<!-- experiment E34 -->

## Why ask

The raider's door closed in B2's late worlds
([Chapter 97](97-does-evolution-close-the-door.md)), and a crew of raiders
did not open it ([Chapter 102](102-a-crew-of-raiders.md)). Both chapters
found the same thing beside the raid itself: the raiders had no children.
- **Their rule:** a child of a third, once the raider holds 30 tokens, as
  [Chapter 51](51-strategies-written-by-hand.md) wrote it.
- **The late worlds:** of the 7,587 agents the patches of Chapter 102 were
  planted on, 4 held that many.

A patch that cannot breed lives only as long as the nodes it holds and the
nodes it takes, and in B2 every agent dies at age 100. The crews died with
the youngest agent they were planted on.

The world's own agents breed as soon as they can give a child ten tokens,
B2's least child ([`min_child`](../notes/settings.md#min_child)). This
experiment gives the raider the same:
- **The breeding raider** raids as the raider does: everything on the
  weakest stranger next door if seven tenths of its tokens outweigh what
  that stranger holds, counted one and a half times; otherwise everything
  at home. It has a child of ten tokens whenever it holds more, joined to
  its teammates as the raider's children are. Each child can raid at once:
  seven tokens staked take a stranger of up to four.
- **The breeding turtle** breeds the same way and never raids. It keeps
  everything at home. It tells breeding alone from breeding and raiding.

Both are planted where Chapters 97 and 102 planted the raiders, 5% of B2's
five worlds played on, at 2,000, 5,000 and 10,000, in the same three places,
beside the lone raiders' and the controls' plantings, which they share with
those chapters.

## The thesis

<!-- thesis E34 -->
> [!quote] The thesis of Experiment 34, written down before any of its runs existed
> **The claim.** A raider that breeds as the world's own agents can takes hold where the barren raider could not, and it is the raid that does it. In B2's worlds played on, at iterations 5,000 and 10,000, where patches of 5% of Chapter 51's raider, which has a child only when it holds 30 tokens, took hold in none of 30 plantings, patches of 5% of a raider that has a child of ten tokens whenever it holds more take hold in at least 5 of the same 30, and in more of them than patches of a turtle that breeds the same way but never raids.
>
> **Why it would be so.** In Experiment 33 neither kind of raider had a child: of the 7,587 agents the patches were planted on, 4 held the 30 tokens their rules wait for. A barren patch lives only as long as the nodes it holds and the nodes it takes, and in B2 every agent dies at age 100; the crews died with the youngest agent they were planted on. The world's own agents have a child as soon as they can give it ten tokens (B2's min_child). A raider that does the same replaces its dead, and each of its children can raid at once: seven tokens staked take a stranger of up to four. The breeding turtle breeds the same way and never raids, and so tells breeding alone from breeding and raiding.
>
> **It holds if:** Breeding raiders take hold in at least 5 of the 30 plantings at 5,000 and 10,000, and in more of them than breeding turtles do: alive after 300 iterations, more than the patch began with, and more than the control planted in the same place.
>
> **It fails if:** Breeding raiders take hold in 4 or fewer of the 30, or in no more of them than breeding turtles: breeding does not open the door, or the raid adds nothing to it; the chapter says which.
<!-- /thesis -->

## The trials

<!-- runs E34 -->
> [!info] The trials behind this chapter
> **180 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **breeding raiders** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies breeding-raider; patches of 5% of the world; places 1, 2, 3.
> - **breeding turtles** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies breeding-turtle; patches of 5% of the world; places 1, 2, 3.
> - **lone raiders** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 45 trials: worlds `pilot:L/B2-long-s001`, `pilot:L/B2-long-s002`, `pilot:L/B2-long-s003`, `pilot:L/B2-long-s004`, `pilot:L/B2-long-s005`; at iteration 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E34`, or ▶ in the Book tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- turns -->
---

← [Chapter 102 · A crew of raiders](102-a-crew-of-raiders.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
