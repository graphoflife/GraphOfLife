# Hand-written strategies in B1

> [!question] Questions of this chapter
> - The hand-written raiders failed in B2's late worlds for reasons that are
>   B2's own rules. Do they fail in a world without those rules?
> - Is the crews' failure a property of pooling stakes, or of B2?

<!-- experiment E35 -->

## Why ask

Every hand-written strategy so far was planted in B2's worlds only
([Chapters 51](strategies-written-by-hand.md),
[96](how-large-a-crowd-a-raider-needs.md),
[97](does-evolution-close-the-door.md),
[102](a-crew-of-raiders.md) and
[103](raiders-that-breed.md)). And the reasons they failed there are B2's
own rules:
- **A home share.** B2 keeps 30% of an agent's tokens at home, and at least
  one ([`keep_home`](settings.md#keep_home)). Seven in ten agents of
  a late patch held a single token and could stake nothing.
- **A holder's bonus.** A holder's own stake counts one and a half times
  ([`incumbency`](settings.md#incumbency)). A raider needs three
  tokens to take a neighbour of one, and a crew's stake that finds no partner
  is lost.
- **Old age.** Every agent dies at age 100
  ([`max_age`](settings.md#max_age)), so a patch that cannot breed
  dies with its founders.
- **A least child of ten tokens**
  ([`min_child`](settings.md#min_child)), which the breeding
  strategies of Chapter 103 give.

B1, the book's first baseline, has none of them. An agent may stake
everything, a holder's stake counts once, a child may start with a single
token, and nobody dies of age. Its worlds stay rich: at 2,000, 5,000 and
10,000 iterations the median agent of its two long worlds holds 4 to 15
tokens, and almost none holds a single token.

**The rules now read the world's rules.** Until now the raiders carried B2's
numbers: they staked 70% of their tokens, and needed more than one and a half
times what the target holds. They now read both from the world they are
planted in. In B2 that changes nothing (the same choices on 20,000 random
views). In B1:
- a raider stakes everything it holds, and needs only more than its target
  holds, so a raid moves it onto the node it takes and leaves its home
  empty; it grows only by its children;
- a crew raider stakes everything as revolution tokens on the weakest
  stranger no richer than itself, and one as rich as its target ties with it
  when the target keeps everything at home;
- the breeding raider and the breeding turtle have a child of B1's least
  size, a single token, whenever they hold more.

**Counted by tokens.** In B1 a child can be a single token, so counting a
patch by its agents rewards whatever splits into the smallest children. This
chapter counts a patch by the tokens it holds. Counted so, B2's verdicts are
unchanged: lone raiders took hold in 4, 0 and 0 of 15 plantings at 2,000,
5,000 and 10,000, crews in none.

The plantings are those of Chapters 97 and 102 in kind: patches of 5% at
2,000, 5,000 and 10,000, in three places, in each of B1's two long worlds.

## The thesis

<!-- thesis E35 -->
> [!quote] The thesis of Experiment 35, written down before any of its runs existed
> **The claim.** The door that closed on hand-written raiders in B2's late worlds, and the crews' failure there, come from B2's own rules. In B1's two long worlds, which keep no share of an agent's tokens at home, count a holder's own stake once, let a child start with a single token and have no old age, patches of 5% planted at iterations 2,000, 5,000 and 10,000, with the raiders' rules reading each world's own, take hold, counted by the tokens they hold, in at least 5 of the 18 plantings of lone raiders, and in at least as many plantings of crews as of lone raiders.
>
> **Why it would be so.** In B2's worlds at 5,000 and 10,000 no lone raider and no crew took hold in 30 plantings each (Chapters 97 and 102), for reasons that are B2's rules: a late patch held two tokens an agent and seven in ten of its agents a single token, which B2's home share keeps from being staked; a holder's own stake counts one and a half times, so a raider needs three tokens to take one, and a crew's stake that finds no partner is lost; and a patch that cannot breed dies with its founders at age 100. B1 has none of these rules, and its worlds stay rich: the median agent holds 4 to 15 tokens at these eras. Reading B1's rules, a raider stakes everything it holds, so a raid moves it onto the node it takes, and it needs only more than the neighbour holds. A child can be a single token in B1, so a patch is counted by the tokens it holds rather than by its agents; counted so, B2's verdicts are the same (lone raiders 4, 0 and 0 of 15; crews none). The breeding raider and the breeding turtle of Chapter 103 are planted too, and reported.
>
> **It holds if:** Counted by tokens (alive after 300 iterations, more tokens than the patch began with, and more than the control planted in the same place), lone raiders take hold in at least 5 of the 18 plantings, and crews in at least as many as lone raiders.
>
> **It fails if:** Lone raiders take hold in 4 or fewer of the 18, or crews in fewer than lone raiders: that verdict of B2 holds in B1 too, and the chapter says which.
<!-- /thesis -->

## The trials

<!-- runs E35 -->
> [!info] The trials behind this chapter
> **90 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **crews** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies crew; patches of 5% of the world; places 1, 2, 3.
> - **breeding raiders** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies breeding-raider; patches of 5% of the world; places 1, 2, 3.
> - **breeding turtles** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies breeding-turtle; patches of 5% of the world; places 1, 2, 3.
> - **lone raiders** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E35`, or ▶ in the Book tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- turns -->
---

← [Chapter 103 · Raiders that breed](103-raiders-that-breed.md) · [Contents](../README.md) · [Chapter 106 · A screen of random worlds](106-a-screen-of-random-worlds.md) →
<!-- /turns -->
