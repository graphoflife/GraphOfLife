# Hand-written strategies in B1

> [!question] Questions of this page
> - The hand-written raiders failed in B2's late worlds for reasons that are
>   B2's own rules. Do they fail in a world without those rules?
> - Is the crews' failure a property of pooling stakes, or of B2?

<!-- experiment E35 -->

## In short

Every strategy written by hand failed in B2's late worlds, and each failure
had a reason in B2's own rules. Experiment 35 planted the same strategies in
B1's two long worlds, which have none of those rules, and they failed faster.
Not one of 90 plantings took hold. The breeding strategies died out within
about ten [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), lone raiders and crews within about a hundred, while
copies of B1's own evolved [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") grew to several times their patch. B1 has
a rule of its own that these strategies were not written for: a link that
carries no [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") in a [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") is cut, and they never stake on each other.

## Why ask

Every hand-written strategy so far was planted in B2's worlds only
([Strategies written by hand](strategies-written-by-hand.md),
[How large a crowd a raider needs](how-large-a-crowd-a-raider-needs.md),
[Does evolution close the door?](does-evolution-close-the-door.md),
[A crew of raiders](a-crew-of-raiders.md) and
[Raiders that breed](raiders-that-breed.md)). And the reasons they failed
there are B2's own rules:
- **A home share.** B2 keeps 30% of an [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.")'s tokens at home, and at least
  one ([`keep_home`](settings.md#keep_home)). Seven in ten agents of a late
  patch held a single token and could stake nothing.
- **A holder's bonus.** A holder's own stake counts one and a half times
  ([`incumbency`](settings.md#incumbency)). A raider needs three tokens to
  take a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") of one, and a crew's stake that finds no partner is lost.
- **Old age.** Every agent dies at age 100
  ([`max_age`](settings.md#max_age)), so a patch that cannot breed dies with
  its [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.").
- **A least child of ten tokens**
  ([`min_child`](settings.md#min_child)), which the breeding strategies of
  [Raiders that breed](raiders-that-breed.md) give.

B1, the notebook's first baseline, has none of them. An agent may stake
everything, a holder's stake counts once, a child may start with a single
token, and nobody dies of age. Its worlds stay rich: at 2,000, 5,000 and
10,000 iterations the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") agent of its two long worlds holds 4 to 15
tokens, and almost none holds a single token.

**The rules now read the world's rules.** Until now the raiders carried B2's
numbers: they staked 70% of their tokens, and needed more than one and a half
times what the target holds. They now read both from the world they are
planted in. In B2 that changes nothing (the same choices on 20,000 random
views). In B1:
- a raider stakes everything it holds, and needs only more than its target
  holds, so a raid moves it onto the node it takes and leaves its home
  empty; it grows only by its children;
- a crew raider stakes everything as [revolution](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") tokens on the weakest
  stranger no richer than itself, and one as rich as its target ties with it
  when the target keeps everything at home;
- the breeding raider and the breeding turtle have a child of B1's least
  size, a single token, whenever they hold more.

**Counted by tokens.** In B1 a child can be a single token, so counting a
patch by its agents rewards whatever splits into the smallest children. This
page counts a patch by the tokens it holds. Counted so, B2's verdicts are
unchanged: lone raiders took hold in 4, 0 and 0 of 15 plantings at 2,000,
5,000 and 10,000, crews in none.

The plantings are those of
[Does evolution close the door?](does-evolution-close-the-door.md) and
[A crew of raiders](a-crew-of-raiders.md) in kind: patches of 5% at 2,000,
5,000 and 10,000, in three places, in each of B1's two long worlds. Beside
each is a control: the patch given frozen copies of one of its own agents'
brains.

## What we expected

If B2's rules closed the door, B1 should open it. Without a home share or a
holder's bonus a raid is cheap, without old age a patch need not breed to
last, and the agents a patch meets are rich enough to be worth taking. So we
expected lone raiders to take hold in B1, and crews at least as often.

<!-- thesis E35 -->
> [!quote] The thesis of Experiment 35, written down before any of its runs existed
> **The claim.** The door that closed on hand-written raiders in B2's late worlds, and the crews' failure there, come from B2's own rules. In B1's two long worlds, which keep no share of an agent's tokens at home, count a holder's own stake once, let a child start with a single token and have no old age, patches of 5% planted at iterations 2,000, 5,000 and 10,000, with the raiders' rules reading each world's own, take hold, counted by the tokens they hold, in at least 5 of the 18 plantings of lone raiders, and in at least as many plantings of crews as of lone raiders.
>
> **Why it would be so.** In B2's worlds at 5,000 and 10,000 no lone raider and no crew took hold in 30 plantings each (Chapters [97](does-evolution-close-the-door.md) and [102](a-crew-of-raiders.md)), for reasons that are B2's rules: a late patch held two tokens an agent and seven in ten of its agents a single token, which B2's home share keeps from being staked; a holder's own stake counts one and a half times, so a raider needs three tokens to take one, and a crew's stake that finds no partner is lost; and a patch that cannot breed dies with its founders at age 100. B1 has none of these rules, and its worlds stay rich: the median agent holds 4 to 15 tokens at these eras. Reading B1's rules, a raider stakes everything it holds, so a raid moves it onto the node it takes, and it needs only more than the neighbour holds. A child can be a single token in B1, so a patch is counted by the tokens it holds rather than by its agents; counted so, B2's verdicts are the same (lone raiders 4, 0 and 0 of 15; crews none). The breeding raider and the breeding turtle of Chapter [103](raiders-that-breed.md) are planted too, and reported.
>
> **It holds if:** Counted by tokens (alive after 300 iterations, more tokens than the patch began with, and more than the control planted in the same place), lone raiders take hold in at least 5 of the 18 plantings, and crews in at least as many as lone raiders.
>
> **It fails if:** Lone raiders take hold in 4 or fewer of the 18, or crews in fewer than lone raiders: that verdict of B2 holds in B1 too, and the chapter says which.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

## What happened

### Nothing took hold

<!-- figure b1raiders/hold -->
![Every strategy in B1's two long worlds](../figures/b1raiders/hold.svg)

**Every strategy in B1's two long worlds.** Of the six plantings at each era (two worlds, three places): crews (red), breeding raiders (orange), lone raiders (violet), breeding turtles (green) and the controls (grey). Left: how many were still alive after 300 iterations; none of the strategies took hold, by its tokens or by its agents. Right: the median iteration a planting's last agent died in.

> [!example]- How to make this figure
> **Runs.** Experiment 35's 90 invasions (`python3 gol_lab.py run E35`): B1's two worlds played on to 10,000 iterations as pilots (`GraphOfLifeRuns/.pilots/L/B1-long-s1` and `s2`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the crew's rule, the breeding raider's, the breeding turtle's, the lone raider's or frozen copies of one patch member's brain (the control), each rule reading B1's home share, holder's bonus and least child.
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the patch's strategy's agents and their tokens after every iteration (each trial's `result.json`).
> 3. Count the plantings of each era whose strategy is alive after 300 iterations, and take the median of the iteration each planting's last agent died in (300 if it lived to the end).
>
> **To make it again:** `python3 book_figures.py b1raiders`.
<!-- /figure -->

Not one of the 90 plantings took hold, counted by tokens or by agents. After
300 iterations no crew, breeding raider or breeding turtle was alive
anywhere, and lone raiders in one planting of 18, at 2,000. Even the controls
were alive in only 2 of the 6 plantings of each [era](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?").

The breeding strategies died first: in the median planting their last agent
died 6 to 11 iterations after planting. Lone raiders lasted 59 to 134
iterations, crews 76 to 120, the controls 130 to 237.

### What the patches did

<!-- figure b1raiders/patches -->
![What became of the patches in B1](../figures/b1raiders/patches.svg)

**What became of the patches in B1.** The median over the eighteen plantings of each kind, at each iteration after planting: the strategy's tokens (left) and agents (right) as a share of what the patch began with, 0 once it has died out.

> [!example]- How to make this figure
> **Runs.** Experiment 35's 90 invasions (`python3 gol_lab.py run E35`): B1's two worlds played on to 10,000 iterations as pilots (`GraphOfLifeRuns/.pilots/L/B1-long-s1` and `s2`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the crew's rule, the breeding raider's, the breeding turtle's, the lone raider's or frozen copies of one patch member's brain (the control), each rule reading B1's home share, holder's bonus and least child.
>
> **Data.** From each trial's `result.json`.
>
> 1. Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by step, until the patch holds 5% of the world.
> 2. Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the patch's strategy's agents and their tokens after every iteration (each trial's `result.json`).
> 3. At each iteration after planting, take the median over the eighteen plantings of the strategy's tokens over the patch's tokens at planting, and of its agents over the patch planted.
>
> **To make it again:** `python3 book_figures.py b1raiders`.
<!-- /figure -->

The strategies lost most of their agents in their very first iteration: the
median patch kept 26% to 28% of its agents for the breeding strategies, 40%
for lone raiders and crews. The controls gained 27% in the same iteration.

B1's rules explain it. In B1 a link that carries no tokens in a game is cut,
wherever it is ([`prune_needs`](settings.md#prune_needs) = 0; B2 cuts an
unused link only where its two ends share at least two other neighbours).
These strategies never stake on each other: each stakes at home, or on a
stranger. So nearly every link inside a patch carries nothing and is cut, and
an agent of the patch with no stranger next door is cut off from the world.
The [cleanup](one-iteration.md#cleanup "The end of every phase: only the largest connected piece of the network lives on, and the tokens of the dead are shared out.") removes it at the end of the game. Evolved brains spread their
stakes over their neighbours as well as their own node, so their links carry
tokens and stay.

Then each strategy failed in its own way:
- **The breeding raider and the breeding turtle** have a child of a single
  token whenever they hold more, and 98% to 99% of the agents of these worlds
  hold two or more. So they had a child every iteration, and their tokens
  went into children of one token, the poorest agents of a rich world. Their
  patches' tokens were gone within ten iterations.
- **The lone raider and the crew** never grew. No planting of either ever
  held more agents than it was planted with. A raid in B1 stakes everything,
  so a raider that takes a node leaves its own: it moves, it does not spread.
  Their patches' tokens drained over about a hundred iterations.
- **The controls**, copies of an evolved brain, spread: at its largest, the
  median planting held 2.5, 5.1 and 6.7 times its patch at the three eras.
  Yet in 4 of the 6 plantings of each era they too had died out by iteration
  300.

### Verdict: refuted

The thesis is refuted. Lone raiders took hold in none of the 18 plantings,
where it asked for at least 5. Crews took hold in as many, none, but the
thesis needed both. So it was not B2's rules alone that kept the raiders out.
In B1, without a home share, a holder's bonus, old age or a least child of
ten tokens, every strategy written by hand failed sooner, and partly by a
rule of B1's own.

## What we learned

- **B2's rules were not what kept the raiders out.** In B1, which has none
  of them, every strategy written by hand failed, and faster.
- **A strategy must fit the world's rules on links.** In B1 a link that
  carries no tokens is cut. A patch whose agents never stake on each other
  cuts itself into pieces in its first game. A strategy written for B1 would
  have to keep its links in use, as evolved brains do.
- **A least child of one token is a trap** for a strategy that breeds
  whenever it can: it pours its tokens into children too poor to keep a
  node.
- **Without a home share, a raid is a move.** A raider that stakes
  everything takes a node and leaves its own, so it cannot grow by raiding.
- **B1's own evolved brains spread where the strategies could not.** Copies
  of one patch agent's brain grew to several times their patch.

## Where to go next

- [Raiders that breed](raiders-that-breed.md): the breeding strategies in
  B2's late worlds.
- [A crew of raiders](a-crew-of-raiders.md): raiders that pool their stakes,
  in B2.
- [Strategies written by hand](strategies-written-by-hand.md): the
  strategies, and their first plantings.
- [B1 and B2: the two baselines](b1-and-b2.md): every rule in which the two
  differ.

## How this was made

Each figure says, folded beneath it, how it is drawn from the trials;
`python3 book_figures.py b1raiders` draws them again. The share of agents
holding two tokens or more is read from the two worlds as they were kept at
each era (`GraphOfLifeRuns/.pilots/L/B1-long-s1` and `s2`,
`ckpt_<iteration>.npz`), and written with every other number of this page to
`book/results/b1raiders.json`.

<!-- runs E35 -->
> [!info]- The trials behind this page
> **90 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 300 iterations, from a world another run kept in its archive.
>
> - **crews** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies crew; patches of 5% of the world; places 1, 2, 3.
> - **breeding raiders** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies breeding-raider; patches of 5% of the world; places 1, 2, 3.
> - **breeding turtles** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies breeding-turtle; patches of 5% of the world; places 1, 2, 3.
> - **lone raiders** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies raider; patches of 5% of the world; places 1, 2, 3.
> - **controls** — 18 trials: worlds `pilot:L/B1-long-s1`, `pilot:L/B1-long-s2`; at iteration 2,000, 5,000, 10,000; strategies resident; patches of 5% of the world; places 1, 2, 3.
>
> To play these trials again: `python3 gol_lab.py run E35`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- turns -->
---

← [Raiders that breed](raiders-that-breed.md) · [Strategies written by hand](../README.md#strategies-written-by-hand) · [The invasion test](invasion-test.md) →

**Also about:** [The baselines B1 and B2](../README.md#the-baselines-b1-and-b2).

> [!info]- Pages that link here (2)
> [A crew of raiders](a-crew-of-raiders.md) · [Raiders that breed](raiders-that-breed.md)
<!-- /turns -->
