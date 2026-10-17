# Meta X · Healthy and rich

> [!question] Questions of this chapter
> - What would a world that might keep evolving look like from the outside?
> - How close does every world the book has made come to that picture?
> - Which worlds come closest, and what do the others miss?

A world in which every agent holds one, two or three tokens does not look like
a world in which much can happen. The author put the picture the other way
round (2026-10-16): a world that might keep evolving would have a network that
is **decentralised, resilient and almost three-dimensional**, and agents that
**differ in almost every statistic** — in wealth, in links, in age, in their
genes and in what they do.

This chapter turns that picture into measurements and reads every world the
book has made against them: 424 worlds when this was last counted, the lab's
runs at their end and the pilot worlds at their last checkpoint, each counted
once.

## Two halves

**The network.** The health bar of [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)
already says what a decentralised, resilient network is, in seven marks: one
piece; a 2-core holding at least 0.8 of the agents; no agent whose loss cuts
off more than 2% of the others, and no link more than 1%; no agent joined to
more than 5% of the others; 3 to 12 links per agent; a dimension between 2.3
and 4 ([How many dimensions does a world have?](24-how-many-dimensions-does-a-world-have.md)).
The author's "almost three" is the middle of that range.

**The agents.** How much they differ is counted without drawing any line. For
tokens, links and ages, the agents are sorted into doubling classes — 1,
2–3, 4–7, 8–15, … — and the **spread** is the effective number of classes they
fill: 1 if every agent is in one class, *k* if they are spread evenly over *k*
([Effective numbers](../notes/effective-number.md)). Agents holding one to
three tokens fill at most two. Genotypes are counted as an effective number per
agent. A world's **diversity score** is the mean of its percentiles among all
worlds on these four.

## Every world on both halves

<!-- figure healthy/map -->
![Healthy, and diverse?](../figures/healthy/map.svg)

**Healthy, and diverse?** Each dot is a world at its last checkpoint. Across: how many of the seven marks of [Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)'s health bar it meets — one piece, a 2-core of at least 0.8, no agent whose loss cuts off more than 2%, no link more than 1%, no agent joined to more than 5% of the others, 3 to 12 links per agent, a dimension of 2.3 to 4 (the dots are spread sideways within each count only so that they do not hide each other). Up: how widely its agents spread over tokens, links, ages and genotypes, as the mean of its percentiles among all worlds. The author's picture of a world that might keep evolving is the top right. B1: B1 at every size and its variants (Experiments 2, 8, 10, 12, 13); brains fixed: Experiment 7; B2: B2 and its variants (Experiments 9, 14, 15, 16), except lapse, where any unused link lapses; long runs: the pilot worlds played to 10,000 iterations; pilots: the other pilot worlds, by the rule set they start from.

> [!example]- How to make this figure
> **Runs.** Every world the book has made that settled (500 iterations or more), each once, at its last checkpoint: the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/profile.json`.
>
> 1. Read each world's checkpoint: its agents' tokens, links, birth times and genotypes.
> 2. The network: pieces, the 2-core, the worst cut by one agent and by one link (one depth-first search), the best-connected agent's share, links per agent; the dimension by ball growth on the same checkpoint (the lab's measure, gol_series).
> 3. The spreads: the effective number of doubling classes the agents fill (exp of the entropy), for tokens, links and ages; genotypes per agent (1 / sum of squared shares, over the agents).
> 4. Rank every world on each spread; the score is the mean of its four percentiles.
>
> **To make it again:** `python3 book_figures.py healthy`.
<!-- /figure -->

The worlds fall into two clouds, and the author's corner — every mark met, the
agents most diverse — is empty.

- **B1's worlds are rich but fragile.** Their agents spread widely: over 5.6
  to 6.9 classes of age, 3.5 to 4.1 of tokens, with a diversity score of 0.67 to
  0.75.
  But their networks meet two to four of the seven marks: only half the agents
  are in the 2-core, and in the median world one agent's loss cuts off 13% to
  35% of the others. Where tokens move along every link and unused links are
  cut ([Chapter 30](30-what-crosses-a-connection.md)), the network is a tree of
  blobs held together by single agents.
- **B2's worlds are healthy but poor.** Their networks meet six or seven marks
  — one piece, the whole world in the 2-core, nothing that one loss could cut
  off, no hub. But three agents in four hold three tokens or fewer, and the
  median agent holds one. Its agents' ages fill about 4 classes against B1's 6
  to 7, since every agent dies at 100. And at 10,000 tokens B2 misses the one
  mark B1 meets: its dimension is 4.5, above the bar and far from three.

> [!info] Added later (2026-10-16, Chapter 80)
> B2's dimension is a matter of its size. In B2's kind of network the
> dimension shown by ball growth rises with the number of agents: B2 at 5,000
> tokens, with 1,200 to 1,800 agents, shows 3.6 to 4.1, and six of its ten
> worlds meet every mark; the mixed worlds of Experiment 17 show about 3.1 below
> 300 agents and 4 above 1,500 (a correlation of 0.78 with the logarithm of the
> size). B1's does not rise: from 40 agents to 70,000 it stays between about
> 2.5 and 4. So for B2's kind of network two marks of the bar are partly marks
> of size — the dimension fails a large world, the hub a small one, and worlds
> of 600 to 1,000 agents met both at 98% of their moments. Health compared
> between worlds of different sizes has to be read with that in mind
> ([Chapter 80](80-mixed-at-birth-ten-long-worlds.md#the-marks-a-world-misses)).

> [!info] Added later (2026-10-17, Chapter 73)
> The rise with size holds for B2 itself and for its mixed worlds, not for
> every variant of B2: with the estate of the dead left to the newborns or to
> the parents, B2's worlds hold more than 5,000 agents at a dimension of 3.4 to
> 3.9 ([Chapter 73](73-who-inherits.md#healthy-and-diverse)).

<!-- figure healthy/poor -->
![Agents with three tokens or fewer](../figures/healthy/poor.svg)

**Agents with three tokens or fewer.** Each dot is a world: the share of its agents holding one, two or three tokens. B1 10k: B1 at 10,000 tokens (Experiment 2); 100k: Experiment 10; sizes: Experiment 8; 5k: Experiments 12 and 13; fixed: brains that never change (Experiment 7); B2 10k: Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15, 16); B2 5k: Experiment 14, except lapse, where any unused link lapses; long runs: the pilot worlds played to 10,000 iterations; pil.: the other pilot worlds, by the rule set they start from.

> [!example]- How to make this figure
> **Runs.** Every world the book has made that settled (500 iterations or more), each once, at its last checkpoint: the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/profile.json`.
>
> 1. Count the agents of each world's checkpoint holding at most three tokens.
>
> **To make it again:** `python3 book_figures.py healthy`.
<!-- /figure -->

<!-- figure healthy/spreads -->
![How widely the agents spread](../figures/healthy/spreads.svg)

**How widely the agents spread.** Each dot is a world. Tokens, links, ages: the effective number of doubling classes its agents fill (1, 2–3, 4–7, …; for ages, age + 1) — 1 if every agent is in one class, k if they are spread evenly over k. Genotypes: the effective number of genotypes per agent (1 for every agent its own genotype, near 0 for one genotype everywhere).

> [!example]- How to make this figure
> **Runs.** Every world the book has made that settled (500 iterations or more), each once, at its last checkpoint: the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/profile.json`.
>
> 1. For each world, exp of the Shannon entropy of its agents' shares over the doubling classes of each quantity; the effective genotypes, 1 / sum of squared shares, per agent.
>
> **To make it again:** `python3 book_figures.py healthy`.
<!-- /figure -->

## The worlds that come closest

Only 56 worlds meet all seven marks, when last counted: 15 of Experiment 14's
B2 worlds at 5,000 tokens (smaller worlds, with a dimension of about 3.8);
eleven of the lab's B2 worlds at 10,000, nine of them without old age
(Experiment 15), one of B2 itself and one whose brains rarely change
(Experiment 9); five alliance pilots, four of them at 3,000 tokens, below the
book's rule of 10,000; and twenty-five pilot worlds built on B2, two of them
long runs.

> [!info] Added later (2026-10-16, the same day)
> This section first counted 34 worlds. A world's dimension was then taken from
> the table of every world, as a mean over the stretch at its end; it is now
> measured on the same checkpoint as the other six marks (the same measure, at
> one moment, [Chapter 24](24-how-many-dimensions-does-a-world-have.md)). The
> two agree for B2's worlds (half of them within 0.12) and swing for B1's,
> whose networks change fast. Worlds that had no entry in the table, and the
> batches finished since, are now counted too. And the genotypes of worlds
> whose brains are mixed at birth are counted by their weights: the
> prototype's rule gave a mixture of two clones — the parent's brain again — a
> number of its own, so those worlds looked richer in genotypes than they were
> (the batch Y world below fell from a diversity score of 0.66 to 0.57).

A healthy network is not enough. Nine of the ten B2 worlds without old age
meet every mark and have stopped: about 0.4 births an iteration in a world of
3,800 agents, 86 to 95% of the agents holding three tokens or fewer, and
nearly all of them of one age, the founders' generation that never dies (their
ages fill 1.3 to 2.3 classes). Their diversity scores are among the lowest of
all. Ranked by how diverse their agents are, the worlds that meet every mark
are:

<!-- table healthy/closest -->
| world | group | at iteration | marks (of 7) | diversity score | median tokens | agents with 3 tokens or fewer | dimension | agents |
|---|---|---|---|---|---|---|---|---|
| `W/norev-mix-s1` | B2 pil. | 1,000 | 7 | 0.69 | 2 | 67% | 3.7 | 2,245 |
| `Q/mix-s1` | B2 pil. | 2,000 | 7 | 0.63 | 5 | 37% | 3.5 | 908 |
| `B2-5000-d6b607-s006` | B2 5k | 1,000 | 7 | 0.62 | 1 | 71% | 3.0 | 1,158 |
| `B2-5000-s001` | B2 5k | 1,075 | 7 | 0.61 | 2 | 74% | 4.0 | 1,518 |
| `W/norev-linktoken-s1` | B2 pil. | 1,000 | 7 | 0.60 | 5 | 42% | 4.0 | 921 |
| `LC/Q-mix-s2` | long runs | 7,500 | 7 | 0.59 | 8 | 19% | 3.9 | 883 |
| `A3k/gain-s1` | B1 pil. | 1,000 | 7 | 0.58 | 2 | 70% | 3.6 | 804 |
| `Y/mix-poorest-s1` | B2 pil. | 1,000 | 7 | 0.57 | 3 | 51% | 3.4 | 1,455 |
| `A3k/ctrl-s2` | B1 pil. | 1,000 | 7 | 0.57 | 2 | 65% | 3.9 | 779 |
| `B2-5000-28aef0-s006` | B2 5k | 1,000 | 7 | 0.56 | 1 | 75% | 3.9 | 1,564 |
| `Q/mix-s2` | B2 pil. | 2,000 | 7 | 0.55 | 11 | 20% | 3.5 | 659 |
| `B2-5000-28aef0-s003` | B2 5k | 1,000 | 7 | 0.53 | 1 | 78% | 3.8 | 1,736 |
<!-- /table -->

The closest are worlds whose **brains are mixed at birth**
([Chapter 78](78-mixed-brains-costly-units.md)): five of the twelve at the top,
the first among them. The two of batch Q meet every mark at a dimension of 3.5,
and their agents are
richer and more varied than in any other healthy world: the median agent holds
5 and 11 tokens, 37% and 20% hold three or fewer, and their wealth spreads over
six doubling classes, wider than in any other world that meets every mark. They
are smaller worlds — 660 to 910 agents for 10,000 tokens, where B2 holds 3,400
— so each agent holds more. They are also among the worlds whose evolution went
on most strongly. The others are the long run of one of them, below; a world
of batch Y, mixing with estates left to the poorest; and at the very top a
world of batch W that mixes without revolutions — larger and poorer (2,245
agents, two in three at three tokens or fewer), but spread widely in age,
links and genotypes.

## Candidates and long runs

A world that comes this close is a **candidate**, and is played on to see
whether it stays. The rule, with no line of its own
(`research/oee/pilots/candidates.py`, list in `research/oee/candidates.md`):
every mark of the health bar met, a diversity score above the median of
Experiment 16's B2 worlds, and at least 10,000 tokens. Three worlds met it
first — the two mixed worlds of batch Q and a world of batch S whose agents
have an eye for a family tag and no revolutions — and three more as batches W
and Y finished: mixing without revolutions, links that hold a token without
revolutions, and mixing with estates left to the poorest. Each is played on to
10,000 iterations in a copy of its own (`GraphOfLifeRuns/.pilots/LC/`). Where
each was chosen, and where it is now:

<!-- table healthy/long -->
| world | group | at iteration | marks (of 7) | diversity score | median tokens | agents with 3 tokens or fewer | dimension | agents |
|---|---|---|---|---|---|---|---|---|
| `Q/mix-s1` | B2 pil. | 2,000 | 7 | 0.63 | 5 | 37% | 3.5 | 908 |
| `LC/Q-mix-s1` | long runs | 5,500 | 7 | 0.51 | 6 | 38% | 3.8 | 1,037 |
| `Q/mix-s2` | B2 pil. | 2,000 | 7 | 0.55 | 11 | 20% | 3.5 | 659 |
| `LC/Q-mix-s2` | long runs | 7,500 | 7 | 0.59 | 8 | 19% | 3.9 | 883 |
| `S/G4-familysense-s2` | B2 pil. | 1,000 | 7 | 0.53 | 1 | 74% | 4.0 | 2,935 |
| `LC/S-G4-familysense-s2` | long runs | 2,000 | 6 | 0.50 | 2 | 72% | 4.3 | 3,192 |
<!-- /table -->

The two mixed worlds go on healthy, and richer than B2's. One swung towards
B2's poverty at 2,500 iterations — half as many agents again, half of them at
three tokens or fewer — and came back: at 5,500, 38% hold three or fewer and
the median agent six. The other grew richer and smaller to 4,500 iterations
(514 agents, the median holding 13 tokens), then larger again: at 7,500, 883
agents, 19% at three tokens or fewer, the median holding eight, every mark met
at a dimension of 3.9. The world with an eye for a family tag has fallen back:
at 2,000 iterations 72% of its agents hold three tokens or fewer, as in B2, and
a dimension of 4.3 has cost it a mark.

The long runs are planned in three steps:

1. **Every candidate to 10,000 iterations** (running). A candidate that falls
   back towards B2's poverty, or loses a mark, says its rule does not hold.
2. **More seeds for what holds.** One world can be luck; a rule whose candidate
   is still healthy and diverse at 10,000 iterations gets five seeds more, from
   the start, to 10,000.
3. **A pre-registered lab experiment** for what holds in most seeds: ten seeds
   to 10,000 iterations, and the best world on to a hundred thousand, as
   [Chapter 69](69-a-hundred-thousand-iterations.md) plans for one world.
   Mixing is now a setting of the engine,
   [`mix_at_birth`](../notes/settings.md#mix_at_birth), off unless a run turns
   it on, and its experiment is written and ready for the lab:
   [Chapter 80](80-mixed-at-birth-ten-long-worlds.md), Experiment 17, ten worlds
   of B2 with mixing for 10,000 iterations.

> [!info] Added later (2026-10-16, Experiment 17)
> Step 3 has run. The ten mixed worlds of Experiment 17
> ([Chapter 80](80-mixed-at-birth-ten-long-worlds.md)) are rich — 11% of the
> agents at three tokens or fewer in the median world at 10,000 iterations —
> but only five of the ten met every mark, so its thesis is refuted; the five
> that did are being played on to 50,000
> ([Chapter 81](81-five-mixed-worlds-fifty-thousand-iterations.md), Experiment
> 18). Two of its worlds are the two mixed worlds of batch Q again, agent for
> agent — the lab's engine and the prototype play a mixed world alike — so
> `Q/mix-s1` and `s2` and their long runs count as Experiment 17's worlds s001
> and s002 from the next count on, and leave the tables above.

The batches still running (T to Y, [Appendix C](C-every-simulation.md)) and the
search for settings by elimination (batch Z: worlds whose every setting is
drawn at random, the values that make worlds worse dropped round after round,
`research/oee/search.md`) add candidates as they finish.

## What this says

- **No world yet fits the picture.** The two rule sets each have half of it:
  B1 the diverse agents, B2 the healthy network. Neither has both.
- **B2's network rules work; its agents are too many and too poor.** A world of
  10,000 tokens that holds 3,400 agents can only give most of them one token.
  What made worlds richer — mixing brains at birth, links that hold a token,
  lapsing links — made them smaller.
- **Age is capped by a rule.** B2's agents die at 100, so their ages cannot
  spread as B1's do. Without old age B2's worlds meet every mark of the bar and
  stop: hardly a birth, nearly every agent poor and of one age (Experiment 15;
  links that hold a token without old age did the same,
  [Chapter 78](78-mixed-brains-costly-units.md)). Something between, a death
  that depends on what an agent does rather than on a fixed age, would let ages
  spread without stopping the world.
- **A world might be judged by this map** before it is judged by its evolution:
  a rule that moves worlds towards the top right is a candidate for the long
  runs and the lifetime contests. The batches now running will be placed on it
  as they finish.

**Related notes:** [Effective numbers](../notes/effective-number.md) ·
[Every setting](../notes/settings.md) ·
[Chapter 46 · How to tell a world that keeps evolving](46-how-to-tell-a-world-that-keeps-evolving.md) ·
[Chapter 48 · A world that cannot break](48-a-world-that-cannot-break.md) ·
[Chapter 70 · How different are the agents?](70-how-different-are-the-agents.md) ·
[Chapter 78 · Brains mixed at birth, units that cost, an eye for raids on kin](78-mixed-brains-costly-units.md) ·
[Meta VI · Every world side by side](74-meta-6-every-world.md)

<!-- turns -->
---

← [Chapter 77 · Meta IX · How alike the agents are](77-meta-9-how-alike.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
