# Healthy and rich

> [!question] Questions of this page
> - What would a world that might keep evolving look like from the outside?
> - How close does every world the notebook has made come to that picture?
> - Which worlds come closest, and what do the others miss?

A world in which every [agent](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") holds one, two or three [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") does not look like
a world in which much can happen. The author put the picture the other way
round (2026-10-16): a world that might keep evolving would have a network that
is **decentralised, resilient and almost three-dimensional**, and agents that
**differ in almost every statistic** — in wealth, in links, in age, in their
genes and in what they do.

This page turns that picture into measurements and reads every world the
notebook has made against them: 424 worlds when this was last counted, the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s
runs at their end and the [pilot](pilot-worlds.md "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") worlds at their last checkpoint, each counted
once.

## Two halves

**The network.** The [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4.") of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)
already says what a decentralised, resilient network is, in seven marks: one
piece; a [2-core](core-trees-leaves.md "What is left of a network when agents with one connection or none are peeled away again and again: the agents on loops or between them.") holding at least 0.8 of the agents; no agent whose loss cuts
off more than 2% of the others, and no link more than 1%; no agent joined to
more than 5% of the others; 3 to 12 links per agent; a [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") between 2.3
and 4 ([How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)).
The author's "almost three" is the middle of that range.

**The agents.** How much they differ is counted without drawing any line. For
tokens, links and ages, the agents are sorted into doubling classes — 1,
2–3, 4–7, 8–15, … — and the **spread** is the effective number of classes they
fill: 1 if every agent is in one class, *k* if they are spread evenly over *k*
([Effective numbers](effective-number.md)). Agents holding one to
three tokens fill at most two. [Genotypes](genotype.md "The identity of a brain: a number that a brain keeps until it changes, and that records whose copy it was.") are counted as an effective number per
agent. A world's **[diversity score](diversity-score.md "How much the agents of a world differ, in one number from 0 to 1: in tokens, links, ages and genotypes.")** is the mean of its percentiles among all
worlds on these four.

> [!info] Added later (17 October 2026)
> Ranked among all worlds, a score moves whenever worlds are added — by up to
> 0.01 between this page and the batches read since. The scores are now
> ranked among a frozen set instead: the 653 worlds of the profile on 17
> October 2026 (`book/results/diversity_reference.json`, made by
> `research/atlas/profile.py --freeze`). A later world gets the percentile its
> spreads would have among them, and no score moves unless the set is
> deliberately made again.

## Every world on both halves

<!-- figure healthy/map -->
![Healthy, and diverse?](../figures/healthy/map.svg)

**Healthy, and diverse?** Each dot is a world at its last checkpoint. Across: how many of the seven marks of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)'s health bar it meets — one piece, a 2-core of at least 0.8, no agent whose loss cuts off more than 2%, no link more than 1%, no agent joined to more than 5% of the others, 3 to 12 links per agent, a dimension of 2.3 to 4 (the dots are spread sideways within each count only so that they do not hide each other). Up: how widely its agents spread over tokens, links, ages and genotypes, as the mean of its percentiles among all worlds. The author's picture of a world that might keep evolving is the top right. B1: B1 at every size and its variants (Experiments 2, 8, 10, 12, 13); brains fixed: Experiment 7; B2: B2 and its variants (Experiments 9, 14, 15, 16), except lapse, where any unused link lapses; long runs: the pilot worlds played to 10,000 iterations; pilots: the other pilot worlds, by the rule set they start from.

> [!example]- How to make this figure
> **Runs.** Every world the notebook has made that settled (500 iterations or more), each once, at its last checkpoint: the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).
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

- **[B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds are rich but fragile.** Their agents spread widely: over 5.6
  to 6.9 classes of age, 3.5 to 4.1 of tokens, with a diversity score of 0.67 to
  0.75.
  But their networks meet two to four of the seven marks: only half the agents
  are in the 2-core, and in the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world one agent's loss cuts off 13% to
  35% of the others. Where tokens move along every link and unused links are
  cut ([What crosses a connection](what-crosses-a-connection.md)), the network is a tree of
  blobs held together by single agents.
- **[B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")'s worlds are healthy but poor.** Their networks meet six or seven marks
  — one piece, the whole world in the 2-core, nothing that one loss could cut
  off, no [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."). But three agents in four hold three tokens or fewer, and the
  median agent holds one. Its agents' ages fill about 4 classes against B1's 6
  to 7, since every agent dies at 100. And at 10,000 tokens B2 misses the one
  mark B1 meets: its dimension is 4.5, above the bar and far from three.

> [!info] Added later (2026-10-16, [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md))
> B2's dimension is a matter of its size. In B2's kind of network the
> dimension shown by ball growth rises with the number of agents: B2 at 5,000
> tokens, with 1,200 to 1,800 agents, shows 3.6 to 4.1, and six of its ten
> worlds meet every mark; the mixed worlds of Experiment 17 show about 3.1 below
> 300 agents and 4 above 1,500 (a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of 0.78 with the logarithm of the
> size). B1's does not rise: from 40 agents to 70,000 it stays between about
> 2.5 and 4. So for B2's kind of network two marks of the bar are partly marks
> of size — the dimension fails a large world, the hub a small one, and worlds
> of 600 to 1,000 agents met both at 98% of their moments. Health compared
> between worlds of different sizes has to be read with that in mind
> ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md#the-marks-a-world-misses)).

> [!info] Added later (2026-10-17, [Who inherits?](who-inherits.md))
> The rise with size holds for B2 itself and for its mixed worlds, not for
> every variant of B2: with the [estate](estate.md "The tokens a dead agent leaves, and who gets them.") of the dead left to the newborns or to
> the parents, B2's worlds hold more than 5,000 agents at a dimension of 3.4 to
> 3.9 ([Who inherits?](who-inherits.md#healthy-and-diverse)).

> [!info] Added later (2026-10-17, [What folds a world](what-folds-a-world.md))
> Why the dimension rises with size in B2's kind of network: those networks
> are nearly small worlds, not spaces. A typical agent's ball grows four to
> six times a step and holds a quarter of a world of 3,400 agents within
> four steps; a typical agent is about as near to everyone as in a random
> network with the same links (1.18 times as far, against 2.3 in B1). A small
> world has no dimension to find, and the number read off its first few
> steps grows with the room it has. B2's worlds start as spaces and fold
> once their first generation dies, as each death is mended with a ring of
> links; the worlds whose estate goes to the newborns or the parents stay
> spaces, of about three dimensions
> ([What folds a world](what-folds-a-world.md)).

<!-- figure healthy/poor -->
![Agents with three tokens or fewer](../figures/healthy/poor.svg)

**Agents with three tokens or fewer.** Each dot is a world: the share of its agents holding one, two or three tokens. B1 10k: B1 at 10,000 tokens (Experiment 2); 100k: Experiment 10; sizes: Experiment 8; 5k: Experiments 12 and 13; fixed: brains that never change (Experiment 7); B2 10k: Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15, 16); B2 5k: Experiment 14, except lapse, where any unused link lapses; long runs: the pilot worlds played to 10,000 iterations; pil.: the other pilot worlds, by the rule set they start from.

> [!example]- How to make this figure
> **Runs.** Every world the notebook has made that settled (500 iterations or more), each once, at its last checkpoint: the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/atlas/profile.json`.
>
> 1. Count the agents of each world's checkpoint holding at most three tokens.
>
> **To make it again:** `python3 book_figures.py healthy`.
<!-- /figure -->

> [!info] Added later (19 October 2026): too even, not too poor
> The author made the picture more precise: what looks unhealthy is not that
> agents hold few tokens but that they hold them **evenly** — almost everyone
> the same few. The share at three tokens or fewer only stood in for that,
> and measured directly it stands in badly. Every world read so far now also
> carries the [Gini coefficient](gini-coefficient.md) of its agents' tokens (0 when all hold the
> same) and the share its [richest tenth](richest-tenth.md "The share of all tokens held by the richest tenth of the agents.") hold
> (`research/atlas/evenness.py`). Across the 502 worlds read at [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.")
> 1,000, the share at three or fewer follows the tokens per agent (a
> correlation of −0.77; −0.92 across the 54 read at 10,000) and not the Gini
> (0.01; 0.52 at 10,000). It counts how many agents share the tokens, not
> how evenly they share them.
>
> **B2 is poor but not even.** Its Gini is 0.49 to 0.56 at iteration 1,000
> and 0.47 to 0.51 at 10,000, above the 0.40 that a random exchange of
> tokens would leave at its three tokens per agent. That reference is the
> spread of greatest [entropy](entropy-and-evenness.md "How evenly something is shared out among the agents, from all in one place to spread perfectly evenly.") at a mean of m tokens, a geometric one, whose
> Gini is (m − 1)/(2m − 1); it draws no line of its own. The evenest worlds
> are others:
> - where every unused link lapses: a median Gini of 0.26 over 29 worlds, at
>   seven tokens per agent, where random exchange would leave 0.46;
> - Experiment 15's children of any size: every agent holds one token, a Gini
>   of 0.02;
> - at 10,000 iterations, four of the five candidates that held: rich worlds
>   with a Gini of 0.37 to 0.44
>   ([The candidates, played on](the-candidates-played-on.md)).
>
> From now on a thesis no longer counts a world's poor. It asks instead that
> the world be **no more even than B2**: a Gini no lower than the least of
> B2's worlds read at the same iteration. Old theses keep their wording;
> [The candidates, played on](the-candidates-played-on.md) and [Five more seeds for the rules that held](the-candidates-played-on.md#five-more-seeds-for-the-rules-that-held) and [Rules in company](rules-in-company.md) say what the clarified reading changes.

<!-- figure healthy/spreads -->
![How widely the agents spread](../figures/healthy/spreads.svg)

**How widely the agents spread.** Each dot is a world. Tokens, links, ages: the effective number of doubling classes its agents fill (1, 2–3, 4–7, …; for ages, age + 1) — 1 if every agent is in one class, k if they are spread evenly over k. Genotypes: the effective number of genotypes per agent (1 for every agent its own genotype, near 0 for one genotype everywhere).

> [!example]- How to make this figure
> **Runs.** Every world the notebook has made that settled (500 iterations or more), each once, at its last checkpoint: the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).
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
(Experiment 15), one of B2 itself and one whose [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") rarely change
(Experiment 9); five alliance pilots, four of them at 3,000 tokens, below the
notebook's rule of 10,000; and twenty-five pilot worlds built on B2, two of them
long runs.

> [!info] Added later (2026-10-17, counted again)
> With batches K, M, R, SW and Y in the count, and the two pilot worlds that
> turned out to be Experiment 17's worlds counted once, 99 worlds of 624 meet
> all seven marks: 26 of the lab's B2 worlds (nine without old age, fifteen of
> B2 or its variants at 5,000 tokens, one of B2 itself, one whose brains
> rarely change), 67 pilot worlds built on B2 — 28 of them with brains mixed
> at birth and one other rule (batch Y,
> [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md)), 12 of batch W — five
> alliance pilots of B1, four of them at 3,000 tokens, and one long run. The closest of all are now
> brains mixed at birth with a faster change of brains (diversity 0.79 and 0.78,
> about 40% of the agents at three tokens or fewer).

> [!info] Added later (2026-10-16, the same day)
> This section first counted 34 worlds. A world's dimension was then taken from
> the table of every world, as a mean over the stretch at its end; it is now
> measured on the same checkpoint as the other six marks (the same measure, at
> one moment, [How many dimensions does a world have?](how-many-dimensions-does-a-world-have.md)). The
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
nearly all of them of one age, the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")' generation that never dies (their
ages fill 1.3 to 2.3 classes). Their diversity scores are among the lowest of
all. Ranked by how diverse their agents are, the worlds that meet every mark
are:

<!-- table healthy/closest -->
| world | group | at iteration | marks (of 7) | diversity score | median tokens | agents with 3 tokens or fewer | dimension | agents |
|---|---|---|---|---|---|---|---|---|
| `Y/mix-mut80-s2` | B2 pil. | 1,000 | 7 | 0.79 | 5 | 36% | 3.8 | 1,238 |
| `Y/mix-mut80-s1` | B2 pil. | 1,000 | 7 | 0.78 | 5 | 41% | 3.8 | 1,299 |
| `Y/mix-keepone-s2` | B2 pil. | 1,000 | 7 | 0.70 | 1 | 77% | 3.4 | 1,537 |
| `Y/mix-kinsense-s2` | B2 pil. | 1,000 | 7 | 0.70 | 8 | 32% | 3.7 | 821 |
| `U/U-BDFG-s1` | B2 pil. | 1,000 | 7 | 0.68 | 8 | 25% | 3.7 | 652 |
| `W/norev-mix-s1` | B2 pil. | 1,000 | 7 | 0.68 | 2 | 67% | 3.7 | 2,245 |
| `Y/mix-linktoken-s1` | B2 pil. | 1,000 | 7 | 0.66 | 8 | 36% | 3.4 | 588 |
| `Y/mix-came-s2` | B2 pil. | 1,000 | 7 | 0.66 | 3 | 53% | 3.7 | 1,521 |
| `M/mix-children-s2` | B2 pil. | 5,000 | 7 | 0.65 | 10 | 23% | 3.3 | 486 |
| `Y/mix-grow-s1` | B2 pil. | 1,000 | 7 | 0.65 | 12 | 23% | 3.3 | 366 |
| `Y/mix-helped-s1` | B2 pil. | 1,000 | 7 | 0.63 | 5 | 38% | 3.7 | 1,075 |
| `Y/mix-nodice-s2` | B2 pil. | 1,000 | 7 | 0.63 | 3 | 53% | 3.5 | 1,468 |
<!-- /table -->

The closest are worlds whose **brains are mixed at birth**
([Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md)): five of the twelve at the top,
the first among them. The two of batch Q meet every mark at a dimension of 3.5,
and their agents are
richer and more varied than in any other [healthy world](what-makes-a-world-interesting.md "A world that meets the six criteria of the screen of random worlds; an interesting one misses at most one."): the median agent holds
5 and 11 tokens, 37% and 20% hold three or fewer, and their wealth spreads over
six doubling classes, wider than in any other world that meets every mark. They
are smaller worlds — 660 to 910 agents for 10,000 tokens, where B2 holds 3,400
— so each agent holds more. They are also among the worlds whose evolution went
on most strongly. The others are the long run of one of them, below; a world
of batch Y, mixing with estates left to the poorest; and at the very top a
world of batch W that mixes without [revolutions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") — larger and poorer (2,245
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
| `S/G4-familysense-s2` | B2 pil. | 1,000 | 7 | 0.53 | 1 | 74% | 4.0 | 2,935 |
| `LC/S-G4-familysense-s2` | long runs | 10,000 | 6 | 0.51 | 2 | 69% | 4.5 | 3,097 |
| `W/norev-linktoken-s1` | B2 pil. | 1,000 | 7 | 0.59 | 5 | 42% | 4.0 | 921 |
| `LC/W-norev-linktoken-s1` | long runs | 10,000 | 6 | 0.45 | 3 | 51% | 4.1 | 1,359 |
| `W/norev-mix-s1` | B2 pil. | 1,000 | 7 | 0.68 | 2 | 67% | 3.7 | 2,245 |
| `LC/W-norev-mix-s1` | long runs | 10,000 | 6 | 0.61 | 2 | 64% | 4.1 | 2,221 |
| `Y/mix-poorest-s1` | B2 pil. | 1,000 | 7 | 0.57 | 3 | 51% | 3.4 | 1,455 |
| `LC/Y-mix-poorest-s1` | long runs | 10,000 | 7 | 0.28 | 1 | 69% | 3.2 | 2,288 |
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

> [!info] Added later (2026-10-17, the long runs at 10,000)
> The two mixed worlds of batch Q are Experiment 17's worlds of [seeds](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 and 2,
> agent for agent, and are read there now
> ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md)): at 10,000 iterations
> one held 2,221 agents, 65% at three tokens or fewer, and the other 744, 6%.
> The four other candidates all reached 10,000, and none held everything it
> had at 1,000: three lost a mark as they grew — their dimension rose above 4,
> at 1,359 to 3,097 agents — and the fourth, mixing with the estate to the
> poorest, kept every mark but grew poor and alike (a diversity score of 0.28,
> 69% at three tokens or fewer). Nineteen new candidates, most of them mixed
> worlds of batch Y, are being played on to 10,000.

The long runs are planned in three steps:

1. **Every candidate to 10,000 iterations** (running). A candidate that falls
   back towards B2's poverty, or loses a mark, says its rule does not hold.
2. **More seeds for what holds.** One world can be luck; a rule whose candidate
   is still healthy and diverse at 10,000 iterations gets five seeds more, from
   the start, to 10,000.
3. **A pre-registered lab experiment** for what holds in most seeds: ten seeds
   to 10,000 iterations, and the best world on to a hundred thousand, as
   [A hundred thousand iterations](a-hundred-thousand-iterations.md) plans for one world.
   Mixing is now a setting of the engine,
   [`mix_at_birth`](settings.md#mix_at_birth), off unless a run turns
   it on, and its experiment is written and ready for the lab:
   [Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md), Experiment 17, ten worlds
   of B2 with mixing for 10,000 iterations.

> [!info] Added later (2026-10-16, Experiment 17)
> Step 3 has run. The ten mixed worlds of Experiment 17
> ([Mixed at birth: ten long worlds](mixed-at-birth-ten-long-worlds.md)) are rich — 11% of the
> agents at three tokens or fewer in the median world at 10,000 iterations —
> but only five of the ten met every mark, so its thesis is refuted; the five
> that did are being played on to 50,000
> ([Five mixed worlds, fifty thousand iterations](mixed-at-birth-ten-long-worlds.md#five-mixed-worlds-fifty-thousand-iterations), Experiment
> 18). Two of its worlds are the two mixed worlds of batch Q again, agent for
> agent — the lab's engine and the prototype play a mixed world alike — so
> `Q/mix-s1` and `s2` and their long runs count as Experiment 17's worlds s001
> and s002 from the next count on, and leave the tables above.

> [!info] Added later (19 October 2026): steps 1 and 2
> Every candidate that can be read at 10,000 iterations now has been
> ([The candidates, played on](the-candidates-played-on.md)): of nineteen, five held,
> eleven fell back — most of them into B2's poverty — and three died out on
> the way; four whose rules exist only in the pilots' prototype wait for
> Experiment 23. Step 2 gave each rule that held five new seeds (Experiment 21,
> [Five more seeds for the rules that held](the-candidates-played-on.md#five-more-seeds-for-the-rules-that-held)): none held more
> often than mixing alone, so step 3 has no rule to take up yet.

The batches still running (T to Y, [Every simulation: what was run, and what was measured](every-simulation.md)) and the
search for settings by elimination (batch Z: worlds whose every setting is
drawn at random, the values that make worlds worse dropped round after round,
`research/oee/search.md`) add candidates as they finish.

> [!info] Added later (2026-10-17, [What folds a world](what-folds-a-world.md))
> The bar's dimension mark does not tell a space from a small world: a B2
> world of 600 to 1,500 agents passes it because it is small
> ([What folds a world](what-folds-a-world.md)). Read with [What folds a world](what-folds-a-world.md)'s stronger
> test — the steps within which a typical agent's ball holds half the world,
> over the same for its [random twin](random-twin.md "A network with the same agents and as many links each, wired at random: what a world would be if it were no space at all.") — none of the 23 candidates is a space:
> they read 1.13 to 1.92, most below 1.5, as folded as B2 (1.18), where B1
> reads 2.3 at 1,500 agents. Of the 388 worlds read at iteration 1,000, those
> that are spaces and not poor are B1's, those whose every unused link lapses
> (rich, but about two-dimensional, and missing three or four marks), and a
> few others: mixing with slow brain change (`Y/mix-mut05-s1`: 2.38, 44% at
> three tokens or fewer, every mark met at a dimension of 2.9, a diversity
> score of 0.48) and two worlds of the search (`Z/r1-15-s1015`: 3.40, 43%,
> five marks, 0.73; `Z/r1-22-s1022`: 3.42, 27%, three marks, 0.83).

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
  [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md)). Something between, a death
  that depends on what an agent does rather than on a fixed age, would let ages
  spread without stopping the world.
- **A healthy network is not yet a space** (added later): the candidates'
  networks are folded small worlds, and "almost three-dimensional" asks more
  than the bar's mark does ([What folds a world](what-folds-a-world.md)).
- **A world might be judged by this map** before it is judged by its evolution:
  a rule that moves worlds towards the top right is a candidate for the long
  runs and the lifetime contests. The batches now running will be placed on it
  as they finish.

**Related notes:** [Effective numbers](effective-number.md) ·
[Every setting](settings.md) ·
[How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md) ·
[A world that cannot break](a-world-that-cannot-break.md) ·
[How different are the agents?](how-different-are-the-agents.md) ·
[Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) ·
[Every world side by side](every-world-side-by-side.md) ·
[What folds a world](what-folds-a-world.md)

<!-- turns -->
---

[Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [What makes a world interesting](what-makes-a-world-interesting.md) →

> [!info]- Pages that link here (23)
> [What this notebook is about](what-this-notebook-is-about.md) · [Every setting](settings.md) · [B1 and B2: the two baselines](b1-and-b2.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [The random twin: is a world a space?](random-twin.md) · [What folds a world](what-folds-a-world.md) · [What holds a world's size](what-holds-a-worlds-size.md) · [How alike the agents are](how-alike-the-agents-are.md) · [What would count as open-ended evolution](open-ended-test.md) · [Rules in company](rules-in-company.md) · [Who inherits?](who-inherits.md) · [Links to strangers](links-to-strangers.md) · [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Eight switches at once](eight-switches-at-once.md) · [What makes a world interesting](what-makes-a-world-interesting.md) · [The health bar](health-bar.md) · [The diversity score](diversity-score.md) · [Brains mixed at birth, in long worlds](mixed-at-birth-ten-long-worlds.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [The candidates, played on and played again](the-candidates-played-on.md) · [Room and riches for the steadiest world](room-for-the-steadiest-world.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
