# Brains mixed at birth, in long worlds

> [!question] Questions of this page
> - Do [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") mixed at birth keep a world healthy and rich for 10,000
>   [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."), or was that the luck of a few [pilot](the-lab.md#pilot-worlds "A world made by hand to see whether an idea is worth an experiment, without a thesis or many seeds.") worlds?
> - What does a world of mixed brains look like after a long life: how many
>   [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions."), how rich, how well knit? And why?
> - Is that health a state a mixed world keeps, or one it passes through?
> - Over 50,000 iterations, does its size keep drifting, does it die out, or
>   does it end up as poor as [B2](b1-and-b2.md "The second baseline: B1 with the rules that keep its network from breaking and its agents from staking or breeding themselves to death.")?

## In short

Mixing a newborn's brain from its parent's and a [neighbour](the-world.md#the-network "An agent joined to another by a connection.")'s makes a B2 world
smaller, so each of its agents holds more [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes."). In ten worlds played for
10,000 iterations (Experiment 17), the [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") world had 11% of its agents at
three tokens or fewer; B2's worlds have 66% to 79%. The agents are rich
because they put off having children, but how long they put it off drifts, and
a world's size and health drift with it. Only five of the ten worlds met every
mark of the [health bar](health-bar.md "The seven marks a world's network must meet to count as healthy: one piece, mostly on loops, no fatal cut, no hub, three to twelve links per agent, a dimension of 2.3 to 4."), so the thesis was refuted. Played on to 50,000
iterations (Experiment 18), those five did not stay healthy: three died out,
the two that lived stayed large and fairly poor, and evolution faded.

## Mixed at birth: ten long worlds

<!-- experiment E17 -->

In B2, a newborn's brain is a copy of its parent's, changed now and then by
[mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype."). With brains **mixed at birth**, a newborn's brain comes from two
agents: its parent and a *mate*, one of the parent's neighbours drawn at
random. For each hidden unit of the brain, a fair coin decides whether the
newborn takes it from the parent or from the mate. B2's brain has a single
hidden layer of 16 units. A unit is taken whole: the weights that lead into
it, its bias, and the weights that lead out of it. So a unit that works
arrives with everything it needs. Then the new brain mutates as usual.

[Healthy and rich](healthy-and-rich.md) read every world the notebook had made
against the author's picture of a world that might keep evolving. The picture
has two halves: a network that is decentralised, resilient and nearly
three-dimensional, and agents that differ in almost every statistic. B2's
worlds meet six of the seven marks of the health bar, the notebook's test of a
healthy network. But their agents do not differ: three in four hold three
tokens or fewer. The worlds that came closest to both halves were those whose
brains are mixed at birth ([Brains mixed at birth, units that cost, an eye for
raids on kin](mixed-brains-costly-units.md)).

Those were pilots. Batch Q made two worlds with mixing, later batches combined
mixing with other rules, and the two worlds of batch Q were played on
([Healthy and rich](healthy-and-rich.md#candidates-and-long-runs)). Now mixing
is a setting of the [engine](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world."), [`mix_at_birth`](settings.md#mix_at_birth), off
unless a run turns it on. So the lab can run it, and for longer than any B2
world has lived in the lab: 10,000 iterations, five times as long as
Experiment 16.

### What the pilots found

When this plan was written (16 October 2026), the worlds read as follows, each
at the iteration named, as [Healthy and rich](healthy-and-rich.md) reads every
world. The [dimension](ball-dimension-and-curvature.md "How fast the number of agents within a distance grows with the distance: as its square in a plane, its cube in space.") is the ball dimension: how fast the number of agents
within a given number of steps grows with the steps. The health bar asks for
2.3 to 4.

| world | iteration | agents | median tokens | 3 tokens or fewer | marks of the health bar (of 7) | dimension |
|---|---|---|---|---|---|---|
| B2, ten worlds (Experiment 16) | 2,000 | 3,064–3,699 | 1–2 | 70–78% | 6 (one world 7) | 3.9–4.8 |
| the same worlds played on (pilots) | 9,500–10,000 | 2,887–3,661 | 1–2 | 66–79% | 6 | 4.3–4.5 |
| mixed, [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 1 (batch Q) | 2,000 | 908 | 5 | 37% | 7 | 3.5 |
| mixed, seed 2 (batch Q) | 2,000 | 659 | 11 | 20% | 7 | 3.5 |
| seed 1, played on | 2,500 | 1,380 | 3 | 52% | 7 | 3.4 |
| … and on | 5,500 | 1,037 | 6 | 38% | 7 | 3.8 |
| seed 2, played on | 4,500 | 514 | 13 | 15% | 7 | 3.6 |
| … and on | 7,500 | 883 | 8 | 19% | 7 | 3.9 |
| mixed, with gifts (batch T) | 1,000 | 1,541 | 4 | 39% | 6 | 4.1 |
| mixed, [estates](estate.md "The tokens a dead agent leaves, and who gets them.") by wealth (batch Y) | 1,000 | 445 | 8 | 31% | 6 | 3.4 |

Two things stand out. First, a mixed world is a quarter to a third of B2's
size for the same tokens, so each agent holds more. Second, the plain mixed
worlds meet the one mark B2 misses: a dimension below 4.

The two long runs do not go the same way, though. One swung towards B2 at
2,500 iterations and came back. The other grew richer and smaller up to 4,500
iterations, and then larger again. Two worlds cannot say which is the rule.

The pilots showed more than the table does. Mixing doubled the [circulation](token-flow.md#flows-that-go-round "The part of a flow that goes round in loops, as against the part that runs from sources to sinks.")
that [Gliders, and who helps whom](gliders-and-who-helps-whom.md) measures:
the part of the flow of tokens that goes round in [loops](bridges.md#loops "A path through the network that comes back to where it started without using a connection twice."). Three things have not
yet been read in the long worlds of this page: that circulation, the kinds of
behaviour of [How different are the agents?](how-different-are-the-agents.md),
and the flow modules of [Flow modules: the groups tokens stay
in](flow-modules.md).

### Ten worlds, ten thousand iterations

Experiment 17 runs B2 at 10,000 tokens with brains mixed at birth: ten seeds,
for 10,000 iterations each. The lab records each run as it records every run:
the graph statistics every 25 iterations, and a whole world every 500. The
whole worlds have two uses. [Eras](eras-and-contests.md "Brains of a world at two moments, its eras, played against each other: do the later ones win?") of a world can be played against each other
in contests. And each world can be read at any of those moments, as [Healthy
and rich](healthy-and-rich.md) reads every world.

B2's own worlds are not run again. Experiment 16 ran them to 2,000 iterations,
and their continuations to 10,000, played as pilots, are the comparison.

So we expected the ten worlds to do what the pilots did: stay small and rich,
and meet every mark of the health bar. A unit of a brain that works can also
spread from one line of descent to another. But one pilot world swung back
towards B2's poverty for a while, so the riches might not last. The thesis
drew two lines at iteration 10,000. In the median world, fewer than half the
agents should hold three tokens or fewer. And at least seven of the ten worlds
should meet every mark.

<!-- thesis E17 -->
> [!quote] The thesis of Experiment 17, written down before any of its runs existed
> **The claim.** Brains mixed at birth keep a world healthy and make it rich over a long life. In B2 with every newborn's brain mixed from its parent's and a neighbour's, at iteration 10,000, fewer than half the agents hold three tokens or fewer in the median world of ten, where three in four do in B2; and at least seven of the ten worlds meet every mark of the health bar.
>
> **Why it would be so.** In the pilots, the worlds whose brains were mixed at birth came closest of all to a world that is healthy and whose agents differ ([Meta X](healthy-and-rich.md)). The two worlds of batch Q and their continuations meet every mark of the health bar, at a dimension of 3.4 to 3.9, with 15% to 52% of their agents at three tokens or fewer: 37% and 20% at 2,000 iterations, 38% and 19% at 5,500 and 7,500. Mixed with gifts as well, 39%, and with estates shared by wealth, 31%, at 1,000 iterations, with six marks each. B2's own worlds hold 70% to 78% of their agents at three tokens or fewer at 2,000 iterations (Experiment 16) and 66% to 79% at 9,500 to 10,000 (the same worlds played on as pilots), and nine in ten of them meet six marks: their dimension is 4.2 to 4.8, above the bar's 4. A mixed world is a quarter to a third of B2's size, so its agents hold more each, and a unit of a brain that works can spread from one line to another. But one of the two continuations swung back towards B2's poverty at 2,500 iterations, half its agents at three tokens or fewer, before it recovered: it may not last.
>
> **It holds if:** At iteration 10,000, each world read from its last checkpoint as research/atlas/profile.py reads every world ([Meta X](healthy-and-rich.md)): the median of the ten worlds' shares of agents holding three tokens or fewer is below 0.5, and at least seven of the ten worlds meet all seven marks of Chapter [46](how-to-tell-a-world-that-keeps-evolving.md)'s health bar — one piece, a 2-core of at least 0.8, no agent whose loss cuts off more than 2% of the others, no link more than 1%, no agent joined to more than 5% of the others, 3 to 12 links per agent, a dimension of 2.3 to 4.
>
> **It fails if:** A median share of 0.5 or more, or fewer than seven worlds meeting every mark: mixing at birth does not keep a world rich and healthy over a long life.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

### What the ten worlds did

<!-- figure mixedlong/sizes -->
![Agents alive, world by world](../figures/mixedlong/sizes.svg)

**Agents alive, world by world.** Each orange line is one world of Experiment 17, its agents after each game (means over 25 iterations); the grey band is B2's ten worlds — their median, middle half and nine in ten. B2's worlds keep their size; the mixed ones swing between a few hundred and two thousand agents, and one died out at iteration 5,376.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each run's `stats.jsonl` (the lab's runs) and `series.jsonl` (B2's continuations).
>
> 1. Read the agents after each game (`nodes`, phase 2).
> 2. For B2, join each lab world (to 2,000) to its continuation (2,000 to 10,000).
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

B2's worlds fill up within 1,500 iterations to some 3,000 to 3,700 agents, and
stay there. The mixed worlds never settle. Each wanders, over thousands of
iterations, between a few hundred agents and two thousand. Seed 3 fell to 40
agents at iteration 2,543 and came back to more than 800. Seed 6 fell to 91
and grew back to nearly 2,000. Seed 9 grew to 1,424 agents at iteration 2,592,
then shrank for nearly three thousand iterations. It died out at 5,376, when
20 agents were left: the lab stops a world there.

<!-- figure mixedlong/poor -->
![Agents holding three tokens or fewer](../figures/mixedlong/poor.svg)

**Agents holding three tokens or fewer.** Each band is ten worlds — the median, the middle half and nine in ten — every 500 iterations: the share of their agents holding one, two or three tokens. A world that died out leaves the band when it does.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each run's whole worlds: the lab's archives (`archive/world_<iteration>.npz`) and the continuations' checkpoints (`ckpt_<iteration>.npz`).
>
> 1. Read every agent's tokens in each whole world kept at a multiple of 500 iterations.
> 2. Count those holding three or fewer, as a share of all.
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

The riches go with the size. In B2, three agents in four hold three tokens or
fewer at every moment of the 10,000 iterations. In the mixed worlds that share
falls from about 40% at iteration 1,000 to between 5% and 25% in the median
world after 3,000. But the worlds spread from none to two thirds.

With the tokens fixed, a world's riches *are* its size. If 152 agents share
10,000 tokens, they hold 66 each; if 2,257 agents share them, 4.4 each.

### Verdict: rich, but only half healthy

<!-- figure mixedlong/verdict -->
![Experiment 17 at iteration 10,000](../figures/mixedlong/verdict.svg)

**Experiment 17 at iteration 10,000.** Each dot is a world at its last checkpoint, read as [Healthy and rich](healthy-and-rich.md) reads every world. Left: the share of its agents holding three tokens or fewer, B2's ten worlds at 10,000 iterations beside the mixed ones. Right: how many of the seven marks of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)'s health bar each mixed world meets; the world that died out is read at its last checkpoint, with 20 agents.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each run's last checkpoint (`checkpoint.npz`; B2's continuations' `ckpt_010000.npz`) (`GraphOfLifeRuns/.pilots/e17_profile.json`).
>
> 1. Count the agents holding three tokens or fewer, as a share of all.
> 2. Measure the seven marks as `research/atlas/profile.py` does.
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

The thesis asked two things of the worlds at iteration 10,000, and got one.

- **Rich: yes, by far.** In the median world, 11% of the agents hold three
  tokens or fewer, against the thesis's limit of a half; B2's worlds hold 66%
  to 79%. Seven of the ten are below 25%. The two exceptions, seeds 1 and 5,
  at 65% and 63%, are the two largest worlds, with 2,221 and 2,257 agents.
- **Healthy: five worlds of ten, where seven were needed.** Seeds 2, 4, 5, 6
  and 10 meet every mark. Three small worlds miss one mark, the [hub](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."): no agent
  may be joined to more than 5% of the others. In seed 3's 152 agents, the
  best-connected agent is joined to 12 others, 7.9% of the world; in seed 8's
  173, to 17 (9.9%); in seed 7's 346, to 47 (13.6%). Seed 1 misses the
  dimension by a hair, 4.05 against the bar's 4. Seed 9 died out.

**Strictly, the thesis is refuted.** And yet five worlds of ten meet every
mark, where none of B2's ten does at 10,000 iterations: each of B2's misses
the dimension, at 4.3 to 4.5.

### Why a mixed world is rich

<!-- figure mixedlong/breeding -->
![Who could have a child, and who has one](../figures/mixedlong/breeding.svg)

**Who could have a child, and who has one.** Left: the share of the agents that could have a child — 11 tokens or more, since a child takes at least 10 and the parent keeps one, and fewer than 8 links. Right: the births of the next 50 iterations, per iteration, per hundred such agents, on a logarithmic axis. Each band is ten worlds every 500 iterations — the median, the middle half and nine in ten; B2's ten worlds are Experiment 16's to 2,000 iterations and the same worlds played on after.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each world kept whole every 500 iterations, and each run's births (`research/oee/checks/e17_breeding.py`, `GraphOfLifeRuns/.pilots/e17_breeding.json`).
>
> 1. In each whole world, count the agents with 11 tokens or more and fewer than 8 links.
> 2. Read the births of the 50 reproduction phases that follow, from the run's record.
> 3. Divide their mean per iteration by the agents counted.
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

Start with death. In every world of both kinds, an agent lives out its hundred
iterations: almost nobody dies young. A world's agents divided by its births
per iteration is how long its agents live on average. In the median stretch of
500 iterations it is 100, in B2 and in the mixed worlds alike. So every world
replaces itself one for one: a birth per hundred agents per iteration. A
world's size is not set by death. It is set by who has children.

Who can have a child? A child takes at least 10 tokens, and its parent keeps
one. And an agent at its valence, the 8 links its brain may hold at most,
cannot have one at all. So the agents that *could* have a child are those with
11 tokens or more and fewer than 8 links.

In B2, 1% to 6% of the agents could have a child, and they have one at a
median rate of 42 per hundred per iteration. An agent that can afford a child
has it within two or three iterations. In the mixed worlds, a third of the
agents could have a child (8% to 89%). They have one at a median rate of 3 per
hundred per iteration: about fourteen times less often.

That is why a mixed world is rich. B2's agents have a child as soon as they
can afford one. So the world fills up until hardly anyone can: tokens go into
children as fast as they come. The mixed worlds' agents put children off. So a
mixed world settles where many could have a child and few do, and each agent
holds more.

And it is why the size drifts. How eagerly an agent breeds is a matter of its
brain. Where the brains are eager, a world grows until few are able; where
they are reluctant, it shrinks until many are. The mixed worlds' eagerness
wanders, from under 1 to 10 per hundred, and the size follows it. Across the
181 moments, the two go together with a [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of 0.87, on logarithms.
B2's brains are eager in every world at every moment.

Why should mixing make agents put children off? A brain spreads in two ways:
through its agent's children, and through every node its agent takes in the
[game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."), which gets a copy of the winner's brain. Taking nodes is by far the main
way. In the mixed worlds, 7.7 nodes per hundred agents change hands in the
median iteration (3 to 18); in B2's, 5.6 (4 to 11). Against that stands one
birth per hundred agents.

Where brains are mixed at birth, a child carries only about half of its
parent's brain, while a node taken carries all of it. A brain that spends its
tokens on children spreads less of itself than one that stakes them on its
neighbours' nodes. And the mate's half spreads for free, whether the mate ever
breeds or not ([What holds a world's
size](what-holds-a-worlds-size.md#why-mixing-makes-agents-reluctant)). So
selection, which keeps working in these worlds (as the contests
[below](#does-evolution-go-on) show), may come to favour putting children off.

It is not simply that the reluctant fight more, though. Within the mixed
worlds, the share of nodes that change hands does not rise as a world grows
small and rich: its correlation with the world's size is −0.08.

<!-- figure mixedlong/switch -->
![Mixing switched at a moment](../figures/mixedlong/switch.svg)

**Mixing switched at a moment.** Each solid line is a world played on as a pilot from the moment the lab had kept it whole, with mixing at birth switched; each dashed line is the same world played on unchanged — the lab's own run of Experiment 17, or B2's continuation. Agents after each game, means over 25 iterations. Batch SW: mixing off in the five worlds of Experiment 17 that met every mark, at iteration 5,000; on in five of B2's worlds, at 2,000.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`). Batch SW: `GraphOfLifeRuns/.pilots/SW/mixoff-sNNN` and `mixon-sNNN` (`research/oee/pilots/make_jobs_TU.py`).
>
> **Data.** From each pilot's `series.jsonl`; the lab runs' `stats.jsonl`.
>
> 1. Start from the lab's whole world at the moment (`run.py --from RUN:ITERATION --set mix_at_birth=…`) and play 2,000 iterations.
> 2. Read the agents after each game in the pilot and in the run it was taken from.
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

Whatever the reason, switching the rule shows that mixing is what does it.
Batch SW took worlds from the lab at a moment, and played them on for 2,000
iterations with the rule switched. All ten went the way the rule says:

- **Without mixing from iteration 5,000**, all five of the worlds that met
  every mark grew: from 893, 744, 2,039, 395 and 1,325 agents to 3,341, 2,797,
  3,243, 2,137 and 2,082, on their way to B2's size. The same worlds with
  mixing stayed far smaller.
- **With mixing from iteration 2,000**, all five of B2's worlds shrank: from
  3,064 to 3,535 agents to between 1,422 and 2,156 — and one, seed 4, to 81,
  nearly dying out. The same worlds without it stayed above 3,000.

How eagerly the able agents breed changed first. Without mixing, it rose
within 1,500 iterations in four of the five worlds: from 1.5–7 births per
hundred per iteration to 14–35. In seed 10 it went from 5 to 10 and back to 5.
With mixing, in B2's worlds, it fell from 26–51 to 4–10. A rule that only five
generations of agents have lived under sets how eagerly they breed, and the
size of the world follows.

### The marks a world misses

<!-- figure mixedlong/size -->
![The marks a world misses, against its size](../figures/mixedlong/size.svg)

**The marks a world misses, against its size.** Each orange dot is a world of Experiment 17 at one of the moments it was kept whole, every 500 iterations; each grey dot a world of B2 at 10,000 tokens — Experiment 16's ten at 2,000 iterations, and the same worlds played on, at 10,000 or more — and each blue one a world of B2 at 5,000 tokens, Experiment 14's ten at 1,000 iterations. Across, on a logarithmic axis: the agents alive. Left: the dimension the network shows by ball growth; right: the share of the others the best-connected agent is joined to.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`). B2 at 5,000 tokens: Experiment 14's ten runs (`B2-5000-s001` … `s010`).
>
> **Data.** From `GraphOfLifeRuns/.pilots/e17_health.json` (`research/oee/checks/e17_health.py`) and `GraphOfLifeRuns/.pilots/atlas/profile.json`.
>
> 1. Read each whole world as `research/atlas/profile.py` reads every world: the dimension by ball growth, the best-connected agent's links over the others.
> 2. Set each against the number of agents alive.
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

Read every 500 iterations, the ten worlds met every mark at 125 of their 190
moments: two in three. They missed only two marks (and one other once), and
each follows the world's size:

- **Small worlds miss the hub.** Below 300 agents, a world missed it at 96% of
  its moments: a best-connected agent with 12 to 47 links is more than 5% of a
  small world.
- **Large worlds miss the dimension.** The dimension the network shows by ball
  growth rises with the number of agents (a correlation of 0.78 with its
  logarithm). Above 1,500 agents it is near 4 or above.

Worlds of 600 to 1,000 agents met every mark at 98% of their moments.

B2 follows the same rule. Its worlds of 10,000 tokens hold 3,000 to 3,700
agents, and all but one miss the dimension, at 4.2 to 4.8. Its worlds of 5,000
tokens ([Experiment 14](a-world-that-cannot-break.md)) hold 1,200 to 1,800
agents, and six of the ten meet every mark. They are as poor as ever: three
agents in four hold three tokens or fewer.

So half of what the thesis asked, every mark of the health bar, was a question
of size. Mixing makes a world smaller, and at that size B2's network meets the
bar. What mixing changes and size does not is how rich the agents are: B2's
worlds of 5,000 tokens are as poor as those of 10,000.

Why does a larger world read a larger dimension? The measure is not to blame.
In [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds, which are true spaces (the agents within *r* steps grow as a
power of *r*), the dimension does not grow with size: from 40 agents to 70,000
it stays between about 2.5 and 4. [What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)
found the reason in B2's network. It is nearly a small world, where every
agent is a few steps from every other, with no dimension of its own to show.
Every B2 world starts as a space and folds once its first generation dies,
each death mended with a ring of links. A larger world then reads a larger
dimension off the same few steps. Brains mixed at birth fold a world the same
way.

The rise is not a law of every world built on B2, though. It holds for B2
itself, at 5,000 and 10,000 tokens, and for its mixed worlds. Other rules move
the dimension at any size. Leave the estate of the dead to the newborns, or to
the parents, and B2 holds more than 5,000 agents at a dimension of 3.4 to 3.9,
and meets every mark ([Who inherits?](who-inherits.md#healthy-and-diverse)).

### Does evolution go on?

<!-- figure mixedlong/evolution -->
![Do later brains beat earlier ones?](../figures/mixedlong/evolution.svg)

**Do later brains beat earlier ones?** Each dot is a world: the share of a world the later era's brains hold after 100 iterations against the earlier era's, the mean of two contests, one in each era's world; the bar is the median world. Orange: the mixed worlds of Experiment 17; grey: B2 — its ten worlds for 1,000 against 500, three of their continuations for the later pairs.

> [!example]- How to make this figure
> **Runs.** The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).
>
> **Data.** From each run's whole worlds every 500 iterations (`GraphOfLifeRuns/.pilots/e17_T100.json`, `long_steps.json`, `long_rejudge_T100.json`).
>
> 1. Seat the brains of the two eras on alternate agents of each era's world and play it on for 100 iterations, an agent's lifetime (`research/oee/checks/rejudge.py`).
> 2. Count the agents the later era's brains hold at the end, as a share of all.
>
> **To make it again:** `python3 book_figures.py mixedlong`.
<!-- /figure -->

Yes, and at first faster than in B2. In each world, the brains of a later
moment met those of an earlier one, as [Why evolution
stops](how-to-tell-a-world-that-keeps-evolving.md#would-we-notice-if-it-went-on) did for B2. Each
contest lasts an agent's [lifetime](age-and-lifetime.md "How many iterations an agent lives, from its birth to its death."), 100 iterations, and is played once in each
era's world. A newborn whose brain is mixed counts for its parent's era.

In the median world, the brains of iteration 1,000 took 84% of a contested
world from those of 500; B2's took 68%. They took more than half in every one
of the ten worlds (72% to 91%). Those of 5,000 took 66% from those of 2,000
(B2: 63%, in three worlds). Those of 10,000 took 54% from those of 5,000 (B2:
55%), more than half in seven worlds of nine.

So mixing does not stop evolution. The steps shrink with time, as they do in
B2, and the first ones are larger. The worlds differ more, though: from 41% to
95% for the last pair. One world's step is uncertain on its own: a contest of
two eras varies by about a tenth from one play to the next. An era of seed 4
played against itself gave 40% and 64%. The median of ten is not uncertain in
that way.

How the brains are counted matters here. The lab's contests told the two eras
apart by the number each brain carries. Mixing makes new brains even where
mutation is switched off, and the numbers those new brains drew counted for
the earlier era. So the first count of these contests said that evolution
stops in mixed worlds. Counted by the parent's era, it goes on.

## Five mixed worlds, fifty thousand iterations

<!-- experiment E18 -->

Whatever else a world that keeps evolving does, it has to *last*. A world that
meets the author's picture for 10,000 iterations and then shrinks to nothing,
or fills up with agents holding a token each, is not a world that keeps
evolving. The author asked for the most promising settings to be run for a
long time, and this is that run. ([A hundred thousand
iterations](a-hundred-thousand-iterations.md) plans a single world twice as
long.)

The first 10,000 iterations left that open. The mixed worlds stayed rich, but
their size drifted, and their health followed it. Small worlds missed the hub
mark and large ones the dimension, while worlds of 600 to 1,000 agents met
every mark at 98% of their moments ([The marks a world
misses](#the-marks-a-world-misses)). One world in ten died out, and two others
fell below a hundred agents before they came back ([What the ten worlds
did](#what-the-ten-worlds-did)). So whether a mixed world stays healthy and
rich depends on where its drift takes it. Ten thousand iterations showed each
world wandering through a range of sizes. Fifty thousand will show whether
that range has edges.

### Five worlds played on

Experiment 18 takes the five worlds of Experiment 17 that met every mark of
the health bar at iteration 10,000: seeds 2, 4, 5, 6 and 10. It plays them on
to iteration 50,000. They are the same runs. The lab goes on from the world
each run saved at iteration 10,000, its checkpoint, with the engine that made
it. It records as before: the graph statistics every 25 iterations, and a
whole world every 500. Nothing is changed, so their first 10,000 iterations
are those of the [section above](#mixed-at-birth-ten-long-worlds). Each world
is read every 500 iterations from the whole worlds the lab keeps, as [Healthy
and rich](healthy-and-rich.md) reads every world.

When the plan was written (16 October 2026), the five worlds stood like this
at iteration 10,000:

| world | agents | median tokens | 3 tokens or fewer | moments with every mark, of 20 (500 to 10,000) | its fewest and most agents, 500 to 10,000 |
|---|---|---|---|---|---|
| seed 2 | 744 | 11 | 6% | 16 | 337–1,437 |
| seed 4 | 628 | 11 | 20% | 17 | 337–1,371 |
| seed 5 | 2,257 | 2 | 63% | 6 | 1,032–2,257 |
| seed 6 | 1,302 | 4 | 42% | 11 | 188–1,688 |
| seed 10 | 618 | 10 | 15% | 16 | 296–1,649 |

We expected the five worlds to stay mostly healthy and rich, because health
came back. Over the first 10,000 iterations, a world that met every mark still
met them 500 iterations later four times in five. A world that missed a mark
met them all again a third of the time. But forty thousand iterations more is
time for each of the five to drift through every size. It might grow small
enough to miss the hub, or large and poor enough to miss the dimension.

<!-- thesis E18 -->
> [!quote] The thesis of Experiment 18, written down before any of its runs existed
> **The claim.** Over a long life, a world whose brains are mixed at birth stays mostly healthy and rich. The five worlds of Experiment 17 that met every mark of the health bar at iteration 10,000, played on to 50,000, meet every mark at half or more of the moments they are kept whole, and at 50,000 at most one of them has half its agents or more holding three tokens or fewer.
>
> **Why it would be so.** Over Experiment 17's 10,000 iterations, these five worlds met every mark at 66 of the 100 moments they were kept whole (every 500 iterations from 500 on): 16, 17, 6, 11 and 16 of 20 each. A world that met every mark still did 500 iterations later four times in five (95 of 120 times, in all ten worlds), and one that missed a mark met them all again a third of the time (21 of 60): health comes back. At iteration 10,000 four of the five had fewer than half their agents at three tokens or fewer (6%, 20%, 42%, 15%) and one had 63%, where B2's worlds have 66% to 79% at every moment. But a mixed world's size drifts, and its health and its riches follow its size: worlds of 600 to 1,000 agents met every mark at 98% of their moments, worlds below 300 at 4% (their best-connected agent is joined to more than 5% of the others), worlds above 1,500 at 35% to 60% (their dimension rises above 4). And with 10,000 tokens, a world of 2,000 agents is a poor one. In ten worlds one shrank below 20 agents and died out, at iteration 5,376; two others fell below a hundred agents, to 40 and to 91, and came back. Forty thousand iterations more is time for every one of the five to drift through those sizes.
>
> **It holds if:** From iteration 10,500 to 50,000, each world read every 500 iterations from its archive as research/atlas/profile.py reads every world ([Meta X](healthy-and-rich.md)): at half or more of the five worlds' 400 moments together, a world meets all seven marks of Chapter [46](how-to-tell-a-world-that-keeps-evolving.md)'s health bar (a world that has died out misses them at every moment after); and at iteration 50,000, at most one of the five worlds has died out or has half its agents or more holding three tokens or fewer.
>
> **It fails if:** Fewer than half the moments meet every mark, or two or more of the five worlds dead or with half their agents or more at three tokens or fewer at 50,000: the health and the riches of a mixed world do not last.
>
> *Written while the notebook still numbered its chapters: each number is a link to the page it meant that day.*
<!-- /thesis -->

### What the five worlds did

<!-- figure longmixed/sizes -->
![Agents alive, world by world](../figures/longmixed/sizes.svg)

**Agents alive, world by world.** Each line is one world, its agents after each game (means over 100 iterations), from its birth in Experiment 17 to iteration 50,000 or to the game after which 20 agents or fewer were left, where the lab stops a world.

> [!example]- How to make this figure
> **Runs.** The five runs of Experiment 18 — Experiment 17's worlds of seeds 2, 4, 5, 6 and 10 (B2 at 10,000 tokens with brains mixed at birth), played on by the lab from iteration 10,000 to 50,000 or until they died out (`python3 gol_lab.py run E18`); their first 10,000 iterations are Experiment 17's.
>
> **Data.** From each run's `stats.jsonl`.
>
> 1. Read the agents after each game (`nodes`, phase 2).
> 2. Average over stretches of 100 iterations.
>
> **To make it again:** `python3 book_figures.py longmixed`.
<!-- /figure -->

Three of the five worlds died out:

- **Seed 10** grew from 620 agents to 1,450 by iteration 13,556, and died out
  at 17,078. In its last thousand iterations it fell from 1,168 agents to 19.
- **Seed 2** wandered between 230 and 740 agents for eight thousand
  iterations. It rose to 989 at iteration 20,874, and was gone 413 iterations
  later, at 21,288.
- **Seed 4** grew to 2,191 agents by iteration 21,260, and stayed near two
  thousand until 28,000. Then it fell to 123 by 34,000, came back to 345, and
  died out at 37,722.

Two worlds lived to 50,000, and stayed large. Seed 5 already held more than
2,000 agents at iteration 10,000, and seed 6 grew past 2,000 within a few
thousand iterations. Over their last 20,000 iterations, both held between
1,262 and 2,252 agents, and 37% to 61% of their agents held three tokens or
fewer: between the mixed worlds' riches and B2's poverty.

### Verdict: refuted on both counts

**The thesis is refuted, on both counts.** It asked that the worlds meet every
mark at half or more of their moments from 10,500 to 50,000. A dead world
counted as missing them all. They met them at 115 of 400, 29%. And it allowed
at most one world dead or with half its agents poor at 50,000. Three had died
out, and seed 5 ended with 56% of its agents at three tokens or fewer. Only
seed 6 ended alive with fewer than half its agents poor, at 43%.

### Health and riches

<!-- figure longmixed/health -->
![Health and riches, every 500 iterations](../figures/longmixed/health.svg)

**Health and riches, every 500 iterations.** Each world read as [Healthy and rich](healthy-and-rich.md) reads every world, from the whole worlds the lab kept every 500 iterations from 10,500 on, while it lived. Left: how many of the seven marks of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)'s health bar it meets (the dots of each world are set a little apart so that they do not hide each other). Right: the share of its agents holding three tokens or fewer.

> [!example]- How to make this figure
> **Runs.** The five runs of Experiment 18 — Experiment 17's worlds of seeds 2, 4, 5, 6 and 10 (B2 at 10,000 tokens with brains mixed at birth), played on by the lab from iteration 10,000 to 50,000 or until they died out (`python3 gol_lab.py run E18`); their first 10,000 iterations are Experiment 17's.
>
> **Data.** From each run's `archive/world_<iteration>.npz` (`research/oee/checks/e17_health.py --seeds 2,4,5,6,10 --from 10500 --to 50000`).
>
> 1. Read each whole world as `research/atlas/profile.py` reads every world.
> 2. Count the marks it meets and its agents at three tokens or fewer.
>
> **To make it again:** `python3 book_figures.py longmixed`.
<!-- /figure -->

The marks a world misses follow its size, as they did in the first 10,000
iterations ([The marks a world misses](#the-marks-a-world-misses)). The worlds
that died missed the hub mark while they were small. The two that lived missed
a mark at 50 of their 80 moments each, and it was always the dimension. They
held 1,300 to 2,300 agents, at a dimension of 3.8 to 4.3: just around the
bar's 4. The riches follow the size too: the dead worlds were rich to the end,
the living ones poorer the larger they grew.

### How a mixed world dies

<!-- figure longmixed/collapse -->
![The worlds that died out](../figures/longmixed/collapse.svg)

**The worlds that died out.** Each line is a world that died out, read every 500 iterations up to its end: left, the share of its agents that could have a child (11 tokens or more and fewer than 8 links); right, how often they had one in the next 50 iterations, per hundred of them. Every agent dies at 100, so a world whose agents all could have a child dies out once they have fewer than one per hundred an iteration.

> [!example]- How to make this figure
> **Runs.** The five runs of Experiment 18 — Experiment 17's worlds of seeds 2, 4, 5, 6 and 10 (B2 at 10,000 tokens with brains mixed at birth), played on by the lab from iteration 10,000 to 50,000 or until they died out (`python3 gol_lab.py run E18`); their first 10,000 iterations are Experiment 17's.
>
> **Data.** From each run's whole worlds and births (`research/oee/checks/e17_breeding.py --seeds 2,4,10 --from 10000 --to 50000`).
>
> 1. In each whole world, count the agents with 11 tokens or more and fewer than 8 links.
> 2. Divide the births per iteration of the next 50 iterations by them.
>
> **To make it again:** `python3 book_figures.py longmixed`.
<!-- /figure -->

Not by a sudden blow. Every agent of these worlds lives out its hundred
iterations. So a world keeps its size only while one agent in a hundred is
born every iteration ([Why a mixed world is
rich](#why-a-mixed-world-is-rich)). The worlds that died had hovered near that
line for thousands of iterations. Then they fell below it, and stayed below.
In the last world kept before each end, births had fallen to 0.39, 0.51 and
0.47 per hundred agents an iteration. Yet a third to three quarters of the
agents could have had a child. In seed 10's last world, 31 agents were left,
every one of them able to have a child, and none had one. Their ages were
spread evenly. This was not a generation dying at once, but a world that
stopped replacing its dead.

### Did evolution go on?

<!-- figure longmixed/contests -->
![Later brains against earlier ones](../figures/longmixed/contests.svg)

**Later brains against earlier ones.** Each dot is a contest between two of a world's moments: the brains of the later moment against those of the earlier, half and half on the same world, for 100 iterations, in each moment's world with two seeds; the bar is the median of the four. Above: the pairs the plan named, for the worlds alive then. Below: the brains of the last world kept before each collapse that still had a few hundred agents, against those of 5,000 iterations before.

> [!example]- How to make this figure
> **Runs.** The five runs of Experiment 18 — Experiment 17's worlds of seeds 2, 4, 5, 6 and 10 (B2 at 10,000 tokens with brains mixed at birth), played on by the lab from iteration 10,000 to 50,000 or until they died out (`python3 gol_lab.py run E18`); their first 10,000 iterations are Experiment 17's.
>
> **Data.** From `GraphOfLifeRuns/.pilots/e18_T100.json` (`research/oee/checks/rejudge.py --seeds 17,18`).
>
> 1. Seat the brains of the two moments on alternate agents of each moment's world and play 100 iterations, an agent's lifetime.
> 2. Count the agents the later moment's brains hold at the end, as a share of all.
>
> **To make it again:** `python3 book_figures.py longmixed`.
<!-- /figure -->

**Hardly.** In the pairs the plan named, the brains of 25,000 beat those of
10,000 in seed 6 (72% of a contested world, the median of four contests)
and narrowly in seed 5 (56%). In
seed 4 they did no better (45%). The brains of 50,000 took 54% from those of
25,000 in seed 5, and lost to them in seed 6 (42%). Over the first 10,000
iterations, the mixed worlds' later brains had won 84%, then 66%, then 54%
([Does evolution go on?](#does-evolution-go-on)). The steps shrink towards
nothing: in the long run, a later era is as likely to lose as to win.

Before the collapses, the contests differ from world to world. In seed 10, the
brains of the last good years took 86% of a contested world from those of
5,000 iterations before. The brains that let the world die out were winners:
selection itself carried it to its end. Biologists call this **evolutionary
suicide**: selection that drives a trait to where the population cannot last.
In seed 4 the last brains lost to their forebears (21%), and in seed 2 they
tied (48%). One world in three is not a law. It is enough to say that a mixed
world can be led to [extinction](extinction.md "A world's end before its last iteration: it has dropped to its threshold of agents or fewer.") by the very brains that win in it.

## What we learned

- **A rich world, here, is a small one.** With the tokens fixed, how rich the
  agents are is how few they are. Mixing brains at birth made the worlds rich
  by making their agents slow to have children. In B2 an agent that can afford
  a child has one within two or three iterations; in a mixed world, after
  thirty or so. Switch mixing off, and the riches go within a thousand
  iterations. Switch it on in B2, and its worlds shrink at once, one of them
  nearly to nothing.
- **Nothing holds the size.** How eagerly agents breed drifts as their brains
  change, and the size, the riches and the marks of health drift with it. The
  riches of a world that keeps evolving would need something to hold them: a
  rule, or something the agents keep up themselves. Batch M tried each rule of
  the engine that might hold a mixed world's size: the estates of the dead to
  the newborns, to the parents or to the poorest; half of an agent's tokens
  kept at home; links to strangers; and a slower change of brains. Those that
  held a world steady held it poor ([What holds a world's
  size](what-holds-a-worlds-size.md#what-would-hold-a-worlds-size)).
- **A mixed world is mortal.** One world of ten died out within 10,000
  iterations, and three of the five that had met every mark at 10,000 died out
  before 40,000. Each died the same way: births fell below the one per hundred
  agents that replaces the dead, while many of its agents could have afforded
  a child. The two that lived stayed large and fairly poor, and missed the
  dimension mark at most of their moments. They ended closer to B2 than to the
  worlds that came closest in [Healthy and rich](healthy-and-rich.md).
- **Evolution goes on, then fades.** Mixing does not stop selection: later
  brains beat earlier ones as they do in B2, and by more at first. Then the
  steps shrink. By 25,000 to 50,000 iterations, an era is as likely to lose as
  to win against the one before it. In one world, the brains that won were
  those that let it die.
- **Two marks of the health bar measure size.** In B2 and its mixed worlds,
  the dimension fails a large world and the hub a small one. Networks are best
  compared at one size.
- **A count has to follow every way its labels are made.** Mixing makes new
  brains even without mutation. Counted by the number each brain carries, the
  contests said evolution stops in mixed worlds. Counted by the parent's era,
  it goes on.

## Where to go next

- [What holds a world's size](what-holds-a-worlds-size.md): what sets how many
  agents a world holds, why mixing loosens it, and what might hold it.
- [Mixed at birth, with parents as
  mates](mixed-at-birth-with-parents-as-mates.md): Experiment 19 takes the
  mate's free ride away, by letting only parents be mates. Its ten worlds all
  lived to 20,000 iterations, richer than B2's but not rich.
- [What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world): why B2's network reads a larger
  dimension the larger it grows, and what keeps a world a space.
- [Healthy and rich](healthy-and-rich.md): every world of the notebook against
  the author's picture of a world that might keep evolving.

## How this was made

The figures of [the first section](#mixed-at-birth-ten-long-worlds) read the
ten runs of Experiment 17. Beside them they read B2's own worlds: Experiment
16's ten, played on as pilots to 10,000 iterations, and a few others that each
figure names. `python3 book_figures.py mixedlong` draws them. The figures of
[the second section](#five-mixed-worlds-fifty-thousand-iterations) read the
five runs of Experiment 18, and `python3 book_figures.py longmixed` draws
them. Each figure says, folded beneath it, which runs it reads and how. A lab
run records every game, so the plans reckoned with about 9 GB of disk for
Experiment 17's ten runs, and 16 GB more for Experiment 18's five.

### The runs of Experiment 17

<!-- runs E17 -->
> [!info]- The runs behind this page
> **10 runs.** Every condition below is run once for every seed (1 to 10), for 10,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **mixed at birth** — changes `mix_at_birth` = on; 10,000 tokens. Runs `B2-10000-4af4a3-s001` … `B2-10000-4af4a3-s010`.
>
> To make these runs again: `python3 gol_lab.py run E17`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | mixed at birth | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 1 to 10 | one run per seed |
> | iterations | 10,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

### The runs of Experiment 18

These are five of Experiment 17's runs, played on from their checkpoints at
iteration 10,000 with the engine that made them. So they keep their names.

<!-- runs E18 -->
> [!info]- The runs behind this page
> **5 runs.** Every condition below is run once for every seed (2, 4, 5, 6, 10), for 50,000 iterations — or until its world dies out. Every setting is that of the baseline **B2** (a world that cannot break) unless the condition changes it.
>
> - **mixed at birth** — changes `mix_at_birth` = on; 10,000 tokens. Runs `B2-10000-4af4a3-s002` … `B2-10000-4af4a3-s010`.
>
> To make these runs again: `python3 gol_lab.py run E18`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.

> [!info]- Every setting of these runs
> | setting | mixed at birth | what it is |
> |---|---|---|
> | [`total_tokens`](settings.md#total_tokens) | 10,000 | The number of tokens in the world, *T*. It never changes during a run. |
> | [`n_nodes`](settings.md#n_nodes) | 0 | How many founders the world starts with. 0 means one founder per hundred tokens: *n* = ⌊*T* / 100⌋. |
> | [`k_neighbors`](settings.md#k_neighbors) | 0 | How many neighbours each founder starts with in the ring. 0 means *k* = max(⌊*n* / 100⌋, 5); an odd *k* is wired as *k* − 1. |
> | [`rewire_p`](settings.md#rewire_p) | 0.2 | In the starting ring, the probability that a connection is moved to a founder chosen at random (Watts–Strogatz). |
> | [`hidden_layers`](settings.md#hidden_layers) | 16 | The widths of the brain's hidden layers, in order from input to output. |
> | [`brain_kind`](settings.md#brain_kind) | float16 | How weights are stored: `float` (64-bit), `float16` (16-bit, computed in 64-bit) or `binary` (−1, 0, +1). |
> | [`brain_bits`](settings.md#brain_bits) | 16 | Only for binary brains: how many input rows encode one number. Unused by float and float16 brains. |
> | [`message_amount`](settings.md#message_amount) | 8 | How many numbers one message holds. Each agent sends one message to itself and one to each neighbour, every phase. |
> | [`random_input_amount`](settings.md#random_input_amount) | 5 | How many random numbers, drawn uniformly from −2 to 2, a brain reads per neighbour, every time it looks. |
> | [`exchange_messages`](settings.md#exchange_messages) | on | Whether agents send and read messages at all. |
> | [`message_prepass`](settings.md#message_prepass) | off | Whether every phase begins with an extra look in which agents only write messages, so the look that acts reads messages written this phase. |
> | [`allow_handover`](settings.md#allow_handover) | off | Whether a parent may move some of its own connections to its newborn child. |
> | [`allow_revolutions`](settings.md#allow_revolutions) | on | Whether a coalition of smaller stakers can take a node from its largest staker (see the note *How a coalition takes a node*). |
> | [`allow_gifting`](settings.md#allow_gifting) | off | Whether agents may give tokens to neighbours during reproduction. Off in both baselines; on in a few pilots, in Experiment 24 and in half the worlds of the screen. |
> | [`random_decisions`](settings.md#random_decisions) | off | The control: every number a brain would produce is replaced by a random draw from the standard normal distribution. |
> | [`prune_after`](settings.md#prune_after) | blotto | After which phase connections that carried no tokens are cut: `blotto` (the game), `reproduction`, or `both`. |
> | [`inactive_window`](settings.md#inactive_window) | phase | How long a connection may go unused: `phase` means it must carry tokens in the phase being judged; `iteration` allows the two last phases. |
> | [`redistribution`](settings.md#redistribution) | uniform | How the tokens of removed agents are shared: `uniform` gives every survivor the same chance at each token; `by_tokens` weights by what a survivor holds; `poorest` raises the poorest to one common level first; `newborns` gives them to this iteration's children, `parents` to their parents, once per child (both `uniform` in an iteration without births); `children` leaves each dead agent's estate to its own living children (`uniform` where there are none). |
> | [`tokens_created_per_phase`](settings.md#tokens_created_per_phase) | 0 | Tokens added to the world at every cleanup. 0 keeps the supply fixed. |
> | [`mutate_on_replication`](settings.md#mutate_on_replication) | on | Whether a brain changes only when it is copied — into a newborn, or into a node its owner has just won — instead of also after every game, wherever it sits. |
> | [`incumbency`](settings.md#incumbency) | 1.5 | How many times over an agent's stake on its own node counts when the game decides who takes the node. 1 is the old rule. |
> | [`birth_links`](settings.md#birth_links) | 2 | How many of its parent's neighbours a newborn is joined to, drawn at random, besides its parent itself; 0 leaves every link to the brain's choice. |
> | [`mend_on_death`](settings.md#mend_on_death) | on | Whether the neighbours of an agent that dies are joined up where its death would leave them in separate pieces. |
> | [`prune_needs`](settings.md#prune_needs) | 2 | How many neighbours the two ends of an unused connection must share for it to be cut; 0 cuts every unused connection. |
> | [`valence`](settings.md#valence) | 8 | The most connections an agent's brain may choose to hold (0 = no limit). An agent at its limit cannot have a child. |
> | [`keep_home`](settings.md#keep_home) | 0.3 | The share of an agent's tokens — rounded up, never less than one — that stays on its own node whatever its brain says; a parent also keeps one. 0 is the old rule. |
> | [`max_age`](settings.md#max_age) | 100 | The age at which an agent dies (0 = never); its tokens are shared out like those of any other dead agent. |
> | [`buy_life`](settings.md#buy_life) | off | Whether an agent that has reached max_age may live on by paying one token an iteration into the estate of the dead — never its last, so one with a single token left dies. Needs max_age. |
> | [`min_child`](settings.md#min_child) | 10 | The fewest tokens a child can be born with; a parent that would give less has no child. |
> | [`child_floor`](settings.md#child_floor) | fixed | The least a child may start with: min_child alone (fixed), or also the median holding of its parent's neighbours (neighbours), so that a world grows only as rich as it is. |
> | [`mix_at_birth`](settings.md#mix_at_birth) | **on** | Whether a newborn's brain is its parent's mixed with a mate's — one of the parent's neighbours, drawn at random — each hidden unit whole from one or the other by an even chance, before it mutates; off, a newborn has a copy of its parent's brain. |
> | [`mix_mates`](settings.md#mix_mates) | any | Under mix_at_birth, who may be the mate: any neighbour of the parent, or only parents — a neighbour with a living child of its own (with none, the newborn's brain is a copy). |
> | [`self_message`](settings.md#self_message) | own | What an agent writes to itself, which each neighbour reads: what its own column says (own), or the largest, number by number, of what it writes to anyone (largest), so that news one column marks reaches every neighbour. |
> | [`link_search`](settings.md#link_search) | off | Whether an agent may reach beyond its neighbours to propose a link: off; jump, a walk as many steps long as the tokens it spends in one iteration, with a link proposed where it ends; walk, a scout that goes one step further in every iteration it pays for; or reach, a jump whose steps are free, as many as the tokens it holds would pay for. A paid step costs one token, paid to the agent it enters, and never the last. |
> | [`link_steering`](settings.md#link_steering) | random | Which way each step of a search goes: to a neighbour drawn at random, or to the one the searcher would most like a link with (looking). |
> | [`link_consent`](settings.md#link_consent) | mutual | Who has to agree to a link a search proposes: both ends (mutual), or only the one who proposes (forced). Nobody at its valence gets a new link either way. |
> | [`proto`](settings.md#proto) | none | Rules first played in the pilots' prototype, by name: came_from and helped_by (an input row each), contest = lottery, pool by genotype or family, kin_sense, tag_rate, mate_pays, grow, unit_token, link_token. Empty in every run but those that ask for one; only the prototype's world plays them (see the settings note). |
> | [`mutation_probability`](settings.md#mutation_probability) | 0.2 | The probability that a brain changes when it is copied to a child, and again, for every brain, after every game. |
> | [`mutation_noise_std`](settings.md#mutation_noise_std) | 0.2 | How large a change to one weight is: a normal draw with standard deviation this times 1/√(fan-in) of its layer. |
> | [`mutation_sparsity`](settings.md#mutation_sparsity) | 0.1 | The share of a brain's numbers a change touches; also the probability, for each weight matrix and bias vector, of a rarer reset that redraws that share of it. |
> | [`extinction_threshold`](settings.md#extinction_threshold) | 20 | A run stops, as extinct, when an iteration ends (after its game) with this many agents or fewer. |
> | seeds | 2, 4, 5, 6, 10 | one run per seed |
> | iterations | 50,000 | how far each run goes, unless its world dies first |
>
> A value in **bold** differs from the baseline B2.
<!-- /runs -->

## How our view changed

- **17 October 2026.** This page said that B2's worlds have the network of the
  author's picture, decentralised, resilient and nearly three-dimensional, and
  lack only the agents. [What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world) found that
  B2's network is nearly a small world, with no dimension of its own. It
  starts as a space, and folds once its first generation dies, each death
  mended with a ring of links. The dimension the health bar reads off its
  first few steps grows with the world's size.
- **17 October 2026.** This page said that the dimension's rise with size
  belongs to "B2's kind of network". [Who
  inherits?](who-inherits.md#healthy-and-diverse) showed that this says too
  much. The rise holds for B2 itself, at 5,000 and 10,000 tokens, and for its
  mixed worlds. But other rules move the dimension at any size. B2 with the
  estate of the dead left to the newborns, or to the parents, holds more than
  5,000 agents at a dimension of 3.4 to 3.9.
- **23 October 2026.** This page said that the two worlds that lived to 50,000
  iterations ended larger and poorer than they began. The whole worlds the lab
  kept say otherwise (`GraphOfLifeRuns/.pilots/e18_health.json`). Seed 5 ended
  smaller and richer than at iteration 10,000: 1,861 agents, 56% of them at
  three tokens or fewer, against 2,257 and 63%. Seed 6 ended about where it
  began: 1,324 agents and 43%, against 1,302 and 42%. Both stayed large and
  fairly poor.
- **24 October 2026.** Two scores of [Did evolution go on?](#did-evolution-go-on)
  were the means of their four contests, 73% and 55%, beside medians for the
  rest. As medians, like the others and like the thesis of
  [Every era against three](every-era-against-three.md), they are 72% and 54%
  (`GraphOfLifeRuns/.pilots/e18_T100.json`).

<!-- turns -->
---

← [The diversity score](diversity-score.md) · [Looking for a healthy, rich world](../README.md#looking-for-a-healthy-rich-world) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) →

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (16)
> [Questioning the mechanics](questioning-the-mechanics.md) · [What B1 is, and what stood in its way](what-the-baseline-world-is.md) · [Building B2: a world that cannot break](a-world-that-cannot-break.md) · [Is a world a space? Dimension, and what folds it](how-many-dimensions-does-a-world-have.md) · [What holds a world's size](what-holds-a-worlds-size.md) · [Every era against three](every-era-against-three.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Who inherits?](who-inherits.md) · [Brains mixed at birth, units that cost, an eye for raids on kin](mixed-brains-costly-units.md) · [Links to strangers](links-to-strangers.md) · [Brains mixed at birth, with every other rule](mixing-with-every-other-rule.md) · [Healthy and rich](healthy-and-rich.md) · [The health bar](health-bar.md) · [Who may be a mate](mixed-at-birth-with-parents-as-mates.md) · [The candidates, played on and played again](the-candidates-played-on.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
