# Candidates for open-ended evolution

> [!question] Questions of this page
> - What must a world have for evolution in it to go on and on?
> - Which of the worlds drawn at random can be ruled out, and why?
> - Which worlds are worth playing on, and how?

## In short

Round 3 drew a hundred worlds at random. This page asks which of them could
evolve without end, and ranks them by how much they evolve. Each world is read
from what it recorded; no world is played again.

- **Ruled out:** 28 of the hundred plainly miss something. 18 died, one is a
  [star](measuring-networks.md#stars-and-hubs "An agent joined to far more agents than most; a star is a world held together by a single hub."), four seemed to pass nothing on and five stand still. The other 72 are
  the *candidate worlds*.
- **What descends is [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.").** A child gets a copy of its parent's brain, and
  a [node](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") won in the [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") gets a copy of the winner's. In the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world the
  game copies brains 1.4 times as often as births do. On the brains' own
  family tree, children act almost exactly as their parents, and the living
  descend from far fewer ancestors than chance would leave.
- **The evolution rank** scores every world that lived by four readings, each
  set against a model of chance: lines that spread beyond luck, novelty beyond
  [mutation](rules-in-detail.md#how-a-brain-changes "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") alone, selection on behaviour, and heredity. Eight of its first ten
  are draws that play the game a [token](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") at a time. Selection is plain in every
  world, but it holds behaviour in place: hardly any line of descent finds
  more new behaviour than mutation alone would.
- **The shortlist** is drawn from health instead: every candidate world among
  the first 13 of any of health's three orders, 21 worlds. Four of them are
  among the rank's first ten, and health and the rank do not go together.

Until 12 November 2026 this page read a family tree of births alone, which
misses every brain copied by a conquest
([How our view changed](#how-our-view-changed)).

## What evolution needs

Evolution needs variation, heredity and selection, and open-ended evolution
needs them to last
([What evolution means here](what-this-notebook-is-about.md#what-evolution-means-here);
[the open-ended test](measuring-evolution.md#what-would-count-as-open-ended-evolution)).
A world that plainly misses one of them cannot evolve open-endedly however long
it runs. So the worlds are treated as suspects in a line-up: each is ruled out
the first time it plainly misses one, with its reason. Every world that lived
is then ranked by evolution ([The evolution rank](#the-evolution-rank)).

## How each pillar is read

Every number comes from what a world recorded. No world is played again, only
brains, in situations taken from the worlds themselves. Two family trees are
read:
- **the births-only tree**: who bore whom, from every birth of every
  [iteration](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."). It misses every brain copied by a conquest. Its readings of
  9 November 2026 give the rules that rule worlds out, and the readings
  [pillar by pillar](#the-candidates-pillar-by-pillar);
- **the brain-copy tree**: every copy of a brain, by birth or by conquest
  ([The brains' own family tree](#the-brains-own-family-tree)). The evolution
  rank reads this one.

The family tree of genotypes in
[Measuring evolution](measuring-evolution.md#the-family-tree) is a third: it
includes conquest copies, but it branches at every mutation, so a brain that
changes where it sits starts a new genotype there.

The readings, and the tree each uses:
- **Generations and the effective population** (both trees). Going back from
  the brains alive at the end, their ancestors grow fewer and fewer, until all
  of them descend from one. In a population of *N* left to chance this takes
  about 2*N* generations. Where it takes a few dozen, the **effective
  population** is tiny, and chance decides which line survives, whatever its
  brains are.
- **Heredity** (both trees). In the worlds kept at the middle of the run and
  at its end, parents and children alive together are played in the same
  situations
  ([How different are the agents?](b1-brains-lineages-and-messages.md#how-different-are-the-agents)).
  The **heredity index** is 1 when a child acts as its parent does (as the
  parent acts like itself from one try to the next), and 0 when it acts like a
  stranger. On the births-only tree a child is played against the agent that
  bore it; on the brain tree, against the brain it was copied from.
- **Fertility passed on** (the births-only tree). Selection changes a
  population only through differences in how many offspring its members
  leave, and only the differences that are passed on add up (Fisher's
  fundamental theorem of natural selection). So we ask whether agents with
  many children have children with many too: the [correlation](statistics.md#two-numbers-together-correlation-a-fitted-line-and-logarithmic-axes "Whether two quantities go up and down together, from −1 to 1.") of a parent's
  and its child's numbers of children, over every pair born from iteration
  100 to two thirds of the run, against the same correlation with the
  children's counts shuffled among the children born in the same iteration.
  The cause need not be in the brain: a rich parent can make rich children,
  and a good place can favour its families.
- **The line of descent and its novelty** (both trees). At every world kept
  (every 250 iterations) we find the ancestor of most of the brains alive at
  the end, and keep its brain: one brain an era, the world's main line. All
  of them are played in the same situations. How far each era lies from every
  earlier one says whether the line keeps reaching behaviour it has not shown
  before (**novelty**), stands still, or comes back to where it was. The rank
  sets it against lines that only mutate.
- **Contemporaries** (no tree). How differently two agents of the same era
  act, above the noise of one brain played twice: the variation selection has
  to choose from.
- **Health** (no tree), by the [proposed order](which-worlds-are-healthy.md).
  It draws the shortlist, and for the rank it only filters.

These follow the measures of the MODES toolbox for open-ended dynamics
([Open-ended evolution: what it would mean](the-ideas-this-builds-on.md#open-ended-evolution-what-it-would-mean)):
*change* and *novelty* along the line of descent, and *ecology* in how many
kinds of behaviour live side by side (the
[same-position share](measuring-health.md#the-same-position-share)
and the effective kinds of staking). Its fourth measure, *complexity*, is not
read yet.

## The worlds ruled out

Of round 3's hundred worlds, 28 are ruled out, each for the first reason it
meets, in this order (research/screen/funnel.py). The other 72 are the
**candidate worlds**.
- **It died** (18 worlds;
  [what became of them](a-screen-of-random-worlds.md#what-round-3-found)).
- **It is a star** (1): [seed](rules-in-detail.md#seeds-and-the-random-stream "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 3020, where one agent is linked to 99.5% of the
  others on average.
- **Its agents hold one number of tokens** (none): 80% or more of the agents
  holding the same amount.
- **It has no heredity** (4): children act like strangers on the births-only
  tree, with a heredity index below 0.1. In all four, brains change after
  every game and not only when they are copied, and two mutate with a
  probability of 0.8. One of them is seed 3077, the worked example of
  [Which worlds are healthy?](which-worlds-are-healthy.md#three-ways-to-put-the-criteria-together):
  the first by the sum of its health's criteria, the 80th of the 82 by
  evolution.
- **It stands still** (5): its line of descent hardly moves from era to era,
  and two agents of one era hardly act differently (both less than 0.05 above
  the noise of playing one brain twice). All five are crowded and poor: from
  6,470 to 21,298 agents, holding 1.4 to 4.6 tokens each on average. An agent
  with a token or two can hardly stake differently from another.

Since 12 November 2026 the rule of no heredity rules nothing out of the
evolution rank. It played children against agents whose brains conquests had
mostly replaced since. On the brains' own tree, seeds 3066 and 3084 pass their
brains on faithfully (heredity indices of 0.97 and 0.96), and seeds 3044 and
3077, whose brains change after every game with a probability of 0.8, keep a
weak heredity (0.26 and 0.29). The other four rules still filter.

The ten worlds that lived and are ruled out:

<!-- table oeecandidates/out -->
| world | round | why it is ruled out |
|---|---|---|
| [s3020](run:B1-30000-511930-s3020) | round 3 | a star: one agent linked to 99.5% of the others on average |
| [s3037](run:B1-30000-845393-s3037) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.01, contemporaries 0.03 above the noise) |
| [s3038](run:B1-30000-1ad8a9-s3038) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.01, contemporaries 0.01 above the noise) |
| [s3044](run:B1-30000-63827e-s3044) | round 3 | no heredity: children act like strangers (index 0.01) |
| [s3066](run:B1-30000-dfd51b-s3066) | round 3 | no heredity: children act like strangers (index 0.07) |
| [s3067](run:B1-30000-4bee8e-s3067) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.02, contemporaries 0.03 above the noise) |
| [s3072](run:B1-30000-f05562-s3072) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.02, contemporaries 0.01 above the noise) |
| [s3077](run:B1-30000-d6eabf-s3077) | round 3 | no heredity: children act like strangers (index 0.06) |
| [s3084](run:B1-30000-ad76aa-s3084) | round 3 | no heredity: children act like strangers (index 0.08) |
| [s3099](run:B1-30000-1ab8a0-s3099) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.01, contemporaries 0.02 above the noise) |
<!-- /table -->

## The candidates, pillar by pillar

These readings are of the births-only tree, as they stood on 9 November 2026.
On 12 November 2026 the brain-copy tree replaced them for every reading but
fertility ([The brains' own family tree](#the-brains-own-family-tree)), and
this section was cut to its medians. On the births-only tree, the median
candidate world has:
- **825 generations** in 3,000 iterations: one every 3.6 iterations;
- **an effective population of about 14**: all the living descend from one
  ancestor 23 generations earlier. It is below 50 in 53 worlds, above 500 in
  only 8;
- **a heredity index of 0.39**, from 0.12 to 0.86;
- **a line of descent that moves slowly**: its late eras lie 0.08 above the
  noise from every earlier era, and in 15 worlds it never comes back to
  behaviour it had shown before.

**Fertility runs in families**, and this reading stands, since it counts
births: in 65 of the 72 worlds a parent's number of children says clearly more
of its children's than chance would (a median of 0.18 above chance in the
correlation).

Three pulls showed across the 72 worlds ([rank correlations](statistics.md#rank-correlation "How alike two orders are: the correlation of the ranks, 1 when the orders agree and 0 when one says nothing of the other.")): heredity against
novelty (−0.37), health against heredity (−0.30), and health with novelty
(+0.23). Where children were most like their parents, the line of descent
changed least; and the healthiest worlds passed on less, but changed a little
more.

Until 12 November 2026 a table here gave every candidate world's births-only
readings, sorted by the mean of its ranks in them. The
[evolution rank](#the-evolution-rank) replaces that mean, and the mean over six
of them that research/screen/funnel.py keeps as `evolution_rank`. The
births-only numbers of the shortlisted worlds stand in
[the shortlist](#the-shortlist)'s table, and the generations and effective
population of every world in
[Every world, by evolution](#every-world-by-evolution).

<!-- figure oeecandidates/heredity_novelty -->
![Like their parents, yet still changing?](../figures/oeecandidates/heredity_novelty.svg)

**Like their parents, yet still changing?** Each dot is a candidate world. Across: how much more a child acts like its parent than like a stranger, in the same situations. Up: how far the later eras of its line of descent lie from every earlier era, above the noise of playing one brain twice. A world in the top right passes on what its agents do and keeps changing it.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds that lived to the end of its run (Experiments 40, 41 and 42; 30,000 tokens, 3,000 iterations)
>
> **Data.** From each world's dossier, `GraphOfLifeRuns/.pilots/screen/candidates/<run>.json` (research/screen/candidates.py: the family tree from every birth, heredity in two kept worlds, the line of descent played in every kept world), its families, `.pilots/screen/families/<run>.json` (research/screen/families.py), and its health (research/screen/health.json), gathered by `research/screen/funnel.py` into `research/screen/candidates.json`.
>
> 1. Read each candidate world's heredity index (the mean of its kept worlds read) and late novelty from `candidates.json`.
>
> **To make it again:** `python3 book_figures.py oeecandidates`.
<!-- /figure -->

<!-- figure oeecandidates/generations_population -->
![How many generations, and how many lines?](../figures/oeecandidates/generations_population.svg)

**How many generations, and how many lines?** Each dot is a candidate world. Across: the mean generation of the agents alive at the end (how many births separate them from the founders). Up: a rough effective population, from how fast the ancestry of the living narrows: the smaller it is, the more chance decides which line survives, and the larger a difference selection needs before it can see it.

> [!example]- How to make this figure
> **Runs and data.** As for the figure *Like their parents, yet still changing?* above.
>
> 1. Read each candidate world's generations and effective population from `candidates.json` (`numbers`).
>
> **To make it again:** `python3 book_figures.py oeecandidates`.
<!-- /figure -->

<!-- figure oeecandidates/fertility -->
![Is having children passed on?](../figures/oeecandidates/fertility.svg)

**Is having children passed on?** Each dot is a candidate world: across every parent and child both born from iteration 100 to two thirds of the run, the correlation between how many children the parent had and how many the child had, less the same correlation with the children's counts shuffled among the children born in the same iteration. Above zero, families differ in how fertile they are, and selection has something to work on.

> [!example]- How to make this figure
> **Runs and data.** As for the figure *Like their parents, yet still changing?* above.
>
> 1. Read each candidate world's `fertility` from `candidates.json` (families.py: correlation less the shuffled mean).
>
> **To make it again:** `python3 book_figures.py oeecandidates`.
<!-- /figure -->

## The shortlist

Health comes in three orders: the six criteria multiplied, added, or the
worst of them ([Which worlds are healthy?](which-worlds-are-healthy.md#three-ways-to-put-the-criteria-together)).
The author (9 November 2026): "the union of the best of these rankings are
candidates." So the **shortlist** is every candidate world among the first 13
of any of the three orders: 21 worlds since later on 9 November 2026, when
health came to count agents on [loops](measuring-networks.md#loops "A path through the network that comes back to where it started without using a connection twice.") per link
([Which worlds are healthy?](which-worlds-are-healthy.md)). Before, it held 24,
and the author chose to play those again with new seeds
([Experiment 43](#round-3s-best-worlds-played-again)); what changed is told in
[How our view changed](#how-our-view-changed).

Health chooses the shortlist, not evolution. The table gives each shortlisted
world's place in the [evolution rank](#the-evolution-rank) beside its health,
and its births-only numbers.

<!-- table oeecandidates/shortlist -->
| world | round | among the first 13 by | health (product) | evolution rank | heredity (births only) | novelty (births only) | generations (births only) | effective population (births only) | fertility passed on | channels open |
|---|---|---|---|---|---|---|---|---|---|---|
| [s3042](run:B1-30000-b82ad0-s3042) | round 3 | product, sum, worst | 0.606 | 59 | 0.14 | 0.15 | 1,180 | 247 | 0.04 | messages, gifts, handover, mates pay, partner search, upkeep, mixing |
| [s4006 s](run:B1-30000-8526cf-s4006) | round 3, strongest first | product, sum, worst | 0.587 | 10 | 0.51 | 0.07 | 940 | 30 | 0.12 | messages, gifts, handover, upkeep, revolutions |
| [s3018](run:B1-30000-af1e28-s3018) | round 3 | product, sum, worst | 0.565 | 75 | 0.16 | 0.10 | 313 | 263 | 0.06 | messages, gifts, handover, mates pay, partner search, mixing |
| [s3080](run:B1-30000-625bc7-s3080) | round 3 | product, sum, worst | 0.549 | 81 | 0.34 | 0.11 | 121 | 11 | 0.04 | messages, gifts, handover, partner search, revolutions |
| [s4002 w](run:B1-30000-5379e2-s4002) | round 3, weakest first | product, sum | 0.544 | 2 | 0.22 | 0.22 | 822 | 8 | 0.37 | messages, gifts, mates pay, partner search, mixing |
| [s3057](run:B1-30000-38aee9-s3057) | round 3 | product, sum, worst | 0.541 | 16 | 0.14 | 0.07 | 597 | 11 | 0.26 | messages, partner search, mixing |
| [s3024](run:B1-30000-f05874-s3024) | round 3 | product, worst | 0.536 | 51 | 0.44 | 0.11 | 878 | 12 | 0.12 | messages, gifts, handover, upkeep, revolutions |
| [s4008 w](run:B1-30000-1cfaa4-s4008) | round 3, weakest first | product, sum | 0.528 | 3 | 0.23 | 0.23 | 662 | 12 | 0.42 | messages, gifts |
| [s3006](run:B1-30000-023c0a-s3006) | round 3 | product, sum | 0.526 | 43 | 0.48 | 0.05 | 652 | 59 | 0.09 | messages, gifts, handover, partner search, upkeep |
| [s3008](run:B1-30000-e19f2d-s3008) | round 3 | product, worst | 0.526 | 67 | 0.62 | 0.14 | 947 | 24 | 0.01 | messages, gifts, partner search, revolutions |
| [s3022](run:B1-30000-298337-s3022) | round 3 | product, worst | 0.525 | 45 | 0.22 | 0.07 | 199 | 9 | 0.25 | messages, partner search |
| [s3023](run:B1-30000-b0f6e3-s3023) | round 3 | product, worst | 0.515 | 46 | 0.34 | 0.06 | 955 | 22 | 0.19 | messages, gifts, upkeep |
| [s3054](run:B1-30000-d9254e-s3054) | round 3 | product, sum | 0.513 | 20 | 0.63 | 0.05 | 224 | 34 | 0.07 | messages, partner search, upkeep |
| [s3087](run:B1-30000-363694-s3087) | round 3 | sum | 0.511 | 60 | 0.54 | 0.04 | 1,179 | 12 | 0.31 | messages, handover, partner search, upkeep |
| [s3064](run:B1-30000-a6a7ec-s3064) | round 3 | sum | 0.511 | 56 | 0.13 | 0.10 | 232 | 2,072 | 0.09 | messages, handover, partner search |
| [s3019](run:B1-30000-b1209a-s3019) | round 3 | worst | 0.485 | 73 | 0.13 | 0.05 | 101 | 1,526 | -0.03 | messages, handover, partner search, upkeep |
| [s3005](run:B1-30000-c18a1f-s3005) | round 3 | worst | 0.481 | 26 | 0.51 | 0.08 | 950 | 35 | -0.05 | messages, handover, mixing |
| [s3045](run:B1-30000-f6d23f-s3045) | round 3 | sum | 0.476 | 55 | 0.22 | 0.14 | 1,237 | 490 | 0.08 | messages, gifts, handover, partner search, upkeep, revolutions |
| [s3058](run:B1-30000-4108dd-s3058) | round 3 | sum | 0.473 | 28 | 0.46 | 0.13 | 142 | 10 | 0.12 | messages, partner search, revolutions |
| [s3093](run:B1-30000-26a341-s3093) | round 3 | worst | 0.457 | 9 | 0.61 | 0.05 | 207 | 9 | 0.08 | messages, gifts, handover, partner search |
| [s4004 w](run:B1-30000-17ac6e-s4004) | round 3, weakest first | worst | 0.345 | 42 | 0.34 | 0.14 | 1,309 | 5 | 0.35 | messages, handover, mates pay, partner search, mixing |
<!-- /table -->

- **Five worlds are among the first 13 of all three orders:** seeds 3042,
  4006 played strongest first, 3018, 3080 and 3057. Seed 3042 is the
  healthiest by the product. The births-only tree gave it a large effective
  population (247) but children little like their parents (0.14). On the
  brain tree its children act almost as their parents (0.95), and it is the
  59th by evolution: selection on its behaviours is among the weakest of the
  round.
- **Seed 3026 has some of the simplest rules of the round:** [messages](rules-in-detail.md#messages "Numbers an agent writes to itself and to each neighbour every phase, which their brains read."),
  handing over, mending, [estates](measuring-lives.md#the-estate-of-the-dead "The tokens a dead agent leaves, and who gets them.") to the newborns, and a cap of 16 links that
  seldom binds; no search, no gifts, no costs on links. The births-only
  readings put it first: the most generations (2,159) and the line of descent
  reaching furthest into new behaviour (0.26 above the noise). The evolution
  rank puts it 19th, its children faithful copies (an index of 1.00). It just
  misses the shortlist: it is the 17th by its worst criterion of health, the
  23rd by the product and the 36th by the sum, since it culls 12% of its
  tokens in a game and its agents act alike.
- **Seed 3092, which meets every one of the author's lines, is not on the
  shortlist:** it is the 30th by the product, the 31st by the sum and the 31st
  by its worst criterion, and the 44th by evolution
  ([Which worlds are healthy?](which-worlds-are-healthy.md#what-the-orders-show)).

## The brains' own family tree

A world copies brains in two ways. A child gets a copy of its parent's brain,
in some worlds mixed with a mate's, and the copy may change. And a node won in
the game keeps its agent, but takes on a copy of the winner's brain. In the
median world of round 3 a game makes 1.4 copies for every birth: from almost
none in seed 3020, a star, to 14 in seed 3004. So what descends in a world is
its brains, not its agents. On 11 November 2026, after an independent review
of this page, the author approved a plan to rank round 3 by evolution on the
brains' own family tree, every reading set against a model of what chance
alone would do.

The **brain-copy tree** follows every copy of a brain, an **instance**. An
instance begins in one of three ways:
- **as a [founder](rules-in-detail.md#the-starting-ring "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts.")'s brain**, at the start;
- **at a birth**: a copy of the brain of the parent that pays for the child
  (mixed with a mate's in the worlds that mix at birth, and the mate is
  recorded where the mate pays);
- **at a conquest**: a copy of the winner's brain, as the game began. Every
  node of one game is won with the brains the game started with.

An instance lasts while it holds its node, until the node is won by another or
dies. A mutation that changes a brain where it sits gives it a new genotype,
but it is the same copy, so it counts as a change of that copy, not as a new
branch. One pass over every [frame](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.") of a world builds the tree: from 2.7 million
copies in the smallest world to 53 million in the largest, 1.6 billion in all.
Most copies are short-lived: a brain holds its node for about one iteration
before a [neighbour](the-world.md#the-network "An agent joined to another by a connection.") wins it.

On this tree every number of evolution changes. In the median world of the 82
that lived:

<!-- table evolutionrank/changed -->
| in the median world | births-only tree | brain-copy tree |
|---|---|---|
| generations by the end | 849 | 2,739 |
| births on the way | — | 959 |
| effective population | 14 | 100 |
| heredity index | 0.41 | 0.96 |
| late novelty above noise | 0.08 | 0.07 |
| shadow worlds' effective population | — | 3,459 |
| copies made by conquest | — | 58% |
<!-- /table -->

- **Many more generations**: a brain of the last iteration has a median of
  2,700 copies behind it, against 850 births on the births-only tree.
- **More lines, but far fewer than chance leaves.** The effective population
  is about 100, against about 3,500 in a shadow world, where every copy's
  parent is drawn at random among the living (below). All the living at the
  end descend from one brain a median of 193 iterations earlier; in their
  shadows, the living of most worlds meet no one ancestor within the run's
  3,000 iterations.
- **Children are copies**, with a heredity index of 0.96. The births-only
  tree played the wrong pairs: by the time a child and the agent that bore it
  were played together, one of their nodes had nearly always been won by a
  neighbour since. Only 4% of those pairs still held the parent's brain and
  the child's.
- **The line of descent moves somewhat less**: 0.07 above the noise from
  every earlier era, against 0.08 on the births-only line.

## The evolution rank

The **evolution rank** orders every world of round 3 that lived by four
readings of evolution on the brain-copy tree, each set against a model of what
would happen without evolution (research/screen/evolution.py, 12 November
2026). It replaces the mean ranks of the births-only readings
([The candidates, pillar by pillar](#the-candidates-pillar-by-pillar)).
Health does not score it. Health of 9 November only filters, by the funnel's
rules: a world that died (the dead are not ranked), a star, one holding, or a
line that stands still ([The worlds ruled out](#the-worlds-ruled-out)).

The four readings and their nulls:

1. **Lines beyond luck.** The shadow world of pure luck
   ([test 3](did-evolution-happen.md#3--a-shadow-world-of-pure-luck)): the
   same copies at the same moments into the same living, but each copy's
   parent drawn at random among the living, 100 times. In each era, the
   clades of the era's first brains are followed to its end: how many still
   have descendants, how large the largest is, and how much of the
   *evolutionary activity* (a clade's size summed over the era, after Bedau
   and Packard) lies in clades more active than 99% of the shadow's. Each is a
   z-score: how many of the shadow's own spreads the real tree lies beyond it.
   The reading is their mean over the twelve eras.
2. **Novelty beyond mutation alone.** The line of descent's brain of
   iteration 250 is given, era by era, as many random changes as the real
   line made, of the world's own kind: six such lines a world. All are played
   in the same situations of every kept world. The reading is the real line's
   late novelty above its noise, less the null lines'. Above 0, the line found
   more new behaviour than mutation alone would; below 0, selection held its
   behaviour in place.
3. **Selection on behaviour**
   ([test 4](did-evolution-happen.md#4--selection-on-behaviour-measured)): in
   every iteration, how strongly four behaviours go with a brain's
   descendants at the next iteration, with its wealth held fixed. The reading
   is the median size of that selection gradient over the four behaviours and
   twelve eras, in standard deviations of the behaviour; a median, since the
   links a newborn gets are heavy-tailed, and a few newborns with dozens would
   otherwise decide it. Its null: the descendants shuffled among the agents of
   each iteration.
4. **Heredity on the brain tree**: the heredity index, played on the tree's
   own parent and child brains, alive together in the worlds kept at
   iterations 1,500 and 3,000. Its null: each child given another pair's
   parent at random, 200 times.

Each reading is scored as the share of the worlds it beats: the best 1, the
worst 1 over their number, as health's measures are scored
([Which worlds are healthy?](which-worlds-are-healthy.md)). The four scores are
combined three ways, as the author asked for health (9 November 2026):
multiplied, added, and the worst of them. A world's score is the product, read
as the four's geometric average, and its rank orders by it; ties go by the
sum, then by the worst. The first ten of each of the three orders, together,
are the rank's bests. Every world's readings, their z-scores and its rank are
in `research/screen/evolution.json`.

What the four readings say across round 3 (tests 3 and 4 tell the first and
the third in full):
- **Lines spread beyond luck in 73 of the 82 worlds**, by more than two of
  the shadow's spreads, and in 69 the common ancestor of the living is more
  recent than in any of its 100 shadows. Three worlds spread their lines more
  slowly than luck would: seed 3077, whose brains change after every game
  with a probability of 0.8, and seeds 3030 and 3094, whose agents are mostly
  newborns and whose links are kept only while paid for (the two taken out of
  Experiment 43). The reading grows with a world's size (a rank correlation
  of 0.67 with its agents): luck narrows a large world's ancestry slowly, so a
  large world beats its shadow by more, in effect as well as in its z-score.
- **Hardly any line of descent finds more than mutation alone.** In 79 of
  the 82 worlds the line moves less far than lines that only mutate; in seeds
  3007, 3044 and 3077 a little further, by 0.01 to 0.02. Selection holds
  behaviour in place, rather than pushing it somewhere new.
- **Selection on behaviour is plain, but behaviours do not follow it.**
  Spreading the stake brings more descendants in 99% of the eras of all the
  worlds; yet a behaviour's mean does not move from era to era the way its
  selection points (a rank correlation of 0.05 in the median world).
- **Heredity is near 1 in all but a few worlds**: 0.9 or more in 68 of the 82,
  so this reading orders the worlds little. The lowest are seeds 3044 and
  3077 (0.26 and 0.29), and seed 3020, the star, whose brains are nearly never
  copied by conquest (0.50).

### The first ten

<!-- table evolutionrank/top -->
| rank | world | round | score | among the first 10 by | lines beyond luck (z) | novelty above the null | selection strength | heredity | copies by conquest | health's rank | on the shortlist | health filters it out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [s4003 s](run:B1-30000-fffc7d-s4003) | round 3, strongest first | 0.779 | product, sum, worst | 21.4 | -0.032 | 0.081 | 0.98 | 58% | 18 | no | — |
| 2 | [s4002 w](run:B1-30000-5379e2-s4002) | round 3, weakest first | 0.732 | product, sum, worst | 14.3 | -0.021 | 0.068 | 1.00 | 67% | 5 | yes | — |
| 3 | [s4008 w](run:B1-30000-1cfaa4-s4008) | round 3, weakest first | 0.703 | product, sum, worst | 14.0 | -0.053 | 0.070 | 1.00 | 64% | 8 | yes | — |
| 4 | [s4007 w](run:B1-30000-04d3eb-s4007) | round 3, weakest first | 0.674 | product, sum, worst | 16.5 | -0.051 | 0.086 | 0.96 | 61% | 80 | no | — |
| 5 | [s4003 w](run:B1-30000-9efb1d-s4003) | round 3, weakest first | 0.656 | product, sum | 28.1 | -0.102 | 0.072 | 1.00 | 45% | 29 | no | — |
| 6 | [s4007 s](run:B1-30000-a8a2d2-s4007) | round 3, strongest first | 0.654 | product, sum, worst | 17.4 | -0.056 | 0.084 | 0.96 | 61% | 77 | no | — |
| 7 | [s4002 s](run:B1-30000-192733-s4002) | round 3, strongest first | 0.639 | product, sum, worst | 14.7 | -0.085 | 0.068 | 1.00 | 66% | 33 | no | — |
| 8 | [s3037](run:B1-30000-845393-s3037) | round 3 | 0.620 | product, sum | 32.8 | -0.067 | 0.050 | 1.00 | 66% | 74 | no | standing still |
| 9 | [s3093](run:B1-30000-26a341-s3093) | round 3 | 0.617 | product, worst | 8.1 | -0.087 | 0.091 | 0.98 | 78% | 31 | yes | — |
| 10 | [s4006 s](run:B1-30000-8526cf-s4006) | round 3, strongest first | 0.596 | product | 15.0 | -0.058 | 0.050 | 0.99 | 45% | 2 | yes | — |
<!-- /table -->

- **Eight of the first ten play the game a token at a time.** Round 3 drew
  eight such [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") and played each twice, weakest first and strongest
  first; twelve of those worlds lived, and eight of them are here. The same
  draw lands near itself both ways: seeds 4003, 4002 and 4007 are in the first
  ten played either way. These twelve worlds beat the rest on all four
  readings, most of all on heredity and novelty. But those two play each brain
  as if it placed its stake all at once, as research/oee/checks/diversity.py
  plays every world's, so for these worlds they read a likeness of the brain
  rather than the way it stakes. On the shadow and on selection, read from
  what the worlds did, three of the twelve are among the first ten.
- **Four of the first ten are on the shortlist:** seeds 4002 and 4008 played
  weakest first, 3093, and 4006 played strongest first.
- **Health filters one of them out:** seed 3037, a crowded and poor world
  whose line of descent stands still and whose agents act alike
  ([The worlds ruled out](#the-worlds-ruled-out)). Its shadow reading is among
  the round's highest, which says its lines sweep, not that they find
  anything new.
- **Health and evolution do not go together.** The rank correlation between
  health's product and the evolution score, over the 82 worlds, is −0.01. The
  first ten of each of the evolution rank's three orders together hold 15
  worlds, five of them on the shortlist; the shortlist's other worlds stand
  from 16th to 81st.

<!-- figure evolutionrank/heredity -->
![Is a child's brain like its parent's?](../figures/evolutionrank/heredity.svg)

**Is a child's brain like its parent's?** Each dot is a world of round 3 that lived. Across: the heredity index as the births-only tree read it, playing a child against the agent that bore it. Up: the same index on the brain-copy tree, playing each brain against the brain it was copied from, by birth or by conquest. 1 means a child acts as its parent does, 0 as a stranger does.

> [!example]- How to make this figure
> **Runs.** As for the figure *Like their parents, yet still changing?* above.
>
> **Data.** From each world's readings on its brain-copy tree, `GraphOfLifeRuns/.pilots/screen/braintree/<run>.json` (research/screen/braintree.py: one pass over every frame, births and conquests both copying brains, and 100 shadow worlds of pure luck; selection.py; brainplay.py, brains of the kept worlds played), gathered by `research/screen/evolution.py` into `research/screen/evolution.json`, with the births-only numbers of `research/screen/candidates.json` beside them.
>
> 1. Read each world's `heredity_births_only` (`beside`) and its brain-tree heredity (`values`) from `evolution.json`.
>
> **To make it again:** `python3 book_figures.py evolutionrank`.
<!-- /figure -->

<!-- figure evolutionrank/ancestry -->
![How far back do the living meet one ancestor?](../figures/evolutionrank/ancestry.svg)

**How far back do the living meet one ancestor?** Each dot is a world of round 3 that lived. Across: how many iterations before the end all its last agents' brains descend from one brain. Up: the same in 100 shadow worlds, whose copies are made at the same moments into the same living but from parents drawn at random (the median; a world whose shadows mostly never meet one ancestor within the run is drawn at 3,000). A dot above the line: the real ancestry narrows faster than luck would narrow it.

> [!example]- How to make this figure
> **Runs and data.** As for the figure *Is a child's brain like its parent's?* above.
>
> 1. Read each world's `common_ancestor_age` and `common_ancestor_age_shadow` (`beside`) from `evolution.json`.
>
> **To make it again:** `python3 book_figures.py evolutionrank`.
<!-- /figure -->

<!-- figure evolutionrank/novelty -->
![Does the line of descent find more than mutation alone?](../figures/evolutionrank/novelty.svg)

**Does the line of descent find more than mutation alone?** Each dot is a world of round 3 that lived. Across: how far the later eras of its line of descent, on the brain-copy tree, lie from every earlier era in the same situations, above the noise of playing one brain twice. Up: the same for six lines that start from the line's brain of iteration 250 and make as many changes as it made, at random. A dot below the line: the real line moved further than mutation alone would take it; above it, less far.

> [!example]- How to make this figure
> **Runs and data.** As for the figure *Is a child's brain like its parent's?* above.
>
> 1. Read each world's `novelty_above_noise` and `novelty_null_above_noise` (`beside`) from `evolution.json`.
>
> **To make it again:** `python3 book_figures.py evolutionrank`.
<!-- /figure -->

### Every world, by evolution

Every world that lived, best first. Beside the brain tree's generations and
effective population stand the births-only tree's, and the shadow's
effective population.

<!-- table evolutionrank/all -->
| rank | world | score | ranks by product, sum, worst | lines beyond luck (z) | novelty above the null | selection strength | heredity | copies by conquest | generations (births only) | effective population (births only, shadow) | on the shortlist | health filters it out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [s4003 s](run:B1-30000-fffc7d-s4003) | 0.779 | 1, 1, 1 | 21.4 | -0.032 | 0.081 | 0.98 | 58% | 3,317 (1,131) | 81 (9, 6,828) | no | — |
| 2 | [s4002 w](run:B1-30000-5379e2-s4002) | 0.732 | 2, 2, 3 | 14.3 | -0.021 | 0.068 | 1.00 | 67% | 4,014 (822) | 77 (8, 5,473) | yes | — |
| 3 | [s4008 w](run:B1-30000-1cfaa4-s4008) | 0.703 | 3, 4, 2 | 14.0 | -0.053 | 0.070 | 1.00 | 64% | 4,014 (662) | 63 (12, 5,724) | yes | — |
| 4 | [s4007 w](run:B1-30000-04d3eb-s4007) | 0.674 | 4, 6, 8 | 16.5 | -0.051 | 0.086 | 0.96 | 61% | 3,654 (828) | 23 (9, 5,628) | no | — |
| 5 | [s4003 w](run:B1-30000-9efb1d-s4003) | 0.656 | 5, 5, 11 | 28.1 | -0.102 | 0.072 | 1.00 | 45% | 1,881 (1,189) | 21 (9, 6,963) | no | — |
| 6 | [s4007 s](run:B1-30000-a8a2d2-s4007) | 0.654 | 6, 8, 7 | 17.4 | -0.056 | 0.084 | 0.96 | 61% | 3,775 (855) | 35 (7, 5,694) | no | — |
| 7 | [s4002 s](run:B1-30000-192733-s4002) | 0.639 | 7, 9, 4 | 14.7 | -0.085 | 0.068 | 1.00 | 66% | 4,225 (953) | 19 (7, 5,588) | no | — |
| 8 | [s3037](run:B1-30000-845393-s3037) | 0.620 | 8, 7, 19 | 32.8 | -0.067 | 0.050 | 1.00 | 66% | 1,600 (347) | 43 (5, 5,388) | no | standing still |
| 9 | [s3093](run:B1-30000-26a341-s3093) | 0.617 | 9, 12, 5 | 8.1 | -0.087 | 0.091 | 0.98 | 78% | 2,323 (207) | 197 (9, 2,464) | yes | — |
| 10 | [s4006 s](run:B1-30000-8526cf-s4006) | 0.596 | 10, 13, 15 | 15.0 | -0.058 | 0.050 | 0.99 | 45% | 3,126 (940) | 119 (30, 5,937) | yes | — |
| 11 | [s3047](run:B1-30000-10c9f1-s3047) | 0.587 | 11, 23, 6 | 7.8 | -0.079 | 0.086 | 0.97 | 66% | 3,695 (662) | 57 (9, 1,416) | no | — |
| 12 | [s3052](run:B1-30000-0d058a-s3052) | 0.584 | 12, 14, 18 | 19.4 | -0.106 | 0.084 | 0.97 | 65% | 3,088 (387) | 129 (15, 6,094) | no | — |
| 13 | [s3071](run:B1-30000-605c0d-s3071) | 0.583 | 13, 10, 24 | 15.4 | -0.114 | 0.200 | 0.97 | 75% | 2,603 (109) | 145 (14, 4,355) | no | — |
| 14 | [s3050](run:B1-30000-ebe08c-s3050) | 0.577 | 14, 18, 13 | 9.6 | -0.045 | 0.088 | 0.94 | 64% | 2,900 (926) | 48 (6, 2,636) | no | — |
| 15 | [s3001](run:B1-30000-68cda7-s3001) | 0.568 | 15, 3, 52 | 50.2 | -0.013 | 0.034 | 1.00 | 21% | 1,535 (1,138) | 25 (15, 7,246) | no | — |
| 16 | [s3057](run:B1-30000-38aee9-s3057) | 0.566 | 16, 15, 20 | 13.3 | -0.071 | 0.185 | 0.92 | 66% | 3,263 (597) | 51 (11, 4,477) | yes | — |
| 17 | [s3076](run:B1-30000-66c764-s3076) | 0.556 | 17, 16, 30 | 12.4 | -0.055 | 0.048 | 1.00 | 44% | 3,351 (1,198) | 16 (9, 3,131) | no | — |
| 18 | [s3062](run:B1-30000-def151-s3062) | 0.555 | 18, 11, 27 | 7.2 | -0.006 | 0.141 | 0.91 | 52% | 4,034 (993) | 36 (11, 1,202) | no | — |
| 19 | [s3026](run:B1-30000-d6955f-s3026) | 0.550 | 19, 19, 28 | 4.4 | -0.056 | 0.071 | 1.00 | 50% | 5,232 (2,159) | 181 (78, 1,279) | no | — |
| 20 | [s3054](run:B1-30000-d9254e-s3054) | 0.548 | 20, 25, 9 | 7.2 | -0.095 | 0.085 | 0.98 | 75% | 2,233 (224) | 491 (34, 2,462) | yes | — |
| 21 | [s4005 s](run:B1-30000-6f6c09-s4005) | 0.546 | 21, 24, 10 | 11.1 | -0.013 | 0.056 | 0.96 | 54% | 2,718 (646) | 31 (10, 3,944) | no | — |
| 22 | [s3014](run:B1-30000-7d129f-s3014) | 0.504 | 22, 22, 35 | 30.1 | -0.101 | 0.113 | 0.90 | 41% | 2,309 (1,242) | 22 (14, 7,246) | no | — |
| 23 | [s3032](run:B1-30000-22865f-s3032) | 0.496 | 23, 28, 38 | 14.8 | -0.080 | 0.086 | 0.90 | 69% | 2,636 (401) | 409 (67, 5,716) | no | — |
| 24 | [s3049](run:B1-30000-c51eb8-s3049) | 0.494 | 24, 41, 14 | 5.8 | -0.082 | 0.073 | 0.97 | 80% | 2,701 (236) | 2,702 (60, 3,954) | no | — |
| 25 | [s3028](run:B1-30000-2eef58-s3028) | 0.482 | 25, 35, 21 | 14.9 | -0.111 | 0.052 | 0.99 | 40% | 3,222 (1,437) | 264 (37, 4,287) | no | — |
| 26 | [s3005](run:B1-30000-c18a1f-s3005) | 0.479 | 26, 36, 32 | 11.6 | -0.041 | 0.047 | 0.96 | 69% | 3,293 (950) | 104 (35, 3,601) | yes | — |
| 27 | [s3038](run:B1-30000-1ad8a9-s3038) | 0.478 | 27, 31, 41 | 19.9 | -0.064 | 0.044 | 0.96 | 63% | 2,158 (858) | 25 (7, 4,869) | no | standing still |
| 28 | [s3058](run:B1-30000-4108dd-s3058) | 0.467 | 28, 46, 25 | 4.4 | -0.061 | 0.070 | 0.96 | 80% | 2,303 (142) | 70 (10, 743) | yes | — |
| 29 | [s3012](run:B1-30000-02fa53-s3012) | 0.466 | 29, 52, 12 | 7.7 | -0.073 | 0.067 | 0.95 | 70% | 2,322 (327) | 101 (9, 2,309) | no | — |
| 30 | [s3078](run:B1-30000-dd27b6-s3078) | 0.461 | 30, 26, 44 | 6.2 | -0.008 | 0.043 | 0.99 | 49% | 3,480 (422) | 23 (5, 668) | no | — |
| 31 | [s3089](run:B1-30000-77ad6a-s3089) | 0.458 | 31, 34, 47 | 15.8 | -0.066 | 0.041 | 0.97 | 35% | 2,185 (844) | 104 (133, 3,522) | no | — |
| 32 | [s3091](run:B1-30000-42d88d-s3091) | 0.458 | 32, 29, 48 | 7.6 | -0.043 | 0.111 | 0.88 | 57% | 3,642 (1,173) | 16 (5, 1,580) | no | — |
| 33 | [s3041](run:B1-30000-965f1e-s3041) | 0.458 | 33, 47, 16 | 5.0 | -0.066 | 0.051 | 0.98 | 82% | 2,177 (212) | 566 (22, 3,016) | no | — |
| 34 | [s3056](run:B1-30000-43985b-s3056) | 0.450 | 34, 50, 22 | 4.4 | -0.081 | 0.084 | 0.95 | 80% | 2,414 (361) | 119 (16, 3,646) | no | — |
| 35 | [s3007](run:B1-30000-8119cd-s3007) | 0.445 | 35, 40, 31 | 6.9 | +0.021 | 0.059 | 0.91 | 48% | 3,807 (1,434) | 89 (37, 884) | no | — |
| 36 | [s3098](run:B1-30000-794ae8-s3098) | 0.444 | 36, 43, 33 | 16.3 | -0.102 | 0.073 | 0.90 | 46% | 2,012 (256) | 42 (6, 2,595) | no | — |
| 37 | [s3061](run:B1-30000-00e3c8-s3061) | 0.442 | 37, 44, 26 | 8.9 | -0.104 | 0.116 | 0.92 | 57% | 3,933 (1,092) | 32 (8, 2,626) | no | — |
| 38 | [s4004 s](run:B1-30000-2bac4d-s4004) | 0.441 | 38, 32, 42 | 4.5 | -0.046 | 0.140 | 0.90 | 54% | 3,774 (1,392) | 26 (6, 680) | no | — |
| 39 | [s4008 s](run:B1-30000-645972-s4008) | 0.435 | 39, 33, 53 | 14.2 | -0.137 | 0.061 | 1.00 | 66% | 3,883 (691) | 32 (14, 5,758) | no | — |
| 40 | [s3043](run:B1-30000-02ed17-s3043) | 0.435 | 40, 38, 49 | 2.7 | -0.081 | 0.079 | 0.98 | 91% | 2,394 (73) | 663 (904, 2,119) | no | — |
| 41 | [s3083](run:B1-30000-0d7e0c-s3083) | 0.432 | 41, 51, 39 | 11.1 | -0.127 | 0.078 | 0.96 | 66% | 1,987 (389) | 137 (11, 2,927) | no | — |
| 42 | [s4004 w](run:B1-30000-17ac6e-s4004) | 0.423 | 42, 30, 45 | 4.2 | -0.018 | 0.138 | 0.89 | 55% | 3,738 (1,309) | 15 (5, 639) | yes | — |
| 43 | [s3006](run:B1-30000-023c0a-s3006) | 0.419 | 43, 58, 17 | 15.7 | -0.105 | 0.056 | 0.94 | 46% | 1,892 (652) | 113 (59, 4,598) | yes | — |
| 44 | [s3092](run:B1-30000-7fc03f-s3092) | 0.419 | 44, 55, 36 | 6.1 | -0.126 | 0.084 | 0.96 | 77% | 2,760 (253) | 1,144 (74, 5,441) | no | — |
| 45 | [s3022](run:B1-30000-298337-s3022) | 0.416 | 45, 37, 50 | 14.6 | -0.103 | 0.124 | 0.87 | 69% | 2,579 (199) | 103 (9, 4,127) | yes | — |
| 46 | [s3023](run:B1-30000-b0f6e3-s3023) | 0.413 | 46, 53, 29 | 4.9 | -0.018 | 0.048 | 0.96 | 49% | 3,875 (955) | 99 (22, 1,080) | yes | — |
| 47 | [s3082](run:B1-30000-1141ea-s3082) | 0.389 | 47, 54, 43 | 9.5 | -0.129 | 0.050 | 1.00 | 47% | 3,238 (1,411) | 59 (26, 3,053) | no | — |
| 48 | [s3021](run:B1-30000-4aff7d-s3021) | 0.379 | 48, 61, 34 | 4.2 | -0.038 | 0.054 | 0.93 | 48% | 3,571 (1,064) | 214 (27, 1,108) | no | — |
| 49 | [s3067](run:B1-30000-4bee8e-s3067) | 0.372 | 49, 17, 73 | 34.5 | -0.211 | 0.089 | 0.97 | 15% | 3,289 (908) | 76 (24, 9,678) | no | standing still |
| 50 | [s3040](run:B1-30000-36d076-s3040) | 0.371 | 50, 59, 55 | 10.6 | -0.139 | 0.078 | 0.96 | 39% | 3,085 (805) | 188 (10, 2,993) | no | — |
| 51 | [s3024](run:B1-30000-f05874-s3024) | 0.369 | 51, 66, 23 | 11.2 | -0.097 | 0.048 | 0.93 | 46% | 2,702 (878) | 75 (12, 3,689) | yes | — |
| 52 | [s3099](run:B1-30000-1ab8a0-s3099) | 0.362 | 52, 21, 74 | 62.1 | -0.042 | 0.019 | 0.97 | 18% | 375 (2,764) | 27 (350, 7,188) | no | standing still |
| 53 | [s3046](run:B1-30000-27bd64-s3046) | 0.361 | 53, 49, 61 | 6.6 | -0.159 | 0.071 | 1.00 | 51% | 4,198 (1,321) | 173 (51, 1,340) | no | — |
| 54 | [s3072](run:B1-30000-f05562-s3072) | 0.355 | 54, 39, 58 | 42.0 | -0.043 | 0.046 | 0.81 | 16% | 1,013 (1,894) | 60 (86, 8,571) | no | standing still |
| 55 | [s3045](run:B1-30000-f6d23f-s3045) | 0.355 | 55, 60, 56 | 1.7 | -0.027 | 0.058 | 0.95 | 51% | 2,874 (1,237) | 202 (490, 850) | yes | — |
| 56 | [s3064](run:B1-30000-a6a7ec-s3064) | 0.354 | 56, 69, 40 | 3.8 | -0.089 | 0.069 | 0.94 | 82% | 2,526 (232) | 458 (2,072, 5,115) | yes | — |
| 57 | [s3059](run:B1-30000-2ec87a-s3059) | 0.326 | 57, 45, 70 | 11.6 | -0.195 | 0.056 | 1.00 | 75% | 2,321 (310) | 93 (12, 3,179) | no | — |
| 58 | [s3013](run:B1-30000-7f7cd5-s3013) | 0.322 | 58, 57, 54 | 30.6 | -0.131 | 0.072 | 0.85 | 45% | 2,163 (500) | 20 (5, 6,613) | no | — |
| 59 | [s3042](run:B1-30000-b82ad0-s3042) | 0.317 | 59, 67, 59 | 10.5 | -0.074 | 0.033 | 0.95 | 63% | 3,369 (1,180) | 1,390 (247, 9,019) | yes | — |
| 60 | [s3087](run:B1-30000-363694-s3087) | 0.317 | 60, 62, 62 | 23.8 | -0.086 | 0.029 | 0.92 | 59% | 4,035 (1,179) | 115 (12, 8,922) | yes | — |
| 61 | [s3070](run:B1-30000-a4d3a0-s3070) | 0.311 | 61, 27, 76 | 20.3 | -0.228 | 0.065 | 1.00 | 56% | 2,109 (774) | 35 (12, 4,803) | no | — |
| 62 | [s3020](run:B1-30000-511930-s3020) | 0.303 | 62, 42, 75 | 20.8 | -0.047 | 0.053 | 0.50 | 1% | 837 (826) | 410 (736, 5,038) | no | a star |
| 63 | [s3051](run:B1-30000-b5a76a-s3051) | 0.302 | 63, 76, 37 | 3.9 | -0.107 | 0.055 | 0.95 | 83% | 2,137 (141) | 44 (4, 420) | no | — |
| 64 | [s3090](run:B1-30000-05ba40-s3090) | 0.289 | 64, 75, 46 | 3.3 | -0.117 | 0.051 | 0.97 | 77% | 2,590 (14) | 279 (365, 959) | no | — |
| 65 | [s3066](run:B1-30000-dfd51b-s3066) | 0.278 | 65, 71, 63 | 0.7 | -0.073 | 0.044 | 0.97 | 58% | 3,837 (1,821) | 5,757 (7,585, 8,835) | no | — |
| 66 | [s3044](run:B1-30000-63827e-s3044) | 0.277 | 66, 20, 79 | 10.9 | +0.022 | 0.110 | 0.26 | 56% | 3,708 (916) | 26 (7, 3,396) | no | — |
| 67 | [s3008](run:B1-30000-e19f2d-s3008) | 0.268 | 67, 74, 51 | 5.3 | -0.127 | 0.037 | 0.98 | 60% | 3,160 (947) | 706 (24, 1,056) | yes | — |
| 68 | [s3030](run:B1-30000-90fc91-s3030) | 0.266 | 68, 48, 77 | -4.6 | -0.105 | 0.115 | 0.98 | 29% | 2,535 (761) | 2,536 (12,248, 11,287) | no | — |
| 69 | [s3004](run:B1-30000-7d5208-s3004) | 0.266 | 69, 63, 60 | 1.7 | -0.146 | 0.081 | 0.98 | 93% | 2,562 (23) | 395 (18, 1,078) | no | — |
| 70 | [s3065](run:B1-30000-ba3afe-s3065) | 0.265 | 70, 73, 57 | 3.4 | -0.100 | 0.034 | 0.98 | 92% | 2,351 (54) | 689 (1,147, 2,143) | no | — |
| 71 | [s3033](run:B1-30000-631e49-s3033) | 0.255 | 71, 65, 66 | 0.6 | -0.022 | 0.060 | 0.89 | 52% | 2,628 (1,083) | 649 (988, 1,050) | no | — |
| 72 | [s3063](run:B1-30000-924e70-s3063) | 0.237 | 72, 56, 67 | 37.2 | -0.178 | 0.083 | 0.78 | 30% | 1,963 (1,053) | 19 (8, 8,339) | no | — |
| 73 | [s3019](run:B1-30000-b1209a-s3019) | 0.220 | 73, 72, 71 | 2.9 | -0.094 | 0.027 | 0.98 | 88% | 2,317 (101) | 830 (1,526, 3,139) | yes | — |
| 74 | [s3016](run:B1-30000-b23381-s3016) | 0.217 | 74, 68, 64 | 0.7 | -0.164 | 0.059 | 1.00 | 44% | 1,926 (1,071) | 75 (31, 715) | no | — |
| 75 | [s3018](run:B1-30000-af1e28-s3018) | 0.205 | 75, 78, 68 | 3.6 | -0.125 | 0.082 | 0.74 | 79% | 2,452 (313) | 754 (263, 2,465) | yes | — |
| 76 | [s3085](run:B1-30000-801b34-s3085) | 0.195 | 76, 77, 69 | 11.5 | -0.087 | 0.028 | 0.79 | 42% | 3,195 (1,303) | 13 (5, 3,058) | no | — |
| 77 | [s3095](run:B1-30000-3e5856-s3095) | 0.179 | 77, 79, 65 | 2.7 | -0.133 | 0.029 | 0.99 | 49% | 1,372 (1,338) | 509 (1,219, 2,663) | no | — |
| 78 | [s3055](run:B1-30000-cca771-s3055) | 0.167 | 78, 80, 72 | 7.9 | -0.134 | 0.048 | 0.70 | 52% | 3,131 (1,033) | 43 (6, 3,010) | no | — |
| 79 | [s3094](run:B1-30000-4ac946-s3094) | 0.147 | 79, 70, 78 | -2.3 | -0.052 | 0.018 | 0.98 | 49% | 4,447 (1,509) | 4,448 (11,316, 10,439) | no | — |
| 80 | [s3077](run:B1-30000-d6eabf-s3077) | 0.116 | 80, 64, 80 | -5.4 | +0.013 | 0.073 | 0.29 | 55% | 3,778 (840) | 18,913 (2,696, 6,618) | no | — |
| 81 | [s3080](run:B1-30000-625bc7-s3080) | 0.094 | 81, 82, 81 | 6.1 | -0.248 | 0.044 | 0.85 | 83% | 2,254 (121) | 165 (11, 1,759) | yes | — |
| 82 | [s3084](run:B1-30000-ad76aa-s3084) | 0.091 | 82, 81, 82 | 0.1 | -0.120 | 0.013 | 0.96 | 59% | 3,157 (976) | 415 (179, 1,012) | no | — |
<!-- /table -->

## What a world's rules make possible

A run of 3,000 iterations shows what a world did, not everything it could do.
Evolution may need far longer than any run we can afford, so it matters, too,
what a world's rules allow its agents to do at all: a strategy its rules do
not allow can never evolve, however long the world runs. Each mechanism the
screen draws opens a channel, and most have a price.

| mechanism | what it makes possible | its price |
|---|---|---|
| messages (4 to 30 numbers to each neighbour) | signals: agents can tell neighbours apart, warn, coordinate their stakes, recognise their kin by a code they learn | a code only works once senders and listeners have evolved it together |
| messages before the game | a round of talk before acting: a stake can answer what a neighbour has just said | — |
| noise inputs (2 to 10 random numbers a neighbour) | a brain can randomise. In a game like Colonel Blotto every fixed way of staking can be beaten by another, so chance in the stakes is worth having | two copies of one brain act differently: behaviour is less heritable |
| gifts | an agent can give tokens: help, payment, support of kin — the raw stuff of cooperation | what is given is lost to the giver |
| handing over | a parent can set its child up with some of its own [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."): a position is inherited | the parent loses those connections |
| mixing at birth (the mate a parent) | a child's brain from two: two lines can combine what each found | a mix can break what worked in each |
| the mate pays | two parents share a child's cost | — |
| search (jump, walk, reach) | agents choose partners beyond their neighbours: they can find allies, or prey, and move in the network | links made far away fold a world into a small world |
| a token in every link, or upkeep | links with a price: a link is kept only where it is worth it | poor agents can hold few links |
| the lottery | the weaker can win: a node goes to a staker drawn in proportion to its stake | chance decides more: selection is weaker |
| [coalitions](rules-in-detail.md#how-a-coalition-takes-a-node "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") (revolutions) | the small can join against the largest staker | — |
| estates to the poor, the young, the family or the neighbours | where the wealth of the dead goes shapes who can have children next | — |
| a cap on links (valence) | no hubs, by rule | at 2, chains and rings only, and an agent at its cap has no child |
| a game in rounds, weakest or strongest first | agents see the board before each token and can answer each other's stakes | — |
| mutation only when copied | a brain changes only when it is copied, into a child or into a node its owner wins, and never while it sits: a genotype lasts, and can be selected | — |
| larger brains (3 to 5 layers of 16 to 50) | more complicated behaviour can be represented | — |

## What could be played next

Of the five ways to go on put to the author on 8 November 2026, the author
chose to play the shortlist's settings again with fresh seeds: Experiment 43
([Round 3's best worlds, played again](#round-3s-best-worlds-played-again)).
Playing each world's eras against each other, the first test of whether
evolution happened at all, is planned as Experiment 45
([Did the brains get better?](round-3-eras-against-each-other.md)).

## Round 3's best worlds, played again

<!-- experiment E43 -->

Experiment 43 plays the shortlist of 9 November 2026 again: each world with
its own drawn settings, exactly as round 3 drew them (the way of staking too,
for the draws played a token at a time), and with as many founders a token as
it had. Each is played twice, with seeds 4301 and 4302, at 50,000 tokens and
for 5,000 iterations. It asks whether a world's health belongs to its
settings, or was the luck of its seed:

<!-- thesis E43 -->
> [!quote] The thesis of Experiment 43, written down before any of its runs existed
> **The claim.** The health of round 3's best worlds belongs to their settings, not to the luck of their seed or to their size. Each of the 24 worlds chosen from health's three orders is played again with two new seeds, at 50,000 tokens instead of 30,000 and for 5,000 iterations instead of 3,000. Scored together with round 3's 82 living worlds, by the same measures averaged from iteration 100 to the end, the new worlds keep the order their settings had: a world's health by the product (the six criteria multiplied, as their geometric average) follows its setting's health in round 3, and two new worlds of one setting differ less than two worlds of round 3 drawn at random.
>
> **Why it would be so.** In round 3 every setting was played once, with one seed and at one size, so a world's place in the orders mixes what its settings do with the luck of its seed. The five draws that lived both when played weakest first and strongest first differ in health by 0.03 to 0.11 (a median of 0.03), against a median of 0.11 between two living worlds of round 3 drawn at random: a hint that settings, not luck, decide most of a world's health. The author (2026-11-08, 2026-11-09) asked for the best worlds played again with new seeds, larger, and more of them than eleven: the 24 among the first 13 of health's three orders.
>
> **It holds if:** Both, over the 24 settings, with a new world that dies counted as health 0: the rank correlation between a setting's health in round 3 and the mean health of its two new worlds is at least 0.5; and the median difference in health between a setting's two new worlds is below 0.11, the median between two living worlds of round 3 drawn at random.
>
> **It fails if:** Either: the rank correlation is below 0.2, or the median difference between a setting's two new worlds is 0.11 or more. Then health is mostly the luck of a seed or the size of a world, and a world's place in round 3 says little about its settings.
<!-- /thesis -->

The thesis names 24 worlds. On 9 November 2026 three of them, whose links ran
away, were taken out (seeds 3094, 3030 and 3095), and the four worlds that had
just joined the shortlist were added: since commit `650c4ab` the experiment
plays 25 settings in 50 runs. The thesis keeps its words. It is judged over
the 21 it was planned with, by health as it was then, without agents on loops
per link (the author's choice); the three taken out are neither counted as
dead nor scored, and the four are reported beside it. The three settings
without a hard limit on links (seeds 3016, 3042 and 3087) are played on,
watched: a run whose links an agent climb past twice what its world held in
round 3 is stopped and shown to the author.

<!-- runs E43 -->
> [!info]- The runs behind this page
> **50 runs.** Every condition below is run once for every seed (4301 to 4302), for 5,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **s3004** — the settings of world 4 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3004), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-cec254-s4301` … `B1-50000-cec254-s4302`.
> - **s3008** — the settings of world 8 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3008), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-253c1a-s4301` … `B1-50000-253c1a-s4302`.
> - **s3016** — the settings of world 16 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3016), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-6a808f-s4301` … `B1-50000-6a808f-s4302`.
> - **s3018** — the settings of world 18 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3018), as drawn; 50,000 tokens. Runs `B1-50000-af1e28-s4301` … `B1-50000-af1e28-s4302`.
> - **s3019** — the settings of world 19 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3019), as drawn; 50,000 tokens. Runs `B1-50000-b1209a-s4301` … `B1-50000-b1209a-s4302`.
> - **s3022** — the settings of world 22 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3022), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-2d6267-s4301` … `B1-50000-2d6267-s4302`.
> - **s3023** — the settings of world 23 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3023), as drawn; 50,000 tokens. Runs `B1-50000-b0f6e3-s4301` … `B1-50000-b0f6e3-s4302`.
> - **s3024** — the settings of world 24 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3024), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-3141af-s4301` … `B1-50000-3141af-s4302`.
> - **s3026** — the settings of world 26 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3026), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-371d3c-s4301` … `B1-50000-371d3c-s4302`.
> - **s3042** — the settings of world 42 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3042), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-4318af-s4301` … `B1-50000-4318af-s4302`.
> - **s3045** — the settings of world 45 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3045), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-e12a73-s4301` … `B1-50000-e12a73-s4302`.
> - **s3051** — the settings of world 51 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3051), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-13aad0-s4301` … `B1-50000-13aad0-s4302`.
> - **s3054** — the settings of world 54 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3054), as drawn; 50,000 tokens. Runs `B1-50000-d9254e-s4301` … `B1-50000-d9254e-s4302`.
> - **s3057** — the settings of world 57 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3057), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-4f0888-s4301` … `B1-50000-4f0888-s4302`.
> - **s3058** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-f44ef4-s4301` … `B1-50000-f44ef4-s4302`.
> - **s3080** — the settings of world 80 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3080), as drawn; 50,000 tokens. Runs `B1-50000-625bc7-s4301` … `B1-50000-625bc7-s4302`.
> - **s3087** — the settings of world 87 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3087), as drawn; 50,000 tokens. Runs `B1-50000-363694-s4301` … `B1-50000-363694-s4302`.
> - **s4002 weakest first** — the settings of world 2 of [Experiment 41](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4002), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-8d0936-s4301` … `B1-50000-8d0936-s4302`.
> - **s4004 weakest first** — the settings of world 4 of [Experiment 41](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4004), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-dad9b8-s4301` … `B1-50000-dad9b8-s4302`.
> - **s4006 strongest first** — the settings of world 6 of [Experiment 42](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4006), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-877f3c-s4301` … `B1-50000-877f3c-s4302`.
> - **s4008 weakest first** — the settings of world 8 of [Experiment 41](a-screen-of-random-worlds.md#the-worlds-of-round-3-a-token-at-a-time) (seed 4008), as drawn; 50,000 tokens. Runs `B1-50000-1cfaa4-s4301` … `B1-50000-1cfaa4-s4302`.
> - **s3005** — the settings of world 5 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3005), as drawn; 50,000 tokens. Runs `B1-50000-c18a1f-s4301` … `B1-50000-c18a1f-s4302`.
> - **s3006** — the settings of world 6 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3006), as drawn; 50,000 tokens. Runs `B1-50000-023c0a-s4301` … `B1-50000-023c0a-s4302`.
> - **s3064** — the settings of world 64 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3064), as drawn but with `n_nodes` = 250; 50,000 tokens. Runs `B1-50000-b8e076-s4301` … `B1-50000-b8e076-s4302`.
> - **s3093** — the settings of world 93 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3093), as drawn but with `n_nodes` = 1,000; 50,000 tokens. Runs `B1-50000-f372e1-s4301` … `B1-50000-f372e1-s4302`.
>
> To make these runs again: `python3 gol_lab.py run E43`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

## What we learned

- **Crowded, poor worlds stand still**: where agents hold a token or two,
  selection has little variation to choose from.
- **Evolution must be read on the brains' own family tree.** A tree of births
  alone misses the copies of conquest, and misreads both heredity and how
  many lines go on.
- **Selection holds behaviour in place.** Lines beat luck and selection on
  behaviour is plain, yet behaviour does not move where selection points, and
  hardly any line of descent gets further than mutation alone would take it.
- **Health is no guide to evolution**: over the 82 worlds that lived, the two
  do not go together.
- **Fertility runs in families in almost every world** (on the births-only
  tree); mending deaths weakens it.

## Where to go next

- [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md):
  how to tell selection from luck in these worlds.
- [Which worlds are healthy? A proposed order](which-worlds-are-healthy.md):
  the health each world brings.
- [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md):
  which settings give heredity, selection and change.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  the notebook's earlier tests of open-ended evolution.

## How this was made

The worlds read are every world of round 3 of the screen that lived to the end
of its run (Experiments 40, 41 and 42); round 1, drawn from an older list of
settings, is left out (the author, 8 November 2026).
Nothing is played again but brains, in situations taken from the worlds
themselves.

- `python3 research/screen/candidates.py <round> --jobs 3` builds each world's
  dossier: its family tree from every birth, heredity in the worlds kept at
  the middle and the end of the run (150 parent-child pairs in 100 situations),
  and its line of descent, played in 40 situations of every kept world. Each
  is kept in `GraphOfLifeRuns/.pilots/screen/candidates/<run>.json`.
- `python3 research/screen/families.py <rounds> --jobs 3` reads whether
  fertility is passed on, with 200 shuffles a world, into
  `.pilots/screen/families/<run>.json`.
- `python3 research/screen/funnel.py E41 E42 E40` rules the worlds out,
  ranks the rest, finds the shortlist and writes
  `research/screen/candidates.json`, from which
  `python3 book_figures.py oeecandidates` draws this page.
- `python3 research/screen/braintree.py` (12 November 2026) reads every
  living world of round 3, one at a time, in one pass over its frames: its
  brain-copy tree, 100 shadow worlds of pure luck, selection on four
  behaviours (`research/screen/selection.py`), and its brains played:
  heredity on the tree and the line of descent against six lines that only
  mutate (`research/screen/brainplay.py`). Each world's readings are kept in
  `GraphOfLifeRuns/.pilots/screen/braintree/<run>.json`. The pass reads only
  the parts of a frame the tree needs, and the stakes of every fifth game.
  `python3 research/screen/evolution.py` ranks the worlds into
  `research/screen/evolution.json`, from which
  `python3 book_figures.py evolutionrank` draws the section
  [The evolution rank](#the-evolution-rank).

## How our view changed

- **9 November 2026, later.** Health came to count agents on loops per link
  ([Which worlds are healthy?](which-worlds-are-healthy.md#not-a-tree-and-not-crowded-with-links)).
  The shortlist went from 24 worlds to 21: seeds 3004, 3016, 3026, 3030, 3051,
  3094 and 3095 left it, and seeds 3005, 3006, 3064 and 3093 joined it. Five
  worlds, not three, are among the first 13 of all three orders. Health
  against heredity reads −0.30, where it read −0.37.
- **9 November 2026, evening: Experiment 43 changed.** The [lab](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") began with the
  slowest worlds, and eight hours in the author saw that they were not
  healthy. Seeds 3094 and 3030, played again, held 10 to 32 links an agent,
  where [B1](b1-and-b2.md "The first baseline: the algorithm exactly as a new run is offered it.")'s worlds hold about 3.5, and took 50 to 100 seconds an iteration.
  Both drew `link_upkeep` `paid_only`: a link lives only while one of its ends
  pays a token for it every iteration, and a new link counts as paid for the
  iteration it is made in. In these worlds most agents are newborns in every
  iteration, each joined to about 25 of its parent's neighbours, and those
  fresh links all stand through their first game. Round 3 had them dense
  already (28 and 10 links an agent against a median of 3.6), and health's
  orders did not see it. On the author's word the two settings were taken out
  of the plan, with seed 3095, the third shortlisted world of the kind; the
  stopped runs stay on disk (`B1-50000-4ac946-s4301` and `-s4302` at about
  iteration 670, `B1-50000-90fc91-s4301` at about 600). The four worlds that
  had joined the shortlist were added with the same two seeds
  ([Round 3's best worlds, played again](#round-3s-best-worlds-played-again)).
- **12 November 2026.** An independent review found that the family tree of
  this page follows births only, though conquests copy brains as often or more,
  and that the shortlist ranks by health. On the brains' own family tree
  ([The brains' own family tree](#the-brains-own-family-tree)), over the 82
  worlds that lived, generations rise from a median of 850 to 2,700, the
  effective population from about 14 to 100, and heredity from 0.41 to 0.96.
  Round 3 was ranked by four readings of evolution against chance
  ([The evolution rank](#the-evolution-rank)); health stands beside the rank
  only as a filter, and the rule of no heredity no longer rules a world out.
  The same day the page was cut to say each thing once. The births-only
  readings keep their medians
  ([The candidates, pillar by pillar](#the-candidates-pillar-by-pillar)); their
  table of every candidate world, with the mean ranks it was sorted by, is in
  the notebook's history
  (`git show 037f997:book/pages/candidates-for-open-ended-evolution.md`). The
  shortlist's history and Experiment 43's change are told only here.

<!-- turns -->
---

← [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md) · [Round 3 of the random screen](../README.md#round-3-of-the-random-screen) · [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md) →

**Also about:** [Does evolution keep going?](../README.md#does-evolution-keep-going).

> [!info]- Pages that link here (15)
> [Welcome](../README.md) · [What this notebook is about](what-this-notebook-is-about.md) · [A screen of random worlds](a-screen-of-random-worlds.md) · [Which worlds are healthy? Three orders](which-worlds-are-healthy.md) · [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md) · [Did evolution happen? Fourteen ways to find out](did-evolution-happen.md) · [Did the brains get better? Round 3's eras against each other](round-3-eras-against-each-other.md) · [A hundred thousand iterations](a-hundred-thousand-iterations.md) · [Glossary](glossary.md) · [Brains replaced by their average](brains-replaced-by-their-average.md) · [How big should a brain be?](how-big-should-a-brain-be.md) · [Why does evolution level off?](why-does-evolution-level-off.md) · [A source and a sink](a-source-and-a-sink.md) · [Measuring health: criteria, the health bar and diversity](measuring-health.md) · [Every simulation: what was run, and what was measured](every-simulation.md)
<!-- /turns -->
