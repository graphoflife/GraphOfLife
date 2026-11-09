# Candidates for open-ended evolution

> [!question] Questions of this page
> - What must a world have for evolution in it to go on and on?
> - Which of the worlds drawn at random can be ruled out, and why?
> - Which worlds are worth playing on, and how?

## In short

Evolution needs variation, heredity and selection, and open-ended evolution
needs them to last. Each of round 3's hundred worlds was read for them from
what it recorded: its whole [family tree](genotype.md "The identity of a brain: a number that a copy keeps and a change replaces, and that remembers the genotype it came from."), how much children act like their
parents, whether having children runs in families, and how far its main line
of descent keeps moving into new behaviour. Of the hundred, 28 are ruled out:
18 died, one is a [star](stars-and-hubs.md "An agent joined to far more agents than most; a star is a world held together by a single hub."), four pass nothing on and five stand still. The other
72 evolve fast but narrowly: a [median](median-and-quantiles.md "The middle value: half of the numbers lie below it and half above.") of 825 generations in 3,000 [iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game."),
but an effective population of only 14, so chance decides much of which line
survives. The healthiest worlds change more than the rest, but they pass on
less. No world leads in both health and evolution: [seed](random-numbers.md "The number that fixes every random choice of a run: the same settings and the same seed make the same world.") 3092 is the healthiest,
seed 3026, with some of the simplest rules of the round, evolves the most, and
seed 3045 stands between them. With thirteen more, they make a shortlist of
sixteen worlds worth playing on.

## What evolution needs

Evolution needs three things, and open-ended evolution needs them to last
([The open-ended test](open-ended-test.md)):
- **variation:** the [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") differ, in what they are and in what they do;
- **heredity:** children are like their parents;
- **selection:** some agents leave more descendants than others *because of*
  what they are, and not by luck alone.

Where all three hold, a population adapts. For it to keep finding new things,
the world must also last, and what its agents do must keep changing rather
than coming to rest or going round in circles. A world that plainly misses one
of these cannot evolve open-endedly however long it runs. So the worlds are
treated as suspects in a line-up: each is ruled out the first time it plainly
misses one, with its reason, and the rest are ranked pillar by pillar, with no
single verdict over all of them.

## How each pillar is read

Every number comes from what a world recorded. No world is played again.

- **The family tree.** A world's [frames](frames-and-stats.md "A whole world as a run records it after one phase: every agent, its tokens and connections.") hold every birth of every iteration,
  so its whole family tree can be built: who descends from whom, and how many
  generations separate the living from the [founders](starting-ring.md "One of the agents a world starts with, each holding the same tokens, on a ring with a few shortcuts."). Going back from the agents
  alive at the end, their ancestors grow fewer and fewer, until all of them
  descend from one. In a population of *N* agents left to chance this takes
  about 2*N* generations. Where it takes a few dozen, the **effective
  population** is tiny, and chance decides which line survives, whatever its
  agents are.
- **Heredity.** In the worlds kept at the middle of the run and at its end,
  children and parents alive together are played in the same situations
  ([How different are the agents?](how-different-are-the-agents.md)). The
  **heredity index** is 1 when a child acts as its parent does (as the parent
  acts like itself from one try to the next), and 0 when it acts like a
  stranger.
- **Fertility passed on.** Selection changes a population only through
  differences in how many children its members leave, and only the differences
  that are passed on add up over the generations (Fisher's fundamental theorem
  of natural selection). So we ask whether agents that have many children have
  children that have many too: the [correlation](correlation.md "Whether two quantities go up and down together, from −1 to 1.") of a parent's and its child's
  numbers of children, over every pair born from iteration 100 to two thirds of
  the run, against the same correlation with the children's counts shuffled
  among the children born in the same iteration. Where it is clearly above
  chance, families differ in how fertile they are, and selection has
  something to work with. The cause need not be in the [brain](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions."): a rich parent
  can make rich children, and a good place can favour its families.
- **The line of descent.** At every world kept (every 250 iterations) we find
  the ancestor of most of the agents alive at the end, and keep its brain: one
  brain an era, the world's main line. All of them are played in the same
  situations. How far each era lies from every earlier one says whether the
  line keeps reaching behaviour it has not shown before (**novelty**), stands
  still, or comes back to where it was.
- **Contemporaries.** How differently two agents of the same era act, above
  the noise of one brain played twice: the variation selection has to choose
  from.
- **Health**, by the [proposed order](which-worlds-are-healthy.md).

These follow the measures of the MODES toolbox for open-ended dynamics
(Dolson, Vostinar, Wiser and Ofria, 2019), read on what GraphOfLife records:
*change* and *novelty* along the line of descent, and *ecology* in how many
kinds of behaviour live side by side (the
[same-position share](what-makes-a-world-interesting.md#the-same-position-share)
and the effective kinds of staking). Its fourth measure, *complexity*, is not
read yet.

## The worlds ruled out

Of round 3's hundred worlds, 28 are ruled out, each for the first reason it
meets, in this order:
- **It died** (18 worlds;
  [what became of them](a-screen-of-random-worlds.md#what-round-3-found)).
- **It is a star** (1): seed 3020, where one agent is linked to 99.5% of the
  others on average.
- **Its agents hold one number of [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.")** (none): 80% or more of the agents
  holding the same amount.
- **It has no heredity** (4): children act like strangers, with a heredity
  index below 0.1. In all four, brains change after every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") and not only
  when they are copied, and two mutate with a probability of 0.8. One of them
  is seed 3077, the worked example of
  [Which worlds are healthy?](which-worlds-are-healthy.md#a-line-read-as-a-share-of-what-it-asks):
  healthy in five criteria of six, but with nothing for evolution to build on.
- **It stands still** (5): its line of descent hardly moves from era to era,
  and two agents of one era hardly act differently (both less than 0.05 above
  the noise of playing one brain twice). All five are crowded and poor: from
  6,470 to 21,298 agents, holding 1.4 to 4.6 tokens each on average. An agent
  with a token or two can hardly stake differently from another.

<!-- table oeecandidates/out -->
| world | round | why it is ruled out |
|---|---|---|
| [s3009](run:B1-30000-f1b401-s3009) | round 3 | died at iteration 920 |
| [s3010](run:B1-30000-30d3da-s3010) | round 3 | died at iteration 202 |
| [s3015](run:B1-30000-f05b17-s3015) | round 3 | died at iteration 214 |
| [s3017](run:B1-30000-44f693-s3017) | round 3 | died at iteration 94 |
| [s3020](run:B1-30000-511930-s3020) | round 3 | a star: one agent linked to 99.5% of the others on average |
| [s3027](run:B1-30000-be02ce-s3027) | round 3 | died at iteration 26 |
| [s3031](run:B1-30000-caf1c5-s3031) | round 3 | died at iteration 142 |
| [s3036](run:B1-30000-1d8c54-s3036) | round 3 | died at iteration 115 |
| [s3037](run:B1-30000-845393-s3037) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.01, contemporaries 0.03 above the noise) |
| [s3038](run:B1-30000-1ad8a9-s3038) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.01, contemporaries 0.01 above the noise) |
| [s3044](run:B1-30000-63827e-s3044) | round 3 | no heredity: children act like strangers (index 0.01) |
| [s3066](run:B1-30000-dfd51b-s3066) | round 3 | no heredity: children act like strangers (index 0.07) |
| [s3067](run:B1-30000-4bee8e-s3067) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.02, contemporaries 0.03 above the noise) |
| [s3069](run:B1-30000-033fbf-s3069) | round 3 | died at iteration 2,442 |
| [s3072](run:B1-30000-f05562-s3072) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.02, contemporaries 0.01 above the noise) |
| [s3074](run:B1-30000-165142-s3074) | round 3 | died at iteration 12 |
| [s3077](run:B1-30000-d6eabf-s3077) | round 3 | no heredity: children act like strangers (index 0.06) |
| [s3079](run:B1-30000-afe35f-s3079) | round 3 | died at iteration 210 |
| [s3081](run:B1-30000-378f0a-s3081) | round 3 | died at iteration 167 |
| [s3084](run:B1-30000-ad76aa-s3084) | round 3 | no heredity: children act like strangers (index 0.08) |
| [s3086](run:B1-30000-1393f4-s3086) | round 3 | died at iteration 719 |
| [s3096](run:B1-30000-76ca8e-s3096) | round 3 | died at iteration 1,030 |
| [s3099](run:B1-30000-1ab8a0-s3099) | round 3 | standing still: its line of descent does not move and its agents act alike (steps 0.01, contemporaries 0.02 above the noise) |
| [s3100](run:B1-30000-13bb83-s3100) | round 3 | died at iteration 1,815 |
| [s4001 w](run:B1-30000-1da684-s4001) | round 3, weakest first | died at iteration 144 |
| [s4005 w](run:B1-30000-a6c23a-s4005) | round 3, weakest first | died at iteration 230 |
| [s4006 w](run:B1-30000-673ef1-s4006) | round 3, weakest first | died at iteration 1,445 |
| [s4001 s](run:B1-30000-ec8a95-s4001) | round 3, strongest first | died at iteration 387 |
<!-- /table -->

## The candidates, pillar by pillar

The 72 candidates evolve fast, but in few lines.
- **Many generations.** A median of 825 generations in 3,000 iterations: one
  every 3.6 iterations.
- **Few lines.** Going back from the living, all of them descend from one
  ancestor a median of 23 generations earlier. That is an effective population
  of about 14 in the median world: below 50 in 53 worlds, above 500 in only 8.
  In most worlds, chance decides which line goes on, whatever its agents do.
- **Children are like their parents, partly.** A median heredity index of 0.39,
  from 0.12 to 0.86.
- **Fertility runs in families.** In 65 of the 72 worlds a parent's number of
  children says clearly more of its children's than chance would (a median of
  0.18 above chance in the correlation).
- **The line of descent moves slowly.** Its late eras lie a median of 0.08
  above the noise from every earlier era, and in 15 worlds it never comes back
  to behaviour it had shown before.

Three pulls show across the worlds (rank correlations over the 72):
- **heredity against novelty (−0.37):** where children are most like their
  parents, the line of descent changes least. Keeping what works and finding
  something new pull against each other;
- **health against heredity (−0.56):** the healthiest worlds pass on less;
- **health with novelty (+0.38):** the healthiest worlds change more.

Each world's rank in every category, 1 the best:

<!-- table oeecandidates/ranks -->
| world | round | size | generations | heredity | population | novelty | contemporaries | fertility | health | riches | mean rank | channels open |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [s3026](run:B1-30000-d6955f-s3026) | round 3 | 58 | 1 | 47 | 14 | 1 | 2 | 42 | 14 | 15 | 21.6 | messages, handover |
| [s3045](run:B1-30000-f6d23f-s3045) | round 3 | 68 | 12 | 58 | 9 | 16 | 18 | 51 | 2 | 5 | 26.6 | messages, gifts, handover, partner search, upkeep, revolutions |
| [s3021](run:B1-30000-4aff7d-s3021) | round 3 | 61 | 23 | 46 | 26 | 11 | 16 | 30 | 17 | 12 | 26.9 | messages, handover, partner search, upkeep |
| [s4008 w](run:B1-30000-1cfaa4-s4008) | round 3, weakest first | 28 | 43 | 55 | 41 | 2 | 10 | 4 | 20 | 45 | 27.6 | messages, gifts |
| [s3095](run:B1-30000-3e5856-s3095) | round 3 | 46 | 7 | 53 | 5 | 9 | 3 | 70 | 29 | 27 | 27.7 | messages, handover, partner search, upkeep |
| [s4003 w](run:B1-30000-9efb1d-s4003) | round 3, weakest first | 8 | 14 | 22 | 51 | 18 | 23 | 16 | 38 | 65 | 28.3 | messages, gifts, handover, upkeep |
| [s4008 s](run:B1-30000-645972-s4008) | round 3, strongest first | 23 | 41 | 52 | 37 | 8 | 8 | 6 | 30 | 50 | 28.3 | messages, gifts |
| [s3062](run:B1-30000-def151-s3062) | round 3 | 51 | 26 | 61 | 45 | 3 | 5 | 26 | 23 | 22 | 29.1 | messages, handover, partner search, upkeep, revolutions |
| [s4003 s](run:B1-30000-fffc7d-s4003) | round 3, strongest first | 10 | 19 | 39 | 53 | 5 | 17 | 19 | 39 | 63 | 29.3 | messages, gifts, handover, upkeep |
| [s4002 w](run:B1-30000-5379e2-s4002) | round 3, weakest first | 26 | 37 | 57 | 58 | 4 | 6 | 12 | 18 | 47 | 29.4 | messages, gifts, mates pay, partner search, mixing |
| [s3042](run:B1-30000-b82ad0-s3042) | round 3 | 5 | 15 | 66 | 12 | 10 | 20 | 58 | 12 | 68 | 29.6 | messages, gifts, handover, mates pay, partner search, upkeep, mixing |
| [s3008](run:B1-30000-e19f2d-s3008) | round 3 | 59 | 30 | 13 | 28 | 15 | 35 | 64 | 13 | 14 | 30.1 | messages, gifts, partner search, revolutions |
| [s3046](run:B1-30000-27bd64-s3046) | round 3 | 55 | 8 | 54 | 19 | 26 | 27 | 36 | 28 | 18 | 30.1 | messages, gifts, partner search, upkeep |
| [s3082](run:B1-30000-1141ea-s3082) | round 3 | 48 | 5 | 3 | 27 | 29 | 44 | 38 | 52 | 25 | 30.1 | messages, upkeep |
| [s3030](run:B1-30000-90fc91-s3030) | round 3 | 1 | 40 | 6 | 1 | 57 | 25 | 35 | 35 | 72 | 30.2 | messages, upkeep |
| [s4002 s](run:B1-30000-192733-s4002) | round 3, strongest first | 21 | 28 | 56 | 62 | 7 | 4 | 10 | 34 | 52 | 30.4 | messages, gifts, mates pay, partner search, mixing |
| [s3024](run:B1-30000-f05874-s3024) | round 3 | 31 | 33 | 34 | 38 | 23 | 11 | 44 | 21 | 42 | 30.8 | messages, gifts, handover, upkeep, revolutions |
| [s3033](run:B1-30000-631e49-s3033) | round 3 | 63 | 21 | 65 | 7 | 6 | 7 | 68 | 31 | 10 | 30.9 | handover, partner search, upkeep, revolutions |
| [s4004 w](run:B1-30000-17ac6e-s4004) | round 3, weakest first | 71 | 9 | 43 | 68 | 12 | 12 | 15 | 47 | 2 | 31.0 | messages, handover, mates pay, partner search, mixing |
| [s4004 s](run:B1-30000-2bac4d-s4004) | round 3, strongest first | 69 | 6 | 48 | 63 | 19 | 15 | 13 | 46 | 4 | 31.4 | messages, handover, mates pay, partner search, mixing |
| [s3094](run:B1-30000-4ac946-s3094) | round 3 | 4 | 2 | 68 | 2 | 40 | 13 | 41 | 48 | 69 | 31.9 | messages, gifts, partner search, upkeep |
| [s3016](run:B1-30000-b23381-s3016) | round 3 | 67 | 22 | 49 | 24 | 56 | 1 | 60 | 3 | 6 | 32.0 | messages, partner search, upkeep |
| [s3055](run:B1-30000-cca771-s3055) | round 3 | 53 | 25 | 38 | 65 | 21 | 14 | 21 | 32 | 20 | 32.1 | messages, handover, revolutions |
| [s3032](run:B1-30000-22865f-s3032) | round 3 | 13 | 49 | 50 | 16 | 35 | 26 | 37 | 11 | 60 | 33.0 | messages, gifts, handover, partner search |
| [s3028](run:B1-30000-2eef58-s3028) | round 3 | 20 | 3 | 2 | 20 | 45 | 72 | 17 | 66 | 53 | 33.1 | messages, gifts, handover, partner search, upkeep, revolutions |
| [s3091](run:B1-30000-42d88d-s3091) | round 3 | 56 | 17 | 51 | 70 | 14 | 22 | 14 | 51 | 17 | 34.7 | messages, gifts, handover, partner search, mixing |
| [s4005 s](run:B1-30000-6f6c09-s4005) | round 3, strongest first | 39 | 45 | 45 | 47 | 13 | 9 | 39 | 42 | 34 | 34.8 | messages, gifts, partner search |
| [s3049](run:B1-30000-c51eb8-s3049) | round 3 | 27 | 58 | 63 | 17 | 20 | 31 | 45 | 7 | 46 | 34.9 | messages, gifts, handover, partner search |
| [s3023](run:B1-30000-b0f6e3-s3023) | round 3 | 64 | 27 | 44 | 29 | 52 | 40 | 34 | 16 | 9 | 35.0 | messages, gifts, upkeep |
| [s4007 w](run:B1-30000-04d3eb-s4007) | round 3, weakest first | 17 | 36 | 20 | 56 | 36 | 37 | 3 | 55 | 56 | 35.1 | messages, handover, partner search |
| [s3064](run:B1-30000-a6a7ec-s3064) | round 3 | 25 | 59 | 71 | 3 | 25 | 28 | 48 | 9 | 48 | 35.1 | messages, handover, partner search |
| [s3018](run:B1-30000-af1e28-s3018) | round 3 | 50 | 54 | 64 | 11 | 28 | 33 | 55 | 4 | 23 | 35.8 | messages, gifts, handover, mates pay, partner search, mixing |
| [s3058](run:B1-30000-4108dd-s3058) | round 3 | 70 | 64 | 32 | 49 | 17 | 30 | 43 | 15 | 3 | 35.9 | messages, partner search, revolutions |
| [s4007 s](run:B1-30000-a8a2d2-s4007) | round 3, strongest first | 16 | 34 | 37 | 61 | 24 | 34 | 7 | 54 | 57 | 36.0 | messages, handover, partner search |
| [s3006](run:B1-30000-023c0a-s3006) | round 3 | 15 | 44 | 27 | 18 | 58 | 29 | 50 | 25 | 58 | 36.0 | messages, gifts, handover, partner search, upkeep |
| [s3001](run:B1-30000-68cda7-s3001) | round 3 | 2 | 18 | 4 | 33 | 63 | 71 | 22 | 45 | 71 | 36.6 | messages, handover, mates pay, upkeep, revolutions, mixing |
| [s4006 s](run:B1-30000-8526cf-s4006) | round 3, strongest first | 14 | 31 | 24 | 25 | 49 | 45 | 46 | 37 | 59 | 36.7 | messages, gifts, handover, upkeep, revolutions |
| [s3014](run:B1-30000-7d129f-s3014) | round 3 | 6 | 11 | 21 | 35 | 64 | 66 | 1 | 62 | 67 | 37.0 | messages, handover, partner search, upkeep |
| [s3092](run:B1-30000-7fc03f-s3092) | round 3 | 18 | 57 | 60 | 15 | 42 | 32 | 54 | 1 | 55 | 37.1 | messages, partner search, upkeep |
| [s3007](run:B1-30000-8119cd-s3007) | round 3 | 43 | 4 | 18 | 21 | 66 | 56 | 29 | 68 | 30 | 37.2 | messages, gifts, handover, mates pay, upkeep, mixing |
| [s3054](run:B1-30000-d9254e-s3054) | round 3 | 45 | 60 | 12 | 23 | 55 | 52 | 53 | 8 | 28 | 37.3 | messages, partner search, upkeep |
| [s3071](run:B1-30000-605c0d-s3071) | round 3 | 19 | 67 | 30 | 36 | 27 | 48 | 33 | 22 | 54 | 37.3 | messages, partner search |
| [s3076](run:B1-30000-66c764-s3076) | round 3 | 32 | 13 | 8 | 55 | 51 | 63 | 2 | 71 | 41 | 37.3 | messages, gifts, handover, mixing |
| [s3080](run:B1-30000-625bc7-s3080) | round 3 | 60 | 66 | 42 | 46 | 22 | 24 | 59 | 5 | 13 | 37.4 | messages, gifts, handover, partner search, revolutions |
| [s3052](run:B1-30000-0d058a-s3052) | round 3 | 12 | 51 | 35 | 34 | 37 | 49 | 24 | 41 | 61 | 38.2 | messages, gifts, handover, partner search |
| [s3087](run:B1-30000-363694-s3087) | round 3 | 9 | 16 | 19 | 40 | 65 | 70 | 20 | 44 | 64 | 38.6 | messages, handover, partner search, upkeep |
| [s3057](run:B1-30000-38aee9-s3057) | round 3 | 30 | 46 | 67 | 43 | 47 | 21 | 27 | 24 | 43 | 38.7 | messages, partner search, mixing |
| [s3047](run:B1-30000-10c9f1-s3047) | round 3 | 41 | 42 | 15 | 57 | 43 | 64 | 23 | 36 | 32 | 39.2 | messages, gifts, handover, partner search |
| [s3050](run:B1-30000-ebe08c-s3050) | round 3 | 44 | 32 | 31 | 64 | 41 | 42 | 25 | 49 | 29 | 39.7 | messages, gifts, handover, mates pay, partner search, mixing |
| [s3089](run:B1-30000-77ad6a-s3089) | round 3 | 22 | 35 | 1 | 13 | 34 | 62 | 67 | 72 | 51 | 39.7 | messages, partner search, upkeep |
| [s3078](run:B1-30000-dd27b6-s3078) | round 3 | 62 | 48 | 10 | 69 | 33 | 67 | 5 | 57 | 11 | 40.2 | messages, gifts, partner search |
| [s3005](run:B1-30000-c18a1f-s3005) | round 3 | 37 | 29 | 23 | 22 | 44 | 53 | 72 | 50 | 36 | 40.7 | messages, handover, mixing |
| [s3040](run:B1-30000-36d076-s3040) | round 3 | 38 | 38 | 40 | 48 | 31 | 41 | 32 | 63 | 35 | 40.7 | messages, gifts, partner search, upkeep |
| [s3061](run:B1-30000-00e3c8-s3061) | round 3 | 47 | 20 | 62 | 60 | 54 | 19 | 18 | 60 | 26 | 40.7 | gifts, mates pay, partner search, revolutions, mixing |
| [s3063](run:B1-30000-924e70-s3063) | round 3 | 3 | 24 | 28 | 59 | 60 | 59 | 8 | 56 | 70 | 40.8 | messages, handover, partner search, upkeep |
| [s3059](run:B1-30000-2ec87a-s3059) | round 3 | 36 | 55 | 29 | 39 | 32 | 36 | 57 | 53 | 37 | 41.6 | messages, gifts, handover, mixing |
| [s3085](run:B1-30000-801b34-s3085) | round 3 | 35 | 10 | 33 | 71 | 69 | 38 | 11 | 69 | 38 | 41.6 | messages, handover, partner search, upkeep |
| [s3004](run:B1-30000-7d5208-s3004) | round 3 | 65 | 71 | 25 | 31 | 39 | 68 | 61 | 10 | 8 | 42.0 | messages, gifts |
| [s3022](run:B1-30000-298337-s3022) | round 3 | 24 | 63 | 59 | 52 | 46 | 39 | 28 | 19 | 49 | 42.1 | messages, partner search |
| [s3070](run:B1-30000-a4d3a0-s3070) | round 3 | 11 | 39 | 7 | 42 | 48 | 69 | 47 | 61 | 62 | 42.9 | messages, gifts, partner search, upkeep |
| [s3019](run:B1-30000-b1209a-s3019) | round 3 | 49 | 68 | 69 | 4 | 53 | 43 | 71 | 6 | 24 | 43.0 | messages, handover, partner search, upkeep |
| [s3051](run:B1-30000-b5a76a-s3051) | round 3 | 72 | 65 | 36 | 72 | 30 | 51 | 40 | 26 | 1 | 43.7 | messages, gifts, partner search, upkeep, revolutions |
| [s3083](run:B1-30000-0d7e0c-s3083) | round 3 | 34 | 50 | 11 | 44 | 38 | 47 | 69 | 67 | 39 | 44.3 | messages, gifts, handover, mates pay, upkeep, mixing |
| [s3041](run:B1-30000-965f1e-s3041) | round 3 | 33 | 61 | 5 | 30 | 68 | 57 | 65 | 43 | 40 | 44.7 | messages, gifts, mates pay, partner search, mixing |
| [s3043](run:B1-30000-02ed17-s3043) | round 3 | 52 | 69 | 41 | 8 | 70 | 65 | 49 | 27 | 21 | 44.7 | messages, gifts, handover, partner search, upkeep |
| [s3013](run:B1-30000-7f7cd5-s3013) | round 3 | 7 | 47 | 17 | 67 | 71 | 61 | 9 | 64 | 66 | 45.4 | messages, partner search, upkeep |
| [s3056](run:B1-30000-43985b-s3056) | round 3 | 40 | 52 | 16 | 32 | 62 | 60 | 56 | 59 | 33 | 45.6 | messages, gifts, handover, mates pay, partner search, mixing |
| [s3012](run:B1-30000-02fa53-s3012) | round 3 | 54 | 53 | 9 | 54 | 50 | 46 | 63 | 65 | 19 | 45.9 | messages, gifts, partner search, upkeep, mixing |
| [s3093](run:B1-30000-26a341-s3093) | round 3 | 42 | 62 | 14 | 50 | 59 | 54 | 52 | 58 | 31 | 46.9 | messages, gifts, handover, partner search |
| [s3065](run:B1-30000-ba3afe-s3065) | round 3 | 57 | 70 | 72 | 6 | 67 | 50 | 66 | 33 | 16 | 48.6 | messages, handover, partner search, upkeep, mixing |
| [s3098](run:B1-30000-794ae8-s3098) | round 3 | 29 | 56 | 26 | 66 | 61 | 55 | 31 | 70 | 44 | 48.7 | messages, gifts, handover, partner search, revolutions |
| [s3090](run:B1-30000-05ba40-s3090) | round 3 | 66 | 72 | 70 | 10 | 72 | 58 | 62 | 40 | 7 | 50.8 | messages, partner search, upkeep, mixing |
<!-- /table -->

<!-- figure oeecandidates/heredity_novelty -->
![Like their parents, yet still changing?](../figures/oeecandidates/heredity_novelty.svg)

**Like their parents, yet still changing?** Each dot is a candidate world. Across: how much more a child acts like its parent than like a stranger, in the same situations. Up: how far the later eras of its line of descent lie from every earlier era, above the noise of playing one brain twice. A world in the top right passes on what its agents do and keeps changing it.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds that lived to the end of its run (Experiments 40, 41 and 42; 30,000 tokens, 3,000 iterations)
>
> **Data.** From each world's dossier, `GraphOfLifeRuns/.pilots/screen/candidates/<run>.json` (research/screen/candidates.py: the family tree from every birth, heredity in two kept worlds, the line of descent played in every kept world), its families, `.pilots/screen/families/<run>.json` (research/screen/families.py), and its health (research/screen/health.json), gathered by `research/screen/funnel.py` into `research/screen/candidates.json`.
>
> 1. Read each candidate's heredity index (the mean of its kept worlds read) and late novelty from `candidates.json`.
>
> **To make it again:** `python3 book_figures.py oeecandidates`.
<!-- /figure -->

<!-- figure oeecandidates/generations_population -->
![How many generations, and how many lines?](../figures/oeecandidates/generations_population.svg)

**How many generations, and how many lines?** Each dot is a candidate world. Across: the mean generation of the agents alive at the end (how many births separate them from the founders). Up: a rough effective population, from how fast the ancestry of the living narrows: the smaller it is, the more chance decides which line survives, and the larger a difference selection needs before it can see it.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds that lived to the end of its run (Experiments 40, 41 and 42; 30,000 tokens, 3,000 iterations)
>
> **Data.** From each world's dossier, `GraphOfLifeRuns/.pilots/screen/candidates/<run>.json` (research/screen/candidates.py: the family tree from every birth, heredity in two kept worlds, the line of descent played in every kept world), its families, `.pilots/screen/families/<run>.json` (research/screen/families.py), and its health (research/screen/health.json), gathered by `research/screen/funnel.py` into `research/screen/candidates.json`.
>
> 1. Read each candidate's generations and effective population from `candidates.json` (`numbers`).
>
> **To make it again:** `python3 book_figures.py oeecandidates`.
<!-- /figure -->

<!-- figure oeecandidates/fertility -->
![Is having children passed on?](../figures/oeecandidates/fertility.svg)

**Is having children passed on?** Each dot is a candidate world: across every parent and child both born from iteration 100 to two thirds of the run, the correlation between how many children the parent had and how many the child had, less the same correlation with the children's counts shuffled among the children born in the same iteration. Above zero, families differ in how fertile they are, and selection has something to work on.

> [!example]- How to make this figure
> **Runs.** every world of round 3 of the screen of random worlds that lived to the end of its run (Experiments 40, 41 and 42; 30,000 tokens, 3,000 iterations)
>
> **Data.** From each world's dossier, `GraphOfLifeRuns/.pilots/screen/candidates/<run>.json` (research/screen/candidates.py: the family tree from every birth, heredity in two kept worlds, the line of descent played in every kept world), its families, `.pilots/screen/families/<run>.json` (research/screen/families.py), and its health (research/screen/health.json), gathered by `research/screen/funnel.py` into `research/screen/candidates.json`.
>
> 1. Read each candidate's `fertility` from `candidates.json` (families.py: correlation less the shuffled mean).
>
> **To make it again:** `python3 book_figures.py oeecandidates`.
<!-- /figure -->

The numbers behind the ranks:

<!-- table oeecandidates/numbers -->
| world | agents | generations | effective population | heredity | fertility passed on | novelty | contemporaries | health |
|---|---|---|---|---|---|---|---|---|
| [s3026](run:B1-30000-d6955f-s3026) | 2,068 | 2159 | 78 | 0.30 | 0.12 | 0.26 | 0.31 | 5.16 |
| [s3045](run:B1-30000-f6d23f-s3045) | 1,411 | 1237 | 490 | 0.22 | 0.08 | 0.14 | 0.22 | 5.89 |
| [s3021](run:B1-30000-4aff7d-s3021) | 1,715 | 1064 | 27 | 0.32 | 0.22 | 0.14 | 0.23 | 5.06 |
| [s4008 w](run:B1-30000-1cfaa4-s4008) | 5,120 | 662 | 12 | 0.23 | 0.42 | 0.23 | 0.26 | 5.05 |
| [s3095](run:B1-30000-3e5856-s3095) | 3,265 | 1338 | 1,219 | 0.25 | -0.02 | 0.15 | 0.29 | 4.82 |
| [s4003 w](run:B1-30000-9efb1d-s4003) | 10,264 | 1189 | 9 | 0.52 | 0.34 | 0.13 | 0.19 | 4.66 |
| [s4008 s](run:B1-30000-645972-s4008) | 5,402 | 691 | 14 | 0.25 | 0.40 | 0.17 | 0.27 | 4.78 |
| [s3062](run:B1-30000-def151-s3062) | 2,569 | 993 | 11 | 0.19 | 0.27 | 0.22 | 0.28 | 4.94 |
| [s4003 s](run:B1-30000-fffc7d-s4003) | 8,805 | 1131 | 9 | 0.36 | 0.33 | 0.19 | 0.23 | 4.65 |
| [s4002 w](run:B1-30000-5379e2-s4002) | 5,278 | 822 | 8 | 0.22 | 0.37 | 0.22 | 0.28 | 5.06 |
| [s3042](run:B1-30000-b82ad0-s3042) | 11,125 | 1180 | 247 | 0.14 | 0.04 | 0.15 | 0.21 | 5.32 |
| [s3008](run:B1-30000-e19f2d-s3008) | 2,063 | 947 | 24 | 0.62 | 0.01 | 0.14 | 0.15 | 5.30 |
| [s3046](run:B1-30000-27bd64-s3046) | 2,418 | 1321 | 51 | 0.23 | 0.18 | 0.10 | 0.18 | 4.82 |
| [s3082](run:B1-30000-1141ea-s3082) | 3,207 | 1411 | 26 | 0.84 | 0.14 | 0.09 | 0.12 | 4.23 |
| [s3030](run:B1-30000-90fc91-s3030) | 19,951 | 761 | 12,248 | 0.68 | 0.18 | 0.05 | 0.18 | 4.71 |
| [s4002 s](run:B1-30000-192733-s4002) | 5,705 | 953 | 7 | 0.22 | 0.39 | 0.17 | 0.29 | 4.74 |
| [s3024](run:B1-30000-f05874-s3024) | 4,614 | 878 | 12 | 0.44 | 0.12 | 0.11 | 0.26 | 5.03 |
| [s3033](run:B1-30000-631e49-s3033) | 1,639 | 1083 | 988 | 0.15 | -0.00 | 0.19 | 0.27 | 4.77 |
| [s4004 w](run:B1-30000-17ac6e-s4004) | 1,186 | 1309 | 5 | 0.34 | 0.35 | 0.14 | 0.25 | 4.42 |
| [s4004 s](run:B1-30000-2bac4d-s4004) | 1,349 | 1392 | 6 | 0.29 | 0.37 | 0.12 | 0.23 | 4.43 |
| [s3094](run:B1-30000-4ac946-s3094) | 11,256 | 1509 | 11,316 | 0.13 | 0.14 | 0.08 | 0.24 | 4.42 |
| [s3016](run:B1-30000-b23381-s3016) | 1,486 | 1071 | 31 | 0.28 | 0.04 | 0.05 | 0.34 | 5.78 |
| [s3055](run:B1-30000-cca771-s3055) | 2,507 | 1033 | 6 | 0.36 | 0.31 | 0.11 | 0.24 | 4.75 |
| [s3032](run:B1-30000-22865f-s3032) | 7,263 | 401 | 67 | 0.28 | 0.18 | 0.08 | 0.18 | 5.34 |
| [s3028](run:B1-30000-2eef58-s3028) | 5,736 | 1437 | 37 | 0.84 | 0.33 | 0.07 | 0.05 | 3.60 |
| [s3091](run:B1-30000-42d88d-s3091) | 2,361 | 1173 | 5 | 0.25 | 0.36 | 0.14 | 0.20 | 4.24 |
| [s4005 s](run:B1-30000-6f6c09-s4005) | 3,866 | 646 | 10 | 0.33 | 0.14 | 0.14 | 0.26 | 4.49 |
| [s3049](run:B1-30000-c51eb8-s3049) | 5,163 | 236 | 60 | 0.17 | 0.12 | 0.11 | 0.17 | 5.58 |
| [s3023](run:B1-30000-b0f6e3-s3023) | 1,612 | 955 | 22 | 0.34 | 0.19 | 0.06 | 0.14 | 5.10 |
| [s4007 w](run:B1-30000-04d3eb-s4007) | 6,489 | 828 | 9 | 0.53 | 0.43 | 0.08 | 0.14 | 4.21 |
| [s3064](run:B1-30000-a6a7ec-s3064) | 5,290 | 232 | 2,072 | 0.13 | 0.09 | 0.10 | 0.18 | 5.55 |
| [s3018](run:B1-30000-af1e28-s3018) | 2,861 | 313 | 263 | 0.16 | 0.06 | 0.10 | 0.16 | 5.74 |
| [s3058](run:B1-30000-4108dd-s3058) | 1,257 | 142 | 10 | 0.46 | 0.12 | 0.13 | 0.17 | 5.16 |
| [s4007 s](run:B1-30000-a8a2d2-s4007) | 6,653 | 855 | 7 | 0.37 | 0.40 | 0.11 | 0.15 | 4.22 |
| [s3006](run:B1-30000-023c0a-s3006) | 6,774 | 652 | 59 | 0.48 | 0.09 | 0.05 | 0.17 | 4.87 |
| [s3001](run:B1-30000-68cda7-s3001) | 17,266 | 1138 | 15 | 0.83 | 0.29 | 0.04 | 0.05 | 4.46 |
| [s4006 s](run:B1-30000-8526cf-s4006) | 7,009 | 940 | 30 | 0.51 | 0.12 | 0.07 | 0.12 | 4.68 |
| [s3014](run:B1-30000-7d129f-s3014) | 10,836 | 1242 | 14 | 0.53 | 0.45 | 0.04 | 0.07 | 3.97 |
| [s3092](run:B1-30000-7fc03f-s3092) | 6,165 | 253 | 74 | 0.20 | 0.07 | 0.08 | 0.17 | 6.00 |
| [s3007](run:B1-30000-8119cd-s3007) | 3,382 | 1434 | 37 | 0.57 | 0.23 | 0.04 | 0.09 | 3.57 |
| [s3054](run:B1-30000-d9254e-s3054) | 3,309 | 224 | 34 | 0.63 | 0.07 | 0.05 | 0.10 | 5.55 |
| [s3071](run:B1-30000-605c0d-s3071) | 6,070 | 109 | 14 | 0.47 | 0.19 | 0.10 | 0.11 | 5.02 |
| [s3076](run:B1-30000-66c764-s3076) | 4,525 | 1198 | 9 | 0.66 | 0.44 | 0.06 | 0.08 | 2.78 |
| [s3080](run:B1-30000-625bc7-s3080) | 1,985 | 121 | 11 | 0.34 | 0.04 | 0.11 | 0.18 | 5.70 |
| [s3052](run:B1-30000-0d058a-s3052) | 7,368 | 387 | 15 | 0.42 | 0.28 | 0.08 | 0.11 | 4.55 |
| [s3087](run:B1-30000-363694-s3087) | 10,034 | 1179 | 12 | 0.54 | 0.31 | 0.04 | 0.06 | 4.49 |
| [s3057](run:B1-30000-38aee9-s3057) | 4,802 | 597 | 11 | 0.14 | 0.26 | 0.07 | 0.20 | 4.91 |
| [s3047](run:B1-30000-10c9f1-s3047) | 3,607 | 662 | 9 | 0.61 | 0.29 | 0.08 | 0.08 | 4.68 |
| [s3050](run:B1-30000-ebe08c-s3050) | 3,312 | 926 | 6 | 0.47 | 0.28 | 0.08 | 0.13 | 4.40 |
| [s3089](run:B1-30000-77ad6a-s3089) | 5,521 | 844 | 133 | 0.86 | -0.00 | 0.09 | 0.09 | 2.21 |
| [s3078](run:B1-30000-dd27b6-s3078) | 1,672 | 422 | 5 | 0.65 | 0.40 | 0.09 | 0.07 | 4.17 |
| [s3005](run:B1-30000-c18a1f-s3005) | 4,022 | 950 | 35 | 0.51 | -0.05 | 0.08 | 0.10 | 4.33 |
| [s3040](run:B1-30000-36d076-s3040) | 3,904 | 805 | 10 | 0.35 | 0.21 | 0.09 | 0.14 | 3.97 |
| [s3061](run:B1-30000-00e3c8-s3061) | 3,236 | 1092 | 8 | 0.17 | 0.33 | 0.05 | 0.22 | 4.00 |
| [s3063](run:B1-30000-924e70-s3063) | 14,473 | 1053 | 8 | 0.47 | 0.39 | 0.05 | 0.09 | 4.20 |
| [s3059](run:B1-30000-2ec87a-s3059) | 4,136 | 310 | 12 | 0.47 | 0.05 | 0.09 | 0.15 | 4.22 |
| [s3085](run:B1-30000-801b34-s3085) | 4,156 | 1303 | 5 | 0.44 | 0.39 | 0.02 | 0.14 | 3.49 |
| [s3004](run:B1-30000-7d5208-s3004) | 1,581 | 23 | 18 | 0.51 | 0.04 | 0.08 | 0.07 | 5.54 |
| [s3022](run:B1-30000-298337-s3022) | 5,394 | 199 | 9 | 0.22 | 0.25 | 0.07 | 0.14 | 5.05 |
| [s3070](run:B1-30000-a4d3a0-s3070) | 7,767 | 774 | 12 | 0.67 | 0.11 | 0.07 | 0.06 | 4.00 |
| [s3019](run:B1-30000-b1209a-s3019) | 3,019 | 101 | 1,526 | 0.13 | -0.03 | 0.05 | 0.13 | 5.61 |
| [s3051](run:B1-30000-b5a76a-s3051) | 1,091 | 141 | 4 | 0.41 | 0.14 | 0.09 | 0.11 | 4.86 |
| [s3083](run:B1-30000-0d7e0c-s3083) | 4,159 | 389 | 11 | 0.63 | -0.01 | 0.08 | 0.11 | 3.58 |
| [s3041](run:B1-30000-965f1e-s3041) | 4,464 | 212 | 22 | 0.74 | 0.00 | 0.03 | 0.09 | 4.49 |
| [s3043](run:B1-30000-02ed17-s3043) | 2,520 | 73 | 904 | 0.35 | 0.09 | 0.02 | 0.07 | 4.85 |
| [s3013](run:B1-30000-7f7cd5-s3013) | 10,511 | 500 | 5 | 0.58 | 0.39 | 0.02 | 0.09 | 3.90 |
| [s3056](run:B1-30000-43985b-s3056) | 3,649 | 361 | 16 | 0.60 | 0.05 | 0.04 | 0.09 | 4.01 |
| [s3012](run:B1-30000-02fa53-s3012) | 2,487 | 327 | 9 | 0.65 | 0.02 | 0.06 | 0.12 | 3.88 |
| [s3093](run:B1-30000-26a341-s3093) | 3,387 | 207 | 9 | 0.61 | 0.08 | 0.05 | 0.10 | 4.15 |
| [s3065](run:B1-30000-ba3afe-s3065) | 2,104 | 54 | 1,147 | 0.12 | -0.00 | 0.03 | 0.11 | 4.75 |
| [s3098](run:B1-30000-794ae8-s3098) | 4,806 | 256 | 6 | 0.49 | 0.21 | 0.04 | 0.10 | 3.42 |
| [s3090](run:B1-30000-05ba40-s3090) | 1,521 | 14 | 365 | 0.13 | 0.03 | -0.00 | 0.09 | 4.60 |
<!-- /table -->

## The shortlist

No world leads in every category, and no weight says how much health is
worth against evolution. So the shortlist asks only this: that no other world
beat a world in both its health (the proposed order's) and its evolution (its
mean rank over the six categories of evolution: heredity, effective population,
novelty, contemporaries, fertility passed on and generations). The worlds that
pass make the **front**. Setting the front aside and asking again gives a
second front, then a third. The first three make the shortlist:

<!-- table oeecandidates/shortlist -->
| front | world | round | health | evolution's mean rank | heredity | novelty | generations | effective population | fertility passed on | channels open |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [s3092](run:B1-30000-7fc03f-s3092) | round 3 | 6.00 | 43.3 | 0.20 | 0.08 | 253 | 74 | 0.07 | messages, partner search, upkeep |
| 1 | [s3045](run:B1-30000-f6d23f-s3045) | round 3 | 5.89 | 27.3 | 0.22 | 0.14 | 1,237 | 490 | 0.08 | messages, gifts, handover, partner search, upkeep, revolutions |
| 1 | [s3026](run:B1-30000-d6955f-s3026) | round 3 | 5.16 | 17.8 | 0.30 | 0.26 | 2,159 | 78 | 0.12 | messages, handover |
| 2 | [s3016](run:B1-30000-b23381-s3016) | round 3 | 5.78 | 35.3 | 0.28 | 0.05 | 1,071 | 31 | 0.04 | messages, partner search, upkeep |
| 2 | [s3042](run:B1-30000-b82ad0-s3042) | round 3 | 5.32 | 30.2 | 0.14 | 0.15 | 1,180 | 247 | 0.04 | messages, gifts, handover, mates pay, partner search, upkeep, mixing |
| 2 | [s3021](run:B1-30000-4aff7d-s3021) | round 3 | 5.06 | 25.3 | 0.32 | 0.14 | 1,064 | 27 | 0.22 | messages, handover, partner search, upkeep |
| 2 | [s3095](run:B1-30000-3e5856-s3095) | round 3 | 4.82 | 24.5 | 0.25 | 0.15 | 1,338 | 1,219 | -0.02 | messages, handover, partner search, upkeep |
| 2 | [s4003 w](run:B1-30000-9efb1d-s4003) | round 3, weakest first | 4.66 | 24.0 | 0.52 | 0.13 | 1,189 | 9 | 0.34 | messages, gifts, handover, upkeep |
| 3 | [s3018](run:B1-30000-af1e28-s3018) | round 3 | 5.74 | 40.8 | 0.16 | 0.10 | 313 | 263 | 0.06 | messages, gifts, handover, mates pay, partner search, mixing |
| 3 | [s3049](run:B1-30000-c51eb8-s3049) | round 3 | 5.58 | 39.0 | 0.17 | 0.11 | 236 | 60 | 0.12 | messages, gifts, handover, partner search |
| 3 | [s3032](run:B1-30000-22865f-s3032) | round 3 | 5.34 | 35.5 | 0.28 | 0.08 | 401 | 67 | 0.18 | messages, gifts, handover, partner search |
| 3 | [s3008](run:B1-30000-e19f2d-s3008) | round 3 | 5.30 | 30.8 | 0.62 | 0.14 | 947 | 24 | 0.01 | messages, gifts, partner search, revolutions |
| 3 | [s4002 w](run:B1-30000-5379e2-s4002) | round 3, weakest first | 5.06 | 29.0 | 0.22 | 0.22 | 822 | 8 | 0.37 | messages, gifts, mates pay, partner search, mixing |
| 3 | [s4008 w](run:B1-30000-1cfaa4-s4008) | round 3, weakest first | 5.05 | 25.8 | 0.23 | 0.23 | 662 | 12 | 0.42 | messages, gifts |
| 3 | [s4008 s](run:B1-30000-645972-s4008) | round 3, strongest first | 4.78 | 25.3 | 0.25 | 0.17 | 691 | 14 | 0.40 | messages, gifts |
| 3 | [s3082](run:B1-30000-1141ea-s3082) | round 3 | 4.23 | 24.3 | 0.84 | 0.09 | 1,411 | 26 | 0.14 | messages, upkeep |
<!-- /table -->

- **The first front is three worlds.** Seed 3026 is the most evolving of the
  round: it lives the most generations (2,159), and its line of descent reaches
  furthest into new behaviour (0.26 above the noise) and never comes back. Its
  rules are among the simplest the round drew: [messages](messages.md "Numbers an agent writes to itself and to each neighbour every phase, which their brains read."), handing over, mending,
  [estates](estate.md "The tokens a dead agent leaves, and who gets them.") to the newborns, and a cap of 16 links that seldom binds; no search,
  no gifts, no costs on links. It misses only the sixth criterion of health: it
  loses a piece of more than 15% of its network about six times every hundred
  iterations. Seed 3092 is the healthiest world, the only one to meet every
  line, but it evolves slowly (253 generations, a heredity index of 0.20).
  Seed 3045 stands between them: health 5.89 of 6, 1,237 generations, and the
  largest effective population of the shortlist but one (490).
- **The second front is five worlds:** seeds 3016, 3042, 3021 and 3095, and
  seed 4003 played weakest first, whose children are the most like their
  parents on the front (0.52).
- **The third front is eight:** seeds 3018, 3049, 3032, 3008 and 3082, and
  three of the draws played a token at a time (4002 and 4008 weakest first,
  4008 strongest first): healthy and changing, but with tiny effective
  populations (8 to 14), and fertility running strongly in their families (0.37
  to 0.42 above chance).

## What a world's rules make possible

A run of 3,000 iterations shows what a world did, not everything it could do.
Evolution may need far longer than any run we can afford, so it matters, too,
what a world's rules allow its agents to do at all: a strategy its rules do
not allow can never evolve, however long the world runs. Each mechanism the
screen draws opens a channel, and most have a price.

| mechanism | what it makes possible | its price |
|---|---|---|
| messages (4 to 30 numbers to each [neighbour](the-world.md#the-network "An agent joined to another by a connection.")) | signals: agents can tell neighbours apart, warn, coordinate their stakes, recognise their kin by a code they learn | a code only works once senders and listeners have evolved it together |
| messages before the game | a round of talk before acting: a stake can answer what a neighbour has just said | — |
| noise inputs (2 to 10 random numbers a neighbour) | a brain can randomise. In a game like Colonel Blotto every fixed way of staking can be beaten by another, so chance in the stakes is worth having | two copies of one brain act differently: behaviour is less heritable |
| gifts | an agent can give tokens: help, payment, support of kin — the raw stuff of cooperation | what is given is lost to the giver |
| handing over | a parent can set its child up with some of its own [connections](the-world.md#the-network "A link between two agents of the network: they are neighbours, and can stake on each other's nodes."): a position is inherited | the parent loses those connections |
| mixing at birth (the mate a parent) | a child's brain from two: two lines can combine what each found | a mix can break what worked in each |
| the mate pays | two parents share a child's cost | — |
| search (jump, walk, reach) | agents choose partners beyond their neighbours: they can find allies, or prey, and move in the network | links made far away fold a world into a small world |
| a token in every link, or upkeep | links with a price: a link is kept only where it is worth it | poor agents can hold few links |
| the lottery | the weaker can win: a node goes to a staker drawn in proportion to its stake | chance decides more: selection is weaker |
| [coalitions](revolution.md "A node taken by stakers who each marked part of their stake revolutionary and together staked more than the largest single stake.") (revolutions) | the small can join against the largest staker | — |
| estates to the poor, the young, the family or the neighbours | where the wealth of the dead goes shapes who can have children next | — |
| a cap on links (valence) | no hubs, by rule | at 2, chains and rings only, and an agent at its cap has no child |
| a game in rounds, weakest or strongest first | agents see the board before each token and can answer each other's stakes | — |
| [mutation](mutation.md "A random change to a brain, made with a set probability when it is copied, and in B1 also after every game: a changed brain is a new genotype.") only when copied | a brain changes only when it is copied, into a child or into a node its owner wins, and never while it sits: a genotype lasts, and can be selected | — |
| larger brains (3 to 5 layers of 16 to 50) | more complicated behaviour can be represented | — |

## What could be played next

From now on only two kinds of world are played (the author, 7 November
2026): more worlds drawn at random as before, and worlds made to follow a
promising candidate. Five ways to go on were put to the author. On 8 November
2026 the author chose the second, larger: the [settings](settings.md "One number or switch of a run's rules, fixed before it starts.") of the best worlds
again, with new seeds, as worlds of 100,000 tokens, and more of them than the
eleven of the first shortlist, since more worlds say more. How many is the
next question for the author, as it is a choice between breadth, days and
disk:

1. **Play the shortlist on.** Every shortlisted world kept its last state, so
   the [lab](the-lab.md "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.") can play it on from iteration 3,000, to 10,000 say, without playing
   its first 3,000 again. Does its health hold? Does its line of descent keep
   reaching new behaviour, or come to rest? A typical world of round 3 took
   about two and a half hours for its 3,000 iterations on one worker, so 7,000
   more take most of a day each.
2. **Play the shortlist's settings again, with fresh seeds.** A world may be
   healthy by its settings or by the luck of its seed. Three new seeds for each
   shortlisted setting say which: a setting that gives a healthy, evolving
   world in two of three is one to keep (the author, 19 October 2026: a few
   seeds are enough to show that something is possible).
3. **Play the eight draws of the game in rounds once more, with every token
   placed at once**, so that each draw is seen in all three ways of staking
   ([A token at a time](how-the-drawn-settings-shape-a-world.md#a-token-at-a-time)).
4. **Draw a fourth round**, as round 3 was drawn but without what this round
   ruled out ([What could be set aside](how-the-drawn-settings-shape-a-world.md#what-could-be-set-aside-and-what-could-be-simplified)).
5. **Play each shortlisted world's eras against each other.** Every world of
   round 3 kept its whole state every 250 iterations: twelve eras. A contest
   between two eras plays their brains against each other in the same world
   ([Eras, contests and invasions](eras-and-contests.md)). If every later era
   beats every earlier one, the world keeps getting better; if the eras win
   and lose in bands, it goes round in circles. The notebook's programme calls
   this the test that a forgetful world cannot pass (`research/Research.md`,
   the cross-time tournament of Cliff and Miller, 1995). It needs no new world,
   only contests: in Experiment 28 one took about half a minute.

## What we learned

- **72 of round 3's hundred worlds remain candidates.** The others died, are a
  star, pass nothing on (all four have brains that change after every game) or
  stand still (all five are crowded and poor).
- **Evolution in round 3 is fast and narrow:** hundreds of generations, but a
  handful of lines, so chance rules which line survives.
- **Heredity and novelty pull against each other,** and health goes with
  novelty, against heredity.
- **Fertility runs in families in almost every world;** mending deaths weakens
  it.
- **No world leads in both health and evolution.** Seed 3092 is the
  healthiest; seed 3026, with simple rules, evolves the most, and loses health
  only to big culls.

## Where to go next

- [Which worlds are healthy? A proposed order](which-worlds-are-healthy.md):
  the health each candidate brings.
- [How the drawn settings shape a world](how-the-drawn-settings-shape-a-world.md):
  which settings give heredity, selection and change.
- [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md):
  the notebook's earlier tests of open-ended evolution.

## How this was made

The candidates are every world of round 3 of the screen that lived to the end
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
