# A world of equals

> [!question] Questions of this page
> - What is the world of equals that formed when mutations were made small?
> - Do [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") still get better inside it?
> - Does it form from founders too, and does it keep a small world alive?
> - Do large mutations first and small ones after give a world that lives
>   and keeps its order?
> - Does a world find its own step, if each brain inherits how much it
>   mutates?

<!-- experiment E54 -->
<!-- experiment E56 -->

## In short

When the mutations of one world were made small, it turned into something
no world of the notebook had been: a few hundred [agents](the-world.md#agents-and-tokens "One member of a world: it lives on a node of the network, holds tokens and has a brain that takes all its decisions.") who all hold about
the same [tokens](the-world.md#agents-and-tokens "The one thing of value in a world: every agent holds a whole number of them, and their total never changes.") and the same number of links, who are never born and never
die, and who all play the same way
([Does a smaller step restart evolution?](does-a-smaller-step-restart-evolution.md#what-experiment-53-found)).
With the [engine](runs-and-records.md#the-lab-the-engine-and-pilots "The program that runs an experiment's plan as runs, each on a frozen copy of the engine, the code that plays a world.")'s own mutation given back, it fell apart within 500
[iterations](one-iteration.md "One step of a world's time: a phase of reproduction, then a phase of the game.").

This page says what that world is and asks two things of it. Experiment 54
plays its later brains against its earlier ones: is it an end point, or do
brains still get better inside it? Experiment 56 starts the same rules from
founders with small mutations, at 40,000 tokens and at 10,000, where they
died out before.

Experiment 56 is played, and its thesis is refuted. Four of its five worlds
died within 71 to 1,322 iterations: small mutations from the start are too
slow for a world of random brains. The fifth became a world of equals
within 1,000 iterations, at 10,000 tokens. Experiment 57, not yet played,
starts with the engine's large mutations and switches to small ones at
iteration 500. Experiment 58, not yet played either, lets every brain
inherit its own step, so that a world can find the switch itself.
Experiment 54 is being played.

## What the world of equals is

It formed under the rules of round 3's world s3058: the largest stake wins
a node, unless a coalition of smaller stakers outweighs it
([How a coalition takes a node](rules-in-detail.md#how-a-coalition-takes-a-node));
an agent may walk the network to make a link with a stranger, who cannot
refuse; every link holds a token; a link nobody staked across is cut.

In the world of equals every agent does the same in every [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone."):

1. It puts one token on each of its [neighbours](the-world.md#the-network "An agent joined to another by a connection.") and one on itself. With 56
   links and 56 tokens that uses them all.
2. It marks every one of them for a coalition.
3. It has no child.

What follows from that:

- On every node the coalition of all the neighbours outweighs any single
  stake. It takes the node, and one of its members, drawn by lot, puts its
  brain there. Every node is won by a coalition in every game.
- Every node gets one token from each neighbour, so about as many as its
  agent gave away. Tokens stay even, and nobody starves.
- Every link carries a stake in both directions, so no link is cut and no
  part of the network is cut off.
- Nobody is born and nobody dies. The agents of iteration 23,000 are those
  of iteration 27,500. Their brains change places in every game: one node
  in fifty keeps its brain.

So the bodies last and the brains move through them. An agent that staked
two tokens on one node would not win it, because the other 55 stakers
together outweigh it. The way of playing protects itself, as long as nearly
everyone follows it.

Three things made it possible, as far as the experiments so far can say.
The coalition rule lets many small stakes beat a large one. The rule that
the largest stake wins makes poor agents lose to rich ones, so having
children does not pay (exact small games, `research/efficient/D2/RESULT.md`).
And small mutations keep a copy doing what its parent did.

It is the most even state its tokens can be in, and it changes less and
less. That could make it an end: order, but no more evolution.

## Do brains still get better inside it?

<!-- thesis E54 -->
> [!quote] The thesis of Experiment 54, written down before any of its runs existed
> **The claim.** In a world of equals brains no longer get better. The brains of a later iteration hold their own against those of an earlier one, and no more.
>
> **Why it would be so.** In Experiment 53's two runs with mutations of 10 numbers every agent does the same thing, a node's new brain is drawn by lot among its stakers, and from iteration 25,500 on the worlds hardly change. If every brain follows the same way of staking, none spreads better than another, and a contest between two moments of such a world is a tie. A score above a half would mean that brains still improve inside the order, which would make it more than an end point.
>
> **It holds if:** A coin is two contests with one deal, the two sides in swapped places; a score is the mean of its coins and its band two standard errors either side. Pooled over the two runs and both worlds of each pair, the later brains' share of the agents has a band that includes 0.5, for 25,000 against 23,000 and for 27,500 against 25,000.
>
> **It fails if:** For either pair the whole band lies above 0.5: brains still get better in a world of equals. A band wholly below 0.5 for either pair is reported as undecided, as is anything else. Reported beside it: 27,500 against 23,000, and each run alone.
<!-- /thesis -->

## Does it form from founders?

The first world of equals came from a world that had run 20,000 iterations
with the engine's mutation. Experiment 56 starts from founders, with
mutations of 10 numbers from the first iteration. At 10,000 tokens the same
rules with the engine's mutation died out in all three runs of Experiment
46, at 744, 1,029 and 2,630 iterations
([Why does evolution level off?](why-does-evolution-level-off.md)).

<!-- thesis E56 -->
> [!quote] The thesis of Experiment 56, written down before any of its runs existed
> **The claim.** With mutations of 10 numbers the rules of s3058 make a world of equals from founders, without the 20,000 iterations of history the first one had, and at 10,000 tokens too, where the same rules with the engine's mutation died out.
>
> **Why it would be so.** In Experiments 50 and 53 a world of 40,000 tokens that had run 20,000 iterations with the engine's mutation turned, once its mutations changed 10 numbers, into a world of equals within 600 to 5,000 iterations: a few hundred agents who all stake one token on every neighbour, marked for a coalition, with no births and no deaths. It fell apart within 500 iterations when the engine's mutation was given back. If that state is what these rules lead to whenever copies are faithful, it should form from founders as well, and it should keep a small world alive: under the engine's mutation all three runs at 10,000 tokens died out, at 744, 1,029 and 2,630 iterations, in culls that cut their network apart (Experiment 46), and in a world of equals no link is cut.
>
> **It holds if:** A run is a world of equals if, over its last 500 iterations, there are fewer than 0.01 births per agent and iteration and at least nine nodes in ten are won by a coalition. Both: all three runs at 10,000 tokens live to iteration 5,000; and at least four of the five runs are a world of equals.
>
> **It fails if:** Two or more of the three runs at 10,000 tokens die out before iteration 5,000, or fewer than two of the five runs are a world of equals. Anything else is undecided. Reported beside it: when each run became a world of equals, if it did, and its agents, links and Gini of tokens.
<!-- /thesis -->

### What Experiment 56 found

The five runs were played on 24 November 2026.

| run | lived to | what it became |
|---|---|---|
| 10,000 tokens, seed 5601 | 71 | died out |
| 10,000 tokens, seed 5602 | 347 | died out |
| 10,000 tokens, seed 5603 | 5,000 | a world of equals from iteration 1,000 on: 108 agents with about 61 links and 62 tokens each, every node won by a coalition, no births |
| 40,000 tokens, seed 5601 | 205 | died out |
| 40,000 tokens, seed 5602 | 1,322 | died out |

**The thesis is refuted**: two of the three runs at 10,000 tokens died out
before iteration 5,000, and only one run of five became a world of equals.

Small mutations from the first iteration are worse for a world's start
than the engine's. Four of five worlds died within 71 to 1,322 iterations;
under the engine's mutation the worlds of this size lived 744 to 2,630. A
world of founders holds brains drawn at random, and a step of ten numbers
is too slow to make them fit before the network breaks.

The one world that got through its start became a world of equals within
1,000 iterations, at 10,000 tokens, from founders. So that state does not
need the 20,000 iterations of history the first one had, nor 40,000 tokens.
It is where these rules go when copies are faithful, if the world lives
long enough to get there.

The two findings point the same way. Large steps find brains that live;
small steps keep what has been found. A world may need the first and then
the second.

## Large steps first, small steps after

<!-- experiment E57 -->

Experiment 57 tries exactly that, on the three worlds that died in
Experiment 46. Each is taken as it stood at iteration 500, after 500
iterations of the engine's mutation, and played on for 5,000 more: twice
with mutations of 10 numbers, and twice as it is.

<!-- thesis E57 -->
> [!quote] The thesis of Experiment 57, written down before any of its runs existed
> **The claim.** A world that starts with the engine's large mutations and goes on with small ones lives, where the same world left with the large ones dies. Played on from iteration 500 with mutations of 10 numbers, the three worlds of s3058 at 10,000 tokens that died out in Experiment 46 live 5,000 iterations more and become worlds of equals.
>
> **Why it would be so.** From founders, mutations of 10 numbers killed four of five worlds within 71 to 1,322 iterations (Experiment 56): founders' brains are drawn at random, and small steps are too slow to make them fit before the world breaks. The engine's mutation finds brains that live, but cannot keep a way of playing that needs everyone to follow it (Experiment 53). The world of equals so far formed twice from a world that had first run with large mutations, and once from founders. So large steps to begin with and small steps after should give both: a world that gets through its start, and then keeps its order. Under the engine's mutation these three worlds died at 744, 1,029 and 2,630 iterations.
>
> **It holds if:** A run is a world of equals if, over its last 500 iterations, there are fewer than 0.01 births per agent and iteration and at least nine nodes in ten are won by a coalition. Both: at least five of the six runs played on with 10 numbers live to iteration 5,500, and at least four of them are worlds of equals; and at most two of the six runs played on as they are live to 5,500.
>
> **It fails if:** Fewer than three of the six runs with 10 numbers live to iteration 5,500, or as many runs live as they are as with 10 numbers. Anything else is undecided.
<!-- /thesis -->

## A step each brain inherits

<!-- experiment E58 -->

Experiment 57 switches from large steps to small ones by hand, at an
iteration somebody chose. A world could find that moment itself, if how
much a mutation changes were the brain's own
([`inherited_step`](settings.md#inherited_step)). Every brain then carries
a step: the number of its numbers one mutation changes. A copy keeps its
parent's step; a mutation first doubles or halves it, by an even chance.
Founders start at a tenth of their numbers, which is what the engine's
mutation touches. No iteration and no step is set by hand.

Experiment 58 plays the rules of s3058 that way from founders: three worlds
at 10,000 tokens and two at 40,000, for 5,000 iterations.

<!-- thesis E58 -->
> [!quote] The thesis of Experiment 58, written down before any of its runs existed
> **The claim.** Where every brain inherits how much a mutation changes it, a world starts with large steps, comes to small ones by itself, and becomes a world of equals. It lives through its start, as worlds with small steps from the first iteration did not.
>
> **Why it would be so.** Large steps find brains that live: under the engine's mutation worlds of these rules lived hundreds to thousands of iterations from founders, where mutations of 10 numbers from the start killed four of five within 71 to 1,322 (Experiment 56). Small steps keep an order that large ones break: the world of equals lasted 4,500 iterations under mutations of 10 numbers and fell apart within 500 under the engine's (Experiment 53). If the step is inherited and itself doubles or halves at every mutation, lines with large steps should do better while brains are still poor, and lines with small steps should take over once there is something to keep: a mutant that decides like its parent keeps its place in a way of playing that needs everyone to follow it.
>
> **It holds if:** Founders start with a step of a tenth of their numbers. A run is a world of equals if, over its last 500 iterations, there are fewer than 0.01 births per agent and iteration and at least nine nodes in ten are won by a coalition. All three: at least four of the five runs live to iteration 5,000; in every run that lives, the middle step of its brains at iteration 5,000 is below a tenth of the founders' step; and at least three of the five runs are worlds of equals.
>
> **It fails if:** Fewer than three of the five runs live to iteration 5,000; or in a run that lives the middle step at 5,000 is as large as the founders' or larger. Anything else is undecided. Reported beside it: the middle step of each run in every world it kept, every 500 iterations.
<!-- /thesis -->

<!-- runs E58 -->
> [!info]- The runs behind this page
> **5 runs.** Every condition below is run once for each of its seeds, for 5,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **inherited step at 10,000 tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `inherited_step` = on, `n_nodes` = 200; 10,000 tokens; seeds 5801 to 5803. Runs `B1-10000-dc698b-s5801` … `B1-10000-dc698b-s5803`.
> - **inherited step at 40,000 tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `inherited_step` = on, `n_nodes` = 800; 40,000 tokens; seeds 5801 to 5802. Runs `B1-40000-2d4867-s5801` … `B1-40000-2d4867-s5802`.
>
> To make these runs again: `python3 gol_lab.py run E58`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

<!-- runs E54 -->
> [!info]- The trials behind this page
> **96 trials**, contests: the world of one era, its agents given brains drawn by coin from two eras, every mutation off, and the later era's share of the agents counted at the end; each plays 100 iterations, from a world another run kept in its archive.
>
> - **seed 5301: 25,000 against 23,000** — 16 trials: worlds `B1-40000-b7eb00-s5301-one41f90`; eras 23,000, 25,000; coins 5,401 to 5,408, every 1.
> - **seed 5301: 27,500 against 25,000** — 16 trials: worlds `B1-40000-b7eb00-s5301-one41f90`; eras 25,000, 27,500; coins 5,409 to 5,416, every 1.
> - **seed 5301: 27,500 against 23,000** — 16 trials: worlds `B1-40000-b7eb00-s5301-one41f90`; eras 23,000, 27,500; coins 5,417 to 5,424, every 1.
> - **seed 5302: 25,000 against 23,000** — 16 trials: worlds `B1-40000-b7eb00-s5302-one41f90`; eras 23,000, 25,000; coins 5,425 to 5,432, every 1.
> - **seed 5302: 27,500 against 25,000** — 16 trials: worlds `B1-40000-b7eb00-s5302-one41f90`; eras 25,000, 27,500; coins 5,433 to 5,440, every 1.
> - **seed 5302: 27,500 against 23,000** — 16 trials: worlds `B1-40000-b7eb00-s5302-one41f90`; eras 23,000, 27,500; coins 5,441 to 5,448, every 1.
>
> To play these trials again: `python3 gol_lab.py run E54`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

<!-- runs E57 -->
> [!info]- The runs behind this page
> **12 runs.** Every condition below is run once for every seed (5701 to 5702), played on from a kept world of its own to iteration 5,500 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **s4601, then 10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-01519c-s5701-onfbb674` … `B1-10000-01519c-s5702-onfbb674`.
> - **s4601, as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-122402-s5701-onfbb674` … `B1-10000-122402-s5702-onfbb674`.
> - **s4602, then 10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-01519c-s5701-onfad1ed` … `B1-10000-01519c-s5702-onfad1ed`.
> - **s4602, as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-122402-s5701-onfad1ed` … `B1-10000-122402-s5702-onfad1ed`.
> - **s4603, then 10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-01519c-s5701-on6669d4` … `B1-10000-01519c-s5702-on6669d4`.
> - **s4603, as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens. Runs `B1-10000-122402-s5701-on6669d4` … `B1-10000-122402-s5702-on6669d4`.
>
> To make these runs again: `python3 gol_lab.py run E57`.
<!-- /runs -->

<!-- runs E56 -->
> [!info]- The runs behind this page
> **5 runs.** Every condition below is run once for each of its seeds, for 5,000 iterations — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **10 numbers at 10,000 tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens; seeds 5601 to 5603. Runs `B1-10000-01519c-s5601` … `B1-10000-01519c-s5603`.
> - **10 numbers at 40,000 tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 800; 40,000 tokens; seeds 5601 to 5602. Runs `B1-40000-b7eb00-s5601` … `B1-40000-b7eb00-s5602`.
>
> To make these runs again: `python3 gol_lab.py run E56`.
<!-- /runs -->

## Where to go next

- [Does a smaller step restart evolution?](does-a-smaller-step-restart-evolution.md):
  Experiments 50 to 53, where the world of equals formed.
- [Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md):
  what one mutation does to a brain.

## How this was made

The plans are `book/experiments/E54.json`, `E56.json`, `E57.json` and
`E58.json`;
Experiment 56 is read by `python3 research/efficient/E50/e56.py`. What the agents
of the world of equals do is read from the statistics of Experiment 53's
runs by `python3 research/efficient/E50/e53.py`.
