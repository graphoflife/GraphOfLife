# A world of equals

> [!question] Questions of this page
> - What is the world of equals that formed when mutations were made small?
> - Do [brains](the-brain.md "The neural network each agent decides with: inputs about itself and its neighbours, outputs that become its decisions.") still get better inside it?
> - Does it form from founders too, and does it keep a small world alive?
> - Do large mutations first and small ones after give a world that lives
>   and keeps its order?
> - Does a world find its own step, if each brain inherits how much it
>   mutates?
> - Can a world of equals be invaded, and can its way of playing spread
>   into another world?

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

Five experiments of this page are played, and all five theses are
refuted. Experiment 60 looked for a step of mutation at which order forms
and breaks, and found it at 30 numbers; its thesis is undecided.
Experiment 59 asked whether a world of equals can be invaded: with half its
agents given other brains, those take most of its tokens; its thesis is
undecided. Experiment 61 played the two worlds of 30 numbers on: one formed
the order a second time and lost it again, the other died out; undecided.
Experiment 62 put a few changed brains into a world of equals: they do as
well as the rest, so the order is not kept by selection; refuted.

Experiment 63 played nine more worlds at steps of 20, 30 and 50 numbers
for 10,000 iterations: with 20 numbers all three formed the order and kept
it; going back and forth happened in none; and two worlds of equals died
out, suddenly.

Experiment 64 put one hoarder into a world of equals, an agent that stakes
everything on one node: it grows rich, up to seven tokens in ten of the
whole world, but no world of 30 ended within 1,000 iterations; refuted.

Taken together: the world of equals is a state these rules fall into when
copies are faithful enough. With a step of 10 or 20 numbers it lasts. With
larger steps it wears away, and a world that loses it, or never finds it,
often dies. One world formed it twice; that was the exception. What the five found together:

- **The world of equals forms again and again**, from a long-run world,
  from founders and from young worlds, at 10,000 and at 40,000 tokens. Six
  of the 20 runs with mutations of 10 numbers or an inherited step ended as
  one, in five separate worlds, and none of the 10 runs with the engine's
  mutation.
- **It lasts while steps stay small**, 4,500 iterations in those
  experiments. With mutations of 100 numbers one world reached it and left
  it again. (Corrected on 25 November 2026: this said it is safe once
  reached. Over 10,000 iterations two worlds of equals died out,
  Experiment 63.)
- **Getting there is a bet.** Most small worlds with small steps died
  sooner than they would have with large ones.
- **Brains still get better inside it for a while**, about 2,000
  iterations, and then stop.
- **A step that brains inherit shrinks at once**, to a handful of numbers.

In more detail: Experiment 56 is played, and its thesis is refuted. Four of its five worlds
died within 71 to 1,322 iterations: small mutations from the start are too
slow for a world of random brains. The fifth became a world of equals
within 1,000 iterations, at 10,000 tokens. Experiment 57 switched from the
engine's mutation to small ones at iteration 500: five of six worlds died
sooner than those left alone, and the sixth became a world of equals.
Experiment 58 let every brain inherit its step: the step fell to a handful
of numbers within 500 iterations in every world, and two of five worlds
lived. Experiment 54 played the world of equals against its own past: its
brains got better for about 2,000 iterations and then stopped.

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

### What Experiment 54 found

The 96 contests were played on 24 November 2026. The later brains' share of
the agents, pooled over the two worlds (16 coins each):

| later against earlier | share | band |
|---|---|---|
| 25,000 against 23,000 | 0.62 | 0.54 to 0.71 |
| 27,500 against 25,000 | 0.52 | 0.46 to 0.59 |
| 27,500 against 23,000 | 0.66 | 0.58 to 0.73 |

**The thesis is refuted**: for 25,000 against 23,000 the whole band lies
above a half. In a world of equals brains still got better, for about 2,000
iterations after it formed. From 25,000 to 27,500 they did not: a tie.

So the world of equals is not an end at once. It does what every world
measured so far did, one step of order higher: better for a while, then no
more. The stretch in which its brains improved is the one in which its
links and tokens evened out
([What Experiment 53 found](does-a-smaller-step-restart-evolution.md#what-experiment-53-found)).
The numbers are made by `python3 research/efficient/E50/e54.py`.

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

### What Experiment 57 found

The twelve runs were played on 24 November 2026. Each started at iteration
500 of a world that had died under the engine's mutation.

| world of Experiment 46 | then 10 numbers, two seeds | as it is, two seeds |
|---|---|---|
| seed 4601 (had died at 2,630) | died at 522 and 529 | died at 1,798 and 534 |
| seed 4602 (had died at 1,029) | died at 953 and 574 | died at 1,360 and 2,467 |
| seed 4603 (had died at 744) | **lived to 5,500**, a world of equals from 1,500 on; died at 740 | died at 2,987 and 2,667 |

**The thesis is refuted.** One of the six runs with small mutations lived;
the thesis asked for five. None of the six runs left as they were lived
either, but they lasted longer: the middle one died near iteration 2,000,
against 574 with small mutations.

Switching to small steps does not rescue a world of this size. It makes a
bet. Five times in six the world died sooner than it would have. Once it
became a world of equals within 1,000 iterations and was then safe: 272
agents at the end, no births, no deaths in its last 500 iterations.

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

### What Experiment 58 found

The five runs were played on 24 November 2026.

| run | lived to | middle step of its brains (founders: 1,357 numbers) | what it became |
|---|---|---|---|
| 10,000 tokens, seed 5801 | 112 | | died out |
| 10,000 tokens, seed 5802 | 4,549 | 4 at iteration 500, then 2 to 16 | died out |
| 10,000 tokens, seed 5803 | 1,206 | 212 at 500, 16 at 1,000 | died out |
| 40,000 tokens, seed 5801 | 5,000 | 40 at 500, then 4 to 32, 8 at the end | a world of equals from iteration 500 on |
| 40,000 tokens, seed 5802 | 5,000 | 8 at 500, and 8 to 16 after | about 2,000 agents with births and deaths; not a world of equals |

**The thesis is refuted**: two of the five runs lived to iteration 5,000,
and the thesis asked for four.

One part of the claim held, and strongly. **Given the choice, every world
chose small steps at once.** Within 500 iterations the middle step fell from
1,357 numbers to between 4 and 212, and it stayed at a handful. A line whose
mutations change little loses fewer of its copies, so it spreads. Nothing
pulls the step up again.

That is also why an inherited step did not help a world through its start:
the step is small before the brains are good. The worlds at 10,000 tokens
died as they did with small steps from the first iteration.

<!-- runs E58 -->
> [!info]- The runs behind this page
> **5 runs.** Every condition below is run once for each of its seeds, for 5,000 iterations — or until its world dies out. Every setting is that of the baseline **B1** (the algorithm exactly as a new run is offered it) unless the condition changes it ([At a glance](settings.md#at-a-glance)).
>
> - **inherited step at 10,000 tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `inherited_step` = on, `n_nodes` = 200; 10,000 tokens; seeds 5801 to 5803. Runs `B1-10000-dc698b-s5801` … `B1-10000-dc698b-s5803`.
> - **inherited step at 40,000 tokens** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `inherited_step` = on, `n_nodes` = 800; 40,000 tokens; seeds 5801 to 5802. Runs `B1-40000-2d4867-s5801` … `B1-40000-2d4867-s5802`.
>
> To make these runs again: `python3 gol_lab.py run E58`, or ▶ in the Research Notebook tab of a computer running `gol_server.py`. Each run writes its settings, seed and engine version beside its frames, in `GraphOfLifeRuns/<run>/meta.json` and `provenance.json`.
<!-- /runs -->

## The step between

<!-- experiment E60 -->

With mutations of 10 numbers the world of equals lasted. With the engine's
mutation it never formed. With 100 numbers it formed once and was lost
again. So between the step that keeps an order for good and the step that
never lets it form, there may be one at which orders keep forming and
breaking: a world that neither stands still nor only churns. Experiment 60
plays the kept world of Experiment 50 on with mutations of 30, 300 and
1,000 numbers, two runs each, for 5,000 iterations.

<!-- thesis E60 -->
> [!quote] The thesis of Experiment 60, written down before any of its runs existed
> **The claim.** The larger the step of a mutation, the less a world of equals lasts, and between the step that keeps it for good and the step that never lets it form there is one at which it forms and is lost again. Played on from the same kept world for 5,000 iterations: with mutations of 30 numbers the world becomes a world of equals and stays one; with 1,000 numbers it is not one at the end; with 300 numbers it is one for part of the time.
>
> **Why it would be so.** From this kept world, mutations of 10 numbers led to a world of equals in both runs, which then lasted; mutations of 100 numbers led to one in one run of two, which was lost again within 2,000 iterations; the engine's mutation, which changes about 1,360 numbers and redraws part of a layer, never led to one (Experiments 50 and 53). A way of playing that needs everyone to follow it is broken by copies that decide otherwise, and how many copies do so grows with the step. A world in which order keeps forming and breaking would be neither the churn of large steps nor the standstill of small ones.
>
> **It holds if:** Each run's 5,000 iterations are ten windows of 500. A window is one of a world of equals if it has fewer than 0.01 births per agent and iteration and at least nine nodes in ten are won by a coalition. All three: with 30 numbers, both runs have at least eight such windows among their last ten; with 1,000 numbers, neither run's last window is one; with 300 numbers, at least one run has between one and seven such windows.
>
> **It fails if:** With 30 numbers both runs have fewer than five such windows; or with 1,000 numbers both runs have more than five. Anything else is undecided. Reported beside it: for every run, which windows are of a world of equals, and how often it entered and left that state.
<!-- /thesis -->

### What Experiment 60 found

The six runs were played on 24 November 2026. Each line is one run's ten
windows of 500 iterations, from iteration 20,000 to 25,000; an `E` is a
window in which the world is a world of equals.

| mutation | seed 6001 | seed 6002 |
|---|---|---|
| 30 numbers | `.EEE......` | `.EEEEEE...` |
| 300 numbers | `..........` | `..........` |
| 1,000 numbers | `..........` | `....` and then died out, at iteration 21,863 |

**The thesis is undecided.** It expected 30 numbers to keep the world of
equals in eight windows of ten; the two runs kept it in three and in six.
It expected 300 numbers to have it for part of the time; neither run had
it at all. Only the claim for 1,000 numbers held.

What the runs show is plainer than the thesis:

- **With 30 numbers the order forms and is lost.** Both worlds were worlds
  of equals within 500 iterations, for 1,500 and for 3,000 iterations.
  Then births and deaths came back, and they ended with 3,408 and 1,682
  agents.
- **With 300 and with 1,000 numbers it never forms.**
- **The step at which order stops lasting lies between 10 and 30 numbers**,
  of the 13,569 a brain of this world holds. With 10 it lasted all 4,500
  iterations played (Experiment 53).
- One world with 1,000 numbers died out, at 40,000 tokens. Under the
  engine's mutation, which changes about as many numbers, worlds of this
  size died too (Experiment 46).


<!-- runs E60 -->
> [!info]- The runs behind this page
> **6 runs.** Every condition below is run once for every seed (6001 to 6002), played on from the world `B1-40000-bbbccd-s4602` kept at iteration 20,000, to iteration 25,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **30 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-8020c3-s6001-on45a3d3` … `B1-40000-8020c3-s6002-on45a3d3`.
> - **300 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 300, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-0c07af-s6001-on45a3d3` … `B1-40000-0c07af-s6002-on45a3d3`.
> - **1,000 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 1,000, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-c745b3-s6001-on45a3d3` … `B1-40000-c745b3-s6002-on45a3d3`.
>
> To make these runs again: `python3 gol_lab.py run E60`.
<!-- /runs -->

### Does order form again?

<!-- experiment E61 -->

So 30 numbers is a step at which order forms and breaks. What a world
that keeps moving would need is that it forms again. Experiment 61 plays
the two worlds on for 10,000 iterations to see. (It was planned on 24
November when three of Experiment 60's six runs were done, both runs with
30 numbers among them.)

<!-- thesis E61 -->
> [!quote] The thesis of Experiment 61, written down before any of its runs existed
> **The claim.** With mutations of 30 numbers a world keeps forming a world of equals and losing it. Played on for 10,000 iterations more, each of the two worlds that formed one and lost it in Experiment 60 forms one again.
>
> **Why it would be so.** In Experiment 60 both worlds with mutations of 30 numbers became worlds of equals within 500 iterations, kept that for 1,500 and 3,000 iterations, and lost it, ending as worlds of 1,700 and 3,400 agents with births and deaths. The world they had started from was such a world too, and it took them 500 iterations to leave it. If nothing in a world that has lost the order stops it from forming again, it should form again, and a world of this step would go on between order and churn without settling in either.
>
> **It holds if:** Each run's 10,000 iterations are twenty windows of 500. A window is one of a world of equals if it has fewer than 0.01 births per agent and iteration and at least nine nodes in ten are won by a coalition. Both runs have at least two such windows.
>
> **It fails if:** Neither run has such a window. Anything else is undecided. Reported beside it: for each run, which windows are of a world of equals, how often it entered and left that state, and its agents in every window.
<!-- /thesis -->

#### What Experiment 61 found

The two runs were played on 24 November 2026. Twenty windows of 500
iterations each, from iteration 25,000 to 35,000; an `E` is a window of a
world of equals.

| world | windows | what happened |
|---|---|---|
| from the run of seed 6001 | `.................` | never formed one again; **died out at iteration 33,061** |
| from the run of seed 6002 | `.............EE.....` | formed one again at iteration 31,500, for 1,000 iterations, and lost it again |

**The thesis is undecided**: one world formed the order a second time, the
other did not.

- **The order can form again.** The second world went from about 4,000
  agents to 335, with nine nodes in ten won by a coalition and almost no
  births, 6,500 iterations after it had lost its first world of equals. A
  thousand iterations later it was a world of 4,000 again. It is the first
  world of the notebook that goes back and forth between two kinds of world
  and settles in neither.
- **A world of this step can also die.** The first world held about 4,300
  agents with no births and hardly a node won by a coalition: neither
  order nor the usual churn, but a world in which nobody has children and
  nobody shares. It shrank from iteration 31,000 on and died out 2,000
  iterations later, at 40,000 tokens.

One world and two formations are a hint. Whether worlds of this step keep
going back and forth, and how many die on the way, needs more of them.

<!-- runs E61 -->
> [!info]- The runs behind this page
> **2 runs.** Every condition below is run once for every seed (6101), played on from a kept world of its own to iteration 35,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **30 numbers, from the run of seed 6001** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens. Run `B1-40000-8020c3-s6101-on697699`.
> - **30 numbers, from the run of seed 6002** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens. Run `B1-40000-8020c3-s6101-onb1e06c`.
>
> To make these runs again: `python3 gol_lab.py run E61`.
<!-- /runs -->

## Can it be invaded, and can it invade?

<!-- experiment E59 -->

A world of equals lasts while everyone in it plays the same way. Experiment
59 asks what happens when half its agents do not: it gives half the agents
of a world of equals the brains of the world that fell back under the
engine's mutation, and half the agents of that world the brains of the
world of equals. It also plays the two worlds of equals against each other.

<!-- thesis E59 -->
> [!quote] The thesis of Experiment 59, written down before any of its runs existed
> **The claim.** The way of playing of the world of equals holds its own ground and cannot take another's. In a world of equals, brains of the world that fell back under the engine's mutation lose to its own; in that other world, the brains of the world of equals do not win.
>
> **Why it would be so.** In a world of equals every agent stakes one token on every neighbour and marks it for a coalition, so on every node the many small stakes together outweigh any single one. A newcomer that keeps half its tokens at home and has children, as the brains of the engine's world do, is outweighed on every node it stakes on while most of its neighbours follow the common way. In the engine's world most agents keep their tokens at home and few nodes are won by a coalition; there a brain that spreads its tokens one to a neighbour gives them away to stakers who do not give back. If so, the world of equals is a state a world must fall into together, and cannot be spread from a few.
>
> **It holds if:** A coin is two contests with one deal, the two sides in swapped places; a score is the mean of its coins and its band two standard errors either side. Pooled over the two seeds, the share of the agents held by the brains of the world of equals after 100 iterations: on the worlds of equals, the whole band is above 0.5; on the engine's worlds, the band is not wholly above 0.5.
>
> **It fails if:** On the worlds of equals the whole band is below 0.5: its own brains lose at home. Anything else is undecided. Reported beside it: the shares of the tokens, and the two worlds of equals against each other.
<!-- /thesis -->

### What Experiment 59 found

The 48 contests were played on 24 November 2026. The share held after 100
iterations by the brains of the world of equals, when half the agents of a
world were given them and half the brains of the world that fell back
under the engine's mutation (8 coins each, both seeds pooled):

| played on | share of the agents | share of the tokens |
|---|---|---|
| the worlds of equals | 0.59 (0.36 to 0.83) | 0.36 (0.24 to 0.48) |
| the engine's worlds | 0.29 (0.12 to 0.45) | 0.24 (0.12 to 0.36) |

**The thesis is undecided.** Its second half holds: in the engine's world
the brains of the world of equals lose. Its first half does not: at home
they do not clearly hold their ground. They end with somewhat more than
half of the agents, but the band includes a half, and they end with about
a third of the tokens, with the whole band below a half.

So a world of equals is not safe against any number of others. With half
its agents playing another way, the others take most of the tokens within
100 iterations. The many small stakes outweigh a large one only while
nearly all neighbours make them. This fits what Experiment 53 saw from the
other side: one copy in five changed by the engine's mutation was enough to
end the order within 500 iterations.

The two worlds of equals against each other: the brains of the first hold
0.42 of the agents, on their own world and on the other's alike. The two
play nearly the same way, and neither world gives its own brains an
advantage. The numbers are made by
`python3 research/efficient/E50/e59.py`.

<!-- runs E59 -->
> [!info]- The trials behind this page
> **48 trials**, contests: the world of one era, its agents given brains drawn by coin from two eras, every mutation off, and the later era's share of the agents counted at the end; each plays 100 iterations, from a world another run kept in its archive.
>
> - **two worlds of equals** — 16 trials: worlds `B1-40000-b7eb00-s5301-one41f90`, `B1-40000-b7eb00-s5302-one41f90`; eras 27,500; coins 5,901 to 5,908, every 1.
> - **equals against the engine's world, seed 5301** — 16 trials: worlds `B1-40000-b7eb00-s5301-one41f90`, `B1-40000-bbbccd-s5301-one41f90`; eras 27,500; coins 5,909 to 5,916, every 1.
> - **equals against the engine's world, seed 5302** — 16 trials: worlds `B1-40000-b7eb00-s5302-one41f90`, `B1-40000-bbbccd-s5302-one41f90`; eras 27,500; coins 5,917 to 5,924, every 1.
>
> To play these trials again: `python3 gol_lab.py run E59`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
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
> To play these trials again: `python3 gol_lab.py run E54`.
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

### Is going back and forth what such worlds do?

<!-- experiment E63 -->

One world formed the order twice; that is one world. Experiment 63 plays
nine: the kept world of Experiment 50 played on for 10,000 iterations with
mutations of 20, 30 and 50 numbers, three runs each. The world is taken
from the copy that Experiment 50's first run kept of it at iteration
20,000, the same agents, brains and links, because the drive that holds
the original was not plugged in.

<!-- thesis E63 -->
> [!quote] The thesis of Experiment 63, written down before any of its runs existed
> **The claim.** A world whose mutations change 20 to 50 numbers does not settle. It forms a world of equals, loses it, and forms it again, and most such worlds live through that.
>
> **Why it would be so.** A world of equals is not kept by selection: a few brains that play otherwise do as well as the rest (Experiment 62), so it wears away as changed brains pile up, faster the larger the step. With 10 numbers it lasted all 4,500 iterations played; with 30 numbers both worlds lost it after 1,500 and 3,000 (Experiment 60). Played on, one of those formed it a second time 6,500 iterations later and lost it again, and the other died out (Experiment 61). If the order is a state these rules fall into whenever copies are faithful enough, a world that has lost it should fall into it again, and a step in this range should give worlds that go on between order and churn.
>
> **It holds if:** Each run's 10,000 iterations are twenty windows of 500. A window is one of a world of equals if it has fewer than 0.01 births per agent and iteration and at least nine nodes in ten are won by a coalition; a stretch is one or more such windows in a row. Both: at least three of the nine runs have two or more separate stretches; and at most three of the nine die out.
>
> **It fails if:** No run has two separate stretches, or five or more of the nine die out. Anything else is undecided. Reported beside it: every run's windows, its stretches and their lengths, by the step of its mutations.
<!-- /thesis -->

#### What Experiment 63 found

The nine runs were played on 24 and 25 November 2026. Twenty windows of 500
iterations each, from iteration 20,000 to 30,000; an `E` is a window of a
world of equals.

| mutation | seed 6301 | seed 6302 | seed 6303 |
|---|---|---|---|
| 20 numbers | `.EEEEEEEEEEEEEEEEEEE` | `.EEEEEEEEEEEEEEEEEEE` | `.EEEEEEEEEEEEEEEEEEE` |
| 30 numbers | `.EEE.EEEEEEEEEEEEEEE` | `.EEEEEEEEEEEEEEEEEE`, then died out at 29,351 | `..............`, then died out at 26,699 |
| 50 numbers | `....................` | `..EEEEE.`, then died out at 23,853 | `..EEEEEEEEEEEEEEEEEE` |

**The thesis is undecided, and the runs speak against its claim.** One run
of nine has two separate stretches, and that is one window's gap in an
order that otherwise held. The thesis asked for three. Three runs died out,
which is as many as it allowed.

- **Going back and forth is not what these worlds do.** A world forms the
  order within 500 to 1,000 iterations and keeps it, or it never forms it.
  The world of Experiment 61 that formed it twice is the exception.
- **With 20 numbers all three worlds formed the order and kept it** for
  9,500 iterations.
- **A world of equals can die, and suddenly.** Two worlds were worlds of
  equals, for 2,500 and for 9,000 iterations, with 536 and about 590
  agents, and were gone a few hundred iterations later. Until this
  experiment no world of equals had died; none had been played longer than
  5,000 iterations. What the page said before, that the order is safe once
  reached, does not hold over 10,000 iterations.
- **The larger the step, the more worlds die**: none of three with 20
  numbers, two of three with 30, one of three with 50; and of the five
  worlds with 30 numbers of Experiments 60, 61 and 63 together, three.

How the two worlds of equals died can be read from their last iterations.
In the world with 30 numbers, 628 agents held about 46 tokens each, the
richest 135. A hundred iterations before the end one agent held 220, then
979, then 1,686. Links were then cut by the hundred in every game, an
agent's links fell from 37 to 3, pieces of the network were culled, and
ten agents were left. The world with 50 numbers already had an agent with
3,556 tokens while it was still in the order.

That fits one way of breaking the order. Where everyone stakes one token on
every neighbour, an agent that puts all its tokens on one neighbour's node
outweighs the coalition there by one, takes the node and everything staked
on it, and can do the same in the next game with twice as much. It grows
richer in every game, the nodes it leaves are staked on by nobody, and the
network round it falls apart. This reading is from two worlds and has not
been tested. A mutation that turns a brain into such a player is rare when
steps are small, and that would be why the order lasts longer then.

(Added on 25 November 2026: [Experiment 64](#does-one-hoarder-end-it) tested
this reading. Such a player does grow rich, as reckoned, but the world
lived on in all 30 trials. One hoarder alone does not explain these
deaths.)

So a step of 20 numbers gives order that lasts, and steps of 30 and more
give worlds that mostly end, in the order or out of it. A step at which a
world lives on while its order comes and goes was not found.

<!-- runs E63 -->
> [!info]- The runs behind this page
> **9 runs.** Every condition below is run once for every seed (6301 to 6303), played on from the world `B1-40000-bbbccd-s5001-on45a3d3` kept at iteration 20,000, to iteration 30,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **20 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 20, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-c8c9f2-s6301-on210139` … `B1-40000-c8c9f2-s6303-on210139`.
> - **30 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-8020c3-s6301-on210139` … `B1-40000-8020c3-s6303-on210139`.
> - **50 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 50, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-3f4b02-s6301-on210139` … `B1-40000-3f4b02-s6303-on210139`.
>
> To make these runs again: `python3 gol_lab.py run E63`.
<!-- /runs -->

### Does one hoarder end it?

<!-- experiment E64 -->

The reading above, that one agent which stakes everything on one node
breaks the order, came from the last iterations of two worlds. Experiment
64 tests it directly. In a world of equals one agent is given its own
brain with only its stakes changed: everything on its own node, or
everything on the node of its poorest neighbour. Its child, its links, its
share for the coalition and its messages stay as its brain decides them,
so its neighbours hear what they heard before. The worlds are the three
with mutations of 20 numbers of Experiment 63, taken at iteration 27,500;
all three kept the order for the 2,500 iterations they were played after
that.

<!-- thesis E64 -->
> [!quote] The thesis of Experiment 64, written down before any of its runs existed
> **The claim.** One agent that stakes everything on one node ends a world of equals. It grows richer in every game until it holds more than a tenth of all the world's tokens, and the world then loses more than half its agents or dies out. It does so whether the node is its own or its poorest neighbour's.
>
> **Why it would be so.** Two worlds of equals died suddenly in Experiment 63, each with one agent that grew rich in its last hundred iterations. In a world of equals every agent stakes one or two tokens on each neighbour and on itself, all marked for the coalition, and holds about as many tokens as are staked on its own node (read from the stakes recorded at iteration 29,999 in the first of the three worlds played here: among 455 agents the largest stake on any node is 1 to 3 tokens, and 95 nodes in 100 are won by the coalition). A coalition takes a node only if its marked tokens outweigh the largest single stake there. An agent that puts all its tokens on one node is that largest stake, and all the others together stake about as much as it holds: by those recorded stakes it would keep its own node in 401 cases of 455 and take its poorest neighbour's in 360. The node keeps everything staked on it, so the hoarder's tokens grow by what its neighbours stake, about one token a neighbour in every game, while it pays nothing back. Nothing in the order answers that: its brains were never selected against such a player, and a few who play otherwise do as well as the rest (Experiment 62).
>
> **It holds if:** A trial takes one agent of a world of equals and gives it its own brain, frozen, with only its stakes written over; it plays 1,000 iterations, or until the hoarder's line is gone or the world has died out. A hoarder takes off if its line at some iteration holds more than 4,000 tokens, a tenth of the world's 40,000. A world ends if it dies out or has fewer than half its first agents at the trial's last iteration. Of the 30 hoarder trials (three worlds, five places, two kinds) at least 20 take off and at least 20 worlds end, and each kind alone has at least 8 of its 15 worlds ending.
>
> **It fails if:** Fewer than 10 of the 30 hoarders take off, or fewer than 10 of the 30 worlds end. Anything else is undecided. Reported beside it, by kind and by world: how many hoarders lose their first game, the iteration at which a hoarder passes 4,000 tokens, its tokens, the world's agents and links over time, and how long the line of the control lasts, the same agent with its brain frozen and unchanged.
<!-- /thesis -->

**The thesis is refuted.** Its first half holds and its second does not.
22 of the 30 hoarders took off, where the thesis asked for 20. No world of
the 30 ended, where it asked for 20, and fewer than 10 refutes it.

- **The move works as reckoned.** No hoarder lost its first game. One that
  stakes everything at home nearly doubled its tokens in the first game, by
  1.9 to 2.0 times in all 15 trials: it keeps its own tokens and gets what
  its neighbours stake on it.
- **It grows very rich.** The richest held 28,118 of the world's 40,000
  tokens after 1,000 iterations, having started with 109. In the first
  world all ten hoarders passed 4,000 tokens within 37 to 137 iterations.
- **The world lives on.** No world died out. In the 24 trials played to
  iteration 1,000 the [median](statistics.md#many-worlds-at-once-the-median-bands-the-settled-life-and-the-spread "The middle value: half of the numbers lie below it and half above.") world kept all its agents with a hoarder at
  home and 97 in 100 with a moving one; the least was 73 in 100. It lost
  links: 12 in 100 with a hoarder at home, 24
  in 100 with a moving one, as medians, and at most 64 in 100. The hardest
  hit went from 484 agents to 353 and from 10,894 links to 3,922, with the
  hoarder holding seven tokens in ten.
- **A hoarder at home can be cut off.** Five of the 15 lost their line
  between iterations 441 and 635, holding 1,029 to 10,705 tokens. In their
  last games each gained one to three tokens a game, as from one or two
  links left, and then the hoarder and one agent of the count were gone
  together. That fits the
  world's own rules: a link that no tokens cross lapses, an agent with no
  link to the rest is culled, and what it held goes back to the world. The
  trials do not record who staked on whom, so this is a reading.
- **The moving hoarder does better than the one at home.** All 15 took
  off, against 7 of 15 at home, and one line of 15 ended.
- **The control is gone at once.** The same agent's brain, frozen and
  unchanged, lasted a median of 2 iterations as a line, because in a world
  of equals the brain on a node is drawn anew from the coalition in every
  game. One control line drifted to 47 copies before it ended at iteration 81.

| world of Experiment 63 (20 numbers) | agents | links each | kind | take off | tokens at 100 | at 500 | at 1,000 | links left at 1,000 |
|---|---|---|---|---|---|---|---|---|
| seed 6301 | 484 | 45 | at home | 5 of 5 | 7,326 | 12,498 | 15,461 | 74% |
| | | | moving | 5 of 5 | 5,686 | 13,527 | 19,689 | 61% |
| seed 6302 | 257 | 103 | at home | 2 of 5 | 1,636 | 3,413 | 5,389 | 88% |
| | | | moving | 5 of 5 | 1,837 | 5,772 | 9,294 | 77% |
| seed 6303 | 533 | 49 | at home | 0 of 5 | 522 | 1,278 | 1,826 | 96% |
| | | | moving | 5 of 5 | 997 | 4,342 | 9,281 | 78% |

Tokens and links are medians of the trials still played at that iteration.

**What this says about the two worlds that died.** One hoarder is not
enough. In Experiment 63 a world was gone about a hundred iterations after
its first rich agent, with links cut by the hundred in every game. Here a
hoarder far richer than that one costs the world a few links a game, and
after 1,000 iterations most of the network stands. So the rich agent of
those last iterations was part of a wider breakdown and not its one cause.
What else failed in those worlds is not known. Three limits of this test:
the hoarder's line here is one agent, where a mutant's line could spread
before it turned; the worlds here mutate by 20 numbers, those by 30 and
50; and at iteration 1,000 the hoarders were still gaining, a sixth to a
fifth more tokens over the last 250 iterations, so what a world looks like
after 5,000 iterations with one is not shown.

The numbers are made by `python3 research/efficient/E50/e64.py`.

<!-- runs E64 -->
> [!info]- The trials behind this page
> **45 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 1,000 iterations, from a world another run kept in its archive.
>
> - **everything at home** — 15 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies home-hoarder; patches of 1 agent; places 1 to 5, every 1.
> - **everything on the poorest neighbour** — 15 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies moving-hoarder; patches of 1 agent; places 1 to 5, every 1.
> - **controls** — 15 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies resident; patches of 1 agent; places 1 to 5, every 1.
>
> To play these trials again: `python3 gol_lab.py run E64`.
<!-- /runs -->

### A few who play otherwise

<!-- experiment E62 -->

Half a world playing otherwise is more than a world of equals can bear.
What it meets in its own life is a few: a copy changed by a mutation now
and then. Experiment 62 marks 100 changed agents and 100 unchanged ones in
each of the two worlds of equals and counts their copies over nine
iterations, as Experiment 49 did in other worlds
([Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md)).

<!-- thesis E62 -->
> [!quote] The thesis of Experiment 62, written down before any of its runs existed
> **The claim.** A world of equals holds against a few who play otherwise. A line whose brain is changed in 1,000 of its numbers leaves fewer copies there than a line left as it is; a line changed in 10 numbers, the step that world mutates by, leaves as many.
>
> **Why it would be so.** With half its agents given other brains a world of equals lost most of its tokens to them (Experiment 59), yet it lasted 4,500 iterations while one copy in five was changed in 10 numbers (Experiment 53). So it should resist deviants while they are few: in a world where every node is won by the coalition of all its stakers, a brain that stakes otherwise is outweighed on every node, and one that keeps its tokens at home or has a child gives up its share of its neighbours' nodes. A change of 10 numbers alters about one decision in a hundred and should cost nothing; a change of 1,000 alters a fifth to two fifths.
>
> **It holds if:** In each of the two worlds, 100 mutant sites and 100 control sites are marked in each of eight assays of 9 iterations. R is the mean count of the mutant sites after six iterations over that of the control sites, pooled over a world's eight assays, and its range is that of 95% of 2,000 shufflings of the labels within each assay. In both worlds: R for 1,000 numbers lies below its shuffled range, and R for 10 numbers lies inside its own.
>
> **It fails if:** In both worlds R for 1,000 numbers lies inside its shuffled range or above it: a few who play otherwise do as well as the rest. Anything else is undecided. Reported beside it: R for 100 numbers, R after three and after nine iterations, and the share of lines still alive.
<!-- /thesis -->

#### What Experiment 62 found

The 48 assays were played on 24 November 2026. R is the mean count of the
changed lines over that of the unchanged ones after six iterations, with
the range of 95% of 2,000 shufflings of the labels.

| world of equals | 10 numbers changed | 100 | 1,000 |
|---|---|---|---|
| seed 5301 | 1.02 (0.79 to 1.27) | 1.11 (0.79 to 1.27) | 1.07 (0.79 to 1.28) |
| seed 5302 | 1.13 (0.79 to 1.28) | 1.20 (0.77 to 1.28) | 0.99 (0.78 to 1.28) |

**The thesis is refuted.** In both worlds a line changed in 1,000 numbers
leaves as many copies as an unchanged one. After three and after nine
iterations it is the same, and as many of its lines are still alive.

So a world of equals does not push out the few who play otherwise. A node's
new brain is drawn by lot among its stakers, and a brain changed in 1,000
numbers still mostly stakes as the others do, so it is drawn as often.
Nothing selects against it in the short run.

That changes how the order should be understood. It is not kept by
selection. It is a state the world falls into and then wears away from, as
changed brains pile up unchecked. How fast it wears is set by the step of a
mutation: with 10 numbers it lasted all 4,500 iterations played, with 30
numbers 1,500 and 3,000, with the engine's mutation fewer than 500. And
with half the agents replaced at once it does not hold at all (Experiment
59).

<!-- runs E62 -->
> [!info]- The trials behind this page
> **48 trials**, assays: in the world of one era some agents' brains are changed once and marked, as many more only marked, every mutation off, and the agents carrying each mark counted at the end; each plays 9 iterations, from a world another run kept in its archive.
>
> - **the world of equals, seed 5301** — 24 trials: worlds `B1-40000-b7eb00-s5301-one41f90`; at iteration 27,500; mutations of 10, 100, 1,000; mutants 100; controls 100; seeds 1 to 8, every 1.
> - **the world of equals, seed 5302** — 24 trials: worlds `B1-40000-b7eb00-s5302-one41f90`; at iteration 27,500; mutations of 10, 100, 1,000; mutants 100; controls 100; seeds 1 to 8, every 1.
>
> To play these trials again: `python3 gol_lab.py run E62`.
<!-- /runs -->

## Where to go next

- [Does a smaller step restart evolution?](does-a-smaller-step-restart-evolution.md):
  Experiments 50 to 53, where the world of equals formed.
- [Are mutants as fit as their parents?](are-mutants-as-fit-as-their-parents.md):
  what one mutation does to a brain.

## How this was made

The plans are `book/experiments/E54.json`, `E56.json`, `E57.json` and
`E58.json`;
Experiments 56, 57 and 58 are read by
`python3 research/efficient/E50/e56.py` (with `E57`, `E58`, `E60`, `E61` or
`E63` after it); the last iterations of the worlds that died by
`python3 research/efficient/E50/autopsy.py`; Experiment 64 by
`python3 research/efficient/E50/e64.py`. What the agents
of the world of equals do is read from the statistics of Experiment 53's
runs by `python3 research/efficient/E50/e53.py`.
