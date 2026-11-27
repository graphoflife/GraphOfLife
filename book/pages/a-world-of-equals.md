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

The [frames](runs-and-records.md#what-a-run-records "A whole world as a run records it after one phase: every agent, its tokens and connections.") of the two worlds that died show who took them: one line of
brains that stake everything on one node, grown from one agent to nearly
all within 85 and 200 iterations. Experiment 65 played such a line by
hand, a hoarder that stakes on all its [neighbours](the-world.md#the-network "An agent joined to another by a connection.")' nodes at once when it
is rich enough: its line took half the world in all 30 trials and 25
worlds ended; confirmed. The world of equals has no defence against it.

Experiment 66 played those 30 trials again under each of three rules that
wealth should not break, chosen by the author. A coalition counted by
heads stops the hoarder in its first [game](one-iteration.md#phase-2-the-game "The second phase of an iteration: every agent stakes all its tokens on its own node and its neighbours', and every node goes to someone.") in all 30 trials, and a world
of equals plays under it as before. A limit on a stake does not stop it
(25 of 30 lines took half the world). Tokens that level across links only
slow it: 11 lines took half the world, 3.5 to 22 times later than
without the rule. The
thesis had levelling stop it, so it is undecided.

The hoarder of Experiment 66 was written by hand. Experiment 67 got back
the lines that evolution made: both worlds that died were played again
from a world they had kept, and came out the same, game for game;
confirmed. Experiment 68 played them on from inside the takeover. Under
the old rule the line took the world again in all 30 trials and 27 worlds
ended. Under a head count no line took it and no world ended, even where
the line already held half of all tokens; confirmed.

Three experiments then asked what whole worlds do under a head count.
Experiment 69 was to play six for 10,000 iterations and was stopped in
its first run, because it would have taken two days; it is not judged.
Experiment 70 played the nine worlds of Experiment 63 again for 1,500
iterations with the rule on: the order formed in two of them, where it
had formed in seven without the rule, and five stayed large and sparse
with one agent holding a tenth of all tokens; confirmed. Experiment 71
played four worlds of equals on under the rule for 7,500 iterations, two
of which had died without it: all four lived, but three had times of many
births while coalitions still won nearly every node; confirmed. So the
head count protects the order where it is and mostly keeps it from
forming where it is not.

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

In the world of equals every agent does the same in every game:

1. It puts one token on each of its neighbours and one on itself. With 56
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
> **2 runs.** Every condition below is run once for every seed (6101), played on or again from a kept world of its own to iteration 35,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **30 numbers, from the run of seed 6001** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens; played on from `B1-40000-8020c3-s6001-on45a3d3` at iteration 25,000. Run `B1-40000-8020c3-s6101-on697699`.
> - **30 numbers, from the run of seed 6002** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens; played on from `B1-40000-8020c3-s6002-on45a3d3` at iteration 25,000. Run `B1-40000-8020c3-s6101-onb1e06c`.
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
> **12 runs.** Every condition below is run once for every seed (5701 to 5702), played on or again from a kept world of its own to iteration 5,500 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **s4601, then 10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens; played on from `B1-10000-122402-s4601` at iteration 500. Runs `B1-10000-01519c-s5701-onfbb674` … `B1-10000-01519c-s5702-onfbb674`.
> - **s4601, as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens; played on from `B1-10000-122402-s4601` at iteration 500. Runs `B1-10000-122402-s5701-onfbb674` … `B1-10000-122402-s5702-onfbb674`.
> - **s4602, then 10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens; played on from `B1-10000-122402-s4602` at iteration 500. Runs `B1-10000-01519c-s5701-onfad1ed` … `B1-10000-01519c-s5702-onfad1ed`.
> - **s4602, as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens; played on from `B1-10000-122402-s4602` at iteration 500. Runs `B1-10000-122402-s5701-onfad1ed` … `B1-10000-122402-s5702-onfad1ed`.
> - **s4603, then 10 numbers** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 10, `n_nodes` = 200; 10,000 tokens; played on from `B1-10000-122402-s4603` at iteration 500. Runs `B1-10000-01519c-s5701-on6669d4` … `B1-10000-01519c-s5702-on6669d4`.
> - **s4603, as it is** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `n_nodes` = 200; 10,000 tokens; played on from `B1-10000-122402-s4603` at iteration 500. Runs `B1-10000-122402-s5701-on6669d4` … `B1-10000-122402-s5702-on6669d4`.
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
What else failed in those worlds was not known when this was written; the
next section reads it from their frames. Three limits of this test:
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

### One line took the worlds that died

A frame records every brain's parent, so the family of the rich agents in
the two worlds that died can be followed back: a line is a brain and every
brain copied from it, changed or not. This reads what Experiment 63
recorded and plays nothing.

| world with 30 numbers, iteration | agents | of the line | its share of all tokens | links each: line, others |
|---|---|---|---|---|
| 29,270 | 674 | 1 | 1% | 7, 37 |
| 29,290 | 676 | 22 | 13% | 28, 35 |
| 29,320 | 767 | 127 | 53% | 15, 22 |
| 29,340 | 991 | 835 | 96% | 6, 6 |
| 29,348 | 416 | 407 | 98% | 3, 2 |
| 29,350 | 10 | | | |

| world with 50 numbers, iteration | agents | of the line | its share of all tokens | links each: line, others |
|---|---|---|---|---|
| 23,400 | 531 | 8 | 5% | 5, 42 |
| 23,500 | 522 | 12 | 39% | 21, 36 |
| 23,540 | 488 | 95 | 78% | 14, 20 |
| 23,560 | 329 | 327 | 100% | 5, 2 |
| 23,620 | 159 | 159 | 100% | 3, – |

- **One line, from one brain.** At the first iteration with five rich
  agents (more than five median holdings each), all of them come from one
  brain. It was first seen at iteration 29,262 in the world with 30
  numbers, 88 iterations before that world's end, and at 23,359 in the
  other. From then on nearly every rich agent is of that line.
- **The line takes the whole world**, from one agent to more than nine in
  ten within 85 iterations in one world and 200 in the other.
- **It spreads in the game and hardly by births.** 5,007 nodes turned to
  the line in a game and 542 agents were born into it in one world; 2,363
  and 45 in the other. Every such node but one in each world was next to an
  agent of the line just before.
- **Its agents stake everything on one node.** One game of the takeover has
  its stakes recorded, iteration 23,500. All twelve agents of the line
  stake everything on one node, six on their own and six on a neighbour's,
  and win all twelve. They hold 11 to 2,591 tokens. Of the 510 others two
  stake on one node.
- **But that alone cannot make a line grow**: an agent that takes a
  neighbour's node with everything leaves its own. Eleven games later the
  nodes of 22 neighbours of one agent of the line turned in a single game.
  That agent held 1,629 tokens and had 63 links, and the line had 12
  agents. So it had spread its tokens: about 26 a node, where the others
  held 33 on average and all the stakers of a node together stake about
  what its holder has.
- **The network dies with the line.** Its agents have fewer links than the
  others from the start, and at the end everyone has three.

So what ended these worlds is a line that hoards and then spreads, and the
rules allow it. A coalition takes a node only if its tokens outweigh the
largest stake there. In a world of equals no stake is large. An agent that
has hoarded more than all its neighbours hold together can outweigh the
coalition on every one of their nodes in one game. Its copies start richer
than their neighbours, and the same happens again. When most agents play
so, few links are staked on, links that no tokens cross lapse, and pieces
of the network are culled. Experiment 62's changed brains did as well as
the rest; this one change does far better, and the order has nothing
against it. Small steps of mutation only make it rare.

That the line spread its stakes is seen in one game. Experiment 65 plays
such a player by hand.

The lines are traced by `python3 research/efficient/E50/autopsy_lines.py`.

### Does a hoarder that spreads end it?

<!-- experiment E65 -->

Experiment 65 gives one agent of a world of equals the two moves together.
It is the hoarder of Experiment 64, at home or moving, until it holds more
than all its neighbours together. Then it stakes every token on its
neighbours' nodes, in proportion to what each holds, so that on each node
its stake is at least what the holder has. A copy that takes a node hoards
again. The worlds, the agents and the length are those of Experiment 64, so
its hoarders, which only hoard, are the comparison.

<!-- thesis E65 -->
> [!quote] The thesis of Experiment 65, written down before any of its runs existed
> **The claim.** A hoarder that spreads once it is rich takes a world of equals and ends it. Its line comes to hold more than half the world's agents, and the world then dies out or loses more than half its agents.
>
> **Why it would be so.** One hoarder that only hoards grows rich and the world lives on (Experiment 64). The two worlds of equals that died in Experiment 63 were each taken by one line of brains: it began with one brain, held every rich agent, and went from one agent to nearly all within 80 to 200 iterations, by turning neighbouring nodes in the game and hardly by births (read from their frames, research/efficient/E50/autopsy_lines.py). In the one game of that takeover whose stakes are recorded, all twelve agents of the line staked everything on one node. But an agent that only does that takes one node and leaves one, so its line cannot grow; and eleven games later the nodes of 22 neighbours of one agent of the line turned in a single game, when it held 1,629 tokens. So the line both hoards and spreads. The rule of the game allows it: a coalition takes a node only if its tokens outweigh the largest stake, and an agent that holds more than all its neighbours together can stake on each of their nodes at least what that neighbour holds, which is about what all the others stake there. Each copy then holds twice a neighbour's tokens and hoards again. A world of such players stakes on few links, the links that no tokens cross lapse, and pieces of the network are culled.
>
> **It holds if:** A trial takes one agent of a world of equals and gives it its own brain, frozen, with only its stakes written over: it hoards, at home or on its poorest neighbour's node, until it holds more than all its neighbours together, and then stakes every token on the neighbours' nodes in proportion to what each holds. It plays 1,000 iterations, or until its line is gone or the world has died out. A line sweeps if at some iteration it holds more than half the world's agents. A world ends if it dies out or has fewer than half its first agents at the trial's last iteration. Of the 30 trials (three worlds, five places, two ways of hoarding) at least 20 lines sweep and at least 20 worlds end.
>
> **It fails if:** Fewer than 10 of the 30 lines sweep, or fewer than 10 of the 30 worlds end. Anything else is undecided. Reported beside it, by way of hoarding and by world: the iteration at which a line first holds two agents or more and at which it holds half the world, how many lines are lost and when, the world's agents and links over time, and the same counts for the hoarders of Experiment 64, which only hoard.
<!-- /thesis -->

**The thesis is confirmed.** All 30 lines swept, where the thesis asked
for 20, and 25 of the 30 worlds ended, where it asked for 20. It is the
first thesis of this page that holds.

- **The line takes half the world within 33 to 134 iterations**, 71 as a
  median when it hoards at home and 89 when it hoards on the move. The
  lines that took the two worlds of Experiment 63 needed 85 and 200.
- **The world then collapses within a few games.** In one trial the line
  held half the agents at iteration 84 and the world had lost half its
  agents by iteration 86. In 28 of the 30 trials that took 1 to 16
  iterations; the other two were in the densest world. 26 of the 30 worlds fell to 20 agents or
  fewer, the mark at which a run counts as died out, at iteration 107 as a
  median. A trial plays on below that mark: 22 worlds ended with one to
  five agents, and in eleven of them one agent held all 40,000 tokens.
- **No hoarder lost its first game, and none failed to spread.** The same
  agents with the hoarders of Experiment 64, which only hoard, ended no
  world of 30.
- **Five worlds lived on, all in the densest of the three worlds** (257
  agents with 103 links each). After the sweep they were worlds of 269 to
  894 agents with about three links each, three to seven in ten of them
  of the line. The world with 50 numbers of Experiment 63 spent its last 300
  iterations in such a state.
- **Two lines were lost after they had swept**, at iterations 117 and 196,
  and left 20 agents each.

| world of Experiment 63 (20 numbers) | agents | links each | hoards | half the world at iteration | 20 agents or fewer at | agents at the end |
|---|---|---|---|---|---|---|
| seed 6301 | 484 | 45 | at home | 71 to 121 | 78 to 128 | 1, 1, 1, 1, 20 |
| | | | moving | 73 to 113 | 105 to 353 | 3, 4, 5, 20, 74 |
| seed 6302 | 257 | 103 | at home | 61 to 101 | 96, 106, 186, never, never | 1, 1, 269, 357, 440 |
| | | | moving | 90 to 134 | 374, 528, 599, never, never | 2, 2, 4, 663, 894 |
| seed 6303 | 533 | 49 | at home | 33 to 56 | 40 to 67 | 1, 1, 1, 1, 1 |
| | | | moving | 71 to 89 | 93 to 116 | 2, 2, 2, 2, 2 |

**What this shows.** A world of equals has no defence against a player
that hoards and then spreads, and one such agent is enough to end it. The
two moves are needed together: hoarding alone leaves the world standing
(Experiment 64), and without a hoard a spread stake is the stake of
everyone else. The coalition guards a node against any stake smaller than
what the coalition stakes there, and so only while nobody is rich.

So the order lasts as long as mutation does not find this player, and
that is why small steps keep it longer: none of three worlds with 20
numbers lost it in 9,500 iterations, and with 30 and 50 numbers two worlds
of equals in four were taken.

**What it does not show.** The player here is written by hand. That the
lines of Experiment 63 hoarded is seen in one recorded game and that they
spread in another; that their brains followed this rule is not shown. And
a trial holds the line's brains fixed, where an evolved line goes on
changing.

The numbers are made by `python3 research/efficient/E50/e64.py E65`.

<!-- runs E65 -->
> [!info]- The trials behind this page
> **30 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 1,000 iterations, from a world another run kept in its archive.
>
> - **hoard at home, then spread** — 15 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies home-spreader; patches of 1 agent; places 1 to 5, every 1.
> - **hoard on the poorest neighbour, then spread** — 15 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies moving-spreader; patches of 1 agent; places 1 to 5, every 1.
>
> To play these trials again: `python3 gol_lab.py run E65`.
<!-- /runs -->

### Three rules that wealth cannot break

<!-- experiment E66 -->

Experiment 65 leaves one way to keep a world of equals: a rule of the game
that a rich agent cannot get round. The author chose three to try, on
26 November 2026. None has a number of its own. Each is a value of the
setting [`wealth_guard`](settings.md#wealth_guard), and a trial plays under
one of them at a time.

- **A coalition counted by heads.** Every agent that flags a token on a
  node counts one, and so does the node's largest staker, whatever they
  stake. Two agents that flag a token there take the node from any stake.
- **A limit on a stake.** An agent stakes on another agent's node at most
  what that node's agent holds. What is over stays on its own node.
- **Tokens that level across links.** After every game one token passes
  across each link from the richer end to the poorer, if they differ by two
  tokens or more.

Experiment 66 plays the trials of Experiment 65 again under each rule: the
same three worlds, the same five agents, the hoarder that spreads at home
and moving. The residents' brains were made under the old rule. So each
rule is also played with no hoarder, to see whether the world of equals
itself lives on under it.

<!-- thesis E66 -->
> [!quote] The thesis of Experiment 66, written down before any of its runs existed
> **The claim.** Two of three rules stop the hoarder that spreads, and a world of equals lives on under them: a coalition counted by heads, and tokens that level across links. The third, a limit on a stake, does not stop it.
>
> **Why it would be so.** The hoarder of Experiment 65 ends a world of equals because a coalition guards a node only against a stake smaller than the coalition's own tokens there. Each rule takes that power away in another way, and none has a number of its own. By heads: every agent that flags a token on a node counts one, and so does the node's largest staker, so two residents that flag a token there take the node from any stake. A hoarder is the largest staker on the node it hoards on, so it should lose that node in its first games, as a resident's frozen copy does. Levelling: after every game one token passes across each link from the richer end to the poorer, if they differ by two or more. A resident stakes about one token on each neighbour's node, so a hoarder gains about one token a link in a game and levelling takes one token a link back; it should not come to hold more than all its neighbours together, which is when it spreads. The limit: an agent stakes on another's node at most what that node's agent holds. The hoarder that spreads stakes just that on each neighbour's node. The coalition there stakes about what the holder holds, and it takes the node only with more than the largest stake, so the hoarder should still take about half the nodes it stakes on, keep its own with the rest of its tokens, and sweep. The residents' brains were made under the old rule, so each rule is also played without a hoarder, to see whether the world of equals itself lives on under it.
>
> **It holds if:** The trials are those of Experiment 65, in the same three worlds at the same five places, with the hoarder that spreads hoarding at home or on its poorest neighbour's node, 30 for each rule; only the rule is changed, for the trial. A line sweeps if at some iteration it holds more than half the world's agents. A world ends if it dies out or has fewer than half its first agents at the trial's last iteration. For each rule six controls play the same worlds 1,000 iterations with no hoarder (a resident's frozen copy in its place, and the trial plays on when that copy is gone), and six more under no rule. The thesis holds if all of this is so: by heads at most 5 of the 30 lines sweep and at most 5 of the 30 worlds end; under levelling the same; under the limit at least 20 of the 30 lines sweep; and by heads and under levelling at most one of the six control worlds ends.
>
> **It fails if:** By heads or under levelling at least 20 of the 30 lines sweep, or at least three of the six control worlds end; or under the limit at most 5 of the 30 lines sweep. Anything else is undecided. Reported beside it, for each rule by itself: how many lines sweep and worlds end, by way of hoarding and by world, and when; how many lines are lost and when; the most tokens a line held; in the controls the world's agents, links and richest holding at the start and the end, beside the six controls under no rule; and the counts of Experiment 65, which played the same trials under no rule (30 of 30 lines swept, 25 of 30 worlds ended).
<!-- /thesis -->

**The thesis is undecided.** Two of its three predictions held and one did
not. By heads no line swept and no world ended, as predicted. Under the
limit 25 of the 30 lines swept, as predicted. But under levelling 11 lines
swept and 9 worlds ended, where the thesis allowed 5: levelling slows the
hoarder and does not stop it. No control world ended under any rule.

| rule | lines that sweep, of 30 | worlds that end, of 30 | lines lost | control worlds that end, of 6 |
|---|---|---|---|---|
| none (Experiment 65) | 30 | 25 | 2 | 0 |
| a coalition counted by heads | 0 | 0 | 30 | 0 |
| a limit on a stake | 25 | 25 | 6 | 0 |
| tokens that level across links | 11 | 9 | 12 | 0 |

- **By heads the hoarder is gone at once.** 29 of the 30 hoarders lost
  their node in their first game and the last in its second, at home and
  moving alike. None held more than 134 tokens, and no world changed by
  more than one agent. The hoarder is the largest staker on the node it
  hoards on, and two neighbours that flag a token there now take it.
- **By heads a world of equals plays as before.** In the densest world the
  two controls by heads are the controls under no rule, iteration for
  iteration, through all 1,000: the same agents, links and richest holding.
  There every stake is one token, so tokens and heads count the same. The
  other two worlds differ from the second iteration on and have 469 to
  546 agents at the end (484 to 565 under no rule).
- **The limit changes nothing for the hoarder at home.** All 15 swept, at
  iteration 37 to 125, 68 as a median; without the rule it was 33 to 121
  and 71. The hoarder that spreads stakes on each node what its holder
  holds, and that is what the limit allows.
- **Under the limit the moving hoarder is faster, or lost.** 10 of 15
  swept, at iteration 19 to 66, 22 as a median, where it was 71 to 134
  without the rule; the other five were lost in their first seven games.
  What is over the limit stays on its own node, so it holds that node
  while it stakes on its neighbour's. All 25 worlds whose line swept
  ended, with 1 to 19 agents.
- **Levelling is a brake.** At home 9 of 15 lines swept, at iteration 293
  to 738, 407 as a median, where it was 71 without the rule.
  Moving, 2 of 15 swept, at 617 and 952. Twelve lines were lost, eight of
  them in their second game. Six of the nine worlds that ended were left
  as one agent with all 40,000 tokens.
- **Levelling brakes harder where agents have more links.** In the densest
  world, with 103 links each, no hoarder at home swept: three were lost in
  their second game, and the other two held 8 and 24 agents and 3,470 and
  8,518 tokens after 1,000 iterations, still growing. In the worlds with
  45 and 49 links each, 9 of 10 swept.
- **Levelling changes the world of equals itself, and it lives.** The
  richest holding of the six controls fell from 212, 123 and 225 tokens to
  89 and 92, 104 and 105, and 53 and 70. The agents stayed: 499 and 501,
  257 and 257, 533 and 533.

| rule | hoards | seed 6301, 45 links each | seed 6302, 103 links each | seed 6303, 49 links each |
|---|---|---|---|---|
| none (Experiment 65) | at home | 5 of 5 sweep, at 71 to 121 | 5 of 5, at 61 to 101 | 5 of 5, at 33 to 56 |
| | moving | 5 of 5, at 73 to 113 | 5 of 5, at 90 to 134 | 5 of 5, at 71 to 89 |
| by heads | at home | 0 of 5 | 0 of 5 | 0 of 5 |
| | moving | 0 of 5 | 0 of 5 | 0 of 5 |
| a limit on a stake | at home | 5 of 5, at 73 to 125 | 5 of 5, at 62 to 100 | 5 of 5, at 37 to 60 |
| | moving | 4 of 5, at 22 to 32 | 4 of 5, at 19 to 26 | 2 of 5, at 21 and 66 |
| levelling | at home | 5 of 5, at 293 to 519 | 0 of 5 | 4 of 5, at 407 to 738 |
| | moving | 0 of 5 | 1 of 5, at 952 | 1 of 5, at 617 |

The control worlds, with no hoarder, at the start and after 1,000
iterations (two trials for each world):

| rule | world | agents | links each | richest holding |
|---|---|---|---|---|
| none | seed 6301 | 484 → 484, 565 | 45.0 → 44.5, 37.8 | 212 → 224, 224 |
| | seed 6302 | 257 → 257, 257 | 103.2 → 103.2, 103.8 | 123 → 123, 122 |
| | seed 6303 | 533 → 526, 531 | 49.4 → 50.3, 49.5 | 225 → 152, 294 |
| by heads | seed 6301 | 484 → 546, 469 | 45.0 → 40.2, 46.1 | 212 → 196, 231 |
| | seed 6302 | 257 → 257, 257 | 103.2 → 103.2, 103.8 | 123 → 123, 122 |
| | seed 6303 | 533 → 533, 525 | 49.4 → 49.4, 50.4 | 225 → 235, 312 |
| a limit on a stake | seed 6301 | 484 → 564, 479 | 45.0 → 38.7, 45.5 | 212 → 197, 226 |
| | seed 6302 | 257 → 257, 257 | 103.2 → 103.2, 103.8 | 123 → 123, 122 |
| | seed 6303 | 533 → 525, 531 | 49.4 → 50.3, 49.5 | 225 → 158, 294 |
| levelling | seed 6301 | 484 → 499, 501 | 45.0 → 43.3, 43.3 | 212 → 89, 92 |
| | seed 6302 | 257 → 257, 257 | 103.2 → 102.7, 103.3 | 123 → 105, 104 |
| | seed 6303 | 533 → 533, 533 | 49.4 → 46.4, 50.0 | 225 → 70, 53 |

Under the limit, too, the densest world plays exactly as under no rule
for all 1,000 iterations, and one of the two trials of seed 6303 does.
Among equals the limit is hardly ever met.

**Where the thesis was wrong.** It reckoned that a hoarder gains about one
token a link in a game and that levelling takes one a link back, so that
nothing is left. Something is left. Over the first ten games a hoarder at
home gained 16 to 119 tokens a game without the rule, and under levelling
still 6 to 40 in the two worlds with fewer links and 4 to 6 in the
densest. A reading, not measured: the tokens that levelling passes to the
neighbours are staked again, in part on the hoarder's node, and come back
to it. And once a line holds several nodes its tokens grow many times
faster, in one trial from 566 to 8,936 within 200 iterations, which one
token a link does not hold back.

**What this shows.** Of the three rules one stops the hoarder that
spreads: a coalition counted by heads. It stops it in the first game, in
every world, and a world of equals plays under it as it did before,
because among equals a stake is already one token. A limit by what the
holder holds is no guard, since it is the very stake that takes a node.
Levelling by one token a link buys time, more with more links, and not
safety.

**What it does not show.** The hoarder is written by hand, for the old
rule. By heads the largest staker on a node is the one who loses it, and a
line that wins by many small stakes was not tried. The residents' brains
were made under the old rule, and the controls play 1,000 iterations: that
a world keeps the order of equals for long under a head count, that it
forms it there, and that mutation finds no other way to break it, are not
shown. Those need worlds played on under the rule.

One mistake was found on the way. The limit crashed in a game in which
some agent had no link, in nine trials, after the world had been taken.
It was corrected while the experiment was being played
(26 November 2026), and those nine were played again from their start. So
55 trials ran on the first copy of the engine and 59 on the corrected one.
The two differ only under the limit, in a game with such an agent, and no
trial that finished on the first copy met one: it would have stopped.

The numbers are made by `python3 research/efficient/E50/e66.py`.

<!-- runs E66 -->
> [!info]- The trials behind this page
> **114 trials**, invasions: a patch of the world of one era given a strategy written by hand, and its agents counted after every iteration; each plays 1,000 iterations, from a world another run kept in its archive.
>
> - **by heads** — 30 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies home-spreader, moving-spreader; patches of 1 agent; places 1 to 5, every 1; with wealth_guard = heads.
> - **a limit on a stake** — 30 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies home-spreader, moving-spreader; patches of 1 agent; places 1 to 5, every 1; with wealth_guard = limit.
> - **levelling** — 30 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies home-spreader, moving-spreader; patches of 1 agent; places 1 to 5, every 1; with wealth_guard = level.
> - **by heads, no hoarder** — 6 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies resident; patches of 1 agent; places 1, 2; with wealth_guard = heads; played on when the strategy is gone.
> - **a limit on a stake, no hoarder** — 6 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies resident; patches of 1 agent; places 1, 2; with wealth_guard = limit; played on when the strategy is gone.
> - **levelling, no hoarder** — 6 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies resident; patches of 1 agent; places 1, 2; with wealth_guard = level; played on when the strategy is gone.
> - **no rule, no hoarder** — 6 trials: worlds `B1-40000-c8c9f2-s6301-on210139`, `B1-40000-c8c9f2-s6302-on210139`, `B1-40000-c8c9f2-s6303-on210139`; at iteration 27,500; strategies resident; patches of 1 agent; places 1, 2; played on when the strategy is gone.
>
> To play these trials again: `python3 gol_lab.py run E66`.
<!-- /runs -->

### The head count against a line that evolved

<!-- experiment E67 -->

Experiment 66 stopped a hoarder written by hand. The lines that took the
two worlds of Experiment 63 were made by mutation, and their brains are
lost: those runs kept a world every 2,500 iterations, and each line lived
a few hundred, between two of them. But a kept world also holds the state
of the world's random numbers. So a run can be played again from a world
it kept, and should come out the same, iteration for iteration.

Experiment 67 plays both worlds again from the last world each kept
before it died, and keeps a world every 10 iterations. If the two plays
agree, the worlds inside the takeover are there to be played on under a
head count, with the line that evolved in them.

<!-- thesis E67 -->
> [!quote] The thesis of Experiment 67, written down before any of its runs existed
> **The claim.** A world played again from a world its run kept, under the same settings and with the random state kept with that world, is the same world. Both worlds of equals that died in Experiment 63 are taken by the same line of brains again and die out at the same iteration.
>
> **Why it would be so.** A world draws all its luck from one stream of random numbers, and a kept world holds the state of that stream beside every agent, brain, link and message. Nothing else decides what happens next. The engine has been changed since those runs were made (hand-written strategies, the setting wealth_guard), but with those off it should play as it did. If that is so, a finished run can be made to keep more of its worlds than it kept: here one every 10 iterations through the takeover, where the runs kept one every 2,500. The worlds inside the takeover hold the brains of the lines that took them, which no kept world has, and a rule can then be tried against a line that evolved and not only against one written by hand.
>
> **It holds if:** Each of the two runs is played again from the last world it kept before it died, iteration 27,500 for the world with 30 numbers and 22,500 for the world with 50, under its own settings and seed, keeping a world every 10 iterations. The thesis holds if in both, after every game, the agents, the tokens each holds and the brain each carries are those of the first play, read from the frames of the two plays, and the world dies out at the same iteration, 29,351 and 23,853.
>
> **It fails if:** In either world some agent, holding or brain differs after some game, or the world dies out at another iteration or not at all. Reported beside it: the first iteration at which the two plays differ, if any, and how many worlds each play kept.
<!-- /thesis -->

**The thesis is confirmed.** Both worlds came out the same. After each of
the 1,851 games of the world with 30 numbers and the 1,353 games of the
world with 50, every agent, its tokens and its brain are those of the
first play, and the worlds die out at iterations 29,351 and 23,853 again.

- **The second plays kept 186 and 136 worlds**, one every 10 iterations,
  where the first kept 4 and 2.
- **The engine had been changed in between.** The first plays ran on the
  engine of 24 November 2026 and the second on that of 26 November, with
  the hand-written strategies and the setting `wealth_guard` added. With
  those off it plays as it did.

**What this shows.** A finished run can be made to keep any world of its
past, as long as it kept one before it. The takeover that ended each world
is now there to start from at any tenth iteration, with the brains of the
line that did it.

**What it does not show.** It is two runs and two versions of the engine.
A change to the engine that draws its random numbers in another order
would break it, and this comparison is how that would be seen.

The two plays are compared by `python3 research/efficient/E50/e67.py`.

<!-- runs E67 -->
> [!info]- The runs behind this page
> **2 runs.** Every condition below is run once for every seed (6302), played on or again from a kept world of its own to iteration 30,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **30 numbers, again** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `n_nodes` = 800; 40,000 tokens; played again from `B1-40000-8020c3-s6302-on210139` at iteration 27,500. Run `B1-40000-8020c3-s6302-on0a0a78`.
> - **50 numbers, again** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 50, `n_nodes` = 800; 40,000 tokens; played again from `B1-40000-3f4b02-s6302-on210139` at iteration 22,500. Run `B1-40000-3f4b02-s6302-ond3e026`.
>
> To make these runs again: `python3 gol_lab.py run E67`.
<!-- /runs -->

### Does the head count stop the lines that evolved?

<!-- experiment E68 -->

Experiment 68 plays both worlds on from inside the takeover. Each is
started at three worlds that Experiment 67 kept: with the line holding
about 4% of all tokens, about 22% and about 54%. From each it is played
1,000 iterations with five seeds under no rule, under a head count, and
under levelling. Nothing is planted and mutation goes on, so the line is
the one that evolved, and it can go on changing.

| world | played on from iteration | agents | of the line | its share of all tokens |
|---|---|---|---|---|
| 30 numbers, seed 6302 | 29,280 | 674 | 9 | 3% |
| | 29,300 | 683 | 19 | 21% |
| | 29,320 | 761 | 127 | 51% |
| 50 numbers, seed 6302 | 23,400 | 531 | 8 | 5% |
| | 23,470 | 526 | 9 | 24% |
| | 23,520 | 512 | 35 | 57% |

<!-- thesis E68 -->
> [!quote] The thesis of Experiment 68, written down before any of its runs existed
> **The claim.** A coalition counted by heads stops the lines that evolution made, too. Played on from inside the takeover under a head count, the two worlds of equals that died live on. Played on under the old rule, with other luck, they end again.
>
> **Why it would be so.** The head count stopped a hoarder written by hand in all 30 trials of Experiment 66. The lines that took the two worlds of Experiment 63 were not written by hand, and nobody knows their brains' rule. What is known is that they hoard: in the one game of a takeover whose stakes are recorded, all twelve agents of the line staked everything on one node, six on their own and six on a neighbour's. Under a head count such a stake counts one, and two neighbours that flag a token on the node take it. So an agent of the line should lose nodes where it took them, and what it holds should pass to those around it. That should hold however much the line holds already, because the count is of heads and the line's agents are few: 9 to 127 among 512 to 761 at the six worlds played on from. Under the old rule the takeover should not hang on luck once the line holds a few tokens in a hundred, since each agent of it outweighs any coalition it meets; so with other seeds the worlds should end as they did.
>
> **It holds if:** Experiment 67 kept the two worlds every 10 iterations. Each is played on from three of them: with the line holding about 4% of all tokens (iterations 29,280 and 23,400), about 22% (29,300 and 23,470) and about 54% (29,320 and 23,520). Each of the six is played on 1,000 iterations with five seeds under no rule, five under a head count and five under levelling; nothing else is changed, and mutation goes on. The line is the brains that come from the one brain all rich agents came from in the first play, and every brain copied from them. A world ends if it dies out or has fewer than half its first agents at the trial's last iteration. A line sweeps if at some iteration it holds more than half the world's agents. The thesis holds if under no rule at least 20 of the 30 worlds end, and by heads at most 5 of the 30 worlds end and at most 5 of the 30 lines sweep.
>
> **It fails if:** By heads at least 15 of the 30 worlds end, or under no rule at most 10 of the 30 worlds end. Anything else is undecided. Reported beside it, with no line drawn: the same counts under levelling, which Experiment 66 found to be a brake; all counts by world and by how much the line held; the line's agents and tokens over time and at the end; and the richest holding.
<!-- /thesis -->

**The thesis is confirmed.** Under no rule 27 of the 30 worlds ended.
Under a head count no world ended and no line swept.

| rule | | 30 numbers: line holds 3% | 21% | 51% | 50 numbers: 5% | 24% | 57% | all |
|---|---|---|---|---|---|---|---|---|
| no rule | lines sweep | 5 of 5, at 48 to 61 | 5, at 29 to 37 | 5, at 13 to 16 | 5, at 196 to 265 | 5, at 65 to 206 | 5, at 25 to 42 | 30 of 30 |
| | worlds end | 4 | 5 | 5 | 4 | 4 | 5 | 27 |
| by heads | lines sweep | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| | worlds end | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| levelling | lines sweep | 0 | 4, at 163 to 343 | 5, at 61 to 105 | 0 | 3, at 359 to 477 | 4, at 101 to 210 | 16 |
| | worlds end | 0 | 3 | 3 | 0 | 2 | 3 | 11 |

Iterations are counted from the start of a trial.

- **Under the old rule the takeover does not hang on luck.** All 30 lines
  swept, from every starting world and with every seed. 27 worlds ended
  with 1 to 164 agents, 15 of them as one agent holding all 40,000
  tokens.
- **Three worlds grew again out of the line.** They fell to 7, 8 and 194
  agents and then grew to 1,135, 4,865 and 3,161 by iteration 1,000, with
  two to three links each. Every agent of them comes from the line. A run
  stops as died out under 20 agents, as the two first plays did; a trial
  plays on, which is why this shows here.
- **By heads the line loses its hoard in the first game.** The table
  below gives what the line held. Even the lines that held half of all
  tokens were down to about a tenth of that after one game. No line ever had
  more agents than at the start.
- **In the world with 30 numbers the line is gone**, in all 15 trials,
  within 29 games.
- **In the world with 50 numbers it lives on as ordinary agents.** In 10
  of the 15 trials one or two agents of the line were left at iteration
  1,000, holding 11 to 251 tokens. In the other five the line was gone at
  iteration 204 to 719.
- **The worlds go back to small holdings.** The richest agent held 166 to
  3,289 tokens at the start, 105 to 297 after ten games and 48 to 251 at
  the end. The world with 50 numbers kept its agents: 531, 526 and 512 at
  the start, 509 to 531 at the end. The world with 30 numbers grew: from
  674, 683 and 761 to between 683 and 1,570.
- **Levelling is a brake here too.** It removed the line only where it
  was smallest: in the world with 30 numbers, started with 3% of the
  tokens, all five lines were gone within 19 games. From 21% and 51%
  there, 9 of 10 swept, later than without the rule. In the world with 50
  numbers, started with 5%, no line swept and none was lost, but two held
  87% and 64% of all tokens at the end. In all, 22 of the 30 lines held
  more than half of all tokens at some iteration.

What the line held under a head count, in tokens, over the five seeds:

| played on from | at the start | after one game | after ten games | line gone |
|---|---|---|---|---|
| 30 numbers, iteration 29,280 | 742 | 0 to 8 | 0 | 5 of 5, by game 3 |
| 29,300 | 6,213 | 0 to 1,020 | 0 | 5 of 5, by game 3 |
| 29,320 | 16,265 | 1,274 to 1,673 | 0 to 184 | 5 of 5, at game 8 to 29 |
| 50 numbers, iteration 23,400 | 1,352 | 73 to 91 | 39 to 125 | 1 of 5, at 719 |
| 23,470 | 7,057 | 1,334 to 3,260 | 48 to 191 | 1 of 5, at 515 |
| 23,520 | 18,515 | 1,146 to 1,219 | 286 to 470 | 3 of 5, at 204 to 653 |

**What this shows.** The head count stops the two lines that evolution
made, as it stopped the hoarder written by hand: at every point of the
takeover that was tried, with every seed, and whether the line held 3% of
all tokens or 57%. And under the old rule the end of these two worlds was
not bad luck. Once the line holds a few tokens in a hundred, it takes the
world with any seed.

**What it does not show.** These are two lines, from two worlds that
differ only in their step, played 1,000 iterations. Both lines were made
under the old rule. No world has yet been played under a head count long
enough for mutation to try its luck against it, and in the world with 50
numbers the line's brains are still there. The world with 30 numbers grew
under the head count, and the two trials that grew most, to 1,456 and
1,570 agents, ended with 17 and 16 links an agent where they started with
32 and 22. Whether that is the rule or this world's own drift
cannot be told apart here, because every trial without the rule has the
line in it. Whether a world keeps the order of equals for long under a
head count, and forms it there, needs worlds played on under the rule for
thousands of iterations.

All 90 trials ran on one copy of the engine. The numbers are made by
`python3 research/efficient/E50/e68.py`.

<!-- runs E68 -->
> [!info]- The trials behind this page
> **90 trials**, sequels: the world of one era played on with its own mutations and a seed of its own, and its agents, and those of one line of brains, counted after every iteration; each plays 1,000 iterations, from a world another run kept in its archive.
>
> - **no rule: 30 numbers at iteration 29280** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,280; seeds 1 to 5, every 1; following a line of 4 brains.
> - **no rule: 30 numbers at iteration 29300** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,300; seeds 1 to 5, every 1; following a line of 15 brains.
> - **no rule: 30 numbers at iteration 29320** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,320; seeds 1 to 5, every 1; following a line of 69 brains.
> - **no rule: 50 numbers at iteration 23400** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,400; seeds 1 to 5, every 1; following a line of 7 brains.
> - **no rule: 50 numbers at iteration 23470** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,470; seeds 1 to 5, every 1; following a line of 9 brains.
> - **no rule: 50 numbers at iteration 23520** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,520; seeds 1 to 5, every 1; following a line of 27 brains.
> - **by heads: 30 numbers at iteration 29280** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,280; with wealth_guard = heads; seeds 1 to 5, every 1; following a line of 4 brains.
> - **by heads: 30 numbers at iteration 29300** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,300; with wealth_guard = heads; seeds 1 to 5, every 1; following a line of 15 brains.
> - **by heads: 30 numbers at iteration 29320** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,320; with wealth_guard = heads; seeds 1 to 5, every 1; following a line of 69 brains.
> - **by heads: 50 numbers at iteration 23400** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,400; with wealth_guard = heads; seeds 1 to 5, every 1; following a line of 7 brains.
> - **by heads: 50 numbers at iteration 23470** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,470; with wealth_guard = heads; seeds 1 to 5, every 1; following a line of 9 brains.
> - **by heads: 50 numbers at iteration 23520** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,520; with wealth_guard = heads; seeds 1 to 5, every 1; following a line of 27 brains.
> - **levelling: 30 numbers at iteration 29280** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,280; with wealth_guard = level; seeds 1 to 5, every 1; following a line of 4 brains.
> - **levelling: 30 numbers at iteration 29300** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,300; with wealth_guard = level; seeds 1 to 5, every 1; following a line of 15 brains.
> - **levelling: 30 numbers at iteration 29320** — 5 trials: worlds `B1-40000-8020c3-s6302-on0a0a78`; at iteration 29,320; with wealth_guard = level; seeds 1 to 5, every 1; following a line of 69 brains.
> - **levelling: 50 numbers at iteration 23400** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,400; with wealth_guard = level; seeds 1 to 5, every 1; following a line of 7 brains.
> - **levelling: 50 numbers at iteration 23470** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,470; with wealth_guard = level; seeds 1 to 5, every 1; following a line of 9 brains.
> - **levelling: 50 numbers at iteration 23520** — 5 trials: worlds `B1-40000-3f4b02-s6302-ond3e026`; at iteration 23,520; with wealth_guard = level; seeds 1 to 5, every 1; following a line of 27 brains.
>
> To play these trials again: `python3 gol_lab.py run E68`.
<!-- /runs -->

### Do worlds with larger steps live under a head count?

<!-- experiment E69 -->

Experiment 68 switched the head count on inside a takeover and played
1,000 iterations. Experiment 69 plays whole histories under it. The six
worlds of Experiment 63 with mutations of 30 and 50 numbers are made
again from the same kept world, with the same seeds, and with the head
count on from the first iteration, for 10,000 iterations. Without the
rule three of the six died out: two as worlds of equals taken by a line
of hoarders, and one that never formed the order.

It asks two things. Do such worlds live when a hoard no longer wins? And
does mutation find another way to break a world, now that the largest
staker loses?

<!-- thesis E69 -->
> [!quote] The thesis of Experiment 69, written down before any of its runs existed
> **The claim.** Under a coalition counted by heads, worlds whose mutations change 30 or 50 numbers live on. Experiment 63 played six such worlds from one kept world and three died out. Played again from that world with the head count switched on, at most one of the six dies out.
>
> **Why it would be so.** Two of the three that died were worlds of equals, and each was taken by one line of hoarders: brains that stake everything on one node, grown from one agent to nearly all within 85 and 200 iterations. Experiment 68 played both worlds on from inside that takeover: without the rule the line took the world again in all 30 trials, and under a head count it lost its hoard in the first game in all 30, whether it held 3% of all tokens or 57%. So if hoarding lines are how such worlds end, a rule under which a hoard counts one head should let them live, however often mutation makes a hoarder. Two things are not known, and this experiment is where they would show. The third world died outside the order, and what ended it has not been read. And under a head count the largest staker no longer wins a node, so mutation may find another way to break a world, one that the old rule never rewarded.
>
> **It holds if:** The six runs of Experiment 63 with 30 and 50 numbers are made again from the same kept world at iteration 20,000, with the same seeds (6301 to 6303) and the same settings but wealth_guard = heads, to iteration 30,000. A world dies out if its run ends as extinct before iteration 30,000. The thesis holds if at most one of the six dies out.
>
> **It fails if:** Three or more of the six die out, as many as without the rule. Two is undecided. Reported beside it, with no line drawn: for each run its twenty windows of 500 iterations by Experiment 63's measure of a world of equals (fewer than 0.01 births per agent and iteration, and at least nine nodes in ten won by a coalition); its agents, the links of an agent and the richest holding over time; and the same for Experiment 63's run of the same step and seed.
<!-- /thesis -->

**Stopped on 27 November 2026, in its first run, and not judged.** The
plan reckoned with about one second an iteration, as in Experiment 63,
where the worlds condensed to a few hundred agents within 500 iterations.
Under the head count the first world did not condense. It stayed at 5,000
to 7,000 agents and played at about 2.7 seconds an iteration, so the six
runs would have taken about two days on the one worker that was free. No
experiment of this notebook may take longer than one day (the author's
rule of 23 November 2026). The thesis stands as written and has no
verdict.

What the first run showed is why the next two experiments ask what they
ask. It is the world with 30 numbers and seed 6301. Without the rule it
was a world of equals from iteration 20,500. Under a head count it began
to condense in the same way and then went back to a large world with two
links an agent. So the question of Experiment 69 is split in two:
[does the order form under a head count](#does-the-order-form-under-a-head-count),
and [does it last there](#does-the-order-last-under-a-head-count)?

<!-- runs E69 -->
> [!info]- The runs behind this page
> **6 runs.** Every condition below is run once for every seed (6301 to 6303), played on from the world `B1-40000-bbbccd-s5001-on45a3d3` kept at iteration 20,000, to iteration 30,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **30 numbers, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-142067-s6301-on210139` … `B1-40000-142067-s6303-on210139`.
> - **50 numbers, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 50, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-eee3e3-s6301-on210139` … `B1-40000-eee3e3-s6303-on210139`.
>
> To make these runs again: `python3 gol_lab.py run E69`.
<!-- /runs -->

### Does the order form under a head count?

<!-- experiment E70 -->

Experiment 70 plays all nine worlds of Experiment 63 again under a head
count, but only 1,500 iterations each: long enough for the order to form,
which without the rule took 500 to 1,000. One of the nine runs is the one
that Experiment 69 left, and the thesis was written after seeing it. It
says so, and counts the other eight apart.

<!-- thesis E70 -->
> [!quote] The thesis of Experiment 70, written down before any of its runs existed
> **The claim.** Under a coalition counted by heads the order of equals does not form out of a churning world. Nine worlds are played 1,500 iterations from the kept world that Experiment 63 started from, with the rule on. At most two of them are worlds of equals in their last 500 iterations, where seven of the nine were without the rule.
>
> **Why it would be so.** This claim rests on one world, and no reckoning stands behind it. Experiment 69 was to play six of these worlds for 10,000 iterations under a head count. It was stopped in its first run, because the world stayed large and slow and the six runs would have taken about two days. That run, with mutations of 30 numbers and seed 6301, is the ground for the claim. Without the rule the same world fell from 5,160 agents with two links each to 910 agents with 30 links each within 325 iterations, and was a world of equals from iteration 20,500. Under a head count it began the same way, 2,931 agents with seven links each at iteration 20,400, and then went back: between 5,000 and 7,200 agents with two links each from iteration 20,500 to 21,300, with two to ten nodes in a hundred won by a coalition, and a richest agent holding 1,800 to 3,200 tokens. Why the head count undoes the condensing is not known. Experiment 66 showed that a world that is already one of equals plays under a head count as before, so the rule does not break the order; what this world shows is that it may keep the order from forming.
>
> **It holds if:** The nine runs of Experiment 63 (mutations of 20, 30 and 50 numbers, seeds 6301 to 6303) are made again from the same kept world at iteration 20,000 with the same settings but wealth_guard = heads, to iteration 21,500. One of the nine exists already: Experiment 69's first run, 30 numbers with seed 6301, which this thesis was written after seeing; it is counted, and the count is also given for the other eight alone. A run is a world of equals in its last window, iterations 21,000 to 21,500, by Experiment 63's measure: fewer than 0.01 births per agent and iteration, and at least nine nodes in ten won by a coalition. The thesis holds if at most two of the nine are.
>
> **It fails if:** Six or more of the nine are worlds of equals in their last window. Anything else is undecided. Without the rule seven of the nine were, in the same window: all three with 20 numbers, two of three with 30 and two of three with 50. Reported beside it, with no line drawn: for each run its three windows of 500 iterations with its agents, the links of an agent, the richest holding, births and the share of nodes won by a coalition, beside Experiment 63's run of the same step and seed.
<!-- /thesis -->

**The thesis is confirmed.** The nine runs were played on 27 November
2026. Under a head count two of the nine worlds were worlds of equals in
iterations 21,000 to 21,500. Without the rule seven were. Of the eight
runs that did not exist when the thesis was written, two were.

Each world in those 500 iterations, under a head count and without the
rule. The richest holding is the middle one of the window: in half of its
iterations the richest agent held more.

| step | seed | by heads: agents | links each | richest | nodes won by a coalition | world of equals | no rule: agents | links each | richest | nodes won by a coalition | world of equals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 numbers | 6301 | 348 | 77 | 214 | 97% | **yes** | 612 | 28 | 235 | 98% | **yes** |
| | 6302 | 3,753 | 2.0 | 3,020 | 0% | no | 536 | 49 | 146 | 99% | **yes** |
| | 6303 | 3,038 | 6.2 | 1,082 | 24% | no | 533 | 50 | 181 | 99% | **yes** |
| 30 numbers | 6301 | 5,638 | 2.1 | 2,178 | 3% | no | 419 | 64 | 211 | 98% | **yes** |
| | 6302 | 987 | 27 | 122 | 82% | no | 493 | 45 | 218 | 95% | **yes** |
| | 6303 | 5,359 | 2.1 | 3,654 | 3% | no | 1,073 | 24 | 155 | 69% | no |
| 50 numbers | 6301 | 5,021 | 2.1 | 4,477 | 2% | no | 2,682 | 3.5 | 960 | 42% | no |
| | 6302 | 4,082 | 2.0 | 3,375 | 0% | no | 564 | 47 | 169 | 93% | **yes** |
| | 6303 | 696 | 38 | 128 | 99% | **yes** | 690 | 39 | 122 | 90% | **yes** |

- **Five worlds stayed large and sparse.** They had 3,753 to 5,638
  agents with two links each, and at most three nodes in a hundred were
  won by a coalition. In each of them one agent held about a tenth of the
  world: 2,178 to 4,477 of the 40,000 tokens, where the richest agent of
  the kept world they all started from held 1,607.
- **Two formed the order within 1,000 iterations**: the world with 20
  numbers and seed 6301, 500 iterations later than without the rule, and
  the one with 50 numbers and seed 6303, 500 iterations sooner.
- **Two were in between.** The world with 30 numbers and seed 6302 was
  condensing, about 500 iterations later than without the rule: 987
  agents with 27 links each, and 82 nodes in a hundred won by a
  coalition. The world with 20 numbers and seed 6303 had 3,038 agents
  with six links each and a quarter of its nodes won by a coalition.
- **No world died out.**

**What this shows.** The rule that protects the order mostly keeps it from
forming. From the same kept world, with the same seeds, seven of nine
worlds were worlds of equals after 1,500 iterations without the rule and
two of nine under a head count. And a world that does not form the order
under a head count is not one of small holdings either. It stays a
churning world with two links an agent, and its richest agent holds more
than the richest did before the rule was switched on.

**What it does not show.** Why. Under a head count two neighbours that
flag a token on a node take it from any staker, so a coalition wins more
easily, not less; yet in five worlds coalitions won almost no nodes. With
two links an agent a node has few stakers, and it may be that there are
too few neighbours for two of them to flag. That has not been read from
the frames. It also does not show that the order never forms: these are
1,500 iterations, and one world was still condensing at the end. And nine
worlds a side is few: by luck alone a split of two against seven comes
up about once in 18 times (Fisher's exact test, 0.057).

The numbers are made by `python3 research/efficient/E50/e69.py E70`.

<!-- runs E70 -->
> [!info]- The runs behind this page
> **9 runs.** Every condition below is run once for every seed (6301 to 6303), played on from the world `B1-40000-bbbccd-s5001-on45a3d3` kept at iteration 20,000, to iteration 21,500 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **20 numbers, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 20, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-c0734c-s6301-on210139` … `B1-40000-c0734c-s6303-on210139`.
> - **30 numbers, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-142067-s6301-on210139` … `B1-40000-142067-s6303-on210139`.
> - **50 numbers, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 50, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens. Runs `B1-40000-eee3e3-s6301-on210139` … `B1-40000-eee3e3-s6303-on210139`.
>
> To make these runs again: `python3 gol_lab.py run E70`.
<!-- /runs -->

### Does the order last under a head count?

<!-- experiment E71 -->

Experiment 71 starts where the order is already there. Four runs of
Experiment 63 with 30 and 50 numbers were worlds of equals at iteration
22,500. Two of them died out later, taken by a line of hoarders, and two
kept the order to the end. Each is played on from its world kept at
22,500 under a head count, to iteration 30,000.

<!-- thesis E71 -->
> [!quote] The thesis of Experiment 71, written down before any of its runs existed
> **The claim.** A world of equals whose mutations change 30 or 50 numbers lasts under a coalition counted by heads. Four such worlds of Experiment 63, two of which died out later without the rule, are played on from iteration 22,500 with the rule on, and at most one dies out before iteration 30,000.
>
> **Why it would be so.** Both worlds of equals that died in Experiment 63 were taken by a line of hoarders, and Experiment 68 played both on from inside that takeover: under a head count the line lost its hoard in the first game in all 30 trials, and without the rule it took the world again in all 30. So the one way these worlds are known to end is closed by the rule. Experiment 66 showed that a world of equals plays under a head count as it did before, for 1,000 iterations. Two things could still end such a world, and this experiment is where they would show. Mutation may find a way to break the order that the old rule never rewarded, since under a head count the largest staker loses. And the order may wear away by itself: Experiment 69's one run suggests that under a head count a churning world does not condense into the order, so a world that leaves it may not come back.
>
> **It holds if:** Experiment 63 kept its worlds every 2,500 iterations. Four of its runs with 30 and 50 numbers were worlds of equals in the 500 iterations from 22,500: 30 numbers with seeds 6301 and 6302, and 50 numbers with seeds 6302 and 6303. Without the rule the second and the third died out, at iterations 29,351 and 23,853, and the other two were worlds of equals to iteration 30,000. Each is played on once from its world kept at 22,500, with its own settings but wealth_guard = heads and the seed 7101, to iteration 30,000. A world dies out if its run ends as extinct before iteration 30,000. The thesis holds if at most one of the four dies out.
>
> **It fails if:** Three or more of the four die out. Two is undecided. Reported beside it, with no line drawn: for each run its fifteen windows of 500 iterations by Experiment 63's measure of a world of equals (fewer than 0.01 births per agent and iteration, and at least nine nodes in ten won by a coalition), and in how many of them it is one; its agents, the links of an agent and the richest holding over time; and the same for Experiment 63's run it was played on from.
<!-- /thesis -->

**The thesis is confirmed.** The four runs were played on 27 November
2026. None of the four worlds died out. Without the rule two of them had,
at iterations 29,351 and 23,853.

Each world from iteration 22,500, under a head count and without the
rule, in windows of 500 iterations. Agents and links are a window's
means, from the first window to the last. The largest holding is the
most that one agent held at any iteration.

| world | rule | windows that are a world of equals | agents | links each | largest holding | at iteration 30,000 |
|---|---|---|---|---|---|---|
| 30 numbers, seed 6301 | by heads | 11 of 15 | 1,006 to 2,462 | 26 to 10 | 119 | lives |
| | no rule | 15 of 15 | 1,006 to 1,007 | 26 | 66 | lives |
| 30 numbers, seed 6302 | by heads | 11 of 15 | 442 to 573 | 57 to 50 | 916 | lives |
| | no rule | 14 of 14 | 452 to 670 | 56 to 36 | 4,827 | died out at 29,351 |
| 50 numbers, seed 6302 | by heads | 10 of 15 | 536 to 2,016 | 46 to 12 | 765 | lives |
| | no rule | 2 of 3 | 536 to 221 | 45 to 6 | 6,005 | died out at 23,853 |
| 50 numbers, seed 6303 | by heads | 15 of 15 | 651 to 559 | 40 to 47 | 154 | lives |
| | no rule | 15 of 15 | 651 to 624 | 41 to 42 | 123 | lives |

- **Both worlds that died without the rule lived under it.** No agent in
  them ever held more than 916 and 765 tokens, where without the rule
  the richest agent held 4,827 and 6,005 shortly before the end.
- **Coalitions kept winning.** In 57 of the 60 windows under a head count
  at least nine nodes in ten were won by a coalition, and in the other
  three 85 to 89 in a hundred.
- **But the order is not as still.** Three of the four worlds left the
  measure for some windows, 13 windows in all, every time with more
  births than the measure allows. Without the rule the same worlds had
  almost none.
- **One world grew and settled again.** The world with 30 numbers and
  seed 6301 grew from 1,006 agents with 26 links each to 2,736 with nine
  links each by iteration 27,000. Then births stopped, and it was a world
  of equals to the end, with about a third of the links it had.
- **One had a burst and came back.** The world with 30 numbers and seed
  6302 grew to 1,490 agents between iterations 25,500 and 26,500, with
  44,825 deaths in 500 iterations and a richest agent holding 916. By
  iteration 28,500 it was back at 482 agents with 56 links each, and its
  richest agent held about 80. Births began again in the last window.
- **One was in a burst when the run ended.** The world with 50 numbers
  and seed 6302 was a world of equals for its first 4,500 iterations.
  Births began at about iteration 26,500. From iteration 28,000 it had
  up to 0.77 births per agent and iteration, and at the end 2,016
  agents with 12 links each and a richest agent holding 765. Even then 96
  nodes in a hundred were won by a coalition.
- **One did not move.** The world with 50 numbers and seed 6303 was a
  world of equals in all 15 windows and lost 92 of its 651 agents.

**What this shows.** A world of equals lasts under a head count, also
with the larger steps that ended it before. That is now seen three times:
against a hoarder written by hand (Experiment 66), against the lines that
evolved (Experiment 68), and over 7,500 iterations of a world's own
mutations here. What it lasts as is new. Under a head count these worlds
have times of many births and deaths while coalitions still win nearly
every node, and the two that came out of such a time came out as worlds
of small holdings again.

**What it does not show.** Each world was played once, with a seed of its
own, so each differs from its run without the rule by the rule and by
luck. One world's burst had not ended; whether it settles, as the other
two did, or goes to the sparse world that Experiment 70 found, is not
known. And what starts a burst has not been read: whether it is a line of
brains that has many children, and whether such a line is better at
anything, is the question for a world that is to keep evolving.

The numbers are made by `python3 research/efficient/E50/e69.py E71`.

<!-- runs E71 -->
> [!info]- The runs behind this page
> **4 runs.** Every condition below is run once for every seed (7101), played on or again from a kept world of its own to iteration 30,000 — or until its world dies out. Every setting is B1's unless the condition changes it.
>
> - **30 numbers, the world of seed 6301, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens; played on from `B1-40000-8020c3-s6301-on210139` at iteration 22,500. Run `B1-40000-142067-s7101-oncc8e47`.
> - **30 numbers, the world of seed 6302, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 30, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens; played on from `B1-40000-8020c3-s6302-on210139` at iteration 22,500. Run `B1-40000-142067-s7101-on9f14d5`.
> - **50 numbers, the world of seed 6302, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 50, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens; played on from `B1-40000-3f4b02-s6302-on210139` at iteration 22,500. Run `B1-40000-eee3e3-s7101-one84bf9`.
> - **50 numbers, the world of seed 6303, by heads** — the settings of world 58 of [Experiment 40](a-screen-of-random-worlds.md#the-worlds-of-round-3) (seed 3058), as drawn but with `mutation_numbers` = 50, `wealth_guard` = heads, `n_nodes` = 800; 40,000 tokens; played on from `B1-40000-3f4b02-s6303-on210139` at iteration 22,500. Run `B1-40000-eee3e3-s7101-on192fc3`.
>
> To make these runs again: `python3 gol_lab.py run E71`.
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
`python3 research/efficient/E50/autopsy.py` and
`autopsy_lines.py`; Experiments 64 and 65 by
`python3 research/efficient/E50/e64.py` (with `E65` after it); Experiments 66,
67 and 68 by `python3 research/efficient/E50/e66.py`, `e67.py` and `e68.py`;
Experiments 69, 70 and 71 by `python3 research/efficient/E50/e69.py` (with
`E70` or `E71` after it). What the agents
of the world of equals do is read from the statistics of Experiment 53's
runs by `python3 research/efficient/E50/e53.py`.
