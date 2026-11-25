# Results of the efficient plan, as they come in

One entry a card. The plan is `plan.md`; each card's folder has the code and
the detail. Nothing here played a world unless it says so.

## Where this leaves the search (24 November 2026, after Experiments 49 to 58)

Open-ended evolution was not found. What was found is why it is not there
in the worlds we had, and a new kind of world.

**Why the old worlds stop**
1. *Variation.* One mutation of the engine makes a brain decide differently
   in a third to two thirds of what it meets (B1), and costs a line about a
   third of its copies within three iterations (Experiment 49).
2. *Selection.* A line's success is mostly its place and luck; its brain
   counts for 1% to 6% of the differences (Experiment 52). Under the
   lottery, which most worlds play, every way of staking returns the same
   tokens on average (D2).
3. *Lives.* In nearly every world an agent lives one or two iterations
   (A8).
4. *The game.* Nothing in one game rewards a body of several agents (D2).

**What small mutations change**
- A way of playing that needs nearly everyone to follow it becomes stable.
  Under the coalition rule that is the **world of equals**: a few hundred
  agents with the same links and tokens, never born, never dying, all
  staking one token on every neighbour (Experiments 50, 53, 56, 57, 58). It
  formed in five separate worlds and in no world with the engine's
  mutation. It is the first lasting structure of many agents that formed by
  itself.
- Its brains still improve for about 2,000 iterations, then stop
  (Experiment 54). It lasted every run of up to 5,000 iterations, but it
  is not safe: over 10,000 iterations two died out of it (Experiment 63).
- Small steps are dangerous for a young or small world: most died sooner
  (Experiments 56, 57).
- If brains inherit their step, it shrinks to a handful of numbers within
  500 iterations, everywhere (Experiment 58).

**Added after Experiments 59 to 62.** The order is not defended: a few
brains that play otherwise do as well as the rest (62), and many of them
take the world's tokens (59). So the world of equals is fallen into, not
selected for, and it wears away at a rate set by the step of a mutation
(60). At 30 numbers one world formed it, lost it, formed it again and lost
it again; another died (61). Nine more worlds (63) showed that this was
the exception: worlds keep the order or never find it, and two worlds of
equals died out of it suddenly. So the order is not safe either.

**Added after Experiment 64.** One hoarder does not explain those deaths:
an agent that stakes everything on one node grows rich, up to seven tokens
in ten of the world, but all 30 worlds lived through 1,000 iterations with
one. What ended the two worlds is open.

**Added after the line autopsy and Experiment 65.** It is no longer open.
Each world was taken by one line of hoarders that multiplies, and a
hand-written hoarder that spreads once it is rich does the same in 30 of
30 trials (25 worlds end). The coalition rule guards a node only against
stakes smaller than the coalition's sum, so the order of equals lasts only
until mutation finds wealth. An order that is to last needs a rule that
wealth cannot break (not tested; a rule change, the author's call).

**So there are two kinds of world, and neither is open-ended.** With large
steps: churn, where nothing that is found is kept. With small steps: an
order that keeps itself and then stands still. And the step itself, left to
evolve, goes to the standing-still side at once, because a line that
changes less loses less.

**What would have to be added, in my judgement** (not tested):
- *A reason to keep changing.* Something outside the agents that moves, so
  that standing still stops paying: tokens that enter and leave (the plan
  *A source and a sink*), or
- *several orders that meet.* A world of equals is one society. Selection
  could work on societies if there were many, each with its own way, in
  contact at their edges. Today the rules merge everything into one piece
  (only the largest piece of the network lives, and anyone may link to
  anyone). Experiment 59 is a first look: one society's brains against
  another's.
- *The step between.* With 100 numbers a world formed an order and lost it
  again. Between "kept for ever" and "never kept" there may be a step at
  which orders keep forming and breaking. That can be mapped in a day.

**What is now cheap.** Each of these takes minutes to hours: how much a
mutation changes (B1), what it costs (the assay, read after 3 to 9
iterations), how much of success is the brain (Experiment 52's design),
playing a kept world on under other rules, and contests between runs. No
question above needs a run of days.

## Wave 1 (23 November 2026)

**The picture so far.** In Experiment 48's three worlds, selection sorts
brains but cannot improve one. A mutation turns a brain into something close
to an unrelated brain; the population has no common way of deciding; what it
does is reshuffled inside bounds; and the rules pay for having children, not
for playing well.

### B1. How big is one mutation, in behaviour? — too coarse

Share of 300 real situations in which a mutant decides differently from its
parent (4,000 mutants of 200 brains for each cell):

| world, iteration | two unrelated brains | engine | 1 number | 10 | 100 | 1,000 |
|---|---|---|---|---|---|---|
| s4003, 15,000 | 76% | 67% | 0.4% | 2.0% | 6.5% | 20% |
| s4003, 5,000 | 62% | 54% | 0.2% | 0.9% | 3.3% | 10% |
| s4008, 15,000 | 83% | 49% | 1.1% | 5.6% | 18% | 43% |
| s4002, 15,000 | 83% | 34% | 0.4% | 2.3% | 7.7% | 21% |

By the card's rule (more than half of the engine's mutants further than a
quarter of the distance between strangers): 69% to 98% are. The engine's
operator is too coarse in all four. In s4003 at 15,000, 47% of mutants are
as far from their parent as a stranger.

### B2. How many things can evolved brains do? — not low-dimensional

Among 1,000 brains of one kept world there are 420 to 910 genotypes. They
agree with the most common first move on a probe 39% to 66% of the time
(random founders: 30% to 35%), and 90% of the variance between them needs
110 to 200 components. No simple rule describes them, with one exception:
in s4008 at 17,500, an agent holding 2 or more tokens has a child in 98% to
99% of cases. A stand-in model (card E2) is not justified.

### A1. Does behaviour keep moving? — bounded reshuffling

Distance between the populations of two kept worlds, by the time between
them (first-token decisions; the floor is two halves of one population):

| world | floor | 2,500 apart | 10,000 apart | longest | slope |
|---|---|---|---|---|---|
| s4003 | 0.06 | 0.43 | 0.47 | 0.49 (12,500) | 0.03 |
| s4008 | 0.07 | 0.27 | 0.27 | 0.25 (15,000) | −0.06 |
| s4002 | 0.06 | 0.36 | 0.33 | 0.31 (25,000) | −0.04 |

Far above the floor at the shortest lag, and no growth with the lag. By the
card's rule: bounded wandering, faster than one kept world to the next. The
populations are also no further from their random founders than from each
other.

### D1. The rules as physics — five laws hold, five were wrong in the plan

Checked on 100 game frames of s4003 (iterations 12,000 to 12,099). Detail in
`D1/RESULT.md`.

Hold exactly: tokens are conserved; a token moves at most one link a game;
a node keeps every token staked on it; a failed attack feeds the defender;
a node with no stake starves.

Corrected:
- These worlds play the **lottery**. The largest stake loses 14.7% of
  contested nodes, and clones' stakes add up. "Many small kin stakes do not
  defend" is false here.
- **Estates go to the parents of this iteration's newborns** (`redistribution`
  = parents), one share for each child, not to everyone alike. 27% of all
  tokens move this way every iteration.
- **A core is safe only within one game.** 62% of nodes whose neighbours
  all share their genotype have a neighbour of another genotype one
  iteration later, mostly a mutant child of kin.
- A link is cut only if no stake crossed it, it was not paid for and it was
  not made this iteration.
- An agent that places its tokens one at a time also sees the board.

Numbers that matter for everything else:
- **An agent lives a median of 1 iteration**; 99% live under 9.
- About 830 agents an iteration die in culls, 390 starve, of 1,900 to 6,400.
- 43% of agents hold 1 token, 23% hold 2.
- **The surface law fails**: 79% to 92% of a territory's nodes are border
  nodes at every size. Territories are thin, not round.
- Help from more than 2 to 4 links away arrives after the agent is dead.

### A2. Where do tokens come from? — the world pays for children

Tokens per agent and iteration in the game, s4003, iterations 12,000 to
12,499:

| group | taken from others | lost to others | from estates |
|---|---|---|---|
| lineages that grew most | 0.10 | 0.10 | 0.49 |
| lineages that shrank most | 0.18 | 0.18 | 0.40 |
| the whole world | 0.20 | 0.20 | 0.45 |

Winning lineages get about five times as much from estates as from the
game, and take no more than they lose. By the card's rule: the world pays
per child. `redistribution` is the first rule to study.

### A6. The entropy cycle

About a fifth of all tokens pass through the estates of the dead every
iteration in all three worlds. s4003 drifts towards its most mixed state
(tokens per agent 4.0 to 2.9, agents with one token 23% to 33%); s4008 moves
away from it and stays there; s4002 falls slowly.

### A5. Does the largest structure grow? — no

Territories of one genotype exist well beyond a shuffled world in all three
(s4003: 340 to 420 of two or more agents against 70 to 150 shuffled; largest
70 to 170 against 10 to 90). In s4002 and s4008 they stay small (largest
about 6 and 12). None grows from window to window by the card's rule. Not
done: lifetimes, family-based territories, the family-pool runs.

### A8. How long do agents live? (added) — one or two iterations, nearly everywhere

From `research/screen/health.json` alone, for round 3's 82 living worlds. An
agent's lifetime is 1 / (births per agent and iteration).

- **Median lifetime: 1.7 iterations.** 70 of the 82 worlds are under 5; only
  4 reach 10 (s3004 19, s3043 19, s3065 18, s3019 12).
- Median holding: 6.9 tokens an agent (1.4 to 31).
- Lifetime goes with where estates go: to the children of the dead 3.5, to
  the neighbourhood 4.4; to newborns, parents or the poorest 1.5 to 1.6.
- The evolution rank does not see it (rank correlation 0.05).

So in nearly every world an agent decides once or twice and is gone. Nothing
that needs a body lasting many iterations (memory, a shell, a partner met
twice) can pay there. What lasts is a line of copies moving through nodes.
The four long-lived worlds are the ones to read for structure; they are on
the external drive.

### A9. The same readings on round 3's long-lived worlds (added)

Worlds s3004, s3043, s3065, s3019 (agents live 12 to 19 iterations), s3080
(8) and s3058 (6), at 30,000 tokens, kept every 250 iterations to 3,000.

| world | engine mutation: decisions changed | as a share of the distance between strangers | behaviour 250 apart | 2,750 apart | slope |
|---|---|---|---|---|---|
| s3004 | 26% | 0.23 | 0.21 | 0.25 | 0.11 |
| s3043 | 14% | 0.12 | 0.17 | 0.30 | 0.23 |
| s3065 | 23% | 0.17 | 0.20 | 0.19 | 0.00 |
| s3019 | 29% | 0.29 | 0.13 | 0.12 | 0.01 |
| s3080 | 41% | 0.48 | 0.08 | 0.08 | 0.01 |
| s3058 | 35% | 0.35 | 0.30 | 0.48 | 0.13 |

(Behaviour: distance between the populations' first-token decisions; the
floor between two halves of one population is 0.06 to 0.07.)

- A mutation is smaller here than in Experiment 48's worlds, yet still
  changes 14% to 41% of decisions. Ten numbers changed: 1% to 7%.
- In every world two brains drawn at random differ on about three quarters
  of the probes, as random founders do. Where a brain sends its first token
  is hardly agreed on anywhere. What is agreed on is having no child in most
  situations (87% to 92%).
- Three kinds of world show: populations that stand still near the noise
  floor (s3080, s3019), populations reshuffled at once and then flat (s3065,
  and Experiment 48's three), and populations that move away slowly (s3043,
  s3004, s3058).

So "bounded reshuffling" (A1) is not the only pattern. The worlds that move
away slowly are the ones with small mutations in place or long lives.
`python3 research/efficient/A9/a9.py`; numbers in `A9/a9.json`.

### C1. The tagged-lineage assay — built

A lab trial of kind `assay` (`gol_tournament.assay`), with tests. Its first
use is Experiment 49 (card C2).

## Wave 2 (23 November 2026, evening)

### D2. Exact small games — size does not pay in the game itself

One game phase, solved exactly: a team of 1 to 4 agents with one joint plan
(a path or a clique, 2 or 3 tokens each) against one outsider, who is linked
to one team node ("one border") or to all of them ("all border") and is as
rich as the whole team or as one member. The team's value is the tokens it
holds afterwards over those it had (1.0: it keeps what it had). Detail and
plans in `D2/RESULT.md`.

| contest | outsider linked to | outsider as rich as | team's value per token, n = 1, 2, 3, 4 |
|---|---|---|---|
| lottery | any | any | 1.00, 1.00, 1.00, 1.00 exactly |
| largest stake | one node | any | 1.00, 1.00, 1.00, 1.00 |
| largest stake | all nodes | the team | 1.00, 0.67, 0.50, 0.40 |
| largest stake | all nodes | one member | 1.00, 1.00, 1.00, 1.00 |
| stakes pooled | one node, or rich outsider | any | 1.00, 1.00, 1.00, 1.00 |
| stakes pooled | all nodes | one member | 1.00, 1.17, 1.17, 1.15 |

- **Under the lottery the staking game is a fair bet.** A staker's chance is
  its share of the stakes on a node and the node holds all of them, so every
  plan gives the tokens it put in, on average. No plan is better than
  another in tokens. 50 of round 3's 82 living worlds and all three of
  Experiment 48's play the lottery. What is left to select there is how
  many nodes a line holds, its children, its links and its estates. This
  fits A2 (income comes from estates, not from the game) and B2 (brains do
  not agree on where the first token goes: it hardly matters).
- **Under the largest stake, one rich agent beats a team of poor ones.**
  Against an outsider as rich as the whole team and linked to all of it, the
  team keeps 67%, 50%, 40% of its tokens as it grows from 2 to 4. Being many
  is a weakness. With one border node the team can only save itself by
  emptying that node.
- **Only pooled stakes let a team gain**, and only against a poorer
  outsider it surrounds.

By the card's rule, size does not pay under any of the three. Condition 4
of the plan fails for the game as it is: nothing in one game rewards a body
of several agents. Limits: one phase, no children, no link cuts, no
estates, teams of at most 4, revolutions left out.

### C2. Experiment 49 — refuted by the letter, and the letter read noise

150 assays, 2 hours 23 minutes. Page: *Are mutants as fit as their parents?*

- **The registered reading (count at iteration 30) decides nothing.** Lines
  die so fast that by 30 one line holds hundreds to thousands of agents and
  the ratio is a coin toss: it ranges from 0 to 3,261, as it does with the
  labels shuffled. By the thesis's own words it is refuted (the ratio for
  one number changed is under 0.8 in two worlds). Lesson for every later
  assay: read after 3 to 9 iterations, not 30.
- **The first iterations show a pattern** (not registered). In s4003 at
  15,000 and at 5,000, a line changed by the engine's mutation has 0.69
  times the copies of an unchanged one after three iterations (shuffled
  range 0.84 to 1.19) and 0.41 and 0.58 times after six. Mutations of 1, 10
  and 100 numbers stay inside the shuffled range everywhere.
- **In s4008 no mutation costs anything**, the engine's included, though it
  changes half of what a brain decides there. What a brain decides hardly
  matters for a line's next few iterations in that world.
- **Lines die fast**: a third alive after 3 iterations, about 3% after 9,
  a few in a thousand after 30.

### C6. Experiment 52: brain, place or luck? — mostly place; the brain where the largest stake wins

400 marked agents of a kept world, replayed six times with other luck and
six times with their brains swapped (24 assays of 9 iterations, minutes).
Correlations of a line's log copies after six iterations:

| world | brain in another place | place with another brain | same brain, same place again |
|---|---|---|---|
| s3058 at 20,000 (largest stake) | 0.25 (0.15 to 0.35) | 0.58 | 0.52 |
| s4003 at 15,000 (lottery) | 0.09 (−0.02 to 0.19) | 0.40 | 0.36 |

Thesis undecided (it holds in s3058, not in s4003). The reading: the place
a line starts in decides most of what repeats; in s3058 about 6% of the
differences between lines travel with the brain, in s4003 about 1%. **How
much selection has to work with is a property of the rules**, and it can be
measured in minutes for any kept world. This is the cheapest test found so
far of whether a world can evolve at all.

### Experiment 50, the branches (before their contests)

The world played on (s3058 at 40,000 tokens, iteration 20,000) had about
5,000 agents, 2.4 links an agent, and one agent linked to a third of all
the others. After 2,500 more iterations:

| branch | agents | links an agent | births per agent | most links one agent has |
|---|---|---|---|---|
| as it is, seed 5001 | 910 | 4.6 | 0.16 | 38 |
| as it is, seed 5002 | 3,850 | 2.9 | 0.20 | 1,578 |
| 10 numbers, seed 5001 | 484 | 55 | 0.000 | 175 |
| 10 numbers, seed 5002 | 683 | 40 | 0.002 | 164 |
| 100 numbers, seed 5001 | 3,162 | 5.2 | 0.08 | 150 |
| 100 numbers, seed 5002 | 6,580 | 2.0 | 0.06 | 6,579 |

(means of the last 500 iterations)

- Both branches with mutations of 10 numbers ended in the same new state:
  500 to 700 agents with 40 to 55 links each, holding about 40 to 55
  tokens, and almost no births, deaths or starving. Brains spread only by
  winning nodes; in the first of the two, 70% to 88% of nodes change hands
  in every game. One reached that state by iteration 20,600, the other
  after 21,500.
- One branch with 100 numbers passed through that state and left it; the
  other became a star: one agent linked to every other.
- One branch as it is thinned to 900 agents and lost its hub; the other
  stayed as the world was.

Six branches, four different worlds. This world has several stable states,
and which one a branch reaches depends on its mutations and its luck. Two
of two for the dense state under the smallest step is suggestive, not
proof. Whether the brains that made it are better is Experiment 51.

### Experiment 53: the core played on — a world of equals under small mutations, collapse under the engine's

From the core world (10 numbers, seed 5001, iteration 22,500), 5,000
iterations more. Thesis (the core lasts under small mutations, falls apart
under the engine's): **refuted in its first half** by its own definition,
because the core spread to the whole world.

- **10 numbers, both seeds:** 476 and 472 agents, the same ones from
  iteration 23,000 (25,000) to 27,500: no births, no deaths. Every agent
  stakes one token on each of its ~56 neighbours and itself, all marked for
  a coalition; every node is won by a coalition every game; 98% of nodes get
  a new brain every game. In one run links even out to 50 to 64 an agent
  and the Gini of tokens falls 0.43 → 0.11 → 0.03.
- **The engine's mutation, both seeds:** within 500 iterations 900 to 1,500
  agents and 55,000 to 150,000 births and deaths per 500 iterations; later
  1,400 to 5,000 agents, and in one run an agent with over 2,000 links.

Reading: a convention that only works if nearly everyone follows it lasts
where copies are faithful and nowhere else. The coalition rule is what
makes "many small stakes" beat "one large stake". It is order, close to the
most mixed state of the tokens, and may be an end point: whether brains
still improve inside it is open.

### Experiment 56: small mutations from founders — four of five worlds die; the fifth becomes a world of equals

s3058's rules from founders with mutations of 10 numbers. At 10,000 tokens:
died at 71 and 347, lived 5,000 (a world of equals from iteration 1,000:
108 agents, 61 links and 62 tokens each). At 40,000 tokens: died at 205 and
1,322. **Thesis refuted.** Small steps are too slow for founders' random
brains; yet the world of equals needs neither the long history nor the
size. Large steps find brains that live, small steps keep what was found:
Experiment 57 plays the worlds that died in Experiment 46 on from iteration
500 with small mutations. The principled form of this is a step size each
brain inherits and that itself mutates.

### Experiment 54: is the world of equals an end? — not at once

96 mirrored contests inside Experiment 53's two worlds of equals. Later
brains' share: 25,000 against 23,000 **0.62 (0.54 to 0.71)**; 27,500
against 25,000 0.52 (0.46 to 0.59); 27,500 against 23,000 0.66 (0.58 to
0.73). Thesis (ties) refuted: brains got better for about 2,000 iterations
after the world of equals formed, then stopped.

### Experiment 57: large steps for 500 iterations, then small — a bet, mostly lost

Experiment 46's three dead worlds (10,000 tokens) played on from iteration
500. With 10 numbers: died at 522, 529, 953, 574, 740; one lived to 5,500 as
a world of equals. As they were: all six died, at 534 to 2,987. Thesis
refuted. Small steps make a small world die sooner, unless it reaches the
world of equals, which is then safe.

### Experiment 58: a step each brain inherits — it shrinks at once

`inherited_step`, founders at 1,357 numbers. In every world the middle step
fell to 4 to 212 within 500 iterations and stayed at a handful (8 at
iteration 5,000 in both survivors). 10,000 tokens: died at 112, 1,206,
4,549. 40,000 tokens: one a world of equals from iteration 500, one alive
with births and deaths. Thesis refuted on survival. **Selection on the step
itself goes only one way, down**: a line that mutates less loses fewer
copies. Nothing rewards exploring.

### The world of equals, counted

Ended as one: 6 of 20 runs with mutations of 10 numbers or an inherited
step (five separate worlds: from a 20,000-iteration world twice, from
founders at 10,000 and at 40,000 tokens, from a 500-iteration world), 0 of
10 runs with the engine's mutation. With 100 numbers one world reached it
and left it.

### Experiments 50 and 51: the verdict — undecided

Contests of the six branches (all 192). (1) End against start, agents'
share: 10 numbers **0.72 (0.60 to 0.84)**; 100 numbers 0.22 (0.14 to 0.30);
as it is 0.30 (0.10 to 0.49). (2) Against the arm as it is: 10 numbers 0.63
(0.42 to 0.83), 100 numbers 0.65 (0.47 to 0.84); by seed 0.28 and 0.97,
0.47 and 0.83. The world before: 20,000 against 17,500 0.91, against 15,000
0.90. By tokens the same, but 10 numbers against the start is 0.55 (0.48 to
0.62).

- The claim's first part holds for 10 numbers; the second is not settled,
  because the branches had become different kinds of world.
- **The source world had not stopped at 20,000**: it beat its past nine
  times in ten, and then the unchanged branches lost to that moment. This
  world swings between kinds of world. A contest between two moments says
  little when the world's kind changed in between.

### Experiment 60: the step between — order forms and breaks at 30 numbers

From the 20,000 world, ten windows of 500 (E = a world of equals): 30
numbers `.EEE......` and `.EEEEEE...`; 300 numbers none in either run; 1,000
numbers none, and one run died at 21,863. Thesis undecided (it expected 30
to keep the order and 300 to have it part of the time). The step at which
order stops lasting lies between 10 and 30 of a brain's 13,569 numbers.
Experiment 61 plays both 30-number worlds on for 10,000 iterations: does
the order form again?

### Experiment 59: can the world of equals be invaded? — yes, by many

Half the agents of a world given the brains of the world of equals, half
those of the world that fell back under the engine's mutation (48 mirrored
contests). Equals' share on their own worlds: agents 0.59 (0.36 to 0.83),
tokens **0.36 (0.24 to 0.48)**; on the engine's worlds: agents 0.29 (0.12
to 0.45), tokens 0.24. Thesis undecided (it holds abroad, not at home). The
convention does not protect itself against a large share of others. The
two worlds of equals against each other: 0.42 at home and away, no home
advantage. Not tested: a few invaders (an assay in the world of equals
would do it in minutes).

### Experiment 61: does order form again at 30 numbers? — once yes, once the world died

Both 30-number worlds played on from 25,000 to 35,000. From seed 6002's
run: `.............EE.....`, a second world of equals at 31,500 for 1,000
iterations (335 agents), then 4,000 agents again. From seed 6001's run:
none, and it died out at 33,061 after a stretch with about 4,300 agents, no
births and hardly any node won by a coalition. Thesis undecided. One world
goes back and forth between order and churn; worlds of this step can also
die.

### Experiment 62: a few deviants in a world of equals — not selected against

48 assays of 9 iterations in the two worlds of equals. Mean copies of
changed lines over unchanged after six iterations: 10 numbers 1.02 and
1.13; 100 numbers 1.11 and 1.20; 1,000 numbers 1.07 and 0.99; every one
inside its shuffled range (about 0.78 to 1.28). **Thesis refuted.** A brain
changed in 1,000 numbers does as well as the rest. **The order is not kept
by selection.** It is a state the world falls into and wears away from as
changed brains pile up: 10 numbers lasted 4,500 iterations and more, 30
numbers 1,500 and 3,000, the engine's mutation under 500.

### Experiment 63: is going back and forth typical? — no; and a world of equals can die

Nine worlds from the 20,000 world, 10,000 iterations (E = a window of 500
as a world of equals). 20 numbers: all three `.EEEEEEEEEEEEEEEEEEE`. 30
numbers: `.EEE.EEEEEEEEEEEEEEE`; 18 windows, then died at 29,351; none, died
at 26,699. 50 numbers: none; 5 windows, then died at 23,853; 18 windows.
Thesis undecided (one run of nine with two stretches; three died).

- Worlds form the order early and keep it, or never form it. Going back
  and forth (Experiment 61) was the exception.
- **Correction: a world of equals is not safe.** Two died out of the order
  within a few hundred iterations, after 2,500 and 9,000 iterations in it.
  Earlier experiments were too short (at most 5,000) to see that.
- Deaths by step: 0 of 3 at 20 numbers, 2 of 3 at 30, 1 of 3 at 50.
- **How they died** (`E50/autopsy.py`): one agent grows rich (the richest
  from 135 tokens to 220, 979, 1,686 within a hundred iterations, where the
  rest hold about 46), links are cut by the hundred each game, pieces are
  culled, the world is gone. Likely mechanism, untested: an agent that puts
  all its tokens on one neighbour's node outweighs the coalition there by
  one, takes everything staked on it, and repeats with twice as much. The
  order has no defence against it; small steps only make it rare.

### Experiment 64: does one hoarder end a world of equals? — no (refuted)

45 invasions of 1,000 iterations in the three 20-number worlds of
Experiment 63 at 27,500: one agent given its own brain, frozen, with only
its stakes written over (new masks `home-hoarder`, `moving-hoarder` in
`gol_strategies.py`), and the same agent's brain unchanged as control.
Read by `E50/e64.py`.

- 22 of 30 hoarders pass 4,000 tokens (all 15 moving, 7 of 15 at home); the
  richest holds 28,118 of 40,000. None loses its first game; at home the
  first game doubles its tokens (1.9 to 2.0 times).
- 0 of 30 worlds end. Agents left at 1,000: median 100% (home), 97%
  (moving), least 73%. Links left: median 88% and 76%, least 36%.
- 5 of 15 home hoarders are gone between 441 and 635 with 1,029 to 10,705
  tokens, after gaining 1 to 3 tokens a game: it fits their links lapsing
  and the cull removing them (not recorded, a reading).
- Growth differs by world: seed 6301 fast (all ten over 4,000 within 137
  iterations), seeds 6302 and 6303 slow. Why is not read.
- Controls last a median of 2 iterations as a line (brains on a node are
  drawn anew from the coalition each game); one drifted to 47 copies.
- Still growing at 1,000 (tokens ×1.16 to ×1.21 over the last 250).
- So the rich agent in Experiment 63's dying worlds was part of a wider
  breakdown, not its one cause.

### Who took the two worlds that died (frames of Experiment 63, nothing played)

`E50/autopsy_lines.py` follows the brain line of the rich agents (frames
record each brain's parent).

- In both worlds all rich agents come from one brain, first seen 88 and
  about 490 iterations before the end (29,262 of 29,350; 23,359 of 23,852).
- The line goes from 1 agent to over 90% of agents in 85 and 200
  iterations and holds 96% to 100% of the tokens. Nodes turn to it in the
  game (5,007 and 2,363), hardly by births (542 and 45); all but one in
  each world next to a line agent.
- In the one recorded game (23,500) all 12 line agents stake everything on
  one node (6 home, 6 a neighbour's) and win; 2 of 510 others do.
- All-in alone cannot grow a line (take one node, leave one). At 23,511,
  22 neighbours of one line agent with 1,629 tokens and 63 links turned in
  one game: it spread its stakes.
- Reading: a line that hoards and then spreads. The coalition rule guards
  a node only against stakes smaller than the coalition's sum, so wealth
  breaks it. Tested by hand in Experiment 65.

### Experiment 65: a hoarder that spreads — the world ends (confirmed)

30 invasions as in Experiment 64, with masks `home-spreader` and
`moving-spreader`: hoard until richer than all neighbours together, then
stake on every neighbour's node in proportion to its holder's tokens.
Read by `E50/e64.py E65`.

- 30 of 30 lines hold more than half the agents, at iteration 33 to 134
  (medians 71 at home, 89 moving). 25 of 30 worlds end (fewer than half
  their agents at 1,000); 26 fall to 20 agents or fewer (median at 107);
  22 end with 1 to 5 agents, 11 with one agent holding all 40,000 tokens.
- The 5 that live are all in the densest world (seed 6302: 257 agents,
  103 links each): afterwards 269 to 894 agents with about 3 links each.
- Comparison: the same agents hoarding only (Experiment 64): 0 of 30 end.
- First confirmed thesis of the series.

## What this changes in the plan

- **Condition 2 fails as far as behaviour goes** (B1). Experiment 49 asks
  whether it fails in fitness too.
- **Condition 4 cannot hold in these three worlds as they are**: agents live
  one iteration, territories are all border, and income comes from having
  children. A shell and a core (cards A3, D3, D4) need agents that last. The
  rule sets to look at next are those where agents live long: no cull of
  whole pieces every game (`mend_on_death`), estates not paid for births.
- All three of Experiment 48's worlds share `contest` = lottery and
  `redistribution` = parents. The evolution rank that chose them rewards
  fast turnover.
