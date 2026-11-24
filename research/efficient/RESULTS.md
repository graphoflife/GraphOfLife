# Results of the efficient plan, as they come in

One entry a card. The plan is `plan.md`; each card's folder has the code and
the detail. Nothing here played a world unless it says so.

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
