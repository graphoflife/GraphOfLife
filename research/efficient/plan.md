# Finding open-ended evolution without long runs

Written 23 November 2026, at the author's request: no experiment longer than
one day, use computation well, think beyond one agent, and write every idea
down so that a helper with less background can carry it out.

This is a plan, not a result. Nothing in it has been run yet, except where a
line says "known".

---

## 1. The idea in one page

We cannot watch open-ended evolution (OEE) happen. It may need billions of
tokens and far more time than we have. So we stop waiting for it and ask
instead **whether its necessary conditions hold**. Each condition can be
checked in hours.

| # | Condition | If it fails |
|---|---|---|
| 1 | **Better brains exist** than the ones a world has reached | the game is at its top: change the game, not the run length |
| 2 | **Mutation can reach them** in small steps | the search is the limit: change how brains vary |
| 3 | **The ladder has no top**: each better strategy can be beaten by a next one, not in a circle | evolution ends, or goes round |
| 4 | **Size and organisation pay**: a group of n agents does better per token than n loners, and more so as n grows | no larger structures will ever be stable |
| 5 | **Larger structures already appear** in the recorded worlds, and their largest scale grows | the rules do not carry structure upward |

If all five hold, OEE is plausible and one long run is worth its cost. If one
fails, that one is what to change, and no long run would have told us.

Five ways to check them, cheapest first:

- **A. Read what is already recorded** (no simulation).
- **B. Play with brains alone** (numbers through a network, no world).
- **C. Short assays in kept worlds** (minutes each, through the lab).
- **D. Think it through as an engineer**: the rules as physics, groups as
  machines (pencil, exact small games, hand-built organisms).
- **E. Other ways to simulate**, only if A to D say they are worth it.

### What is known already

- **Evolution levels off early everywhere it was measured.** Round 3
  (Experiment 45): better by iteration 500, then flat. Experiment 48, stopped
  on 23 November at 17,193, 27,509 and 18,301 iterations: all three worlds
  beat their founders by iteration 5,000 (scores 1.0, 0.87, 1.0); the eight
  later checks against 5,000 iterations before average 0.53 (0.50, 0.25, 0.87,
  0.51, 0.63, 0.50, 0.38, 0.64), each with a band that includes 0.5.
- **A better strategy existed and evolution did not find it.** A raider
  written by hand took 40% and 62% of two evolved B2 worlds, and 12 units of
  a brain can compute it (pages *Can a strategy written by hand beat
  evolution?* and *What a brain can compute*). This points at condition 2.
- **One mutation is very large**: it changes 10% to 30% of all weights, and
  in 57% to 94% of cases redraws part of a layer (review of 22 November).
- **Worlds grow crowded and poor.** In Experiment 48's s4003 world an agent
  holds about 2 tokens (3,600 to 4,400 agents on 10,000 tokens). With 2
  tokens there are only a handful of possible moves.
- **Seed and size barely change a world's statistics** (Experiment 43), so
  one seed and 10,000 tokens are enough for most readings.
- **Family pools kept worlds alive** and made many poor agents (Experiment
  46). Whether their brains kept getting better is unknown.
- **Novelty is above its null in none of round 3's 82 worlds**; selection
  and heredity show in all.

---

## 2. Rules for everyone who works on this

1. **One day at most** for any experiment, counted as wall-clock time on
   this computer. State the expected cost before starting. Stop and report
   if it will be exceeded.
2. **Worlds are only played through the lab** (a plan in
   `book/experiments/`, started with ▶ or at the author's word). Reading
   files and running numbers through brains is not a simulation and needs
   no lab. Never run background pilots.
3. **Every world has at least 10,000 tokens.** Exact small games (section D2)
   are mathematics, not worlds, and are exempt.
4. **No tuned constants in rules.** Settings of a measurement (how long an
   assay lasts, how many copies are planted) are allowed; write them down.
5. **Write the decision rule first.** Each card below has one. Do not change
   it after seeing the numbers; add a dated note instead.
6. **Report honestly**: the number of samples, the noise, what failed, what
   was skipped.
7. **Never delete or move run data.** The external drive (`Speicher2`) is
   unreliable: read from it, never write to it, and say so if a read fails.
8. **Work inside the worktree** `.claude/worktrees/notebook`. Never `cd` out
   of it; use absolute paths. Set
   `GOL_RUNS_DIR=/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns`.
9. **At most 3 heavy processes at once.**
10. Put each card's code and notes in `research/efficient/<card>/`, with a
    `LOG.md` (what was done, in order, with numbers) and a `RESULT.md` (one
    page: the answer to the card's question, the evidence, the caveats).
    Simple English.

### Where things are

| Thing | How to read it |
|---|---|
| A run's settings | `gol_store.load_meta(run_id)["config"]`, then `SimConfig.from_dict(...)` |
| Its statistics, one row per phase | `gol_record.read_stats(run_id)` |
| One frame (a phase) | `gol_store.read_frame(run_id, index)`; index = 2 × iteration + phase − 1. Keys: `ids`, `tokens`, `brain_ids`, `parent_ids`, `ages`, `edges`, `cleanup`, `decisions` |
| Decisions of a game frame | `decisions["allocations"]` (each: `agent`, `tokens`, `targets`, `alloc`), `decisions["winners"]` (each: `node`, `winner`, `amount`) |
| A kept world with every brain | `gol_store.archived(run_id)` lists the iterations; `gol_store.load_archive(run_id, it)` gives arrays: `ids`, `tokens`, `edges`, `brain_ids`, `W0`, `b0`, … (first axis = agent) |
| A world object from a kept world | `GraphOfLife.from_checkpoint(blob, cfg)` |
| What a brain sees and answers | `f = world._precompute_features()`; `X = world._inputs(u, [u] + f[1][u], f[0], f[2], f[3], f[4], f[5])`; `Y = world.brains[u].forward(X)`; rows of `Y` are named by `world.heads` |
| A contest of two eras | `gol_tournament.contest(...)`; as a lab trial: kind `contest` |
| Hand-written strategies | `gol_strategies.py` (`turtle`, `raider`, `crew`, …; planted by an `invasion` trial) |
| The brain-copy tree | `research/screen/braintree.py` |

**The main test bed** is Experiment 48: three rule sets at 10,000 tokens, on
the internal disk, with a whole world kept every 2,500 iterations and every
frame.

| Run | World | Reached | Note |
|---|---|---|---|
| `B1-10000-085a32-s4801` | s4003, strongest first | 17,193 | best for assays |
| `B1-10000-f511a5-s4801` | s4002, weakest first | 27,509 | mixes brains at birth: avoid for assays |
| `B1-10000-1cfaa4-s4801` | s4008, weakest first | 18,301 | brains mutate in place |

Other data: round 3's 82 worlds and Experiment 46 (family pools) are on the
drive; Experiment 43's nine unfinished runs at 50,000 tokens are local.

**Before any card**, read one frame and one kept world and print their keys.
Do not trust this table blindly.

---

## 3. The cards

Each card: the question, why it matters, what it costs, the steps, what to
report, and the decision rule. The order to do them in is in section 4.

### A. Reading what is recorded

#### A1. Does behaviour keep moving? (the wandering test)

**Question.** In behaviour space, does a world's population move ever further
from where it was, or does it wander inside a bounded region?

**Why.** A world that stopped is bounded. A world that drifts moves like a
random walk. A world that evolves in a direction moves away steadily. The
contests cannot tell "stopped" from "going round"; this can.

**Cost.** Minutes. No simulation.

**Steps.**
1. Take one run of Experiment 48 and its kept worlds (every 2,500).
2. Make a **probe set**: from the kept world in the middle of the run, pick
   300 agents at random (fixed seed). For each, compute its input matrix `X`
   (see the table). Keep these 300 matrices. They are the questions every
   brain will be asked. Use a copy of the world's random stream so the noise
   inputs are the same every time.
3. For each kept world, pick 300 brains at random. Feed every probe to every
   brain. From the answer `Y` keep the decisions, not the raw numbers: the
   share of tokens given to a child, the share of stake kept on the own node
   (softmax or the engine's own apportioning of the stake row), whether it
   spreads or goes all in. Experiment 48's worlds place their stakes a token
   at a time with the board in view; here the stake row is read once, with
   an empty board. That is a fingerprint of the brain, not its exact play. One **behaviour vector** per brain: these numbers
   over all 300 probes.
4. For each kept world, the **mean behaviour vector** over its 300 brains,
   and the spread between brains.
5. For every pair of kept worlds, the distance between their mean vectors.
   Plot distance against the time between them (log-log). Fit the slope.
6. A null for the size of noise: split one kept world's brains in two halves
   at random; the distance between the halves' means is the floor.
7. Repeat for the other two runs.

**Report.** The plot, the slope, the floor.

**Decision rule.** Slope near 0 and distances near the floor: stopped.
Distances above the floor that level off: bounded wandering. Slope about 0.5
with no levelling: drift. Slope near 1: directed change.

#### A2. Where do tokens come from? (the budget)

**Question.** For the lineages that win, which channel feeds them: taking
other agents' stakes, estates of the dead, or gifts?

**Why.** It says what the rules really reward. If most income is estates
shared out per head, the best strategy is to have as many nodes as possible,
which explains crowded, poor worlds and tells us which rule to look at.

**Cost.** About an hour of reading frames. No simulation.

**Steps.**
1. Take 500 consecutive iterations of one Experiment 48 run (1,000 frames).
2. In each game frame, for each node: the tokens it ends with are the sum of
   all stakes on it. Split them into: stakes by the winner itself; stakes by
   others of the winner's genotype; stakes by strangers (this is **prey**).
3. For each staker: tokens staked on nodes it did not win are **lost**.
4. Tokens an agent got from the dead are **estate income**: what it holds
   in the frame minus what its node held after the contest (step 2). First
   read `_frame` and `_cleanup_and_redistribute` in the engine to see what
   `delta` and `cleanup` record.
5. Sum per genotype family (use `brain_ids`; group by the ancestor at the
   start of the window through `parent_brain_ids`).
6. Compare the ten lineages that grew most with the ten that shrank most.

**Report.** A table: prey, lost, estate income per agent and iteration, for
winners, losers and the world as a whole.

**Decision rule.** If estate income is larger than prey for the winners,
write "the world pays per head" and flag the `redistribution` rule as the
first rule to study in D1.

#### A3. Is there a hard shell and a soft core already?

**Question.** Do territories of one kin group hold their tokens at the border
and move tokens from inside outwards?

**Why.** This is the author's picture of an organism. If evolved worlds
already show it, larger structure is starting. If not, D4 builds it by hand.

**Cost.** One to two hours. No simulation. Needs the drive for family pools.

**Steps.**
1. Define a **territory**: a connected set of nodes of one family. In the
   family-pool runs of Experiment 46 (`B1-10000-4819e3-s4601` to `s4603`) the
   family is in the frame. In other worlds use one genotype and all
   genotypes within 5 mutations of it.
2. In 200 frames spread over a run, find all territories of at least 10
   nodes.
3. A **border node** has a neighbour outside the territory. All others are
   **core**.
4. Per territory: mean tokens of border nodes and of core nodes; net flow of
   stakes from core to border (from `allocations`); the same per unit of
   threat (the richest outside neighbour).
5. Null: shuffle which nodes are called border inside each territory, keeping
   their number.

**Report.** Border-to-core ratio of tokens, net outward flow, against the
null, by territory size.

**Decision rule.** Shell and core exists if border nodes hold more than core
nodes beyond the null in most territories **and** the ratio grows with size.

#### A4. Is selection acting on groups or on single agents?

**Question.** Of the change in a trait, how much comes from differences
between families and how much from differences inside them?

**Why.** A major transition needs selection between groups to outweigh
selection inside groups. The Price equation splits the two exactly.

**Cost.** One to two hours. No simulation. Needs the drive.

**Steps.**
1. In a family-pool run, take windows of 50 iterations.
2. Trait z of an agent: the share of its stake it keeps on its own node
   (from `allocations`). Fitness w: how many nodes carry its brain's
   descendants 50 iterations later (use the brain-copy tree or follow
   `brain_ids` and `parent_brain_ids`).
3. Between-family part: covariance over families of (family mean fitness,
   family mean trait). Within-family part: the average over families of the
   covariance inside each family. Their sum is the selection part of the
   trait's change.
4. Repeat with two more traits: share given to a child; share staked on kin.

**Report.** The two parts per trait, with their signs, over the run.

**Decision rule.** "Groups are units" if the between-family part is larger in
size than the within-family part for most windows.

#### A5. Does the largest structure grow?

**Question.** How large are territories, how long do they live, and does the
largest keep growing?

**Why.** Condition 5. Unbounded evolution of structure should show as a size
distribution whose tail keeps stretching.

**Cost.** One hour. No simulation.

**Steps.**
1. Territories as in A3, in every 25th frame.
2. Size distribution (log-log), the largest size, and the mean lifetime of a
   territory (follow by family or genotype).
3. The same for 4 windows of the run, to see a trend.
4. Compare family pools with the Experiment 48 worlds.

**Report.** The four distributions per run, the largest size over time.

**Decision rule.** Growth if the largest size and the tail rise from window
to window and are not explained by more agents (divide by the number of
agents).

#### A6. The entropy cycle

**Question.** How far from its most mixed state does a world stay, and how
fast do tokens cycle between concentrated and spread out?

**Why.** A closed world with a fixed number of tokens tends to its most
mixed state, where nothing more happens. Structure needs a steady flow
through it, like a flame. The game concentrates tokens; death and estates
spread them. That cycle is the world's only "energy" flow.

**Cost.** Minutes, from statistics alone.

**Steps.**
1. Per iteration: the entropy of the token distribution (from the frame's
   `tokens`), and the entropy a random sharing of the same tokens among the
   same agents would have. Their difference is the **distance from mixed**.
2. The flow: tokens shared out from the dead each iteration
   (`redistributed`) over all tokens.
3. Both over time, for the three Experiment 48 runs and three family-pool
   runs.

**Report.** Two curves per run.

**Decision rule.** None alone. Use it with C: do worlds with a larger steady
flow and a larger distance from mixed show more beneficial mutations?

#### A7. Does a bigger world evolve faster?

**Question.** Does the time back to the common ancestor grow with world size?

**Why.** If a sweep takes the same number of iterations at 10,000 and 50,000
tokens, a bigger world is only more of the same, and small worlds are enough.
That decides how to spend computer time.

**Cost.** Building brain-copy trees: a few hours. No simulation.

**Steps.**
1. Build trees (`braintree.py`) for s4002 and s4008 at 10,000 tokens
   (Experiment 48), 30,000 (round 3, drive) and 50,000 (Experiment 43,
   local, unfinished runs are fine).
2. Read the common ancestor's age and the effective population from each,
   over the same window of iterations.

**Report.** Three sizes × two worlds: ancestor age, effective population,
agents.

**Decision rule.** "Size does not help" if the ancestor age changes by less
than a factor 1.5 from 10,000 to 50,000 tokens.

### B. Brains alone

#### B1. How big is one mutation, in behaviour?

**Question.** How much does one mutation change what a brain does?

**Why.** Condition 2. If nearly every mutant behaves very differently, small
improvements cannot be found.

**Cost.** Minutes.

**Steps.**
1. Probes as in A1 (300 real input matrices from a kept world).
2. Take 200 brains of that world. For each, make 20 mutants with the
   engine's own operator: copy the brain, set its `cfg` to a copy with
   `mutation_probability` 1.0, call `mutate`.
3. Behaviour vectors as in A1. The **step** of a mutant is the distance to
   its parent's vector. Scale: divide by the typical distance between two
   unrelated brains of the same world.
4. Repeat with smaller operators written for this test: change exactly k
   weights by the engine's noise, for k = 1, 10, 100, 1,000, with no redraw.

**Report.** The distribution of steps for each operator; the share of
mutants that are nearly neutral (step under 1% of the scale) and the share
that are as far as a stranger (step over 50%).

**Decision rule.** The engine's operator is "too coarse" if more than half
of its mutants are further than 25% of the scale from their parent.

#### B2. How many things can evolved brains do?

**Question.** How many independent directions of behaviour does a world's
population use, and what is the common policy in plain words?

**Why.** If all brains do nearly the same simple thing, the plateau is an
equilibrium, and a fast stand-in model (E2) becomes possible.

**Cost.** Minutes.

**Steps.**
1. Behaviour vectors of 1,000 brains of one kept world on the A1 probes.
2. Principal components: how many are needed for 90% of the variance?
3. The same per kept world over time.
4. Describe the mean policy: fit the kept-at-home share and the child share
   as simple functions of the agent's own tokens, its number of links and
   its neighbours' tokens (a small table or a line fit). Check how much of
   the brains' decisions that simple rule explains.

**Report.** Components over time; the simple rule and its fit.

**Decision rule.** "Low-dimensional" if 5 or fewer components give 90% and
the simple rule explains more than 80% of decisions.

### C. Short assays in kept worlds

Assays are lab trials. Each lasts 20 to 50 iterations with mutation off. One
costs one to three minutes at 10,000 tokens.

#### C1. Build the tagged-lineage assay (a tool)

**Question.** Can we measure the fitness of many brains at once in one short
contest?

**Why.** It makes everything in C cheap: one contest measures 100 brains.

**Cost.** One day to build and check.

**Steps.**
1. Add a trial kind `assay` beside `contest` (`gol_trial.py`,
   `gol_strategies.TRIAL_NEEDS`). It takes a kept world, a list of **sites**
   (agent indices) and, for each site, a brain to put there or "keep".
2. Every site gets its own label (a brain id above a marker, as
   `gol_tournament.LABEL_B` does). Mutation is off, so every copy keeps its
   label.
3. Play T iterations. After each, count the nodes that carry each label.
   The result: one series of counts per site.
4. A site's **fitness** is its count at T (0 if it died out).
5. Checks, each a test in `tests/`:
   - sites that keep their own brain, labelled only: the world plays
     exactly as without labels (same token totals each iteration);
   - a world that dies out gives no result (`gol_tournament.died_out`);
   - known signal: plant 100 brains of Experiment 48's iteration 0 into the
     world of iteration 5,000 beside 100 labelled residents. The founders
     must do clearly worse (the contests gave 1.0).
6. Do not use worlds that mix brains at birth (s4002).

**Report.** The tests, and the founders-against-residents check with its
numbers.

**Decision rule.** The tool is ready when the known signal shows with 100
sites in one contest.

#### C2. Are any mutations good? (the distribution of fitness effects)

**Question.** Of the engine's mutants, what share does better than its
parent? And of smaller mutants?

**Why.** Conditions 1 and 2 in one measurement.

**Cost.** About 3 hours of lab time for 3,000 mutants.

**Steps.**
1. World: `B1-10000-085a32-s4801` at 15,000. Also at 5,000 for comparison.
2. One contest: choose 200 sites at random. At 100, replace the brain by a
   mutant of itself. The other 100 keep their brain: the controls.
3. Operators: the engine's own; and k = 1, 10, 100 weights (from B1). Ten
   contests each, with different seeds.
4. Compare the fitness of mutant sites and control sites in the same
   contest: the difference of means, and the share of mutant sites above
   the controls' 90th percentile. Test with a permutation of the labels.
5. Take the 20 best mutants of each operator and test each again, alone, in
   10 new contests with 20 sites each. The first round overstates them.

**Report.** Per operator: share of harmful, neutral and helpful mutants, and
how many of the 20 best held up.

**Decision rule.**
- No operator gives mutants that hold up: evolution here is **at a top**.
  Go to C3 to check with a stronger search.
- Small operators give some and the engine's gives none: **the search is
  the limit**. This is the main finding; go to section 5.
- The engine's gives some: selection is too weak to keep them; go to A7.

#### C3. Search directly for a better brain (the oracle)

**Question.** Can a directed search find a brain that beats the residents?

**Why.** Evolution in the world is a slow search. A direct search does in an
hour what the world may need 100,000 iterations for. If even this finds
nothing, the game is at its top under these rules.

**Cost.** Half a day.

**Steps.**
1. Start from a resident brain of the kept world at 15,000.
2. One generation: make 100 variants (small steps, k = 10 weights), plant
   each at 10 sites in one assay with 100 control sites, T = 30. Three
   contests in parallel give 300 variants.
3. Keep the best 10 by fitness. Make the next generation from them.
4. Thirty generations. Every fifth, test the current best against the
   starting brain in 10 fresh contests.
5. At the end, a full check: the best brain against the residents in 20
   mirrored contests (`gol_tournament.contest`, with the best brain dealt to
   half the agents).
6. Also try starting from the hand-written raider's weights
   (`raider_weights`), if the world's brain shape allows.

**Report.** Fitness of the best brain by generation; the final check.

**Decision rule.** "Better brains exist" if the final check is above 0.5
with its whole band. Then describe what the found brain does differently
(B2's simple rule on it).

#### C4. Climb the ladder by hand (best response after best response)

**Question.** If we let each better brain take over and search again, does it
go on, stop, or go round?

**Why.** Condition 3. This is thousands of years of evolution done by
directed steps.

**Cost.** One day for about five rungs. Only after C3 succeeds.

**Steps.**
1. Rung 0: the residents. Rung 1: the brain C3 found.
2. Fill the kept world with rung 1 (every agent gets it), let it settle 50
   iterations with mutation off, keep that world.
3. Run C3 in that world to find rung 2. Repeat.
4. After each new rung, play it against **every** earlier rung (10 mirrored
   contests each). Fill in the table of all against all.

**Report.** The table; the number of rungs that beat all earlier ones.

**Decision rule.** Open if each rung beats all earlier ones for five rungs
and the search is not getting harder. A circle if a rung loses to one two
steps back. A top if the search fails.

#### C5. Did brains keep getting better under family pools?

**Question.** In the three family-pool runs, do later brains beat earlier
ones after iteration 5,000?

**Why.** They are the only worlds with strong groups that lived 20,000
iterations.

**Cost.** About 6 hours of lab time. Needs the drive.

**Steps.**
1. Eras 2,500, 5,000, 10,000, 15,000, 20,000. Pairs: each against the one
   before, and each against 5,000.
2. Four coins each, mirrored, own deal for every coin (as the readout does
   since 22 November).
3. A contest in a family world keeps the eras' families apart. That gives
   the era with larger families larger pools. So report family sizes of the
   two eras beside every score.

**Report.** Scores with bands, pooled over the three runs.

**Decision rule.** "Still improving" if the pooled score of later against
the era before, over the pairs after 5,000, is above 0.5 with its band.

### D. Thinking as an engineer

#### D1. The rules as physics: one sheet

**Question.** What are the conservation and movement laws of tokens, and what
do they allow a group to do?

**Why.** Everything else in D rests on it, and it costs only thought.

**Cost.** Half a day of reading the engine and writing. Check every law on
recorded frames.

**Steps.** Write each law as one sentence and a formula, then check it on
100 recorded game frames (it must hold exactly; if not, the law is wrong).
Start from these, which were read from the engine on 23 November:

1. **Tokens are conserved.**
2. **A token moves one link per game**: staking on a neighbour's node puts
   the token on that node.
3. **A node keeps every token staked on it**, whoever wins it.
4. **Only neighbours can stake on a node.** So a node whose neighbours are
   all kin cannot be attacked: a core is safe by construction. Check the
   exception: where the rules let a stranger make a link from afar
   (`link_search`, with `link_consent` = forced), a core node can get a
   foreign neighbour.
5. **The contest sees single stakes, not sums** (`contest` = largest): many
   small kin stakes do not defend, and they go to the attacker if it wins.
   Under `lottery` the chance is in proportion to the stake, so the stakes
   of clones add up by themselves. Under `pool` they add by rule.
6. **A failed attack feeds the defender**: the attacker's stake stays on the
   node.
7. **Upkeep**: a node with no stake on it starves; a link with no stake
   across it is cut; a piece cut off from the largest piece dies.
8. **What an attacker sees** is the tokens a node holds now, and six
   quantiles of its neighbours' tokens, not the stakes about to arrive.
9. **Estates**: how the dead's tokens are shared (`redistribution`).

Then derive:
- **The defence law.** A border node b is safe if the largest single stake
  on it is at least the largest stake an outside neighbour can make. A group
  with W tokens, N nodes and B border nodes can put about (W − N) / B on
  each border node.
- **The surface law.** If B grows more slowly than N (as a surface does
  against a volume), defence per border node grows with size. Compute B
  against N for real territories (from A3).
- **The delay.** Tokens need as many games as links to cross a territory.
  How large can it be before help comes too late?
- **Income per head.** With estates shared equally, each living node earns
  the same. Compare with what a raid earns (A2).
- **Poverty.** If income per head dominates, the best body is many nodes
  with one token each. State what that does to the game (with 1 or 2 tokens
  there is almost nothing to decide).

**Report.** The sheet, each law marked "holds exactly on 100 frames" or
corrected.

**Decision rule.** None; it feeds D2 to D5.

#### D2. Exact small games

**Question.** In the smallest cases, does a team do better per token than
loners, and does that grow with team size?

**Why.** Condition 4, answered exactly, in seconds.

**Cost.** One day to write and run.

**Steps.**
1. A small game: a few nodes (a pair, a triangle, a line of 3 to 6, a
   triangle with one outside neighbour), a few tokens each (1 to 6), one
   game phase. Each player picks how to split its tokens over itself and its
   neighbours.
2. Two sides: a **team** (all its members play one joint plan) against an
   **outsider**, or two teams. The payoff of a side: tokens on the nodes it
   holds after the phase (and the number of such nodes).
3. List all joint plans of each side. Fill the payoff table from the
   engine's own rules (`_resolve_winner`, `_lottery`, `_pooled`).
4. Solve the zero-sum game. If no solver is installed, use regret matching:
   each side keeps, for every plan, how much better it would have done; it
   plays plans in proportion to the positive regrets; the average play
   converges. Run 100,000 rounds.
5. Do this for each contest rule: largest, lottery, largest with
   revolutions, family pools.
6. For each: the value per token of a team of n = 1, 2, 3, 4, 5, 6 against
   an outsider with as many tokens as the whole team.

**Report.** A table: rule × team size → value per token. And for each rule,
the best team plan in words (who stakes where).

**Decision rule.** "Size pays" under a rule if the value per token rises
with n. A rule where it falls or stays flat cannot carry larger structures.

#### D3. How much better could groups have played? (the coordination gap)

**Question.** With perfect internal routing of their own tokens, how many of
the nodes that real territories lost could they have kept?

**Why.** If the gap is large, organisation would pay and the question is
whether brains can find it. If it is small, the rules leave nothing to gain.

**Cost.** Two hours. No simulation.

**Steps.**
1. Territories and border nodes as in A3, in 200 game frames.
2. For each border node: the **threat** is the largest stake an outsider
   actually put on it in that frame.
3. The planner's rule: every node gets 1 token; then border nodes get
   tokens up to their threat plus 1, from tokens held within one link.
4. Count the border nodes that were lost in the record but would have held
   under the planner. Count how many tokens the planner had left over.

**Report.** Share of lost border nodes that were avoidable, by territory
size.

**Decision rule.** "A large gap" if more than a third of the losses were
avoidable.

#### D4. Build an organism by hand

**Question.** Does a hand-built group with a shell and a core take hold in an
evolved world, and does a larger one do better?

**Why.** The author's idea, tested directly. It also shows evolution what to
find, as the raider did.

**Cost.** One day. Uses `invasion` trials (minutes each).

**What exists.** `gol_strategies.py` has rules that know their own kind by a
password in their messages, and a way to plant a patch. Earlier lessons: a
strategy must have children as the world's agents do, or it dies with its
founders; and late worlds were too poor for raids.

**Steps.**
1. Write the strategy `cell`:
   - it knows kin by the password;
   - a member with a non-kin neighbour is **shell**; it says so in its
     message to kin, one iteration later;
   - a **core** member keeps 1 token and stakes the rest on shell
     neighbours, or towards them (on the neighbour that reports the
     shortest way to the shell);
   - a **shell** member stakes on itself, and has a child towards the
     outside when it holds more than twice its richest outside neighbour;
   - every member keeps each kin link alive with 1 token.
2. Check which engine settings it needs. A core member must know whether
   **any** neighbour is foreign; a brain reads one neighbour at a time, and
   only `self_message` = largest carries that across. Note it.
3. Plant patches of 10, 30, 100 and 300 in Experiment 48 kept worlds, three
   places each, T = 100.
4. Beside each, plant `breeding-turtle` and `breeding-raider` of the same
   size as controls.
5. Then two variants: `bait` (the shell shows few tokens and gets its
   defence from the core in the same game) and `anti-cell` (a raider that
   reads the neighbours' quantiles and avoids baits).

**Report.** Share of the world each patch holds over time, by size.

**Decision rule.** "Organisms work here" if the cell outgrows both controls
and larger patches do better per member than smaller ones.

#### D5. Is there a top? (the scaling argument)

**Question.** From D1 to D4: under which rules does the advantage of size
never end, and what limits it?

**Why.** This is the conceptual answer to "would it go on for ever", without
simulating for ever.

**Cost.** Thought, after D1 to D4.

**Steps.** Write, with the numbers from D1 to D4:
1. the gain from size (the surface law);
2. each cost that grows with size: the delay; mutants inside the body that
   keep tokens for themselves (use A4); the need to stay joined to the
   largest piece; poverty;
3. for each rule set, the size at which cost meets gain, or "none";
4. what a body needs to pass each limit (for example, lines of rich nodes
   that carry tokens fast: an inner structure).

**Report.** One page: "under rules X the ladder of sizes has no top because
…", or "it tops out at about n because …".

**Decision rule.** The rule set with no top, if there is one, is the
candidate for the one long run.

### E. Other ways to simulate

Only after A to D.

#### E1. A faster engine

**Question.** Where does the time go, and can brain passes be done in
batches?

**Steps.**
1. Profile 20 iterations replayed from a kept world (through a lab trial).
2. If most time is in single brain passes and in building inputs: group
   agents by number of neighbours and pass each group as one stack of
   matrices.
3. The fast engine is a new strain. Check it against the old one as
   Experiment 43 did: the same statistics within the noise between seeds.

**Decision rule.** Worth it if the profile promises 10 times or more.

#### E2. A stand-in world

Only if B2 says behaviour is low-dimensional.

**Steps.**
1. Replace brains by B2's simple rule with a few inherited numbers.
2. Write the two phases with arrays only (no loop over agents).
3. Check it against Experiment 43's and 48's statistics.
4. Use it to ask what a real world cannot be asked in a day: millions of
   agents, hundreds of thousands of iterations, which rules give growing
   territories.

**Decision rule.** Trust it only for questions where it matches the real
world's record first.

#### E3. Many short runs in a chain

**Steps.** Ten worlds for 2,000 iterations (hours); the winners by contest
seed the next ten. Each link is a whole experiment under a day with its own
result (winners against the link before).

**Decision rule.** Only if C2 finds that good mutations exist but are lost.

---

## 4. Order of work

| Wave | Cards | Total cost | Gives |
|---|---|---|---|
| 1 | D1, B1, B2, A1, A6 | half a day, no lab | the laws; how coarse mutation is; stopped or wandering |
| 2 | C1, then C2 | one day building, 3 hours of lab | **the main answer**: game limit or search limit |
| 2, beside it | A2, A3, A5, D2 | one day, no lab | what the rules reward; whether size pays |
| 3 | C3, then C4; D3, D4; A4, C5 (drive) | one day each | the ladder; organisms |
| 4 | D5; then E1 to E3 if called for | thought | the candidate for one long run |

Cards in one wave do not depend on each other and can be given to different
helpers. A helper needs: this file, the card's name, and nothing else.

---

## 5. What each outcome would mean

- **Search is the limit** (C2: small steps help, the engine's do not). Change
  how brains vary: each brain carries its own step size, inherited and
  itself mutated, so no number is set by hand. Test it with C2 and C3
  again, not with a long run.
- **The game is at a top** (C2 and C3 find nothing). Longer runs are
  pointless under these rules. D2 says which contest rule lets size pay;
  move there.
- **A circle** (C4). The game has depth but no direction. Look for rules
  where D2's value rises with size.
- **Size pays and organisms work** (D2, D4), but evolved worlds show no
  shell and core (A3). Then evolution cannot find the organism: back to the
  search, or seed worlds with built organisms and watch what evolution makes
  of them.
- **Worlds pay per head** (A2, D1). Poverty is built into how estates are
  shared. Poor agents have no choices. Study `redistribution` first.
- **Everything holds.** Then, and only then, one long run of the candidate
  rule set, with the organisms seeded, is worth its days.
