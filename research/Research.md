# Graph of Life — research notes

Working notes towards a paper. Not a paper: this is the design, the
definitions, the hypotheses, the protocols, and an honest account of what is
and is not yet measurable. Written 2026-09-04.

Everything labelled **measured** was run and the numbers are given. Everything
else is conjecture, and marked as such. The distinction matters more than
usual here, because the interesting claims are exactly the ones that are easy
to assert and hard to establish.

---

## 1. The system, stated precisely

A paper needs this in one place, in a form somebody could reimplement.

**State.** An undirected graph `G_t = (V_t, E_t)`. Each agent `u ∈ V_t` holds

- an integer token count `τ(u) ≥ 0`,
- a policy `π_u` — a feed-forward network, never trained, only copied and
  perturbed,
- an inbox of message vectors from its neighbours, one per neighbour and one
  from itself.

**Conservation.** `Σ_u τ(u) = T` for all `t`, unless
`tokens_created_per_phase > 0`. Tokens are neither created nor destroyed by any
rule except the global injection and the resurrection of an extinct world.
This is the system's energy analogue.

**Locality.** An agent observes and acts only on `N[u] = {u} ∪ N(u)`. Nothing
in one iteration moves information further than one hop. This is the system's
speed-of-light limit: a perturbation at `u` cannot influence an agent at graph
distance `d` in fewer than `d` phases.

**One iteration is two phases.**

*Reproduction.* Every solvent agent observes `N[u]` in one pass, writes a
message to each target, and decides a fraction of its pile to spend on a child.
The child gets exactly those tokens, a mutated copy of the parent's policy, and
a set of links chosen by the parent from `N[u]`. The parent may additionally
*hand over* one of its own edges: the edge moves rather than being copied.
Then cleanup.

*The game.* Every agent observes once, writes messages, and stakes its **entire**
pile across `N[u]` — either spread in proportion to a score or all on one
target, its own choice. A node's new balance is everything staked on it.
`resolve` decides who takes each node: the largest staker (the *hegemon*)
unless a coalition of smaller stakers who flagged part of their stake as
*revolt* outweighs everyone above them plus the hegemon, in which case the node
goes to the strongest staker in the rung that tipped it. The winner's policy is
**copied into the node**. Links that carried no tokens are cut. Then cleanup,
then every surviving policy mutates.

*Cleanup*, after both phases: agents holding nothing die; then everything
outside the largest connected component dies; the dead's tokens are scattered
uniformly over the survivors.

**Selection.** There is no fitness function. A policy spreads by two channels
only: reproduction (a child carries a mutated copy) and conquest (a winner's
policy overwrites the loser's). Nothing optimises anything.

---

## 2. Is there an optimal strategy?

### 2.1 What theory already says

The staking sub-game is a **Colonel Blotto game** on the neighbourhood: a fixed
budget divided across several contested fronts, winner-take-all per front.
Blotto games are the canonical example of a game with **no pure-strategy Nash
equilibrium** — the equilibrium is a distribution over allocations (Borel 1921;
Roberson 2006). If the local game inherits that property, then "the optimal
strategy" does not exist as a pure policy, and a population cannot converge to
one without becoming invadeable.

Three features push this system further from equilibrium than textbook Blotto:

1. **The budget is endogenous.** An agent's stake next round is whatever was
   staked *on* it this round. Budget and payoff are the same quantity, so the
   game is a dynamical system rather than a one-shot contest.
2. **Revolutions are an explicit non-transitivity generator.** With them off,
   a node goes to whoever staked most — a total order, and total orders admit
   dominant strategies. With them on, a coalition of small stakers beats a
   large one, which is precisely a rock-paper-scissors structure: *big beats
   medium, medium beats small, coalition of small beats big.*
3. **Conquest replicates the winner's policy into the loser's node.** Selection
   is not just differential survival, it is direct strategy transmission
   between neighbours — closer to cultural transmission than to inheritance.

**Conjecture C1.** Revolutions are the mechanic that makes the strategy space
non-transitive, and non-transitivity is a *necessary* condition for the
strategic dynamics to be open-ended in this system. With
`allow_revolutions = False` the population should converge to a narrow band of
policies and stay there; with it on it should keep turning over.

This is a clean, cheap ablation and it should be experiment number one.

### 2.2 Groups

Conquest makes a locally dominant policy overwrite its neighbours', so a winning
lineage produces a **clonal patch**: a connected region of near-identical
policies. Within a patch, relatedness is ≈ 1, which is the strongest possible
condition for Hamilton's rule and for group-level adaptation. Between patches,
competition happens at boundaries.

This is the structure in which a **transition in individuality** (Maynard Smith
& Szathmáry 1995) could occur: a patch that coordinates its staking — for
example, its interior agents feeding tokens to its boundary agents — would
behave as a single competitive unit. Whether that happens is an empirical
question with a specific signature, and it is measurable (see §5.4).

**Conjecture C2.** The system's long-run dynamics are between-patch, not
between-agent, and the effective unit of selection drifts upward over time.

---

## 3. Can open-ended evolution happen here?

### 3.1 The honest prior

Probably **not without at least one addition**, and the reason is structural
rather than a matter of parameters.

Tokens are strictly conserved and every interaction is appropriative. There is
**no positive-sum interaction available anywhere in the rule set**. Two agents
that coordinate perfectly cannot produce more than two agents that ignore each
other; they can only redistribute. Every major transition in biological
evolution — chromosomes, eukaryotes, multicellularity, eusociality — rests on
interactions where the whole exceeds the sum, usually through division of
labour or metabolic complementarity.

A strictly zero-sum world with a fixed resource can still show unbounded
*strategic* churn (arms races, Red Queen dynamics) but is a poor candidate for
unbounded *complexity growth*, because complexity has a cost and no channel to
pay for itself.

This is the single most important design question, and it is stated as a
hypothesis rather than a conclusion because it might be wrong: self-generated
niches can appear in strictly competitive worlds (Chromaria; Soros & Stanley
2014), and the graph topology is itself a resource that agents shape.

**Hypothesis H1 (zero-sum ceiling).** Under strict conservation and purely
appropriative interaction, evolutionary activity is *bounded* (Bedau class 2):
novelty continues but the diversity of persistently-used adaptations saturates.
Adding an endogenous positive-sum channel moves it to unbounded (class 3).

### 3.2 The scale question

**Conjecture C3 (mixing length).** Open-endedness requires the world to be
large compared to its mixing length. In a well-mixed population one strategy
sweeps and the system is effectively one agent; in a spatially extended one,
many strategies coexist and boundaries keep generating novelty (Nowak & May
1992). The relevant ratio is diameter to the distance information travels in
the time a sweep takes.

At `T = 10^6` tokens the seed graph is 10,000 agents and the carrying capacity
scales with `T`, since an agent needs at least one token to live. Whether that
is *interestingly* bigger depends on whether the evolved graph stays
small-world (`D ~ log N`, everything mixes, one strategy can sweep) or becomes
lattice-like (`D ~ N^{1/d}`, many quasi-independent domains). The box-counting
dimension already implemented is the right instrument for exactly this.

**Prediction P3.** If `D` grows sub-logarithmically with `N`, larger worlds
will *not* be qualitatively richer — they will be many copies of the same
dynamics. If `D` grows as a power of `N`, they will be.

---

## 4. What is missing from the model

Ordered by how much I expect each to matter.

### 4.1 No positive-sum interaction — *the big one*

There is no rule by which cooperation produces surplus. `tokens_created_per_phase`
injects tokens globally and unconditionally, which is weather, not production.

**Proposed mechanic — mutual flow yields.** An edge that carries tokens in
*both* directions in the same game phase yields a small number of new tokens,
split between its ends. This is minimal, local, conserves the spirit of the
design (nothing is created except by an interaction), and converts the game
from zero-sum to variable-sum. It also creates an immediate defection problem
— staking on a neighbour is a cost — which is the substrate for everything
interesting in evolutionary game theory.

The theory here is well developed: Nowak's five rules (2006), and specifically
**network reciprocity**, which predicts cooperation is favoured when the
benefit-to-cost ratio exceeds the mean degree, `b/c > k` (Ohtsuki, Hauert,
Lieberman & Nowak 2006). This system has an evolving `k` (**measured**: mean
degree falls from ~3.9 to ~3.2 over 45 iterations), so it could pass through
that threshold from either side — and the agents partly control `k` themselves,
which makes the threshold an evolvable target rather than a parameter. That is
a genuinely novel setting for that result.

### 4.2 The graph can only erode

Edges are created **only** at birth, from the parent's existing neighbourhood,
and destroyed **every iteration** by pruning every zero-flow edge. There is no
mechanic by which an existing agent forms a new connection.

**Measured**: mean degree falls from ~3.9 to ~3.2 over 45 iterations across
five seeds (seed graph `k = 6`). The graph is thinning. Long-run behaviour is
unmeasured and is a priority: a graph that tends to a tree has a diameter that
grows, a fragile largest component, and mass extinction by disconnection.

**Proposed mechanic — reach.** An agent may spend tokens to create an edge to
a neighbour's neighbour (exactly two hops, preserving locality). Cost paid in
tokens keeps conservation. This makes topology an evolvable trait with a price
rather than a one-shot inheritance, and turns the system into a proper adaptive
network (Gross & Blasius 2008).

### 4.3 No persistent public signal

Messages are private, per-edge, and last one phase. There is no stigmergy — no
mark left on a node or edge that persists, decays, and is readable by whoever
arrives later. Stigmergy is how most biological collectives coordinate without
central control, and it is a very cheap addition: one decaying vector per node.

### 4.4 No identity, no tags

An agent cannot tell *who* it is dealing with beyond the content of a message.
There is no heritable tag, so no tag-based assortment (Riolo, Cohen & Axelrod
2001), no reputation, no partner choice. Adding a small heritable tag vector
that agents can observe would open green-beard and reciprocity routes that are
currently closed.

### 4.5 No environmental heterogeneity

Every node is identical. All niches must be self-generated. This is possible
but hard; giving nodes intrinsic variation (a per-node yield, or a per-node
cost of living) would supply exogenous niches to specialise into.

### 4.6 Death has only two causes

Starvation and disconnection. No ageing, no density dependence, no
catastrophes. Periodic disturbance is a classic driver of diversity (the
intermediate disturbance hypothesis) and is one line of code.

---

## 5. Methods

### 5.1 What already exists

`gol_series.py` computes ~60 per-frame statistics, and they are parity-tested
against a JavaScript reimplementation. Directly relevant: `gini`,
`tokenEntropy`, `degreeEntropy`, `meanDegree`, `transitivity`, `diameter`,
`meanPathLength`, `components`, `dimension` (box-counting, greedy true-ball
covering), the scale-free exponent (MLE with KS-selected `k_min`),
`revolutions`, `revoltShare`, `spreadShare`, `selfAllocationShare`,
`prunedEdges`, `starved`, `orphaned`.

This is a substantial measurement apparatus and most of the structural work is
already done.

### 5.2 What does not exist, and blocks everything in §6

**There is no lineage identity.** `brain_id` is reassigned on *every successful
mutation*, and mutation runs on every agent every iteration. So `brain_id` is a
genotype *version* number, not a clade label.

**Measured**: in a 502-agent population there were **502 distinct
`brain_id`s** — one per agent. The two statistics that look like lineage
measures, `distinctBrains` and `distinctLineages`, are therefore not measuring
lineage diversity; they are close to population size. Any claim about lineage
turnover computed from them today would be an artefact.

The information needed is nonetheless recorded: each frame carries `brain_ids`
and `parent_brain_ids`, so the full phylogenetic forest is reconstructible
offline from a run with `export_every = 1`. **Building that reconstruction is
prerequisite work for §6.** Deliverable: a `research/phylogeny.py` that reads a
run's frames and emits, per iteration, the set of extant clades under a
coalescence depth, with clade abundances.

### 5.3 Definitions to fix before measuring

- **Clade.** The set of agents whose most recent common ancestor is at depth
  ≤ `d` in the `parent_brain_id` forest. `d` is a free parameter and results
  must be shown to be robust across it.
- **Persistent adaptation.** Following the MODES convention (Dolson, Vostinar,
  Wiser & Ofria 2019): a lineage is counted only if it still has descendants
  after a filter interval, which removes the noise of transient mutants. This
  is essential here, where every agent mutates every iteration.
- **Evolutionary activity** (Bedau & Packard 1992; Bedau, Snyder & Packard
  1998). Activity `A_i(t)` = cumulative presence of clade `i`. Report new
  activity, mean activity, and diversity — each **normalised against a neutral
  shadow**, without which none of the three means anything.
- **Neutral shadow.** The same run with selection removed but demography intact:
  replace the conquest winner with a uniformly random staker on that node.
  Population dynamics, token flow and graph dynamics are preserved; only the
  *direction* of selection is destroyed. This is the control every OEE claim
  will be compared against.

### 5.4 Metrics designed for the specific questions

**Non-transitivity index (for C1).** Sample policies from iterations
`t_1 < … < t_m` of one run. Build an arena: a fixed small graph, two policies
seeded on opposite halves, run `k` iterations, record which holds more nodes.
Fill a dominance matrix `W`, then count **intransitive triads** — triples where
`A > B > C > A` — against the expectation under a random-tournament null. A
population converging on an optimum gives a near-transitive matrix; a Red Queen
gives an intransitive one. This is the CIAO methodology of Cliff & Miller
(1995), and it is the sharpest available test of "is there an optimal
strategy".

**Communication use.** Mutual information between an agent's received messages
and its subsequent allocation, against a control where inboxes are shuffled
between agents. If the two are indistinguishable, communication has not
evolved, whatever the message vectors contain. Also worth measuring across the
`message_prepass` ablation, since that option exists precisely to change what
an agent knows when it acts.

**Multilevel selection (for C2).** Partition the graph into communities, then
apply the Price equation to decompose the change in a trait's population mean
into between-community and within-community covariance. A rising between-group
share is the quantitative signature of the unit of selection moving upward.

**Patch structure.** Assortativity of clade identity across edges; the
distribution of clonal patch sizes; the fraction of edges that are
within-clade. Whether the world becomes a mosaic of clones, and at what scale.

**Mixing length (for C3).** Perturb one agent's tokens and measure how the
divergence from an unperturbed twin run spreads with graph distance and time.
Gives the actual information velocity, against the theoretical bound of one hop
per phase.

---

## 6. Experiments

Each is stated as: question, design, response variables, and what would falsify
the hypothesis. All are local runs; disk is cheap, RAM is the binding
constraint (§7).

**E1 — Mechanic ablation.** Full factorial over `allow_revolutions`,
`allow_handover`, `exchange_messages`, `message_prepass`, with ≥ 30 seeds per
cell (the null A/B in §8 is the reason for that number). Responses: survival to
a fixed horizon, evolutionary activity class, non-transitivity index, mean
degree trajectory. *Falsifies C1 if* the non-transitivity index is
indistinguishable between revolutions on and off.

**E2 — Neutral shadow.** Every E1 cell repeated with the random-winner control.
Provides the normaliser for all activity statistics. Without this, E1 measures
nothing.

**E3 — Scale sweep.** `T ∈ {2·10³, 10⁴, 10⁵, 10⁶}` with a *fixed* small binary
brain so that world size and agent complexity are not confounded. Responses:
population, diameter, box dimension, number of coexisting clades, mixing
length. *Tests P3.*

**E4 — Cross-time tournament.** One long run, policies checkpointed every `n`
iterations, all-pairs arena as in §5.4. *The direct test of "is there an
optimal strategy".*

**E5 — Positive-sum extension.** Implement mutual-flow yields (§4.1) as an
option, then repeat E1/E2 with it on. *Tests H1 — the single most important
experiment in this document.*

**E6 — Long run.** One run, as long as patience allows, `export_every = 1`, for
the activity statistics. Everything else is a pilot for this.

---

## 7. Constraints and threats to validity

**RAM, not disk.** Memory is population × policy size. The default float brain
is ~10,000 weights at 8 bytes = 80 KB per agent, so 10,000 agents is ~800 MB of
weights alone. Reaching `T = 10⁶` therefore *requires* small binary brains
(1 byte per weight). **This confounds world size with agent capacity**, and E3
must hold the brain fixed and report that it did.

**Extinction is common.** **Measured**: 9 of 20 binary runs died within 25
iterations. Any statistic conditioned on survival is conditioned on a
non-random subsample. Report extinction as a primary outcome, not as missing
data.

**Small samples lie.** **Measured, and learned the hard way**: a 6-seed
comparison showed 2 extinctions versus 0 and a 56% lift in median population;
the same comparison at 20 seeds showed 9 versus 9 and no difference. Nothing
below ~30 seeds should be reported as an effect.

**The pilot statistics are not what they appear.** See §5.2. Do not use
`distinctBrains` or `distinctLineages` as diversity measures.

**No reference implementation.** Results here cannot be compared against Avida
or Tierra directly. Comparisons must be structural (activity classes) rather
than quantitative.

---

## 8. Preliminary observations

All small, all pilots, none conclusive.

| observation | measurement | status |
|---|---|---|
| Mean degree declines | 3.9 → 3.2 over 45 iterations, 5 seeds, seed `k = 6` | the graph erodes; long-run unknown |
| No lineage identity | 502 agents, 502 distinct `brain_id`s | blocks all lineage statistics |
| Extinction is common | 9/20 binary runs dead inside 25 iterations | affects every downstream design |
| Encoding resolution | binary input ladder: 15 → 36 levels after the split | no measured survival effect at n = 20 |
| Binary decisions are coarse | ~9% of paired heads exactly tied; 17 distinct staking scores vs 936 for float | output layer width, not input encoding |
| Float input balance | noise is 35% of first-layer variance from 5 of 54 inputs | unnormalised inputs; nobody chose this |

---

## 9. Literature to engage with

Listed from working knowledge; every one needs checking against the actual
paper before it is cited.

**Open-ended evolution.** Bedau & Packard (1992), measurement of evolutionary
activity. Bedau, Snyder & Packard (1998), classification of long-term
evolutionary dynamics — the class 1/2/3 scheme this document uses. Taylor et
al. (2016), *Open-Ended Evolution: Perspectives from the OEE Workshop in York*.
Packard et al. (2019), overview and editorial introduction. Banzhaf et al.
(2016), defining and simulating open-ended novelty. Dolson, Vostinar, Wiser &
Ofria (2019), the MODES toolbox — the most directly reusable methodology here.
Soros & Stanley (2014), necessary conditions via Chromaria. Standish (2003), on
what open-ended even means.

**Artificial life systems.** Ray (1991), Tierra. Ofria & Wilke (2004), Avida.
Channon, Geb. Yaeger, Polyworld. These are the comparison class.

**Evolutionary graph theory.** Lieberman, Hauert & Nowak (2005), evolutionary
dynamics on graphs — amplifiers and suppressors of selection. Ohtsuki, Hauert,
Lieberman & Nowak (2006), `b/c > k`. Nowak & May (1992), spatial chaos. Nowak
(2006), five rules. Gross & Blasius (2008), adaptive coevolutionary networks —
the closest existing framing for a graph the agents rewire themselves.

**Game theory.** Borel (1921) and Roberson (2006) on Colonel Blotto and the
absence of pure equilibria. Cliff & Miller (1995) on CIAO plots and measuring
progress in coevolution.

**Selection theory.** Price (1970), the covariance decomposition. Maynard Smith
& Szathmáry (1995), the major transitions. Wilson & Sober on multilevel
selection.

---

## 10. Immediate next steps

1. `research/phylogeny.py` — reconstruct the lineage forest from a run's
   frames. Everything in §6 depends on it and nothing else does.
2. The neutral-shadow control as a config option (random conquest winner).
3. The arena for cross-time tournaments (§5.4). Small, self-contained, and it
   answers the headline question on its own.
4. Then E1 and E2, at ≥ 30 seeds.
5. Mutual-flow yields as an option, then E5.

Everything before step 4 is instrumentation. It is worth resisting the
temptation to run big experiments first: the pilots in §8 already show that the
obvious statistics measure the wrong thing, and a large run analysed with them
would produce a confident, wrong answer.
