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
2. **Revolutions break the total order.** With them off, a node goes to
   whoever staked most. That is a total order on stake size, it holds in every
   subset of the contestants, and total orders admit dominant strategies.

   With them on, the winner is not a function of pairwise strength at all.
   **Demonstrated against the engine's own `_resolve_winner`**: three stakers
   H (100, none flagged as revolt), M (60, all flagged), S (50, all flagged).
   Pairwise H beats M and H beats S; add S to the H-versus-M contest and **M
   wins**, by revolution, while S itself wins nothing. A revolution cannot fire
   in a two-way contest at all — it would need one agent's revolt to exceed the
   leader's whole stake — so this is irreducibly a coalition effect.

   In choice-theoretic terms this violates independence of irrelevant
   alternatives: who wins depends on who else is present. That is what removes
   the ordering, and it is the precondition for cyclic dominance among evolved
   policies — but a cycle among *policies* is a stronger claim than this
   demonstration supports, and establishing one is what E4 is for.
3. **Conquest replicates the winner's policy into the loser's node.** Selection
   is not just differential survival, it is direct strategy transmission
   between neighbours — closer to cultural transmission than to inheritance.

**Conjecture C1.** Revolutions are the mechanic that makes the strategy space
non-transitive, and non-transitivity is a *necessary* condition for the
strategic dynamics to be open-ended in this system. With
`allow_revolutions = False` the population should converge to a narrow band of
policies and stay there; with it on it should keep turning over.

The demonstration above establishes the *mechanism* — the order is gone. It
does not establish that evolved policies actually cycle, only that nothing in
the rules forbids it.

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

### 3.1 A retracted argument, and what survives it

An earlier draft of this document argued that strict token conservation
**precludes** unbounded complexity, on the grounds that "complexity has a cost
and no channel to pay for itself". That argument is wrong, in two separate
ways, and the correction matters enough to record rather than quietly delete.

**It is wrong about this system.** Complexity here has no cost. Every brain has
the same architecture — the layer sizes are a global constant — so a genome is
a fixed-length vector of weights and there is nothing for an agent to spend on
being more complicated. Whatever bounds this system, it is not the price of
complexity.

**It is wrong about zero-sum games generally.** Van Valen's Red Queen (1973) is
a zero-sum formulation of evolution and is the standard account of *sustained*
adaptation: in a world where one lineage's gain is another's loss, everything
must keep evolving to stand still. Host–parasite arms races, Sims' (1994)
competitive coevolution, and every deep two-player game are zero-sum and
strategically unbounded. Zero-sum forbids nothing.

**The specific counter-argument that defeats it.** Organisation does not need
to *produce* surplus to pay for itself; it only needs to take a larger *share*.
A structure of many agents that holds tokens better than the same agents
separately is favoured, and nothing in conservation forbids that structure from
being larger, or from there being more of them. Empires are zero-sum and scale
anyway. Conservation bounds the *total*, not the *organisation of it* — and the
total is a parameter.

So the honest position is the weaker one: **conservation bounds the population
at `T`, and therefore bounds how many lineages and how much structure can
coexist. It bounds nothing else.** Raising `T` raises that ceiling
proportionally, which makes "make the world big enough that it never settles"
a real answer rather than an evasion.

### 3.1a What still gives me pause, stated better

Three arguments that are not the retracted one.

**The genome cannot grow.** Layer sizes are fixed for a run, so a new adaptation
must be a re-encoding of existing weights and never a new structure. Tierra and
Avida both let genome length change, and unbounded growth in *encoded*
complexity is normally what "class 3" ends up meaning. This is a real limit and
it is a parameter away from being lifted — variable hidden layers, or a
duplication operator.

**The state space does not expand.** Banzhaf et al. (2016) separate novelty
*within* a fixed space of possibilities from novelty that *enlarges* the space.
Here the space is settled at configuration time: a graph, an integer per node,
weights of a fixed shape. No rule can bring a new *kind* of thing into
existence. Whether that is fatal is genuinely contested — it is close to the
central open question of the field — but it is the strongest available argument
against, and it has nothing to do with conservation.

**Locality may cap coherent structure.** *New, and the most interesting of the
three.* Information moves one hop per phase. A structure of diameter `D` needs
`D` phases for one side to learn anything about the other, so a structure can
only behave as a unit if it is smaller than the distance information travels in
the time the world stays put. That predicts a **maximum coherent size set by the
ratio of information speed to the rate of change** — which is a physical-feeling
limit of the same kind as the one that stops organisms signalling across
arbitrary distances, and it would bound structure even with `T` infinite.

**Hypothesis H1 (revised).** Conservation is not the binding constraint. What
bounds structure is the coherence limit: the largest lasting structure has a
diameter comparable to the number of phases over which the neighbourhood it
sits in stays recognisable. *Falsified if* the largest persistent clonal patch
keeps growing with `T` at fixed churn.

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

## 3.4 Castles: making the intuition measurable

The criterion that prompted the retraction above was put plainly: *if many
nodes together form a kind of castle, and the castle can get larger and there
can be more of them, that is already open-ended evolution.* That is a
structural criterion rather than a genetic one, it is a reasonable thing to
mean by the term, and — unlike most definitions in this literature — it can be
measured directly.

**Definition.** A **castle** at iteration `t` is a connected set of agents `C`
such that

1. **it holds together** — the token flow along edges inside `C` exceeds the
   flow across its boundary, by some ratio `ρ > 1`;
2. **it persists** — a set overlapping `C` in at least half its members
   satisfies (1) at `t + 1`.

That is all. An earlier draft also required every member to share a lineage,
and that was a mistake: it builds the answer into the question. A clonal
requirement guarantees relatedness ≈ 1, which guarantees Hamilton's rule is
satisfied, which is assuming the mechanism one is trying to detect — and it
makes the more interesting object invisible by construction. Both kinds of
transition occur in biology: multicellularity is clonal, eukaryogenesis was
symbiotic.

**Lineage composition is therefore a measured property of a castle, not a
condition on it.** Every castle gets a *mixedness*: the number of distinct
clades among its members, and the share held by the largest. `ρ ≈ 1` clonal
patches and mixed mutualisms are then two findings rather than one definition
and one blind spot.

**Symbiosis is already available in the rules**, which is worth stating because
it means this is not a hypothetical. **Demonstrated against `_resolve_winner`**:
an agent defending its own node with a self-stake of 100, and a neighbour of
any lineage staking `x` on it. At `x < 100` the neighbour's tokens arrive, the
defender keeps its node *and its brain*, and the giver is poorer — a gift.
Only at `x >` the self-stake does the stake become a conquest and overwrite the
defender. (At exactly equal it is a coin toss, as ties are.) Flagging the gift
as revolt changes nothing, because a revolution needs a coalition and one
revolter is not one. So unrelated agents can already feed each other without
taking each other over, and no new mechanic is needed for a symbiotic castle to
be possible — only for it to be *worth* it.

**The numbers to watch**, each as a function of `T`:

- **size** — members of the largest persistent castle;
- **count** — how many exist at once;
- **lifetime** — how long the longest-lived one survives;
- **mixedness** — clades inside it, and whether that rises or falls with size.

**Hypothesis H4 (castles).** Castle size and count grow without bound as `T`
grows; castle *lifetime* does not, being capped by the coherence limit of
§3.1a. If all three grow, the structural criterion for open-endedness is met.
If size grows but lifetime saturates, castles are real but transient — an
ecology of empires rather than a transition in individuality.

**Hypothesis H5 (symbiosis).** Mixed castles occur, and the lineages in them
are interdependent rather than merely adjacent. The test is a perturbation: cut
one clade out of a castle and compare what remains against the same cut made on
a size-matched random subset. If the remainder loses more than the control, the
castle was doing something the parts were not.

Detecting mixed castles at all would be the more interesting result, because
conquest is a homogenising force — winning a node overwrites its brain — so a
mixed castle has to be actively maintained against a rule that is constantly
trying to make it clonal.

The interesting case is the third: size and count growing with `T` while
lifetime is flat would mean the world gets *wider* without getting *deeper*,
which is the sharpest version of the disagreement this section is about and is
worth settling before anything else in §6.

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

It looked as though the information was nonetheless recorded — each frame
carries `brain_ids` and `parent_brain_ids`. **It is not.** Building
`research/phylogeny.py` established that the frames are insufficient, and the
reason is specific: winning a node copies the winner's brain, and the copy is
then mutated at the end of the phase. Only the second of those two ids is ever
written to a frame, so the copy — which is the *link in the chain* — is
invisible.

**Measured: 49% of the brain ids a run creates never appear in any frame.**
Reconstructing from frames alone therefore breaks every chain within a step or
two and hands back one clade per agent, which is the very artefact the tool
exists to avoid. On a 50-iteration run it reported 524 clades in a population
of 593, and a dominant-clade turnover rate of 0.68 — all noise.

Two consequences:

**For runs made in process**, `research/phylogeny.py --simulate` traces every
link as the engine makes it, and the reconstruction is then exact (`unrooted:
0`). The research in §6 can proceed today on traced runs.

**For runs recorded to disk**, this has now been fixed in the engine.
`brain_id` names a *genotype*: a copy keeps its source's id, because it is the
same genotype, and only mutation allocates a new one. Ids that never reach a
frame fell from **49% to 4.5%**, and reconstruction from frames alone now gives
answers identical to the traced version — 1 clade, 100% top share, 4 turnovers,
coalescence lag 18, against 5 unrooted stragglers instead of thousands.
`distinctBrains` counts genotypes for the first time, so `SERIES_VERSION` went
to 16.

Runs recorded **before** that change cannot have their ancestry rebuilt, and
nothing can recover it: the missing ids were never written down. The Lineage
view detects them by their root share and says so rather than drawing a picture
of nothing.

**What is still true.** `brain_id` remains a genotype version, not a clade
label — it changes on every mutation, and every agent mutates every iteration.
Clades are *derived* from the forest rather than stored on an agent, because a
clade is a choice of anchor and there is no single right one: founder clades
collapse to one within about seventeen iterations, so a stored founder label
would read "1" forever, and the informative measure is a sliding window, which
needs the forest anyway.

`distinctLineages` never counted lineages — it counts distinct *parents* among
the living, one hop back. **Measured** at the end of a 25-iteration run: 176
agents, 146 genotypes, **105 distinct parents, and 1 actual founder clade**. It
is now called `distinctParents`, which is what it is.

A real family count came with it. `cladesInWindow` counts how many separate
families the living divide into, a family being everything descended from one
agent alive `CLADE_WINDOW` iterations ago (8). It is computed in the series
builder rather than in `frame_stats`, because it needs history and a frame
statistic by definition does not — which also keeps the Python and JavaScript
implementations in parity, since a single frame cannot produce it on either
side.

It uses a **rolling** window of ancestry, not the whole forest, so it stays
bounded on a long run, and it is left **absent** rather than guessed whenever
the chain is broken: `export_every > 1`, or a run long enough that the series
had to be sampled down. Ancestry is a chain and a chain cannot be sampled.

**Measured** over 30 iterations of a 40-founder world: agents ranged 56 → 205
and genotypes 48 → 168 while families stayed between **20 and 28**. So it is
not tracking the population, which is exactly what the statistic it replaces
was doing.

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

**E7 — Castles against scale.** `T` over as wide a range as memory allows, with
the brain held fixed, measuring castle size, count, lifetime and mixedness
(§3.4) and the time for the founding lineages to collapse to one. *Tests H4, and settles
whether "big enough that it never finishes" is a real answer.* This is the
experiment the disagreement in §3.1 turns on, and it should come first.

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
| **One founder sweep, then none** | 50 founding clades → 1 by iteration ~17; afterwards the coalescence lag grows at one per iteration | see below |
| Mean degree declines | 3.9 → 3.2 over 45 iterations, 5 seeds, seed `k = 6` | the graph erodes; long-run unknown |
| Frames cannot carry ancestry | 49% of brain ids never reached a frame | **fixed**: a copy keeps its genotype's id, now 4.5% |
| No lineage identity | 502 agents, 502 distinct `brain_id`s | blocks all lineage statistics |
| Extinction is common | 9/20 binary runs dead inside 25 iterations | affects every downstream design |
| Encoding resolution | binary input ladder: 15 → 36 levels after the split | no measured survival effect at n = 20 |
| Binary decisions are coarse | ~9% of paired heads exactly tied; 17 distinct staking scores vs 936 for float | output layer width, not input encoding |
| Float input balance | noise is 35% of first-layer variance from 5 of 54 inputs | unnormalised inputs; nobody chose this |

### 8.1 The first real lineage result

From a traced 40-iteration run (seed 5, 2500 tokens, 50 founders), clades named
by founder:

- The 50 founding lineages collapse to **one** by about iteration 17. One
  founder's descendants hold 100% of the population thereafter.
- After that sweep the **coalescence lag grows by one per iteration** — 16, 18,
  20, … 28 — which means the population's most recent common ancestor is a
  *fixed* brain from around iteration 10 and **nothing has swept since**.
- Dominant-clade turnover over the whole run: 3 changes in 80 frames.

So the early dynamics are a hard selective sweep and the later dynamics are
not. Under a sliding five-iteration window, recent ancestry is much livelier —
62 clades, top share 6.7%, 50 turnovers — so sub-lineages do keep replacing one
another locally without any of them fixing globally.

That combination, a single early sweep followed by no fixation, is exactly the
signature that needs a neutral shadow (§5.3) to interpret: it is equally
consistent with strong frequency-dependent selection maintaining diversity and
with selection having simply stopped mattering. Distinguishing those two is
what §6's E2 is for. Forty iterations is also far too short to claim either.

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
