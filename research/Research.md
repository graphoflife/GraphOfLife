# Graph of Life — the road to open-ended evolution

## The goal

> **Build open-ended evolution in this substrate, and prove it.**

Not "find out whether it is there". The working assumption is that this
substrate can carry open-ended evolution, that the gap between what it does now
and what it needs to do is a list of mechanics, and that the list is short. This
document is that list, the argument for why the substrate deserves the effort,
the definition we are aiming at, and the measurements that will settle it.

Every mechanic proposed here is **optional and off by default**, so that every
run ever made stays reproducible and comparable. §5 is how an experiment names
the exact algorithm it was run on.

---

## 0. Stance, and the one thing it does not licence

The stance is a decision about where effort goes. We assume the target is
reachable and spend our time on how to reach it, rather than on adjudicating
the current build. When a mechanic fails to produce what it was supposed to,
that is information about which mechanic to try next, not a verdict on the
programme.

The one thing this does not licence is misreporting a number. A programme built
on a measurement that was massaged fails later, more expensively, and in a way
that is hard to trace. So: **conjecture freely, label it as conjecture, and
report what the runs actually did.** Everything below marked **measured** was
run and the number is given; everything else is design.

That is not caution, it is the only way the strategy pays. We are going to make
a lot of changes and we need to know which one worked.

---

## 1. What we are trying to build

"Open-ended evolution" names several different targets, and a system can hit
one while missing another. Choosing which one we are aiming at is the most
consequential decision in this document, because it determines what counts as
success and therefore what we build.

### 1.1 The definitions on offer

| definition | the thing that must grow without bound | can this substrate satisfy it? |
|---|---|---|
| **Bedau–Packard activity** (1992, 1998) — class 3: new components keep appearing *and* the stock of components keeps growing | cumulative adaptive activity, against a neutral shadow | Yes, once components persist |
| **MODES** (Dolson et al. 2019) — change, novelty, complexity, ecology, each with a persistence filter | four separate curves | Yes for change/novelty/ecology; complexity needs §4.6 |
| **Banzhaf et al. 2016 / Taylor et al. 2016** — variation vs innovation vs **transition**; novelty that *enlarges the space* | the dimensionality of the possibility space | Not with a fixed genome. This is the hard one |
| **Standish 2003** — measurable growth in the complexity of evolved entities | entity description length | Needs a growable entity |
| **Adams, Zenil, Davies, Walker 2017** — unbounded evolution requires the *transition rules* to depend on state | the rule set itself | Yes, via §4.7 |
| **Sayama — cardinality leap** | the *number of components* a system is made of | Yes: population is unbounded, and structures are made of agents |
| **Maynard Smith & Szathmáry 1995** — major transitions: new levels of individuality | levels in the hierarchy | This is the target |
| **Kauffman — the adjacent possible** | the set of things reachable in one step from what exists | Yes, and this is the key to the conservation objection |

### 1.2 The definition we adopt

> **Open-ended evolution here means unbounded growth in the space of realised
> multi-agent organisations — structures that persist, that are reproduced, and
> whose description cannot be bounded in advance by the configuration.**

The object of evolution is the **organisation**, not the genome.

This is a deliberate choice and worth defending, because it is what makes the
programme winnable.

**The genome route is closed by construction.** Layer sizes are fixed for a
run, so the genotype space is a fixed-dimensional box and every run is a walk
inside it. Under Banzhaf's grading that is *variation*, permanently, and no
amount of running changes it. If we define OOE as growth in genome complexity,
this substrate is disqualified before the first experiment — unless §4.6 lands.

**The organisation route is open.** The number of distinct labelled structures
on *n* agents grows super-exponentially in *n*, and *n* is bounded only by the
token supply, which is a parameter we control. Nothing in the rules caps how
many agents can participate in one organisation, how many organisations can
coexist, or how they can be arranged. The space is not fixed-dimensional; it
grows with the population.

**This also disposes of the conservation objection.** Tokens are conserved.
Arrangements are not. A conserved currency bounds how many agents can be alive
at once and *nothing else* — not how they are connected, not what they do to
each other, not how many levels of organisation sit between one agent and the
whole. Conservation is a carrying capacity, and carrying capacities do not
prevent evolution; every real ecosystem has one. What conservation gives us for
free is a **selection pressure with no fitness function**, which is exactly
what an open-ended system needs and what most artificial systems have to fake.

Kauffman's adjacent possible is the frame: each new organisation makes further
organisations reachable that were not reachable before. That is growth in the
reachable space driven by what has been reached, with a fixed currency.

### 1.3 What would count as proof

Four claims, in order of increasing strength. Each has a measurement in §6.

- **P1 — There is evolution.** Lineages retain identity long enough for
  selection to act, and selection changes the outcome. *(§6.1, §6.2)*
- **P2 — There is cumulative adaptation.** A late population beats its own
  ancestors under identical conditions. *(§6.3)*
- **P3 — There are organisations.** Persistent multi-agent structures exist
  above chance, survive the turnover of their members, and are reproduced.
  *(§6.4, §6.5)*
- **P4 — The space of organisations grows without bound.** New kinds keep
  appearing, the stock keeps growing, and neither is levelling off within the
  longest run we can afford. *(§6.6)*

P4 is the claim. P1–P3 are the ladder to it, and each is worth publishing on
its own.

---

## 2. Why this substrate deserves the effort

Most artificial life systems have to bolt on the things this one has for free.
Stated plainly, because it is the case for the programme.

**The interaction topology is endogenous.** Agents build the graph. Avida runs
on a fixed grid, Tierra on a linear memory soup, Polyworld on a plane. Here,
who can reach whom is a product of what the agents did — reproduction wires a
newborn into its parent's neighbourhood, and links that carry nothing are
pruned. Evolutionary graph theory (Lieberman, Hauert & Nowak 2005) shows the
graph determines selection strength; a system where the graph is itself
evolving is a setting that literature does not have a result for.

**The game has no optimum to converge to.** Colonel Blotto has no pure-strategy
equilibrium (Borel 1921; Roberson 2006), and the revolution rule destroys even
the pairwise ordering: whether A beats B depends on who else is present.
*(Measured: demonstrated on the engine's own resolver — §A.3.)* A system with a
best strategy runs to it and stops. This one cannot, which is a precondition
for open-endedness that most systems have to engineer around.

**Selection has no fitness function.** Tokens are life, wealth and voting power
at once. Nothing is optimised toward a goal; what survives, survives. Bedau's
critique of artificial systems is largely that their fitness functions bound
what can be discovered. There is no such bound here.

**Randomness is evolvable.** Each discrete decision is paired with a MODE head,
so an agent decides for itself whether that decision is read as a probability
or a maximum. The degree of stochasticity is part of the genome. That is
unusual, and it means exploration/exploitation balance is under selection
rather than under a parameter.

**There are already two inheritance channels.** Node lineage (who spawned whom)
and brain lineage (whose genome won a node) are tracked separately and can
diverge. A genome moving between unrelated agents by conquest is structurally
horizontal transfer, which is the substrate for symbiosis and for the major
transitions in §4.9.

**Conserved tokens give an intrinsic carrying capacity.** Population size is
self-regulating with no external cap, which means an ecology can find its own
equilibrium rather than being held at one.

---

## 3. What blocks it now

Three blockers, in the order they must be cleared. The first invalidates
measurements of the other two, so the order is not negotiable.

### 3.1 Heredity — *blocking everything*

**Measured.** `blotto_phase` mutates every living brain every iteration at
`mutation_probability` 0.5. Mutation is applied to the population rather than
at replication.

- A genotype's median life is **1 iteration**; 1% last more than five.
- A lineage is **45% of the way to a stranger after one iteration, 92% after
  ten** — in weights and in behaviour alike. Heredity half-life ≈ 3 iterations.
- With mutation off entirely the median genotype life is still 3, because
  conquest overwrites a node's brain. **Two independent shredders.**
- Replacing the whole conquest rule with a coin among stakers leaves the
  population 38% larger and 37% more diverse, at 30 seeds.

Full numbers in §A.1. Fix in §4.1.

Nothing measured before this is cleared means what it appears to mean. Every
statistic this project built — diversity, lineage forests, flow-module
compression, activity — looks *healthy* under zero heredity, because none of
them measures retention.

### 3.2 The possibility space cannot expand

No rule brings a new kind of thing into existence. Layer sizes are fixed, token
types are one, the set of decisions an agent can make is fixed at
configuration. Under §1.2 this is survivable — organisations can grow even if
genomes cannot — but it caps how far the programme can go, and §4.6/§4.7 lift
it.

### 3.3 Nothing is positive-sum

There is no rule by which cooperation produces surplus.
`tokens_created_per_phase` injects tokens globally and unconditionally, which is
weather, not production. Without local surplus there is no reason for a
multi-agent organisation to be worth more than its parts, and organisations are
the object we have chosen to evolve. Fix in §4.4.

---

## 4. The mechanics

Every one of these is **optional, off by default, and named** so an experiment
can cite exactly which are on (§5). Ordered by expected leverage. Confidence is
stated because most of these are conjecture.

### 4.1 `mutate_on_replication` — tie mutation to copying
*Confidence: high. This is the one that has to be first.*

Delete the world-wide mutation loop at the end of `blotto_phase`; a child is
already a mutated copy of its parent in `_spawn_child`. A genome then changes
only when it is copied, which is what makes a lineage a lineage.

**What it does not fix:** conquest still overwrites a node's brain, so a genome
vanishes when its carrier loses. That is not a defect — it *is* how a
successful genome spreads, and it is why brain lineage must be followed rather
than node lineage.

**Success:** heredity half-life bounded by how long a lineage survives rather
than by the mutation rate, and the §6.2 ablation separating in the *right*
direction.

### 4.2 `germline` — separate the genome from the working copy
*Confidence: medium. Only if 4.1 is not enough.*

Give each agent an unmutated germline genome alongside the expressed one.
Conquest copies the expressed genome; reproduction copies the germline. This
is Weismann's barrier, and it makes lineage identity survive conquest, which
4.1 alone does not.

### 4.3 `token_colours` — conserve the count, not the kinds
*Confidence: medium-high. This is the most interesting idea here.*

Give each token a colour from a set of size *c*. Agents have a heritable
affinity vector; a token of a colour an agent has no affinity for is worth
less to it. Conversion between colours is possible at a loss, or only through
particular agents.

**Why this matters more than it looks.** It answers the conservation objection
at the mechanical level rather than the philosophical one: the *number* of
tokens stays fixed while the space of colour-distributions over the population
is unbounded. It creates **niches** immediately — an agent specialised in a
colour cannot be displaced by a generalist on that colour's terms — and niches
are what MODES measures as ecology and what every open-ended system has.

It also creates a reason for **trade**, which is the cheapest possible route to
positive-sum interaction (§4.4) without minting anything.

Allow `c` itself to grow — a mutation that splits a colour in two — and the
possibility space expands, which is §3.2 addressed without touching the genome
architecture.

### 4.4 `mutual_flow_yield` — make cooperation produce something
*Confidence: medium-high.*

An edge carrying tokens in **both** directions in the same game phase yields a
small number of new tokens, split between its ends. Nothing is created except
by an interaction. This converts the game from constant-sum to variable-sum and
creates an immediate defection problem, which is the substrate for everything
in evolutionary game theory.

The theory is developed: network reciprocity predicts cooperation is favoured
when `b/c > k` (Ohtsuki, Hauert, Lieberman & Nowak 2006). This system has an
evolving `k` — *measured*: mean degree falls ~3.9 → ~3.2 over 45 iterations —
and the agents partly control it, which makes the threshold an evolvable target
rather than a parameter. That is a genuinely novel setting for that result and
is publishable on its own.

### 4.5 `edge_proposal` — let the graph grow, not only erode
*Confidence: medium.*

Edges are currently created **only** at birth and destroyed **every** iteration
by pruning. The topology erodes by default. Add a head by which two agents can
both propose a link and have it form if both do, at a token cost.

Without this, no organisation can *build* anything — it can only inherit
whatever topology birth happened to give it.

### 4.6 `growable_layers` — let the genome gain capacity
*Confidence: low-medium, high value.*

A mutation that adds a hidden unit, at a token cost proportional to the added
capacity. This is the one mechanic that would satisfy Banzhaf's *transition*
grade directly, and the only one that makes a complexity measure (§6.6) mean
what Standish means by it.

The cost is what stops it running away: capacity has to pay for itself.
Currently complexity has no cost here, which is why it also has no reason to
grow.

### 4.7 `local_rules` — agents that change their own rules
*Confidence: low, highest ceiling.*

Adams, Zenil, Davies & Walker (2017) argue that unbounded evolution in a
dynamical system requires the transition rules themselves to depend on state.
Concretely here: let brain outputs set an agent's *own* mutation rate,
reproduction threshold, or edge-forming permission. Evolvable evolvability.

This is the mechanic most likely to produce something nobody predicted, and
the most likely to produce degenerate runaway. Worth trying late, with the
ablation ready.

### 4.8 `contracts` — a second replicator
*Confidence: low, speculative.*

Let an agent carry a small heritable structure describing how it interacts with
a specific neighbour, copied and mutated independently of the brain. A second
replicator on a different timescale is how several natural major transitions
worked, and it gives organisations something to inherit that is *theirs* rather
than their members'.

### 4.9 `structure_replication` — the major transition
*Confidence: very low, this is the endgame.*

If a group of agents satisfies a closure condition (§6.4 defines candidates),
allow the group to seed a copy of itself elsewhere in the graph at a token
cost. This is the explicit version of what we hope emerges on its own. Building
it in is arguably cheating; having it available tells us what the measurements
look like when the phenomenon is definitely present, which is the calibration
every detector in §6 needs.

---

## 5. Strain labelling — how to name an algorithm

An experiment must cite the exact algorithm it ran on, and that citation must
still resolve in a year when six more mechanics exist.

### 5.1 The three kinds of setting

| kind | definition | examples | in the strain id? |
|---|---|---|---|
| **Mechanic** | changes *which code paths can execute* | `allow_revolutions`, `brain_kind`, every flag in §4 | **yes** |
| **Parameter** | changes magnitudes only | `total_tokens`, `n_nodes`, `hidden_layers`, `mutation_probability`, `seed` | no — cited separately |
| **Infrastructure** | no effect on the run | `checkpoint_every`, `export_every`, `export_decisions` | no |

### 5.2 The strain identifier

```
gol-<SPEC>[+<mechanic>[=<value>]]...
```

- `SPEC` is an integer in `gol_config.py`, currently **1**. It is bumped
  **only** when behaviour changes in a way that cannot be expressed as an
  optional flag — a bug fix that changes results, or a change of default.
- Every mechanic has a **frozen default**, chosen so the default reproduces
  SPEC-1 behaviour. Defaults never change; that is the whole trick.
- The identifier lists only mechanics whose value **differs from the frozen
  default**, sorted alphabetically. Booleans that are on appear bare.

**The property that makes this work:** adding a new mechanic that defaults to
off does not change the identifier of any existing strain. `gol-1` means the
same thing forever.

Examples:

```
gol-1                                        the algorithm as it stands today
gol-1+mutate_on_replication                  heredity fixed, nothing else
gol-1+mutate_on_replication+token_colours=4  heredity plus four token colours
gol-1+brain_kind=binary                      the binary brain, no new mechanics
gol-2+mutate_on_replication                  same flags, after a spec bump
```

### 5.3 Citing an experiment

A strain pins the mechanics. It does not pin magnitudes, and it does not pin
bugs. So an experiment record carries four things:

```
strain:  gol-1+mutate_on_replication
setup:   total_tokens=2500 n_nodes=50 k_neighbors=6 hidden_layers=[12,10]
         mutation_probability=0.5
seeds:   1..30
commit:  a1b2c3d
```

`commit` is the backstop: it reproduces the code exactly, including anything
the strain scheme failed to capture. `strain` is what makes two experiments
*comparable* — the thing you group by and put on an axis.

### 5.4 The registry

`research/strains.md` records every strain an experiment has ever used: the
identifier, the mechanics it turns on, the commit that introduced them, and a
one-line note on what it was for. A strain is never removed and never
redefined. If a mechanic's meaning has to change, it gets a new name.

### 5.5 Implemented

1. ✅ `SPEC = 1`, and `MECHANICS` / `PARAMETERS` / `INFRASTRUCTURE` in
   `gol_config.py` classifying every setting.
2. ✅ `SimConfig.strain_id()`.
3. ✅ The strain written into run metadata (both backends), the checkpoint, and
   the series cache — every store that can be found on its own.
4. ✅ `research/strains.md`, with the nine unbuilt mechanics named and
   defaulted in advance.

Two guards keep it honest, both in `tests/test_engine.py`: every setting must
be classified as exactly one of the three kinds, so a new one cannot be added
without deciding; and every frozen default must equal the actual default, so
`gol-1` always names a world you get by asking for nothing.

The reserved mechanics are deliberately **not** config fields. Setting one
raises rather than quietly producing a strain name for a run that ignored it —
a label that lies is worse than no label.

---

## 6. The measurement battery

Each measurement names the claim it serves (§1.3) and what would count as
success. **Every one needs a control**; the recurring failure in this project
has been a real number that meant nothing without one.

### 6.1 Heredity half-life — *serves P1*
Descendant-to-ancestor distance against a background of unrelated pairs, in
weights and in behaviour, at increasing horizons. **Success:** the curve stops
saturating within the run; retention is limited by lineage survival, not by the
clock. *(Exists: `research/pilot_heredity.py`.)*

### 6.2 Selection ablation — *serves P1*
The same world with the conquest winner decided by a coin among stakers.
**Success:** the two separate, and the real rule is the better one.
*(Exists.)*

### 6.3 Cross-time tournament (CIAO) — *serves P2, the decisive one*
Replay a late population against its own ancestors under identical conditions
(Cliff & Miller 1995). **Success:** a clean gradient — later beats earlier
everywhere. Banded patterns mean cycling, which is *also* interesting and is
what non-transitivity predicts.

This is the measurement that cannot be fooled by a system that forgets: a
population with no heredity cannot beat its own past. It is the highest-value
thing to build, checkpoints already hold what it needs, and it answers P2 on
its own.

### 6.4 Organisation detection — *serves P3*
An organisation is a set of agents whose joint behaviour persists while its
membership turns over. Three candidate detectors, all partly built:

- **Flow modules** (map equation, Rosvall & Bergstrom 2008) — groups a
  token-walk tends to stay inside. *Measured:* 41.5% compression, longest-lived
  48 frames, ~23% membership turnover per frame. **No null model yet, so this
  number currently means nothing.**
- **Token cycles** — circulation without conquest. Never looked for.
- **Periodic birth/death patterns** — a structure that regenerates on a period.
  Nothing detects period > 1.

**Success:** any detector finding structure significantly above a
degree-preserving rewired null.

### 6.5 Information-theoretic individuality — *serves P3*
Krakauer, Bertschinger, Olbrich, Flack & Ay (2020): an individual is a subset
that propagates information about its own past into its own future in excess of
what its environment supplies. The flow modules are a cheap proxy for this; the
real quantity is computable on small groups. **Success:** groups with
self-predictive information above the environmental baseline, persisting.

### 6.6 Unbounded growth — *serves P4, the claim*
Three curves, each against a neutral shadow:
- **Activity** (Bedau) — cumulative presence of persistent components.
- **Novelty and complexity** (MODES) — with a persistence filter, which matters
  enormously here because everything mutates.
- **Organisation count and size** — how many distinct organisations exist and
  how large the largest is.

**Success:** none of the three levelling off within the longest run affordable,
and all three above their shadows. **Failure that is still informative:** a
clean asymptote, which tells us which mechanic to add next.

### 6.7 Topology — the medium an organisation would have to live in
An organisation is a subgraph, so whether organisations can exist is partly a
question about the graph they would occupy. `Graphs.md` works this out: the
shape that supports mesoscale structure is a well-connected core with structure
attached to it, and it is identifiable three independent ways — bridge balance
and whisker sizes, the spectral gap `λ₂`, and the distribution of per-edge
effective resistance. ✅ The first of those three is built: `cutRisk`,
`cutRiskBefore`, `coreShare` and `ricciCurvature` are recorded every frame, and
each claim has a panel in the **Theses** tab stating in advance what would
confirm and what would refute it. `λ₂` and effective resistance are not.

Two things make this urgent rather than decorative. **No edge in this algorithm
is ever created between nodes more than two hops apart**, so the small-world
shortcuts the world starts with are a budget that is spent and never refilled —
which is the sharpest argument yet for §4.5. And because cleanup keeps only the
largest component, a bridge with a tenth of the population behind it *is* a
tenth-of-the-population extinction, waiting for the zero-flow prune to reach it.
**Control:** the Watts–Strogatz graph the run started from, and a
degree-preserving rewire of each frame.

Measured once already (`research/pilot_topology.py`, and Appendix A.8): the
first is confirmed against the control by a wide margin, the second is not
supported and was not even askable until the engine began taking the reading
before the cull rather than after it.

### 6.8 Ablation as standing method
Soros & Stanley 2014: for each mechanic in §4, run with and without and show
the effect disappears. A property that survives every ablation was never caused
by the mechanism claimed. Every headline result gets this treatment before it
is believed.

---

## 7. What the comparison class does

Worth knowing what we are measured against, and where this substrate is
genuinely different.

| system | what made it open-ended (or not) | what this substrate has instead |
|---|---|---|
| **Tierra** (Ray 1991) | self-replicating machine code in shared memory; genome length can grow; got parasites and hyper-parasites within hours | fixed genome; but an evolving *topology*, which Tierra has no analogue of |
| **Avida** (Ofria & Wilke 2004) | rewarded task hierarchy; showed complex functions need rewarded intermediate steps | no reward function at all — a strength for open-endedness, a weakness for measurement |
| **Geb** (Channon 2001) | unbounded genome plus coevolution; passed Bedau class 3 by its author's measure | this is the bar for §6.6 |
| **Polyworld** (Yaeger 1994) | embodiment and ecology in a 2D world | ecology is what §4.3 and §4.4 are for |
| **Novelty search** (Lehman & Stanley 2011) | abandoning the objective entirely | we have no objective to abandon |
| **Chemlambda / autopoietic sets** | new *kinds* of entity from combination | §4.9 is the analogue and is not built |

The pattern: **every system that convincingly showed open-endedness let its
entities grow.** That is the strongest argument against the current build and
the reason §4.3 and §4.6 exist. Our bet is that letting the *organisation* grow
is as good as letting the *genome* grow — and if that bet is wrong, §4.6 is the
fallback.

---

## 8. The programme

Ordered so each step's measurement is meaningful when it is taken.

**Phase 0 — instrumentation.** §5.5, the strain scheme. Then §6.3, the
cross-time tournament, because it is the decisive measurement and it is
independent of every mechanic below.

**Phase 1 — make it evolve.** `mutate_on_replication`. Measure §6.1, §6.2,
§6.3 against `gol-1`. This is the paper "a graph-structured Blotto world
evolves", and it is not open-endedness yet.

**Phase 2 — give it an ecology.** `token_colours`, then `mutual_flow_yield`,
then `edge_proposal`. Measure §6.4, §6.5. Expect organisations here or not at
all.

**Phase 3 — let it grow.** `growable_layers`, `local_rules`. Measure §6.6.

**Phase 4 — the honest attempt to break it.** §6.8 across every mechanic, and
the longest run affordable, looking for the asymptote.

Rules for the whole programme, learned the hard way in this project:
- **Nothing below ~30 seeds is an effect.** Two results here reversed between
  6 seeds and 20.
- **No number without its null.** Conquest cycles looked abundant in the
  thousands and came out *below* chance against a random-neighbour null.
- **Grep the callers before believing a mechanism matters.** A defect analysed
  for two days turned out to be in a function with no callers.

---

## 9. Risks

Stated so that hitting one is recognised rather than explained away.

- **The organisation bet fails.** Organisations never appear above a null even
  with ecology. Then §1.2's reframe was wrong and the genome route (§4.6) is
  the only one left.
- **Runaway.** `local_rules` produces agents that set their own mutation rate
  to zero and freeze, or to one and dissolve. Expected; the ablation catches it.
- **Detector-driven results.** We build a detector, it finds something, and the
  something is the detector. This has already happened once here (conquest
  cycles). Every detector gets a null and a positive control — §4.9 exists
  partly to *be* that positive control.
- **The measurement outruns the compute.** Bedau class 3 needs long runs.
  Checkpointing and the frame window exist; the budget is the constraint.
- **Definition drift.** Choosing §1.2 because it is winnable is legitimate;
  quietly re-choosing it *after* seeing results is not. §1.2 is fixed now, in
  writing, before the experiments.

---

## Appendix A — measured facts, carried over

Everything here was run. These survive the change of goal because they are
observations, not arguments.

### A.1 Heredity *(`research/pilot_heredity.py`)*

Genotype lifetime, by mutation rate:

| `mutation_probability` | median life | 90th pct | ever > 5 iterations |
|---|---|---|---|
| 0.50 (default) | **1** | 3 | **1.0%** |
| 0.20 | 1 | 5 | 7.4% |
| 0.05 | 2 | 7 | 15.0% |
| 0.00 | 3 | 22 | 31.6% |

Lineage memory — distance from an agent's own ancestor as a fraction of the
distance between unrelated agents; 1.0 means nothing is left:

| iterations later | weights | behaviour |
|---|---|---|
| 1 | 0.45 | 0.45 |
| 2 | 0.53 | 0.53 |
| 3 | 0.58 | 0.56 |
| 5 | 0.71 | 0.70 |
| 10 | **0.92** | **0.89** |

Selection ablation, 30 seeds:

| winner chosen by | extinct | median agents | distinct brains | mean degree |
|---|---|---|---|---|
| stake (as shipped) | 0/30 | 430 | 373 | 3.10 |
| **chance** | 0/30 | **594** | **511** | 3.24 |

### A.2 Scale
Extinction falls with founder count: 3/3 dead at 6 founders, 0/3 at 100.
Founding lineages still coexist after 60 iterations in 2/3, 3/3, 2/3 of runs at
40, 100, 250 founders. **Scale is a real answer, not an evasion.**

### A.3 The strategy space has no order
Revolutions make the winner depend on who else is present rather than on
pairwise strength. Demonstrated on the engine's own `_resolve_winner`: H beats
M and H beats S pairwise, but adding S to the H-versus-M contest makes **M**
win, while S itself wins nothing.

### A.4 Flow modules
41.5% compression of a token-walk, longest-lived module 48 frames, ~23%
membership turnover per frame. **No null model — this number does not yet
count as evidence.**

### A.5 Conquest cycles are not structure
Thousands per run; 2-cycles at **0.87–0.91×** a random-neighbour null. Below
chance. The abundance was the trap.

### A.6 Symbiosis is already expressible
An agent defending its node with a self-stake while a neighbour adds a smaller
stake is a gift the rules already allow. Demonstrated against `_resolve_winner`.

### A.7 The brains
A noise input is ~4.4× as loud as a magnitude in the first layer (38.5% of
variance over 5 inputs), and removing noise entirely changes survival not at
all over 30 seeds. Widening a binary brain's last hidden layer takes it from 1
to 10 distinct staking scores and cuts coin-flip ties from 27% to ~4%, and
changes survival not at all. **How finely an agent can state a preference is
not currently what is being selected on** — consistent with §3.1.

---
### A.8 Topology *(`research/pilot_topology.py`)*
Two long runs, against a Watts–Strogatz graph rebuilt at the same size and mean
degree. The graph ends with **22–30% of its edges being bridges where the
control has 0.0%**, clustering at 0.02–0.06 against 0.24, and path length above
the small-world ratio — it has left the shape it started as entirely. The
departure happens **early**: by the first sampled window of a long run it is
already at 19% bridges, so a long run's trend describes the aftermath, not the
transition.

Two things that cut against the easy reading. `orphaned` is non-zero in **every
recorded frame** — the cull is continuous, not catastrophic, so the tree-like
drift is being punished all along and happens regardless. And clustering
*falls* rather than rising, so newborns are mostly not closing triangles: the
graph grows by adding thin pendant structure, which is whisker growth.

## Appendix B — literature

Rendered for a reader in the site's **Literature** tab, with a note on what
each means here. Summaries there are from working knowledge and need checking
against the papers. Graph structure is a second document, in the **Graphs** tab:
tree-likeness and expansion, the core-and-whiskers shape real networks take, the
four ways graphs get categorised, and the Wolfram Physics Project's geometric
estimators — one of which this project already uses.

**Open-endedness:** Bedau & Packard 1992; Bedau, Snyder & Packard 1998; Dolson,
Vostinar, Wiser & Ofria 2019 (MODES); Banzhaf et al. 2016; Taylor et al. 2016;
Packard et al. 2019; Standish 2003; Adams, Zenil, Davies & Walker 2017; Soros &
Stanley 2014; Lehman & Stanley 2011.

**Systems:** Ray 1991 (Tierra); Ofria & Wilke 2004 (Avida); Channon 2001 (Geb);
Yaeger 1994 (Polyworld).

**Evolutionary game theory on graphs:** Nowak & May 1992; Lieberman, Hauert &
Nowak 2005; Ohtsuki, Hauert, Lieberman & Nowak 2006; Nowak 2006; Gross &
Blasius 2008.

**The game:** Borel 1921; Roberson 2006; Van Valen 1973; Cliff & Miller 1995.

**Individuality and structure:** Maynard Smith & Szathmáry 1995; Price 1970;
Krakauer, Bertschinger, Olbrich, Flack & Ay 2020; Crutchfield & Hanson 1993;
Rosvall & Bergstrom 2008; Kauffman (adjacent possible); Sayama (cardinality
leap).

**Method:** Kimura 1968 and neutral models generally; Lenski's LTEE — the
citrate innovation took ~31,000 generations and depended on potentiating
mutations that were invisible when they happened. Our runs are hundreds of
iterations. That is the timescale corrective, and it is why §6.6's failure mode
is "we did not run long enough" and why that has to be distinguishable from "it
levelled off".
