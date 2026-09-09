# Graphs

What kinds of graph there are, how the kinds are told apart, and which of those
measurements this substrate should be making. Read against the same goal as
`Literature.md`: the organisation we are trying to grow has to live *in* a
topology, so knowing which topology we have is not background reading.

Sources checked September 2026. The **Here** blocks are the part that earns an
entry its place — what to copy, what to measure, or what it rules out.

---

## 1. A tree, and its several opposites

A tree is over-determined. Four different properties all hold at once, and each
one is a *different* thing to negate — so "the opposite of a tree" has no single
answer, and which one you mean decides what you should measure.

| a tree is… | negate it and you get | the extreme case |
|---|---|---|
| acyclic — `m = n - 1` | a graph with many independent cycles | the complete graph `K_n`, `m = n(n-1)/2` |
| entirely bridges — every edge disconnects | a bridgeless graph | any 2-edge-connected graph |
| cut in half by one edge | a graph with **no sparse cut at any scale** | an **expander** |
| of treewidth 1 | a graph with no small balanced separator | treewidth `Θ(n)` |
| 0-hyperbolic — its metric *is* a tree | a metric far from any tree | a grid or a long cycle, `δ ≈ diameter/2` |

These come apart, and the counterexample that proves it is the **grid**. A grid
is nothing like a tree: treewidth `√n`, hyperbolicity huge, cycles everywhere.
It is also nothing like an expander: cut it down the middle and you sever `√n`
edges out of `n`, which is a sparse cut. So "tree ↔ complete graph" is not an
axis anything can be ranked on. There are at least three axes and a graph can
sit anywhere on each.

The opposite in the sense that matters for a **decentralised** structure — no
chokepoints, no part that can be severed cheaply — is the **expander**. Formally
the Cheeger constant, or edge expansion:

```
h(G) = min over S, |S| <= n/2  of  |edges leaving S| / |S|
```

`h` is large when *every* way of splitting the graph is expensive. A tree has
`h ≈ 1/n`; a random 3-regular graph has `h` bounded away from zero as `n` grows.
Cheeger's inequality ties `h` to the second-smallest eigenvalue `λ₂` of the
normalised Laplacian — the **spectral gap** — which is why `λ₂` is usually used
as the working definition: it is the one you can actually compute.

> **Here.** "Decentralised" is not a vibe, it is `λ₂`. ✅ Now measured, and the
> answer is **0.00086** — near enough to zero that this graph is very nearly in
> two pieces. See §6, which also records how the first algorithm tried gave a
> number forty times larger that turned out to be a reading of its own
> iteration count.

---

## 2. How far from a tree — the scalars

Everything below is a number, on one graph, in polynomial time. Marked ✅ where
`gol_series.py` already computes it and the series carries it per frame.

| measure | tree gives | expander gives | ours |
|---|---|---|---|
| **circuit rank** `m - n + c` — independent cycles, the first Betti number | 0 | `Θ(m)` | ✅ `cycleRank` |
| **loop density** — circuit rank over edges | 0 | → 1 | ✅ `loopDensity` |
| **bridge count** — edges whose removal disconnects | `n - 1`, all of them | 0 | ✅ `bridges` |
| **transitivity** — closed triples over triples | 0 | ≈ 0 in a sparse expander | ✅ `transitivity` |
| **assortativity** — Newman's degree correlation | — | — | ✅ `assortativity` |
| **degree exponent** — power-law tail fit | — | — | ✅ `degreeExponent` |
| **ball-growth dimension** — shell size against radius | ill-defined, grows exponentially | ill-defined, grows exponentially | ✅ `dimension` |
| **box dimension** — boxes needed to cover, against box size | — | — | ✅ `boxDimension` |
| **degeneracy / arboricity** — min forests covering `G` (Nash-Williams) | 1 | `> 1` | ✗ |
| **treewidth** — width of the best tree decomposition | 1 | `Θ(n)` | ✗ |
| **δ-hyperbolicity** — Gromov's 4-point condition | 0 | `O(log n)`, and *not* small | ✗ |
| **spectral gap** `λ₂` / **conductance** | `→ 0` | bounded below | ✅ `spectralGap` |
| **effective resistance** per edge | 1 on every edge | `≈ (n-1)/m` on every edge | ✗ |

Three of the missing four deserve their own note.

### Effective resistance, and Foster's theorem
Treat each edge as a one-ohm resistor and measure the resistance `R_e` across
the edge's own endpoints. `R_e = 1` exactly when the edge is a bridge — no
parallel path — and falls toward zero as the edge is bypassed by more and more
alternatives. It is also, by Kirchhoff, the probability that the edge appears in
a uniformly random spanning tree. So it is a **continuous, per-edge measure of
bridge-ness**, where the bridge count is only its 0/1 shadow.

**Foster's theorem (Foster 1949):** for a connected graph, the sum over all
edges is fixed —

```
sum over e of R_e  =  n - 1
```

Tree-likeness is a **conserved quantity**, `n - 1` of it, distributed over `m`
edges. A tree spends exactly one unit per edge and has `m = n - 1`, so it is the
unique graph where every edge is saturated. Everything else spreads the same
budget more thinly, mean `(n-1)/m`.

> **Here.** A conserved quantity spread over the graph, whose *distribution* is
> the entire content because the total is fixed by `n`. That is the same shape
> as the token rule this whole simulation is built on, and it makes the natural
> summary obvious: not the mean, but the Gini coefficient or the maximum — the
> code already has `_gini` for tokens and it would apply unchanged.

### Gromov δ-hyperbolicity
For any four nodes, form the three pairwise distance sums; the two largest
differ by at most `2δ`. Trees give `δ = 0`; cycles and grids give roughly half
the diameter. It measures whether the *metric* is tree-like, independently of
whether the graph is. Naïvely `O(n⁴)`; sampled quadruples are the standard
practice and are what the empirical literature uses.

### Treewidth
The width of the best tree decomposition — cover the graph with overlapping
bags arranged in a tree, such that every edge is inside some bag and each node's
bags form a connected subtree. Width is the largest bag minus one. NP-hard
exactly, but greedy elimination orderings (min-degree, min-fill) give usable
upper bounds, and that is what Adcock, Sullivan & Mahoney used on real networks.

---

## 3. The thing in between — and it has a name

The description to pin down: *decentralised, loosely intertwined, no bridge with
more than ~10% of the nodes on one side, but still plenty of small tree-like
structures hanging off local points.*

That is a real and well-studied object. Three literatures found it from three
directions, and two of them are describing the same thing.

### (a) Locally tree-like, globally an expander
**Benjamini & Schramm 2001**, local weak convergence; standard for sparse random
graphs.

A random `d`-regular graph converges locally to the infinite `d`-regular
**tree**: pick a node, look at its neighbourhood out to any fixed radius, and
with high probability you see a tree, because the girth grows like `log n`. Yet
the same graph is an **expander** — `λ₂` bounded away from zero, no sparse cut at
any scale. Zoom in and it is a tree; zoom out and it is the least tree-like
object there is. No contradiction: the cycles exist, they are just all long.

This is the *homogeneous* version of the description. Every neighbourhood looks
the same.

### (b) An expander-like core with whiskers — this is the one
**Leskovec, Lang, Dasgupta & Mahoney 2008/2009**, *Internet Mathematics* 6(1),
over 100 real networks.

They define the **network community profile**: for each size `k`, the best
conductance achievable by any set of `k` nodes. Plotted against `k` the curve
falls to a minimum at around **100 nodes** and then rises steadily. Verbatim from
the abstract: they observe *"tight communities that are barely connected to the
rest of the network at very small size scales"*, while larger ones *"gradually
'blend into' the expander-like core of the network and thus become less
'community-like'."*

The small tight pieces are **whiskers** — subgraphs attached to the rest of the
graph by a *single edge*. They are what makes the curve fall. The core, and the
way whiskers root into it, is what makes it rise.

So: no good large cut, many small barely-attached appendages. That is the
description, exactly, and it is what real social and information networks
actually look like. They also report that **no standard generative model
reproduces this** — not Erdős–Rényi, not preferential attachment, not
Watts–Strogatz. Forest-fire growth was the only one that came close.

> **Here.** The intuition is right and it is the empirically common case, not an
> exotic one. It also gives the formal handle: **whisker = a component hanging
> off a bridge**. Which means the object to compute is the bridge tree.

### The bridge tree, and how to make "10%" precise
Contract every 2-edge-connected component to a point. What remains is a **tree**,
and its edges are exactly the bridges of the original graph. Every graph
decomposes this way, uniquely. So:

- **bridge balance** — for each bridge, removing it splits the graph into `a` and
  `n - a`; take `min(a, n-a)/n`. The condition *"no bridge with more than 10% on
  one side"* is `max over bridges of that ratio < 0.10`.
- **whiskers** — the small sides. Their size distribution is the "many small
  tree-like structures at local points" half of the description.
- **the 2-core** — iteratively delete degree-1 nodes. What survives is the graph
  with all its hanging trees stripped off. Core size over `n` says how much of
  the graph is periphery.

`_structure()` in `gol_series.py:145` **already finds every bridge** by
depth-first search, and then keeps only how many there were. The side sizes are
one more pass over the same DFS — the low-link values that identify a bridge
also delimit the subtree beneath it, so the split is available at the moment the
bridge is found and is currently discarded.

### (c) Tree-like at large scale, expander-like at small
**Adcock, Sullivan & Mahoney 2013**, ICDM, tree decompositions and
δ-hyperbolicity on real social and information networks.

The same object from the other side. Measured by treewidth and hyperbolicity,
large networks are *more* tree-like than random graphs of the same size — but
the tree-likeness is a large-scale property, and tree-decomposition structure
correlates strongly with **core-periphery** organisation. (b) and (c) are one
structure: a dense core, a tree-shaped fringe, and the fringe is what both the
NCP dip and the low hyperbolicity are seeing.

The support is mixed rather than clean — the follow-up work (*Tree decompositions
and social graphs*, 2014) is careful that "social networks have small treewidth"
is not established. Take the method, not the headline.

> **Here.** Three measures, three answers, one structure. If our graph is doing
> anything organised, this is the shape it is most likely to be doing, and none
> of the three numbers that would show it are currently computed.

---

## 4. How graphs get categorised — four traditions

Worth separating, because they are genuinely different projects and do not
reduce to one another. Asking "what kind of graph is this" without saying which
tradition you are in has no answer.

### (i) Structural graph theory — by forbidden substructure
Classes are defined by what they *cannot contain*. Planar graphs exclude `K₅`
and `K₃,₃` as minors; chordal graphs exclude long induced cycles; bounded
treewidth excludes large grid minors.

**Nešetřil & Ossona de Mendez**, *Sparsity* (2012), put the sparse classes in one
hierarchy, each strictly inside the next:

```
bounded degree  ⊂  bounded treewidth  ⊂  excluded minor
                ⊂  excluded topological minor
                ⊂  bounded expansion  ⊂  nowhere dense  ⊂  somewhere dense
```

The device is the **shallow minor**: contract disjoint connected subgraphs of
radius at most `r`, then ask how dense the result can get. Bounded expansion
means bounded average degree at every depth `r`; nowhere dense is the weaker
requirement that some graph is excluded at each depth. The payoff is exact:
first-order model checking is fixed-parameter tractable on a hereditary class
**precisely** when it is nowhere dense. A structural boundary that is also an
algorithmic one.

Relevant caveat: **real complex networks are usually not in these classes** —
heavy-tailed degrees break bounded degree immediately. There is work placing
random network models inside bounded expansion (*Structural sparsity of complex
networks*, 2019), but it is not the default.

### (ii) Network science — by which generator reproduces the statistics
Erdős–Rényi, Watts–Strogatz small-world, Barabási–Albert preferential
attachment, the configuration model, stochastic block models, forest fire. A
graph is "scale-free" or "small-world" if the generator's signature statistics
match.

Two standing cautions:

- **Broido & Clauset 2019**, *Nature Communications* 10:1017, "Scale-free
  networks are rare". Nearly 1000 datasets, proper likelihood-ratio testing
  against log-normal alternatives: strong scale-free structure is rare, only 4%
  in the strongest category, and log-normal fits as well or better for most.
  Fitting a straight line to a log-log degree plot is not evidence.
- **Leskovec et al. above:** no standard generator reproduces the community
  profile of real graphs. The generative taxonomy does not span the observed
  space.

> **Here.** We already fit `degreeExponent` with an R² and a tail cut, which is
> better than most, but the honest reading of Broido & Clauset is that the
> number should not be quoted without a log-normal comparison. Filed, not fixed.

### (iii) Spectral and geometric — by the geometry the metric implies
`λ₂`, Cheeger constant, effective resistance, hyperbolicity, and **discrete
curvature**. Ollivier-Ricci curvature of an edge compares the optimal-transport
distance between the neighbourhood distributions of its endpoints against the
edge length: positive where neighbourhoods overlap (triangles, communities),
negative where the edge is a bottleneck between different regions. Ni, Lin, Luo
& Gao (*Scientific Reports* 9:9984, 2019) do community detection by Ricci flow —
repeatedly stretch negatively-curved edges until the communities fall apart.

And a theorem that ties this tradition to the last one — **Salez 2021**, *sparse
expanders have negative curvature*: bounded-degree expanders with non-negative
Ollivier-Ricci curvature **do not exist**. "Decentralised" and "negatively
curved" are not two properties, they are one property in two vocabularies.

### (iv) Mesoscale and functional — by what a process on the graph does
Modularity, the map equation, k-core decomposition, core-periphery models,
motifs, the NCP. The unit is not the whole graph and not the node, but the
*module*, and the criterion is behavioural: where does a random walk get
trapped, which edges carry flow, what survives peeling.

> **Here.** This is the tradition we are already in — the Flow modules view is
> the map equation. What we do not have is the axis it should be read against:
> a module is only meaningful relative to how modular the graph is *at all*,
> which is (iii)'s `λ₂`. A partition of an expander is an artefact, and we
> cannot currently tell whether ours is one.

---

## 5. The Wolfram Physics Project

**Wolfram 2020**, *A Class of Models with the Potential to Represent Fundamental
Physics*, *Complex Systems* 29(2); the technical introduction at
wolframphysics.org; Gorard and others at the Wolfram Institute since.

The model: a **hypergraph** rewritten by local rules. Space is not a background,
it is the connectivity of the hypergraph and nothing else — no embedding, no
coordinates. Repeated rewriting generates a **causal graph** of which updates
enabled which; running every possible update order at once generates the
**multiway graph**; slicing that across branches gives **branchial** space.
**Causal invariance** — different update orders converging on the same causal
graph — is the property that makes any of it well-defined.

### What they have that is directly usable

**Emergent dimension from ball growth.** Pick a node, count how many nodes lie
within `r` hops. In `d`-dimensional space that volume grows like `r^d`, so the
log-log slope estimates `d`. Dimension is not put in, it comes out — and it need
not be an integer, and can vary across the graph.

This project **already does this**. `web/js/graphstats.js:678` says so in as many
words: *"Ball-growth dimension, in the spirit of the Wolfram Physics Project."*
`gol_series.py:300` walks the shells, fits `log(shell)` against `log(r)`, and
adds one.

**Curvature from the same measurement, which we throw away.** The volume of a
geodesic ball in a curved `d`-dimensional space is not `r^d` but

```
V(r) = ω_d r^d [ 1 - R r² / (6(d+2)) + O(r⁴) ]
```

with `R` the Ricci scalar. Differentiating for the shell our code actually
measures:

```
S(r) = d ω_d r^(d-1) [ 1 - R r² / (6d) + O(r⁴) ]
```

Positive curvature means balls are smaller than flat space predicts, negative
means larger. The existing fit takes `log S(r)` against `log r` and keeps the
slope. Adding an `r²` term to that same regression recovers `R` — the curvature
is already in the data being fitted and is being discarded as residual.

### What they have to say about measuring emergent complexity
This is where the honest answer is uncomfortable, because their position is
close to *you cannot*, and it is argued rather than assumed.

- **The four classes** (NKS, and the cellular-automaton work from 1983) are a
  *qualitative* taxonomy — uniform, periodic, chaotic, complex. Class 4 is the
  interesting one. Deliberately visual; Wolfram's own notes on defining
  complexity survey the numerical proposals and conclude none of them work.
- **The Principle of Computational Equivalence** says that above a low threshold
  essentially all systems are computationally equivalent. If true, complexity
  comparisons between two systems that have both crossed that threshold have
  almost nothing to measure — the ordering flattens.
- **Computational irreducibility** says the only way to know what a system does
  in `n` steps is to run `n` steps. No cheap complexity statistic can exist in
  general, because it would be the shortcut.
- **Observer theory and the second law** (Wolfram 2023, *Computational
  Foundations for the Second Law of Thermodynamics*) makes the consequence
  explicit: entropy increases because a **computationally bounded observer**
  cannot track the detail. Complexity is not a property of the system, it is a
  property of the pair (system, observer). What looks structured depends
  entirely on the coarse-graining the observer can afford.

> **Here.** Two opposite things, and both are real.
>
> The **tools** are cheap, concrete and half-implemented already — take the
> dimension estimator (done), take the curvature term (one regression coefficient
> away), and note that discrete curvature independently arrives from tradition
> (iii) with a theorem attached.
>
> The **philosophy is hostile to this programme.** `Research.md` §1.2 commits us
> to unbounded growth in realised organisations, which is a number that must
> rise forever; the Principle of Computational Equivalence says that above a low
> bar there is no such ordering. We should not pretend to be neutral about this:
> we are betting against it, alongside Bedau and Standish, who both insist the
> number exists.
>
> But the observer argument has a **constructive** reading and it is the same
> lesson `Literature.md` §6 already draws from Rosas et al. and the map
> equation: if complexity is only defined relative to a coarse-graining, then
> **state the coarse-graining first**, before measuring. Which is a discipline,
> not a defeat.
>
> On standing: the physics is not peer-reviewed in the ordinary sense and is
> contested. Cite the graph estimators, which stand on their own as
> Bishop–Gromov volume comparison; do not cite the physics.

---

## 6. What this changes here

### The finding: this graph cannot build long-range structure
Every edge the engine creates is created in one of two places, and both are
local:

- `_spawn_child` (`GraphOfLifeSimple.py:971`) links the newborn to members of
  `candidates`, which is `[parent] + parent's neighbours` (`:887`).
- the handover loop (`:926`) moves an edge from the parent to the child, and its
  far endpoint is again a parent's neighbour.

So **no edge is ever created between two nodes more than two hops apart.** New
edges close triangles; they never build shortcuts. Meanwhile edges are destroyed
freely — every zero-flow edge is cut each Blotto phase (`:1155`), and every death
takes its edges with it.

The world starts as `nx.watts_strogatz_graph` (`:1560`), whose small-world
property comes entirely from `rewire_p` long-range shortcuts. **Those shortcuts
are a budget that is spent and cannot be replenished.** This is a structural
statement about the algorithm, not a hypothesis about a run, and it is exactly
what `Research.md` §4.5 (`edge_proposal`) was reserved for — now with a reason
sharper than "the graph only erodes".

### The consequence: tree-likeness *is* the extinction risk
`_cleanup_and_redistribute` keeps only the largest connected component
(`:1268`) and kills everything else as `orphaned`. So the graph can never
actually disconnect — the cull enforces it. Which means:

**A bridge with 40% of the nodes on the small side is a 40% mass extinction,
waiting for the zero-flow prune to touch that one edge.**

Bridge balance is therefore not a descriptive statistic here. It is a leading
indicator of population collapse, and both halves of the test are already
recorded per frame: `bridges` and `orphaned`.

### What was measured — `research/pilot_topology.py`

Run against the recorded series, with a Watts–Strogatz graph rebuilt at the
same size and degree as the control. **No conclusions drawn here**; these are
the numbers, and the Theses tab states in advance what would confirm or refute
each claim.

**T1, the shortcut budget.** Confirmed, and by a wide margin. Two long runs
(4,985 and 2,773 iterations, populations growing to 31k and 13.6k) end with
**22–30% of edges being bridges, against 0.0% for the control**; clustering at
0.02–0.06 against 0.24; path length above the small-world ratio. A short run
measured from its beginning moves every one of the six metrics in the predicted
direction, path length included, `rho` between +0.5 and +0.8.

The refinement worth keeping: **the departure happens early.** By the first
sampled window of a long run the graph is already at 19% bridges while the
control is at 0, so a long run's *trend* is a second-order story on top of a
transition that has already finished. Two of the six metrics move the wrong way
across a long run for exactly that reason. Measure the first few hundred
iterations, not the last few thousand.

**T2, fragility leading the cull.** Not supported, and initially not testable.
Correlations sit near zero at every lag with no peak, and the scattered |z| > 3
cells are what 36 tests produce by chance. Two reasons, both now fixed or named:
the series is stored every 4–8 frames so short lags were invisible; and
`bridges` was counted *after* cleanup, in the same frame as `orphaned`, so a
cull that severs a side of a bridge moved both at once and cause could not be
told from effect. The engine now takes the reading **before** the cull, which
makes the question askable — but only for runs recorded since.

One caveat found on the way and not yet fixed: differencing an autocorrelated
series induces negative correlation at lag 1, which shows up as a column
alternating `+,-,+,-` of roughly equal size. That is an artefact, not a signal.
Pre-whitening both sides before correlating is the fix. **Until it is done, an
alternating column is no result.**

### Now measured every frame

| metric | what it is | where |
|---|---|---|
| `cutRisk` | largest share of the population one edge can sever — the 10% rule, as a number | post-cleanup graph |
| `cutRiskBefore` | the same, on the graph before the cull, so it can be read as a cause | engine, per phase |
| `coreShare` | fraction left after peeling degree-1 nodes until none remain: the 2-core against the whiskers | post-cleanup graph |
| `ricciCurvature` | the `r²` term the ball-growth fit was discarding | post-cleanup graph |
| `spectralGap` | λ₂ of the normalised Laplacian — one number for how hard the population is to cut in two | largest component |
| `lightningScore`, `cyclingShare`, `lightningLongest` | how much of a Blotto phase's token flow goes round in closed loops, Σ hops² and the share of tokens in them | game-phase allocations |
| `netLightningScore`, `netCyclingShare`, `netLightningLongest` | the same on flow with reciprocal amounts cancelled, so no two-hop loop can exist and every circuit is real | game-phase allocations |
| `flowImbalance`, `netFlowShare` | the exact ceiling circulation lives under, and how much flow survives cancellation | game-phase allocations |

`bridge_splits()` and `two_core_size()` live in `GraphOfLifeSimple.py` because
the engine needs them on the live graph, and `gol_series` imports them rather
than growing a third copy. The browser mirror is `graphstats.js`, compared
value for value by `tests/test_stats_parity.py`.

### λ₂, and what it settled

**Measured: 0.00086** on a 35,000-node frame of `GOL_26_08_31_n001`. Essentially
zero, and that answers the question it was added for.

The worry was that a Flow-modules partition might be an artefact — the map
equation always returns *something*, and a division drawn through an expander
means nothing. λ₂ near zero says the opposite: this graph has a cheap cut, so
divisions in it are real. Cheeger's inequality makes that formal — `h(G)² / 2 ≤
λ₂ ≤ 2 h(G)` — so a small λ₂ is a *promise* that a good cut exists. It also
corroborates everything else measured here: a fifth to a third of edges are
bridges, `cutRisk` is high, and the graph is core-plus-whiskers. Those are four
readings of one fact.

**The algorithm had to be replaced, and how it failed is worth recording.**
Power iteration separates the top two eigenvalues at a rate set by the space
between them, and this graph has none — the spectrum is a cluster. Its estimate
*halved every time the iteration count doubled*: 0.038, 0.019, 0.010, 0.0045,
0.0021 at 25, 50, 100, 200, 400 passes. Any of those would have been a reading
of the iteration count, and the first one would have been reported as
"moderately connected". Lanczos on the same graphs is exact to the last bit
wherever a closed form exists to check it — complete graphs, rings up to 80 —
and resolves the real frame in 312ms. This is the same trap as the conquest
cycles and the flow modules: a number that looks like a measurement until it is
asked what it would be if the method were run differently.

### Still worth adding, in order of value per line of code
1. **Effective resistance per edge**, summarised by Gini and maximum — Foster
   fixes the total at `n - 1`, so only the spread carries information. Reuses
   `_gini`. Exact computation is too slow per frame at 30k nodes and would need
   the sampled form.
2. **δ-hyperbolicity** on sampled quadruples, and a greedy **treewidth** upper
   bound. Most expensive, and the two whose literature support is shakiest.
3. **Whisker size distribution**, not only the worst one. `bridge_splits`
   already returns every split; only the maximum is currently kept.

### On `edge_proposal`, revised

The first version of this section read the local-rewiring finding as a missing
mechanic. That is probably wrong, and the data is why.

Culls do not arrive as rare catastrophes. `orphaned` is **non-zero in every
single recorded frame** of both long runs. So the punishment for going
tree-like is not waiting to happen — it is already continuous, and the graph
goes tree-like anyway. What is missing is not the pressure but a **response**:
no agent has any action whose effect is "make my neighbourhood harder to
sever". Reproduction is the only structural verb, it is local, and it is cheap.

Which argues for keeping the mechanic local and *costly* rather than adding a
free one: an agent offers a link to a node it can already reach in two hops,
both ends must agree, and it costs tokens. Redundancy becomes something bought
out of the same conserved supply everything else comes out of, and the
question — is resilience worth paying for — becomes one the agents answer
rather than one the rules answer for them. Whether it is needed at all is what
§6.7's measurements are for, so it waits on them.

### And the tie back to the goal
`Research.md` §1.2 adopts *unbounded growth in the space of realised multi-agent
organisations*. An organisation is a subgraph. Whether such subgraphs can exist
at all is a question about the topology they would have to live in — and §3 says
the shape that supports them is a core with structure hanging off it, which is
measurable in three independent ways. Right now we cannot say whether we have
one, because we measure the number of bridges and not what is on either side of
them.

---

## Sources

- [Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters](https://arxiv.org/abs/0810.1355) — Leskovec, Lang, Dasgupta & Mahoney 2008/2009
- [Statistical Properties of Community Structure in Large Social and Information Networks](https://cs.stanford.edu/people/jure/pubs/ncp-www08.pdf) — the WWW'08 version
- [Tree-Like Structure in Large Social and Information Networks](https://www.stat.berkeley.edu/~mmahoney/pubs/treelike-icdm13.pdf) — Adcock, Sullivan & Mahoney 2013
- [Tree decompositions and social graphs](https://arxiv.org/abs/1411.1546) — the follow-up, and the caution
- [Metric tree-like structures in real-world networks: an empirical study](https://arxiv.org/pdf/1402.3364)
- [On the Hyperbolicity of Small-World and Tree-Like Random Graphs](https://arxiv.org/pdf/1201.1717) — Chen et al. 2012
- [Sparsity: Graphs, Structures, and Algorithms](https://link.springer.com/book/10.1007/978-3-642-27875-4) — Nešetřil & Ossona de Mendez 2012
- [Structural sparsity of complex networks: bounded expansion in random models and real-world graphs](https://arxiv.org/pdf/1406.2587)
- [Scale-free networks are rare](https://www.nature.com/articles/s41467-019-08746-5) — Broido & Clauset 2019
- [Rare and everywhere: perspectives on scale-free networks](https://www.nature.com/articles/s41467-019-09038-8) — the reply
- [Cheeger's Inequality and the Sparsest Cut Problem](https://homes.cs.washington.edu/~shayan/courses/approx/adv-approx-17.pdf) — lecture notes
- [Foster's Theorems](https://mathworld.wolfram.com/FostersTheorems.html) — Foster 1949, sum of effective resistances is `n - 1`
- [Sparse expanders have negative curvature](https://arxiv.org/abs/2101.08242) — Salez 2021
- [Community Detection on Networks with Ricci Flow](https://arxiv.org/abs/1907.03993) — Ni, Lin, Luo & Gao 2019
- [Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature](https://www.nature.com/articles/s41467-021-24884-1)
- [Sparse graphs and their Benjamini-Schramm limits: a spectral tour](https://arxiv.org/html/2510.10299)
- [A clarified typology of core-periphery structure in networks](https://www.science.org/doi/10.1126/sciadv.abc9800) — Gallagher et al. 2021
- [Wolfram Physics Project — technical introduction: curvature](https://www.wolframphysics.org/technical-introduction/limiting-behavior-and-emergent-geometry/curvature/)
- [Wolfram Physics Project — branchial graphs and multiway causal graphs](https://www.wolframphysics.org/technical-introduction/the-updating-process-in-our-models/branchial-graphs-and-multiway-causal-graphs/)
- [The Wolfram Physics Project: A One-Year Update](https://writings.stephenwolfram.com/2021/04/the-wolfram-physics-project-a-one-year-update/) — Wolfram 2021
- [Computational Foundations for the Second Law of Thermodynamics](https://writings.stephenwolfram.com/2023/02/computational-foundations-for-the-second-law-of-thermodynamics/) — Wolfram 2023
- [Defining Complexity — history of complexity definitions](https://www.wolframscience.com/nks/notes-10-4--history-of-complexity-definitions/) — NKS notes
- [Computational Irreducibility](https://www.wolframscience.com/nks/p737--computational-irreducibility/) — NKS p737
- [Charting a Course for "Complexity": Metamodeling, Ruliology and More](https://writings.stephenwolfram.com/2021/09/charting-a-course-for-complexity-metamodeling-ruliology-and-more/) — Wolfram 2021
