# Plans for the chapters still to come

Working notes, mine, for every chapter of the book that is planned but not
yet written — what to check before trusting anything, what to find out, what
to watch for, what each needs built. Not for readers; the book says the same
things properly once a chapter exists. Chapter numbers are the book's
(`book/book.json`). Written 2026-10-09; update as chapters land.

Standing rules for every experiment chapter (from the book's method,
Chapter 24 · Meta I, and `research/Research.md` §8):

- Thesis in the plan JSON *before* any of the experiment's runs exist; never
  reworded afterwards; stale chapter numbers get a note after the thesis.
- Thirty seeds per condition unless the question is within-world; measure the
  settled life (iteration 500 on); the dead counted apart — and remember that
  "dead" means "fell to 20 agents" (Chapter 38), so where extinction matters
  run a threshold-0 condition beside.
- Every number gets a null. Every detector gets a null and, where possible, a
  positive control.
- Figures through `book_chapters/`, captions say what every mark is, recipes
  say how to redo it by hand.
- Each chapter ends with what it means for open-ended evolution (Part V) and,
  where it bears on it, for the physics aim (Chapter 7).

---

## Part IV · One change at a time

### 39 · How fast should brains change? (Experiment 9, ready)

**Question.** Is the baseline's birth rate (≈ 3.1 per hundred agents per
iteration) something selection finds, or a mutation–selection balance that
moves with the mutation probability *p*? (Thesis: the balance; pilots at seed 1
gave 2.0 / 2.9 / 8.0 births per hundred at *p* = 0.05 / 0.2 / 0.8.)

**Check first.**
- All 150 runs on one engine (`gol_lab.py analyse` refuses otherwise).
- At *p* = 0.05, did worlds pass through a single-founder bottleneck as in the
  baseline (Chapter 11)? Low variation may leave founders' quirks in place —
  compare with the frozen/teeming split of Chapter 37.
- Extinction per condition, with Wilson intervals; whether the dead differ.

**Find out.**
- Births per hundred vs *p*: monotone? linear in *p* (mutation–selection
  balance predicts roughly linear for small *p*)? a jump (error threshold)?
- Genotype lifetimes, sweeps (`ancestor.moves`), families vs *p* — the
  heredity numbers the OEE ladder (P1) needs. Expect genotypes per agent ≈
  rising strongly with *p* (pilots 0.32 / 0.60 / 0.98).
- World size, Gini, home share, kept share vs *p*. The *p* = 0.05 pilot was a
  small (729 agents), unequal (Gini 0.60) world — is that general?
- Between-world spread (sd/mean) vs *p*: low *p* should spread worlds out
  (towards Chapter 37's changeless worlds).
- The wander: autocorrelation vs *p* (Chapter 37 found changeless worlds hardly
  wander — does the wander scale with *p*?).

**Observe.** Bands over time per condition; dot columns of settled means; a
"births vs *p*" figure with the baseline and the changeless worlds of Chapter 37
as the *p* → 0 end (careful: sparsity 0 is not *p* = 0 — it renames).

**Pitfalls.** At high *p*, genotype numbers churn every game, so genotype-based
lineage statistics measure mutation, not descent — follow lines, not numbers.

### 40 · Brains replaced by their average

**Question.** Does a world need what brains *respond to*, or only their
average behaviour? (The user's idea, 2026-10-09.)

**Build first (engine option, off by default, new strain).** A decision source
`empirical_policy` reading a table measured from settled baseline frames:
- reproduction: P(child | token class, degree class); the child share's
  distribution per token class (keep the atoms at ½ and 1!); P(join a
  candidate); P(hand over a connection);
- game: P(spread); if spread, the home share's distribution per degree class,
  and how stakes spread over neighbours (even, or the observed dispersion);
  P(all-in target = home); the revolutionary share's distribution;
- messages: zero, or draws from the observed message distribution
  (Chapter 19).
Two levels: unconditional (one distribution per decision) and conditional (on
the agent's own token and degree classes). Table measured from the 26 worlds,
iterations 500–2,999, stored in the repo with its provenance.

**Check first.** That the table reproduces the marginals it was measured from:
run a world of such agents and compare the decision statistics with the
baseline's (they must match by construction at the start; drift tells how
much the network shape feeds back).

**Find out.** Agents, connections, Gini, births, cut-offs, wander, bridges,
dimension (Chapter 23) of the empirical-policy worlds vs the baseline. If they
match: brains add only their average (consistent with Chapter 27's even split).
If not: which decisions' dependence on inputs matters — ablate one decision at
a time (only stakes from the table, the rest from brains, and so on).

**Observe.** Whether such worlds die (Chapter 37's chance worlds died in 3
iterations — do these?), and whether they freeze or teem.

**Pitfalls.** The table conditions on states that the policy itself changes;
a mismatch can be a feedback, not "brains matter". Report both levels.

### 41 · What keeps wealth spread? (`allow_revolutions` off)

**Question.** Coalitions decide half of all nodes (Chapter 15). Without them,
does wealth concentrate (the original Experiment 4 thesis), do hubs become
holdable, do partnerships last longer (Chapter 34's *w* ≈ 0.05)?

**Find out.** Gini, top-decile share, kept share by degree class (Chapter 27's
2.2% at 50+ connections), pair persistence (Chapter 34's figure), births,
world size, sweeps. Who wins nodes: largest staker always — does that favour
hoarders?

**Observe.** Any world that "freezes" around a few rich hubs.

### 42 · Meta III

Write once 37–41 are analysed. Re-read the ladder: what moved for P1
(heredity: Chapter 39), for the null (Chapter 40), for structure (41). Update
the obstacles list of Chapter 36 and the order of Part V. Fold in what the six
Part II chapters (18–23) changed.

### 43 · How big should a brain be? (`hidden_layers`) — PRIORITY, run next after 39

**Why it moved up.** Chapter 18: the baseline brain is nearly blind — gain
0.0015 (founders 0.0006), each sigmoid layer passes ×0.23–0.27, the last
(linear) ×1.2. Selection only sets constants; first-layer weights sit at the
mutation–drift balance (0.67 = 0.57 × √1.40) for every input group, noise
included; brains in one world are as far apart as unrelated founders. Every
Part V rule (incumbency, tags, cooperation that pays) presumes brains that
respond. Chapter 19: messages are genotype names — kin recognition is one
comparison away, blocked by the gain. So this is the first obstacle (Meta II's
added callout).

**Conditions (one per row, 30 seeds each, 10,000 tokens, 3,000 iterations).**
- `hidden_layers` = [] (linear: 154 → 45, gain ≈ 1 by construction);
- one hidden layer of 50 (gain ≈ ¼);
- the baseline's five (gain ≈ 10⁻³);
- if the engine allows it without a new strain: tanh or ReLU hidden units,
  or weights scaled so each layer's gain starts near 1 (Glorot/He). Check
  `gol_config.py` for an activation or init option first; adding one is an
  engine change — off by default, new strain, ask the user before.

**Find out.** First, the probes of Chapter 18 on the new worlds: gain, the
three sensitivities (messages, noise, neighbours' wealth), the listen figure
(do input groups separate from the drift balance?), the even-split fallback
share. Then Chapter 19's message stats (does anything other than the name get
said? explained-by-state share up from 3.5%?). Then the world statistics and
the Chapter 34 kin-discrimination ratio. P2 test: cross-time tournament.

**Pitfalls.** Mutation per weight is scaled by fan-in; changing layers changes
how much a "change" does. Report the effective change size. A linear brain's
outputs are unbounded functions of log-tokens: watch for all-in on extremes.
Cost: fewer weights, cheaper per agent (Appendix A).

**Thesis idea.** "With a gain near 1, brains respond: doubling neighbours'
wealth changes the stakes of ≥ 20% of agents (baseline 1%), and stakes depart
from the even split (R² of the random-walk fit, Chapter 27, below 0.46)." 

### 44 · Does size change the dynamics?

Now largely served by Chapter 51 (worlds of 100,000 tokens, 30 seeds, 1,500
iterations) against the baseline's first 1,500 iterations: sweeps, the wander's
time scale, the sawtooth of cuts (Chapter 38), extinction. Decide when 51 is
analysed whether 44 remains a separate chapter or folds into 51/52.

---

## Part V · Towards open-ended evolution

Every chapter here needs an engine option (off by default, its own strain
label, `research/strains.md`). Measurement battery: `research/Research.md` §6 —
heredity half-life, selection ablation (winner by coin), cross-time tournament
(needs a tournament harness: load brains from two checkpoints of one world and
let them play; checkpoints only exist at run ends today — plan checkpoints at
fixed iterations for these experiments: `record.checkpoint_every`).

### 45 · Mutation at replication, and nowhere else

**Build.** `mutate_on_replication`: no change after the game; change only when
a brain is copied — into a child (already) and into a won node (new).

**Find out.** Heredity half-life (distance of a lineage to its ancestor over
time, in weights — Chapter 18's tools — and in behaviour); genotype lifetimes;
sweeps; whether selection now separates good from bad (selection ablation).

**Observe.** Whether worlds freeze (Chapter 37's changeless worlds froze when
nothing varied — but here copies still vary).

### 46 · Can a brain hold its place? (incumbency)

**Build.** The agent's own stake on its node counts double (or a coalition
must outweigh hegemon + home). One parameter, two or three values.

**Find out.** Kept share by degree class (hubs!), pair persistence *w*
(Chapter 34), whether partnerships appear (reciprocity beyond the even split),
inequality, world size.

**Pitfall.** Incumbency may simply freeze the world (no conquest, no
spread) — measure births and sweeps too.

### 47 · Kin that know each other (heritable tags)

**Build.** A few numbers per brain (a tag), copied with it, changed rarely;
each agent sees its candidates' tags (new input rows — changes input width,
so a new strain).

**Find out.** Kin discrimination ratio (Chapter 34: 1.01 in the baseline);
Hamilton accounting Σ(rB − C); whether tag-groups form regions; green-beard
dynamics (tags without kinship — Riolo, Cohen, Axelrod 2001).

**Check.** Tags must not be a covert channel the brains already had: compare
with a condition where tags are random per agent (not inherited).

### 48 · When cooperation produces (mutual flow yield)

**Build.** A connection with tokens flowing both ways in a game yields *y* new
tokens to its ends; every holding decays by a small rate so the total stays
bounded (choose the decay so the expected total stays near *T*).

**Find out.** Reciprocity beyond the even split; defection (one-way stakes
exploiting a partner); whether clusters of mutual exchange form and persist
(P3 candidates); total tokens over time (stability).

**Pitfalls.** Runaway growth; a yield that every pair gets by the even split
anyway (Chapter 34: 58% of mutual pairs exchange 1↔1) makes it a flat subsidy —
maybe yield only on stakes above a threshold, or only on matched stakes.

### 49 · An open world

**Build.** (a) Pieces that separate live on (no cull) — every piece is its own
world; (b) local long-range links: a child may join an agent within *k* steps
of its parent (*k* small), or reach grows with tokens spent (Chapter 7's
locality caution).

**Find out.** Number and sizes of pieces over time; divergence between pieces
(genotype distance, behaviour); whether pieces reconnect; deaths by cutting
vanish — what replaces them?

**Pitfall.** Without the cull, tiny pieces of a few agents may linger forever —
decide how to measure "a world" (largest piece? all pieces?).

### 50 · Meta IV

What moved on the ladder (P1–P4); which mechanic combinations to run next.

---

## Part VI · Towards a physics: space, scale and locality

### 51 · Worlds of 100,000 tokens (Experiment 10, ready)

**Runs.** B1 at 100,000 tokens, seeds 1–30, 1,500 iterations, four workers.
The lab's plan (after fixing its memory fit, 2026-10-09): about 40 h on four
workers, 52 GB of disk (frames and checkpoints), up to 3.7 GB of memory per
run (Experiment 8 measured 2.8–3.4 GB at 102,400 tokens). Disk: 185 GB free
on 2026-10-09; Experiment 9 needs 48 GB — both fit.

The lab's memory estimate had been 27.8 GB a run: the old fit took the
largest ratio of (peak − 300 MB) to brain bytes over all runs, which small
runs (all overhead) inflated to ×68. The lab would have run E10 one run at a
time (≈ 150 h). Now a least-squares line raised to bound every run
(`gol_lab._memory_line`; test in tests/test_lab.py).

**Check first.** Extinction (none expected at this size); the youth (does a
world of 1,000 founders pass through a single-founder bottleneck as worlds of
100 do? — Chapter 11: families fall from 100 to ~20); per-token agents ≈ 0.14
(Chapter 38).

**Find out (thesis, in E10.json: lines take a world over as fronts; the time
to a single founder is 2–5× the baseline's median 175, i.e. 350–875).**
Pilot: E08 at 102,400 tokens reached a single founder at 400, 400, 525
(world_pass caches; 51,200: 275, >600, >600; 25,600: >600, 575, 325). Neutral
coalescent would give ×10 (≈ 1,750), a panmictic sweep ×1.3 (≈ 230). Also:
moves of the common ancestor over 100–1,499 (baseline, lab definition: median
3.5, mean 4.25 over 28 worlds); the front speed (diameter ÷ time) against the
light cone of Chapter 7 (≤ 1 step per iteration); how many lines hold large
regions at once; families per agent. Regions: do families occupy regions
(Chapter 33) of a typical size that does not grow with the world? Space: the
dimension measures of Chapter 23 at scale (→ Chapter 52); distances; cut
sizes (Chapter 38's sawtooth). Scale-free: tails with 14,000 agents per frame
(→ 53).

**Observe.** The sawtooth; whether big cuts remove whole family regions
(selection on regions!).

### 52 · Dimensions at scale

Chapter 23's tools on the 100,000-token worlds: local dimension *d(r)* from ball
growth out to *r* ≈ 30; spectral dimension *d_s(t)* out to *t* of thousands
(random walks on 14,000-node graphs: numpy edge-list propagation, fine);
distance scaling across 3,200–409,600 tokens (Chapter 38). Question: is there a
plateau at 3? Does the dimension depend on scale (CDT: 2 at small, 4 at large)?
Compare with degree-preserving nulls at the same size.

### 53 · Which properties have no scale?

Finite-size scaling: for each quantity *X*, does the distribution of *X/N^a*
collapse across sizes (Experiment 8's ten sizes, Chapter 51's large worlds)?
Candidates: the share of a world cut off per game (Chapter 38: same at every
size), cluster/branch sizes behind bridges, family-region sizes, token tails
(γ ≈ 2.3, Chapter 29), degree tails, loops per agent. Also correlation
functions (Chapter 22): exponential (a scale ξ) or power law (none)? Teach
data collapse with a known critical system (percolation on a lattice) as the
positive control.

### 54 · Local rules only

**Build (three engine options).** (a) Local share-out: tokens of the dead go to
their former neighbours (or stay as a "fossil" node that neighbours can win?);
(b) no global cull: separated pieces live on (same as 49a); (c) local random
numbers: a counter-based generator keyed by (seed, agent id, iteration,
purpose) — Philox-style, so runs stay exactly repeatable (Chapter 8's tests must
pass under the new option).

**Find out.** Each alone vs the baseline: world size, inequality (the lottery
was the poor's basic income — Chapter 26), deaths, sweeps; then all three
together = "the local world", the substrate for 55.

### 55 · How far does a difference travel?

Damage spreading on the local world (54c is a prerequisite: with one global
stream the difference is everywhere at once). Twin runs from one checkpoint:
add one token to one agent; record the set of agents whose state differs
(tokens, brain id, connections) every phase; the maximum and mean network
distance of the difference from its origin vs time → a speed (≤ 4 connections
per phase by Chapter 7's cone); the Hamming distance between the twins vs time
→ chaos (grows) or order (fades) (Derrida & Weisbuch 1986). Many origins,
many times; positive control: a lattice cellular automaton with known speed.

### 56 · A source and a sink

**Design options.** (a) Token quality: tokens carry a "freshness"; a stake
spends it; spent tokens regenerate slowly (or only at "stars"); only fresh
tokens count for winning nodes. (b) Stars: a few fixed nodes that release
tokens at a rate, and a uniform sink (every holding loses a fraction) so the
total is stationary. **Find out.** Entropy production; whether stable
structures sit in the flow (dissipative structures); gradients around stars;
whether life-like persistence (P3) is commoner near sources.

### 57 · Particles

**Detectors (each with a null and a positive control).** (1) Bound tokens:
persistent cycles of *net* flow (Chapter 27: almost none in the baseline) —
track the same cycle over games; (2) groups that outlast their members
(membership turnover vs group persistence — flow modules with a rewired null,
`research/Research.md` §6.4); (3) local beats: power spectra of single agents'
or regions' token holdings, peaks against a random-walk null; (4) moving
patterns: a group whose "centre" drifts through the network.

### 58 · One agent in a sea of average agents

Needs Chapter 40's empirical policy. **Design.** (a) A large world (10,000–
100,000 tokens) of empirical-policy agents, run to stationarity; (b) insert a
probe — an agent with a real (evolved, from a checkpoint) brain, or a small
group of kin — as a leaf of a bath agent, holding the median tokens; (c) many
insertions (different places/times) → distributions. **Measure.** Tokens over
time (extraction rate), survival time, copies of its brain (conquests +
births → invasion fitness), local depletion of the bath. **Define success
explicitly** — tokens, survival or copies differ: a hoarder gains tokens among
even-splitters but spreads its brain by birth only (Chapter 35: conquest
spreads brains 18× more than birth). **Variants.** A "ghost reservoir"
boundary (cheap, open system, breaks conservation at the border); a group
probe (bound state?). **Theory.** Invasion fitness (Metz et al. 1992),
ESS (Maynard Smith & Price 1973), reservoir/grand canonical ensemble.

### 59 · Meta V

What the measurements say about space, scale and matter; which rule changes to
combine with Part V's.

---

## Part VII · The long run

### 60 · A hundred thousand iterations (Experiment 11, planned, not runnable yet)

**Question.** The user's, 2026-10-09: what if there simply was not enough time
to learn cooperation? One world of B1 at 204,800 tokens (seed 1; its first 600
iterations are E08's `B1-204800-s001`), run for up to 100,000 iterations while
healthy. Every 500 iterations: does it behave as expected, stay within the
range of the known, and does anything stand out? Thesis (draft, pinned in
`book/experiments/drafts/E11.json`): time is not what is missing — the wander
levels off, cooperation does not appear, the gain stays below 0.005.

**Numbers behind the plan** (measured 2026-10-09, scratchpad scripts; redo
when building):
- Cost model: 27,312 agents, 24.7 s/iteration, 5.9 GB peak → 29 days for
  100k on one worker. E08's three 204,800-token worlds measured 22–28 s/it
  with four workers busy, peak 5.3–6.5 GB.
- Disk: frames 0.65 MB (phase 1) + 1.58 MB (phase 2, decisions) per
  iteration → 223 GB for 100k; stats 2 KB/row → 400 MB; checkpoint 0.83 GB.
  With windows of 17 iterations per 500-block: ~8 GB of frames.
- The wander (E02, 26 worlds, phase-2 rows from 500 on): sd of block means
  / sd of single iterations at B = 500 is 0.35–0.63; Hurst exponent 0.78–0.89
  over B = 10…500 (nodes, gini, heldHomeShare, maxDegree, distinctBrains,
  cladesInWindow ≈ 0.88; revoltShare, starved, totalFlow ≈ 0.78).
- Naive sentinel (100-iteration blocks, robust z against the blocks of
  500–1,499 of the same world): 5.2% of blocks beyond |z| = 6, 1.9% beyond
  10; three-in-a-row runs common (maxDegree 17×, medianDegree 10×, leaves
  10×). So "stands out" must be judged against the wander — hence the
  variogram in the thesis and the "beyond the whole earlier range by half that
  range, five blocks running" rule, to be calibrated on E02 before launch.
- Cooperation ranges in the 26 baseline worlds: kin ratio 0.89–1.10
  (median 1.01); same two brains one game later 3.3–8.6% (median 5.0%); gain
  0.0011–0.0017 (founders 0.0006).
- E02 settled deaths at 313, 1,547 and 2,184 (plus one at 4): hazard
  ≈ 3.7·10⁻⁵ per iteration at 10,000 tokens → a 10,000-token world reaching
  100k iterations has a few percent chance. That is why there is no cheap
  small-world companion arm; whether a 204,800-token world can die at all is
  itself a finding.

**Build first (only after E10 has finished — item 1 is engine).**
1. *Frames in windows, statistics always* (gol_config, gol_run, gol_record,
   gol_store). `SimConfig.records(i)` and `frames_before(i)` generalised to
   windows (`every`, `last`); INFRASTRUCTURE, so no strain change. The
   recorder observes every iteration's frames in memory; only window frames
   are written. Rows ≠ frames from then on: `_cut_back` by iteration (rows
   carry `iteration`), `_frame` only on rows whose frame was kept. Resume:
   checkpoints only at window ends (`checkpoint_every == every`), and a window
   at least CLADE_WINDOW (8) iterations long, so the families can be warmed
   from it; `previous` = its last frame. Lift the refusal in gol_run for runs
   recorded this way only. Every reader that assumes row *i* ↔ frame *i*:
   build_series (chart from stats.jsonl for such runs), `frame_at` (Phase 2.2
   of the review plan), the viewer's timeline. Tests: windowed run == whole
   run, row for row except `_frame`; its frames are the whole run's frames;
   a run resumed at every block end == one never stopped (the lab will resume
   it 200 times if it advances block by block). Prove B1 unchanged with
   `tools/trajectory_digest.py --stats`.
2. *Blocks and a sentinel* (gol_lab, new non-engine `gol_sentinel.py`).
   Prefer a worker that runs on to 100k, with the lab reading each finished
   block's rows and stopping the worker when unhealthy, over 200 restarts
   (each loads 0.8 GB). Health: extinct; tokens ≠ 204,800 or a statistic
   newly missing/NaN; s/it > 2× the median of the last 10 blocks, twice
   running; memory or disk short within 10 blocks. Stands out (never stops):
   outside E10's band (scale-free stats) 3 blocks running; new level as
   above; cooperation outside the baseline range in a window; gain > 0.005
   at a kept checkpoint. Writes `<run>/blocks.jsonl` and a lab log line per
   block; the Book tab shows the latest. Ring buffer of the last two
   block-end checkpoints; on a flag, pin the one before the flagged block.
3. *Kept checkpoints*: `<run>/checkpoints/<iteration>.npz`, every 10,000
   and pinned ones; listed by the lab; openable by the analysis.
4. *Analysis kind `longrun`*: variograms of the eight statistics (near
   2,000–5,000, far 20,000–50,000, ratio ≤ 2 for ≥ 7); Chapter 34's kinship
   and partners on each window (parameterise `cooperation.kinship` and
   `partners` by iteration range — the Phase 2 Measure registry is the place);
   Chapter 18's gain on 300 agents from each kept checkpoint; block-mean bands
   for everything of Chapter 9.
5. Then move the draft to `book/experiments/E11.json` unchanged and switch the
   chapter's thesis and runs blocks to the generated markers.

**Check first (once it runs).** Frames at iterations 483–499 identical to E08's
`B1-204800-s001` frames 966–999 (cladesInWindow rows will differ: E08 was
recorded before the family-count fix, 5db3860). Throughput and `_peakMB` flat
across blocks (a leak over four weeks shows here first). Two rows per
iteration, no gaps at block boundaries.

**Find out.** The thesis's three numbers. Beyond them: does a world this size
ever die or split for good; how often one line takes over (needs ancestry
across the gaps — either a genealogy file of (genotype, parent, born), about
5,400 new genotypes an iteration, ~10 GB raw, or a live statistic in the
recorder; decide when building); whether the births' decline of youth
(Chapter 12) continues; the gain and the probes of Chapter 18 at each kept
checkpoint; the cross-time tournament (P2) from the kept checkpoints, which
needs a tool of its own (seed a world with brains from two checkpoints).

**Observe.** The variogram curves themselves (levelling, rising, a step); the
block-mean bands over 100k iterations beside E02/E10 bands; every flag the
sentinel raises, with the replay of the stretch around it from the pinned
checkpoint.

**Pitfalls.** A flag is not a finding until the stretch is replayed and looked
at. One world: anything found must be checked on seeds 2 and 3, run to the
same point. Thresholds of the sentinel are tuned before the run, never during
it. Heavy statistics every 25 iterations at 27k agents are part of the 25 s;
check that spectral and box dimension do not dominate.

---

## Analyses that need no new runs (candidates for later chapters)

- How the statistics move together: cross-correlations and lead–lag between
  the stats over time (does inequality lead the number of agents? do cut-offs
  follow births?) — from stats files only.
- Age at the periphery (Chapter 33's suggestion): age against distance from
  the hubs, all 26 worlds.
- Why worlds reproduce less as they age (Chapter 12, Chapter 28): births per
  agent against time, per lineage — a candidate adaptation.
- Do family regions persist as organisations? (Chapter 36's first P3 candidate.)
  Chapter 22: genotypes cluster within 2 steps only (33× chance at 1 step,
  below chance from 4). Next: the same by *line* — agents sharing an ancestor
  25/50/100 iterations back (world_pass family tree) — domain size vs time.
- Why do the richest leaves die? (Chapter 20: richest hundredth with one
  connection dead within 10 games 53% vs 16% of all leaves.) The frames don't
  record each death's cause; infer from the frame before cleanup? (phase-2
  frame is after cleanup — need tokens just before; maybe from `decisions`
  allocations: an agent staked on by nobody starves, else it was cut off.)
- The two-game rhythm (Chapter 20): predict lag-1 vs lag-2 rank correlation
  from the even-split matrix's spectrum on the actual networks (most negative
  eigenvalues of D⁻¹(A+I)); check the hub–leaf explanation quantitatively.
- Dimension over life at fixed *scale* beyond r = 3 needs larger worlds (52).
- Chapter 23 core (2-core) reads lower (2.7 balls, 1.5 walks at 60k agents):
  is the core threadlike at large scales (loops only local)? Count cycle-basis
  lengths; compare the core's spectral dimension with branched polymers (4/3).
