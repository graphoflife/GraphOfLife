# OEE programme — research log (scratch; becomes the book's Part V)

Goal (user, 2026-10-12): an algorithm on GraphOfLife where open-ended evolution
emerges; a statistic that keeps rising (ceiling grows with tokens); network
neither tree nor complete, resilient, large, decentralised, ~3D, few single
points of failure, emerging from local rules (token-limited long-range search
allowed). Pilots by me, big experiments for the user. Teach in the book.

## Tools (scratchpad/oee)
- proto.py   ProtoWorld/ProtoConfig: engine + prototype mechanics (P dict),
             identical to the engine with everything off (equiv.py).
- run.py     one pilot world: series.jsonl (per iteration counters + health
             every 10), ckpt_*.npz every 100 (with founder roots).
- tourney.py cross-time tournament: half the agents get era-A brains, half
             era-B, mutation off, T=20 iterations, share held by each era.
- probe.py   decisions of the agents of a checkpoint (repro share, link, spread,
             home share, revolution share, handover, gain, seek, accept).
- summary.py window means of the series.

## Round 1 (5,000 tokens, 1,500 it): B1 vs brains that see ([16]) x mutation rule
- gain: 5 layers 0.0006, [16] 0.20, [16,16] 0.04, [] 1.0 (founder brains).
- Seeing worlds are fragile: H16-s1 fell to 1 agent at ~550 and regrew at
  ~1,100; H16mor-s1 decayed to 25; H16mor-s2 to 1; H16morsp-s1 fell to 1 at
  ~40 and regrew at ~490. B1 worlds steady at ~1,000 agents.
- Mutation only at birth (morsp): s1 froze — one genotype from ~800 on, no
  births, every node retaken by its own kin each game, ages climbing to 650.
  Longer lives as stasis, not progress.
- Collapse mechanism in H16mor: births go to zero (median agent's repro share
  0) while a few deaths per iteration continue — the world drains.
- Every evolved world (blind or seeing): the median agent never has a child,
  spreads its stake, keeps ~40% at home, flags all of it revolutionary.
  Conquest is the replication channel; births are a minority's act.
- Conquest churn: 40-90% of nodes change brain every game, ~all to kin.
- Tournament H16morsp-s1 (eras 600..1500): 900 beats 600 (0.75), later eras
  only 0.52-0.55 against 900/1200.

## Ideas queue
- incumbency (home stake counts h times) → births become the replication
  channel, lifespans meaningful, small stakes become gifts. Round 2.
- juvenile protection (newborn untakeable J iterations). Round 2.
- memory inputs (what each neighbour staked on me) → reciprocity, deterrence.
- seek (token-limited walk + mutual consent) → redundancy, partner choice.
- mutual-flow yield financed by a holding tax → cooperation pays.
- growable brains → possibility space that can grow.

## Round 2-3 and the window pilots (5,000 / 3,000 tokens)
- Incumbency 2 (home stake counts double): churn falls (15-25% of nodes
  conquered per game instead of 40-90%), populations survive and grow, births
  return as the replication channel. But networks become tree-like (mean degree
  2.2-2.8, articulation 30-50%) and H16inc2all-s1 collapsed at ~900 through
  repeated huge cuts (669 agents orphaned in one game).
- Tournament H16inc2all-s1: 200 beats 100 (0.93), 300 beats 200 (0.71) — then
  500/700/900 LOSE to 200-300 (0.25-0.37). The world went peaceful (conquest
  1-3%), defences decayed; old aggressive brains win. Elo vs the past can fall.
- Tournament B1-s1: 100 beats founders 0.74; from 200 on every pair ~0.50 —
  the baseline stops improving after ~200 iterations.
- Mutual-stake yield (tax 5%): stars (max degree 36-90% of agents) and
  collapse — the yield accrues to whoever has most mutual links. First version
  also taxed agents' last token (bug, fixed).
- Seek walks: few links made (1-10 per iteration), little effect.
- Window 4/16 phases: early networks much better (2-core 0.8-0.95, degree 4-6)
  but the population explodes to ~1.3 tokens per agent and conquest vanishes;
  leaves remain (16-19%) — not newborns but agents whose links lapsed (1-token
  agents can keep one link a game).
- Minimum child endowment 4/8: fewer, richer agents; health still poor.

## Space alone (tissue*.py): what network do births and deaths make?
No brains: births (child joined to parent + k of its neighbours), random deaths,
the cull. 1,500 agents.
- k=0..3, no repair: big single cuts always (node cut 0.17-0.44) — local growth
  plus random death makes a tree of blobs; nothing closes large loops.
- Full path repair (all neighbours of the dead joined in a path): node cut 0.006,
  but dimension 4.5 (contraction).
- Partial repair (probability q): cuts stay large.
- Walk links (p 0.02-0.05, L 4-16): node cut 0.05-0.34.
- **k=2 + neighbourhood repair (r=1: only groups of the dead agent's neighbours
  that are not linked to each other are joined, least-connected members
  first): degree 4.7, 2-core 99.5%, 3-core 90%, node cut 0.5-0.9%, edge cut
  0.5%, dimension 3.5-3.65, max degree 1.4-2.2%.** Two seeds. This is the user's
  network, from two local rules.
- Repair keeps every dead agent's neighbourhood in one piece, so the world can
  never split from a death; with safe pruning (an unused link lapses only if its
  ends share a neighbour) it cannot split from pruning either: the cull never
  fires.

## The game on the new space (p, q, a, g, v, k, kk)
- birth_links=2 + repair + safe_prune(1): the world never splits (orphaned = 0)
  but shrinks: starvation 30-60% a game. Diagnosis: the starving are newborns
  (46% age 0, 83% age <= 2) with 1-2 tokens who stake everything away —
  "kamikaze children": a parent spends a child as a missile (its stake can take a
  neighbour for the lineage). Each death triggers repairs -> degree climbs to
  11-14 -> stakes thinner -> more starvation.
- newborn_home (first game at home) only delays the kamikaze by a game.
- Without incumbency the worlds collapse into a few rich agents; with
  incumbency 2 into a hoarding "queen" (one agent with 57% of all tokens, staking
  everything at home, unassailable, reproducing; everyone else holds 1 token).
- safe_prune(1) can thin the mesh into a cactus (triangles glued at single
  agents = cut vertices); safe_prune(2) (two common neighbours) cannot.
- Children linked to the two neighbours the LINK head likes most -> hubs (max
  degree 30% of agents): preferential attachment. Random choice + valence fix it.
- keep_one (last token stays home): no starvation at all; with cheap births the
  population fills up to ~1.2 tokens per agent and freezes (no conquest).
- keep_one + min_child 10 + valence 8 + max_age + incumbency 1.5, 10k tokens:
  3,000-5,000 agents, no starvation, no cuts, conquest 6-10% a game, births
  2-3%, gini 0.5-0.6. Running.
- m=8 messages without the pre-pass: half the cost per agent (0.26 vs 0.53 ms).

## Why the game stops improving (2026-10-12, evening)
- Elo against the past, v3fix-s1 (tamed world, eras 0-1000): 0, +133, +160,
  +164, +171, +166, +137 — a plateau from iteration ~200, like B1 (+283, +329,
  +345, +327, +344, +320). v3grow-s1: +128, +130, +128, +114, +125, +113.
- Which inputs change an evolved agent's stakes (sense.py; agents with >= 3
  tokens only — most agents hold 1-2 and have no choice to make): redrawing the
  noise changes 25-58% of decisions; own state 2-7%, the neighbour's state
  4-23%, messages 2-23%, memory of last game's stakes 11-24% (more in worlds
  with a yield).
- Reading: in a Colonel Blotto contest the best play is mixed — be
  unpredictable. Evolution finds randomisation (the noise inputs are its dice)
  within ~200 iterations, and nothing beats a good mixed strategy, so nothing
  more is selected. A contest game alone has no headroom; information has to
  pay somewhere — cooperation (who reciprocates?) or coordination.
- Brain size (grow brains with a neutral shadow): v2 (handover on) active rose
  ~1.4 units above the shadow in 2 seeds; v3 (handover off), 4 seeds, active ~
  shadow. A rising mean size is mostly the left wall: a size bounded below
  drifts up with nothing selecting it. The shadow is what shows it.
- Tamed world v3 = MOR + incumbency 1.5 + birth_links 2 + mend + prune_needs 2
  + valence 8 (chosen links) + keep_one + max_age 100 + min_child 10 +
  handover off, [16] brains, 8-number messages, no pre-pass: ~1,900 agents at
  5k tokens for 3,000 iterations, 2-core 100%, node cut 0.02, edge cut 0.01,
  max degree 1%, mean degree 4.3, dimension 3.3, conquest 6-9% a game,
  births 1%, every death by age.

## Zombies, and what keep_one cost (2026-10-12, late)
- Tokens per agent in the tamed world (v3fix-s1 at 1,500): 75% of agents hold
  exactly 1 token, 8% hold 2; the game is played by a fifth of the world. Yield
  worlds: 80% at 1. B1 (blind): 1% at 1, 43% at 3-5, 37% at 6-10.
  Reason: keep_one makes all-in attacks free (an agent cannot die of them), so
  seeing brains stake everything but the protected token on one neighbour;
  most agents are left with that token and nothing else.
- Without keep_one (same rules otherwise) worlds starve out: 1 agent holding
  all 5,000 tokens by iteration 300 in both seeds, 58-138 agents with a yield.
  Without noise inputs and without keep_one: 19 and 116 agents at 1,000.
- keep_frac f (at least ceil(f t) tokens stay home) as a middle way: running
  with f = 0.3 and 0.5.
- Cross-size: era-800 brains of the 5,000-token world hold 0.531 (0.47-0.60
  over six contests) against era-800 brains of the 1,250-token world, in both
  worlds' arenas: a hint, one run each, that the bigger world evolved the
  better brains. A design for the user: many seeds, sizes 1k-100k tokens.
- Elo vs founders, 1,250 tokens: +305 at 50, plateau +330 from 300 (faster and
  higher than at 5,000 — Elo against one's own founders does not compare
  across sizes; cross-size contests do).
- Yield 0.03 (keep_one world): gross circulation (cycling share) rises over
  time (0.014 -> 0.134 in s2), mutual pairs stay few (most agents have one
  spare token).
- Next: support stakes (allies can help a neighbour hold its node; support can
  never take a node) — cooperation with a value of its own, no yield needed.

## Every depth pilot stops too (2026-10-13)
Cross-time tournaments (eras 0, 100, 200, 500, 1000; T = 20; one coin, both
worlds), Elo against the founders, and the step score (each era against the one
before it):
- U/support-s1 (support stakes, keep_frac 0.3): +155, +194, +319, +337; steps
  0.80 0.59 0.72 0.61 — the only world still beating its predecessor at 1,000.
  Seed 2: +342 at 100, then flat (steps 0.87 0.52 0.47 0.47).
- U/nodice (no random inputs): +134..+184 then flat in both seeds.
- X (3,000 tokens; memory inputs; control / memory_len 6 + grow brains / yield
  0.05 / fog): plateaus at +100 to +250 from iteration 100-200 in all eight;
  late steps 0.44-0.59 except memgrow-s2 (0.64, 0.59) and yield-s2 (0.76 last).
- Support is selected against: support tokens fall from ~200-300 to ~20 per
  iteration (of 5,000) by iteration 600 — a gift with no surplus (support counts
  1x, the home stake 1.5x).
- Group sizes (connected clone domains, kin-within-4 domains) SHRINK over time
  in every pilot; net lightning (cycling share of the net flow) stays ~0.
- Fog of war ends conquest (0.2-0.4% of nodes a game) and with it evolution
  (1-2 mutations per iteration).
Reading: the plateau is the game's — a local Blotto has a mixed equilibrium —
not the brains'; more sight, memory or capacity cannot help. Cooperation needs
a surplus of its own.

## Batch A: cooperation with a surplus (launched 2026-10-13; data lost)
proto.py gained `allies` (ALLY head; mutual consent; stakes pooled by side —
owner + its allies vs. other bidders grouped by alliance; an ally can never
take an ally's node), `kin_pool` k (the same pooling for genotypes within k
mutations) and `support_gain` (support counts g times in defence), each with a
`same_side` input row; run.py records sideDomains/sideLargest/sideMean.
Smoke test at 1,000 tokens, 40 iterations: allies -> 421 allied pairs among 192
agents, one bloc of 165, conquests 4 (control 43): a giant peaceful bloc forms
at once. Batch A (ctrl/allies/kin/gain x seeds 1-2, 3,000 tokens, 1,000
iterations, revolutions off everywhere) was running when a reboot wiped /tmp
(2026-10-14): rerun it from research/oee/pilots.

## Experiment 13, analysed (2026-10-14)
Thesis holds (8/10 MOR seeing worlds extinct, 0/10 baseline below 50). But no
world of any condition passes the health bar, and the networks of the seeing
worlds are not clearly worse than the baseline's; what kills worlds is the
steady cull (1.9% of agents per iteration in the baseline from 500 on, 5.3%
with one hidden layer) against births. Births probe (era brains in the
founders' world, 10 iterations, 2 draws; oee.children): founders 73-75%, later
eras 1-17%; in 6 of 12 extinct worlds the last brains had no child at all;
surviving one-hidden-layer worlds' last brains 7-43% (median 17%) vs baseline
0-13% (median 3%). Seeing makes reproduction a real, losable decision.

## Experiment 14, first look (2026-10-14, partial)
B2 ("all the rules"), ten worlds at 5,000 tokens to 1,000: 2-core 1.000, cut
risk 0.000, hub 2-3%, 5.0-5.8 links, no agent cut off from 200 on; dimension
3.45-3.97 in nine worlds and 4.19 in seed 5 (above the bar's 4). Handover put
back: every world fails (2-core 0.28-0.87, cut risk 1.2-6.8%).

## Batch A at 10,000 tokens (2026-10-14; the author: every simulation ≥ 10,000 tokens from now)
Rerun of batch A (ctrl / allies / kin_pool 2 / support_gain 2, seeds 1-2, 1,000
iterations, revolutions off; data in GraphOfLifeRuns/.pilots/A10k). The 3,000-token
runs (A3k) were stopped and kept only as hints.
- **Alliances grow.** In allies-s1 the founders' random ally decisions make one
  bloc of 54% of the agents at iteration 60; it decays to 15% by 310, then the
  largest alliance grows steadily: 28% (360), 52% (560), 71% (760), 83% (960);
  allied pairs 1,450 -> 7,020; conquest falls to ~2% of nodes a game. Seed 2:
  64% -> 19-22% at 370-490 -> 46% at 910-970. Same shape, slower.
- Support at gain 2 is still given up (631 -> 17 tokens a game in s1).
- Kin pooling: peace (conquest 1.6%), kin sides of 30-80 agents.
- Coin tournaments (eras 0/100/200/500/1000, T 20): ctrl +126 +204 +230 +196;
  gain +222 +294 +306 +284; kin plateau from 200 (1000 vs 500: 0.494); allies
  +113 +127 +135 +143 — the only one still creeping up (steps 0.51-0.53, 1000
  vs 100: 0.566).
- A coin per agent breaks up the alliances whose advantage is spatial, so
  tourney.py gained `--layout halves` (each era a contiguous half, BFS from a
  random agent); running with T 50.
- Next (batch AL, running): allies at 10k for 3,000 iterations (s3, s4) and at
  20k for 2,000 (s1, s2): does the bloc reach the whole world, what happens
  after, and does it take longer in a world twice the size?

## Batch AL and N0: alliances over 3,000 iterations, at 20,000 tokens, and without the home advantage (2026-10-15)
- AL (proto-B2 + memory + allies, revolutions off): the largest alliance reaches
  0.99/0.97 of the world at 2,000 (10k, s3/s4) and stays; 0.91/0.96 at 2,000 in
  the 20k worlds (s1 slower: half the world first at 720, s2 at 120). Allied
  connections 0.98 of all at 2,000 (10k-s3). Conquest 1.8-1.9 per 100 agents a
  game (500-1,000) -> 0.2-0.4 (last 500). Kin groups (4 mutations) stay at 2%
  while the alliance holds 99%: the habit spreads, not a family.
- Tournament AL 10k-s3 (coins, T 20, eras 0/500/1000/1500/2000): +178 +198 +190
  +209, steps 0.74 0.53 0.52 0.52.
- N0 (the same with incumbency 1): largest alliance 0.37 / 0.71 at 1,000
  (against 0.83 / 0.44 with 1.5); conquest 6.0 / 3.1 (against 2.2 / 3.4).
  Alliances form without the home advantage; two seeds cannot tell the sizes apart.
- Health in every allied world (last 500): 2-core 1.0, no cut, hub <= 1.6%,
  4.9-5.5 links, dimension 4.1-4.8 (above the bar's 4; highest at 20k).
- Written up as Chapter 56 (Alliances), figures in book_chapters/pilots.py.

## Batch N: rules without numbers, B3 (2026-10-14/15)
B1's game (incumbency 1, no keep rule, no age limit, min child 1, any unused
link lapses, no birth links, no valence) + one hidden layer of 16, MOR, mending,
memory; B3a with allies, B3b with allies + birth_allies, B3c without allies.
- s1: B3b and B3c died out (to one agent by iteration ~80 and ~120); B3a lived as
  a single alliance (side 1.0) on a near-tree: 3,342 agents, 2.09 links/agent,
  dimension 1.1, 71% of agents articulation points, worst single-agent cut 31%;
  no conquest; deaths ~ births ~ 10% of agents per iteration, by being cut off.
- s2 (running): all three alive at 67-200 iterations, a world of mayflies: mean
  age 1.4-8, a third of the agents starving each game, births half the world.

## Batch S and G: the original switches, and two other contests (started 2026-10-15)
Engine mode (`--proto engine=true`: the engine's own phases, only bookkeeping;
equiv.py checks it identical to the engine under B2 and three switch variants).
10,000 tokens, 1,000 iterations, ckpt every 100, seeds 1-2 (jobs_S.txt):
S6 B1, S7 B1 without revolutions; S0 B2, S1 B2 without revolutions, S2 B2 +
gifts (allow_gifting, prune_after=reproduction, inactive_window=iteration), S3
the same with prune_needs 0 (a link must carry a stake or a gift), S4 prune_needs
0 alone, S5 redistribution by_tokens; G1 B2 without revolutions + a lottery
contest (contest=lottery: a node goes to a staker drawn in proportion to its
weighed stake), G2 B2 without revolutions + stakes pooled by genotype
(pool=genotype, plus the input row "same genotype").

## The dice were there from the start (2026-10-15)
sense.py (share of agents with >= 3 tokens whose stake division changes when
one input group is perturbed) on the founders, iteration 0, of three 10k worlds
(A10k ctrl-s1, allies-s1, AL allies10k-s3): noise 0.58 / 0.54 / 0.49, own state
0.20 / 0.22 / 0.14, its state 0.19 / 0.20 / 0.13, messages 0.42 / 0.40 / 0.34,
memory 0.35 / 0.40 / 0.37. Evolved (iteration 1,000; 3,000 for AL): noise
0.38 / 0.47 / 0.58, own state 0.09 / 0.01 / 0.04, its state 0.10 / 0.07 / 0.09,
messages 0.15 / 0.11 / 0.16, memory 0.30 / 0.19 / 0.18. Kin pooled 1,000:
noise 0.58, own 0.24, its 0.17, messages 0.26, memory 0.15.
=> Random brains already roll the dice; evolution does not add randomness, it
turns down everything else (own state x2-20, messages x2-4). Peaceful allied
worlds keep the dice as much as the warring control, where randomising has no
strategic value: the dice may be kept because nothing selects against them.
Chapter 50's "evolution finds randomisation" needs this baseline; the lottery
contest (G1, where dice are useless) is the test.

## The invasion test (2026-10-15; manual.py, invade_B2_2000.json)
E16's B2 worlds s001 and s002 at iteration 2,000, a 5% patch (177 / 153 agents)
given a strategy, 300 iterations, seeds 1-2; control: frozen copies of one
resident brain in the same patch.
- s002: factory 0.477 / 0.515 of the world at 300 (still rising), raider
  0.395 / 0.500; resident control 0.070 / 0.008.
- s001: factory and raider die out (by 135-196 and 148-175 iterations); resident
  control 0.080 / 0.049.
- turtle dies out everywhere within ~100 iterations (old bodies die of age; it
  never conquers and rarely breeds).
=> In one of two evolved worlds a simple rule (all you can stake on the poorest
non-teammate neighbour you are sure to take, else all at home) takes half the
world: the evolved play is not "the answer". The pipeline adds little over the
raider. Residents of s001 and s002 look alike on the probes (home share ~0.8,
all-in mode, 19-21% of divisions change when a neighbour's message is replaced
by the strategies' signature). Running: all ten E16 worlds (raider, factory,
resident; invade_B2_2000_ten.json) and a lone raider without password (lone;
invade_B2_2000_lone.json).

## Dice in a lottery (2026-10-15)
sense.py, noise share: B2 (S0-s1) founders 0.66, 500: 0.44, 1,000: 0.42; lottery
(G1-s1) 500: 0.67, 800: 0.57; genotype pooling (G2-s1) 500: 0.62, 900: 0.57.
Under a lottery contest randomising is worthless, yet the brains roll the dice
as much as under plurality without revolutions, the right control (S1-s1:
founders 0.62, 500: 0.59; G1-s1 at 1,000: 0.48; G2-s1 at 1,000: 0.54). B2 with
revolutions (S0) rolled them less (0.42-0.44), one seed each. With the founders' baseline (0.49-0.66) this says the
dice are inherited, not discovered: Chapter 50's mixed-strategy reading is not
what the probes show.

## The password is the edge (2026-10-15)
Lone raider (the raider without password: every neighbour a stranger, no
message, children joined to nobody chosen), worlds s002 and s001 at 2,000, 5%
patch, seeds 1-2: dies out in all four plantings within 96-130 iterations, also
in s002 where the raider with password took 40-50% of the world. A raider
without a way to tell its own raids its own: its poorest neighbour is often its
own newborn. Evolution would have to find the signature and its reading at once
(a green beard), which no single mutation does.
Next (running): G3, B2 without revolutions + kin_sense (one input row: is this
candidate's brain identical to mine? no pooling), seeds 1-2 (jobs_G3.txt).
Compare S1 (no sense), G3 (sense), G2 (sense + pooling).

## Number-free candidates, first smoke tests (2026-10-15; one seed, 5,000 tokens, 150 iterations)
proto `fission` (birth by division: a brain that would give at least half its
tokens divides into two equal halves; less is no child), engine mode, one
hidden layer of 16, MOR, mending, revolutions on.
- division alone (+ gifts, every unused link lapsing, nothing kept, 3,000
  tokens): starves out by 60 iterations (32 agents): division stops missiles
  but not agents staking themselves to death.
- division + half kept at home + gifts/lapse: lives (4,228 agents at 150),
  median token 1 (half of one token is one: a world of single-token agents that
  can never divide), births falling (947 -> 278 a game), no starvation, 2.9 links.
- division + half kept at home + B2's space (birth links 2, prune_needs 2): births
  stop (median token 1), nobody dies (no age limit): a frozen world, mean age
  rising 21 -> 132.
=> Number-free replacements must also stop tokens fragmenting into units and
must keep turnover going once nobody dies of age.

## B1 without revolutions keeps improving? (2026-10-15, lead)
Coin tournament (T 20, eras 0/100/200/500/1000) of S7-B1norev-s1: +145 +79
+177 +275; 1000 vs 500 = 0.726 (0.90 0.63 0.68 0.70), 1000 vs 200 = 0.819 (all
four > 0.5). B1 with revolutions (S6-s2): +352 +378 +352 +312, 1000 vs 500 0.414.
B1 without revolutions is a mayfly world (84 of 100 nodes change hands a game,
28 births per 100 agents an iteration, mean age 7, lifetime 3; dimension 2.1, a
worst cut of 30-34%): many more generations per iteration, and every brain
mutates after every game (no MOR). Single contests are noisy (0.02-1.00).
Checking: S7-B1norev-s2 moved to the front; a long run, seed 3, 3,000
iterations (GraphOfLifeRuns/.pilots/L/B1norev-long-s3).

## A single raider never starts (2026-10-15; invade_B2_2000_single.json)
World s002 at 2,000 (where a 5% patch of raiders takes 40-50%), a patch of ONE
agent, seeds 1-20, 200 iterations: the raider dies out in 20/20 (lived 2-180
iterations, peak 1-28 agents, median life ~34); a frozen resident in 20/20 too
(lived 2-134, peak 1-5). The raider grows larger before it dies, but never
establishes: an Allee effect. A raider needs a crowd of teammates and a way to
know them - three things at once that no single mutation makes.
probe.py (eras 0/500/1000): B1 without revolutions (S7-s1) repro share 0.50 /
0.50 / 0.09, links 0.75 / 1.0 / 1.0, spread 0.45 / 0.98 / 0.99, home 0.29 / 0.32 /
0.31, handover 0.50 / 0.97 / 0.50; B1 with revolutions (S6-s1) repro 0.50 / 0.00
/ 0.00, revolutionary share 0.5 / 1.0 / 1.0, handover 0.5 / 0.0 / 1.0. Gain
0.001-0.002 in both (blind brains): what changes between 500 and 1,000 without
revolutions is a few constants (how much a child gets, what it inherits).

## Ten worlds (2026-10-15; invade_B2_2000_ten.json, seed 1 each)
End shares at 300 (raider / factory / resident): s001 0 / 0 / .080, s002 .395 /
.477 / .070, s003 .616 / .004 / 0, s004 0 / 0 / .026, s005 0 / 0 / .061, s006 0 /
.428 / 0, s007 0 / 0 / .084, s008 0 / 0 / .001, s009 0 / 0 / 0, s010 0 / 0 /
.038. Three worlds of ten fall to one of the two (s003 to the raider only, s006
to the factory only); seven resist both. Weak points differ between worlds.

## Family tags (2026-10-15; proto tag_rate)
A tag carried with every brain; a copy keeps its parent's tag; a brain that
mutates takes a new tag with probability tag_rate (a mutation parameter), so a
family is all genotypes since one tag. Smoke test (3,000 tokens, 60 its,
tag_rate 0.05): families of mean size 29 (kin_sense by family) and 68 (pool by
family) against clone groups of 3-7. Pilots running: G4 (family sense) and G5
(stakes pooled by family), B2 without revolutions, 10k, seeds 1-2 (jobs_G45.txt).
The question: given a password and a crowd for free, does evolution find the
raiders' play, and does it keep going?
S7-B1norev-s2 tournament: +161 +234 +383 +303 (100/200/500/1000); steps 0.68 0.62
0.66 0.40; 1000 vs 500 = 0.395. Improvement goes on to 500 (later than B2's 200),
then falls back. Contest shares within a pair range 0.09-1.00 (a mayfly world).
B1 without revolutions evolves longer, not without end; the long run decides.
Switches, seed 1 (1000 vs 500): gifts 0.54 (+263 at 100: more to learn early),
all lapse 0.57 (+520 at 100), gifts+lapse 0.47 (noisy, breaking world), estates
by wealth 0.50; B1 0.54 (s2 0.41).

## The long run and the family tags (2026-10-15, afternoon)
- B1norev-long-s3, 3,000 iterations: improvement to 500 (500 beat 0/100/200 in
  11 of 12 contests), then back (1,000 vs 500 = 0.32, 1,500 vs 500 = 0.11). The
  world shrank from ~1,000 agents (1,000-1,750) to a few dozen after 2,000 (13
  at its smallest, 44 at 3,000): extinct by the lab's rule. Contests in the
  eras 2,000-3,000 ended with 9-40 agents; their Elo (+551 at 3,000, residual
  2.3 logits) is meaningless. The B1-without-revolutions lead is withdrawn:
  longer evolution (to 500, once to 1,000), not open-ended.
- G4 family sense: family mean 30-39, largest 3-12% after 500, conquest 8 per
  100; steps after 200: 0.50 0.48 / 0.47 0.54. G5 pooled by family: family mean
  153-491, largest 57% at 410 -> 12% at 710 -> ~30% (s1), fighting 1.3-1.6;
  steps after 200: 0.53 0.43 / 0.54 0.52. Families rise and fall, brains stop.

## Experiment 9: what the brains use, by rate (2026-10-15; probe_lab.py, iteration 1,500)
Share of agents (>= 3 tokens, of 400 sampled) whose division of stakes changes
when one input group is perturbed, seeds 1-3, rates 0.05 / 0.2 / 0.8:
own state .105 .064 .084 | .077 .027 .034 | .077 .044 .096
its state .073 .073 .079 | .066 .075 .069 | .146 .074 .090
messages  .145 .118 .274 | .264 .156 .154 | .146 .125 .198
noise     .411 .364 .300 | .538 .408 .314 | .338 .346 .316
founders (0.05, s1, iteration 0): own .141 its .200 messages .435 noise .653.
No rate keeps brains less deaf; faster change rolls the dice a little less.
Groups of clones at 1,500 (seed 1): mean 8.2 at 0.05, 3.0 at 0.2, 1.2 at 0.8.

## Experiment 9 analysed (2026-10-15)
book_chapters/rates.py: cross-rate contests at 1,500 (rates-cross, 200 contests,
gol_tournament.contest with brains from two runs), steps per world
(tournament-steps, eras 0/100/200/500/1000/1500, against the previous),
clone groups at 1,500 (clones-1500). Every pair ties (medians 0.49-0.52), every
rate stops by ~500 (1,500 vs 1,000: 0.49-0.51); worlds differ in genes (0.20 ->
0.90 genotypes per agent), clone groups (~8 -> 1.2) and size (3,730 -> 2,600).
Thesis held.

## Would we notice open-ended evolution? (2026-10-15, evening; the author's question)
- Power of the yardstick (E16 B2): 2,000 vs 500 median 0.482, SE 0.005 across
  worlds (one contest +-0.046, a world's two-contest mean +-0.033); 95% upper
  bound 0.492: between 500 and 2,000 B2 did not improve, not even slowly.
  Beyond 2,000, and in contests longer than 20 iterations, nothing is known.
- No Red Queen signature: a later era's share at home minus away, pairs from
  200 on, mean -0.006 (positive in 46 of 100 pairs; 39/100 without dice).
- Exploitability (research/oee/checks/exploitability.py): per agent, the
  nodes it would hold after one game with its best reply to the others' play
  (K = 8 draws of the dice, everyone else as drawn), minus what it holds with
  its own play. B2, median of 10 worlds: 0 0.87 | 100 1.16 | 200 1.49 | 500
  1.33 | 1,000 1.21 | 1,500 1.26 | 2,000 1.20 (holds 1.00 -> 1.23 -> 1.13; 87%
  of agents gain >= half a node at 2,000). Without dice: 0.94 -> 1.75 (200) ->
  1.33. Evolution moves away from a one-game equilibrium, then stays put.
- A continuation from a lab archive is exact (run.py --from RUN:IT; 100
  iterations from B2-10000-s001 at 1,900 identical to the lab's in agents,
  links and Gini).
- Long runs started (jobs_L.txt, GraphOfLifeRuns/.pilots/L): the ten B2 worlds
  of E16 continued 2,000 -> 10,000 (checkpoints every 500, the lab's archives
  0..2,000 copied in), and B1 at 10,000 tokens, seeds 1-2, 10,000 iterations.
  Then: tournaments 2,000 vs 10,000 etc. (detects drift of ~2 Elo per 1,000
  iterations), exploitability at 10,000, contests of T = 100.

## Lifetime contests: evolution did NOT stop (2026-10-15, evening)
research/oee/checks/long_contests.py, E16's ten B2 worlds, contests of T = 100
(an agent's lifetime) instead of 20, once in each era's world:
- 2,000 vs 500: median 0.734 (0.65-0.83), 10 of 10 worlds above a half
  (0.700 in 500's world, 0.771 in 2,000's); token share 0.78.
- 2,000 vs 200: 0.820 (0.75-0.92), 10/10.   200 vs 100: 0.749, 10/10.
- At T = 20 the same pair 2,000 vs 500 is 0.482: the later brains start a little
  behind and overtake over a life.
- Controls (contest_controls.py, T = 100): era 2,000 against itself 0.476
  (0.44-0.54) - a tie; labels swapped (2,000 as A, 500 as B): B holds 0.257.
=> The 20-iteration yardstick is blind to what pays over a lifetime. Chapter
50's "evolution stops by 200-500" and every pilot verdict drawn from T = 20
tournaments must be re-examined. Running: steps at T = 100 (0->100 ... 1,500->
2,000, and 1,000->2,000), long_steps.json.

## Structures and help: gliders, rings, feuds (2026-10-15, late; the author's question)
"Find a new way to define cooperation; maybe there are cyclic structures, like
gliders in the Game of Life, harder to spot because they depend on the
topology." Read the record of play of E16's ten B2 worlds and B1's first ten
baseline worlds, games 1,000-2,000 (every game recorded with decisions), and of
alliance pilots played on 200 iterations from a checkpoint with their games
recorded (research/oee/checks/pilot_frames.py; proto.py now keeps
last_winners). Tools in research/oee/checks: structures, gliders, rotations,
circulation, kin_flows, decisive. Chapter 58 (new; Meta IV -> 59, Part VI/VII
+1, research/tools/renumber-2026-10-15b.json).
- Chains of takeovers (a move = a takeover with >= half the taker's tokens; a
  chain = moves each made from the node the last took). B2: 99.6% of takeovers
  are moves; chains >= 3 are 23x chance (18-54), 31% of takeovers, longest
  18-32; mover keeps its node 93% (PUFFERS: a lineage copying itself outward),
  packets 3 -> 2 tokens. B1: 11% of takeovers are moves; chains >= 3 11x chance
  (6.5-26), longest 69-175; mover keeps its node 9%, starves 8% (GLIDERS), packets
  grow 18 -> 82 tokens (rich nomads). Back to the node before (blinker) 1.37x
  chance in B2, 1.27x in B1; triangles rare / at chance. Reach = a random walk's
  (0.8 B2, 1.0 B1): no direction. Guns bursty (var/mean 2.6, 13), no period.
- Musical chairs: B1 72% of takeovers by agents losing their own node the same
  game; swaps at chance (s001: 17,150 vs 17,096); B2 swaps 0.65x chance, no
  rotations.
- Channels: B2 recur 1.5x chance; B1 at chance (stakes on ~all neighbours).
  Lasting mutual stakes are FEUDS in B2 (kept 0.80 vs 0.89, 10/10 worlds); B1
  no effect; controls 0.49 vs 0.86-0.88. Among allies 85%/56% of lasting pairs
  are allied, yet they keep less (0.94 vs 0.97): help goes where the attack is,
  so outcome comparisons measure need, not benefit.
- Flows (Helmholtz-Hodge per 50 games, sign-flip chance): gifts run DOWNHILL in
  20/20 worlds (circulation - chance: B2 -0.09, B1 -0.13), and the slope lasts
  (gradient persistence 0.20, 0.50): tribute; B1 drains to the well-connected
  (rank r -0.28/-0.33, two worlds), B2 weakly to the rich (-0.16). Takeovers go
  round more than chance (+0.07, +0.24) but never in the same place twice
  (persistence 0.00). Only with alliances do gifts go round (+0.07, +0.04;
  AL s3 at 2,000: +0.14); controls -0.13/-0.08; no ring persists (<= 0.09).
- Help defined by the rules (decisive.py): effect of u's stake on v = P(v keeps
  | all stakes) - P(v keeps | all but u's), the engine's rule counted exactly
  (checked against 4,000 engine draws per contest, 300 contests). Help possible
  only via revolutions or pooled sides. B2: 0.005-0.028 nodes saved per 100
  agents a game (vs 6-7 lost), falling over evolution (0.03-0.07 at 100 ->
  0.002-0.012 at 1,900). B1: 2.4 (1.5-4.4) saved vs 31 lost, 2.3% of stakes on
  others; a by-product of the revolutionary crowd: returned 0.76x
  exposure-matched chance (10/10 below), net help runs one way (-0.08 vs
  sign-flip, 10/10), clones 1.03x; doubles 100 -> 1,000, then no trend.
  Alliances: rescue ~nil (0.016-0.085 per 100 a game) although agents with an
  ally keep 99-100% vs 90-95%: alliances protect by PEACE (an ally cannot take
  the node) and EXCHANGE (gifts go round), not rescue. Controls (no
  revolutions, no alliances): no stake can help, none did.
- Rare help is RETURNED above chance (pooled over stretches): B2 156 pairs vs
  28 expected (10/10 worlds), allies 26 vs 3.6, 10 vs 0.1, 6 vs 0.5; clones
  1.2x (B2). But the situations that allow help are symmetric (two neighbours'
  crowds against one attacker; mutual alliances), and in B1/B2 no input tells
  an agent who staked on it: returned help is the trace reciprocity would
  leave, not proof of it. A before/after timing test is degenerate (each pair
  of events counts once each way) and was dropped.
- Kin: B2 clones get ~1.4x their share of stakes, gifts AND takeovers
  (proximity); B1 1.0x stakes, 1.1x gifts, 1.5x takeovers. No kin favour.
- A reciprocity artefact caught: help "returned" 3.5x chance against a
  density null fell to 0.8x against an exposure-matched one.

## Lifetime steps: B2 improves at every step to 2,000 (2026-10-15, evening)
research/oee/checks/long_contests.py, E16's ten B2 worlds, T = 100, once in
each era's world (long_steps.json): later era's share, median (range), worlds
above one half:
- 100 vs 0: 0.794 (0.71-0.86) 10/10;  200 vs 100: 0.749 (0.65-0.88) 10/10
- 500 vs 200: 0.790 (0.71-0.91) 10/10;  1,000 vs 500: 0.682 (0.64-0.76) 10/10
- 1,500 vs 1,000: 0.618 (0.53-0.79) 10/10;  2,000 vs 1,500: 0.595 (0.52-0.65) 10/10
- 2,000 vs 1,000: 0.667 (0.57-0.76) 10/10.
=> Evolution in B2 never stopped; it slows (0.68 -> 0.62 -> 0.60 per 500
iterations). The long runs to 10,000 will show whether it slows towards a tie.

## Earlier pilots re-judged a lifetime long (2026-10-15, night)
rejudge.py, T = 100, once in each era's world, 188 contests + 12 self-contests
(control 0.44-0.58). Later era 1,000 vs 500, mean of the two homes:
- A10k allies 0.62/0.59, ctrl 0.70/0.62, kin 0.80/0.59, gain 0.69/0.97.
  AL s3: 3,000 vs 1,000 0.69, vs 2,000 0.52; s4: 0.88, 0.69.
- S1-norev 0.76/0.61, S2-gifts 0.87/0.83, S3-giftslapse 0.25/0.62 (erratic),
  S4-lapse 0.65/0.53, S5-bytokens 0.65/0.62; S6-B1 0.37/0.68,
  S7-B1norev 0.35/0.16.
- G1-lottery 0.47/0.56, G2-genotype 0.59/0.65, G3-kinsense 0.72/0.80,
  G4-familysense 0.62/0.67, G5-familypool 0.46/0.78.
- B3a s1 0.48, s2 0.92; B3b s2 0.22 (all-or-nothing contests in worlds held by
  one alliance).
- Lab E12 (B1-5000): median 0.49 (0.31-0.77) vs 500, 0.56 vs 300.
=> B2's family keeps improving 500 -> 1,000 almost everywhere; B1's does not;
the lottery stops evolution. Chapters 46, 54, 55, 56, 57 corrected. Control S0-B2
(rejudge_control.txt): 1,000 vs 500 0.692/0.670, vs 100 0.839/0.901.

## B2 to 10,000: a plateau after ~5,000 (2026-10-15, night)
long_worlds.txt, T = 100, once in each era's world (s001/s002/s003):
5,000 vs 2,000 0.690/0.590/0.632; 10,000 vs 2,000 0.613/0.615/0.650;
10,000 vs 5,000 0.505/0.574/0.551; 10,000 vs 9,500 0.476/0.456/0.593.
Self-contests 0.44-0.58. Chapter 50 updated. jobs_XL.txt continues the three
to 50,000 (run.py --resume last) when jobq_L ends.

## Batch P: the two inputs of Chapter 58 (2026-10-16)
jobs_P.txt: B2 (10,000 tokens, 2,000 iterations), came_from, helped_by, both,
two seeds each; link_token and link_token+seek one seed. Lifetime contests
(batch_P_T100.json) and 200 recorded games from 2,000 (frames_P_*.json):
- Contests 1,000 vs 500: B2 0.68/0.71 0.64/0.70; came 0.78/0.71 0.76/0.77;
  helped 0.78/0.73 0.74/0.72; both-s1 0.71/0.67. 2,000 vs 1,000: B2 0.65/0.66
  0.73/0.80; came 0.64/0.70 0.56/0.68; helped 0.62/0.59 0.60/0.63.
- Help (nodes saved per 100 agents a game): B2 0.009/0.009; came 0.009/0.011;
  helped-s1 0.006; both 0.012/0.007. Returned: 0 with any input.
- Chains >= 3 moves in 200 games: B2 2,978/2,861; came 4,015/3,488; both
  4,511/2,296; helped-s1 3,026. Lag-2 return / chance: B2 1.24/1.35; came
  1.54/1.42; both 1.62/1.36; helped-s1 1.38.
=> came_from used to go back (blinkers), helped_by unused. Written into Ch. 58
(Added later) and Meta VIII (Ch. 76).

## Batch Q so far (2026-10-16)
Contests (batch_Q_T100.json): grow 0.59-0.85 (keeps evolving); took_kin
0.59-0.74 (like B2); mix s1 500->1,000 0.58/0.62 but 1,000->2,000 0.34/0.49,
s2 0.41/0.30 and 0.40/0.39 — NOT a result: in contests every mixed child drew a
fresh id below LABEL_B and counted as the earlier era's (fixed in tourney.contest:
side by ancestry); replayed in batch_Q_mix_T100.json.
Frames: mix 971/673 agents, conquest cost 12.7/9.7 per 100 a game (B2 ~6),
circulation 0.84/0.83 (flip 0.63); unit tokens 1,439/1,312 agents, help
0.04/0.08 (B2 0.009); link tokens without old age 1,630/1,912 agents, conquest
0.6/0.1: peaceful. grow frames redone (pilot_frames did not restore growable
brains before; fixed).
