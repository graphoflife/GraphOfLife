# What we know — the consolidated findings

A living list of what the book has established, with where it is shown. When a
later result changes an earlier one, the earlier line is corrected here and the
chapter gets an "Added later" note. Numbers are medians over worlds unless said
otherwise. Last consolidated: 2026-10-15.

## The baseline B1 (Parts II and III; 30 worlds of 10,000 and 30 of 100,000 tokens)

**Brains**
- Nearly blind: a change in what a brain sees reaches what it does reduced to a
  gain of 0.0015 (founders 0.0006) — five sigmoid layers each pass about a
  quarter (Ch. 19; note *How a signal fades through layers*).
- Selection set only constants: few children (3.6% of agents would have one at
  100,000 tokens; 2.6% at 10,000), the stake spread over all candidates (98%),
  every token revolutionary (Ch. 19).
- 42% of agents (58% at 10,000 tokens) score every candidate ≤ 0 and fall back to
  the rules' even split; the extra home stake is largest-remainder rounding, not
  the brains (Ch. 19).
- First-layer weights sit at the mutation–drift balance, noise inputs included (Ch. 19).
- Messages name the writer's genotype (within-writer distance 0.0005 vs 0.55
  between) — tags already exist, unread (Ch. 20).

**The game and the tokens**
- The game is close to an even split: a lazy random walk, tokens at rest ∝
  degree + 1; the walk's prediction fits with slope 0.94 and R² 0.45 in worlds
  without stars (0.65 and 0.34 with them) (Ch. 16, 21, 27–29).
- Wealth order is half forgotten in about 50 games (Ch. 21).
- Connections are gained ∝ k^1.02 and lost ∝ k^1.20 at 100,000 tokens (k^0.94
  and k^1.11 at 10,000) (Ch. 22). Token tail: a power law, γ = 2.22 from 49
  tokens up, bootstrap p = 0.37 at 100,000 tokens (γ = 2.29, p = 0.92 at
  10,000) (Ch. 32).

**Heredity and selection**
- Brains spread by conquest 17–18× more than by birth (Ch. 39).
- 97% of new genotypes arise in brains that are not being copied (mutation after
  every game) (Ch. 39).
- A world of 10,000 tokens descends from one branch about every 350 iterations;
  takeover time ∝ tokens^0.77 (E10 thesis refuted) (Ch. 18, 60).
- Families live in patches that widen like a random walk (reach 7 → 12 steps for
  lags 25 → 200) (Ch. 37, 60).
- Hazard falls with age to ~1% per game; no ageing; the oldest (~850) is set by
  numbers; implied median age 55 = observed (Ch. 12–13).
- Worlds whose brains never change end in very different places; evolving
  worlds all end alike (E07, Ch. 41).

**The network**
- The world is 4.5× longer (2× at 10,000 tokens) and its spectral gap 33×
  smaller than a degree-preserving random network's; no new connection ever
  shortens a distance; correlation length 1–2 steps for tokens, ages and
  connections, 3–4 for genotypes (Ch. 22–24, 35).
- Dimension 3.3 by balls, ~1.9 by walks at 60,000 agents; the core holds 60% of
  agents and 73% of tokens (Ch. 24–25).
- 7 of 30 worlds of 100,000 tokens are stars (one agent joined to ≥ 10%); a star's
  centre is a place handed to newborn children (dynasties) (Ch. 17, 33).
- Big cuts are quiet borders of poor agents (1 token, 2 links) (Ch. 36).
- Size (E08, Ch. 42): agents and connections ∝ tokens (exponents 1.00, 1.04);
  per-agent statistics size-free, shape not (clustering ×0.44, distances ×2);
  fluctuations do not average out (one game can cut off half a 60,000-agent
  world); distances ∝ N^0.25–0.30.

## Part V: engineering open-ended evolution

**The yardstick (Ch. 46, E12; 10 worlds of 5,000 tokens)**
- B1 learns in its first 50–100 iterations (+361 Elo against its founders at 50,
  +400 at 100) and nothing after: 1,000 vs 300 = 0.483; each era ties with the
  one before (0.47–0.52); a dominance tournament crowns as many champions as
  chance (null mean 3.05, observed 3.1). Thesis held.
- Health bar (Ch. 46): one piece; 2-core ≥ 0.8; worst agent cut ≤ 2%; worst link
  cut ≤ 1%; best-connected agent ≤ 5% of the others; 3–12 links per agent; ball
  dimension 2.3–4. B1 at 5,000 tokens fails it in every world (2-core 0.51,
  link cut 8%, hub 9%).

**Brains that see (Ch. 47, E13)**
- One hidden layer of 16: founder gain 0.2. Worlds die out: 4/10, and 8/10 with
  change only when copied (MOR); B1 never below 74 agents. Thesis held.
- Not the network alone: no world of any condition passes the bar, seeing
  networks no worse than B1's. Every world loses 2–5% of its agents to the cull
  each iteration and lives while births keep up.
- Births probe (era brains in the founders' world): in 6 of 12 dead worlds the
  last brains had no child at all; surviving seeing worlds' brains had more
  children than B1's (median 17 vs 3 per 100 agent-iterations). Sight makes
  reproduction a decision evolution can take away. Cause (selection vs drift in
  small worlds) open.
- MOR alone at birth only ("spawn") froze a pilot world: no births, no change.

**Space (Ch. 48, model research/oee/space.py; E14)**
- Births alone make a tree of blobs: newborns close only triangles; one agent's
  loss can cut 37% (links 2), 41% (parent only).
- A death mended in a ring never splits the world (proof in Ch. 48); with 2 birth
  links: cuts 1.4% / 1.2%, dimension 3.7, 4.5 links. 3 birth links pass the bar.
  Chain mending stretches the world (dimension 2.3); mending everything makes
  hubs and dimension 4.9; a link cap hurts without brains (5.7% / 3.0%).
- In the game: unused links lapse only if their ends share ≥ 2 neighbours
  (prune_needs 2; with 1 a cactus can form); valence 8 against preferential
  attachment by brains; no handover.
- E14 (analysed 2026-10-14): under B2 nobody was cut off from 200 to 1,000 in
  any of ten worlds; 2-core 1.000; no link and (in the median frame) no agent
  whose loss cuts anything off, worst moment 1.2%; hub 2.1–3.1%, 5.0–5.8
  links; dimension 3.45–4.19 (median 3.86) — seed 5 above the bar's 4, so the
  pre-registered thesis is refuted by that one line. Every ablation fails in
  9–10 worlds: no mending 16,000–43,000 agents cut off (deaths by age cut;
  worlds knit at links but hung at agents, worst agent cut 16%); any unused
  link lapses → dimension 2.0, link cut 20%, one extinction; no birth links →
  25–434 cut off; no valence → 8.7 links (up to 17.7), dimension > 4 in 7;
  handover → 40% outside the 2-core.

**Who dies (Ch. 49, E15 planned)**
- Kamikaze children: with links at birth and mending, newborns with 1–2 tokens
  stake everything away (46% of the starving are age 0).
- keep_one (last token home) makes all-in attacks free: 75% of agents end with
  exactly one token (zombies). keep_home 0.3 (a share) avoids it.
- Without any keep rule worlds starve to one agent.

**Why evolution stops (Ch. 50; pilots, research/oee/pilots/LOG.md)**
- Every depth mechanic tried plateaus within 100–200 iterations: memory inputs,
  memory of 6 games + growable brains, a mutual-stake yield, fog of war,
  support stakes, no random inputs; brain size above its neutral shadow did not
  rise when handover was off.
- E16 (analysed 2026-10-15; 10 B2 worlds of 10,000 tokens to 2,000, and 10
  without random inputs): thesis held — 2,000 vs 500 = 0.482 (0.471–0.502) and
  0.477 (0.452–0.519) in the median world. Steps 0.68 (0→100), 0.56 (100→200),
  0.52 (200→500), then 0.49–0.50. Later is a little worse: 2,000 vs 500 below
  one half in 9/10 (7/10 without dice). Ratings +90, +124, +126, +128, +120,
  +120 (B2 median). Dominance: 3.2 champions against 2.3 by chance (contest
  sigma 0.046 at 10,000 tokens vs 0.117 at 5,000); the excess is the step to 200.
- The Blotto reading is NOT supported by the probes (2026-10-15): founders
  already change 49–66% of stake divisions when their dice are redrawn; evolved
  brains 38–59% — evolution keeps the dice and turns the senses down (own state
  14–22% → 1–9%, messages 34–48% → 11–20%). Under a lottery contest, where dice
  are worthless, evolved brains roll them as much (48–67% vs 59% under Blotto
  without revolutions). And the game is not solved: a hand-written raider takes
  half of an evolved B2 world (Ch. 51). Evolution stops because the search
  stops, not because the game has no better play.
- Support stakes at face value are selected against (a gift without surplus).
- Fog of war ends conquest and with it evolution.
- Group sizes (clone and kin domains) shrink over time in every pilot; net token
  circulation stays ~0.

**Alliances (Ch. 56; pilots A10k, AL, N0 — two seeds each)**
- Alliances by mutual consent with pooled stakes are taken up — the first rule
  of Part V evolution adopts rather than gives up: the largest alliance holds
  83/44% of the world at 1,000 iterations, 99/97% at 2,000 (10,000 tokens),
  91/96% at 2,000 (20,000 tokens); allied connections 98% at 2,000.
- It spreads as a habit (say yes to every neighbour), not as a family: kin groups
  (4 mutations) stay at 2% while the alliance holds 99%.
- Peace: conquest from 7.6–8.0 per 100 agents a game (no alliances) to 2.2–3.4
  (500–1,000) and 0.2–0.4 (last 500 of 3,000). Selection by conquest ends.
- Tournaments flat: +113→+143 (100→1,000), +178→+209 (500→2,000); steps
  0.51–0.53; territorial contests (halves, T 50): 1,000 vs 500 = 0.41, later
  brains lose, as in the control.
- Without the home advantage (incumbency 1) alliances still form (37/71% at
  1,000); two seeds cannot tell the sizes apart.
- Kin pooling (within 2 mutations): peace (0.6–1.8) with small sides (2–14%).
  Support counted twice: still given up.
- A rising statistic with a ceiling set by the world, but one convention
  spreading once — not open-ended evolution (the author's objection).

**How fast brains change (Ch. 52, E09; analysed 2026-10-15; B2 at 10,000 tokens,
mutation probability per copy 0.05/0.1/0.2/0.4/0.8, 10 seeds each, to 1,500)**
- Thesis held on both counts: every pair of rates ties in the median seed
  (0.49–0.52), and every rate has stopped (1,500 vs 1,000: 0.49–0.51).
- Not even how soon differs: every rate's steps are 0.66–0.70 (0→100),
  0.54–0.56 (100→200), then ties from ~500 on. Each rate holds 0.49–0.51
  against all the others.
- What the rate changes is the world: genotypes per agent 0.20 → 0.90, clone
  groups ~8 → 1.2 agents, agents 3,730 → 2,600 (the extra agents of slow
  worlds are single-token ones: median tokens 1.0 at 0.05/0.1, 2.0 at 0.8),
  Gini ~0.5 throughout. Probes: no rate keeps brains less deaf.
- No mutation–selection balance or error threshold in play up to 0.8. More
  variation, or larger families, do not find the better play of Ch. 51.

**The yardstick was too short (2026-10-15, evening; the author's question)**
- Contests of 20 iterations miss what pays over a lifetime (B2: everyone dies at
  100). In contests of T = 100, E16's B2 brains of 2,000 beat those of 500
  (0.73, 10/10 worlds), of 200 (0.82); 200 beats 100 (0.75). At T = 20 the same
  pair was 0.48. Controls at T = 100: self vs self 0.48, labels swapped 0.26.
- So "evolution stops by 200–500" (Ch. 50, 52, and the pilot chapters 54–56)
  holds only for what a 20-iteration contest can see. Steps at T = 100 and the
  ten B2 worlds continued to 10,000 are being run.
- Detection otherwise: E16's T = 20 data exclude even slow improvement between
  500 and 2,000 (2,000 vs 500 = 0.482 ± 0.005); no Red Queen signature (home
  minus away −0.006). Exploitability (one game): founders 0.87 → 1.49 (200) →
  ~1.2 from 1,000 on — the brains move away from a one-game equilibrium and
  stay there (note *Exploitability*).

**Strategies written by hand (Ch. 51; manual.py, the invasion test)**
- In E16's B2 world s002 at 2,000 a 5% patch of raiders (all they can stake on
  the poorest stranger they are sure to take, else home; a password so they
  never attack a teammate) reaches 40/50% in 300 iterations, a factory 48/52%,
  the residents' own brains 1/7%. In s001 both die out. Turtles die of age.
- Without the password (lone raider) it dies out everywhere (97-122
  iterations), also in s002: the edge is knowing one's own. A single raider
  never establishes (0/20 in s002; peaks up to 28 agents): an Allee effect.
- Ten worlds (one planting each): the raider took 40/62% of two worlds, the
  factory 43/48% of two (three worlds in all; seed 2 by both); in seven both
  died out; the control never above 8%. Which strategy wins depends on the
  world — evolved worlds differ in their weak points.
- So better play exists, but needs a raid, a password and a crowd at once — a
  valley single mutations cannot cross (the green-beard problem).

**Other contests (Ch. 55; batch G, engine mode, B2 without revolutions)**
- Lottery (Tullock): almost nothing learned (+14/+23 Elo at 1,000 vs +100/+139
  under Blotto); dice still rolled (48–67% vs 59%) where worthless.
- Stakes pooled by genotype: fighting cut to a third (3.2 vs 6.7–9.0 per 100 a
  game), clone groups 8–11 (vs 3), slowly growing in one world; ratings stop
  as under Blotto; on turf later brains lose (0.35).
- Kin sense alone: no groups (3); one world crept upward (1,000 vs 100: 0.67),
  the other did not (0.53). The sense is used less than by founders (12–16% vs 26%).
- Family tags (tag_rate 0.05, a mutation parameter; 1 copy in 100 starts a
  family): sensed — families of 30–39, no change; pooled — families of
  150–490 that rise and fall like empires (one held 57% at 410, 12% at 710,
  ~30% at the end), fighting 1.3–1.6. Brains stop either way (1,000 vs 500:
  0.43–0.54). A password and a crowd for free are not enough: the play itself
  is what evolution does not find.

**The original switches (Ch. 54; batch S, engine mode)**
- The pilot B2 of seed 1 equals the lab's run B2-10000-s001 for all 1,000
  iterations (agents and links): engine mode reproduces the lab exactly.
- B1 without revolutions is a mayfly world (births 14–96 per 100 agents an
  iteration, conquest 70–90, mean age 2–16, worst cut 30–34%); with
  revolutions B1 has old agents (age 80–160), stars (hub 23–40%). In one world
  without revolutions the brains of 1,000 beat those of 500 (0.73).
- Seed 2 of B1 without revolutions: improvement to 500 (+383), then back
  (1,000 vs 500 = 0.40). Seed 3 run for 3,000: improvement to 500 (500 beat 0,
  100, 200 in 11/12 contests), then back (1,000 vs 500 0.32, 1,500 vs 500
  0.11), and the world shrank to a few dozen agents after 2,000 (13 at its
  smallest) — extinct by the lab's rule; later contests meaningless. Without
  revolutions B1 evolves longer (to 500, once to 1,000), not without end.
- In B2 no switch keeps evolution going: 1,000 vs 500 between 0.47 and 0.57 in
  every variant (B2 0.50/0.50). Gifts and lapsing links give more to learn at
  first (+263/+157 and +520/+256 at 100 vs +102/+85), then stop. Revolutions
  are rare in B2 (0.1–0.3 nodes per 100 a game) and change nothing; in B1 they
  take 28–39 of the 46–55 nodes that change hands a game.
- Gifts on B2's space: harmless. Gifts with every unused link lapsing: the
  network breaks as without gifts (dimension ~2, worst agent cut ~20%,
  1,300–1,450 agents): brains do not pay for the links that matter.

**Structures and help (Ch. 58; E16 B2 and B1 baseline worlds, games 1,000–2,000;
alliance pilots played on 200 iterations; research/oee/checks)**
- Takeover chains (moves with >= half the taker's tokens, each from the node the
  last took) are 23x chance in B2 and 11x in B1 for >= 3 moves. B2's are
  puffers (mover keeps its node 93%; packets 2–3 tokens); B1's are gliders
  (keeps 9%; packets grow 18 -> 82 tokens). They blink a little (back to the
  node before 1.37x / 1.27x chance), wander like random walks (no direction:
  an agent cannot know where it came from), and guns fire without a period.
- Lasting mutual stakes are feuds, not partnerships (B2 kept 0.80 vs 0.89, 10/10).
- Gifts run downhill in every world without alliances (circulation 0.09–0.13
  below the sign-flip level) along slopes that last (0.20 B2, 0.50 B1): tribute,
  to the well-connected in B1. Takeovers go round, but no ring lasts 50 games.
  With alliances gifts go round (+0.04 to +0.14): exchange.
- Help defined by the rules (a stake without which the holder would have lost
  its node; counted exactly) exists only via revolutions or pooled sides. B2:
  ~0.01 nodes saved per 100 agents a game, falling over evolution. B1: 2.4
  (1.5-4.4), a by-product: returned 0.76x exposure-matched chance, runs one
  way, clones 1.03x (10/10 worlds). Alliances: rescue ~nil (0.02-0.09); they
  protect by peace and exchange. Rare help (B2, allies) is returned 4-75x
  chance, but in situations symmetric by nature (shared attacker, mutual
  alliance) and, in B1/B2, without any input saying who helped: the trace of
  reciprocity, not reciprocity.
- Outcome comparisons of the helped against the rest measure need, not
  benefit (allied pairs keep less, allied agents more); help has to be defined
  by the rules.

**Lifetime steps (E16 B2, T = 100; 2026-10-15)**
- Every step from the founders to 2,000 is won by the later era in 10/10 worlds:
  0.79 (100 vs 0), 0.75, 0.79, 0.68 (1,000 vs 500), 0.62, 0.60 (2,000 vs 1,500);
  2,000 vs 1,000 0.67. B2 keeps evolving, more slowly as it goes.
- Continued to 10,000 (three worlds): 5,000 vs 2,000 0.69/0.59/0.63, 10,000 vs
  2,000 0.61/0.62/0.65, 10,000 vs 5,000 0.51/0.57/0.55, 10,000 vs 9,500
  0.48/0.46/0.59. A plateau after ~5,000; runs to 50,000 under way (jobs_XL).

**Earlier pilots re-judged a lifetime long (T = 100; rejudge.py, 2026-10-15)**
- Control: eras against themselves 0.44–0.58 (12 contests, pilots and lab).
- B2-based pilots keep evolving 500 -> 1,000 (later era's share, two seeds):
  B2 itself (S0) 0.69/0.67, alliances 0.62/0.59, their control 0.70/0.62, kin pooled 0.80/0.59, support
  doubled 0.69/0.97; Blotto without revolutions 0.76/0.61, gifts 0.87/0.83,
  lapse 0.65/0.53, estate by tokens 0.65/0.62; genotype pooling 0.59/0.65, kin
  sense 0.72/0.80, family sense 0.62/0.67, family pooling 0.46/0.78. Long
  alliance worlds: 3,000 beats 1,000 (0.69/0.88) and 2,000 in one (0.69; 0.52).
- Exceptions: the lottery (0.47/0.56) - a ticket does not pay for a better brain;
  gifts that lapse (breaking networks, erratic); B3 (all-or-nothing contests).
- B1 does not: E12's ten worlds 0.49 median (0.31-0.77) for 500 -> 1,000;
  B1 pilots 0.37/0.68, B1 without revolutions 0.35/0.16 (later brains lose).
- The short-contest verdicts of Chapters 54-57 ("tournaments flat") were wrong
  for B2's family of worlds and right for B1's.

**What goes with what (Appendix B; research/atlas; 2.3 million agent rows from E16's B2 and E02's
B1 worlds, games 1,000-2,000; world statistics of the same worlds and of E10's thirty)**
- Token curvature drives change at ~1/5 of the even split's pace in both (slope 0.20 B2, 0.21 B1;
  r ~0.3); in B1 through nodes changing hands (peaks keep 10%, level agents 56%; among kept
  agents r -0.01), in B2 among agents that stay (r 0.28; 88-98% keep).
- Wealth persistence over 10 games: rank 0.78 B1, 0.41 B2; connections 0.92 in both.
- 30 of 374 non-definitional pairs reverse sign B2 vs B1, mostly where wealth sits: tokens vs
  connections -0.18/+0.56, PageRank -0.14/+0.65, age -0.17/+0.36; the rich go all-in in B2
  (+0.42) and spread in B1 (-0.45).
- B2: nothing among 21 quantities predicts losing one's node (max |r| 0.16). B1: home share +0.47,
  connections -0.43.
- Logarithmic axes reveal much in B1 (curvature vs neighbours' connections 0.05 -> 0.64; tokens
  vs targets 0.23 -> 0.67), little in B2 (narrow distributions).
- World over time (changes): B1's scattered estate levels (Gini -0.78); B2's hoarding unlevels
  (Gini +0.43). Between E10's 30 worlds: PC1 = 47% of 79 statistics = the star (hub share).

**Open (being tested)**
- Rules without numbers (B3, Ch. 57): all seven of B2's numbers removed at once
  — s1: without alliances extinct, with alliances a peaceful near-tree.
- The original switches judged for OEE (Ch. 54) and two other contests, a
  lottery and stakes pooled by genotype (Ch. 55): batch S/G running.
- Strategies written by hand in evolved worlds (Ch. 51): in E16's B2 world s002
  at 2,000 a 5% patch of raiders (all it can stake on the poorest stranger it is
  sure to take, else home; a password so teammates never fight) reaches 40/50%
  of the world in 300 iterations, a factory 48/52%; the resident control 7/1%.
  In world s001 both die out (control 8/5%). The turtle dies out everywhere
  (age). Ten worlds, a lone raider without password, and single-agent
  invasions running.
