# What we know — the consolidated findings

A living list of what the book has established, with where it is shown. When a
later result changes an earlier one, the earlier line is corrected here and the
chapter gets an "Added later" note. Numbers are medians over worlds unless said
otherwise. Last consolidated: 2026-10-14.

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
  takeover time ∝ tokens^0.77 (E10 thesis refuted) (Ch. 18, 54).
- Families live in patches that widen like a random walk (reach 7 → 12 steps for
  lags 25 → 200) (Ch. 37, 54).
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
- Reading: a local Blotto game has a mixed equilibrium; evolution finds
  randomisation (the noise inputs, 25–58% of decision changes) quickly, and then
  nothing beats it. Information has to pay somewhere else — cooperation needs a
  surplus of its own.
- Support stakes at face value are selected against (a gift without surplus).
- Fog of war ends conquest and with it evolution.
- Group sizes (clone and kin domains) shrink over time in every pilot; net token
  circulation stays ~0.

**Open (being tested)**
- Cooperation with a surplus: alliances by mutual consent with pooled stakes,
  kin pooling, support worth more than the home stake (proto.py; batch A lost in
  a reboot, to rerun).
