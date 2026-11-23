# D2 LOG (2026-11-23)

1. Read plan.md section 2 and card D2, D1/RESULT.md, and the code in GraphOfLife.py: `blotto_phase` 2121-2300
   (every agent stakes all tokens on itself or a neighbour; `new_tokens[v] = incoming_totals[v]` line 2262; a node
   with no offer gets 0 tokens, lines 2243-2249), `_resolve_winner` 2395 (largest stake, ties random among the tied
   agents; with no flagged tokens the mob is empty), `_lottery` 2381 (chance = stake / total), `_pooled` 2349 (one offer
   per side; the owner is the maker of its own side, so the owner keeps the node when its side wins; equal pools -> random),
   `_claims` 2370 (with revolutions off nothing is flagged, so it changes nothing).
2. Wrote d2.py. Game: team of n agents (joint plan = one split of w tokens per agent) vs one outsider node. Team payoff =
   sum over nodes of (tokens staked on the node) x (chance the team wins the node). Rules: largest, lottery, pool.
   Revolutions (d) skipped: it needs the flagged part of each stake as a free choice (REV_FRACTION is a brain output);
   not modelled rather than guessed.
3. Solver: double oracle. A small restricted game is solved by regret matching (RM+ with linear averaging, 5,000 rounds
   in the loop; then plain regret matching, 200,000 rounds, on the final support). After each solve the two sides' best
   replies over ALL plans (full table never stored) give a certified lower and upper bound on the value.
   Value = mean of the bounds; gap = upper - lower. The loop stops at gap < 0.004 (first cases) / 0.008 (later cases;
   tolerance changed after the first runs because some "largest, all border" cases took 150 s) or at a 150 s deadline.
4. Checked conservation: for 300 random plan pairs per case and rule, team + outsider expected tokens = n*w + outsider
   tokens, max error 0.0 (exact in floating point). A node nobody stakes on holds 0 tokens, so no token dies with a node.
5. 56 cases (path n=1..4, clique n=2..4, 2 attachments, w=2,3, outsider A and B) x 3 rules. Largest team table:
   clique n=4 all border w=3: 1,500,625 team plans x 1,820 outsider plans = 2.7e9 cells (over the 2 million limit), so it was
   never built; the double oracle only touches best replies. Other big cases: path n=4 all border w=3 A, 40,000 x 1,820.
6. Results in d2_result.json (one entry per case, with team and outsider plans, gaps, seconds), table.md.
   Largest gap 0.0108 (clique n=4 all border w=3 A, largest rule: stopped at the 150 s deadline; just over the 0.01 aim).
   Second largest 0.0082; all others < 0.0065.
