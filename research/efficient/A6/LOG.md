# A6 log
- Read one frame (085a32, iteration 50, game frame). Keys: iteration, phase, nodes_before, ids, tokens, brain_ids, parent_brain_ids, parent_ids, ages, delta, edges, cleanup, summary, decisions. cleanup has `redistributed`. Tokens held by agents sum to exactly 10,000 in every sampled frame.
- a6.py: every 50th iteration's game frame (index 2*it+1; the plan says 2*it+phase-1, so phase 2 = 2*it+1). Entropy in nats. Random sharing: each token thrown to a random agent, mean of 20 draws, seed 1. Output a6.csv (1,259 rows: 343 / 550 / 366 per run). summ.py makes summ.txt.
- Run time 19 s. No plots (matplotlib is not installed): the curves are the CSV columns `dist` and `flow` against `iter`.
- Family-pool runs skipped (drive unreliable), as instructed.
