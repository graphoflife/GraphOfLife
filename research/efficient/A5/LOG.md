# A5 log
- Same frame layout as A6. Territory = connected set of agents (edges whose two ends are both agents in the frame; ids looked up) with the same brain_ids, found by union-find.
- a5.py: every 100th iteration's game frame (171 / 275 / 183 frames). Null: brain ids permuted over the agents of the same frame (network kept), 3 shuffles per frame, averaged. Output a5.csv, summ.txt (4 equal windows of frames), sizes.pkl (pooled sizes).
- Run time 15 s. Not done: step 2's lifetime of a territory (no follow-through of genotypes over frames), step 4 (family pools). No log-log plot (no matplotlib): the size-bin counts are in summ.txt.
