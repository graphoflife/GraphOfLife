# D1 log (2026-11-23)
1. Read the plan (section 2, D1, A2). Printed the config of `B1-10000-085a32-s4801`: contest lottery, allocation strongest_first,
   redistribution parents, link_search off, link_upkeep paid_or_used, prune_after blotto, keep_home 0, link_token false.
2. Read one pair of frames (iteration 12000): keys `ids, tokens, brain_ids, parent_brain_ids, parent_ids, ages, delta, edges, cleanup, summary, decisions`
   (`delta` exists; `parent_brain_ids` too, which the plan's key list lacks). Game frame decisions: `allocations` (`agent, tokens, spread, targets, alloc`),
   `winners` (`node, winner, amount`), `pruned_edges`. Reproduction frame decisions: `births, gifts, upkeep, pruned_edges`.
   The game frame's ids are the survivors; the allocations are for the graph of the reproduction frame (before the cull).
3. Read the engine: `blotto_phase`, `stake_in_rounds`, `Board`, `_lottery`, `_cleanup_and_redistribute`, `_share_out`, `_stale_edges`,
   `_add_edge`, `_pay_upkeep`, `_spawn_child`, `_birth_links`, `_precompute_features`, `_inputs`, `_frame`.
4. `d1_check.py` (22 s): 100 iterations, all laws; every check passed. `d1_extra.py` (17 s): core drift, territories, ages.
5. Results in `d1_result.json`, `d1_extra.json`. Wrote RESULT.md.
Settings of the measurement: iterations 12,000-12,099; kin = same genotype id; a territory = connected set of same-genotype survivors.
Skipped: frame checks of reproduction-phase transfers (gifts, upkeep, estates in phase 1).
