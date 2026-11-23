# A2 log (2026-11-23)
1. Plan: 500 iterations (12,000-12,499) of `B1-10000-085a32-s4801`, one lineage per genotype present at the start, followed by `parent_brain_ids`.
2. `a2_budget.py` reads every frame pair; for each stake in the game it names winner and staker, splits into own / same genotype /
   same lineage different genotype / other lineage (prey); lost = tokens staked on a node won by another lineage; forfeited = tokens
   on nodes that died (their estate goes to the pool); estate income = tokens in the frame minus tokens staked on the node.
   Following parent ids: no gaps (0 of 355,756 genotypes unresolved), 29 s for 500 iterations.
3. Whole window (12,000-12,499) as one: **the lineage grouping collapses**: 1,530 lineages at the start, one at the end
   (every node descends from one start genotype); the same happens within every 100-iteration block. So "ten that grew most vs ten that shrank most" has no content over 500 iterations.
4. Used 20 blocks of 25 iterations, lineages re-defined at the start of each block (1 to 10 lineages alive at the end of a block).
   In each block: winners = the up to 10 lineages with the most nodes gained, losers = the 10 with the most lost; pooled over blocks.
5. Identity per lineage and block: tokens at the end - tokens at the start of the game = prey - lost - forfeited + estate: holds exactly for all
   27,367 lineage-blocks. `a2_table.py` makes the table. Data: `a2_rows_*.json`, `a2_table.json`.
