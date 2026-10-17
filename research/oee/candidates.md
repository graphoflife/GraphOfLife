# Candidates for long runs

Made by `research/oee/pilots/candidates.py` from the profile of every world (Meta X). A candidate
meets every mark of Chapter 46's health bar, its agents are more diverse than B2's own (a diversity
score above 0.51, the median of Experiment 16's B2 worlds), and it has at least 10,000 tokens.
Each is played on to 10,000 iterations in `GraphOfLifeRuns/.pilots/LC/`.

| world | what it is | diversity | median tokens | 3 tokens or fewer | dimension | agents | long run |
|---|---|---|---|---|---|---|---|
| `Y/mix-mut80-s2` | B2 with mutation_probability = 0.8, mix = True | 0.79 | 5 | 36% | 3.8 | 1,238 | from 1,000 to 10,000: queued |
| `Y/mix-mut80-s1` | B2 with mutation_probability = 0.8, mix = True | 0.78 | 5 | 41% | 3.8 | 1,299 | from 1,000 to 10,000: queued |
| `Y/mix-keepone-s2` | B2 with keep_home = 0.01, mix = True | 0.70 | 1 | 77% | 3.4 | 1,537 | from 1,000 to 10,000: queued |
| `Y/mix-kinsense-s2` | B2 with kin_sense = True, mix = True | 0.70 | 8 | 32% | 3.7 | 821 | from 1,000 to 10,000: queued |
| `U/U-BDFG-s1` | B2 with allow_revolutions = False, random_input_amount = 0, valence = 0, link_token = True, mix = True | 0.68 | 8 | 25% | 3.7 | 652 | from 1,000 to 10,000: queued |
| `W/norev-mix-s1` | B2 with allow_revolutions = False, mix = True | 0.68 | 2 | 67% | 3.7 | 2,245 | from 1,000 to 10,000: running or done |
| `Y/mix-linktoken-s1` | B2 with valence = 0, link_token = True, mix = True | 0.66 | 8 | 36% | 3.4 | 588 | from 1,000 to 10,000: queued |
| `Y/mix-came-s2` | B2 with came_from = True, mix = True | 0.66 | 3 | 53% | 3.7 | 1,521 | from 1,000 to 10,000: queued |
| `M/mix-children-s2` | B2 with mix_at_birth = True, redistribution = children | 0.65 | 10 | 23% | 3.3 | 486 | from 5,000 to 10,000: queued |
| `Y/mix-grow-s1` | B2 with brain_kind = grow, hidden_layers = [32], mix = True | 0.65 | 12 | 23% | 3.3 | 366 | from 1,000 to 10,000: queued |
| `Y/mix-helped-s1` | B2 with helped_by = True, mix = True | 0.63 | 5 | 38% | 3.7 | 1,075 | from 1,000 to 10,000: queued |
| `Y/mix-nodice-s2` | B2 with random_input_amount = 0, mix = True | 0.63 | 3 | 53% | 3.5 | 1,468 | from 1,000 to 10,000: queued |
| `Y/mix-pool-s1` | B2 with mix = True, pool = genotype | 0.63 | 6 | 34% | 3.3 | 816 | from 1,000 to 10,000: queued |
| `W/norev-mix-s2` | B2 with allow_revolutions = False, mix = True | 0.63 | 3 | 55% | 3.9 | 1,343 | from 1,000 to 10,000: queued |
| `Y/mix-keepone-s1` | B2 with keep_home = 0.01, mix = True | 0.61 | 1 | 79% | 3.7 | 1,791 | from 1,000 to 10,000: queued |
| `W/norev-linktoken-s1` | B2 with allow_revolutions = False, valence = 0, link_token = True | 0.59 | 5 | 42% | 4.0 | 921 | from 1,000 to 10,000: running or done |
| `Y/mix-poorest-s1` | B2 with redistribution = poorest, mix = True | 0.57 | 3 | 51% | 3.4 | 1,455 | from 1,000 to 10,000: running or done |
| `K/reach-random-forced-s1` | B2 with link_consent = forced, link_search = reach | 0.54 | 4 | 49% | 4.0 | 1,931 | from 1,000 to 10,000: queued |
| `Y/mix-lottery-s2` | B2 with contest = lottery, mix = True | 0.53 | 5 | 32% | 3.9 | 1,137 | from 1,000 to 10,000: queued |
| `Y/mix-unittoken-s2` | B2 with brain_kind = grow, hidden_layers = [32], mix = True, unit_token = True | 0.53 | 6 | 43% | 3.6 | 521 | from 1,000 to 10,000: queued |
| `S/G4-familysense-s2` | B2 with allow_revolutions = False, kin_sense = True, tag_rate = 0.05 | 0.53 | 1 | 74% | 4.0 | 2,935 | from 1,000 to 10,000: running or done |
| `Y/mix-came-s1` | B2 with came_from = True, mix = True | 0.52 | 4 | 48% | 3.4 | 1,284 | from 1,000 to 10,000: queued |
| `Y/mix-familysense-s1` | B2 with kin_sense = True, mix = True, tag_rate = 0.05 | 0.52 | 10 | 20% | 3.7 | 707 | from 1,000 to 10,000: queued |
