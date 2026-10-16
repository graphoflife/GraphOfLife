# Candidates for long runs

Made by `research/oee/pilots/candidates.py` from the profile of every world (Meta X). A candidate
meets every mark of Chapter 46's health bar, its agents are more diverse than B2's own (a diversity
score above 0.50, the median of Experiment 16's B2 worlds), and it has at least 10,000 tokens.
Each is played on to 10,000 iterations in `GraphOfLifeRuns/.pilots/LC/`.

| world | what it is | diversity | median tokens | 3 tokens or fewer | dimension | agents | long run |
|---|---|---|---|---|---|---|---|
| `Q/mix-s1` | B2 with mix = True | 0.64 | 5 | 37% | 3.5 | 908 | from 2,000 to 10,000: running or done |
| `Q/mix-s2` | B2 with mix = True | 0.55 | 11 | 20% | 3.6 | 659 | from 2,000 to 10,000: running or done |
| `S/G4-familysense-s2` | B2 with allow_revolutions = False, kin_sense = True, tag_rate = 0.05 | 0.53 | 1 | 74% | 3.9 | 2,935 | from 1,000 to 10,000: running or done |
