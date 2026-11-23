# A6 result: the entropy cycle
Distance from mixed = H(random sharing) - H(real), in nats. Flow = redistributed / 10,000. Means of 10 samples (500 iterations) at start, middle, end.

| Run | Distance start / mid / end | Flow start / mid / end | Tokens per agent | Share of agents with 1 / 2 / 3+ tokens (start -> end) |
|---|---|---|---|---|
| 085a32 | 0.120 / 0.057 / 0.027 | 0.234 / 0.148 / 0.226 | 3.97 / 3.67 / 2.85 | 0.23/0.23/0.54 -> 0.33/0.28/0.39 |
| f511a5 | 0.410 / 0.281 / 0.234 | 0.289 / 0.228 / 0.200 | 7.9 / 6.0 / 5.7 | 0.15/0.16/0.69 -> 0.16/0.17/0.67 |
| 1cfaa4 | 0.324 / 0.453 / 0.438 | 0.276 / 0.226 / 0.209 | 5.9 / 6.1 / 6.5 | 0.20/0.17/0.63 -> 0.20/0.16/0.64 |

Trends (from iteration 1000 on):
- 085a32 drifts towards the mixed state: the distance falls to 0.03 and agents grow in number (2,800 -> 3,700) while tokens per agent fall. By the end the world is almost a random sharing. Flow stays around 0.15-0.23.
- f511a5: the distance falls from 0.41 to 0.23, slowly. Flow falls a little (0.29 -> 0.20).
- 1cfaa4: the distance rises (0.32 -> 0.45) and stays there. Flow falls a little.
- Every run has a flow of roughly 0.2 of all tokens per iteration, but it swings a lot from sample to sample (sd of flow about 0.15-0.18, of agent count 500-900). The world is not steady between samples: it breathes. The fitted slopes (see summ.txt) are small against this noise, so "trend" is only visible at the scale of the start/mid/end means above.
- Correlation of distance with flow across samples: 0.73 (085a32), 0.30 (f511a5), -0.10 (1cfaa4). Only 085a32 shows the idea "more flow, more distance from mixed".

Caveats: one sample per 50 iterations of a quantity that swings every iteration, so these are snapshots of a cycle, not its period. The cycle's speed was not measured (this needs every iteration). Entropy is of agent shares only. The use with C (beneficial mutations) was not done. Family-pool runs not read.
