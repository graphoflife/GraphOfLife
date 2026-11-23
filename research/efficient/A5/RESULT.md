# A5 result: does the largest structure grow?
Territories of at least 2 agents, mean per frame, per window of the run (null = shuffled brain ids, same network).

| Run, window | Agents | Territories (null) | Largest, mean (null) | Largest / agents | Share of agents in territories >= 10 (null) |
|---|---|---|---|---|---|
| 085a32 W1 | 3359 | 416 (112) | 101 (24) | 0.030 | 0.203 (0.036) |
| W2 | 2800 | 355 (66) | 87 (17) | 0.030 | 0.132 (0.022) |
| W3 | 3050 | 397 (93) | 74 (12) | 0.023 | 0.154 (0.011) |
| W4 | 3118 | 343 (152) | 174 (87) | 0.055 | 0.335 (0.107) |
| f511a5 W1-W4 | ~1700 | 135-152 (1) | 5.4-5.8 (1.7) | 0.0035-0.0037 | 0.000 (0.000) |
| 1cfaa4 W1 | 1798 | 196 (7) | 12.7 (2.3) | 0.0074 | 0.014 (0.000) |
| W2 / W3 / W4 | ~1650 | 192 / 185 / 179 | 12.3 / 12.1 / 11.8 | 0.0078 / 0.0082 / 0.0076 | 0.013 / 0.012 / 0.009 |

Largest territory ever seen in a window (085a32): 351, 675, 458, 692. f511a5: 8-10. 1cfaa4: 24-26. Pooled size distributions per window (counts per frame in bins 2,3,4-5,6-9,10-19,20-49,50-99,100+) are in summ.txt.

Reading:
- All three worlds have far more territory than the shuffled null (which keeps the network and the family sizes). For f511a5 and 1cfaa4 the excess is large and clear, but territories are small: size 2-4 mostly, f511a5 never reaches 10. f511a5 mixes brains at birth, so near neighbours differ, which fits.
- 085a32 has big territories (largest 100-700 agents, up to 20-30% of agents in territories of 10+). Its null is also big in W4 (few genotypes dominate: 970 brains), so the excess there is smaller (0.335 vs 0.107), still real. The largest does not rise steadily: it falls from W1 to W3 (101 -> 74), then jumps in W4 (174, 0.055 of agents). Largest/agents moves the same way.
- 1cfaa4 and f511a5 are flat across all four windows.
- 1cfaa4 mutates every brain in place by chance each iteration, so a genotype splits quickly and territories are cut up. The territories we see (up to 26) are the ones that survive this, but the real structure (families of near-identical brains) is under-counted. Exact-equality territories understate size here; a family-based territory (using parent_brain_ids) would be the fairer measure. Not done.

Decision (rule: "growth if the largest size and the tail rise from window to window and are not explained by more agents"): NO sustained growth. 1cfaa4 and f511a5 are flat. 085a32 does not rise window to window (down, down, then up in W4); agent number is steady, so the W4 jump is not an agent-count effect, but one jump after three flat or falling windows, in one world, is not growth. Marginal: 085a32 W4 deserves a look with all frames (the tail p99 also rose, 29 -> 63, null 13 -> 44).

Caveats: one sampled frame per 100 iterations; 3 shuffles per frame; territories by exact genotype only; lifetime of territories (step 2) and family-pool comparison (step 4) not done.
