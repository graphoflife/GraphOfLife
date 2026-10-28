"""How many champions would a dominance tournament crown by chance?

Read every tournament of an experiment (E12 by default), estimate the spread of
one contest's share around its pair's mean (from the two contests of each pair
of eras from the second on), then simulate tournaments in which every era after
the founders is exactly as good as every other (true score 0.5 between them)
and the founders lose every contest, and count champions with the same rule
(gol_tournament.dominance).

    python3 champion_null.py                                   # Experiment 12
    python3 champion_null.py 'B2-10000-s0*.tournament-long.json' 0,100,200,500,1000,1500,2000
"""
import glob, json, sys
import numpy as np
sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
import gol_tournament as tour

pattern = sys.argv[1] if len(sys.argv) > 1 else "B1-5000-s0*.tournament.json"
runs = sorted(glob.glob("/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns/.book/" + pattern))
diffs, real = [], []
for p in runs:
    v = json.load(open(p))
    by = {}
    for c in v["contests"]:
        if c["share"] is not None:
            by.setdefault((c["a"], c["b"]), []).append(c["share"])
    for (a, b), s in by.items():
        if a > 0 and len(s) == 2:
            diffs.append(s[0] - s[1])
    real.append(len(tour.dominance(v["eras"], v["contests"])))
sigma = float(np.std(diffs) / np.sqrt(2))
eras = [int(e) for e in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 50, 100, 200, 300, 500, 700, 1000]
rng = np.random.RandomState(1)
counts = []
for _ in range(20000):
    contests = []
    for j, b in enumerate(eras):
        for a in eras[:j]:
            for _home in range(2):
                share = 0.95 if a == 0 else float(np.clip(0.5 + sigma * rng.randn(), 0, 1))
                contests.append({"a": a, "b": b, "share": share})
    counts.append(len(tour.dominance(eras, contests)))
counts = np.array(counts)
print(f"worlds {len(runs)}; one contest's spread around its pair: sigma = {sigma:.3f} (from {len(diffs)} pairs)")
print("champions found in the worlds:", real)
print("by chance (all later eras equal):", {k: round(float(np.mean(counts == k)), 3) for k in range(2, 6)},
      "mean", round(float(counts.mean()), 2))
