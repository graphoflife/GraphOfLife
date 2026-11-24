"""Card C2, Experiment 49: are mutants as fit as their parents? Reads the assays the lab played; plays nothing."""
import json, os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
SHUFFLES = 2000


def read(name="E49"):
    sets = {}
    for spec in gol_plan.experiment_runs(name):
        try:
            with open(os.path.join(gol_store.run_dir(spec.run_id), "result.json")) as f:
                result = json.load(f)["result"]
        except (OSError, ValueError):
            continue
        sets.setdefault((spec.condition, str(spec.trial["operator"])), []).append(result)
    return sets


def reading(results, rng, at=None):
    """R = mean count of mutant sites over that of control sites, pooled over assays; p by shuffling labels within each assay."""
    played = [r for r in results if r.get("sites")]
    if not played:
        return None
    def count(site):
        return site["count"] if at is None else site["series"][at]
    counts = [np.array([count(s) for s in r["sites"]], dtype=float) for r in played]
    labels = [np.array([s["mutant"] for s in r["sites"]]) for r in played]
    def ratio(lab):
        m = sum(c[l].sum() for c, l in zip(counts, lab)); n = sum(l.sum() for l in lab)
        k = sum(c[~l].sum() for c, l in zip(counts, lab)); q = sum((~l).sum() for l in lab)
        return (m / n) / (k / q) if k else float("inf")
    R = ratio(labels)
    shuffled = np.array([ratio([rng.permutation(l) for l in labels]) for _ in range(SHUFFLES)])
    lower = (1 + (shuffled <= R).sum()) / (SHUFFLES + 1)
    upper = (1 + (shuffled >= R).sum()) / (SHUFFLES + 1)
    mut = np.concatenate([c[l] for c, l in zip(counts, labels)]); con = np.concatenate([c[~l] for c, l in zip(counts, labels)])
    top = np.quantile(con, 0.9)
    return {"assays": len(played), "died_out_worlds": len(results) - len(played), "mutants": int(mut.size), "controls": int(con.size),
            "R": float(R), "p_lower": float(lower), "p_upper": float(upper),
            "null_R_2.5_97.5": [float(np.quantile(shuffled, q)) for q in (0.025, 0.975)],
            "mean_count": [float(mut.mean()), float(con.mean())],
            "died_out": [float((mut == 0).mean()), float((con == 0).mean())],
            "above_controls_90th": [float((mut > top).mean()), float((con > top).mean())],
            "largest": [float(mut.max()), float(con.max())]}


def by_horizon(results, rng, horizons=(3, 6, 9)):
    """The same reading at earlier iterations, from each site's series: not what the thesis reads, reported beside it."""
    played = [r for r in results if r.get("sites")]
    if not played:
        return {}
    every = played[0]["every"]
    rows = {}
    for h in horizons:
        r = reading(results, rng, at=h // every)
        mut = np.array([s["series"][h // every] for x in played for s in x["sites"] if s["mutant"]], dtype=float)
        con = np.array([s["series"][h // every] for x in played for s in x["sites"] if not s["mutant"]], dtype=float)
        rows[h] = {"R": r["R"], "p_lower": r["p_lower"], "p_upper": r["p_upper"], "null": r["null_R_2.5_97.5"],
                   "alive": [float((mut > 0).mean()), float((con > 0).mean())]}
    return rows


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    out = {}
    name = sys.argv[1] if len(sys.argv) > 1 else "E49"       # E62 is read the same way, at 3, 6 and 9 iterations
    sets = read(name)
    for (world, op), results in sorted(sets.items(), key=lambda kv: (kv[0][0], (0, 0) if kv[0][1] == "engine" else (1, int(kv[0][1])))):
        r = reading(results, rng)
        out[f"{world} | {op}"] = r
        if r is None:
            print(f"{world:<18} {op:>6}: nothing played yet"); continue
        print(f"{world:<18} {op:>6}: R = {r['R']:.3f} (shuffled 95%: {r['null_R_2.5_97.5'][0]:.2f} to {r['null_R_2.5_97.5'][1]:.2f}; p below {r['p_lower']:.4f}, above {r['p_upper']:.4f}); "
              f"mean count {r['mean_count'][0]:.2f} vs {r['mean_count'][1]:.2f}; died out {r['died_out'][0]:.0%} vs {r['died_out'][1]:.0%}; "
              f"above the controls' 90th percentile {r['above_controls_90th'][0]:.1%} vs {r['above_controls_90th'][1]:.1%}; {r['assays']} assays")
    print()
    print("Earlier iterations (not what the thesis reads): R, the 95% range of R with labels shuffled, and lines still alive")
    for (world, op), results in sorted(sets.items(), key=lambda kv: (kv[0][0], (0, 0) if kv[0][1] == "engine" else (1, int(kv[0][1])))):
        rows = by_horizon(results, rng)
        out[f"{world} | {op}"] = {**(out.get(f"{world} | {op}") or {}), "earlier": rows}
        if rows:
            print(f"{world:<18} {op:>6}: " + "; ".join(
                f"after {h}: R {r['R']:.2f} ({r['null'][0]:.2f} to {r['null'][1]:.2f}), alive {r['alive'][0]:.0%} vs {r['alive'][1]:.0%}"
                for h, r in rows.items()))
    json.dump(out, open(os.path.join(HERE, "c2.json" if name == "E49" else f"{name.lower()}.json"), "w"), indent=1)
