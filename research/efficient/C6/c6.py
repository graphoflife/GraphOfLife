"""Card C6, Experiment 52: is a line's success in its brain, in its place, or luck? Reads the assays the lab played; plays nothing."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
RESAMPLES = 2000


def read(name="E52"):
    worlds = {}
    for spec in gol_plan.experiment_runs(name):
        try:
            with open(os.path.join(gol_store.run_dir(spec.run_id), "result.json")) as f:
                result = json.load(f)["result"]
        except (OSError, ValueError):
            continue
        if result.get("sites"):
            worlds.setdefault(spec.condition.split(":")[0], []).append((spec.trial, result))
    return worlds


def scores(result, at, kind):
    series = np.array([s["series"][at // result["every"]] for s in result["sites"]], dtype=float)
    return np.log1p(series) if kind == "copies" else (series > 0).astype(float)


def corr(a, b, rows):
    a, b = a[rows], b[rows]
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def reading(plays, at, kind, rng):
    plain = [scores(r, at, kind) for t, r in plays if "swap" not in t]
    moved = [(np.array([s["brain_of"] for s in r["sites"]]), scores(r, at, kind)) for t, r in plays if "swap" in t]
    if len(plain) < 2 or not moved:
        return None
    n = plain[0].size
    own = np.mean(plain, axis=0)                              # brain j in its own place j
    place_other = np.mean([y for _, y in moved], axis=0)      # place j with other brains
    brain_other = np.zeros(n); times = np.zeros(n)
    for whose, y in moved:                                    # brain whose[j] played in place j
        np.add.at(brain_other, whose, y); np.add.at(times, whose, 1)
    brain_other /= np.maximum(times, 1)
    half_a, half_b = np.mean(plain[0::2], axis=0), np.mean(plain[1::2], axis=0)
    all_rows = np.arange(n)
    out = {"plain": len(plain), "moved": len(moved), "sites": int(n),
           "mean_score": float(own.mean()), "r_brain": corr(own, brain_other, all_rows),
           "r_place": corr(own, place_other, all_rows), "r_same": corr(half_a, half_b, all_rows)}
    boots = {"r_brain": [], "r_place": [], "r_same": [], "place_minus_brain": []}
    for _ in range(RESAMPLES):
        rows = rng.randint(0, n, size=n)
        rb, rp, rs = corr(own, brain_other, rows), corr(own, place_other, rows), corr(half_a, half_b, rows)
        boots["r_brain"].append(rb); boots["r_place"].append(rp); boots["r_same"].append(rs); boots["place_minus_brain"].append(rp - rb)
    for k, v in boots.items():
        out[k + "_band"] = [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
    return out


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    out = {}
    for world, plays in sorted(read().items()):
        for at, kind in ((6, "copies"), (3, "alive"), (9, "copies")):
            r = reading(plays, at, kind, rng)
            out[f"{world} | {kind} after {at}"] = r
            if r is None:
                print(f"{world}: too few assays yet"); break
            b = lambda k: f"{r[k]:+.3f} ({r[k + '_band'][0]:+.3f} to {r[k + '_band'][1]:+.3f})"
            print(f"{world:<18} {kind} after {at}: r_brain {b('r_brain')}; r_place {b('r_place')}; r_same {b('r_same')}; "
                  f"place minus brain {r['place_minus_brain_band'][0]:+.3f} to {r['place_minus_brain_band'][1]:+.3f}; {r['plain']} + {r['moved']} assays of {r['sites']} sites")
    json.dump(out, open(os.path.join(HERE, "c6.json"), "w"), indent=1)
