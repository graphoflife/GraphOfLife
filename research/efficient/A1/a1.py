"""Card A1: does a world's behaviour keep moving away from where it was? Nothing here plays a world."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe as P
import gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES, BRAINS = 300, 600          # 600 brains an era, in two halves of 300: one half against another era's, or its own other half


def population(beh, rows, most):
    """What a set of brains does on every probe: the share with a child, and the share sending the first token to each candidate."""
    child = (beh["child"][rows] > 0).mean(axis=0)
    first = np.stack([(beh["first"][rows] == c).mean(axis=0) for c in range(most)])      # (candidates, probes)
    return child, first


def distance(a, b):
    """Mean over probes: the gap in the share with a child, the total variation between the first-token
    distributions, and the gap in the share whose first token stays at home (candidate 0)."""
    return (float(np.abs(a[0] - b[0]).mean()), float(0.5 * np.abs(a[1] - b[1]).sum(axis=0).mean()),
            float(np.abs(a[1][0] - b[1][0]).mean()))


def card(name, run_id):
    eras = gol_store.archived(run_id)
    cfg = P.config(run_id)
    asked = P.probes(run_id, eras[len(eras) // 2], PROBES)
    most = max(p["candidates"] for p in asked)
    rng = np.random.RandomState(3)
    halves = {}
    for era in eras:
        blob = gol_store.load_archive(run_id, era)
        n = len(blob["ids"])
        rows = rng.choice(n, size=min(BRAINS, n), replace=False)
        beh = P.behaviour(*P.layers(blob, np.sort(rows)), asked, cfg)
        k = len(rows) // 2
        order = rng.permutation(len(rows))
        halves[era] = (population(beh, order[:k], most), population(beh, order[k:], most), n,
                       float(blob["tokens"].mean()), len(set(blob["brain_ids"].tolist())))
    rows = []
    for i, a in enumerate(eras):
        for b in eras[i:]:
            d = distance(halves[a][0], halves[b][1])
            rows.append({"a": a, "b": b, "lag": b - a, "child": d[0], "first": d[1], "home": d[2]})
    floor = {k: float(np.mean([r[k] for r in rows if r["lag"] == 0])) for k in ("child", "first", "home")}
    by_lag = {}
    for r in rows:
        if r["lag"] and r["a"] > 0:                      # the founders are random brains: left out of the walk
            by_lag.setdefault(r["lag"], []).append(r)
    table = [{"lag": lag, "pairs": len(v), "child": float(np.mean([r["child"] for r in v])),
              "first": float(np.mean([r["first"] for r in v])),
              "home": float(np.mean([r["home"] for r in v]))} for lag, v in sorted(by_lag.items())]
    slopes = {}
    for k in ("child", "first", "home"):
        usable = [(t["lag"], t[k]) for t in table if t["pairs"] >= 2 and t[k] > 0]
        if len(usable) >= 3:
            x, y = np.log([u[0] for u in usable]), np.log([u[1] for u in usable])
            slopes[k] = float(np.polyfit(x, y, 1)[0])
    from_founders = [{"era": r["b"], "child": r["child"], "first": r["first"], "home": r["home"]} for r in rows if r["a"] == 0 and r["lag"]]
    return {"world": name, "run": run_id, "eras": eras, "floor": floor, "by_lag": table, "slopes": slopes,
            "from_founders": from_founders,
            "eras_info": [{"era": e, "agents": halves[e][2], "tokens_each": halves[e][3], "genotypes": halves[e][4]} for e in eras]}


if __name__ == "__main__":
    out = []
    for name in ("s4003", "s4008", "s4002"):
        r = card(name, P.RUNS[name]); out.append(r)
        print(f"\n{name}: eras {r['eras'][0]}..{r['eras'][-1]}; floor (two halves of one era): child {r['floor']['child']:.3f}, first token {r['floor']['first']:.3f}, home {r['floor']['home']:.3f}")
        print("  from the founders: " + "  ".join(f"{d['era']}: {d['child']:.2f}/{d['first']:.2f}/{d['home']:.2f}" for d in r["from_founders"]))
        for t in r["by_lag"]:
            print(f"  lag {t['lag']:>6} ({t['pairs']:>2} pairs, founders left out): child {t['child']:.3f}  first token {t['first']:.3f}  home {t['home']:.3f}")
        print("  slope of log distance on log lag:", {k: round(v, 2) for k, v in r["slopes"].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "a1.json"), "w"), indent=1)
