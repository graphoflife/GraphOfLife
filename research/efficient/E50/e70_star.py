"""Experiment 70, read afterwards: what the worlds are that did not form the order under a head count.
Two readings, of the runs' statistics and frames; plays nothing.   python3 research/efficient/E50/e70_star.py

1. The star. For each run, in its last 500 iterations: the links of the most linked agent over the number of
   agents, and the share of agents with one link.
2. Who keeps a node. In every 25th game of the first 500 iterations, for every node: its links, how many
   neighbours flag a token on it, and whether its brain won it again."""
import gzip, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e56
import gol_plan, gol_record, gol_store

GAMES = range(20001, 20500, 25)


def star(run_id, lo=21000, hi=21500):
    rows = [r for r in gol_record.read_stats(run_id) if r["phase"] == 2 and lo <= r["iteration"] < hi]
    return {"hub": float(np.median([r["maxDegree"] / r["nodes"] for r in rows])),
            "one_link": float(np.median([r["leaves"] / r["nodes"] for r in rows]))}


def kept(run_id, counts):
    """Add this run's games to counts[links][flaggers] = [nodes, kept by their own brain]."""
    for it in GAMES:
        path = os.path.join(gol_store.run_dir(run_id), "frames", f"frame_{2 * it + 1:05d}.json.gz")
        with gzip.open(path) as f:
            decisions = json.load(f)["decisions"]
        flaggers = {}
        for a in decisions["allocations"]:
            for target, flagged in zip(a["targets"], a["revolt"]):
                if target != a["agent"] and flagged:
                    flaggers[target] = flaggers.get(target, 0) + 1
        winner = {w["node"]: w["winner"] for w in decisions["winners"]}
        for a in decisions["allocations"]:
            links = min(len(a["targets"]) - 1, 3)                      # 3 stands for three or more
            cell = counts.setdefault(links, {}).setdefault(min(flaggers.get(a["agent"], 0), 2), [0, 0])
            cell[0] += 1
            cell[1] += winner.get(a["agent"]) == a["agent"]


if __name__ == "__main__":
    before = {(s.condition, s.lab["seed"]): s for s in gol_plan.experiment_runs("E63")}
    out = {"star": {}, "kept": {"by heads": {}, "no rule": {}}}
    print("in iterations 21,000 to 21,500: the most linked agent's links over all agents, and the share with one link")
    for spec in gol_plan.experiment_runs("E70"):
        step = spec.condition.split(",")[0]
        partner = before[step, spec.lab["seed"]]
        row = out["star"][f"{step}, seed {spec.lab['seed']}"] = {"by heads": star(spec.run_id), "no rule": star(partner.run_id)}
        print(f"  {step}, seed {spec.lab['seed']}: by heads {row['by heads']['hub']:.2f} and {row['by heads']['one_link']:.2f}; "
              f"no rule {row['no rule']['hub']:.2f} and {row['no rule']['one_link']:.2f}")
        kept(spec.run_id, out["kept"]["by heads"])
        kept(partner.run_id, out["kept"]["no rule"])
    print(f"\nnodes whose own brain won them again, in {len(GAMES)} games of iterations 20,001 to 20,476, all nine worlds")
    for rule, counts in out["kept"].items():
        for links in sorted(counts):
            cells = counts[links]
            n, k = sum(c[0] for c in cells.values()), sum(c[1] for c in cells.values())
            by = "; ".join(f"{fl}{'+' if fl == 2 else ''} flaggers {c[1] / c[0]:.0%} of {c[0]}" for fl, c in sorted(cells.items()))
            print(f"  {rule}, {links}{'+' if links == 3 else ''} links: {k / n:.0%} of {n}   ({by})")
    with open(os.path.join(e56.HERE, "e70_star.json"), "w") as f:
        json.dump(out, f, indent=1)
