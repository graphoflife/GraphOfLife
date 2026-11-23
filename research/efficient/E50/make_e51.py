"""Writes book/experiments/E51.json: the contests of Experiment 50. Every coin is two contests with one deal,
the two sides in swapped places; every coin of every set and world has a deal of its own."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import gol_plan

SOURCE, AT, END, COINS = "B1-40000-bbbccd-s4602", 20000, 22500, 4
runs = {(s.condition, s.lab["seed"]): s.run_id for s in gol_plan.experiment_runs("E50")}
deal = [0]


def coins(world, world_era, x, y):
    """COINS coins on one world: sides x and y, each (run, era), once in each place, with a deal of its own a coin."""
    items = []
    for _ in range(COINS):
        deal[0] += 1
        for a, b in ((x, y), (y, x)):
            item = {"run": world, "home": world_era, "a": a[1], "b": b[1], "seed": deal[0]}
            if a[0] != world:
                item["a_run"] = a[0]
            if b[0] != world:
                item["b_run"] = b[0]
            items.append(item)
    return items


sets = []
for (cond, seed), rid in runs.items():
    sets.append({"name": f"{cond}, seed {seed}: 22,500 against 20,000",
                 "items": coins(rid, AT, (rid, AT), (rid, END)) + coins(rid, END, (rid, AT), (rid, END))})
for arm in ("10 numbers", "100 numbers"):
    for seed in (5001, 5002):
        mine, theirs = runs[(arm, seed)], runs[("as it is", seed)]
        sets.append({"name": f"{arm} against as it is, seed {seed}, at 22,500",
                     "items": coins(mine, END, (theirs, END), (mine, END)) + coins(theirs, END, (theirs, END), (mine, END))})
for before in (17500, 15000):
    sets.append({"name": f"the world before: 20,000 against {before:,}",
                 "items": coins(SOURCE, before, (SOURCE, before), (SOURCE, AT)) + coins(SOURCE, AT, (SOURCE, before), (SOURCE, AT))})
plan = {"id": "E51", "title": "The contests of Experiment 50: the branches against their start and against each other",
        "trials": {"kind": "contest", "T": 100, "workers": 3, "sets": sets}}
with open(os.path.join(ROOT, "book", "experiments", "E51.json"), "w") as f:
    json.dump(plan, f, indent=1, ensure_ascii=False); f.write("\n")
found = gol_plan.trials_of("E51")
print(len(found), "contests in", len(sets), "sets;", deal[0], "deals")
