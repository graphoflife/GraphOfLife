"""Experiment 66: three rules that wealth cannot break, each against the hoarder that spreads, and each with no hoarder.
Reads the invasions the lab played; plays nothing.   python3 research/efficient/E50/e66.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e64
import gol_plan, gol_store

WAYS = {"home-spreader": "at home", "moving-spreader": "moving", "resident": "no hoarder"}


def read(name="E66"):
    trials = []
    for spec in gol_plan.experiment_runs(name):
        try:
            with open(os.path.join(gol_store.run_dir(spec.run_id), "result.json")) as f:
                result = json.load(f)
        except (OSError, ValueError):
            continue
        trials.append({"set": spec.condition, "world": spec.trial["run"].split("-")[3], "place": spec.trial["place"],
                       "way": WAYS[spec.trial["strategy"]], "series": result["result"]["series"],
                       "seconds": result["seconds"]})
    return trials


def control(trial):
    """A world played with no hoarder: what it was and what it is at the end."""
    s = trial["series"]
    first, last = s[0], s[-1]
    each = lambda p: round(2 * p["links"] / p["agents"], 1) if p["agents"] else 0
    return {"world": trial["world"], "place": trial["place"], "played": last["t"],
            "agents": [first["agents"], min(p["agents"] for p in s), last["agents"]],
            "links_each": [each(first), each(last)],
            "richest": [first["richest"], max(p["richest"] for p in s), last["richest"]],
            "richest_median": int(np.median([p["richest"] for p in s])),
            "world_ends": last["agents"] == 0 or last["agents"] < first["agents"] / 2}


if __name__ == "__main__":
    trials = read()
    out = {}
    for name in dict.fromkeys(t["set"] for t in trials):
        mine = [t for t in trials if t["set"] == name]
        if name.endswith("no hoarder"):
            rows = out[name] = [control(t) for t in mine]
            print(f"\n{name}: {len(rows)} worlds; end {sum(r['world_ends'] for r in rows)}")
            for r in sorted(rows, key=lambda r: (r["world"], r["place"])):
                print(f"  {r['world']} place {r['place']}: played {r['played']}; agents first, least, last {r['agents']}; "
                      f"links each first, last {r['links_each']}; richest first, most, last {r['richest']}, "
                      f"median {r['richest_median']}")
            continue
        rows = out[name] = [{**e64.reading(t), "way": t["way"], "richest_most": max(p["richest"] for p in t["series"])}
                            for t in mine]
        print(f"\n{name}: {len(rows)} trials; sweep {sum(r['sweeps_at'] is not None for r in rows)}; "
              f"worlds end {sum(r['world_ends'] for r in rows)}; lines lost {sum(r['line_gone_at'] is not None for r in rows)}; "
              f"minutes {sum(t['seconds'] for t in mine) / 60:.0f}")
        for way in ("at home", "moving"):
            for world in sorted({r["world"] for r in rows}):
                part = sorted((r for r in rows if r["way"] == way and r["world"] == world), key=lambda r: r["place"])
                show = lambda key: [r[key] for r in part]
                print(f"  {way}, {world}: sweeps at {show('sweeps_at')}; line gone at {show('line_gone_at')}; "
                      f"played {show('played')}; most tokens {show('most_tokens')}; most copies {show('most_copies')}; "
                      f"agents {show('agents_start')[0]} -> {show('agents_end')}; under 20 at {show('under_20_at')}; "
                      f"ends {sum(show('world_ends'))}")
    with open(os.path.join(e64.HERE, "e66.json"), "w") as f:
        json.dump(out, f, indent=1)
