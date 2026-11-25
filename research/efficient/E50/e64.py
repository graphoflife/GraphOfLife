"""Experiments 64 and 65: does one hoarder end a world of equals, and one that spreads once it is rich?
Reads the invasions the lab played; plays nothing.   python3 research/efficient/E50/e64.py [E64|E65]"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
TENTH = 4000            # a tenth of the world's 40,000 tokens: the thesis's "takes off"
SHOWN = (0, 1, 2, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000)


def read(name="E64"):
    trials = []
    for spec in gol_plan.experiment_runs(name):
        try:
            with open(os.path.join(gol_store.run_dir(spec.run_id), "result.json")) as f:
                series = json.load(f)["result"]["series"]
        except (OSError, ValueError):
            continue
        trials.append({"set": spec.condition, "world": spec.trial["run"].split("-")[3], "place": spec.trial["place"],
                       "series": series})
    return trials


def reading(trial):
    s = trial["series"]
    first, last = s[0], s[-1]
    over = next((p["t"] for p in s if p["tokens"] > TENTH), None)
    died = last["agents"] == 0
    return {"world": trial["world"], "place": trial["place"], "played": last["t"],
            "start_tokens": first["tokens"], "agents_start": first["agents"], "links_start": first.get("links"),
            "lost_first_game": len(s) > 1 and s[1]["mine"] == 0 and s[1]["agents"] > 0,
            "line_gone_at": next((p["t"] for p in s if p["mine"] == 0 and p["agents"] > 0), None),
            "takes_off_at": over, "most_tokens": max(p["tokens"] for p in s),
            "most_copies": max(p["mine"] for p in s),
            "two_at": next((p["t"] for p in s if p["mine"] >= 2), None),
            "sweeps_at": next((p["t"] for p in s if p["agents"] and p["mine"] > p["agents"] / 2), None),
            "least_agents": min(p["agents"] for p in s), "least_links": min(p.get("links") or 0 for p in s),
            "halved_at": next((p["t"] for p in s if p["agents"] < first["agents"] / 2), None),
            "under_20_at": next((p["t"] for p in s if p["agents"] <= 20), None),
            "died_out_at": last["t"] if died else None,
            "agents_end": last["agents"], "links_end": last.get("links"),
            "world_ends": died or last["agents"] < first["agents"] / 2,
            "at": {str(p["t"]): [p["tokens"], p["mine"], p["agents"], p.get("links")] for p in s if p["t"] in SHOWN}}


if __name__ == "__main__":
    name_ = sys.argv[1] if len(sys.argv) > 1 else "E64"
    trials = read(name_)
    out = {}
    for name in dict.fromkeys(t["set"] for t in trials):
        rows = [reading(t) for t in trials if t["set"] == name]
        out[name] = rows
        print(f"\n{name}: {len(rows)} trials; sweep {sum(r['sweeps_at'] is not None for r in rows)}, "
              f"take off {sum(r['takes_off_at'] is not None for r in rows)}, "
              f"worlds end {sum(r['world_ends'] for r in rows)} (died out {sum(r['died_out_at'] is not None for r in rows)}), "
              f"lost the first game {sum(r['lost_first_game'] for r in rows)}")
        for r in sorted(rows, key=lambda r: (r["world"], r["place"])):
            print(f"  {r['world']} place {r['place']}: starts with {r['start_tokens']:4d} tokens; played {r['played']:4d}; "
                  f"line gone at {r['line_gone_at']}; two agents at {r['two_at']}; half the world at {r['sweeps_at']}; "
                  f"over {TENTH:,} at {r['takes_off_at']}; most {r['most_tokens']:6d} tokens, "
                  f"{r['most_copies']} copies; agents {r['agents_start']} -> {r['agents_end']}, links {r['links_start']} -> {r['links_end']}; "
                  f"died out at {r['died_out_at']}")
            print("      t: tokens, copies, agents, links  " + "  ".join(f"{t}: {v[0]}, {v[1]}, {v[2]}, {v[3]}" for t, v in r["at"].items()))
    hoard = [r for name, rows in out.items() if name != "controls" for r in rows]
    if hoard:
        print(f"\nall hoarders: {len(hoard)} trials; sweep {sum(r['sweeps_at'] is not None for r in hoard)}; "
              f"take off {sum(r['takes_off_at'] is not None for r in hoard)}; "
              f"worlds end {sum(r['world_ends'] for r in hoard)}")
    with open(os.path.join(HERE, name_.lower() + ".json"), "w") as f:
        json.dump(out, f, indent=1)
