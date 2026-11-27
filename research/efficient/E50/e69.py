"""Experiment 69: the six worlds of Experiment 63 with mutations of 30 and 50 numbers, played again from the same
kept world under a head count. Each is read beside Experiment 63's run of the same step and seed, window by window.
Reads the runs' statistics; plays nothing.   python3 research/efficient/E50/e69.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e56
import gol_plan, gol_record, gol_store


def windows(spec):
    """A run's windows of 500 iterations, each with the richest holding met in it; None if the run is not made."""
    try:
        meta = gol_store.load_meta(spec.run_id)
    except (OSError, ValueError):
        return None
    rows = gol_record.read_stats(spec.run_id)
    end, out = meta["iteration"], []
    for lo in range(spec.start - spec.start % 500, end, 500):
        w = e56.window(rows, lo, lo + 500)
        if w:
            rich = [r["maxTokens"] for r in rows if r["phase"] == 2 and lo <= r["iteration"] < lo + 500]
            out.append({**w, "richest_most": int(max(rich)), "richest_median": float(np.median(rich)),
                        "brains": float(np.mean([r["distinctBrains"] for r in rows
                                                 if r["phase"] == 2 and lo <= r["iteration"] < lo + 500
                                                 and r.get("distinctBrains") is not None] or [0]))})
    return {"run": spec.run_id, "iteration": end, "status": meta.get("status"), "windows": out,
            "marks": "".join("E" if e56.equals(w) else "." for w in out)}


if __name__ == "__main__":
    before = {(s.condition, s.lab["seed"]): s for s in gol_plan.experiment_runs("E63")}
    out = {}
    for spec in gol_plan.experiment_runs("E69"):
        step = spec.condition.split(",")[0]
        pair = {"by heads": windows(spec), "no rule": windows(before[step, spec.lab["seed"]])}
        out[f"{step}, seed {spec.lab['seed']}"] = pair
        print(f"\n{step}, seed {spec.lab['seed']}")
        for rule, run in pair.items():
            if run is None:
                print(f"  {rule}: not started")
                continue
            print(f"  {rule}: at {run['iteration']} ({run['status']}); windows {run['marks']}")
            for w in run["windows"]:
                print(f"     {w['from']:>5}: {w['agents']:6.0f} agents, {w['links']:5.1f} links each, richest {w['richest_most']:6d} "
                      f"(median {w['richest_median']:6.0f}), Gini {w['gini']:.2f}, births per agent {w['births_per_agent']:.3f}, "
                      f"won by a coalition {w['coalition']:.0%}, deaths {w['deaths']}, brains {w['brains']:.0f}")
    died = [name for name, pair in out.items() if pair["by heads"] and pair["by heads"]["status"] == "extinct"]
    print(f"\ndied out by heads: {len(died)} of {len(out)} {died}")
    with open(os.path.join(e56.HERE, "e69.json"), "w") as f:
        json.dump(out, f, indent=1)
