"""Experiments 69, 70 and 71: worlds of Experiment 63 played under a head count, from the kept world it started from
(69 and 70) or from a world one of its runs kept (71). Each run is read beside its partner without the rule, window by
window. Reads the runs' statistics; plays nothing.   python3 research/efficient/E50/e69.py [E70 | E71]"""
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
    name = sys.argv[1] if len(sys.argv) > 1 else "E69"          # E70 and E71 are read alike
    before = {(s.condition, s.lab["seed"]): s for s in gol_plan.experiment_runs("E63")}
    by_run = {s.run_id: s for s in before.values()}
    out = {}
    for spec in gol_plan.experiment_runs(name):
        step = spec.condition.split(",")[0]
        # Its partner without the rule: the run of Experiment 63 it was played on from (Experiment 71), or the
        # one of the same step and seed, started from the same kept world (Experiments 69 and 70).
        partner = by_run.get(spec.lab["from"]["run"]) or before[step, spec.lab["seed"]]
        label = f"{step}, seed {partner.lab['seed']}"
        pair = {"by heads": windows(spec), "no rule": windows(partner)}
        if pair["no rule"]:                                       # the same windows as the run under the rule
            pair["no rule"] = {**pair["no rule"], "windows": [w for w in pair["no rule"]["windows"] if w["from"] >= spec.start - spec.start % 500]}
            pair["no rule"]["marks"] = "".join("E" if e56.equals(w) else "." for w in pair["no rule"]["windows"])
        out[label] = pair
        print(f"\n{label}")
        for rule, run in pair.items():
            if run is None:
                print(f"  {rule}: not started")
                continue
            print(f"  {rule}: at {run['iteration']} ({run['status']}); windows {run['marks']}")
            for w in run["windows"]:
                print(f"     {w['from']:>5}: {w['agents']:6.0f} agents, {w['links']:5.1f} links each, richest {w['richest_most']:6d} "
                      f"(median {w['richest_median']:6.0f}), Gini {w['gini']:.2f}, births per agent {w['births_per_agent']:.3f}, "
                      f"won by a coalition {w['coalition']:.0%}, deaths {w['deaths']}, brains {w['brains']:.0f}")
    made = {label: pair["by heads"] for label, pair in out.items() if pair["by heads"]}
    died = [label for label, run in made.items() if run["status"] == "extinct"]
    last = [label for label, run in made.items() if run["marks"].endswith("E")]
    print(f"\n{name}: {len(made)} of {len(out)} runs made; died out by heads: {len(died)} {died}; "
          f"a world of equals in the last window played: {len(last)} {last}")
    with open(os.path.join(e56.HERE, f"{name.lower()}.json"), "w") as f:
        json.dump(out, f, indent=1)
