"""Experiment 73: the four worlds of Experiment 71 played on under a head count from iteration 30,000 to 37,500. Each
run is read window by window with Experiment 63's measure of a world of equals and with the star's measure (more
than half the agents with one link, by the median over a window), beside the run of Experiment 71 it was played on
from. Reads the runs' statistics; plays nothing.   python3 research/efficient/E50/e73.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e56, e69
import gol_plan, gol_record


def one_link(run_id, lo):
    """The median share of agents with one link in the window of 500 iterations from lo."""
    rows = [r for r in gol_record.read_stats(run_id) if r["phase"] == 2 and lo <= r["iteration"] < lo + 500]
    return float(np.median([r["leaves"] / r["nodes"] for r in rows])) if rows else None


def read(spec):
    """A run's windows, each with its share of agents with one link; its marks: E a world of equals, S a star."""
    run = e69.windows(spec)
    if run is None:
        return None
    for w in run["windows"]:
        w["one_link"] = one_link(spec.run_id, w["from"])
    run["marks"] = "".join("S" if w["one_link"] > 0.5 else "E" if e56.equals(w) else "." for w in run["windows"])
    return run


if __name__ == "__main__":
    before = {s.run_id: s for s in gol_plan.experiment_runs("E71")}
    out, bad = {}, []
    for spec in gol_plan.experiment_runs("E73"):
        label = spec.condition.replace("the world of ", "").replace(", played on", "")
        pair = out[label] = {"played on": read(spec), "before": read(before[spec.lab["from"]["run"]])}
        print(f"\n{label}")
        for name, run in pair.items():
            if run is None:
                print(f"  {name}: not started")
                continue
            print(f"  {name}: at {run['iteration']} ({run['status']}); windows {run['marks']}")
            for w in run["windows"]:
                print(f"     {w['from']:>5}: {w['agents']:6.0f} agents, {w['links']:5.1f} links each, one link {w['one_link']:.0%}, "
                      f"richest {w['richest_most']:5d}, births per agent {w['births_per_agent']:.3f}, "
                      f"won by a coalition {w['coalition']:.0%}, deaths {w['deaths']}")
        run = pair["played on"]
        if run and (run["status"] == "extinct" or run["marks"].endswith("S")):
            bad.append(label)
    made = {label: pair["played on"] for label, pair in out.items() if pair["played on"]}
    print("\nbeside: windows that are a world of equals; windows off the measure by births (0.01 or more) and by coalitions "
          "(under nine in ten); separate times of many births; agents and links in the first and last window; "
          "the largest share with one link; the largest holding")
    for label, pair in out.items():
        for name in ("before", "played on"):
            run = pair[name]
            if run is None:
                continue
            ws = run["windows"]
            many = [w["births_per_agent"] >= 0.01 for w in ws]
            run["summary"] = {"equals": run["marks"].count("E"), "windows": len(ws), "by_births": sum(many),
                              "by_coalitions": sum(w["coalition"] < 0.9 for w in ws),
                              "times_of_births": sum(m and not (i and many[i - 1]) for i, m in enumerate(many)),
                              "most_one_link": max(w["one_link"] for w in ws), "largest": max(w["richest_most"] for w in ws)}
            q = run["summary"]
            print(f"  {label}, {name}: {q['equals']} of {q['windows']}; {q['by_births']} by births, {q['by_coalitions']} by coalitions; "
                  f"{q['times_of_births']} times; {ws[0]['agents']:.0f} agents with {ws[0]['links']:.1f} links to "
                  f"{ws[-1]['agents']:.0f} with {ws[-1]['links']:.1f}; one link {q['most_one_link']:.0%}; largest {q['largest']}")
    key = made.get("50 numbers, seed 6302")
    print(f"\nE73: {len(made)} of {len(out)} runs made; ended badly (extinct, or a star in the last window): {len(bad)} {bad}; "
          f"50 numbers, seed 6302: {key['marks'].count('E') if key else 0} windows a world of equals")
    with open(os.path.join(e56.HERE, "e73.json"), "w") as f:
        json.dump(out, f, indent=1)
