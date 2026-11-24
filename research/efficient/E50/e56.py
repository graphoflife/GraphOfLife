"""Reads Experiment 56: the rules of s3058 from founders with mutations of 10 numbers, at 10,000 and at 40,000 tokens.
A run is a world of equals if, over its last 500 iterations, there are fewer than 0.01 births per agent and iteration
and at least nine nodes in ten are won by a coalition (the thesis's words). Plays nothing."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path[:0] = [ROOT, HERE]
import gol_plan, gol_record, gol_store


def window(rows, lo, hi):
    g = [r for r in rows if r["phase"] == 2 and lo <= r["iteration"] < hi]
    b = [r for r in rows if r["phase"] == 1 and lo <= r["iteration"] < hi]
    if not g or not b:
        return None
    m = lambda k: float(np.mean([r[k] for r in g if r.get(k) is not None]))
    births = float(np.mean([(r.get("births") or 0) / r["nodes_before"] for r in b if r.get("nodes_before")]))
    return {"from": lo, "agents": m("nodes"), "links": m("meanDegree"), "most_links": m("maxDegree"), "tokens": m("meanTokens"),
            "gini": m("gini"), "births_per_agent": births, "coalition": m("revolutions") / m("nodes"),
            "deaths": int(sum((r.get("starved") or 0) + (r.get("orphaned") or 0) for r in g + b))}


def equals(w):
    return w is not None and w["births_per_agent"] < 0.01 and w["coalition"] >= 0.9


def steps(run_id):
    """Under inherited_step: the middle step of a run's brains, and its quartiles, in every world it kept."""
    out = []
    for era in gol_store.archived(run_id):
        blob = gol_store.load_archive(run_id, era)
        if "brain_steps" in blob:
            k = blob["brain_steps"]
            out.append({"era": era, "median": float(np.median(k)), "quartiles": [float(np.quantile(k, 0.25)), float(np.quantile(k, 0.75))],
                        "least": int(k.min()), "most": int(k.max())})
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "E56"          # E56, E57 and E58 are read alike
    out = {}
    for s in gol_plan.experiment_runs(name):
        try:
            meta = gol_store.load_meta(s.run_id)
        except (OSError, ValueError):
            print(f"{s.condition}, seed {s.lab['seed']}: not started"); continue
        rows = gol_record.read_stats(s.run_id)
        end = meta["iteration"]
        ws = [w for w in (window(rows, lo, lo + 500) for lo in range(s.start - s.start % 500, end, 500)) if w]
        first = next((w["from"] for w in ws if equals(w)), None)
        last = window(rows, max(0, end - 500), end)
        kept_steps = steps(s.run_id)
        out[f"{s.condition}, seed {s.lab['seed']}"] = {"run": s.run_id, "iteration": end, "status": meta.get("status"), "windows": ws,
                                                       "world_of_equals": equals(last), "first_window_of_equals": first,
                                                       "steps": kept_steps}
        print(f"\n{s.condition}, seed {s.lab['seed']}: at {end} ({meta.get('status')}); a world of equals in its last 500: {equals(last)}; first such window from {first}")
        for w in ws:
            print(f"   {w['from']:>5}: {w['agents']:6.0f} agents, {w['links']:5.1f} links (most {w['most_links']:5.0f}), {w['tokens']:5.1f} tokens, Gini {w['gini']:.2f}, "
                  f"births per agent {w['births_per_agent']:.3f}, won by a coalition {w['coalition']:.0%}, deaths {w['deaths']}")
        marks = "".join("E" if equals(w) else "." for w in ws)
        stretches = [len(x) for x in marks.split(".") if x]
        out[f"{s.condition}, seed {s.lab['seed']}"].update(marks=marks, stretches=stretches)
        print(f"   windows of 500: {marks}   stretches of a world of equals: {stretches}")
        if kept_steps:
            print("   middle step (quartiles) in the worlds it kept: " + "  ".join(
                f"{k['era']}: {k['median']:.0f} ({k['quartiles'][0]:.0f}-{k['quartiles'][1]:.0f})" for k in kept_steps))
    json.dump(out, open(os.path.join(HERE, f"{name.lower()}.json"), "w"), indent=1)
