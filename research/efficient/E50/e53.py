"""Reads Experiment 53: the core world of Experiment 50 played on with mutations of 10 numbers and with the engine's own.
The thesis reads whether a world has a core at iteration 27,500; beside it, what each run's agents do. Plays nothing."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path[:0] = [ROOT, HERE]
import gol_plan, gol_record, gol_store
from shape import shape


def has_core(r):
    """The thesis's own words: at least 100 agents with more than half the best-linked agent's links, at least half
    of the links possible among them made, holding at least half of the tokens agents hold."""
    return r["core"] >= 100 and r["core_links_made"] >= 0.5 and r["core_share_of_tokens"] >= 0.5


def windows(run_id, spans):
    rows = gol_record.read_stats(run_id)
    out = []
    for lo, hi in spans:
        g = [r for r in rows if r["phase"] == 2 and lo <= r["iteration"] < hi]
        b = [r for r in rows if r["phase"] == 1 and lo <= r["iteration"] < hi]
        if not g:
            continue
        m = lambda k: float(np.mean([r[k] for r in g if r.get(k) is not None]))
        out.append({"from": lo, "to": hi, "agents": m("nodes"), "links": m("meanDegree"), "most_links": m("maxDegree"),
                    "least_links": m("minDegree"), "tokens": m("meanTokens"), "most_tokens": m("maxTokens"), "gini": m("gini"),
                    "births": int(sum(r.get("births") or 0 for r in b)),
                    "deaths": int(sum((r.get("starved") or 0) + (r.get("orphaned") or 0) for r in g + b)),
                    "kept_by_own_agent": m("heldHomeShare"), "home_share": m("selfAllocationShare"),
                    "flagged": m("revoltShare"), "taken_by_coalition": m("revolutions") / m("nodes"), "genotypes": m("distinctBrains")})
    return out


if __name__ == "__main__":
    out = {}
    for s in gol_plan.experiment_runs("E53"):
        meta = gol_store.load_meta(s.run_id)
        kept = [shape(s.run_id, era) for era in gol_store.archived(s.run_id)]
        w = windows(s.run_id, [(lo, lo + 500) for lo in range(22500, 27500, 500)])
        out[f"{s.condition}, seed {s.lab['seed']}"] = {"run": s.run_id, "iteration": meta["iteration"], "kept": kept, "windows": w}
        last = kept[-1]
        print(f"\n{s.condition}, seed {s.lab['seed']} (at {meta['iteration']}): at {last['era']} a core by the thesis's words: {has_core(last)} "
              f"({last['core']} agents, {last['core_links_made']:.0%} of the links among them made, {last['core_share_of_tokens']:.0%} of the tokens)")
        print("   cores in the worlds it kept: " + " ".join("yes" if has_core(k) else "no" for k in kept))
        for x in w[::3] + w[-1:]:
            print(f"   {x['from']}-{x['to']}: {x['agents']:.0f} agents, {x['links']:.1f} links each ({x['least_links']:.0f} to {x['most_links']:.0f}), {x['tokens']:.1f} tokens each (most {x['most_tokens']:.0f}), Gini {x['gini']:.2f}; "
                  f"{x['births']} births, {x['deaths']} deaths; at home {x['home_share']:.1%}, flagged {x['flagged']:.1%}, nodes taken by a coalition {x['taken_by_coalition']:.0%}, kept by own agent {x['kept_by_own_agent']:.0%}")
    json.dump(out, open(os.path.join(HERE, "e53.json"), "w"), indent=1)
