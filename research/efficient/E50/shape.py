"""The shape of Experiment 50's branches, from the worlds they kept: is there a core, how complete, who holds the tokens,
how old are its agents? Plays nothing.   python3 research/efficient/E50/shape.py"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store


def shape(run, era):
    b = gol_store.load_archive(run, era)
    n, e, tok = len(b["ids"]), b["edges"], b["tokens"].astype(float)
    deg = np.bincount(e.ravel(), minlength=n)
    core = deg > deg.max() / 2            # the core: agents with more than half the links of the best-linked one
    k = int(core.sum())
    inside = int((core[e[:, 0]] & core[e[:, 1]]).sum())
    cross = int((core[e[:, 0]] ^ core[e[:, 1]]).sum())
    age = era - b["born_at"]
    return {"era": era, "agents": n, "links": int(len(e)), "median_links": float(np.median(deg)), "most_links": int(deg.max()),
            "core": k, "core_links_made": inside / max(1, k * (k - 1) // 2), "core_tokens_each": float(tok[core].mean()),
            "core_share_of_tokens": float(tok[core].sum() / tok.sum()), "core_median_age": float(np.median(age[core])),
            "fringe": int(n - k), "fringe_links_each": float(np.median(deg[~core])) if k < n else None,
            "fringe_tokens_each": float(tok[~core].mean()) if k < n else None,
            "fringe_median_age": float(np.median(age[~core])) if k < n else None,
            "links_fringe_to_core": cross, "links_fringe_to_fringe": int(len(e)) - inside - cross}


if __name__ == "__main__":
    out = {}
    for s in gol_plan.experiment_runs("E50"):
        for era in gol_store.archived(s.run_id):
            r = shape(s.run_id, era); out[f"{s.condition}, seed {s.lab['seed']}, {era}"] = r
            print(f"{s.condition:<12} {s.lab['seed']} at {era}: {r['agents']:>5} agents, median {r['median_links']:.0f} links, most {r['most_links']:>5}; "
                  f"core {r['core']:>4} ({r['core_links_made']:.0%} of its links made, {r['core_tokens_each']:.0f} tokens each, {r['core_share_of_tokens']:.0%} of all, age {r['core_median_age']:.0f}); "
                  f"fringe {r['fringe']} ({r['fringe_tokens_each'] or 0:.1f} tokens each, age {r['fringe_median_age'] or 0:.0f})")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape.json"), "w"), indent=1)
