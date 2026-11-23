"""Card B1: how much does one mutation change what a brain decides? Nothing here plays a world."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe as P

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES, PARENTS, MUTANTS = 300, 200, 20
OPERATORS = ["engine", 1, 10, 100, 1000]


def card(name, run_id, at):
    world, blob, cfg = P.world_at(run_id, at)
    asked = P.probes(run_id, at, PROBES)
    rng = np.random.RandomState(7)
    ids = sorted(world.G.nodes())
    parents = [world.brains[int(u)] for u in rng.choice(ids, size=PARENTS, replace=False)]
    base = P.behaviour(*P.stack(parents), asked, cfg)
    # The scale: two brains of the world drawn at random (clones included, as the world has them).
    pairs = [(i, j) for i, j in zip(rng.permutation(PARENTS), rng.permutation(PARENTS)) if i != j]
    strangers = [P.differ(base, i, base, j) for i, j in pairs]
    scale = {k: float(np.mean([d[k] for d in strangers])) for k in ("child", "first", "any")}
    distinct = len({b.brain_id for b in parents})
    out = {"world": name, "run": run_id, "iteration": at, "probes": len(asked), "parents": PARENTS,
           "distinct_genotypes_among_parents": distinct, "agents": len(ids),
           "tokens_per_agent": float(np.mean([world.tokens[u] for u in ids])),
           "strangers_differ": scale, "operators": {},
           "numbers_in_a_brain": int(sum(w.size for w in parents[0].weights) + sum(b.size for b in parents[0].biases))}
    for op in OPERATORS:
        steps = {"child": [], "first": [], "any": []}
        for i, parent in enumerate(parents):
            kids = [P.mutant(parent, cfg, rng) if op == "engine" else P.nudged(parent, cfg, op, rng) for _ in range(MUTANTS)]
            beh = P.behaviour(*P.stack(kids), asked, cfg)
            for m in range(MUTANTS):
                d = P.differ(beh, m, base, i)
                for k in steps: steps[k].append(d[k])
        a = np.array(steps["any"]); rel = a / scale["any"]
        out["operators"][str(op)] = {
            "mutants": int(a.size),
            "probes_decided_differently": {k: float(np.mean(v)) for k, v in steps.items()},
            "median_any": float(np.median(a)), "quartiles_any": [float(np.quantile(a, q)) for q in (0.25, 0.75)],
            "relative_to_strangers_median": float(np.median(rel)),
            "share_identical": float(np.mean(a == 0)),
            "share_nearly_neutral_under_1pct_of_scale": float(np.mean(rel < 0.01)),
            "share_over_25pct_of_scale": float(np.mean(rel > 0.25)),
            "share_over_50pct_of_scale": float(np.mean(rel > 0.50)),
            "share_as_far_as_a_stranger": float(np.mean(rel >= 1.0))}
    return out


if __name__ == "__main__":
    results = []
    for name, at in (("s4003", 15000), ("s4003", 5000), ("s4008", 15000), ("s4002", 15000)):
        t = time.time(); r = card(name, P.RUNS[name], at); results.append(r)
        print(f"\n{name} at {at}: {r['agents']} agents, {r['tokens_per_agent']:.2f} tokens each, {r['distinct_genotypes_among_parents']} genotypes among {PARENTS} parents; "
              f"two strangers decide differently on {r['strangers_differ']['any']:.1%} of probes (child {r['strangers_differ']['child']:.1%}, first token {r['strangers_differ']['first']:.1%})  [{time.time()-t:.0f}s]")
        for op, o in r["operators"].items():
            print(f"  {op:>6}: differs on {o['probes_decided_differently']['any']:.1%} of probes (median {o['median_any']:.1%}); vs strangers median {o['relative_to_strangers_median']:.2f}; "
                  f"identical {o['share_identical']:.0%}, <1% {o['share_nearly_neutral_under_1pct_of_scale']:.0%}, >25% {o['share_over_25pct_of_scale']:.0%}, >50% {o['share_over_50pct_of_scale']:.0%}, >=stranger {o['share_as_far_as_a_stranger']:.0%}", flush=True)
    json.dump(results, open(os.path.join(HERE, "b1.json"), "w"), indent=1)
