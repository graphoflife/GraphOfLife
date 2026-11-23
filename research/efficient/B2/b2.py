"""Card B2: how many different things do a world's brains do, and what is the common policy? Nothing here plays a world."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe as P
import gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES, BRAINS = 300, 1000


def components(M):
    """How many principal components hold 90% of the variance between brains, and the participation ratio."""
    C = M - M.mean(axis=0)
    s = np.linalg.svd(C, compute_uv=False) ** 2
    if s.sum() == 0:
        return 0, 0.0
    share = np.cumsum(s) / s.sum()
    return int(np.searchsorted(share, 0.9) + 1), float(s.sum() ** 2 / (s ** 2).sum())


def card(name, run_id, eras):
    cfg = P.config(run_id)
    all_eras = gol_store.archived(run_id)
    asked = P.probes(run_id, all_eras[len(all_eras) // 2], PROBES)
    tokens = np.array([p["tokens"] for p in asked]); cands = np.array([p["candidates"] for p in asked])
    rng = np.random.RandomState(5)
    out = {"world": name, "run": run_id, "eras": []}
    for era in eras:
        blob = gol_store.load_archive(run_id, era)
        n = len(blob["ids"])
        rows = np.sort(rng.choice(n, size=min(BRAINS, n), replace=False))
        beh = P.behaviour(*P.layers(blob, rows), asked, cfg)
        home, child = beh["home"].astype(float), (beh["child"] > 0).astype(float)
        M = np.concatenate([home, child], axis=1)
        k90, pr = components(M)
        # How alike the brains are: the share of (brain, probe) decisions equal to the most common decision on that probe.
        def with_mode(x):
            vals = [np.bincount(x[:, j]).max() / x.shape[0] for j in range(x.shape[1])]
            return float(np.mean(vals))
        agree_first, agree_child = with_mode(beh["first"]), with_mode(beh["child"])
        geno = blob["brain_ids"][rows]
        uniq, counts = np.unique(geno, return_counts=True)
        # The common policy, in a table: by the tokens the agent holds.
        policy = []
        for lo, hi, label in ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 9, "4-9"), (10, 10 ** 9, "10+")):
            sel = (tokens >= lo) & (tokens <= hi)
            if sel.sum():
                policy.append({"tokens": label, "probes": int(sel.sum()), "first_token_home": float(home[:, sel].mean()),
                               "has_child": float(child[:, sel].mean()),
                               "child_tokens_share": float((beh["child"][:, sel] / tokens[sel]).mean())})
        out["eras"].append({"era": era, "agents": n, "tokens_each": float(blob["tokens"].mean()), "brains": len(rows),
                            "genotypes_among_them": int(len(uniq)), "largest_clone_share": float(counts.max() / len(rows)),
                            "components_for_90pct": k90, "participation_ratio": pr,
                            "agree_with_most_common_first": agree_first, "agree_with_most_common_child": agree_child,
                            "first_token_home": float(home.mean()), "has_child": float(child.mean()), "policy": policy})
    out["probe_tokens"] = {str(k): int((tokens == k).sum()) for k in sorted(set(tokens.tolist()))[:8]}
    out["probe_candidates_median"] = float(np.median(cands))
    return out


if __name__ == "__main__":
    res = []
    for name in ("s4003", "s4008", "s4002"):
        eras = gol_store.archived(P.RUNS[name])
        pick = [e for e in eras if e in (0, 2500, 5000, 10000, 15000)] + ([eras[-1]] if eras[-1] > 15000 else [])
        r = card(name, P.RUNS[name], pick); res.append(r)
        print(f"\n{name}: probes hold tokens {r['probe_tokens']}, median {r['probe_candidates_median']:.0f} candidates")
        for e in r["eras"]:
            print(f"  era {e['era']:>6}: {e['agents']} agents, {e['tokens_each']:.2f} tokens each; {e['genotypes_among_them']} genotypes in {e['brains']} brains (largest clone {e['largest_clone_share']:.0%}); "
                  f"90% of variance in {e['components_for_90pct']} components (participation {e['participation_ratio']:.1f}); "
                  f"same as most common: first token {e['agree_with_most_common_first']:.0%}, child {e['agree_with_most_common_child']:.0%}; home first {e['first_token_home']:.0%}, child {e['has_child']:.0%}")
        last = r["eras"][-1]
        for p in last["policy"]:
            print(f"     at era {last['era']}, holding {p['tokens']:>4} tokens ({p['probes']} probes): first token home {p['first_token_home']:.0%}, has a child {p['has_child']:.0%}, gives it {p['child_tokens_share']:.0%} of its tokens", flush=True)
    json.dump(res, open(os.path.join(HERE, "b2.json"), "w"), indent=1)
