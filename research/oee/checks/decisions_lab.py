"""
What a lab world's brains decide, at one archived iteration: a profile of every agent's decisions.

For every agent (after one game, so messages are fresh; random inputs drawn once):
the share of its tokens its brain would give a child, whether that would make a
child (at least min_child and the parent not crowded), the share of candidates a
newborn would be linked to, whether it spreads its stake or goes all in, and the
share of its tokens it keeps at home.

    python3 research/oee/checks/decisions_lab.py RUN:ITERATION [...]
"""
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import gol_store as store
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, _apportion, _share_of_first


def profile(run, it):
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    w = GraphOfLife.from_checkpoint(store.load_archive(run, it), cfg)
    w.step(record_decisions=False)
    w.rng = np.random.RandomState(5)
    log_deg, neighs, q_tok, q_deg, log_tok, at_risk = w._precompute_features()
    H = w.heads
    out = {"repro": [], "child": [], "child_tokens": [], "link": [], "spread": [], "home": [], "tokens": []}
    for u in sorted(w.G.nodes()):
        t = int(w.tokens.get(u, 0))
        if t <= 0:
            continue
        cands = [u] + list(neighs[u])
        Y = w._observe(u, cands, log_deg, q_tok, q_deg, log_tok, at_risk)
        frac = np.mean(Y[H["REPRO_FRACTION"], :], axis=1)
        share = _share_of_first(frac[0], frac[1])
        wanted = int(np.floor(share * t))
        child = w._endowment(u, t, wanted)
        L = Y[H["LINK"], :]
        link = float(np.mean([_share_of_first(L[0, c], L[1, c]) for c in range(1, len(cands))])) if len(cands) > 1 else 0.0
        mode = np.mean(Y[H["BLOTTO_MODE"], :], axis=1)
        scores = np.asarray(Y[H["BLOTTO"], :], float)
        kept = max(1, int(np.ceil(cfg.keep_home * t))) if cfg.keep_home > 0 else 0
        stake = t - kept
        a = _apportion(scores, stake) if mode[0] > mode[1] else np.eye(len(cands), dtype=int)[int(np.argmax(scores))] * stake
        a[0] += kept
        out["repro"].append(share)
        out["child"].append(1.0 if child >= 1 else 0.0)
        out["child_tokens"].append(child if child >= 1 else np.nan)
        out["link"].append(link)
        out["spread"].append(1.0 if mode[0] > mode[1] else 0.0)
        out["home"].append(a[0] / t)
        out["tokens"].append(t)
    tok = np.array(out["tokens"])
    rich = tok >= 2 * cfg.min_child
    return {"agents": len(tok), "rich (>= 2 min_child)": float(rich.mean()),
            "repro share": float(np.mean(out["repro"])),
            "would have a child": float(np.mean(out["child"])),
            "of the rich, would have a child": float(np.mean(np.array(out["child"])[rich])) if rich.any() else float("nan"),
            "child tokens (median)": float(np.nanmedian(out["child_tokens"])) if np.any(np.isfinite(out["child_tokens"])) else float("nan"),
            "link share": float(np.mean(out["link"])), "spread": float(np.mean(out["spread"])),
            "home share": float(np.mean(out["home"]))}


if __name__ == "__main__":
    for target in sys.argv[1:]:
        run, it = target.rsplit(":", 1)
        p = profile(run, int(it))
        print(run, it, {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()}, flush=True)
