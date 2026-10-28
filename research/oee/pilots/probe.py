"""
What the agents of a checkpoint do: their decisions on the world as it stood.

For every agent (or a sample) the brain is run on its real candidate set, and
the decisions are read off the heads exactly as the engine reads them, without
acting on any of them:

  repro    the share of its tokens a child would get
  link     the share of candidates a newborn would be joined to (expected)
  spread   whether it spreads its stake (1) or goes all in (0)
  home     the share of its stake it keeps on its own node
  hhi      how concentrated its stake on the neighbours is (sum of squared shares)
  rev      the revolutionary share of what it stakes on others
  hand     the share of its connections it would hand to a newborn
  gain     RMS output change per RMS input change (inputs jittered by 0.1)

    python3 probe.py RUNDIR [--eras 0,500,1000] [--sample 300]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from GraphOfLifeSimple import _apportion, _share_of_first
from proto import ProtoWorld, ProtoConfig, set_proto


def p_yes(yes, no, myes, mno):
    """Probability that _choose_binary says yes."""
    if myes > mno:
        return _share_of_first(yes, no)
    if yes == no:
        return 0.5
    return float(yes > no)


def probe(run, it, sample, rng):
    meta = json.load(open(os.path.join(run, "config.json")))
    set_proto(meta.get("proto", {}))
    if meta["cfg"].get("brain_kind") == "grow":
        import grow
        grow.install()
    cfg = ProtoConfig.from_dict(meta["cfg"], stored=False)
    with np.load(os.path.join(run, f"ckpt_{it:06d}.npz")) as z:
        blob = {k: z[k] for k in z.files}
    w = ProtoWorld.from_checkpoint(blob, cfg)
    import grow
    grow.restore_active(w, blob)
    if w.G.number_of_nodes() < 2:
        return None
    feats = w._precompute_features()
    log_deg, neighs, q_tok, q_deg, log_tok, at_risk = feats
    H = w.heads
    nodes = sorted(w.G.nodes())
    if sample and len(nodes) > sample:
        nodes = list(rng.choice(nodes, sample, replace=False))
    out = {k: [] for k in ("repro", "link", "spread", "home", "hhi", "rev", "hand", "gain",
                           "seek", "accept")}
    for u in nodes:
        cands = [u] + list(neighs[u])
        X = w._inputs(u, cands, *feats[:0], log_deg, q_tok, q_deg, log_tok, at_risk) \
            if False else w._inputs(u, cands, log_deg, q_tok, q_deg, log_tok, at_risk)
        Y = w.brains[u].forward(X)
        eps = rng.normal(0, 0.1, size=X.shape)
        Y2 = w.brains[u].forward(X + eps)
        out["gain"].append(float(np.sqrt(np.mean((Y2 - Y) ** 2)) / np.sqrt(np.mean(eps ** 2))))
        frac = np.mean(Y[H["REPRO_FRACTION"], :], axis=1)
        out["repro"].append(_share_of_first(frac[0], frac[1]))
        L, LM = Y[H["LINK"], :], Y[H["LINK_MODE"], :]
        out["link"].append(np.mean([p_yes(L[0, c], L[1, c], LM[0, c], LM[1, c]) for c in range(len(cands))]))
        mode = np.mean(Y[H["BLOTTO_MODE"], :], axis=1)
        spread = bool(mode[0] > mode[1])
        out["spread"].append(float(spread))
        t = max(1, int(w.tokens[u]))
        scores = np.asarray(Y[H["BLOTTO"], :], dtype=float)
        if spread:
            alloc = _apportion(scores, t).astype(float)
        else:
            alloc = np.zeros(len(cands)); alloc[int(np.argmax(scores))] = t
        out["home"].append(alloc[0] / t)
        others = alloc[1:]
        out["hhi"].append(float(np.sum((others / others.sum()) ** 2)) if others.sum() > 0 else np.nan)
        if "REV_FRACTION" in H and len(cands) > 1:
            R = Y[H["REV_FRACTION"], :]
            out["rev"].append(np.mean([_share_of_first(R[0, c], R[1, c]) for c in range(1, len(cands))]))
        if "HANDOVER" in H and len(cands) > 1:
            Hd, HM = Y[H["HANDOVER"], :], Y[H["HANDOVER_MODE"], :]
            out["hand"].append(np.mean([p_yes(Hd[0, c], Hd[1, c], HM[0, c], HM[1, c]) for c in range(1, len(cands))]))
        if "SEEK" in H:
            s = np.mean(Y[H["SEEK"], :], axis=1)
            out["seek"].append(_share_of_first(s[0], s[1]))
            A = Y[H["ACCEPT"], :]
            out["accept"].append(np.mean([p_yes(A[0, c], A[1, c], LM[0, c], LM[1, c]) for c in range(len(cands))]))
    return {k: (float(np.nanmedian(v)) if v else None) for k, v in out.items()} | {
        "n": w.G.number_of_nodes(), "mean_home": float(np.mean(out["home"])),
        "mean_spread": float(np.mean(out["spread"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--eras", default="")
    ap.add_argument("--sample", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    for run in a.runs:
        eras = ([int(x) for x in a.eras.split(",")] if a.eras else
                sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(run, "ckpt_*.npz"))))
        print(run)
        print(f"{'it':>6s} {'n':>5s} " + " ".join(f"{k:>7s}" for k in
              ("repro", "link", "spread", "home", "hhi", "rev", "hand", "gain", "seek", "accept")))
        for it in eras:
            if not os.path.exists(os.path.join(run, f"ckpt_{it:06d}.npz")):
                continue
            r = probe(run, it, a.sample, rng)
            if r is None:
                print(f"{it:6d}  (world too small)")
                continue
            print(f"{it:6d} {r['n']:5d} " + " ".join(
                f"{r[k]:7.3f}" if r[k] is not None else f"{'-':>7s}" for k in
                ("repro", "link", "mean_spread", "mean_home", "hhi", "rev", "hand", "gain", "seek", "accept")))


if __name__ == "__main__":
    main()
