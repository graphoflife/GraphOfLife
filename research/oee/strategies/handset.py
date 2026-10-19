"""
A raider written in a brain's own weights (Chapter 94): the checks.

manual.py's raider is a rule in Python: it reads the input rows an evolved brain reads and writes the
output rows it writes. gol_strategies.raider_weights sets the same rule by hand into a brain of B2's
shape — 66 inputs, one hidden layer of 16 sigmoid units, 19 linear outputs, the weights in float16 —
and says unit by unit how. This file checks it against the rule on situations a raider actually met,
and plants it in the rule's place.

    python3 handset.py record B2-10000-s002 --era 2000 --T 40 --seed 1      a short trial: plants
        manual.py's raiders as manual.py does and keeps every situation a raider met
    python3 handset.py check FILES... [--out agreement.json]                 decision by decision
    python3 handset.py save                                                  writes raider_B2.npz here
    python3 handset.py invade B2-10000-s002 --seeds 1,2 --T 300              manual.py's invasion with
        the weight-file raider and with the rule, on the same seeds
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))

import manual as M  # noqa: E402  (research/oee/pilots/manual.py)

OUT = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "strategies")
EPS = 0.02                      # manual.py's password band


def raider_brain(cfg, margin: float = 0.69):
    """The raider's weights, as the engine sets them (gol_strategies.raider_weights)."""
    import gol_strategies
    return gol_strategies.raider_weights(cfg, margin)


# ---------------------------------------------------------------------------
# Situations a raider met
# ---------------------------------------------------------------------------

def record(run: str, era: int, T: int, seed: int, share: float = 0.05) -> str:
    """Plant manual.py's raiders as manual.invade does, play T iterations, keep every X a raider read."""
    import gol_store as store
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    cfg.extinction_threshold = 0
    world = GraphOfLife.from_checkpoint(store.load_archive(run, era), cfg)
    world.rng = np.random.RandomState(seed)
    rng = np.random.RandomState(1000 + seed)
    kept: List[np.ndarray] = []

    def recording(cfg_, X):
        kept.append(np.array(X, dtype=float))
        return M.raider(cfg_, X)

    patch = M.patch_of(world, share, rng)
    for u in patch:
        world.brains[u] = M.ManualBrain(cfg, M.MANUAL_ID, policy=recording)
    for t in range(T):
        world.step(record_decisions=False)
        mine = sum(1 for u in world.G.nodes() if world.brains[u].brain_id >= M.MANUAL_ID)
        if not mine:
            break
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"raider_situations_{run}_{era}_s{seed}.npz")
    widths = np.array([x.shape[1] for x in kept])
    np.savez_compressed(path, X=np.concatenate(kept, axis=1), widths=widths, run=run, era=era, seed=seed,
                        T=T, raiders_left=mine)
    print(f"{run} at {era}, seed {seed}: {len(kept):,} situations over {t + 1} iterations, "
          f"{mine} raiders left", flush=True)
    return path


def situations(path: str) -> List[np.ndarray]:
    with np.load(path) as z:
        X, widths = z["X"], z["widths"]
    cuts = np.concatenate([[0], np.cumsum(widths)])
    return [X[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]


# ---------------------------------------------------------------------------
# Deciding as the engine does
# ---------------------------------------------------------------------------

def decisions(cfg, heads, X: np.ndarray, Y: np.ndarray) -> Dict[str, object]:
    """What the engine would do with Y: the stake, the child, the child's links, the message, revolts."""
    from GraphOfLifeSimple import _apportion, _share_of_first
    lay = cfg.input_layout()
    me = int(round(np.expm1(X[lay["own_tokens"][0], 0])))
    kept = max(1, int(np.ceil(cfg.keep_home * me))) if cfg.keep_home > 0 else 0
    stake = me - kept
    scores = Y[heads["BLOTTO"], :]
    mode = np.mean(Y[heads["BLOTTO_MODE"], :], axis=1)
    if mode[0] > mode[1]:
        alloc = _apportion(scores, stake)
    else:
        alloc = np.zeros(X.shape[1], dtype=int)
        alloc[int(np.argmax(scores))] = stake
    alloc[0] += kept
    frac = np.mean(Y[heads["REPRO_FRACTION"], :], axis=1)
    wanted = int(np.floor(_share_of_first(frac[0], frac[1]) * me))
    wanted = max(0, min(me, wanted))
    if cfg.keep_home > 0:
        wanted = min(wanted, me - 1)
    child = wanted if wanted >= cfg.min_child else 0
    lk, lm = Y[heads["LINK"], :], Y[heads["LINK_MODE"], :]
    links = tuple(int(c) for c in range(1, X.shape[1])
                  if not lm[0, c] > lm[1, c] and lk[0, c] > lk[1, c])
    start = heads["MESSAGE_START"]
    said = np.tanh(Y[start:start + cfg.message_amount, :])
    rev = Y[heads["REV_FRACTION"], :] if "REV_FRACTION" in heads else None
    revolts = 0 if rev is None else sum(int(np.floor(_share_of_first(rev[0, c], rev[1, c]) * alloc[c]))
                                       for c in range(X.shape[1]))
    return {"alloc": alloc, "child": child, "links": links if child else (), "said": said, "revolts": revolts}


def check(paths: List[str], margin: float) -> Dict[str, float]:
    import gol_store as store
    from gol_config import SimConfig
    from GraphOfLifeSimple import build_heads
    counts: Dict[str, int] = {}
    total = 0
    for path in paths:
        with np.load(path) as z:
            run = str(z["run"])
        cfg = SimConfig.from_dict(store.load_meta(run)["config"])
        heads = build_heads(cfg)
        brain = raider_brain(cfg, margin)
        for X in situations(path):
            a = decisions(cfg, heads, X, M.raider(cfg, X))
            b = decisions(cfg, heads, X, brain.forward(X))
            v = M.View(cfg, X)
            role = np.where(v.is_self, 0, np.where(v.mate, 1, 2))
            ta, tb = int(np.argmax(a["alloc"][1:] > 0)) + 1 if a["alloc"][1:].any() else 0, \
                int(np.argmax(b["alloc"][1:] > 0)) + 1 if b["alloc"][1:].any() else 0
            same = {
                "stake (same column)": np.array_equal(a["alloc"], b["alloc"]),
                "stake (same kind of target)": (role[ta], v.tokens[ta]) == (role[tb], v.tokens[tb]),
                "child": a["child"] == b["child"],
                "child's links": a["links"] == b["links"],
                "password heard by teammates": bool(np.all(np.abs(b["said"][:M.SIGN, :]
                                                                   - M.SIGNATURE[:, None]) < EPS)),
                "no revolution tokens": a["revolts"] == b["revolts"] == 0,
            }
            same["every decision"] = (same["stake (same kind of target)"] and same["child"]
                                      and same["child's links"] and same["password heard by teammates"]
                                      and same["no revolution tokens"])
            for k, ok in same.items():
                counts[k] = counts.get(k, 0) + int(ok)
            total += 1
    return {k: c / total for k, c in counts.items()} | {"situations": total}


def save(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raider_B2.npz")) -> str:
    """The weights, for B2's layout (any B2 run's settings give the same brain)."""
    import gol_store as store
    from gol_config import SimConfig
    cfg = SimConfig.from_dict(store.load_meta("B2-10000-s002")["config"])
    b = raider_brain(cfg)
    np.savez(path, W0=b.weights[0], b0=b.biases[0], W1=b.weights[1], b1=b.biases[1],
             inputs=cfg.n_inputs(), hidden=cfg.hidden_layers[0], outputs=cfg.n_outputs(),
             what="manual.py's raider as a B2 brain (research/oee/strategies/handset.py)")
    print(path, flush=True)
    return path


def invade(task) -> Dict[str, object]:
    """manual.invade with the weight-file raider, frozen (it never mutates), or with the rule itself."""
    run, era, seed, T, share, rule = task
    import gol_store as store
    import gol_strategies
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    cfg.extinction_threshold = 0
    world = GraphOfLife.from_checkpoint(store.load_archive(run, era), cfg)
    world.rng = np.random.RandomState(seed)
    patch = gol_strategies.plant(world, "raider" if rule else "raider-weights", share, seed)
    series = []
    for t in range(T + 1):
        if t:
            world.step(record_decisions=False)
        ids = list(world.G.nodes())
        mine = [u for u in ids if world.brains[u].brain_id >= M.MANUAL_ID]
        series.append({"t": t, "agents": len(ids), "mine": len(mine),
                       "tokens": int(sum(world.tokens[u] for u in mine))})
        if not mine or not ids:
            break
    return {"run": run, "era": era, "seed": seed, "rule": rule, "patch": len(patch), "series": series}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("run")
    r.add_argument("--era", type=int, default=2000)
    r.add_argument("--T", type=int, default=40)
    r.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("check")
    c.add_argument("paths", nargs="+")
    c.add_argument("--margin", type=float, default=0.69)
    c.add_argument("--out", default="", help="write each file's agreement here, as JSON")
    sub.add_parser("save")
    i = sub.add_parser("invade")
    i.add_argument("runs", nargs="+")
    i.add_argument("--era", type=int, default=2000)
    i.add_argument("--seeds", default="1")
    i.add_argument("--T", type=int, default=300)
    i.add_argument("--patch", type=float, default=0.05)
    i.add_argument("--jobs", type=int, default=4)
    i.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.run, a.era, a.T, a.seed)
    elif a.cmd == "save":
        save()
    elif a.cmd == "invade":
        import json
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(r, a.era, int(sd), a.T, a.patch, rule) for r in a.runs for sd in a.seeds.split(",")
                 for rule in (True, False)]
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(invade, tasks))
        for r in results:
            last = r["series"][-1]
            print(f"{r['run']} seed {r['seed']} {'rule   ' if r['rule'] else 'weights'}: "
                  f"{last['mine']}/{last['agents']} = {last['mine'] / max(1, last['agents']):.3f} at t={last['t']}",
                  flush=True)
        if a.out:
            json.dump(results, open(a.out, "w"))
    else:
        import json
        found = {}
        for path in a.paths:
            found[os.path.basename(path)] = got = check([path], a.margin)
            print(os.path.basename(path))
            for k, v in got.items():
                print(f"  {k:30s} {v:.4f}" if k != "situations" else f"  {k:30s} {v:,}")
        if a.out:
            json.dump({"margin": a.margin, "files": found}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
