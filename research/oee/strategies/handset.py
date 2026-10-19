"""
A raider written in a brain's own weights (Chapter 94).

manual.py's raider is a rule in Python: it reads the input rows an evolved brain reads and writes the
output rows it writes. Here the same rule is set by hand into a B2 brain — 66 inputs, one hidden layer
of 16 sigmoid units, 19 linear outputs, the weights stored in float16 — and checked against the rule
on situations a raider actually met. If they decide alike, the raider is a brain B2 could grow.

The hidden units (four are left at zero):
  0      this column is the agent itself                       (the `self` row)
  1-8    the password: for each of the first four numbers a neighbour wrote, a wall at s - eps and one
         at s + eps; the lower minus the upper is 1 inside the band and 0 outside, so the four bands
         add to 4 for a teammate and to about 0 for anyone else. The self row shuts every wall, so the
         agent's own column never counts as a teammate: what it heard from itself is a phase old, and
         a node just won or an agent just born has not spoken yet
  9      the neighbour can be taken: log1p(own tokens) - log1p(its tokens) beyond a margin
  10     how weak the neighbour is: falling with its tokens
  11     rich enough for a child: 30 tokens or more
The outputs:
  BLOTTO        A * takeable + D * weak - P * password + Q * self, read all-in (BLOTTO_MODE 0, 1): the
                weakest stranger that can be taken, else home, never a teammate
  REPRO         (r * rich, 1 - r * rich): a third of the tokens to a child when rich (r a hair above 1/3,
                so that float16 rounding does not take a token off the child)
  LINK          yes = self + password / 4 against no = 1/2, decided (LINK_MODE 0, 1): the child joins
                its parent's teammates
  REV_FRACTION  (0, 1): no revolution tokens
  MESSAGE       atanh of the signature, then zeros: the password

    python3 handset.py record B2-10000-s002 --era 2000 --T 40 --seed 1      a short trial: plants
        manual.py's raiders as manual.py does and keeps every situation a raider met
    python3 handset.py check FILES...                                        decision by decision
    python3 handset.py save                                                  writes raider_B2.npz here
    python3 handset.py invade B2-10000-s002 --seeds 1,2 --T 300 [--manual]   manual.py's invasion with
        the weight-file raider (frozen, as manual.py's strategies are) or, with --manual, the rule
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
WALL = 10000.0                  # steepness of the password walls: 200 steps inside the band, and no
                                # remainder outside it that could tip a choice between equal targets
STEEP = 1500.0                  # steepness of the rich unit: 29 and 30 tokens lie 0.033 apart in log1p
TAKE = 6000.0                   # steepness of the takeable unit: 3 against 1 lies 0.003 past the margin
#: the score's weights: takeable, weak, password, self (see the conditions in `raider_brain`)
A, D, P, Q = 40.0, 1.0, 10.0, 5.0
SHUT = 30000.0                  # what the self row takes off every password wall


def raider_brain(cfg, margin: float = 0.69):
    """
    The raider as a B2 brain. `margin` is where "can be taken" falls in log1p space: the rule's own
    line, 0.7 * mine > 1.5 * theirs, is log(15/7) = 0.762 for large holdings and lower for small ones,
    and no straight line in log1p space draws it exactly (13 against 6 is taken, 15 against 7 is not,
    and both differ by log 2). Just below log 2 it matches every pair the recorded raiders met, since
    they hold one to a few tokens: 3 against 1 is taken, 2 against 1 and 6 against 3 are not.

    The score keeps three orders with the margins chosen: a stranger that can be taken above home even
    when its messages happen to match three of the password's four numbers (A - 3P > Q + D), home above
    a stranger that cannot be taken (Q > D), home above any teammate (Q + 4P > A + D).
    """
    from GraphOfLifeSimple import BRAIN_KINDS, build_heads
    lay = cfg.input_layout()
    heads = build_heads(cfg)
    brain = BRAIN_KINDS[cfg.brain_kind](cfg, M.MANUAL_ID)
    n_in, n_hid, n_out = cfg.n_inputs(), cfg.hidden_layers[0], cfg.n_outputs()
    W1, b1 = np.zeros((n_hid, n_in)), np.zeros((n_hid, 1))
    W2, b2 = np.zeros((n_out, n_hid)), np.zeros((n_out, 1))

    self_row, own, its = lay["self"][0], lay["own_tokens"][0], lay["its_tokens"][0]
    heard = lay["message_from_it"][0]
    W1[0, self_row], b1[0] = 40.0, -20.0
    for i, s in enumerate(M.SIGNATURE):
        lo, hi = 1 + 2 * i, 2 + 2 * i
        W1[lo, heard + i], b1[lo] = WALL, -WALL * (s - EPS)
        W1[hi, heard + i], b1[hi] = WALL, -WALL * (s + EPS)
        W1[lo, self_row] = W1[hi, self_row] = -SHUT
    W1[9, own], W1[9, its], b1[9] = TAKE, -TAKE, -TAKE * margin
    W1[10, its], b1[10] = -1.0, 3.0
    W1[11, own], b1[11] = STEEP, -STEEP * 0.5 * (np.log1p(29) + np.log1p(30))

    blotto = heads["BLOTTO"]
    W2[blotto, 9], W2[blotto, 10], W2[blotto, 0] = A, D, Q
    for i in range(4):
        W2[blotto, 1 + 2 * i], W2[blotto, 2 + 2 * i] = -P, P
    mode = heads["BLOTTO_MODE"]
    b2[mode.start + 1] = 1.0                            # all-in on the best
    rf = heads["REPRO_FRACTION"]
    third = float(np.float16(1 / 3) + np.float16(2 ** -12))   # the float16 just above a third
    W2[rf.start, 11], W2[rf.start + 1, 11], b2[rf.start + 1] = third, -third, 1.0
    link, lmode = heads["LINK"], heads["LINK_MODE"]
    W2[link.start, 0] = 1.0
    for i in range(4):
        W2[link.start, 1 + 2 * i], W2[link.start, 2 + 2 * i] = 0.25, -0.25
    b2[link.start + 1] = 0.5
    b2[lmode.start + 1] = 1.0                           # decide, do not draw
    if "REV_FRACTION" in heads:
        b2[heads["REV_FRACTION"].start + 1] = 1.0
    start = heads["MESSAGE_START"]
    b2[start:start + M.SIGN, 0] = M.atanh(M.SIGNATURE)

    brain.weights = [W1.astype(brain.dtype), W2.astype(brain.dtype)]
    brain.biases = [b1.astype(brain.dtype), b2.astype(brain.dtype)]
    for w in brain.weights + brain.biases:
        assert np.all(np.isfinite(w)), "a weight float16 cannot hold"
    return brain


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
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    cfg.extinction_threshold = 0
    world = GraphOfLife.from_checkpoint(store.load_archive(run, era), cfg)
    world.rng = np.random.RandomState(seed)
    rng = np.random.RandomState(1000 + seed)
    patch = M.patch_of(world, share, rng)
    rng.randint(len(patch))                         # manual.invade draws its control's brain here
    brain = None if rule else raider_brain(cfg)
    for u in patch:
        world.brains[u] = (M.ManualBrain(cfg, M.MANUAL_ID, policy=M.raider) if rule
                           else M.ManualBrain(cfg, M.MANUAL_ID, frozen=brain))
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
