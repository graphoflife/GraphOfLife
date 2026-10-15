"""
Hand-written strategies in place of brains, and an invasion test.

A strategy reads exactly what a brain reads — the input rows X, one column per
candidate, column 0 the agent itself — and writes exactly what a brain writes:
the output rows Y that the engine turns into decisions. So it plugs into the
world where a brain does and can do nothing an evolved brain could not.

Teammates recognise each other by what they write: the first four numbers of
every message a strategy sends are a fixed signature (an evolved brain could
write the same; it is a password, not a privilege). The fifth carries a number
a strategy wants its teammates to know, such as its distance from the frontier.

The invasion test: take an evolved world from a run's archive, give a
contiguous patch of agents (the nearest few per cent of a random agent) a
strategy, play on under the world's own rules, and follow the share of agents
the strategy holds. A control patch gets frozen copies of the world's own
brains (the same brains, marked and not mutated), so a strategy is judged
against what a patch of the residents would do from the same place.

    python3 manual.py RUN --era 2000 --strategies factory,turtle,raider,resident \\
        --seeds 1,2 --T 300 --patch 0.05 --jobs 3 --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

MANUAL_ID = 1_900_000_000          # brain ids at or above this belong to a strategy
SIGNATURE = np.array([0.9, -0.9, 0.9, -0.9])
SIGN = 4                            # message numbers taken by the signature
CAP = 19                            # distances are sent as d / 20, d at most this


def atanh(v):
    return np.arctanh(np.clip(v, -0.999, 0.999))


class View:
    """What one look shows a strategy, read off the input rows."""

    def __init__(self, cfg, X):
        lay = cfg.input_layout()
        row = lambda name: X[lay[name][0]:lay[name][1], :]
        self.n = X.shape[1]
        self.me_tokens = float(np.expm1(row("own_tokens")[0, 0]))
        self.tokens = np.expm1(row("its_tokens")[0, :])
        self.is_self = row("self")[0, :] > 0.5
        heard = row("message_from_it")            # what each candidate wrote to me
        self.mate = np.all(np.abs(heard[:SIGN, :] - SIGNATURE[:, None]) < 0.02, axis=0) & ~self.is_self
        self.word = np.round(heard[SIGN, :] * 20) if heard.shape[0] > SIGN else np.zeros(self.n)
        self.enemy = ~self.mate & ~self.is_self


class Writer:
    """Builds the output rows from plain decisions."""

    def __init__(self, cfg, n):
        from GraphOfLifeSimple import build_heads
        self.h = build_heads(cfg)
        self.cfg = cfg
        self.Y = np.zeros((cfg.n_outputs(), n))
        self.child(0.0)
        self.no_links()
        self.revolt(0.0)
        self.say(0)

    def child(self, share: float) -> None:
        """Give this share of my tokens to a child (0: none)."""
        s = self.h["REPRO_FRACTION"]
        self.Y[s.start, :], self.Y[s.start + 1, :] = share, 1.0 - share

    def no_links(self) -> None:
        L, M = self.h["LINK"], self.h["LINK_MODE"]
        self.Y[L.start, :], self.Y[L.start + 1, :] = 0.0, 1.0
        self.Y[M.start, :], self.Y[M.start + 1, :] = 0.0, 1.0     # decide, do not draw

    def link(self, cols) -> None:
        """Join a newborn to these candidates."""
        L = self.h["LINK"]
        self.Y[L.start, cols], self.Y[L.start + 1, cols] = 1.0, 0.0

    def stake(self, amounts) -> None:
        """Spread the staked tokens in proportion to `amounts` (all zero: everything at home)."""
        a = np.maximum(0.0, np.asarray(amounts, dtype=float))
        if a.sum() <= 0:
            a = np.zeros(len(a))
            a[0] = 1.0
        self.Y[self.h["BLOTTO"], :] = a
        m = self.h["BLOTTO_MODE"]
        self.Y[m.start, :], self.Y[m.start + 1, :] = 1.0, 0.0        # spread, as given

    def revolt(self, share: float) -> None:
        if "REV_FRACTION" in self.h:
            r = self.h["REV_FRACTION"]
            self.Y[r.start, :], self.Y[r.start + 1, :] = share, 1.0 - share

    def say(self, word: float) -> None:
        start = self.h["MESSAGE_START"]
        m = self.cfg.message_amount
        vec = np.zeros(m)
        vec[:SIGN] = SIGNATURE
        if m > SIGN:
            vec[SIGN] = min(word, CAP) / 20.0
        self.Y[start:start + m, :] = atanh(vec)[:, None]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def turtle(cfg, X):
    """Everything at home; a child of a third when rich; the child joined to its parent alone."""
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    w.stake(np.eye(v.n)[0])
    if v.me_tokens >= 30:
        w.child(1 / 3)
    return w.Y


def raider(cfg, X):
    """Alone: all on the weakest stranger next door if that can take it, else everything at home."""
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    staked = 0.7 * v.me_tokens                     # B2 keeps at least 30% at home anyway
    target = None
    if v.enemy.any():
        j = int(np.argmin(np.where(v.enemy, v.tokens, np.inf)))
        if staked > 1.5 * v.tokens[j]:
            target = j
    w.stake(np.eye(v.n)[target if target is not None else 0])
    if v.me_tokens >= 30:
        w.child(1 / 3)
        w.link(np.flatnonzero(v.mate | v.is_self))
    return w.Y


def lone(cfg, X):
    """The raider without a team: no password, every neighbour a stranger, children joined to nobody chosen."""
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    w.Y[w.h["MESSAGE_START"]:w.h["MESSAGE_START"] + cfg.message_amount, :] = 0.0
    others = ~v.is_self
    target = None
    if others.any():
        j = int(np.argmin(np.where(others, v.tokens, np.inf)))
        if 0.7 * v.me_tokens > 1.5 * v.tokens[j]:
            target = j
    w.stake(np.eye(v.n)[target if target is not None else 0])
    if v.me_tokens >= 30:
        w.child(1 / 3)
    return w.Y


def factory(cfg, X):
    """
    A team with a pipeline. Every member knows how far it is from the team's
    frontier — 0 if a stranger is next door, else one more than the nearest
    teammate says — and writes it to its teammates. The interior pumps its
    tokens downhill, towards the frontier; the frontier spends everything it can
    stake on the weakest stranger next door when that can take it, and otherwise
    holds. The interior breeds when rich, and joins its children to teammates.
    """
    v, w = View(cfg, X), Writer(cfg, X.shape[1])
    frontier = bool(v.enemy.any())
    if frontier:
        d = 0
    else:
        heard = v.word[v.mate]
        d = int(min(CAP, (heard.min() if heard.size else CAP - 1) + 1))
    w.say(d)
    staked = 0.7 * v.me_tokens
    amounts = np.zeros(v.n)
    if frontier:
        j = int(np.argmin(np.where(v.enemy, v.tokens, np.inf)))
        if staked > 1.5 * v.tokens[j]:
            amounts[j] = 1.0
    elif v.mate.any():
        nearest = np.where(v.mate, v.word, np.inf)
        amounts[int(np.argmin(nearest))] = 1.0
    w.stake(amounts)
    if not frontier and v.me_tokens >= 30:
        w.child(1 / 3)
        w.link(np.flatnonzero(v.mate | v.is_self))
    return w.Y


STRATEGIES: Dict[str, Callable] = {"turtle": turtle, "raider": raider, "lone": lone, "factory": factory}


class ManualBrain:
    """A brain that is a strategy, or a frozen resident brain marked as one."""

    __slots__ = ("cfg", "brain_id", "parent_brain_id", "policy", "frozen", "weights", "biases")

    def __init__(self, cfg, brain_id, policy=None, frozen=None):
        self.cfg, self.brain_id, self.parent_brain_id = cfg, brain_id, -1
        self.policy, self.frozen = policy, frozen
        self.weights = frozen.weights if frozen is not None else []
        self.biases = frozen.biases if frozen is not None else []

    def forward(self, x):
        return self.frozen.forward(x) if self.frozen is not None else self.policy(self.cfg, np.asarray(x, float))

    def copy(self):
        b = ManualBrain(self.cfg, self.brain_id, self.policy, self.frozen)
        b.parent_brain_id = self.parent_brain_id
        return b

    def mutate(self, brain_id, rng):
        return False


# ---------------------------------------------------------------------------
# The invasion test
# ---------------------------------------------------------------------------

def patch_of(world, share, rng):
    """The agents nearest a random agent, by steps, until `share` of the world."""
    from collections import deque
    nodes = sorted(world.G.nodes())
    want = max(1, int(round(share * len(nodes))))
    start = nodes[rng.randint(len(nodes))]
    seen, order, queue = {start}, [], deque([start])
    while queue and len(order) < want:
        u = queue.popleft()
        order.append(u)
        for x in sorted(world.G[u]):
            if x not in seen:
                seen.add(x)
                queue.append(x)
    return order


def invade(task) -> Dict[str, Any]:
    run, era, strategy, seed, T, share = task
    import gol_store as store
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    cfg.extinction_threshold = 0
    world = GraphOfLife.from_checkpoint(store.load_archive(run, era), cfg)
    world.rng = np.random.RandomState(seed)
    rng = np.random.RandomState(1000 + seed)
    patch = patch_of(world, share, rng)
    resident = world.brains[patch[rng.randint(len(patch))]]
    for k, u in enumerate(patch):
        if strategy == "resident":
            world.brains[u] = ManualBrain(cfg, MANUAL_ID, frozen=resident.copy())
        else:
            world.brains[u] = ManualBrain(cfg, MANUAL_ID, policy=STRATEGIES[strategy])
    series = []
    for t in range(T + 1):
        if t:
            world.step(record_decisions=False)
        ids = list(world.G.nodes())
        mine = [u for u in ids if world.brains[u].brain_id >= MANUAL_ID]
        tok = sum(world.tokens[u] for u in mine)
        series.append({"t": t, "agents": len(ids), "mine": len(mine), "tokens": int(tok)})
        if not mine or not ids:
            break
    return {"run": run, "era": era, "strategy": strategy, "seed": seed, "patch": len(patch), "series": series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--era", type=int, default=2000)
    ap.add_argument("--strategies", default="factory,turtle,raider,resident")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--patch", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tasks = [(r, a.era, s, int(sd), a.T, a.patch) for r in a.runs for s in a.strategies.split(",")
             for sd in a.seeds.split(",")]
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(invade, tasks))
    for r in results:
        s = r["series"]
        pts = " ".join(f"{x['t']}:{x['mine'] / max(1, x['agents']):.3f}" for x in s[:: max(1, len(s) // 6)])
        print(f"{r['run']} {r['strategy']:9s} seed {r['seed']} patch {r['patch']}: share {pts} | end {s[-1]['mine']}/{s[-1]['agents']}")
    if a.out:
        json.dump(results, open(a.out, "w"))


if __name__ == "__main__":
    main()
