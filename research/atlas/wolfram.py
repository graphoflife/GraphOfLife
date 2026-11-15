#!/usr/bin/env python3
"""
Round 3's worlds read with the tools of the Wolfram Physics Project (Wolfram 2020): a world as a network
that rewrites itself, how local its rewrites are, how far a cause can travel in a phase and how many such
steps wide a world is, and whether its riches bend its space. One pass over each of the 82 worlds of
round 3 that lived (Experiments 40, 41 and 42: the worlds of research/screen/evolution.json), from its
frames and its statistics.

    python3 research/atlas/wolfram.py [--jobs 2] [--only RUN ...]   GraphOfLifeRuns/.pilots/atlas/wolfram/<run>.json
    python3 research/atlas/wolfram.py --gather                       research/atlas/wolfram.json

Four readings of each world:

  rewrites   at every 100th iteration from 100 to 3,000, both phases: each phase's changes to the network
             read as the rewrites of a Wolfram model. Agents born, agents dead (starved, culled, aged), links
             made and links cut. A link made between two agents that were there before is placed by how many
             steps apart they were just before it: 2 closes a triangle, the most local link a rule can make.
             A newborn's link is placed by how far its other end was from the newborn's parent. Of the
             links beyond two steps, at most FAR_PLACED a phase are placed exactly, the rest in proportion.
  space      the viewer's `dimension` and `ricciCurvature` (balls out to 5 steps, every 25th iteration),
             from the run's stats.jsonl.
  cones      from 32 agents drawn at random after the games of iterations 1,000, 2,000 and 2,900, three cones
             followed for 40 phases: news, as far as the rules let it go in a phase (4 steps, 3 in a world
             without the look that only talks); matter, a step a phase (a stake, a gift, a child); and the
             tokens and brains that did move (stakes, conquests, gifts, children, a mate's share). The global
             rules, the cull, the share-out and the one random stream, are left out: they reach the whole
             world at once.
  curvature  in the world after its last game, every agent's ball out to the radius at which the typical
             ball holds a quarter of the world. The world's dimension d and Ricci scalar R come from
             ln V(r) = a + d ln r − R r² / (6 (d + 2)) (Wolfram 2020, section 4); each agent's own R from the
             same with d held at the world's. Each R is set against the agent's tokens, the tokens around it,
             its links, and how much was rewritten at it in the last 20 iterations.

Numpy and networkx only, like the rest of the notebook's code; every random choice is seeded by the run's
own seed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import gol_store as store                        # noqa: E402
from research.atlas.space import half_reach      # noqa: E402

EVOLUTION = os.path.join(ROOT, "research", "screen", "evolution.json")
OUT = os.path.join(ROOT, "research", "atlas", "wolfram.json")
KEY = np.int64(1) << 32          # a link's key: its smaller agent id times KEY plus the larger
REWRITE_AT = range(100, 3001, 100)
CONE_AT = (1000, 2000, 2900)
CONE_PHASES = 40
CONE_SOURCES = 32
ACTIVITY_PHASES = 40
RADII = 5                         # balls out to 5 steps, as the viewer's `dimension` and `ricciCurvature`
FAR = 8                          # distances 0 … 7, and 8 for "8 or more, or in another piece"
#: Links beyond two steps placed exactly in one phase; any more are placed in proportion to these. Only s3094,
#: whose agents link far by the thousand in a phase, was read with this cap when the page was made.
FAR_PLACED = 256
SETTINGS = ("message_prepass", "link_search", "allocation", "redistribution", "mend_on_death", "link_token",
            "link_upkeep", "valence", "contest", "allow_revolutions", "allow_gifting", "mix_mates", "upkeep_to")


def readings_dir() -> str:
    return os.path.join(store.BASE_DIR, ".pilots", "atlas", "wolfram")


def worlds() -> List[str]:
    return [w["id"] for w in json.load(open(EVOLUTION))["worlds"]]


# ---------------------------------------------------------------------------
# A network in compressed rows, enough to spread sets of agents through it

class Rows:
    """A symmetric 0/1 matrix in compressed rows: row u lists u's neighbours."""

    def __init__(self, n: int, a: np.ndarray, b: np.ndarray) -> None:
        a, b = np.r_[a, b].astype(np.int64), np.r_[b, a].astype(np.int64)
        order = np.lexsort((b, a))
        self.n = n
        self.indices = b[order]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=n))]).astype(np.int64)
        self.degree = np.diff(self.indptr)
        self._starts = np.flatnonzero(self.degree > 0)

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        if len(self._starts):
            out[self._starts] = np.add.reduceat(x[self.indices], self.indptr[self._starts], axis=0)
        return out

    def neighbours(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def spread(self, x: np.ndarray, steps: int) -> np.ndarray:
        """Every agent within `steps` of the agents marked in each column of x."""
        for _ in range(steps):
            x = ((x + self @ x) > 0).astype(np.float32)
        return x

    def sub(self, keep: np.ndarray) -> "Rows":
        """The network among the agents `keep` (places), renumbered in that order."""
        new = np.full(self.n, -1)
        new[keep] = np.arange(len(keep))
        u = np.repeat(np.arange(self.n), self.degree)
        ok = (new[u] >= 0) & (new[self.indices] >= 0) & (u < self.indices)
        return Rows(len(keep), new[u[ok]], new[self.indices[ok]])


class Net:
    """The agents of one frame, their places in its list, and its network."""

    def __init__(self, frame: Dict[str, Any]) -> None:
        self.frame = frame
        self.ids = np.asarray(frame["ids"], np.int64)
        self.n = len(self.ids)
        self._sort = np.argsort(self.ids, kind="stable")
        self._sorted = self.ids[self._sort]
        e = np.asarray(frame["edges"], np.int64).reshape(-1, 2)
        e = e[e[:, 0] != e[:, 1]]
        self.keys = np.unique(np.minimum(e[:, 0], e[:, 1]) * KEY + np.maximum(e[:, 0], e[:, 1]))
        a, b = self.place(self.keys // KEY), self.place(self.keys % KEY)
        ok = (a >= 0) & (b >= 0)
        self.rows = Rows(self.n, a[ok], b[ok])
        self.degree = self.rows.degree

    def place(self, ids: np.ndarray) -> np.ndarray:
        """Each id's place in this frame's list of agents, or -1 for an agent not in it."""
        ids = np.asarray(ids, np.int64)
        if self.n == 0:
            return np.full(ids.shape, -1)
        p = np.clip(np.searchsorted(self._sorted, ids), 0, self.n - 1)
        return np.where(self._sorted[p] == ids, self._sort[p], -1)


def typical_balls(rows: Rows, sources: np.ndarray, rmax: int = 40) -> np.ndarray:
    """The typical number of agents within r steps, r = 0 … rmax: the geometric mean over the sources, so
    that the few beside a hub do not outweigh the rest (as book_chapters.space.ball_growth)."""
    x = np.zeros((rows.n, len(sources)), np.float32)
    x[sources, np.arange(len(sources))] = 1.0
    sizes = [x.sum(axis=0)]
    for _ in range(rmax):
        x = rows.spread(x, 1)
        sizes.append(x.sum(axis=0))
    return np.exp(np.log(np.stack(sizes, axis=1)).mean(axis=0))


def distances(net: Net, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Steps between agents a[i] and b[i] (places in net); FAR for FAR or more, or out of reach."""
    out = np.full(len(a), FAR, np.int64)
    if len(a) == 0:
        return out
    out[a == b] = 0
    key = np.minimum(net.ids[a], net.ids[b]) * KEY + np.maximum(net.ids[a], net.ids[b])
    linked = (a != b) & np.isin(key, net.keys)
    out[linked] = 1
    rest = np.flatnonzero((a != b) & ~linked)
    far = []
    for i in rest:
        if np.intersect1d(net.rows.neighbours(a[i]), net.rows.neighbours(b[i]), assume_unique=True).size:
            out[i] = 2
        else:
            far.append(i)
    far = np.asarray(far, np.int64)
    rest = far[:0]
    if len(far) > FAR_PLACED:
        rng = np.random.default_rng(len(far))
        placed = np.sort(rng.choice(far, FAR_PLACED, replace=False))
        rest, far = np.setdiff1d(far, placed), placed
    for start in range(0, len(far), 128):
        part = far[start:start + 128]
        x = np.zeros((net.n, len(part)), np.float32)
        x[a[part], np.arange(len(part))] = 1.0
        x = net.rows.spread(x, 2)
        for steps in range(3, FAR):
            x = net.rows.spread(x, 1)
            hit = (x[b[part], np.arange(len(part))] > 0) & (out[part] == FAR)
            out[part[hit]] = steps
    if len(rest):
        out[rest] = rng.choice(out[far], len(rest), replace=True)
    return out


def hist(values: np.ndarray) -> List[int]:
    return np.bincount(values, minlength=FAR + 1)[:FAR + 1].tolist()


# ---------------------------------------------------------------------------
# Rewrites

def rewrites_of_phase(before: Net, after: Net) -> Dict[str, Any]:
    """One phase's changes to the network, as the rewrites of a Wolfram model."""
    fr = after.frame
    was = before.place(after.ids)
    newborn = was < 0
    made = np.setdiff1d(after.keys, before.keys, assume_unique=True)
    cut = np.setdiff1d(before.keys, after.keys, assume_unique=True)
    mu, mv = made // KEY, made % KEY
    bu, bv = before.place(mu), before.place(mv)
    old = (bu >= 0) & (bv >= 0)
    # A newborn's link: how far its other end was from the newborn's parent, just before the phase.
    parent_of = dict(zip(after.ids[newborn].tolist(), np.asarray(fr["parent_ids"], np.int64)[newborn].tolist()))
    one = (bu >= 0) != (bv >= 0)
    child = np.where(bu >= 0, mv, mu)[one]
    other = np.where(bu >= 0, bu, bv)[one]
    parent = before.place(np.array([parent_of.get(int(c), -1) for c in child], np.int64))
    known = parent >= 0
    cu, cv = cut // KEY, cut % KEY
    lapsed = (after.place(cu) >= 0) & (after.place(cv) >= 0)
    clean = fr.get("cleanup") or {}
    return {
        "agents": int(before.n),
        "born": int(newborn.sum()),
        "died": int(before.n - (was >= 0).sum()),
        "starved": int(clean.get("starved", 0)),
        "culled": int(clean.get("orphaned", 0)),
        "aged": int(clean.get("aged", 0)),
        "made_old": hist(distances(before, bu[old], bv[old])),
        "made_birth": hist(distances(before, parent[known], other[known])),
        "made_birth_unplaced": int((~known).sum()),
        "made_newborns": int(((bu < 0) & (bv < 0)).sum()),
        "cut_lapsed": int(lapsed.sum()),
        "cut_dead": int((~lapsed).sum()),
        "links": int(len(before.keys)),
    }


def add(total: Dict[str, Any], part: Dict[str, Any]) -> None:
    for k, v in part.items():
        if isinstance(v, list):
            total[k] = [a + b for a, b in zip(total.get(k, [0] * len(v)), v)]
        else:
            total[k] = total.get(k, 0) + v


def rewrites(run: str) -> Dict[str, Any]:
    total: Dict[str, Any] = {"reproduction": {}, "game": {}, "iterations": []}
    last = store.count_frames(run) // 2 - 1
    for t in REWRITE_AT:
        if t > last:
            break
        before = Net(store.read_frame(run, 2 * t - 1))
        middle = Net(store.read_frame(run, 2 * t))
        after = Net(store.read_frame(run, 2 * t + 1))
        add(total["reproduction"], rewrites_of_phase(before, middle))
        add(total["game"], rewrites_of_phase(middle, after))
        total["iterations"].append(t)
    return total


# ---------------------------------------------------------------------------
# Cones

def flows(fr: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Who passed tokens or a brain to whom in this phase (agent ids): stakes, conquests, gifts, children,
    a mate's share."""
    src: List[int] = []
    dst: List[int] = []
    dec = fr.get("decisions") or {}
    for a in dec.get("allocations", []):
        for t, k in zip(a["targets"], a["alloc"]):
            if k > 0:
                src.append(a["agent"])
                dst.append(t)
    for w in dec.get("winners", []):
        src.append(w["winner"])
        dst.append(w["node"])
    for b in dec.get("births", []):
        if b.get("child") is not None:
            src.append(b["agent"])
            dst.append(b["child"])
    for g in dec.get("gifts", []):
        src.append(g[0])
        dst.append(g[1])
    for m in dec.get("mate_paid", []):
        src.append(m[0])
        dst.append(m[1])
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def cones(run: str, cfg: Dict[str, Any], t_end: int, seed: int) -> Optional[Dict[str, Any]]:
    """Past cones: from agents alive after the game of iteration t_end, back phase by phase, every agent of
    an earlier phase that could have had an effect on them. Followed backwards, a cone is never cut short
    by the cull, which removes the agents a cone going forwards has reached."""
    end = 2 * t_end + 1
    if end >= store.count_frames(run) or end - CONE_PHASES < 0:
        return None
    net = Net(store.read_frame(run, end))
    rng = np.random.default_rng(seed)
    sources = rng.choice(net.n, min(CONE_SOURCES, net.n), replace=False)
    s = len(sources) + 1
    news_speed = 4 if cfg.get("message_prepass", True) else 3
    # One column per agent drawn, and a last one for the whole world: the past of everyone alive now.
    x = {}
    for kind in ("news", "matter", "flow"):
        x[kind] = np.zeros((net.n, s), np.float32)
        x[kind][sources, np.arange(s - 1)] = 1.0
        x[kind][:, -1] = 1.0
    balls = typical_balls(net.rows, rng.choice(net.n, min(40, net.n), replace=False))
    out = {"t_end": t_end, "news_speed": news_speed, "agents": int(net.n),
           "half_reach": half_reach(balls.tolist(), net.n), "balls": balls.tolist(),
           "share": {k: [] for k in x}, "whole": {k: [] for k in x}, "world": []}
    for k in range(1, CONE_PHASES + 1):
        # The phase that led from `prev` to `net`: news went out within the network it began with.
        prev = Net(store.read_frame(run, end - k))
        back = prev.place(net.ids)
        old = back >= 0
        born = np.flatnonzero(~old)
        parent = prev.place(np.asarray(net.frame["parent_ids"], np.int64)[born])
        has = parent >= 0
        new = {}
        for kind, speed in (("news", news_speed), ("matter", 1)):
            z = np.zeros((prev.n, s), np.float32)
            z[back[old]] = x[kind][old]
            w = np.zeros((prev.n, s), np.float32)
            np.maximum.at(w, parent[has], x[kind][born[has]])
            new[kind] = np.maximum(prev.rows.spread(z, speed), prev.rows.spread(w, speed - 1))
        # Tokens and brains: whoever passed some to a member of the cone in that phase joins it.
        a, b = flows(net.frame)
        pa, pb = prev.place(a), net.place(b)
        ok = (pa >= 0) & (pb >= 0)
        y = np.zeros((prev.n, s), np.float32)
        y[back[old]] = x["flow"][old]
        np.add.at(y, pa[ok], x["flow"][pb[ok]])
        new["flow"] = (y > 0).astype(np.float32)
        for kind in x:
            share = new[kind].sum(axis=0) / max(prev.n, 1)
            out["share"][kind].append(share[:-1].round(5).tolist())
            out["whole"][kind].append(round(float(share[-1]), 5))
        out["world"].append(int(prev.n))
        x, net = new, prev
    return out


# ---------------------------------------------------------------------------
# Curvature

def all_balls(rows: Rows, limit: float, rmax: int = 30, batch: int = 256) -> np.ndarray:
    """Every agent's ball, V[x, r] = agents within r steps of x, for r = 0 … one past the radius at which
    the typical ball (the geometric mean over the agents) first holds `limit` agents, at most rmax."""
    cols: List[np.ndarray] = []
    for lo in range(0, rows.n, batch):
        hi = min(rows.n, lo + batch)
        x = np.zeros((rows.n, hi - lo), np.float32)
        x[np.arange(lo, hi), np.arange(hi - lo)] = 1.0
        sizes = [x.sum(axis=0)]
        for _ in range(rmax):
            x = rows.spread(x, 1)
            sizes.append(x.sum(axis=0))
            if sizes[-1].min() >= limit:
                break
        sizes += [sizes[-1]] * (rmax + 1 - len(sizes))
        cols.append(np.stack(sizes, axis=1))
    v = np.concatenate(cols, axis=0)
    typical = np.exp(np.log(v).mean(axis=0))
    stop = int(np.argmax(typical >= limit)) if (typical >= limit).any() else rmax - 1
    return v[:, :stop + 2]


def ranks(a: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties given their average rank."""
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[np.argsort(a, kind="mergesort")] = np.arange(1, len(a) + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=r)[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def partial_spearman(y: np.ndarray, x: np.ndarray, *held: np.ndarray) -> float:
    """Spearman's rho of y and x, with the ranks of what is held equal regressed out of both."""
    m = np.vstack([np.ones(len(y))] + [ranks(z) for z in held]).T

    def resid(r: np.ndarray) -> np.ndarray:
        return r - m @ np.linalg.lstsq(m, r, rcond=None)[0]
    return float(np.corrcoef(resid(ranks(y)), resid(ranks(x)))[0, 1])


def activity(run: str, last: int, net: Net) -> np.ndarray:
    """Rewrites at each agent of `net` over the last ACTIVITY_PHASES phases, per phase: links made or cut at
    it, children it had, and its node taken by another brain."""
    count = np.zeros(net.n)
    prev = Net(store.read_frame(run, last - ACTIVITY_PHASES))
    for i in range(last - ACTIVITY_PHASES + 1, last + 1):
        cur = Net(store.read_frame(run, i)) if i < last else net
        changed = np.setxor1d(prev.keys, cur.keys, assume_unique=True)
        for ends in (changed // KEY, changed % KEY):
            p = net.place(ends)
            np.add.at(count, p[p >= 0], 1)
        births = (cur.frame.get("decisions") or {}).get("births", [])
        p = net.place(np.asarray([b["agent"] for b in births if b.get("child") is not None], np.int64))
        np.add.at(count, p[p >= 0], 1)
        both = prev.place(cur.ids)
        kept = both >= 0
        brains_now = np.asarray(cur.frame["brain_ids"], np.int64)[kept]
        brains_then = np.asarray(prev.frame["brain_ids"], np.int64)[both[kept]]
        p = net.place(cur.ids[kept][brains_now != brains_then])
        np.add.at(count, p[p >= 0], 1)
        prev = cur
    return count / ACTIVITY_PHASES


def largest_piece(rows: Rows) -> np.ndarray:
    g = nx.Graph()
    g.add_nodes_from(range(rows.n))
    u = np.repeat(np.arange(rows.n), rows.degree)
    g.add_edges_from(zip(u.tolist(), rows.indices.tolist()))
    return np.asarray(sorted(max(nx.connected_components(g), key=len)), np.int64)


def curvature(run: str) -> Dict[str, Any]:
    last = store.count_frames(run) - 1
    net = Net(store.read_frame(run, last))
    act = activity(run, last, net)
    big = largest_piece(net.rows)
    rows = net.rows.sub(big)
    n = len(big)
    v = all_balls(rows, n / 2, rmax=RADII)
    typical = np.exp(np.log(v).mean(axis=0))
    # The viewer's radii, 1 to 5, less any at which the typical ball already holds half the world.
    radii = [r for r in range(1, min(RADII, v.shape[1] - 1) + 1) if typical[r] < n / 2]
    out: Dict[str, Any] = {"agents": int(n), "all_agents": int(net.n), "typical_ball": typical.tolist(),
                           "radii": radii}
    if len(radii) < 3:
        return out
    r = np.asarray(radii, float)
    lv = np.log(typical[radii])
    if len(radii) >= 4:
        a, d, k = np.linalg.lstsq(np.vstack([np.ones_like(r), np.log(r), -r ** 2]).T, lv, rcond=None)[0]
    else:
        a, d = np.linalg.lstsq(np.vstack([np.ones_like(r), np.log(r)]).T, lv, rcond=None)[0]
        k = np.nan
    out.update({"dimension": float(d), "ricci": float(6 * (d + 2) * k) if np.isfinite(k) else None})
    # Each agent's own curvature, its dimension held at the world's.
    y = np.log(v[:, radii]) - d * np.log(r)[None, :]
    coef = np.linalg.lstsq(np.vstack([np.ones_like(r), -r ** 2]).T, y.T, rcond=None)[0]
    ricci = 6 * (d + 2) * coef[1]
    tokens = np.asarray(net.frame["tokens"], float)[big]
    degree = rows.degree.astype(float)
    around = (tokens + rows @ tokens.astype(np.float32)) / (1 + degree)
    # The links of an agent's neighbours, on average: a second shell is large where they are many.
    beside = (rows @ degree.astype(np.float32)) / np.maximum(degree, 1)
    act = act[big]
    out["agent"] = {
        "ricci": {"median": float(np.median(ricci)), "q1": float(np.quantile(ricci, 0.25)),
                  "q3": float(np.quantile(ricci, 0.75)), "positive": float((ricci > 0).mean())},
        "rho_tokens": spearman(ricci, tokens), "rho_around": spearman(ricci, around),
        "rho_degree": spearman(ricci, degree), "rho_activity": spearman(ricci, act),
        "partial_tokens_degree": partial_spearman(ricci, tokens, degree),
        "partial_around_degree": partial_spearman(ricci, around, degree),
        "partial_activity_degree": partial_spearman(ricci, act, degree),
        "partial_tokens_links": partial_spearman(ricci, tokens, degree, beside),
        "partial_around_links": partial_spearman(ricci, around, degree, beside),
        "partial_activity_links": partial_spearman(ricci, act, degree, beside),
        "rho_beside": spearman(ricci, beside), "rho_around_beside": spearman(around, beside),
        "rho_tokens_degree": spearman(tokens, degree), "rho_activity_degree": spearman(act, degree),
        "rho_activity_tokens": spearman(act, tokens),
        "activity": {"median": float(np.median(act)), "mean": float(act.mean())},
    }
    # The balls around the richest and the poorest tenth, against the typical ball.
    order = np.argsort(tokens, kind="stable")
    tenth = max(1, n // 10)
    rel = np.log(v) - np.log(typical)[None, :]
    out["deciles"] = {"poorest": rel[order[:tenth]].mean(axis=0).tolist(),
                      "richest": rel[order[-tenth:]].mean(axis=0).tolist(),
                      "degree_poorest": float(degree[order[:tenth]].mean()),
                      "degree_richest": float(degree[order[-tenth:]].mean()),
                      "degree_all": float(degree.mean())}
    return out


# ---------------------------------------------------------------------------
# Space, from the run's own statistics

def space(run: str) -> Dict[str, Any]:
    rows = []
    with open(os.path.join(store.run_dir(run), "stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("dimension") is not None and r.get("phase") == 2:
                rows.append([r["iteration"], r["dimension"], r.get("ricciCurvature"), r.get("nodes")])
    return {"rows": rows}


# ---------------------------------------------------------------------------

def one(run: str) -> str:
    path = os.path.join(readings_dir(), f"{run}.json")
    if os.path.exists(path):
        return f"{run}: kept"
    meta = store.load_meta(run)
    cfg = meta["config"]
    seed = int((meta.get("lab") or {}).get("seed") or 23)
    out = {"id": run, "seed": seed, "settings": {k: cfg.get(k) for k in SETTINGS}, "space": space(run),
           "rewrites": rewrites(run),
           "cones": [c for c in (cones(run, cfg, t0, seed + t0) for t0 in CONE_AT) if c],
           "curvature": curvature(run)}
    os.makedirs(readings_dir(), exist_ok=True)
    store.write_json(path, out)
    return f"{run}: read"


def compact(w: Dict[str, Any]) -> Dict[str, Any]:
    """What the page reads of a world's readings: each cone as the median over its agents, rounded."""
    r = lambda v, k=4: [round(float(x), k) if x is not None else None for x in v]
    cones = [{"t_end": c["t_end"], "news_speed": c["news_speed"], "agents": c["agents"],
              "half_reach": c["half_reach"], "world": c["world"],
              # The share of the world within two steps of a typical agent: a stranger drawn at random is
              # further away than that with the rest.
              "within2": round(float(c["balls"][2]) / c["agents"], 6),
              "agent": {k: r(np.median(np.asarray(v), axis=1)) for k, v in c["share"].items()},
              "whole": {k: r(v) for k, v in c["whole"].items()}} for c in w["cones"]]
    space = [[row[0], round(row[1], 4), None if row[2] is None else round(row[2], 4), row[3]]
             for row in w["space"]["rows"]]
    cur = dict(w["curvature"])
    cur["typical_ball"] = r(cur.get("typical_ball", []), 2)
    if "deciles" in cur:
        cur["deciles"] = {k: (r(v) if isinstance(v, list) else v) for k, v in cur["deciles"].items()}
    return {"id": w["id"], "seed": w["seed"], "settings": w["settings"], "rewrites": w["rewrites"],
            "space": {"rows": space}, "cones": cones, "curvature": cur}


def again(run: str) -> str:
    """A world's curvature read again, into its readings kept."""
    path = os.path.join(readings_dir(), f"{run}.json")
    if not os.path.exists(path):
        return f"{run}: not read yet"
    out = json.load(open(path))
    out["curvature"] = curvature(run)
    store.write_json(path, out)
    return f"{run}: curvature read again"


def gather() -> None:
    rows = []
    for run in worlds():
        path = os.path.join(readings_dir(), f"{run}.json")
        if os.path.exists(path):
            rows.append(compact(json.load(open(path))))
    with open(OUT, "w") as f:
        json.dump({"worlds": rows, "of": len(worlds())}, f, separators=(",", ":"))
    print(f"{len(rows)} of {len(worlds())} worlds gathered into {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gather", action="store_true")
    ap.add_argument("--curvature", action="store_true", help="read the curvature again, into readings kept")
    a = ap.parse_args()
    if a.gather:
        gather()
        return
    if a.curvature:
        todo = a.only or worlds()
        run = map if a.jobs <= 1 else ProcessPoolExecutor(a.jobs).map
        for line in run(again, todo):
            print(line, flush=True)
        return
    todo = a.only or worlds()
    if a.jobs <= 1:
        for run in todo:
            print(one(run), flush=True)
    else:
        with ProcessPoolExecutor(a.jobs) as pool:
            for line in pool.map(one, todo):
                print(line, flush=True)


if __name__ == "__main__":
    main()
