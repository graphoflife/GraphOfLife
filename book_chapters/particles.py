# -*- coding: utf-8 -*-
"""
Particles: patterns that outlast what carries them.

Does anything in these worlds keep its identity while the agents, tokens and
brains that make it up are replaced? Three detectors, each beside a null that
says what chance would make, and each tried on a pattern known to be there:

- **Rings** (bound tokens): the net flow of a stretch of games, split into
  the part that runs downhill and the part that goes round loops (as in
  research/oee/checks/circulation.py); a ring that lasts gives two stretches
  in a row the same circulation. Positive control: rings planted in the
  recorded flow, on real loops of the network, carrying a set share of the
  circulation.
- **Beats**: the spectrum of each long-lived agent's tokens, and of the
  tokens around it, against the spectrum of a random walk with the same two
  memories, one slow and one that swings from game to game (an AR(2) process
  with real roots, fitted to each stretch), which has no rhythm of its own.
  Positive controls: beats planted in the real series, one a stretch and one
  shared by all; and the frozen world's two-game cycle, for the swing, whose
  even-split bound for a hub and its leaves is known.
- **Groups** that outlast their members, and whether they move: the flow
  modules of the last fifty games (web/js/flowmodules.js through
  tools/flowmodules_cli.js), followed from game to game, against the same
  games with every agent's partners dealt at random (the network's regions)
  and with every module's members dealt at random (chance overlaps); and how
  far a long-lived module's last members stood from its first.

The worlds: five ordinary worlds and the largest star of B1 at 100,000
tokens (Experiment 10), the five healthiest worlds of round 3 by the product
(Experiments 40 and 42), and, as a control in which everything lasts, a world
whose brains never change and which froze (Experiment 7). Each is read over
the last WINDOW games of its life, in one pass over its frames; what the pass
finds is kept beside the run (.book/<run>.particles.json).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import zlib
from array import array
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import book_data as D
import book_figures as F
import gol_store as store
from book_figures import DEGREE_CLASSES, chapter, recipe
from book_chapters.common import GREEN, GREY, ORANGE, RED, VIOLET, WORLD_SETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "tools", "flowmodules_cli.js")
HEALTH = os.path.join(ROOT, "research", "screen", "health.json")

#: The games read: the last WINDOW of a run's life.
WINDOW = 500
#: Stretches of games, in games, whose net flow is added up before it is split; at most RING_PAIRS
#: stretches in a row are read for each, from the start of the window. One game shows the two-game
#: swing of the flow itself; the others are even, so that a swing back and forth cancels in each.
RING_WINDOWS = (1, 2, 10, 50)
RING_PAIRS = 100
#: Rings planted as the positive control, on triangles and squares of the connections two stretches
#: in a row share: the share of the circulation's energy they carry, and the share of those
#: connections they run along.
PLANTED = (0.05, 0.2)
PLANTED_EDGES = 0.02
#: Lengths, in games, of the stretches of one agent's life whose spectrum is taken.
SEGMENTS = (32, 128)
#: Planted beats, by the share of a stretch's variance they carry (amplitude a: a² of the variance).
BEAT_AMPLITUDES = (0.25, 0.5, 0.75, 1.0)
#: The level of the test for a beat: a stretch is flagged when its highest peak over the random walk
#: would be beaten by chance less than once in a hundred.
BEAT_LEVEL = 0.99
#: How many random-walk series make the yardstick for each length.
BEAT_DRAWS = 200000
#: How many times its scatter a fitted swing memory must lie below the even split's −1/2 to count as
#: a two-game swing that outlasts a hub's (2.5 times: chance, about once in 160).
SWING_SPREAD = 2.5
#: The second positive control: one beat planted in every stretch, of this period in games and with
#: the amplitude that gives it a sixteenth of each stretch's variance.
SHARED_PERIOD = 5
SHARED_AMPLITUDE = 0.25
#: Stretches are read so many at a time, so that their spectra fit in memory.
CHUNK = 10000
#: Games of flow modules followed, at the end of the window.
GROUP_GAMES = 50
#: A module followed at least this many games is asked whether it outlasted its members and moved.
LONG_LIFE = 10
#: Bumped whenever what the pass makes changes, so a kept copy made by older code is made again.
VERSION = 2


# ---------------------------------------------------------------------------
# The worlds
# ---------------------------------------------------------------------------

#: B1 at 100,000 tokens: the first five seeds whose world is not held together by one hub, and the
#: largest star (The largest and the smallest).
B1_SEEDS = (1, 3, 4, 5, 6)
STAR_SEED = 28
#: How many of round 3's healthiest worlds are read.
ROUND3_TOP = 5
#: The five the page read: round 3's healthiest by the product on 9 November 2026, before health counted
#: agents on loops per link. Kept by name, so the page reads the same worlds whatever the order says since
#: (the fifth by the product is now seed 4002 played weakest first, and seed 3057 the sixth).
ROUND3_READ = ("B1-30000-b82ad0-s3042", "B1-30000-8526cf-s4006", "B1-30000-af1e28-s3018",
               "B1-30000-625bc7-s3080", "B1-30000-38aee9-s3057")
#: The control: brains that never change (Experiment 7), a world that froze long before its end.
FROZEN = "B1-10000-a8525d-s002"


def round3_worlds() -> List[Dict[str, Any]]:
    """The healthiest worlds of round 3 by the product when the page was made, in that order: {id, label, rank}."""
    with open(HEALTH, encoding="utf-8") as f:
        worlds = {w["id"]: w for w in json.load(f)["worlds"]}
    tag = {"E41": " w", "E42": " s"}
    return [{"id": rid, "label": f"s{worlds[rid]['seed']}{tag.get(worlds[rid]['round'], '')}", "rank": i,
             "round": worlds[rid]["round"]} for i, rid in enumerate(ROUND3_READ[:ROUND3_TOP], 1)]


def worlds() -> List[Dict[str, Any]]:
    """Every world the page reads, with the group it is drawn in and a short label."""
    e10 = WORLD_SETS["E10"]
    out = [{"id": e10.run(s), "label": f"s{s}", "group": "B1"} for s in B1_SEEDS]
    out.append({"id": e10.run(STAR_SEED), "label": f"s{STAR_SEED}", "group": "B1 star"})
    out += [{"id": w["id"], "label": w["label"], "group": "round 3"} for w in round3_worlds()]
    out.append({"id": FROZEN, "label": "frozen", "group": "frozen"})
    return out


GROUP_COLOUR = {"B1": ORANGE, "B1 star": RED, "round 3": GREEN, "frozen": VIOLET}


# ---------------------------------------------------------------------------
# Rings: the circulation of a stretch's net flow
# ---------------------------------------------------------------------------

#: Two agent ids make one key: the lower times this, plus the higher.
KEY = 1 << 32


def _pressure(iu: np.ndarray, iv: np.ndarray, flow: np.ndarray, n: int, tol: float = 1e-6,
              iters: int = 5000) -> np.ndarray:
    """
    The potential phi whose differences phi(u) − phi(v) explain as much of the
    flow as possible: least squares, by conjugate gradients on the network's
    Laplacian, preconditioned by the degrees (research/oee/checks/circulation.py).
    """
    deg = (np.bincount(iu, minlength=n) + np.bincount(iv, minlength=n)).astype(float)

    def lap(x: np.ndarray) -> np.ndarray:
        return deg * x - np.bincount(iu, weights=x[iv], minlength=n) - np.bincount(iv, weights=x[iu], minlength=n)

    d = np.bincount(iu, weights=flow, minlength=n) - np.bincount(iv, weights=flow, minlength=n)
    inv = 1.0 / np.maximum(deg, 1.0)
    x = np.zeros(n)
    r = d.copy()
    z = inv * r
    p = z.copy()
    rz = r @ z
    dd = max(d @ d, 1e-300)
    for _ in range(iters):
        if r @ r <= tol * tol * dd:
            break
        ap = lap(p)
        alpha = rz / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        z = inv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def circulation(keys: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    The part of a net flow (one value a connection, keyed lower·KEY + higher)
    that goes round loops. A connection on no loop — on a dead-end branch —
    carries none of it, and the rest does not depend on such connections, so
    they are pruned first, dead end by dead end, down to the network's core:
    the same answer, and far fewer, far better behaved equations to solve.
    """
    u, v = keys // KEY, keys % KEY
    _, inverse = np.unique(np.concatenate([u, v]), return_inverse=True)
    iu, iv = inverse[:len(u)], inverse[len(u):]
    n = int(inverse.max()) + 1
    keep = np.ones(len(keys), dtype=bool)
    while True:
        deg = np.bincount(iu[keep], minlength=n) + np.bincount(iv[keep], minlength=n)
        dead = keep & ((deg[iu] == 1) | (deg[iv] == 1))
        if not dead.any():
            break
        keep &= ~dead
    out = np.zeros(len(keys))
    if keep.any():
        cu, cv, cf = iu[keep], iv[keep], flow[keep]
        _, back = np.unique(np.concatenate([cu, cv]), return_inverse=True)
        bu, bv = back[:len(cu)], back[len(cu):]
        phi = _pressure(bu, bv, cf, int(back.max()) + 1)
        out[keep] = cf - (phi[bu] - phi[bv])
    return out


def _correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _loops(keys: np.ndarray, rng: np.random.Generator) -> List[List[int]]:
    """
    Short loops for planted rings, on the connections `keys`: for connections
    taken in a random order, a triangle through it, or else a square, each
    connection used by one loop at most, until the loops run along
    PLANTED_EDGES of the connections.
    """
    adjacency: Dict[int, set] = {}
    for code in keys.tolist():
        a, b = divmod(code, KEY)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    used: set = set()
    loops: List[List[int]] = []
    wanted = max(3, int(PLANTED_EDGES * len(keys)))
    covered = 0

    def free(*nodes: int) -> bool:
        return not any((min(a, b), max(a, b)) in used for a, b in zip(nodes, nodes[1:] + nodes[:1]))

    def square(u: int, v: int) -> Optional[List[int]]:
        """
        The square u–v–y–x with the smallest x, then the smallest y, whose
        connections are all free. Its candidates are counted out from whichever
        end has fewer neighbours: from a hub's side, a hub with thousands of
        leaves would be searched in full for every leaf.
        """
        if len(adjacency[u]) <= len(adjacency[v]):
            pairs = [(x, y) for x in adjacency[u] if x != v for y in adjacency[x] & adjacency[v] if y not in (u, v)]
        else:
            pairs = [(x, y) for y in adjacency[v] if y not in (u, v) for x in adjacency[y] & adjacency[u] if x != v]
        pairs.sort()
        return next(([u, v, y, x] for x, y in pairs if free(u, v, y, x)), None)

    for code in keys[rng.permutation(len(keys))].tolist():
        if covered >= wanted:
            break
        u, v = divmod(code, KEY)
        if (u, v) in used:
            continue
        loop = next(([u, v, w] for w in sorted(adjacency[u] & adjacency[v]) if free(u, v, w)), None)
        if loop is None:
            loop = square(u, v)
        if loop is None:
            continue
        used |= {(min(a, b), max(a, b)) for a, b in zip(loop, loop[1:] + loop[:1])}
        loops.append(loop)
        covered += len(loop)
    return loops


def _planted(loops: List[List[int]], keys: np.ndarray) -> np.ndarray:
    """One token round every loop, as a flow on the connections `keys` (positive from the lower id to the higher)."""
    out = np.zeros(len(keys))
    for loop in loops:
        steps = list(zip(loop, loop[1:] + loop[:1]))
        codes = np.array([min(a, b) * KEY + max(a, b) for a, b in steps], dtype=np.int64)
        out[np.searchsorted(keys, codes)] += np.array([1.0 if a < b else -1.0 for a, b in steps])
    return out


class Rings:
    """The circulation of every stretch of `length` games, and how much of it the next stretch repeats."""

    def __init__(self, length: int, rng: np.random.Generator) -> None:
        self.length = length
        self.rng = rng
        self.codes: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.games = 0
        self.previous: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.rows: List[Dict[str, Any]] = []
        self.stretches = 0

    @property
    def done(self) -> bool:
        return self.stretches > RING_PAIRS

    def add(self, src: np.ndarray, dst: np.ndarray, amount: np.ndarray) -> None:
        if self.done:
            return
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        self.codes.append(lo.astype(np.int64) * KEY + hi.astype(np.int64))
        self.values.append(np.where(src < dst, amount, -amount).astype(float))
        self.games += 1
        if self.games == self.length:
            self._close()

    def _close(self) -> None:
        codes = np.concatenate(self.codes) if self.codes else np.zeros(0, dtype=np.int64)
        values = np.concatenate(self.values) if self.values else np.zeros(0)
        self.codes, self.values, self.games = [], [], 0
        self.stretches += 1
        keys, inverse = np.unique(codes, return_inverse=True)
        flow = np.bincount(inverse, weights=values, minlength=len(keys)) if len(keys) else np.zeros(0)
        if len(keys) < 10:
            self.previous = None
            return
        circ = circulation(keys, flow)
        slope = flow - circ
        energy = float((flow ** 2).sum())
        row: Dict[str, Any] = {"share": float((circ ** 2).sum() / energy) if energy else None,
                               "edges": int(len(keys))}
        if self.previous is not None:
            keys0, circ0, slope0 = self.previous
            common, i0, i1 = np.intersect1d(keys0, keys, assume_unique=True, return_indices=True)
            row["common"] = len(common) / len(keys0)
            row["persistence"] = _correlation(circ0[i0], circ[i1])
            row["slope"] = _correlation(slope0[i0], slope[i1])
            # The positive control: rings on loops of the connections both stretches used, the same
            # rings in both, as a lasting ring would be, each as strong as makes the rings carry the
            # share PLANTED of the first stretch's circulation on those connections. Two stretches
            # whose own rings have nothing in common then correlate by about share / (1 + share).
            loops = _loops(common, self.rng)
            ring = _planted(loops, common)
            along = float((ring ** 2).sum())
            held = float((circ0[i0] ** 2).sum())
            row["planted_edges"] = along / len(common) if len(common) else 0.0
            for share in PLANTED:
                if along and held:
                    s = math.sqrt(share * held / along)
                    row[f"planted_{share}"] = _correlation(circ0[i0] + s * ring, circ[i1] + s * ring)
        self.previous = (keys, circ, slope)
        self.rows.append(row)

    def summary(self) -> Dict[str, Any]:
        def med(key: str) -> Optional[float]:
            v = [r[key] for r in self.rows if r.get(key) is not None]
            return float(np.median(v)) if v else None

        def mean(key: str) -> Optional[float]:
            v = [r[key] for r in self.rows if r.get(key) is not None]
            return float(np.mean(v)) if v else None
        pairs = [r for r in self.rows if r.get("persistence") is not None]
        return {"games": self.length, "stretches": len(self.rows), "pairs": len(pairs),
                "share": med("share"), "persistence": med("persistence"), "persistence_mean": mean("persistence"),
                "persistence_q": ([float(q) for q in np.quantile([r["persistence"] for r in pairs], [0.1, 0.9])]
                                  if pairs else None),
                "slope": med("slope"), "common": med("common"), "edges": med("edges"),
                "planted": {str(s): med(f"planted_{s}") for s in PLANTED},
                "planted_edges": med("planted_edges")}


# ---------------------------------------------------------------------------
# Beats: spectra against a random walk with the same memories
# ---------------------------------------------------------------------------

_YARDSTICK: Dict[int, Tuple[np.ndarray, float]] = {}
#: Bins of the two fitted memories (the roots, from −1 to 1) in which the yardstick is read.
ROOT_BINS = np.linspace(-1.0, 1.0, 21)
#: The periodogram is read on a grid PAD times finer than the stretch's own, so that a beat whose
#: period falls between two of the stretch's frequencies is not missed.
PAD = 4


def frequencies(s: int) -> np.ndarray:
    """The frequencies a stretch of S games is read at, a game⁻¹: from 1/S (a period of S games) to 1/2."""
    return np.arange(PAD, PAD * s // 2 + 1) / (PAD * s)


def inside(s: int) -> np.ndarray:
    """The frequencies a beat is looked for at: periods from 3 games to half the stretch."""
    f = frequencies(s)
    return (f >= 2 / s) & (f <= 1 / 3)


def _fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For stretches x (one a row) the random walk with two memories that comes
    closest to each, and what is left over. An agent's tokens drift slowly,
    and swing from game to game where the even split overshoots; one memory
    for each, a ≥ b, makes an AR(2) process with real roots,

        x_t − m = (a + b)(x_{t−1} − m) − a b (x_{t−2} − m) + noise,

    whose spectrum, σ² / (|1 − a e^{−2πif}|² |1 − b e^{−2πif}|²), is high at the
    longest periods or at two games, or both, and never higher anywhere between.
    The memories come from the stretch's correlations with itself one and two
    games on (Yule and Walker); where those would make the two memories a
    damped rhythm (complex roots), the nearest pair of equal real memories is
    taken instead, so that no rhythm is put into the yardstick. Returned: a, b,
    the periodogram at `frequencies(S)`, and the spectrum of the random walk
    there, scaled to the stretch's own variance.
    """
    s = x.shape[1]
    x0 = x - x.mean(axis=1, keepdims=True)
    var = np.maximum((x0 ** 2).sum(axis=1), 1e-12)
    r1 = (x0[:, 1:] * x0[:, :-1]).sum(axis=1) / var
    r2 = (x0[:, 2:] * x0[:, :-2]).sum(axis=1) / var
    p1 = r1 * (1 - r2) / np.maximum(1 - r1 ** 2, 1e-9)
    p2 = (r2 - r1 ** 2) / np.maximum(1 - r1 ** 2, 1e-9)
    p2 = np.where(p1 ** 2 + 4 * p2 < 0, -p1 ** 2 / 4, p2)
    root = np.sqrt(np.maximum(p1 ** 2 + 4 * p2, 0))
    a = np.clip((p1 + root) / 2, -0.98, 0.98)
    b = np.clip((p1 - root) / 2, -0.98, 0.98)

    def shape(f: np.ndarray) -> np.ndarray:
        z = np.exp(-2j * np.pi * f)[None, :]
        return 1 / (np.abs(1 - a[:, None] * z) ** 2 * np.abs(1 - b[:, None] * z) ** 2)
    periodogram = np.abs(np.fft.rfft(x0, n=PAD * s, axis=1)[:, PAD:]) ** 2 / s
    # Scaled so that, over the stretch's own frequencies, the random walk holds as much variance as the
    # stretch does.
    own = np.arange(1, s // 2 + 1) / s
    held = (np.abs(np.fft.rfft(x0, axis=1)[:, 1:]) ** 2 / s).mean(axis=1) / shape(own).mean(axis=1)
    return a, b, periodogram, shape(frequencies(s)) * held[:, None]


def _bin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = len(ROOT_BINS) - 1
    ia = np.clip(np.digitize(a, ROOT_BINS) - 1, 0, k - 1)
    ib = np.clip(np.digitize(b, ROOT_BINS) - 1, 0, k - 1)
    return ia * k + ib


def yardstick(s: int) -> Tuple[np.ndarray, float]:
    """
    How high the highest peak over the random walk climbs by chance: random
    walks of S games with two memories drawn evenly from −0.95 to 0.95, fitted
    as a real stretch is; the BEAT_LEVEL quantile of their highest ratio of
    periodogram to random walk at the periods a beat is looked for, in each bin
    of the two fitted memories (or over all of them, where a bin holds fewer
    than 200). Seeded, so it is the same every time.
    """
    if s not in _YARDSTICK:
        rng = np.random.default_rng(1000 + s)
        cells, peaks = [], []
        for start in range(0, BEAT_DRAWS, CHUNK):
            n = min(CHUNK, BEAT_DRAWS - start)
            m1, m2 = rng.uniform(-0.95, 0.95, n), rng.uniform(-0.95, 0.95, n)
            e = rng.standard_normal((n, s + 50))
            x = np.zeros_like(e)
            for t in range(2, s + 50):
                x[:, t] = (m1 + m2) * x[:, t - 1] - m1 * m2 * x[:, t - 2] + e[:, t]
            a, b, periodogram, null = _fit(x[:, 50:])
            cells.append(_bin(a, b))
            peaks.append((periodogram / null)[:, inside(s)].max(axis=1))
        cell, peak = np.concatenate(cells), np.concatenate(peaks)
        everywhere = float(np.quantile(peak, BEAT_LEVEL))
        k = (len(ROOT_BINS) - 1) ** 2
        level = np.full(k, everywhere)
        for c in range(k):
            here = peak[cell == c]
            if len(here) >= 200:
                level[c] = np.quantile(here, BEAT_LEVEL)
        _YARDSTICK[s] = (level, everywhere)
    return _YARDSTICK[s]


def flagged(a: np.ndarray, b: np.ndarray, periodogram: np.ndarray, null: np.ndarray, s: int) -> np.ndarray:
    """Whether each stretch's highest peak over its random walk is higher than chance makes it once in a hundred."""
    level, _ = yardstick(s)
    return (periodogram / null)[:, inside(s)].max(axis=1) > level[_bin(a, b)]


def beats(series: np.ndarray, degree: np.ndarray, s: int, rng: np.random.Generator) -> Dict[str, Any]:
    """
    The spectra of stretches of S games (rows of `series`), each against its
    own random walk: how many have a peak chance would not make; the mean
    ratio of spectrum to random walk, by the agents' connections; the two
    fitted memories by connections; and two positive controls — the share of
    beats planted one a stretch that the test finds, by their strength, and
    the mean ratio with one weak beat of SHARED_PERIOD games planted in every
    stretch.
    """
    out: Dict[str, Any] = {"stretches": int(len(series))}
    if not len(series):
        return out
    flat = series.std(axis=1) == 0
    out["flat"] = int(flat.sum())
    # A stretch is classed by its agent's connections, averaged over it and rounded.
    xs, degs = series[~flat].astype(float), np.maximum(1, np.rint(degree[~flat]))
    out["read"] = int(len(xs))
    if not len(xs):
        return out
    # The swing is read from the second memory. On a hub and its k leaves the even split multiplies a
    # departure from rest by 1/(k + 1) − 1/2 a game, never by less than −1/2 (A two-game rhythm). A
    # swing memory further below −1/2 than its scatter over S games (about 1/√S) allows, SWING_SPREAD
    # times over, is a swing that lasts longer than any hub with leaves makes it: a candidate beat of
    # two games. (Denser shapes can swing harder: two hubs that share many leaves, at about −2/3.)
    below = -0.5 - SWING_SPREAD / math.sqrt(s)
    out["swing_below"] = below
    nf = len(frequencies(s))
    classes = DEGREE_CLASSES.labels
    sums = {label: {"n": 0, "flagged": 0, "swing": 0, "ratio": np.zeros(nf), "slow": [], "swings": []}
            for label in classes}
    found = {str(a): 0 for a in BEAT_AMPLITUDES}
    shared = np.zeros(nf)
    total = np.zeros(nf)
    hits = swings = 0
    slow_all, swing_all = [], []
    t = np.arange(s)
    for start in range(0, len(xs), CHUNK):
        x, deg = xs[start:start + CHUNK], degs[start:start + CHUNK]
        a, b, periodogram, null = _fit(x)
        ratio = periodogram / null
        hit = flagged(a, b, periodogram, null, s)
        hits += int(hit.sum())
        swings += int((b < below).sum())
        total += ratio.sum(axis=0)
        slow_all.append(a)
        swing_all.append(b)
        for label, here in zip(classes, DEGREE_CLASSES.masks(deg)):
            if not here.any():
                continue
            c = sums[label]
            c["n"] += int(here.sum())
            c["flagged"] += int(hit[here].sum())
            c["swing"] += int((b[here] < below).sum())
            c["ratio"] += ratio[here].sum(axis=0)
            c["slow"].append(a[here])
            c["swings"].append(b[here])
        # Planted beats, one a stretch: a sine of a period drawn between 3 games and S/2, at a phase
        # drawn at random, with an amplitude that gives it the share a² of the stretch's variance.
        sd = x.std(axis=1, keepdims=True)
        period = rng.uniform(3, s / 2, (len(x), 1))
        phase = rng.uniform(0, 2 * np.pi, (len(x), 1))
        wave = np.sqrt(2) * np.sin(2 * np.pi * t[None, :] / period + phase)
        for amp in BEAT_AMPLITUDES:
            found[str(amp)] += int(flagged(*_fit(x + amp * sd * wave), s).sum())
        # One beat for all: SHARED_PERIOD games, a sixteenth of each stretch's variance, a phase of its own.
        wave = np.sqrt(2) * np.sin(2 * np.pi * t[None, :] / SHARED_PERIOD + rng.uniform(0, 2 * np.pi, (len(x), 1)))
        _, _, p_periodogram, p_null = _fit(x + SHARED_AMPLITUDE * sd * wave)
        shared += (p_periodogram / p_null).sum(axis=0)
    n = len(xs)
    slow, swing = np.concatenate(slow_all), np.concatenate(swing_all)
    out.update({"flagged": hits / n, "flagged_n": hits, "swing": swings / n,
                "slow": [float(q) for q in np.quantile(slow, [0.1, 0.25, 0.5, 0.75, 0.9])],
                "swing_root": [float(q) for q in np.quantile(swing, [0.1, 0.25, 0.5, 0.75, 0.9])],
                "periods": (1 / frequencies(s)).tolist(), "ratio": (total / n).tolist(),
                "planted": {amp: k / n for amp, k in found.items()}, "shared": (shared / n).tolist()})
    out["by_class"] = {}
    for label in classes:
        c = sums[label]
        if c["n"] < 5:
            continue
        out["by_class"][label] = {
            "n": c["n"], "flagged": c["flagged"] / c["n"], "swing": c["swing"] / c["n"],
            "ratio": (c["ratio"] / c["n"]).tolist(),
            "slow": [float(q) for q in np.quantile(np.concatenate(c["slow"]), [0.25, 0.5, 0.75])],
            "swing_root": [float(q) for q in np.quantile(np.concatenate(c["swings"]), [0.1, 0.25, 0.5, 0.75])]}
    return out


# ---------------------------------------------------------------------------
# Groups: flow modules, followed, against two nulls
# ---------------------------------------------------------------------------

def _cli(frames: List[Dict[str, Any]], args: List[str]) -> Dict[str, Any]:
    text = "\n".join(json.dumps(f, separators=(",", ":")) for f in frames)
    done = subprocess.run(["node", CLI] + args, input=text, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"flowmodules_cli.js {' '.join(args)}: {done.stderr.strip()[-600:]}")
    return json.loads(done.stdout)


def follow(parts: List[Dict[str, Any]], floor: float = 0.3) -> List[Dict[str, Any]]:
    """
    The modules of every frame, followed as the Research view follows them
    (web/js/flowmodules.js, track): best overlap first, each module used once,
    none below the floor. Each track: the frame it began in, how many frames it
    was followed, and its members in the first and in the last.
    """
    tracks: List[Dict[str, Any]] = []
    previous: Dict[int, set] = {}          # track -> its members in the frame before
    for k, part in enumerate(parts):
        groups: Dict[int, set] = {}
        for agent, label in zip(part["ids"], part["module"]):
            if label >= 0:
                groups.setdefault(label, set()).add(agent)
        owner = {a: t for t, members in previous.items() for a in members}
        rank_old = {t: i for i, t in enumerate(previous)}
        pairs = []
        for new_rank, (label, members) in enumerate(groups.items()):
            shared: Dict[int, int] = {}
            for a in members:
                t = owner.get(a)
                if t is not None:
                    shared[t] = shared.get(t, 0) + 1
            for t, n in shared.items():
                overlap = n / (len(previous[t]) + len(members) - n)
                if overlap >= floor:
                    pairs.append((-overlap, rank_old[t], new_rank, t, label))
        pairs.sort()
        taken_old, assigned = set(), {}
        for _, _, _, t, label in pairs:
            if t in taken_old or label in assigned:
                continue
            taken_old.add(t)
            assigned[label] = t
        now: Dict[int, set] = {}
        for label, members in groups.items():
            t = assigned.get(label)
            if t is None:
                t = len(tracks)
                tracks.append({"from": k, "life": 0, "first": members})
            tracks[t]["life"] += 1
            tracks[t]["last"] = members
            now[t] = members
        previous = now
    return tracks


def _bfs_from(adjacency: Dict[int, set], sources: set, targets: set, limit: int = 60) -> Dict[int, int]:
    """Steps from the nearest source to each target reached (within `limit`)."""
    dist = {s: 0 for s in sources if s in adjacency}
    queue = deque(dist)
    left = set(targets) - set(dist)
    while queue and left:
        x = queue.popleft()
        if dist[x] >= limit:
            continue
        for y in adjacency[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                left.discard(y)
                queue.append(y)
    return {t: dist[t] for t in targets if t in dist}


def _adjacency(pairs: np.ndarray) -> Dict[int, set]:
    adjacency: Dict[int, set] = {}
    for a, b in pairs.tolist():
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def drift(tracks: List[Dict[str, Any]], nets: List[np.ndarray], parent: Dict[int, int]) -> List[List[Any]]:
    """
    For every module followed at least LONG_LIFE frames: [frames followed, whether none of its first
    members is among its last, how far its last members stood from its first when it began]. A last
    member born since stands where its nearest forebear alive then stood; the distance is the median,
    over the last members, of the steps to the nearest first member, on the network the first frame's
    stakes ran on (`nets`: its connections, a pair a row, one array a frame).
    """
    out = []
    networks: Dict[int, Dict[int, set]] = {}
    for track in tracks:
        if track["life"] < LONG_LIFE:
            continue
        k0 = track["from"]
        if k0 not in networks:
            networks[k0] = _adjacency(nets[k0])
        adjacency = networks[k0]
        first, last = track["first"], track["last"]
        placed = set()
        for a in last:
            steps = 0
            while a not in adjacency and a in parent and steps < 10000:
                a, steps = parent[a], steps + 1
            if a in adjacency:
                placed.add(a)
        reached = _bfs_from(adjacency, first, placed)
        far = float(np.median(list(reached.values()))) if reached else None
        out.append([track["life"], not (first & last), far])
    return out


def _groups(cut: List[Dict[str, Any]], nets: Dict[int, np.ndarray], parent: Dict[int, int]) -> Dict[str, Any]:
    """The flow modules of the last games, three ways, and where the long-lived ones went."""
    out: Dict[str, Any] = {"games": len(cut)}
    if len(cut) < 2:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        for mode, args in (("flow", []), ("partners", ["--neighbours", "1"]), ("members", ["--shuffle", "1"])):
            path = os.path.join(tmp, f"{mode}.jsonl")
            answer = _cli(cut, args + ["--outlast"] + (["--partitions", path] if mode != "members" else []))
            out[mode] = {"lives": answer["lives"], "outlast": answer.get("outlast"),
                         "summary": answer.get("summary"), "sizes": answer.get("sizes")}
            if mode != "members":
                with open(path, encoding="utf-8") as f:
                    parts = [json.loads(line) for line in f if line.strip()]
                out[mode]["drift"] = drift(follow(parts), [nets[p["iteration"]] for p in parts], parent)
    return out


# ---------------------------------------------------------------------------
# One pass over a world's last games
# ---------------------------------------------------------------------------

def _want(run_id: str) -> Dict[str, Any]:
    return {"iteration": store.load_meta(run_id).get("iteration"), "version": VERSION}


def _seed(run_id: str, salt: int) -> int:
    return zlib.crc32(f"{run_id}:{salt}".encode()) & 0x7FFFFFFF


@D.measure("particles", field="key", want=_want)
def particles_pass(run_id: str) -> Dict[str, Any]:
    """
    One reading of a world's last WINDOW games, for the three detectors and the
    clocks they are measured by:

      clocks   how long the carriers last: a token on one node (the share of
               the tokens staked at home), a brain on one node (the share of
               nodes taken by a neighbour), an agent (Little's law: agents
               over newcomers a game), and the best-connected place, handed
               on to the same node or to a child of its holder (dynasties)
      rings    for each stretch length of RING_WINDOWS, the circulation's
               share, how much of it and of the downhill part the next stretch
               repeats, and the same with rings planted
      beats    for each stretch length of SEGMENTS, the spectra of agents'
               tokens and of the tokens of each agent with its neighbours
      groups   the flow modules of the last GROUP_GAMES games, three ways,
               and how far the long-lived ones went
    """
    D.needs_decisions(run_id, "who staked on whom")
    last = D.last_iteration(run_id)
    first = max(0, last - WINDOW + 1)
    group_from = max(first, last - GROUP_GAMES + 1)
    rings = {w: Rings(w, np.random.default_rng(_seed(run_id, w))) for w in RING_WINDOWS}
    # An agent's tokens, the tokens of it and its neighbours, and its connections, after every game
    # while it lives: three growing arrays of single floats, which a world of 20,000 agents can hold.
    runs: Dict[int, Tuple[array, array, array]] = {}
    stretches = {s: {"agent": [], "around": [], "degree": []} for s in SEGMENTS}
    home = staked = 0.0
    taken = contested = 0
    agents_seen: List[int] = []
    newcomers: List[int] = []
    top: List[Tuple[int, int]] = []                     # best-connected agent after each game, and its parent
    cut: List[Dict[str, Any]] = []
    nets: Dict[int, np.ndarray] = {}
    parent: Dict[int, int] = {}
    previous_ids: Optional[set] = None

    def close(agent: int) -> None:
        values = [np.frombuffer(v, dtype=np.float32) for v in runs.pop(agent)]
        for s in SEGMENTS:
            for k in range(len(values[0]) // s):
                part = slice(k * s, (k + 1) * s)
                stretches[s]["agent"].append(values[0][part])
                stretches[s]["around"].append(values[1][part])
                stretches[s]["degree"].append(float(values[2][part].mean()))

    for t in range(first, last + 1):
        game = D.frame_at(run_id, t, 2)
        ids = game["ids"]
        if not ids:
            break
        allocations = (game.get("decisions") or {}).get("allocations") or []
        winners = (game.get("decisions") or {}).get("winners") or []
        src, dst, amount = [], [], []
        links = []
        for r in allocations:
            u = r["agent"]
            staked += r["tokens"]
            for v, x in zip(r["targets"], r["alloc"]):
                if v == u:
                    home += x
                    continue
                if u < v:
                    links.append((u, v))
                if x > 0:
                    src.append(u)
                    dst.append(v)
                    amount.append(x)
        # The network the game was played on: every agent's candidates, each connection once.
        pairs = np.unique(np.array(links, dtype=np.int64).reshape(-1, 2), axis=0) if links else \
            np.zeros((0, 2), dtype=np.int64)
        taken += sum(1 for w in winners if w["winner"] != w["node"])
        contested += len(winners)
        src_a, dst_a, amount_a = (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
                                  np.array(amount, dtype=float))
        for ring in rings.values():
            ring.add(src_a, dst_a, amount_a)

        tokens = np.asarray(game["tokens"], dtype=float)
        index = {a: i for i, a in enumerate(ids)}
        edges = np.asarray(game["edges"], dtype=np.int64).reshape(-1, 2)
        if len(edges):
            ia = np.array([index[a] for a in edges[:, 0].tolist()])
            ib = np.array([index[b] for b in edges[:, 1].tolist()])
            around = tokens + np.bincount(ia, weights=tokens[ib], minlength=len(ids)) \
                + np.bincount(ib, weights=tokens[ia], minlength=len(ids))
            degree = np.bincount(ia, minlength=len(ids)) + np.bincount(ib, minlength=len(ids))
        else:
            around, degree = tokens.copy(), np.zeros(len(ids), dtype=int)
        alive = set(ids)
        for a in [a for a in runs if a not in alive]:
            close(a)
        for a, x, y, z in zip(ids, tokens.tolist(), around.tolist(), degree.tolist()):
            kept = runs.get(a)
            if kept is None:
                kept = runs[a] = (array("f"), array("f"), array("f"))
            kept[0].append(x)
            kept[1].append(y)
            kept[2].append(z)
        agents_seen.append(len(ids))
        if previous_ids is not None:
            newcomers.append(len(alive - previous_ids))
        previous_ids = alive
        parents = game.get("parent_ids") or [-1] * len(ids)
        h = int(np.argmax(degree))
        top.append((ids[h], parents[h]))
        if t >= group_from:
            for a, p in zip(ids, parents):
                if p is not None and p >= 0:
                    parent.setdefault(a, p)
            cut.append({"iteration": game["iteration"], "phase": 2, "ids": ids,
                        "decisions": {"allocations": allocations}})
            nets[game["iteration"]] = pairs
    for a in list(runs):
        close(a)

    # Dynasties of the best-connected place: the same node, or a child of the one before, game after game.
    dynasties, length = [], 0
    for k, (who, his_parent) in enumerate(top):
        if k and (who == top[k - 1][0] or his_parent == top[k - 1][0]):
            length += 1
        else:
            if length:
                dynasties.append(length)
            length = 1
    if length:
        dynasties.append(length)

    rng = np.random.default_rng(_seed(run_id, 7))
    out_beats = {}
    for s in SEGMENTS:
        degree = np.array(stretches[s]["degree"], dtype=float)
        out_beats[str(s)] = {
            unit: beats(np.array(stretches[s][unit], dtype=float).reshape(-1, s), degree, s, rng)
            for unit in ("agent", "around")}
        # The stretches are read; let them go before the modules' program takes its own share of memory.
        stretches[s] = {}
    births = float(np.mean(newcomers)) if newcomers else 0.0
    return {
        "from": first, "to": last,
        "clocks": {"token_home": home / staked if staked else None,
                   "token_stay": 1 / (1 - home / staked) if staked and home < staked else None,
                   "taken": taken / contested if contested else None,
                   "brain_stay": contested / taken if taken else None,
                   "agents": float(np.mean(agents_seen)) if agents_seen else 0.0,
                   "newcomers": births,
                   "agent_life": float(np.mean(agents_seen)) / births if births else None,
                   "dynasties": dynasties,
                   "dynasty_longest": max(dynasties) if dynasties else 0,
                   "games": len(agents_seen)},
        "rings": {str(w): ring.summary() for w, ring in rings.items()},
        "beats": out_beats,
        "groups": _groups(cut, nets, parent),
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def _med(values: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _lives_at_least(lives: Dict[str, int], games: int) -> Optional[float]:
    total = sum(lives.values())
    return sum(n for life, n in lives.items() if int(life) >= games) / total if total else None


def _life_quantile(lives: Dict[str, int], q: float) -> Optional[float]:
    if not lives:
        return None
    life = np.repeat([int(k) for k in lives], [int(v) for v in lives.values()])
    return float(np.quantile(life, q))


def _drift(rows: List[List[Any]]) -> Dict[str, Optional[float]]:
    """The long-lived modules of a world: how many, the share renewed, and how far they went (all, and renewed)."""
    far = [r[2] for r in rows if r[2] is not None]
    renewed_far = [r[2] for r in rows if r[1] and r[2] is not None]
    return {"modules": len(rows), "renewed": float(np.mean([r[1] for r in rows])) if rows else None,
            "far_mean": float(np.mean(far)) if far else None, "far_median": float(np.median(far)) if far else None,
            "far_2": float(np.mean([f >= 2 for f in far])) if far else None,
            "far_renewed": float(np.mean(renewed_far)) if renewed_far else None,
            "life": float(np.mean([r[0] for r in rows])) if rows else None}


#: Random walks whose swing memory is known, (slow memory, swing memory): hubs with leaves alone swing at
#: −1/2 at most, two hubs sharing many leaves at about −2/3.
SWING_CASES = ((0.85, -0.5), (0.5, -0.5), (0.0, -0.5), (0.85, -2 / 3))


def swing_chance(s: int, draws: int = 20000) -> List[Dict[str, float]]:
    """
    How often a random walk with a known swing memory is fitted below the line
    of a swing that outlasts a hub's, in stretches of S games, and how its
    fitted memory comes out: what the share of real stretches below the line
    is to be set beside. Seeded, so it is the same every time.
    """
    rng = np.random.default_rng(2000 + s)
    below = -0.5 - SWING_SPREAD / math.sqrt(s)
    out = []
    for slow, swing in SWING_CASES:
        e = rng.standard_normal((draws, s + 50))
        x = np.zeros_like(e)
        for t in range(2, s + 50):
            x[:, t] = (slow + swing) * x[:, t - 1] - slow * swing * x[:, t - 2] + e[:, t]
        _, fitted, _, _ = _fit(x[:, 50:])
        out.append({"slow": slow, "swing": swing, "below": float(np.mean(fitted < below)),
                    "median": float(np.median(fitted)), "sd": float(fitted.std())})
    return out


def _codes(ids: List[str]) -> str:
    """'`a`, `b` and `c`'."""
    named = ["`" + i + "`" for i in ids]
    return named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]


def _runs_text(ws: List[Dict[str, Any]]) -> str:
    b1 = [w["id"] for w in ws if w["group"] == "B1"]
    star = [w["id"] for w in ws if w["group"] == "B1 star"]
    r3 = [w["id"] for w in ws if w["group"] == "round 3"]
    return (f"Twelve worlds, each read over the last {WINDOW} games of its run. Five worlds of B1 at 100,000 tokens "
            f"(Experiment 10: {_codes(b1)}, the first five seeds whose world is not held together by one hub) and "
            f"its largest star ({_codes(star)}), iterations 1,000 to 1,499. The five healthiest worlds of round 3 "
            f"by the product ([Which worlds are healthy?](which-worlds-are-healthy.md); Experiments 40 and 42: "
            f"{_codes(r3)}), 30,000 tokens, iterations 2,500 to 2,999. And a world whose brains never change and "
            f"which froze long before its end (Experiment 7, `{FROZEN}`), iterations 2,500 to 2,999.")


DATA = ("each run's frames, read once by `particles_pass` in `book_chapters/particles.py` (every game's "
        "`decisions.allocations` and `decisions.winners`, and the agents, tokens, connections and parents after "
        "it) and kept in `GraphOfLifeRuns/.book/<run>.particles.json`")


@chapter
def particles(ch: F.Chapter) -> None:
    ws = worlds()
    kept = {w["id"]: particles_pass(w["id"]) for w in ws}
    runs = _runs_text(ws)
    real = [w for w in ws if w["group"] != "frozen"]
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for w in ws:
        by_group.setdefault(w["group"], []).append(w)
    ch.number("worlds", [{"id": w["id"], "label": w["label"], "group": w["group"],
                          "from": kept[w["id"]]["from"], "to": kept[w["id"]]["to"],
                          "agents": kept[w["id"]]["clocks"]["agents"]} for w in ws])

    # ---- 1. The clocks: how long a carrier lasts, and how long the lasting patterns do.
    def module_life(r: Dict[str, Any], mode: str = "flow") -> Optional[float]:
        g = r["groups"].get(mode) or {}
        return _life_quantile(g.get("lives") or {}, 0.99)

    clocks = [("a token on a node", lambda r: r["clocks"]["token_stay"]),
              ("a brain on a node", lambda r: r["clocks"]["brain_stay"]),
              ("an agent", lambda r: r["clocks"]["agent_life"]),
              ("a group of the flow", module_life),
              ("the best-connected place", lambda r: float(r["clocks"]["dynasty_longest"]))]
    series = []
    for k, (name, get) in enumerate(clocks):
        for g, members in by_group.items():
            values = [get(kept[w["id"]]) for w in members]
            pts = [(x, v) for x, v in zip(F.jitter(k, len(values), 11 + k + 7 * len(series), 0.25), values)
                   if v is not None]
            if pts:
                series.append({"label": g if k == 0 else None, "x": [p[0] for p in pts], "y": [p[1] for p in pts],
                               "points": True, "size": 6, "colour": GROUP_COLOUR[g], "alpha": 0.85})
    ch.figure("clocks", title="How long things last",
              x={"categories": [c[0] for c in clocks]}, y={"label": "games (logarithmic)", "log": True, "min": 1,
                                                           "max": 1000},
              series=series,
              caption=(f"Each dot is a world, read over the last {WINDOW} games of its run. A token on a node: how "
                       "many games a token stays on the node it is on, on average, from the share of all tokens "
                       "staked at home. A brain on a node: how many games a node goes, on average, before a "
                       "neighbour takes it and its brain is copied in. An agent: how long an agent lives, the agents "
                       "alive over the "
                       "newcomers a game (no dot for the frozen world, in which nobody is born or dies). A group of "
                       "the flow: of the flow modules of the last 50 games, the life that one in a hundred reaches "
                       "or passes (50 at most, the games followed). The best-connected place: the longest run of "
                       f"games in which the best-connected agent stayed the same or was a child of the one before "
                       f"({WINDOW} at most). B1: five worlds of B1 at 100,000 tokens; B1 star: its largest star; "
                       "round 3: the five healthiest worlds of round 3; frozen: a world whose brains never change."),
              recipe=recipe(runs, DATA, [
                  "A token: 1/(1 − h), with h the share of all tokens staked in the games that went to the "
                  "staker's own node.",
                  "A brain: the nodes on which anyone staked over those taken by a neighbour.",
                  "An agent: the mean number of agents after a game over the mean number that were not alive "
                  "after the game before (Little's law).",
                  "A group: flow modules followed as on [Flow modules](measuring-tokens.md#following-groups-through-time) "
                  "(web/js/flowmodules.js through tools/flowmodules_cli.js); the 99th percentile of their lives.",
                  "A place: after every game the agent with the most connections; a dynasty goes on while the "
                  "next one is the same agent or its child."]))
    for g, members in by_group.items():
        ch.number(f"clocks:{g}", {name: {"median": _med([get(kept[w["id"]]) for w in members]),
                                          "min": min((v for v in (get(kept[w["id"]]) for w in members)
                                                      if v is not None), default=None),
                                          "max": max((v for v in (get(kept[w["id"]]) for w in members)
                                                      if v is not None), default=None)}
                                   for name, get in clocks})

    # ---- 2. Rings.
    lengths = [int(w) for w in RING_WINDOWS]
    persistence, slope = [], []
    for w in ws:
        r = kept[w["id"]]["rings"]
        for out, key in ((persistence, "persistence"), (slope, "slope")):
            ys = [r[str(L)][key] for L in lengths]
            out.append(F.line(lengths, ys, None, GROUP_COLOUR[w["group"]], width=1.5, alpha=0.8))
            # A dot at every value, so that a world with a value at one length only is seen too.
            out.append(F.line(lengths, ys, None, GROUP_COLOUR[w["group"]], points=True, size=5, alpha=0.9))
    labels = []
    for g in by_group:
        labels.append(F.line([lengths[0]], [None], g, GROUP_COLOUR[g], width=2))
    planted = []
    for share, dash in zip(PLANTED, ([5, 4], [2, 3])):
        ys = [_med([kept[w["id"]]["rings"][str(L)]["planted"].get(str(share)) for w in real]) for L in lengths]
        planted.append(F.line(lengths, ys, f"rings planted, {share:.0%} of the circulation", GREY, width=2,
                              dash=dash))
        for L, y in zip(lengths, ys):
            ch.number(f"rings:planted:{share}:{L}", y)
    ch.grid("rings", [
        {"title": "rings: the circulation",
         "x": {"label": "games in a stretch", "log": True}, "y": {"label": "correlation", "min": -1, "max": 1},
         "series": persistence + planted + labels, "guides": [{"axis": "y", "at": 0, "label": ""}]},
        {"title": "the downhill part",
         "x": {"label": "games in a stretch", "log": True}, "y": {"label": "correlation", "min": -1, "max": 1},
         "series": slope, "guides": [{"axis": "y", "at": 0, "label": ""}], "legend": False}],
        columns=2, title="Do tokens go round the same loops?",
        caption=("The net flow of a stretch of games (what each connection carried one way, minus what came back) "
                 "is split into a part that runs downhill and a part that goes round loops, the circulation. Left: "
                 "the correlation of the circulation of one stretch with that of the next, over the connections "
                 "both used; a ring that lasts gives two stretches the same circulation, and a correlation near 1. "
                 "Each line is a world (orange: B1; red: its largest star; green: round 3; violet: the frozen "
                 "world, whose circulation turns round every game, and whose net flow over any even number of "
                 "games is nothing at all). Grey: the median of the eleven worlds that are not frozen, with rings "
                 "planted on triangles and squares of the connections both stretches used, the same rings in both, "
                 "carrying 5% or 20% of the circulation there. Right: the same for the downhill part. A swing of "
                 "the flow back and forth from one game to the next shows at one game, below 0, and cancels in the "
                 "stretches of an even number of games."),
        recipe=recipe(runs, DATA, [
            "For every game, the tokens each agent staked on each neighbour's node.",
            "Add them up over stretches of " + ", ".join(str(L) for L in lengths[:-1]) + f" and {lengths[-1]} "
            "games, one way minus the other: the net flow of each connection (from the first game read, at most "
            f"{RING_PAIRS + 1} stretches in a row).",
            "Find the potential that explains as much of the net flow as possible as running downhill (least "
            "squares, by conjugate gradients on the network's Laplacian, after pruning dead-end branches, on which "
            "no flow goes round); the circulation is the rest ([Flows that go round](measuring-tokens.md#flows-that-go-round)).",
            "Correlate each stretch's circulation, and its downhill part, with the next stretch's, over the "
            "connections both used; take the median over the pairs of stretches.",
            "Positive control: on the connections two stretches share, loops of three or four connections "
            f"running along {PLANTED_EDGES:.0%} of them, each connection used once; the same ring on each loop "
            "in both stretches, as strong as makes the rings carry 5% or 20% of the first stretch's circulation "
            "on those connections (sum of squares); the same correlation. Rings that last, and nothing else, "
            "would give about 0.05 and 0.17."]))
    for w in ws:
        r = kept[w["id"]]["rings"]
        ch.number(f"rings:{w['label']}", {L: {k: r[L][k] for k in ("persistence", "slope", "share", "common",
                                                                     "pairs", "persistence_q")} for L in r})
    for g, members in by_group.items():
        ch.number(f"rings:{g}", {str(L): {k: _med([kept[w["id"]]["rings"][str(L)][k] for w in members])
                                          for k in ("persistence", "slope", "share", "common")} for L in lengths})

    # ---- 3. Beats.
    def beat(r: Dict[str, Any], s: int, unit: str = "agent") -> Dict[str, Any]:
        return r["beats"][str(s)][unit]

    def enough(b: Dict[str, Any]) -> bool:
        return b.get("read", 0) >= 100

    def spectrum_panel(title: str, members: List[Dict[str, Any]], s: int) -> Dict[str, Any]:
        # Periods from two games to half the stretch: anything slower is a trend inside the stretch.
        out = []
        shared = []
        periods = np.zeros(0)
        for w in members:
            b = beat(kept[w["id"]], s)
            if not enough(b):
                continue
            periods = np.array(b["periods"])
            keep = periods <= s / 2
            out.append(F.line(periods[keep], np.array(b["ratio"])[keep], None, GROUP_COLOUR[w["group"]], width=1.5,
                              alpha=0.85))
            shared.append(np.array(b["shared"])[keep])
        if shared:
            out.append(F.line(periods[periods <= s / 2], np.median(np.array(shared), axis=0),
                              f"a beat of {SHARED_PERIOD} games planted in every stretch", GREY, width=2, dash=[5, 4]))
        return {"title": title, "x": {"label": "period, games (logarithmic)", "log": True, "min": 2, "max": s / 2},
                "y": {"label": "spectrum over the random walk's", "min": 0}, "series": out,
                "guides": [{"axis": "y", "at": 1, "label": ""}]}
    b1 = by_group["B1"] + by_group["B1 star"]
    r3 = by_group["round 3"]
    panels = [spectrum_panel(f"B1 at 100,000 tokens: stretches of {max(SEGMENTS)} games", b1, max(SEGMENTS)),
              spectrum_panel(f"round 3: stretches of {min(SEGMENTS)} games", r3, min(SEGMENTS))]
    # Found against planted: the share of stretches flagged, by the share of the variance a planted beat carries.
    sens = []
    xs = [0.0] + [a * a for a in BEAT_AMPLITUDES]
    for name, members, s, colour, dash in (("B1, 128 games", b1, 128, ORANGE, None),
                                           ("B1, 32 games", b1, 32, ORANGE, [5, 4]),
                                           ("round 3, 32 games", r3, 32, GREEN, None)):
        rows = [beat(kept[w["id"]], s) for w in members]
        rows = [b for b in rows if enough(b)]
        if not rows:
            continue
        ys = [_med([b["flagged"] for b in rows])] + [_med([b["planted"][str(a)] for b in rows])
                                                      for a in BEAT_AMPLITUDES]
        sens.append(F.line(xs, ys, name, colour, width=2, dash=dash))
        ch.number(f"beats:found:{name}", dict(zip(["real"] + [str(a) for a in BEAT_AMPLITUDES], ys)))
    panels.append({"title": "stretches with a beat",
                   "x": {"label": "share of a stretch's variance a planted beat carries", "min": 0, "max": 1},
                   "y": {"label": "share of stretches", "min": 0, "max": 1}, "series": sens,
                   "guides": [{"axis": "y", "at": 1 - BEAT_LEVEL, "label": "chance"}]})
    # The swing memory of an agent's tokens, by its connections, against a hub's.
    memory = []
    labels = DEGREE_CLASSES.labels
    for w in ws:
        s = 32 if w["group"] == "round 3" else max(SEGMENTS)
        b = beat(kept[w["id"]], s)
        by = b.get("by_class") or {}
        pts = [(k + (0.12 if w["group"] == "round 3" else -0.12), by[c]["swing_root"][2])
               for k, c in enumerate(labels) if c in by]
        if pts:
            memory.append({"label": None, "x": [p[0] for p in pts], "y": [p[1] for p in pts], "points": True,
                           "size": 5, "colour": GROUP_COLOUR[w["group"]], "alpha": 0.85})
    # A hub of k leaves, at a typical k of each class from two connections on (a leaf swings with its hub).
    hub = [1 / (k + 1) - 0.5 for k in (2, 3.5, 7, 25, 100)]
    memory.append(F.line(list(range(1, len(labels))), hub, "a hub with only leaves, 1/(k + 1) − 1/2", GREY,
                         width=1.5, dash=[5, 4]))
    panels.append({"title": "the swing, by connections",
                   "x": {"label": "connections", "categories": labels},
                   "y": {"label": "swing memory (median)", "min": -1, "max": 0.25}, "series": memory,
                   "guides": [{"axis": "y", "at": -0.5 - SWING_SPREAD / math.sqrt(max(SEGMENTS)),
                               "label": "a swing that outlasts a hub's"}]})
    ch.grid("beats", panels, columns=2, title="Does anything keep a rhythm?",
            caption=("Top: the spectrum of agents' tokens, game by game, divided by the spectrum of a random walk "
                     "with the same two memories, one slow and one that swings back and forth, fitted to each "
                     "stretch; averaged over all the stretches of a world, at periods from two games to half a "
                     "stretch (anything slower is a trend inside the stretch). Flat means no rhythm beyond the "
                     "random walk's. Each line is a world (orange: B1; red: its largest star; green: the round-3 worlds "
                     f"with at least 100 stretches). Dashed grey: the same with a weak beat of {SHARED_PERIOD} games "
                     "planted in every stretch, carrying a sixteenth of its variance (the median of the worlds). "
                     "Bottom left: the share of stretches with a peak, at a period from 3 games to half the "
                     "stretch, that a random walk makes less than once in a hundred; at 0, the real stretches; "
                     "further right, with a beat of a random period planted in each, carrying that share of its "
                     "variance (the median of the worlds). Bottom right: the swing memory of agents' tokens, by "
                     "their connections: how much of a swing is left a game later, with its sign (stretches of 128 "
                     "games; 32 in round 3, drawn a little to the right); one dot per world and class, the median. "
                     "Dashed: what the even split leaves of a swing between a hub and its leaves alone; below the "
                     "dotted line, a swing that lasts longer than any such hub's."),
            recipe=recipe(runs, DATA, [
                "For every agent, its tokens after every game it was alive in the games read.",
                f"Cut each agent's life into stretches of {min(SEGMENTS)} and of {max(SEGMENTS)} games, back to back; "
                "leave out the stretches in which its tokens never change.",
                "For each stretch, its correlations with itself one and two games on, r1 and r2; the AR(2) "
                "process they give (Yule and Walker), with real roots a ≥ b, the slow memory and the swing memory "
                "(where the roots would be complex, the nearest pair of equal real roots); its spectrum "
                "σ² / (|1 − a e^(−2πif)|² |1 − b e^(−2πif)|²), scaled to the stretch's variance; and the "
                f"periodogram, on a grid {PAD} times finer than the stretch's own frequencies.",
                f"Chance: {BEAT_DRAWS:,} random walks of each length with two memories drawn from −0.95 to 0.95, "
                "fitted the same way; in each bin of the two fitted memories, the highest ratio of periodogram to "
                "spectrum, at periods from 3 games to half the stretch, that one in a hundred reaches. A real "
                "stretch whose highest ratio is above it is flagged.",
                "Positive controls: a sine of a period drawn between 3 games and half the stretch, at a random "
                "phase, added to every stretch with the share 1/16, 1/4, 9/16 or 1 of its variance, and the "
                f"stretches flagged again; and a sine of {SHARED_PERIOD} games with 1/16 of the variance added to "
                "every stretch, averaged as the real ones.",
                f"The two-game swing: a swing memory b below −1/2 − {SWING_SPREAD}/√S. Its chance: 20,000 random "
                "walks with slow memory 0.85, 0.5 or 0 and swing memory −1/2, and with 0.85 and −2/3, fitted the "
                "same way, and the share of them below the line."]))
    ch.number("beats:swing_chance", {str(s): swing_chance(s) for s in SEGMENTS})
    for w in ws:
        bb = kept[w["id"]]["beats"]
        ch.number(f"beats:{w['label']}", {
            s: {u: {k: bb[s][u].get(k) for k in ("stretches", "read", "flat", "flagged", "swing", "slow",
                                                  "swing_root", "planted")}
                for u in bb[s]} for s in bb})
        for s in bb:
            for u in bb[s]:
                per = bb[s][u].get("by_class") or {}
                ch.number(f"beats:class:{w['label']}:{s}:{u}", {c: {"n": v["n"], "flagged": v["flagged"],
                                                                     "swing": v["swing"], "slow": v["slow"],
                                                                     "swing_root": v["swing_root"]}
                                                                 for c, v in per.items()})
                b = bb[s][u]
                if b.get("periods"):
                    per_ = np.array(b["periods"])
                    middle = (per_ >= 3) & (per_ <= int(s) / 2)
                    at = int(np.argmin(np.abs(per_ - SHARED_PERIOD)))
                    ch.number(f"beats:ratio:{w['label']}:{s}:{u}", {
                        "max": float(np.max(np.array(b["ratio"])[middle])),
                        "min": float(np.min(np.array(b["ratio"])[middle])),
                        "at_shared": float(b["ratio"][at]), "shared_at_shared": float(b["shared"][at]),
                        "two": float(b["ratio"][-1])})

    # ---- 4. Groups: how long modules live, three ways.
    games = [1, 2, 5, 10, 20, 40, 50]
    panels = []
    for g, members in by_group.items():
        def curve(mode: str) -> List[Optional[float]]:
            return [_med([_lives_at_least(kept[w["id"]]["groups"].get(mode, {}).get("lives") or {}, L)
                          for w in members]) for L in games]
        floor = lambda ys: [max(y, 1e-4) if y is not None else None for y in ys]
        flow, partners, dealt = curve("flow"), curve("partners"), curve("members")
        panels.append({"title": g + (f" ({len(members)} worlds)" if len(members) > 1 else ""),
                       "x": {"label": "games", "log": True, "min": 1, "max": 50},
                       "y": {"label": "share living at least this long", "log": True, "min": 1e-4, "max": 1},
                       "series": [F.line(games, floor(flow), "the flow", GROUP_COLOUR[g], width=2),
                                  F.line(games, floor(partners), "partners dealt", GROUP_COLOUR[g], width=1.5,
                                         dash=[5, 4]),
                                  F.line(games, floor(dealt), "members dealt", GREY, width=1.5, dash=[2, 3])]})
        ch.number(f"groups:lives:{g}", {str(L): {"flow": f, "partners": p, "members": m}
                                        for L, f, p, m in zip(games, flow, partners, dealt)})
    ch.grid("groups", panels, columns=2, title="Groups that last",
            caption=("Of all the flow modules of the last 50 games of each world, the share followed for at least "
                     "so many games (the median of each kind's worlds; logarithmic axes, a share of 0 drawn at the "
                     "bottom). The flow: as it was. Partners dealt: the same games with every agent's stakes on its "
                     "neighbours dealt among the same neighbours at random, which keeps the network and undoes the "
                     "agents' choice of whom to stake on. Members dealt: the real modules with their agents dealt "
                     "among them at random, game by game, which keeps only their sizes. B1: five worlds of B1 at "
                     "100,000 tokens; B1 star: its largest star; round 3: the five healthiest worlds of round 3; "
                     "frozen: a world whose brains never change, the positive control, whose modules all last."),
            recipe=recipe(runs, DATA, [
                "For each of the last 50 games, the flow between each pair of neighbours: the tokens staked "
                "either way.",
                "Group the agents by the map equation (web/js/flowmodules.js, through tools/flowmodules_cli.js) "
                "and follow the groups from game to game: best overlap first, each used once, none below 0.3 of "
                "the members two modules share ([Flow modules](measuring-tokens.md#following-groups-through-time)).",
                "Do the same with every agent's stakes on its neighbours dealt among them at random "
                "(`--neighbours 1`) and with each game's agents dealt among modules of its sizes (`--shuffle 1`).",
                "The share of modules followed for at least so many games; the median of each kind's worlds."]))

    # ---- 5. Whether long-lived groups renew themselves, and whether they move.
    cats = [w["label"] for w in ws]
    renewed, moved = [], []
    for mode, colour_of, dx in (("flow", lambda w: GROUP_COLOUR[w["group"]], -0.12), ("partners", lambda w: GREY, 0.12)):
        for k, w in enumerate(ws):
            d = _drift(kept[w["id"]]["groups"].get(mode, {}).get("drift") or [])
            label = ("the flow" if mode == "flow" else "partners dealt") if k == 0 else None
            if d["renewed"] is not None:
                renewed.append({"label": label, "x": [k + dx], "y": [d["renewed"]], "points": True, "size": 6,
                                "colour": colour_of(w), "alpha": 0.9})
            if d["far_mean"] is not None:
                moved.append({"label": label, "x": [k + dx], "y": [d["far_mean"]], "points": True, "size": 6,
                              "colour": colour_of(w), "alpha": 0.9})
            ch.number(f"drift:{w['label']}:{mode}", d)
    ch.grid("drift", [
        {"title": "long-lived modules that kept none of their first members",
         "x": {"categories": cats}, "y": {"label": "share of those modules", "min": 0, "max": 1},
         "series": renewed},
        {"title": "how far they went",
         "x": {"categories": cats}, "y": {"label": "steps from the first members (mean)", "min": 0},
         "series": moved, "legend": False}],
        columns=1, title="Groups that outlast their members, and whether they move",
        caption=("Of the flow modules of the last 50 games followed for 10 games or more, in each world (the "
                 "worlds' short names; B1 at 100,000 tokens, its star s28, the five round-3 worlds, and the frozen "
                 "world). Top: the share whose members at the end included none of those it began with. Bottom: "
                 "how far its last members stood from its first when it began, in steps through the network of "
                 "its first game; a member born since stands where its nearest forebear alive then stood; the "
                 "median over the last members, averaged over the modules. Coloured: the flow as it was; grey: "
                 "with every agent's partners dealt at random among its neighbours, the network's own regions."),
        recipe=recipe(runs, DATA, [
            "Follow the flow modules of the last 50 games as for the figure above, the real ones and those of the "
            "flow with partners dealt.",
            "For every module followed 10 games or more: its members in the first game and in the last.",
            "Renewed: none of the first members is among the last.",
            "Moved: each last member, or its nearest forebear that was alive in the module's first game (parents "
            "from the frames' `parent_ids`); its steps to the nearest first member on the network that game's "
            "stakes ran on; the median over the last members; the mean over the modules."]))
    for g, members in by_group.items():
        ch.number(f"drift:{g}", {mode: {k: _med([_drift(kept[w["id"]]["groups"].get(mode, {}).get("drift") or [])[k]
                                                 for w in members]) for k in ("modules", "renewed", "far_mean",
                                                                              "far_median", "far_2", "far_renewed",
                                                                              "life")}
                                 for mode in ("flow", "partners")})
        ch.number(f"groups:{g}", {mode: {
            "at_least_10": _med([_lives_at_least(kept[w["id"]]["groups"].get(mode, {}).get("lives") or {}, 10)
                                 for w in members]),
            "turnover": _med([(kept[w["id"]]["groups"].get(mode, {}).get("summary") or {}).get("meanTurnover")
                              for w in members]),
            "born": _med([(kept[w["id"]]["groups"].get(mode, {}).get("summary") or {}).get("regenerationShare")
                          for w in members]),
            "compression": _med([(kept[w["id"]]["groups"].get(mode, {}).get("summary") or {}).get("compression")
                                 for w in members]),
            "size": _med([(lambda s: sum(int(k) * v for k, v in s.items()) / sum(s.values()) if s else None)(
                kept[w["id"]]["groups"].get(mode, {}).get("sizes") or {}) for w in members])}
            for mode in ("flow", "partners", "members")})

    # ---- Every world in one table.
    def pct(v: Optional[float]) -> str:
        return "—" if v is None else f"{100 * v:.1f}%"

    def num(v: Optional[float], digits: int = 1) -> str:
        return "—" if v is None else f"{v:,.{digits}f}"
    rows = []
    for w in ws:
        r = kept[w["id"]]
        c, g = r["clocks"], r["groups"]
        b32 = beat(r, 32)
        flow_d, part_d = _drift(g.get("flow", {}).get("drift") or []), _drift(g.get("partners", {}).get("drift") or [])
        rows.append([{"text": w["label"], "href": f"run:{w['id']}"}, w["group"], f"{c['agents']:,.0f}",
                     num(c["agent_life"]), num(c["token_stay"], 2), num(c["brain_stay"], 2),
                     num(r["rings"]["2"]["persistence"], 3),
                     f"{pct(b32.get('flagged'))} of {b32.get('read', 0):,}",
                     " / ".join(pct(_lives_at_least(g.get(m, {}).get("lives") or {}, 10))
                                for m in ("flow", "partners", "members")),
                     f"{pct(flow_d['renewed'])} / {pct(part_d['renewed'])}",
                     f"{num(flow_d['far_mean'], 2)} / {num(part_d['far_mean'], 2)}"])
    ch.number("table:worlds", {
        "head": ["world", "kind", "agents", "an agent lives, games", "a token stays, games",
                 "a brain stays, games", "rings: what the next two games repeat", "beats flagged, stretches of 32",
                 "modules living 10+ games: flow / partners dealt / members dealt",
                 "of those, renewed: flow / partners", "of those, steps moved: flow / partners"],
        "rows": rows})
