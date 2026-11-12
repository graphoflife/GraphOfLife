"""
The brain-copy tree of round 3's worlds, and its shadow world of pure luck
(2026-11-11).

An independent review found that the family trees of research/screen/
candidates.py follow births only, although a world copies brains far more
often by conquest: a node won in the game keeps its own agent id and gets a
copy of the winner's brain (GraphOfLife.blotto_phase). So a world's agents
are not what descends; its brains are. This follows brain *instances*:

- an instance begins at a founder, at a birth (its parent is the paying
  parent's instance; the mate is recorded where the frame names one, under
  mate_pays) or at a conquest (its parent is the winner's instance at that
  game, as the game reads the brains from before it: every conquest of one
  game copies from the brains the game began with);
- it lasts while it holds its node: until the node is conquered, or dies;
- a mutation in place (after every game, unless brains change only when
  copied) changes the brain's genotype id but not the instance: it is
  counted as a change of the instance, and kept in the genotype forest.

One pass over a world's frames builds the tree (`Pass`); the same pass keeps
what test 4 (research/screen/selection.py) needs, the decisions of every agent
alive at the start of each iteration and how many descendants it had at the
next. The tree is then read (`readings`):

- the share of copies made by birth and by conquest;
- generations: copy events along each path, births and conquests apart;
- the ancestors of the final population back in time, and their latest
  common ancestor;
- the line of descent: in each kept era, the ancestor instance of the most
  final agents;
- an effective population, from how fast the ancestry narrows on this tree;
- heredity pairs (parent and child instances alive together) for
  research/screen/brainplay.py, and the genotype events along the line;
- the shadow world of pure luck (test 3 of book/pages/did-evolution-happen.md):
  the same copy events, when and how many, and the same living population, but
  each copy's parent drawn uniformly among the living at that moment. Only the
  events are shuffled, never the brains, so a shadow costs a few seconds: each
  parent is a hash of (seed, replicate, instance), the same whichever way the
  tree is walked, so the ancestry of the final population is traced backwards
  and the clades of each era are counted forwards from the same draws.

    python3 research/screen/braintree.py                  every living world of round 3, one at a time
    python3 research/screen/braintree.py RUN [RUN ...]    those worlds
    python3 research/screen/braintree.py --spot-check     conquests per agent per game in three worlds

Readings are kept in $GOL_RUNS_DIR/.pilots/screen/braintree/<run>.json, item by
item as they are made; the arrays of the tree in a work folder beside it, until
the world's readings are complete.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import gol_store  # noqa: E402
from gol_config import SimConfig  # noqa: E402

#: The shadow world's replicates, and the seed every random draw of this file starts from.
SHADOWS, SEED = 100, 11
#: Parent-child pairs sampled in each kept world for heredity (as research/screen/candidates.py plays them).
PAIRS = 150
#: How far back, in iterations, the ancestors of the final population are reported (as candidates.lineage does).
LAGS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000)
#: The traits of test 4, read from the decisions of every agent alive at the start of an iteration.
TRAITS = ("child_share", "links", "spread", "home")
#: What each sum of a trait's row is (research/screen/selection.py reads them by these names).
SUMS = ("n", "z", "zz", "w", "ww", "wz", "s", "sz", "x", "xx", "wx", "zx", "sx")

# How a genotype came about, in the genotype forest.
UNSEEN, FOUNDER, MUTATION, EITHER, MIX = -1, 0, 1, 2, 3
# EITHER: a child's new genotype whose parent is the paying parent's, in a world that mixes at birth: one
# change, a mix or a mutation, which the ids cannot tell apart. MIX: the hidden genotype of a mix that then
# mutated (its id is the child's minus one, never written to a frame).


def readings_dir() -> str:
    return os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "braintree")


def work_dir(run_id: str) -> str:
    return os.path.join(readings_dir(), "work", run_id)


# ---------------------------------------------------------------------------
# One pass over the frames: the tree, and the decisions with what came of them
# ---------------------------------------------------------------------------

class _Grow:
    """A numpy array indexed by a growing id, filled with `fill` where nothing was written."""

    def __init__(self, dtype: Any, fill: int, size: int = 1 << 16) -> None:
        self.a = np.full(size, fill, dtype=dtype)
        self.fill = fill
        self.top = 0

    def room(self, largest: int) -> None:
        if largest >= self.a.size:
            size = self.a.size
            while largest >= size:
                size *= 2
            more = np.full(size, self.fill, dtype=self.a.dtype)
            more[:self.a.size] = self.a
            self.a = more
        self.top = max(self.top, largest + 1)

    def __setitem__(self, index: np.ndarray, value: Any) -> None:
        if np.size(index):
            self.room(int(np.max(index)))
            self.a[index] = value

    def __getitem__(self, index: np.ndarray) -> np.ndarray:
        index = np.asarray(index)
        out = np.full(index.shape, self.fill, dtype=self.a.dtype)
        inside = (index >= 0) & (index < self.a.size)
        out[inside] = self.a[index[inside]]
        return out

    def trimmed(self) -> np.ndarray:
        return self.a[:self.top]


def _find(sorted_ids: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where each of `query` stands in `sorted_ids`, and whether it is there at all."""
    pos = np.searchsorted(sorted_ids, query)
    clipped = np.minimum(pos, max(0, sorted_ids.size - 1))
    found = (pos < sorted_ids.size) & (sorted_ids[clipped] == query) if sorted_ids.size else np.zeros(query.shape, bool)
    return clipped, found


def _need(sorted_ids: np.ndarray, query: np.ndarray, what: str) -> np.ndarray:
    pos, found = _find(sorted_ids, query)
    if not found.all():
        raise ValueError(f"{what}: {int((~found).sum())} agents not alive where they must be "
                         f"(the first {query[~found][:3].tolist()})")
    return pos


class Pass:
    """
    The state of a world between two frames, kept agent by agent in the order of the frame's ids, and what the
    pass writes as it goes. Feed it every frame in order (`frame`), then `finish`.
    """

    def __init__(self, folder: str, founders: int, total_tokens: int, mixing: bool, eras: Sequence[int]) -> None:
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.mixing = mixing
        self.eras = sorted(int(a) for a in eras if a > 0)
        n = int(founders)
        # The founders are agents 0..n-1, with genotypes 1..n (GraphOfLife.__init__ numbers both in order).
        self.ids = np.arange(n, dtype=np.int64)
        self.inst = np.arange(n, dtype=np.int64)
        self.geno = np.arange(1, n + 1, dtype=np.int64)
        self.gen = np.zeros(n, np.int32)
        self.bgen = np.zeros(n, np.int32)
        self.cgen = np.zeros(n, np.int32)
        self.chg = np.zeros(n, np.int32)
        self.born = np.full(n, -1, np.int32)
        self.tokens = np.full(n, total_tokens / max(1, n))
        self.next = n
        self.files = {name: open(os.path.join(folder, f"{name}.bin"), "wb") for name in ("parent", "start", "alive")}
        self.files["parent"].write(np.full(n, -1, np.int32).tobytes())
        self.files["start"].write(np.full(n, -1, np.int16).tobytes())
        self.files["alive"].write(self.inst.astype(np.int32).tobytes())
        self.first = [0, n]          # instances made at phase f are first[f + 1] .. first[f + 2] - 1
        self.alive_off = [0, n]      # the instances alive in frame f are alive[alive_off[f + 1]:alive_off[f + 2]]
        self.gparent = _Grow(np.int64, -2)
        self.gkind = _Grow(np.int8, UNSEEN)
        self.gparent[np.arange(1, n + 1)] = -1
        self.gkind[np.arange(1, n + 1)] = FOUNDER
        self.mates: List[np.ndarray] = []
        self.rows: Dict[str, List[Any]] = {"t": [], "census": [], "births": [], "births_lived": [], "conquests": [],
                                           "conquests_lived": [], "kept": [], "one": [], "hidden": [], "odd": [],
                                           "conquest_same": [], "conquest_changed": [], "mates": [],
                                           "changes": [], "starved": []}
        self.sums = {k: [] for k in TRAITS}
        self.snapshots: Dict[str, np.ndarray] = {}
        self._snapshot(-1, self.ids, None)
        self.life_sum = 0.0
        self.life_count = 0
        self.life_hist = np.zeros(16, np.int64)     # instances by lifetime: 0, 1, 2-3, 4-7, ... iterations
        self.change_hist = np.zeros(16, np.int64)   # instances by changes in place: 0, 1, 2-3, ...
        self.anomalies: Dict[str, int] = {}
        self._census()

    # -- bookkeeping ------------------------------------------------------------

    def _note(self, what: str, count: int) -> None:
        if count:
            self.anomalies[what] = self.anomalies.get(what, 0) + int(count)

    def _ended(self, born: np.ndarray, changes: np.ndarray, frame: int) -> None:
        """Instances that end with this frame's phase, by when they began and how often they changed in place."""
        if not np.size(born):
            return
        life = np.maximum(0, frame - born) // 2
        self.life_sum += float(life.sum())
        self.life_count += int(life.size)
        np.add.at(self.life_hist, np.minimum(15, np.floor(np.log2(life + 1)).astype(int)), 1)
        np.add.at(self.change_hist, np.minimum(15, np.floor(np.log2(changes + 1)).astype(int)), 1)

    def _new(self, parents: np.ndarray, frame: int) -> np.ndarray:
        k = parents.size
        made = self.next + np.arange(k, dtype=np.int64)
        self.next += k
        self.files["parent"].write(parents.astype(np.int32).tobytes())
        self.files["start"].write(np.full(k, frame, np.int16).tobytes())
        return made

    def _register(self, genos: np.ndarray, parents: np.ndarray, kind: int, what: str) -> None:
        """New genotypes, each with the one it came from; one seen before is counted as odd and left as it was."""
        if not genos.size:
            return
        genos, first = np.unique(genos, return_index=True)
        parents = parents[first]
        fresh = self.gkind[genos] == UNSEEN
        self._note(f"{what}: a new genotype seen before", int((~fresh).sum()))
        self.gparent[genos[fresh]] = parents[fresh]
        self.gkind[genos[fresh]] = kind

    def _census(self) -> None:
        """The agents alive at the start of an iteration: the units test 4 follows through it."""
        n = self.ids.size
        self.unit = np.arange(n, dtype=np.int64)          # each agent's unit: itself, or the unit it descends from
        self.unit_inst = self.inst.copy()
        self.unit_x = np.log(np.maximum(self.tokens, 1.0))
        self.z = {k: np.full(n, np.nan) for k in TRAITS}

    def _snapshot(self, frame: int, ids: np.ndarray, parent_ids: Optional[np.ndarray]) -> None:
        a = (frame + 1) // 2
        if frame != -1 and a not in self.eras:
            return
        for key, value in (("ids", ids), ("inst", self.inst), ("geno", self.geno), ("gen", self.gen),
                           ("bgen", self.bgen), ("cgen", self.cgen), ("tokens", self.tokens)):
            self.snapshots[f"{a}:{key}"] = np.asarray(value).copy()
        if parent_ids is not None:
            self.snapshots[f"{a}:parent_ids"] = np.asarray(parent_ids, np.int64)

    def _alive(self, frame: int) -> None:
        self.files["alive"].write(self.inst.astype(np.int32).tobytes())
        self.alive_off.append(self.alive_off[-1] + self.inst.size)
        self.first.append(self.next)
        if len(self.alive_off) != frame + 3:
            raise ValueError(f"frame {frame} out of order")

    # -- the two phases --------------------------------------------------------

    def frame(self, f: int, frame: Dict[str, Any]) -> None:
        if (frame["iteration"], frame["phase"]) != (f // 2, 1 + f % 2):
            raise ValueError(f"frame {f} is iteration {frame['iteration']} phase {frame['phase']}")
        if "decisions" not in frame:
            raise ValueError(f"frame {f} records no decisions: the tree cannot be built")
        if f % 2 == 0:
            self._reproduction(f, frame)
        else:
            self._game(f, frame)

    def _reproduction(self, f: int, frame: Dict[str, Any]) -> None:
        d = frame["decisions"]
        births = d["births"]
        k = len(births)
        parent = np.array([b["agent"] for b in births], np.int64)
        child = np.array([b["child"] for b in births], np.int64)
        if k and np.any(np.diff(child) <= 0):
            raise ValueError(f"frame {f}: children's ids not in order")
        pp = _need(self.ids, parent, f"frame {f} births")
        made = self._new(self.inst[pp], f)
        # The units' decisions at birth: an agent has at most one child in a phase.
        unit = self.unit[pp]
        self.z["child_share"][unit] = (np.array([b["invested"] for b in births], float)
                                       / np.maximum(np.array([b["tokens_before"] for b in births], float), 1.0))
        self.z["links"][unit] = np.array([len(b["links"]) for b in births], float)
        mates = d.get("mate_paid") or []
        if mates:
            m = np.asarray(mates, np.int64).reshape(-1, 3)
            mp, mfound = _find(self.ids, m[:, 0])
            cp, cfound = _find(child, m[:, 1])
            ok = mfound & cfound
            self.mates.append(np.stack([made[cp[ok]], self.inst[mp[ok]]], axis=1))
        ids = np.asarray(frame["ids"], np.int64)
        geno = np.asarray(frame["brain_ids"], np.int64)
        pgeno = np.asarray(frame["parent_brain_ids"], np.int64)
        q, carried = _find(self.ids, ids)
        kq, newborn = _find(child, ids) if k else (np.zeros(ids.size, np.int64), np.zeros(ids.size, bool))
        stranger = ~(carried | newborn)
        self._note("agents from nowhere (resurrected)", int(stranger.sum()))
        # Carried agents keep their instance and genotype through the reproduction phase.
        moved = carried & (geno != self.geno[q])
        self._note("a carried agent's genotype changed in reproduction", int(moved.sum()))
        # Newborns: a copy of the paying parent's genotype, or a change of it (a mix or a mutation), or a mix
        # that then mutated, whose hidden middle genotype is the child's id minus one.
        nb = np.flatnonzero(newborn)
        g, pg = geno[nb], pgeno[nb]
        paying = self.geno[pp[kq[nb]]]
        kept = g == paying
        one = ~kept & (pg == paying)
        hidden = ~kept & ~one & (pg == g - 1) & (self.gkind[pg] == UNSEEN)
        odd = ~(kept | one | hidden)
        self._register(g[one], paying[one], EITHER if self.mixing else MUTATION, f"frame {f} births")
        self._register(pg[hidden], paying[hidden], MIX, f"frame {f} hidden mixes")
        self._register(g[hidden], pg[hidden], MUTATION, f"frame {f} hidden mixes")
        self._register(g[odd], pg[odd], EITHER if self.mixing else MUTATION, f"frame {f} odd births")
        # The next state, in the frame's order.
        alive_before = np.zeros(self.ids.size, bool)
        alive_before[q[carried]] = True
        gone = np.flatnonzero(~alive_before)
        self._ended(self.born[gone], self.chg[gone], f)
        died_at_birth = k - int(newborn.sum())
        self._ended(np.full(died_at_birth, f), np.zeros(died_at_birth, np.int32), f)
        src = np.where(carried, q, 0)
        par = np.where(newborn, pp[kq], 0) if k else np.zeros(ids.size, np.int64)
        new = {}
        for name in ("gen", "bgen", "cgen", "unit"):
            old = getattr(self, name)
            new[name] = np.where(carried, old[src], 0).astype(old.dtype)
        new["gen"][newborn] = self.gen[par[newborn]] + 1
        new["bgen"][newborn] = self.bgen[par[newborn]] + 1
        new["cgen"][newborn] = self.cgen[par[newborn]]
        new["unit"][newborn] = self.unit[par[newborn]]
        new["unit"][stranger] = -1
        inst = np.where(carried, self.inst[src], -1)
        inst[newborn] = made[kq[newborn]]
        if stranger.any():                               # an agent from nowhere is a founder of its own
            inst[stranger] = self._founders(int(stranger.sum()), f)
        chg = np.where(carried, self.chg[src], 0).astype(np.int32)
        born = np.where(carried, self.born[src], f).astype(np.int32)
        self.ids, self.inst, self.geno = ids, inst, geno
        self.gen, self.bgen, self.cgen, self.unit = new["gen"], new["bgen"], new["cgen"], new["unit"]
        self.chg, self.born = chg, born
        self.tokens = np.asarray(frame["tokens"], float)
        self._alive(f)
        self._births = {"births": k, "births_lived": int(newborn.sum()), "kept": int(kept.sum()),
                        "one": int(one.sum()), "hidden": int(hidden.sum()), "odd": int(odd.sum()),
                        "mates": int(len(mates))}

    def _founders(self, count: int, f: int) -> np.ndarray:
        made = self.next + np.arange(count, dtype=np.int64)
        self.next += count
        self.files["parent"].write(np.full(count, -1, np.int32).tobytes())
        self.files["start"].write(np.full(count, f, np.int16).tobytes())
        return made

    def _game(self, f: int, frame: Dict[str, Any]) -> None:
        d = frame["decisions"]
        won = d["winners"]
        node = np.array([w["node"] for w in won], np.int64)
        winner = np.array([w["winner"] for w in won], np.int64)
        taken = node != winner
        pn = _need(self.ids, node[taken], f"frame {f} conquered nodes")
        pw = _need(self.ids, winner[taken], f"frame {f} winners")
        census = int(self.unit_inst.size)
        made = self._new(self.inst[pw], f)
        # Every conquest of a game copies from the brains the game began with; the brain it replaces ends.
        self._ended(self.born[pn], self.chg[pn], f)
        inst, geno = self.inst.copy(), self.geno.copy()
        gen, bgen, cgen, unit = self.gen.copy(), self.bgen.copy(), self.cgen.copy(), self.unit.copy()
        chg, born = self.chg.copy(), self.born.copy()
        inst[pn] = made
        geno[pn] = self.geno[pw]
        gen[pn] = self.gen[pw] + 1
        bgen[pn] = self.bgen[pw]
        cgen[pn] = self.cgen[pw] + 1
        unit[pn] = self.unit[pw]
        chg[pn] = 0
        born[pn] = f
        # The units' stakes: only an agent that is still its census self (a newborn of this iteration is not).
        # A lean read of the frames leaves most games' stakes unread (None): their traits are not seen.
        stakes = d.get("allocations")
        if stakes:
            agent = np.array([a["agent"] for a in stakes], np.int64)
            held = np.array([a["tokens"] for a in stakes], float)
            spread = np.array([a["spread"] for a in stakes], float)
            home = np.array([a["alloc"][0] for a in stakes], float)
            pa = _need(self.ids, agent, f"frame {f} stakes")
            u = self.unit[pa]
            own = (u >= 0) & (self.inst[pa] == self.unit_inst[np.maximum(u, 0)])
            self.z["spread"][u[own]] = spread[own]
            self.z["home"][u[own]] = home[own] / np.maximum(held[own], 1.0)
        ids = np.asarray(frame["ids"], np.int64)
        fg = np.asarray(frame["brain_ids"], np.int64)
        fpg = np.asarray(frame["parent_brain_ids"], np.int64)
        q = _need(self.ids, ids, f"frame {f} survivors")
        expected = geno[q]
        changed = fg != expected
        wrong = changed & (fpg != expected)
        self._note("a genotype changed from one the agent did not hold", int(wrong.sum()))
        self._register(fg[changed], np.where(wrong, fpg, expected)[changed], MUTATION, f"frame {f} mutations")
        chg[q[changed]] += 1
        lived = np.zeros(self.ids.size, bool)
        lived[q] = True
        gone = np.flatnonzero(~lived)            # the brains the cleanup removed: a conquest's copy, if it took one
        self._ended(born[gone], chg[gone], f)
        conquered_alive = np.isin(q, pn)
        # Test 4: each unit's descendants alive now (itself included, if its census instance still lives).
        u_now = unit[q]
        w = np.bincount(u_now[u_now >= 0], minlength=census).astype(float)
        survived = np.zeros(census, float)
        same = (u_now >= 0) & (inst[q] == self.unit_inst[np.maximum(u_now, 0)])
        survived[u_now[same]] = 1.0
        for k in TRAITS:
            z = self.z[k]
            seen = np.flatnonzero(~np.isnan(z))
            zz, ww, ss, xx = z[seen], w[seen], survived[seen], self.unit_x[seen]
            self.sums[k].append([seen.size, zz.sum(), (zz * zz).sum(), ww.sum(), (ww * ww).sum(), (ww * zz).sum(),
                                 ss.sum(), (ss * zz).sum(), xx.sum(), (xx * xx).sum(), (ww * xx).sum(),
                                 (zz * xx).sum(), (ss * xx).sum()])
        b = getattr(self, "_births", {})
        r = self.rows
        r["t"].append(f // 2)
        r["census"].append(census)
        for key in ("births", "births_lived", "kept", "one", "hidden", "odd", "mates"):
            r[key].append(b.get(key, 0))
        r["conquests"].append(int(taken.sum()))
        r["conquests_lived"].append(int(conquered_alive.sum()))
        r["conquest_same"].append(int((conquered_alive & ~changed).sum()))
        r["conquest_changed"].append(int((conquered_alive & changed).sum()))
        r["changes"].append(int(changed.sum()))
        r["starved"].append(int(self.ids.size - len(won)))
        # The next state.
        self.ids, self.inst, self.geno = ids, inst[q], fg
        self.gen, self.bgen, self.cgen, self.unit = gen[q], bgen[q], cgen[q], unit[q]
        self.chg, self.born = chg[q], born[q]
        self.tokens = np.asarray(frame["tokens"], float)
        self._alive(f)
        self._snapshot(f, ids, np.asarray(frame["parent_ids"], np.int64))
        self._census()

    # -- the end ----------------------------------------------------------------

    def finish(self) -> Dict[str, Any]:
        for name, handle in self.files.items():
            handle.close()
        self._ended(self.born, self.chg, 2 * len(self.rows["t"]))
        np.save(os.path.join(self.folder, "first.npy"), np.asarray(self.first, np.int64))
        np.save(os.path.join(self.folder, "alive_off.npy"), np.asarray(self.alive_off, np.int64))
        np.save(os.path.join(self.folder, "gparent.npy"), self.gparent.trimmed())
        np.save(os.path.join(self.folder, "gkind.npy"), self.gkind.trimmed())
        np.save(os.path.join(self.folder, "mates.npy"),
                np.concatenate(self.mates) if self.mates else np.zeros((0, 2), np.int64))
        np.savez(os.path.join(self.folder, "snapshots.npz"), **self.snapshots)
        rows = {k: np.asarray(v) for k, v in self.rows.items()}
        rows.update({f"sums_{k}": np.asarray(v, float).reshape(-1, len(SUMS)) for k, v in self.sums.items()})
        np.savez(os.path.join(self.folder, "rows.npz"), **rows)
        summary = {"instances": int(self.next), "frames": len(self.alive_off) - 2,
                   "genotypes": int(self.gparent.top), "anomalies": self.anomalies,
                   "lifetime_mean": self.life_sum / max(1, self.life_count),
                   "lifetime_hist": self.life_hist.tolist(), "change_hist": self.change_hist.tolist()}
        with open(os.path.join(self.folder, "pass.json"), "w") as fh:
            json.dump(summary, fh)
        return summary


#: The games whose stakes the pass reads: every STAKES-th. A game frame's stakes are half its text, only test 4
#: reads them, and the tree needs none; the births, which test 4 also reads, come with every reproduction frame.
STAKES = 5


def _flat(text: bytes, key: bytes) -> List[int]:
    """A flat list of numbers under a top-level key of a frame's compact JSON, parsed alone."""
    start = text.index(b'"' + key + b'":[') + len(key) + 3
    return json.loads(text[start:text.index(b"]", start) + 1])


def _until(text: bytes, key: bytes, after: bytes) -> Any:
    """The list under `key`, which the key `after` follows, parsed alone."""
    start = text.index(b'"' + key + b'":[') + len(key) + 3
    return json.loads(text[start:text.index(b'],"' + after + b'":', start) + 1])


def lean_frame(run_id: str, f: int, stakes: bool) -> Dict[str, Any]:
    """
    What the pass reads of a frame, cut out of its text and parsed alone: gol_store writes a frame as compact
    JSON with its keys in a fixed order (GraphOfLife._frame), so each list can be found by its key. The links,
    ages, gifts and the like are never parsed, and a game's stakes only where `stakes` asks for them.
    """
    import zlib
    with open(gol_store.frame_path(run_id, f), "rb") as fh:
        text = zlib.decompress(fh.read(), 31)
    head = json.loads(text[:text.index(b',"nodes_before":')] + b"}")
    out = {"iteration": head["iteration"], "phase": head["phase"]}
    for key in ("ids", "tokens", "brain_ids", "parent_brain_ids", "parent_ids"):
        out[key] = _flat(text, key.encode())
    if out["phase"] == 1:
        d = {"births": _until(text, b"births", b"gifts")}
        if b'"mate_paid":[' in text:
            d["mate_paid"] = _until(text, b"mate_paid", b"pruned_edges")
    else:
        d = {"winners": _until(text, b"winners", b"pruned_edges"),
             "allocations": _until(text, b"allocations", b"winners") if stakes else None}
    out["decisions"] = d
    return out


def frames_of(run_id: str, count: int, lean: bool = True) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """A run's frames in order: leanly read (`lean_frame`, the stakes of every STAKES-th game), or whole."""
    for f in range(count):
        yield f, (lean_frame(run_id, f, (f // 2) % STAKES == 0) if lean else gol_store.read_frame(run_id, f))


def config_of(run_id: str) -> SimConfig:
    return SimConfig.from_dict(gol_store.load_meta(run_id)["config"])


def build(run_id: str, until: int, frames: Optional[Iterable[Tuple[int, Dict[str, Any]]]] = None,
          cfg: Optional[SimConfig] = None, eras: Optional[Sequence[int]] = None,
          folder: Optional[str] = None) -> Dict[str, Any]:
    """One pass over a world's 2 * until frames: the brain-copy tree and the rows of test 4, written to `folder`."""
    cfg = cfg or config_of(run_id)
    folder = folder or work_dir(run_id)
    eras = eras if eras is not None else [a for a in gol_store.archived(run_id) if 0 < a <= until]
    p = Pass(folder, cfg.resolved_n(), cfg.total_tokens, bool(cfg.mix_at_birth), eras)
    started = time.time()
    for f, frame in (frames if frames is not None else frames_of(run_id, 2 * until)):
        p.frame(f, frame)
    out = p.finish()
    out["seconds"] = time.time() - started
    with open(os.path.join(folder, "done.json"), "w") as fh:
        json.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# The tree, as the pass wrote it
# ---------------------------------------------------------------------------

class Built:
    """A world's brain-copy tree, read from its work folder (the large arrays mapped, not loaded)."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.parent = np.memmap(os.path.join(folder, "parent.bin"), np.int32, "r")
        self.start = np.memmap(os.path.join(folder, "start.bin"), np.int16, "r")
        self.alive = np.memmap(os.path.join(folder, "alive.bin"), np.int32, "r")
        self.first = np.load(os.path.join(folder, "first.npy"))
        self.alive_off = np.load(os.path.join(folder, "alive_off.npy"))
        self.size = int(self.parent.size)
        self.last = int(self.alive_off.size) - 3          # the last frame
        snap = np.load(os.path.join(folder, "snapshots.npz"))
        self.snap: Dict[int, Dict[str, np.ndarray]] = {}
        for key in snap.files:
            a, name = key.split(":")
            self.snap.setdefault(int(a), {})[name] = snap[key]
        with open(os.path.join(folder, "pass.json")) as fh:
            self.summary = json.load(fh)

    def alive_at(self, frame: int) -> np.ndarray:
        """The instances alive in a frame (-1: the founders), in the order of the frame's agents."""
        return np.asarray(self.alive[self.alive_off[frame + 1]:self.alive_off[frame + 2]], np.int64)

    def made_at(self, frame: int) -> Tuple[int, int]:
        """The instances a phase made: the first, and one past the last."""
        return int(self.first[frame + 1]), int(self.first[frame + 2])

    def up_to(self, inst: np.ndarray, frame: int) -> np.ndarray:
        """Each instance's ancestor alive in `frame`: the instance itself if it was made by then."""
        x = np.asarray(inst, np.int64).copy()
        late = np.asarray(self.start[x]) > frame
        while late.any():
            x[late] = self.parent[x[late]]
            late = np.asarray(self.start[x]) > frame
        return x

    def path_kinds(self, inst: int, frame: int) -> Tuple[int, int]:
        """The births and conquests on the way from an instance up to its ancestor alive in `frame`."""
        births = conquests = 0
        x = int(inst)
        while int(self.start[x]) > frame:
            if int(self.start[x]) % 2 == 0:
                births += 1
            else:
                conquests += 1
            x = int(self.parent[x])
        return births, conquests


def census_frame(at: int) -> int:
    """The frame of the agents alive at the start of iteration `at`: after the game before it (-1: the founders)."""
    return 2 * at - 1


# ---------------------------------------------------------------------------
# Draws of the shadow world: a parent for every copy, the same however the tree is walked
# ---------------------------------------------------------------------------

_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def draw(seed: int, rep: Any, inst: Any, n: int) -> np.ndarray:
    """
    For each replicate and instance, a whole number from 0 to n - 1, uniform and the same whenever it is asked
    for: the position, among the living, of the instance's parent in that shadow. A hash (splitmix64) of the
    seed, the replicate and the instance, so that the backward and the forward walks meet the same parents.
    """
    with np.errstate(over="ignore"):                     # the hash wraps around 2**64 on purpose
        salt = (np.asarray(rep, np.uint64) + np.uint64(seed % (1 << 32)) * np.uint64(1 << 32)) * _M2
        z = np.asarray(inst, np.uint64) * _G + salt
        z = z ^ (z >> np.uint64(30))
        z = z * _M1
        z = z ^ (z >> np.uint64(27))
        z = z * _M2
        z = z ^ (z >> np.uint64(31))
        return ((z >> np.uint64(32)) * np.uint64(n) >> np.uint64(32)).astype(np.int64)


# ---------------------------------------------------------------------------
# The ancestry of the final population, traced back
# ---------------------------------------------------------------------------

def trace_back(b: Built, reps: Optional[Sequence[int]], seed: int = SEED,
               keep: Sequence[int] = ()) -> Dict[str, Any]:
    """
    The final population's ancestry, frame by frame back to the founders: on the real tree (`reps` None) or
    in each shadow replicate of `reps`. Each lineage carries how many final agents descend through it and the
    copies made along their paths, so that at the common ancestor the mean number of copies from it to the
    final agents is read off. Returned: the number of distinct ancestors alive at the start of every iteration,
    per replicate; the frame of the latest common ancestor (-2: none) and the mean copies from it; the founders'
    lines left and the copies from the founders; and at the frames of `keep` (the real tree), the ancestors and
    how many final agents each has.
    """
    final = b.alive_at(b.last)
    n = final.size
    R = 1 if reps is None else len(reps)
    reps_arr = None if reps is None else np.asarray(reps, np.int64)
    M = np.int64(b.size)
    key = (np.arange(R, dtype=np.int64)[:, None] * M + final[None, :]).ravel()
    count = np.ones(key.size)
    jumps = np.zeros(key.size)
    iterations = (b.last + 1) // 2
    ancestors = np.zeros((R, iterations + 1), np.int64)   # at the start of iteration t, t = 0 .. iterations
    ancestors[:, iterations] = n
    common = np.full(R, -2, np.int64)
    copies = np.full(R, np.nan)
    keep = sorted(set(int(k) for k in keep))
    kept: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if b.last in keep:
        kept[b.last] = (final.copy(), np.ones(n))
    f = b.last
    for f in range(b.last, -1, -1):
        inst = key % M
        moving = np.flatnonzero(np.asarray(b.start[inst]) == f)
        if moving.size:
            sub = inst[moving]
            if reps_arr is None:
                up = np.asarray(b.parent[sub], np.int64)
            else:
                living = b.alive_at(f - 1)
                up = living[draw(seed, reps_arr[key[moving] // M], sub, living.size)]
            key[moving] = (key[moving] // M) * M + up
            jumps[moving] += count[moving]
            key, inverse = np.unique(key, return_inverse=True)
            count = np.bincount(inverse, count)
            jumps = np.bincount(inverse, jumps)
        # The lineages are now the ancestors alive in frame f - 1.
        if (f - 1) in keep:
            kept[f - 1] = ((key % M).copy(), count.copy())
        if (f - 1) % 2 == 1:
            per = np.bincount(key // M, minlength=R)
            ancestors[:, f // 2] = per
            lone = (per == 1) & (common == -2)
            if lone.any():
                where = np.flatnonzero(lone)
                common[where] = f - 1
                copies[where] = jumps[np.searchsorted(key // M, where)] / n
            if (common != -2).all() and (not keep or keep[0] >= f - 1):
                ancestors[:, :f // 2] = 1
                break
    reached = f == 0 and (common == -2).any()
    per = np.bincount(key // M, minlength=R)
    founder_copies = np.bincount(key // M, jumps, minlength=R) / n if reached else np.full(R, np.nan)
    return {"ancestors": ancestors, "common": common, "copies": copies, "kept": kept, "final": n,
            "founder_lines": per if reached else np.ones(R, np.int64), "founder_copies": founder_copies}


def effective_population(n: int, copies: Optional[float], ancestors_left: int) -> Optional[float]:
    """
    An effective population from how fast the ancestry narrows (Kingman's coalescent): n lineages narrow to k
    in about 2 N (1/k - 1/n) generations, so N is about G / (2 (1/k - 1/n)), with G the copies from the
    ancestors to the final agents. With one ancestor left this is about G / 2, the reading funnel.py makes on
    the births-only tree; with k founders' lines still apart, about k G / 2.
    """
    if copies is None or not math.isfinite(copies) or copies <= 0 or ancestors_left >= n:
        return None
    return float(copies / (2.0 * (1.0 / ancestors_left - 1.0 / n)))


# ---------------------------------------------------------------------------
# The clades of each era, counted forward
# ---------------------------------------------------------------------------

def era_clades(b: Built, a: int, h: int, reps: Optional[Sequence[int]], seed: int = SEED,
               batch_elements: int = 40_000_000) -> Dict[str, Any]:
    """
    The clades of the instances alive at the start of iteration a, followed to the start of iteration h: at
    every iteration's start, how many of the living descend from each. On the real tree (`reps` None) or in
    each shadow replicate. Per replicate: the share of the clades still alive at h (lines that persist), the
    largest clade's share of the living at h, the mean number of iterations a clade lives (up to h - a), and
    each clade's evolutionary activity, its size summed over every iteration's start (Bedau and Packard 1992).
    """
    fa, fh = census_frame(a), census_frame(h)
    anchor = b.alive_at(fa)
    na = anchor.size
    base, end = b.made_at(fa + 1)[0], b.made_at(fh)[1]
    width = na + (end - base)
    order = np.argsort(anchor)
    ranked = anchor[order]

    def local(g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, np.int64)
        out = np.empty(g.size, np.int64)
        late = g >= base
        out[late] = na + (g[late] - base)
        early = ~late
        if early.any():
            pos = np.minimum(np.searchsorted(ranked, g[early]), na - 1)
            if not np.array_equal(ranked[pos], g[early]):
                raise ValueError(f"a parent outside the clades of iteration {a}")
            out[early] = order[pos]
        return out

    reps_arr = None if reps is None else np.asarray(reps, np.int64)
    R = 1 if reps is None else reps_arr.size
    step = max(1, min(R, batch_elements // max(1, width)))
    persist, top, life = np.zeros(R), np.zeros(R), np.zeros(R)
    activity = np.zeros((R, na))
    for lo in range(0, R, step):
        hi = min(R, lo + step)
        rb = hi - lo
        lab = np.empty((rb, width), np.int32)
        lab[:, :na] = np.arange(na, dtype=np.int32)[None, :]
        before = local(anchor)
        act = np.zeros((rb, na))
        lived = np.zeros((rb, na))
        offsets = (np.arange(rb, dtype=np.int64) * na)[:, None]
        sizes = np.zeros((rb, na))
        for f in range(fa + 1, fh + 1):
            g0, g1 = b.made_at(f)
            if g1 > g0:
                new = slice(na + g0 - base, na + g1 - base)
                if reps_arr is None:
                    up = local(np.asarray(b.parent[g0:g1], np.int64))[None, :]
                else:
                    idx = draw(seed, reps_arr[lo:hi][:, None], np.arange(g0, g1, dtype=np.int64)[None, :],
                               before.size)
                    up = before[idx]
                lab[:, new] = np.take_along_axis(lab, np.broadcast_to(up, (rb, g1 - g0)), axis=1)
            now = local(b.alive_at(f))
            if f % 2 == 1:                                 # the start of an iteration: count the clades
                flat = (lab[:, now].astype(np.int64) + offsets).ravel()
                sizes = np.bincount(flat, minlength=rb * na).reshape(rb, na).astype(float)
                act += sizes
                lived += sizes > 0
            before = now
        total = sizes.sum(axis=1)
        persist[lo:hi] = (sizes > 0).mean(axis=1)
        top[lo:hi] = sizes.max(axis=1) / np.maximum(1.0, total)
        life[lo:hi] = lived.mean(axis=1)
        activity[lo:hi] = act
    return {"persist": persist, "top": top, "life": life, "activity": activity, "clades": na}


# ---------------------------------------------------------------------------
# A world's tree, read
# ---------------------------------------------------------------------------

def _plain(x: Any) -> Any:
    """Numbers numpy made, as JSON writes them; a number that is not finite as None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def rows_of(folder: str) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(folder, "rows.npz")) as z:
        return {k: z[k] for k in z.files}


def copies(rows: Dict[str, np.ndarray], eras: Sequence[int], p: float, mixing: bool) -> Dict[str, Any]:
    """
    How brains were copied: the share of copies made by birth and by conquest, over the run and era by era;
    births and conquests per agent alive at the start of an iteration; and what a copy did to the genotype.
    A world that mixes at birth gives a child a genotype of its own when the mix differs from the parent's
    brain (a share q of births) and again when it mutates (p, the world's mutation_probability): a birth that
    shows a hidden middle genotype both mixed and mutated (q p of them), so q is read from those, and of the
    births with one change, the share r = q (1 - p) / (one change) were mixes.
    """
    t = rows["t"]
    b, c = float(rows["births"].sum()), float(rows["conquests"].sum())
    census = np.maximum(1, rows["census"]).astype(float)
    per_era = []
    bounds = [0] + [int(e) for e in eras]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (t >= lo) & (t < hi)
        bb, cc = float(rows["births"][m].sum()), float(rows["conquests"][m].sum())
        per_era.append({"from": lo, "to": hi, "births": bb, "conquests": cc,
                        "conquest_share": cc / (bb + cc) if bb + cc else None})
    lived = float(rows["kept"].sum() + rows["one"].sum() + rows["hidden"].sum() + rows["odd"].sum())
    q = r = None
    if mixing and lived and p > 0:
        q = min(1.0, float(rows["hidden"].sum()) / (lived * p))
        one = float(rows["one"].sum()) / lived
        r = min(1.0, q * (1 - p) / one) if one > 0 else 0.0
    games = rows["census"] > 0
    # The games the independent review read (frames 3999 to 4001 in a world of 3,000 iterations).
    spot = np.flatnonzero(np.isin(t, (1999, 2000, 2001)) & games)
    return {"births": b, "conquests": c, "birth_share": b / (b + c) if b + c else None,
            "conquests_per_agent_2000": float(np.mean(rows["conquests"][spot] / census[spot])) if spot.size else None,
            "conquest_share": c / (b + c) if b + c else None,
            "births_lived": float(rows["births_lived"].sum()), "conquests_lived": float(rows["conquests_lived"].sum()),
            "births_per_agent": float(np.mean(rows["births"][games] / census[games])),
            "conquests_per_agent": float(np.mean(rows["conquests"][games] / census[games])),
            "conquests_per_birth": c / b if b else None,
            "per_era": per_era,
            "birth_genotypes": {"kept": float(rows["kept"].sum()), "one_change": float(rows["one"].sum()),
                                "mixed_and_mutated": float(rows["hidden"].sum()), "odd": float(rows["odd"].sum())},
            "mixing_rate": q, "mix_share_of_one_change": r,
            "conquest_genotypes": {"kept": float(rows["conquest_same"].sum()),
                                   "changed": float(rows["conquest_changed"].sum())},
            "mates_recorded": float(rows["mates"].sum())}


def genotype_events(gparent: np.ndarray, gkind: np.ndarray, g_end: int, g_start: int,
                    limit: int = 1_000_000) -> Optional[str]:
    """
    The changes from genotype g_start to g_end, oldest first, one letter each: m a mutation, x a mix, e one
    change at a birth that was a mix or a mutation (a world that mixes). None where the forest does not lead
    from one to the other.
    """
    letters = []
    g = int(g_end)
    while g != int(g_start):
        if not 0 <= g < gkind.size or gkind[g] in (UNSEEN, FOUNDER) or len(letters) > limit:
            return None
        letters.append({MUTATION: "m", EITHER: "e", MIX: "x"}[int(gkind[g])])
        g = int(gparent[g])
    return "".join(reversed(letters))


def _where(snap: Dict[str, np.ndarray], inst: int) -> int:
    hit = np.flatnonzero(snap["inst"] == inst)
    if not hit.size:
        raise ValueError(f"instance {inst} not alive where the line says")
    return int(hit[0])


def line_of_descent(b: Built, real: Dict[str, Any], eras: Sequence[int]) -> List[Dict[str, Any]]:
    """
    In each kept era, the ancestor instance of the most final agents (ties, as at the end where every final
    agent is its own, go to one descending from the era before's), with its node, genotype and generations;
    and from the era before, the copies on its own path and the genotype changes along it.
    """
    gparent, gkind = (np.load(os.path.join(b.folder, f"{n}.npy")) for n in ("gparent", "gkind"))
    out: List[Dict[str, Any]] = []
    prev: Optional[Tuple[int, int]] = None
    for e in sorted(eras):
        inst, cnt = real["kept"][census_frame(e)]
        best = cnt.max()
        cands = np.sort(inst[cnt == best])
        if prev is not None and cands.size > 1:
            on = cands[b.up_to(cands, census_frame(prev[1])) == prev[0]]
            cands = on if on.size else cands
        lead = int(cands[0])
        s = b.snap[e]
        i = _where(s, lead)
        row = {"era": int(e), "instance": lead, "node": int(s["ids"][i]), "genotype": int(s["geno"][i]),
               "share": float(best / real["final"]), "generation": int(s["gen"][i]),
               "births": int(s["bgen"][i]), "conquests": int(s["cgen"][i])}
        if prev is not None:
            fp = census_frame(prev[1])
            anc = int(b.up_to(np.array([lead]), fp)[0])
            sp = b.snap[prev[1]]
            row["nested"] = anc == prev[0]
            row["path_births"], row["path_conquests"] = b.path_kinds(lead, fp)
            row["events"] = genotype_events(gparent, gkind, row["genotype"], int(sp["geno"][_where(sp, anc)]))
        out.append(row)
        prev = (lead, e)
    return out


def pairs_at(b: Built, a: int, seed: int = SEED, count: int = PAIRS) -> Dict[str, Any]:
    """
    Parent and child instances alive together in the world kept at `a`, by how the child was made, and `count`
    of them drawn at random, as nodes, for heredity (research/screen/brainplay.py). Beside them, the pairs the
    births-only tree played there (a child and the agent that bore it, both alive), and the share of those that
    are still parent and child on the brain tree: the child still holds its brain from birth, and the parent
    still holds the brain that bore it.
    """
    s = b.snap[a]
    inst, ids = s["inst"], s["ids"]
    par = np.asarray(b.parent[inst], np.int64)
    order = np.argsort(inst)
    pos = np.minimum(np.searchsorted(inst[order], par), inst.size - 1)
    has = (par >= 0) & (inst[order][pos] == par)
    child = np.flatnonzero(has)
    parent = order[pos[has]]
    born = np.asarray(b.start[inst[child]]) % 2 == 0
    rng = np.random.RandomState(seed + a)
    pick = np.sort(rng.choice(child.size, size=min(count, child.size), replace=False)) if child.size else []
    drawn = [[int(ids[child[j]]), int(ids[parent[j]]), "birth" if born[j] else "conquest"] for j in pick]
    pid = s["parent_ids"]
    idpos = np.minimum(np.searchsorted(ids, pid), ids.size - 1)
    node_pair = (pid >= 0) & (ids[idpos] == pid)
    birth_brain = np.asarray(b.start[inst]) % 2 == 0
    true = node_pair & birth_brain & (par == inst[idpos])
    return {"agents": int(ids.size), "pairs": int(child.size), "birth_pairs": int(born.sum()),
            "conquest_pairs": int((~born).sum()), "drawn": drawn,
            "births_only_pairs": int(node_pair.sum()),
            "births_only_still_brain_pairs": float(true.sum() / node_pair.sum()) if node_pair.sum() else None}


def tree_readings(b: Built, cfg: SimConfig, until: int) -> Dict[str, Any]:
    """Everything of the real tree but the shadow: copies, generations, ancestry, the line, heredity pairs."""
    eras = sorted(e for e in b.snap if 0 < e <= until)
    rows = rows_of(b.folder)
    real = trace_back(b, None, keep=[census_frame(e) for e in eras])
    n = real["final"]
    common = int(real["common"][0])
    generations = {}
    for e in sorted(b.snap):
        s = b.snap[e]
        if not s["gen"].size:
            continue
        generations[str(e)] = {"agents": int(s["gen"].size), "mean": float(s["gen"].mean()),
                               "min": int(s["gen"].min()), "max": int(s["gen"].max()),
                               "births": float(s["bgen"].mean()), "conquests": float(s["cgen"].mean())}
    iterations = (b.last + 1) // 2
    anc = real["ancestors"][0]
    back = {str(lag): int(anc[iterations - lag]) for lag in LAGS if iterations - lag >= 0}
    if common != -2:
        ne = effective_population(n, float(real["copies"][0]), 1)
        age = (b.last - common) / 2.0
    else:
        ne = effective_population(n, float(real["founder_copies"][0]), int(real["founder_lines"][0]))
        age = None
    return {"instances": b.size, "agents_end": n, "pass": b.summary,
            "copies": copies(rows, eras, float(cfg.mutation_probability), bool(cfg.mix_at_birth)),
            "generations": generations, "ancestors": back,
            "ancestors_curve": anc[::10].tolist(),
            "common_ancestor_at": (common + 1) // 2 if common != -2 else None,
            "common_ancestor_age": age,
            "common_ancestor_copies": float(real["copies"][0]) if common != -2 else None,
            "founder_lines": int(real["founder_lines"][0]) if common == -2 else 1,
            "effective_population": ne,
            "line": line_of_descent(b, real, eras),
            "pairs": {str(a): pairs_at(b, a) for a in (until // 2, until) if a in b.snap}}


# ---------------------------------------------------------------------------
# The shadow world of pure luck, against the real tree
# ---------------------------------------------------------------------------

def against(real: float, shadow: np.ndarray, more: bool) -> Dict[str, Any]:
    """
    A real number against its shadow distribution: the shadow's mean and spread, the z-score, and the empirical
    chance of a shadow as far or further in the direction that says "not luck" (more or less than chance).
    """
    shadow = np.asarray(shadow, float)
    shadow = shadow[np.isfinite(shadow)]
    if not shadow.size or real is None or not math.isfinite(real):
        return {"real": real, "shadow": None, "sd": None, "z": None, "p": None}
    mean, sd = float(shadow.mean()), float(shadow.std(ddof=1)) if shadow.size > 1 else 0.0
    sign = 1.0 if more else -1.0
    z = sign * (real - mean) / sd if sd > 0 else None
    beyond = (shadow >= real) if more else (shadow <= real)
    return {"real": float(real), "shadow": mean, "sd": sd, "z": z,
            "p": float((1 + beyond.sum()) / (shadow.size + 1))}


def shadow_readings(b: Built, until: int, reps: int = SHADOWS, seed: int = SEED) -> Dict[str, Any]:
    """
    The real tree against `reps` shadows of pure luck: the same copies at the same moments, and the same living,
    but every copy's parent drawn uniformly among those alive just before it (`draw`). Read in both: how fast the
    final population's ancestry narrows (its common ancestor, its ancestors back in time, its effective
    population); and in every era, how the clades of its first living fare by its end (how many persist, the
    largest's share, how long they live) and their evolutionary activity: a clade is significantly active where
    its summed size beats 99% of the shadow's clades of that era (Bedau, Snyder and Packard 1998), and the share
    of all activity those clades hold is compared with the shadow's.
    """
    eras = sorted(e for e in b.snap if 0 < e <= until)
    started = time.time()
    real = trace_back(b, None)
    sh = trace_back(b, np.arange(reps), seed)
    n = real["final"]
    iterations = (b.last + 1) // 2

    def age(common: int) -> float:
        return (b.last - common) / 2.0 if common != -2 else math.inf

    real_age = age(int(real["common"][0]))
    ages = np.array([age(int(c)) for c in sh["common"]])
    coalescence = {"real": real_age if math.isfinite(real_age) else None,
                   "shadow_coalesced": float(np.isfinite(ages).mean()),
                   "shadow_median": float(np.median(ages)) if np.isfinite(np.median(ages)) else None,
                   "p": float((1 + (ages <= real_age).sum()) / (reps + 1)) if math.isfinite(real_age) else None}
    if np.isfinite(ages).all() and math.isfinite(real_age):
        coalescence.update({k: v for k, v in against(real_age, ages, more=False).items() if k in ("shadow", "sd", "z")})
    lags = {}
    for lag in LAGS:
        t = iterations - lag
        if t < 0:
            continue
        lags[str(lag)] = against(math.log(real["ancestors"][0][t]), np.log(sh["ancestors"][:, t]), more=False)
        lags[str(lag)]["real_count"] = int(real["ancestors"][0][t])
        lags[str(lag)]["shadow_count"] = float(np.median(sh["ancestors"][:, t]))

    def ne(trace: Dict[str, Any], r: int) -> Optional[float]:
        if int(trace["common"][r]) != -2:
            return effective_population(n, float(trace["copies"][r]), 1)
        return effective_population(n, float(trace["founder_copies"][r]), int(trace["founder_lines"][r]))

    real_ne = ne(real, 0)
    shadow_ne = np.array([ne(sh, r) or np.nan for r in range(reps)], float)
    clades = []
    bounds = [0] + eras
    for a, h in zip(bounds[:-1], bounds[1:]):
        rc = era_clades(b, a, h, None)
        sc = era_clades(b, a, h, np.arange(reps), seed)
        line = np.quantile(sc["activity"].ravel(), 0.99)
        ra = rc["activity"][0]
        total = ra.sum()
        shares = (np.where(sc["activity"] > line, sc["activity"], 0).sum(axis=1)
                  / np.maximum(1e-300, sc["activity"].sum(axis=1)))
        clades.append({
            "from": a, "to": h, "clades": rc["clades"],
            "persist": against(float(rc["persist"][0]), sc["persist"], more=False),
            "top": against(float(rc["top"][0]), sc["top"], more=True),
            "life": against(float(rc["life"][0]), sc["life"], more=False),
            "active_share": against(float(ra[ra > line].sum() / total) if total else math.nan, shares, more=True),
            "active_count": against(float((ra > line).sum()), (sc["activity"] > line).sum(axis=1), more=True),
            "top_activity": against(float(ra.max() / total) if total else math.nan,
                                    sc["activity"].max(axis=1) / np.maximum(1e-300, sc["activity"].sum(axis=1)),
                                    more=True),
        })

    def mean_z(key: str) -> Optional[float]:
        zs = [c[key]["z"] for c in clades if c[key]["z"] is not None]
        return float(np.mean(zs)) if zs else None

    return {"replicates": reps, "seed": seed, "coalescence": coalescence, "ancestors": lags,
            "effective_population": {"real": real_ne, "shadow": float(np.nanmedian(shadow_ne))
                                     if np.isfinite(shadow_ne).any() else None},
            "eras": clades,
            "z": {k: mean_z(k) for k in ("persist", "top", "life", "active_share", "active_count", "top_activity")},
            "seconds": time.time() - started}


# ---------------------------------------------------------------------------
# A world's readings, kept item by item as they are made
# ---------------------------------------------------------------------------

#: What a world's readings hold once complete.
ITEMS = ("tree", "shadow", "selection", "heredity", "line")


def _path(run_id: str) -> str:
    return os.path.join(readings_dir(), f"{run_id}.json")


def rows_path(run_id: str) -> str:
    return os.path.join(readings_dir(), f"{run_id}.rows.npz")


def kept(run_id: str) -> Dict[str, Any]:
    return json.load(open(_path(run_id))) if os.path.exists(_path(run_id)) else {}


def keep(run_id: str, key: str, value: Any) -> None:
    os.makedirs(readings_dir(), exist_ok=True)
    have = kept(run_id)
    have[key] = _plain(value)
    with open(_path(run_id) + ".tmp", "w") as fh:
        json.dump(have, fh)
    os.replace(_path(run_id) + ".tmp", _path(run_id))


def world(run_id: str, until: int, cfg: Optional[SimConfig] = None,
          frames: Optional[Iterable[Tuple[int, Dict[str, Any]]]] = None, shadows: int = SHADOWS,
          brains: bool = True, clean: bool = True, say: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    Whatever of one world's readings is not yet kept: its tree read (one pass over its frames, kept in a work
    folder until every reading is made), its shadow, its selection, and its brains played.
    """
    import selection as SEL
    have = kept(run_id)
    folder = work_dir(run_id)
    if any(k not in have for k in ("tree", "shadow", "selection")):
        if not os.path.exists(os.path.join(folder, "done.json")):
            shutil.rmtree(folder, ignore_errors=True)
            started = time.time()
            build(run_id, until, frames=frames, cfg=cfg, folder=folder)
            say(f"  {run_id}: tree built in {time.time() - started:.0f} s")
        b = Built(folder)
        cfg = cfg or config_of(run_id)
        eras = sorted(e for e in b.snap if 0 < e <= until)
        if "tree" not in have:
            started = time.time()
            keep(run_id, "tree", tree_readings(b, cfg, until))
            say(f"  {run_id}: tree read in {time.time() - started:.0f} s")
        if "shadow" not in have:
            keep(run_id, "shadow", shadow_readings(b, until, shadows))
            say(f"  {run_id}: shadow read in {kept(run_id)['shadow']['seconds']:.0f} s")
        rows = rows_of(folder)
        os.makedirs(readings_dir(), exist_ok=True)
        np.savez_compressed(rows_path(run_id), **rows)
        if "selection" not in have:
            keep(run_id, "selection", SEL.summary(rows, eras))
        del b
    have = kept(run_id)
    if brains:
        import brainplay as BP
        tree = have["tree"]
        started = time.time()
        if "heredity" not in have:
            keep(run_id, "heredity", {at: BP.heredity(run_id, int(at), pairs) for at, pairs in tree["pairs"].items()})
        if "line" not in have:
            keep(run_id, "line", BP.line_and_null(run_id, tree["line"],
                                                  tree["copies"].get("mix_share_of_one_change"), until))
        say(f"  {run_id}: brains played in {time.time() - started:.0f} s")
    have = kept(run_id)
    if clean and all(k in have for k in ITEMS):
        shutil.rmtree(folder, ignore_errors=True)
    return have


def living_worlds() -> List[Tuple[str, int, float]]:
    """Round 3's worlds that lived to their end (research/screen/health.json), the smallest first."""
    with open(os.path.join(HERE, "health.json")) as fh:
        worlds = json.load(fh)["worlds"]
    living = [(w["id"], int(w["until"]), float((w.get("averages") or {}).get("agents") or 0))
              for w in worlds if w["fate"] == "lived"]
    return sorted(living, key=lambda w: (w[2], w[0]))


#: The independent review's spot check (2026-11-11): conquests per agent per game, from a few frames.
SPOT = {"B1-30000-b82ad0-s3042": 0.75, "B1-30000-d6955f-s3026": 0.89, "B1-30000-625bc7-s3080": 0.66}


def conquests_per_agent(run_id: str, games: Sequence[int]) -> Dict[str, Any]:
    """In each game named, the nodes won by another agent, per agent alive at the start of its iteration."""
    out = {}
    for t in games:
        before = gol_store.read_frame(run_id, census_frame(t))
        won = gol_store.read_frame(run_id, 2 * t + 1)["decisions"]["winners"]
        out[str(t)] = sum(1 for w in won if w["node"] != w["winner"]) / max(1, len(before["ids"]))
    return {"games": out, "mean": float(np.mean(list(out.values())))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", help="run ids (default: every living world of round 3, the smallest first)")
    ap.add_argument("--spot-check", action="store_true", help="conquests per agent per game, against the review's")
    ap.add_argument("--shadows", type=int, default=SHADOWS)
    ap.add_argument("--no-brains", action="store_true", help="leave the brains unplayed")
    ap.add_argument("--keep-work", action="store_true", help="keep the tree's arrays once the readings are made")
    a = ap.parse_args()
    if a.spot_check:
        for run_id, review in SPOT.items():
            got = conquests_per_agent(run_id, (1999, 2000, 2001))
            print(f"{run_id}: {got['mean']:.2f} conquests per agent per game (the review: {review:.2f}) {got['games']}")
        return
    todo = [(r, 3000) for r in a.runs] if a.runs else [(r, u) for r, u, _ in living_worlds()]
    started = time.time()
    failed = []
    for i, (run_id, until) in enumerate(todo, 1):
        t0 = time.time()
        try:
            world(run_id, until, shadows=a.shadows, brains=not a.no_brains, clean=not a.keep_work,
                  say=lambda s: print(s, flush=True))
            print(f"[{i}/{len(todo)}] {run_id} done in {time.time() - t0:.0f} s "
                  f"({(time.time() - started) / 3600:.2f} h in all)", flush=True)
        except Exception as e:  # noqa: BLE001 - the others go on; listed at the end
            failed.append((run_id, repr(e)))
            print(f"[{i}/{len(todo)}] {run_id} FAILED: {e!r}", flush=True)
        # The rank as it stands, from every world read so far (research/screen/evolution.py, read afresh each
        # time, so that it can be mended while the pass goes on).
        try:
            import importlib
            import evolution
            importlib.reload(evolution).write()
        except Exception as e:  # noqa: BLE001 - the rank is remade at the end in any case
            print(f"  the rank was not written: {e!r}", flush=True)
    if failed:
        print(f"{len(failed)} failed: " + "; ".join(f"{r} {e}" for r, e in failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
