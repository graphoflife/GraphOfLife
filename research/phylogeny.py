#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct the lineage forest of a run, and measure what it does over time.

Why this exists
---------------
The engine has no lineage identity. `brain_id` is reassigned on every
successful mutation, and every agent mutates every iteration, so a population
of five hundred agents carries five hundred distinct brain ids — one each. The
two statistics that look like lineage measures, `distinctBrains` and
`distinctLineages`, are therefore close to population size and say nothing
about ancestry. Anything built on them would be an artefact.

The information is nonetheless recorded. Every frame carries `brain_ids` and
`parent_brain_ids`, and those pairs, accumulated across a run, are a forest:
one node per genotype version ever seen, one edge from each version to the one
it came from. This assembles that forest and asks it the questions that need
asking before any claim about open-endedness can be made.

What "descended from" means here
--------------------------------
An agent's clade is named by **which agent alive at some earlier iteration it
descends from**, not by a count of generations. That choice is deliberate.
Making a child copies the parent's brain and then mutates the copy, so two new
ids appear where only the second is ever recorded: depth measured in observed
steps is a contraction of the true number of replications, and the contraction
factor is not constant. Anchoring on a *time* instead is robust to that,
because it only asks which recorded ancestor was alive when, and both of those
facts are recorded exactly.

The consequence to keep in mind: this reconstructs the genealogy of surviving
genotypes, not every replication event that ever happened. For lineage
dynamics that is the right object. For counting mutations it is not.

Requires `export_every = 1`. With a coarser export the parent of a recorded
brain may itself never have been recorded, and the forest silently gains roots
that are not founders — `--check` reports how often that happened so the damage
is visible rather than assumed.

Usage
-----
    python3 research/phylogeny.py --simulate --iterations 60
    python3 research/phylogeny.py --run GOL_26_09_04_n001
    python3 research/phylogeny.py --simulate --window 10   # sliding anchor
    python3 research/phylogeny.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NO_PARENT = -1


# ---------------------------------------------------------------------------
# The forest
# ---------------------------------------------------------------------------

class Forest:
    """
    Every genotype version ever recorded, and where each came from.

    Built by feeding it frames in order. A brain is a node; the edge to its
    parent is whatever `parent_brain_ids` said the first time that brain was
    seen. A brain's parent never changes once observed — the engine writes it
    at the moment of copying or mutating and never rewrites it — so the first
    sighting is authoritative and later ones are ignored.
    """

    __slots__ = ("parent", "first_seen", "unrooted")

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.first_seen: Dict[int, int] = {}
        # Brains whose parent was named but never itself recorded. With
        # export_every = 1 these are the founders and the hidden intermediates
        # of the very first iteration; with a coarser export they are damage.
        self.unrooted: Set[int] = set()

    def observe(self, iteration: int, brain_ids: Iterable[int],
                parent_brain_ids: Iterable[int]) -> None:
        for brain, parent in zip(brain_ids, parent_brain_ids):
            self.link(iteration, brain, parent)

    def link(self, iteration: int, brain: int, parent: int) -> None:
        brain, parent = int(brain), int(parent)
        if brain in self.first_seen:
            return
        self.first_seen[brain] = iteration
        self.parent[brain] = parent
        if parent != NO_PARENT and parent not in self.first_seen:
            self.unrooted.add(parent)

    def ancestor_at(self, brain: int, when: int,
                    memo: Optional[Dict[int, int]] = None) -> int:
        """
        The recorded ancestor of `brain` that was already alive at `when`.

        Walks up until it reaches a brain first seen at or before `when`, or
        runs out of forest. `memo` is per-`when`: pass the same dict for every
        brain at one anchor time and each chain is walked once, because the
        chains of a population share almost all of their length.
        """
        if memo is None:
            memo = {}
        path: List[int] = []
        node = brain
        while True:
            if node in memo:
                answer = memo[node]
                break
            seen = self.first_seen.get(node)
            if seen is None or seen <= when:
                answer = node
                break
            up = self.parent.get(node, NO_PARENT)
            if up == NO_PARENT:
                answer = node
                break
            path.append(node)
            node = up
        for step in path:
            memo[step] = answer
        memo[node] = answer
        return answer

    def coalescence_time(self, extant: Iterable[int], now: int) -> Optional[int]:
        """
        The latest iteration at which every extant brain had one common
        ancestor, or None if they never did.

        The count of distinct ancestors can only fall as the anchor moves
        earlier, so this is a binary search rather than a walk. A late
        coalescence means a recent sweep: everything alive descends from one
        agent of the recent past. A coalescence stuck near zero means founding
        lineages have coexisted the whole way.
        """
        extant = list(extant)
        if not extant:
            return None

        def one_ancestor(when: int) -> bool:
            memo: Dict[int, int] = {}
            first = self.ancestor_at(extant[0], when, memo)
            return all(self.ancestor_at(b, when, memo) == first for b in extant)

        if one_ancestor(now):
            return now
        if not one_ancestor(0):
            return None

        lo, hi = 0, now                       # true at lo, false at hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if one_ancestor(mid):
                lo = mid
            else:
                hi = mid
        return lo


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _shannon(counts: Iterable[int]) -> float:
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    # abs() keeps a single clade at 0.0 rather than -0.0, which is the same
    # number and reads like a mistake.
    return abs(-sum((c / total) * math.log(c / total) for c in counts))


def walk(frames: Iterable[dict], anchor: int = 0,
         window: Optional[int] = None) -> Iterator[dict]:
    """
    Feed frames in, get one row of lineage statistics per frame out.

    `anchor` fixes the clades at one iteration — 0 names them by founder, which
    answers "do the original lineages persist". `window` instead anchors them a
    fixed distance behind the present, which answers "how fast does ancestry
    turn over", and is the measure that keeps meaning something after the
    founders have all been replaced.
    """
    forest = Forest()
    for frame in frames:
        iteration = int(frame["iteration"])
        brains = [int(b) for b in frame["brain_ids"]]

        # A traced run hands over every link, including the ones no frame can
        # carry. Fed first, so the chain is whole before anything is asked of
        # it; the frame's own pairs then add nothing it does not already know.
        for child, parent in frame.get("lineage_links", ()):
            forest.link(iteration, child, parent)
        forest.observe(iteration, brains, frame["parent_brain_ids"])

        at = max(0, iteration - window) if window is not None else anchor
        memo: Dict[int, int] = {}
        clades: Dict[int, int] = {}
        for brain in brains:
            root = forest.ancestor_at(brain, at, memo)
            clades[root] = clades.get(root, 0) + 1

        population = len(brains)
        top = max(clades.items(), key=lambda kv: kv[1]) if clades else (None, 0)
        yield {
            "iteration": iteration,
            "phase": int(frame.get("phase", 0)),
            "population": population,
            "anchor": at,
            "clades": len(clades),
            "shannon": _shannon(clades.values()),
            "top_clade": top[0],
            "top_share": (top[1] / population) if population else 0.0,
            "coalescence": forest.coalescence_time(brains, iteration),
            "forest_nodes": len(forest.first_seen),
            "unrooted": len(forest.unrooted - set(forest.first_seen)),
        }


def summarise(rows: List[dict]) -> dict:
    """The handful of numbers a run is worth describing by."""
    if not rows:
        return {}
    last = rows[-1]
    turnovers = sum(1 for a, b in zip(rows, rows[1:])
                    if a["top_clade"] != b["top_clade"])
    coal = [r["coalescence"] for r in rows if r["coalescence"] is not None]
    return {
        "frames": len(rows),
        "iterations": last["iteration"],
        "final_population": last["population"],
        "final_clades": last["clades"],
        "final_shannon": last["shannon"],
        "final_top_share": last["top_share"],
        "dominant_turnovers": turnovers,
        "turnover_rate": turnovers / max(1, len(rows) - 1),
        "forest_nodes": last["forest_nodes"],
        "unrooted": last["unrooted"],
        # How far behind the present the population's common ancestor sits at
        # the end: small means lineages coexist, large means a recent sweep.
        "final_coalescence_lag": (last["iteration"] - coal[-1]) if coal else None,
    }


# ---------------------------------------------------------------------------
# Where frames come from
# ---------------------------------------------------------------------------

def frames_from_run(run_id: str) -> Iterator[dict]:
    """Frames of a run recorded by gol_server.py, in order."""
    import gol_store as store

    count = store.count_frames(run_id)
    if not count:
        raise SystemExit(f"run {run_id} has no frames recorded")
    for index in range(count):
        yield store.read_frame(run_id, index)


def frames_from_simulation(iterations: int, tokens: int, seed: int,
                           trace: bool = True, **overrides) -> Iterator[dict]:
    """
    Frames from a run made here and now.

    With `trace`, every link is captured as the engine makes it rather than
    read back off the frames — which matters, because the frames do not have
    them all. Winning a node copies the winner's brain and then, at the end of
    the phase, mutates the copy; only the second of those is ever written down,
    so the copy is a hole in the chain. **Measured: 49% of the brain ids a run
    creates never appear in any frame.** Reconstructing from frames alone
    therefore breaks every chain within a step or two and hands back one clade
    per agent — the very artefact this file exists to avoid.

    Tracing is only possible in process. A run recorded to disk needs the
    engine to write down a lineage of its own; see research/Research.md.
    """
    import random

    import numpy as np

    from gol_config import SimConfig
    from GraphOfLifeSimple import new_world

    random.seed(seed)
    np.random.seed(seed)
    settings = dict(total_tokens=tokens, n_nodes=50, k_neighbors=6,
                    hidden_layers=[12, 10], seed=seed)
    settings.update(overrides)
    world = new_world(SimConfig(**settings))

    links: List[Tuple[int, int]] = []
    if trace:
        made, copied, mutated = world._new_brain, world._copy_brain, world._mutate_brain

        def new_brain():
            brain = made()
            links.append((brain.brain_id, NO_PARENT))
            return brain

        def copy_brain(source):
            clone = copied(source)
            links.append((clone.brain_id, source.brain_id))
            return clone

        def mutate_brain(brain):
            before = brain.brain_id
            mutated(brain)
            if brain.brain_id != before:
                links.append((brain.brain_id, before))

        world._new_brain, world._copy_brain, world._mutate_brain = (
            new_brain, copy_brain, mutate_brain)

        # The founders exist before any of that was in place.
        for brain in world.brains.values():
            links.append((brain.brain_id, brain.parent_brain_id))

    for _ in range(iterations):
        for frame in world.step(record_decisions=False):
            if trace:
                frame = {**frame, "lineage_links": list(links)}
                links.clear()
            yield frame
        if world.is_extinct():
            return


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def sparkline(values: List[float], width: int = 48) -> str:
    """A run's shape in one line of text, for reading in a terminal."""
    if not values:
        return ""
    marks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    lo, hi = min(sampled), max(sampled)
    if hi <= lo:
        return marks[0] * len(sampled)
    return "".join(marks[min(len(marks) - 1,
                             int((v - lo) / (hi - lo) * len(marks)))]
                   for v in sampled)


def write_csv(rows: List[dict], path: str) -> None:
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# A forest with a known answer
# ---------------------------------------------------------------------------

def selftest() -> int:
    """
    Checks against a forest small enough to reason about by hand.

        iteration 0:  1        2          two founders
        iteration 1:  3←1      4←2
        iteration 2:  5←3      6←3        2 died out; everything is 1's
    """
    frames = [
        {"iteration": 0, "phase": 2, "brain_ids": [1, 2], "parent_brain_ids": [-1, -1]},
        {"iteration": 1, "phase": 2, "brain_ids": [3, 4], "parent_brain_ids": [1, 2]},
        {"iteration": 2, "phase": 2, "brain_ids": [5, 6], "parent_brain_ids": [3, 3]},
    ]

    rows = list(walk(frames, anchor=0))
    assert [r["clades"] for r in rows] == [2, 2, 1], [r["clades"] for r in rows]
    assert rows[-1]["top_clade"] == 1, "everything at the end descends from founder 1"
    assert rows[-1]["top_share"] == 1.0

    # Anchored on the present, every distinct brain is its own clade.
    now = list(walk(frames, window=0))
    assert [r["clades"] for r in now] == [2, 2, 2]

    # Coalescence: the founders never share an ancestor, so at iterations 0 and
    # 1 there is none. At 2 both agents descend from brain 3, which was alive
    # at iteration 1.
    assert [r["coalescence"] for r in rows] == [None, None, 1], \
        [r["coalescence"] for r in rows]

    # A parent that was never recorded is a root, and is counted as one.
    broken = list(walk([
        {"iteration": 5, "phase": 2, "brain_ids": [9], "parent_brain_ids": [8]},
    ]))
    assert broken[0]["clades"] == 1
    assert broken[0]["unrooted"] == 1, "an unseen parent should be reported"

    # One agent is its own common ancestor, right now — not at the root. The
    # first version of this test asserted otherwise and was wrong.
    lone = list(walk([
        {"iteration": 0, "phase": 2, "brain_ids": [1], "parent_brain_ids": [-1]},
        {"iteration": 7, "phase": 2, "brain_ids": [2], "parent_brain_ids": [1]},
    ]))
    assert lone[-1]["coalescence"] == 7, lone[-1]["coalescence"]

    # Two lineages that split at a known moment, to exercise the search rather
    # than its end points. Both descend from brain 1, which lived at iteration
    # 0, and from nothing later — so that is where they coalesce.
    split = [{"iteration": 0, "phase": 2, "brain_ids": [1], "parent_brain_ids": [-1]}]
    left, right = 1, 1
    for i in range(1, 30):
        left, right = 100 + i, 200 + i
        split.append({"iteration": i, "phase": 2,
                      "brain_ids": [left, right],
                      "parent_brain_ids": [100 + i - 1 if i > 1 else 1,
                                           200 + i - 1 if i > 1 else 1]})
    assert list(walk(split))[-1]["coalescence"] == 0

    # Long enough that a naive walk per brain would be quadratic, to show the
    # per-anchor memo is doing its job.
    chain = [{"iteration": 0, "phase": 2, "brain_ids": [0], "parent_brain_ids": [-1]}]
    for i in range(1, 4000):
        chain.append({"iteration": i, "phase": 2,
                      "brain_ids": [i], "parent_brain_ids": [i - 1]})
    tail = list(walk(chain, anchor=0))[-1]
    assert tail["clades"] == 1, "one chain is one clade whatever its length"
    assert tail["forest_nodes"] == 4000

    print("phylogeny selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", help="a run id recorded by gol_server.py")
    source.add_argument("--simulate", action="store_true",
                        help="make a run here and now instead")
    source.add_argument("--selftest", action="store_true")

    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--tokens", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--anchor", type=int, default=0,
                        help="name clades by their ancestor alive at this iteration")
    parser.add_argument("--window", type=int, default=None,
                        help="instead, by their ancestor this many iterations back")
    parser.add_argument("--csv", help="write the per-frame rows here")
    args = parser.parse_args()

    if args.selftest or not (args.run or args.simulate):
        return selftest()

    frames = (frames_from_run(args.run) if args.run
              else frames_from_simulation(args.iterations, args.tokens, args.seed))

    rows = list(walk(frames, anchor=args.anchor, window=args.window))
    if not rows:
        print("no frames")
        return 1

    facts = summarise(rows)
    naming = (f"ancestor {args.window} iterations back" if args.window is not None
              else f"ancestor alive at iteration {args.anchor}")

    print(f"clades named by: {naming}")
    print(f"{'iter':>5} {'pop':>7} {'clades':>7} {'shannon':>8} {'top share':>10} "
          f"{'coalescence lag':>16}")
    step = max(1, len(rows) // 20)
    for row in rows[::step]:
        lag = ("-" if row["coalescence"] is None
               else str(row["iteration"] - row["coalescence"]))
        print(f"{row['iteration']:>5} {row['population']:>7,} {row['clades']:>7,} "
              f"{row['shannon']:>8.3f} {row['top_share']:>9.1%} {lag:>16}")

    print()
    print(f"  population   {sparkline([r['population'] for r in rows])}")
    print(f"  clades       {sparkline([float(r['clades']) for r in rows])}")
    print(f"  diversity    {sparkline([r['shannon'] for r in rows])}")
    print()
    for key in ("frames", "iterations", "final_population", "final_clades",
                "final_shannon", "final_top_share", "dominant_turnovers",
                "turnover_rate", "final_coalescence_lag", "forest_nodes",
                "unrooted"):
        value = facts.get(key)
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"  {key:<22} {value}")

    if facts.get("unrooted"):
        share = facts["unrooted"] / max(1, facts["forest_nodes"])
        print(f"\n  {facts['unrooted']:,} brains ({share:.0%}) name a parent that was"
              f"\n  never recorded, so their chains stop there and every one of them"
              f"\n  counts as its own founder. Winning a node copies the winner's"
              f"\n  brain and then mutates the copy at the end of the phase, and only"
              f"\n  the second of those is written down — measured, 49% of the brain"
              f"\n  ids a run creates never appear in any frame."
              f"\n\n  These clade counts are not trustworthy. Use --simulate, which"
              f"\n  traces the links as the engine makes them, until the engine"
              f"\n  records a lineage of its own.")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\n  wrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
