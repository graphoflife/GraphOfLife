"""
How much of a world is cut off at once (the author, 2026-11-08): "Some
simulations have huge parts culled all the time. If everywhere a little bit is
culled, like leaves, it is not bad; but if every few iterations a whole branch,
maybe 40% of the network, it is quite bad."

At the end of every phase the engine removes the agents that starved, and then
culls everything no longer joined to the largest piece of the network (the
run's `orphaned`, "Culled" in the web app's statistics). A run records how many
agents starved and how many were culled, not which, nor in how many pieces, so
both are read from the network as the phase left it, just before its cleanup:
the frame before it, changed as the phase's own record says (the newborns and
their links, the links handed over to them, the links searches made, the links
that lapsed). The rebuilt network is the recorded one: the links it leaves
between the survivors are the next frame's, in every phase checked (mending
adds its own).

Which of the removed starved is read from what they held (`holdings`). In a
game, an agent holds what was staked on it; in reproduction, what it had,
changed by its birth, its gifts, its mates and its upkeep. A removed agent
linked to a survivor starved, since a culled one is by definition not joined to
the survivors. A few agents' holdings the record leaves open: under link_token
a link that lapses gives its token to one of its two ends, drawn at random, and
a searcher that steps pays the agent it steps on, unrecorded. An agent at its
last token there may have been paid one and culled, or not and starved; the
counts of the starved and the culled settle most of them, and the rest bound the
piece between the agents surely culled and every agent not surely starved
(`pieces`).

The pieces culled are then the culled agents grouped by their links; under
mend_on_death, which joins a dead agent's neighbours up again, the culled
agents of one piece of the network cut off by the phase's lapses or handovers.
In every phase whose holdings are all settled the pieces add up to the run's
own count of the culled.

For each phase from iteration 100 on: the largest piece culled, as a share of
the agents the network had just before the cleanup, surely and at most. Kept per
world in GraphOfLifeRuns/.pilots/screen/culls/<run>.json, with how often a piece
of more than each share was culled, and the share a phase culled in all its
pieces.

    python3 research/screen/culls.py E41 E42 E40 [--jobs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_store  # noqa: E402

#: The first iteration read, as everywhere the screen reads a world.
SETTLE = 100
#: The shares of the world a piece is counted above, for how often each was culled.
SIZES = (0.05, 0.15, 0.25, 0.4)

Network = Dict[int, Set[int]]


def readings_dir() -> str:
    return os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "culls")


def before_cleanup(prev: Dict[str, Any], frame: Dict[str, Any]) -> Network:
    """The network as a phase left it just before its cleanup: the frame before, changed as the phase's record says."""
    adj: Network = {u: set() for u in prev["ids"]}

    def link(a: int, b: int) -> None:
        if a != b and a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    def cut(a: int, b: int) -> None:
        adj.get(a, set()).discard(b)
        adj.get(b, set()).discard(a)

    for a, b in prev["edges"]:
        link(a, b)
    record = frame.get("decisions") or {}
    births = record.get("births") or []
    for birth in births:
        adj.setdefault(birth["child"], set())
    for birth in births:
        for v in birth["links"]:
            link(birth["child"], v)
    # Handed over once every child is born, in the order of the births, as the engine does: a link a parent no
    # longer has, because its neighbour handed it to a child of its own, is not handed again.
    for birth in births:
        parent, child = birth["agent"], birth["child"]
        for v in birth["handed_over"]:
            if v in adj.get(parent, ()):
                link(child, v)
                cut(parent, v)
    for u, w, made in record.get("proposals") or []:
        if made:
            link(u, w)
    for a, b in record.get("pruned_edges") or []:
        cut(a, b)
    return adj


def holdings(prev: Dict[str, Any], frame: Dict[str, Any],
             config: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, Optional[int]]]:
    """
    What each agent held just before the phase's cleanup, at least and at most: at most None where tokens it was
    paid may be missing from the record. Only whether an agent held a token matters to the cleanup, so a searcher,
    which never pays its last token for a step, is counted as holding what it had.
    """
    record = frame.get("decisions") or {}
    token = bool(config.get("link_token"))
    if frame["phase"] == 2:
        # A game: everything is staked, and a node holds what was staked on it, nothing if nobody staked on it.
        least: Dict[int, int] = {u: 0 for u in prev["ids"]}
        for stake in record.get("allocations") or []:
            for v, amount in zip(stake["targets"], stake["alloc"]):
                least[v] = least.get(v, 0) + int(amount)
        most: Dict[int, Optional[int]] = dict(least)
        if token:
            # A lapsed link's token went to one of its two ends.
            for a, b in record.get("pruned_edges") or []:
                for x in (a, b):
                    if x in most:
                        most[x] += 1
        return least, most
    held = {u: int(t) for u, t in zip(prev["ids"], prev["tokens"])}
    births = record.get("births") or []
    linked = {u: set() for u in prev["ids"]}
    for a, b in prev["edges"]:
        linked[a].add(b)
        linked[b].add(a)
    for birth in births:
        # The parent gives its child what it invests; under link_token the child pays a token for each link it made.
        held[birth["agent"]] = birth["tokens_before"] - birth["invested"]
        held[birth["child"]] = birth["invested"] - (len(birth["links"]) if token else 0)
        linked[birth["child"]] = set(birth["links"])
        for v in birth["links"]:
            linked.setdefault(v, set()).add(birth["child"])
    for birth in births:
        parent, child = birth["agent"], birth["child"]
        for v in birth["handed_over"]:
            if v in linked.get(parent, ()):
                if token and v in linked[child]:
                    held[child] += 1          # a handed link joining one the child has leaves its token with it
                linked[child].add(v)
                linked[v].add(child)
                linked[parent].discard(v)
                linked[v].discard(parent)
    for giver, taker, amount in record.get("gifts") or []:
        held[giver] -= amount
        held[taker] = held.get(taker, 0) + amount
    for mate, child, amount in record.get("mate_paid") or []:
        held[mate] -= amount
        held[child] = held.get(child, 0) + amount
    for payer, other in record.get("upkeep") or []:
        held[payer] -= 1
        if config.get("upkeep_to") == "end":
            held[other] = held.get(other, 0) + 1
    least, most = dict(held), dict(held)
    search = config.get("link_search") or "off"
    for u, w, made in record.get("proposals") or []:
        if made and token:
            # The link a search made cost a token, after steps that may have spent all but the last.
            least[u] = least[u] - 1 if search == "reach" else min(least[u] - 1, 0)
            most[u] = most[u] - 1
        if search == "jump" and w != u:
            least[w] = least.get(w, 0) + 1    # a jump ends on the agent it proposes to, and pays for the step
    if search in ("jump", "walk"):
        most = {u: None for u in most}        # anyone may have been stepped on and paid for it
    return least, most


def _components(nodes: Iterable[int], adj: Network) -> List[Set[int]]:
    within = set(nodes)
    seen: Set[int] = set()
    out = []
    for start in within:
        if start in seen:
            continue
        seen.add(start)
        stack, piece = [start], set()
        while stack:
            u = stack.pop()
            piece.add(u)
            for v in adj[u]:
                if v in within and v not in seen:
                    seen.add(v)
                    stack.append(v)
        out.append(piece)
    return out


def pieces(adj: Network, alive: Set[int], starved: int, culled: int, least: Dict[int, int],
           most: Dict[int, Optional[int]], mend: bool) -> Tuple[int, int, bool]:
    """
    The most agents culled in one piece, surely and at most, and whether the record contradicts the reading (an
    agent that surely held a token where it must have starved, more starved or culled than the run counts, pieces
    that do not add up to the culled).
    """
    removed = set(adj) - alive
    parts = _components(removed, adj)
    apart = [p for p in parts if all(adj[u].isdisjoint(alive) for u in p)]
    if mend:
        # No death splits the network under mending, so whatever of the removed still hangs on a survivor starved.
        dead = set().union(*(p for p in parts if not all(adj[u].isdisjoint(alive) for u in p)))
    else:
        dead = {u for u in removed if not adj[u].isdisjoint(alive)}
    dead |= {u for u in removed if most.get(u) is not None and most[u] <= 0}
    cut = {u for u in removed if least.get(u, 0) >= 1}
    wrong = bool(cut & dead) or len(dead) > starved or len(cut) > culled
    open_ = removed - dead - cut
    if len(dead) == starved:
        cut, open_ = cut | open_, set()
    elif len(dead) + len(open_) == starved:
        dead, open_ = dead | open_, set()
    if mend:
        # A piece is a part of the network the phase cut off, less those of its agents that starved.
        surely = max((len(p & cut) for p in apart), default=0)
        at_most = max((len(p - dead) for p in apart), default=0)
        total = sum(len(p - dead) for p in apart)
    else:
        surely = max((len(p) for p in _components(cut, adj)), default=0)
        at_most = max((len(p) for p in _components(cut | open_, adj)), default=0)
        total = len(cut | open_)
    return surely, at_most, wrong or (not open_ and total != culled)


def culls(run_id: str, until: int) -> Dict[str, Any]:
    """Every phase's largest piece culled, from iteration SETTLE to the end of the run, and how often each size."""
    config = gol_store.load_meta(run_id)["config"]
    lost: List[Tuple[int, int, float, float]] = []
    every: List[float] = []
    wrong = 0
    prev = gol_store.read_frame(run_id, 2 * SETTLE - 1)
    for k in range(2 * SETTLE, 2 * until):
        frame = gol_store.read_frame(run_id, k)
        adj = before_cleanup(prev, frame)
        cleanup = frame.get("cleanup") or {}
        starved = int(cleanup.get("starved") or 0) + int(cleanup.get("aged") or 0)
        culled = int(cleanup.get("orphaned") or 0)
        n = max(1, len(adj))
        if culled:
            least, most = holdings(prev, frame, config)
            surely, at_most, bad = pieces(adj, set(frame["ids"]), starved, culled, least, most,
                                          bool(config.get("mend_on_death")))
            wrong += bad
        else:
            surely = at_most = 0
        lost.append((int(frame["iteration"]), int(frame["phase"]), surely / n, at_most / n))
        every.append(culled / n)
        prev = frame
    sure = np.array([s for _, _, s, _ in lost])
    most = np.array([m for _, _, _, m in lost])
    iterations = until - SETTLE
    return {"phases": len(lost), "iterations": iterations, "contradicted": wrong,
            # The share of the network a phase culls in all its pieces, leaves and branches alike.
            "cut_off": float(np.mean(every)) if every else 0.0,
            "largest": float(most.max()) if most.size else 0.0,
            "median": float(np.median(most)) if most.size else 0.0,
            # Phases whose largest piece the record leaves open, and how wide open at most.
            "open": int((most != sure).sum()), "widest": float((most - sure).max()) if most.size else 0.0,
            # How many times, every hundred iterations, a piece of more than each share of the world was culled:
            # surely, and at most.
            "per_100": {f"{s:g}": [float((sure > s).sum() * 100.0 / iterations),
                                   float((most > s).sum() * 100.0 / iterations)] for s in SIZES},
            "lost": [[t, p, round(s, 5), round(m, 5)] for t, p, s, m in lost if m > SIZES[0]]}


def _path(run_id: str) -> str:
    return os.path.join(readings_dir(), f"{run_id}.json")


def kept(run_id: str) -> Dict[str, Any]:
    return json.load(open(_path(run_id))) if os.path.exists(_path(run_id)) else {}


def read(item: Tuple[str, int]) -> str:
    run_id, until = item
    if not os.path.exists(_path(run_id)):
        out = culls(run_id, until)
        os.makedirs(readings_dir(), exist_ok=True)
        with open(_path(run_id) + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(_path(run_id) + ".tmp", _path(run_id))
    return run_id


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rounds", nargs="+")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    todo = []
    for name in a.rounds:
        for spec in gol_plan.experiment_runs(name):
            if not os.path.exists(os.path.join(gol_store.run_dir(spec.run_id), "meta.json")):
                continue
            meta = gol_store.load_meta(spec.run_id)
            if meta.get("status") != "extinct" and int(meta.get("iteration") or 0) >= spec.until:
                todo.append((spec.run_id, spec.until))
    print(f"{len(todo)} worlds lived to their end", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(read, item): item for item in todo}
        for future in as_completed(futures):
            try:
                run_id = future.result()
                print("read", run_id, "contradicted", kept(run_id).get("contradicted"), flush=True)
            except Exception as e:  # noqa: BLE001 - the others go on
                print("failed", futures[future][0], repr(e), flush=True)


if __name__ == "__main__":
    main()
