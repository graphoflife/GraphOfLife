"""
Structures in pilot worlds: record a pilot's games the way the lab records its own.

The structure checks of this folder (structures, gliders, circulation, kin flows)
read a lab run's recorded games. Pilot worlds keep only checkpoints. This goes
on from a pilot's checkpoint for a number of iterations, records each game in
the lab's form (who staked what on whom, who took each node, brains before and
after) and runs the checks on those records.

    python3 research/oee/checks/pilot_frames.py PILOTDIR --at 500 --games 200 [--out out.json]
"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))

import numpy as np

import book_data as D

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(pilot, at):
    from proto import ProtoConfig, ProtoWorld, restore_tags, set_proto
    meta = json.load(open(os.path.join(pilot, "config.json")))
    set_proto(meta.get("proto", {}))
    cfg = ProtoConfig.from_dict(meta["cfg"], stored=False)
    cfg.extinction_threshold = 0
    with np.load(os.path.join(pilot, f"ckpt_{at:06d}.npz"), allow_pickle=False) as z:
        blob = {k: z[k] for k in z.files}
    world = ProtoWorld.from_checkpoint(blob, cfg)
    restore_tags(world, blob)
    return world


def record(world, games):
    """Step `games` iterations; frames keyed (iteration, phase) as book_data.frame_at returns them."""
    from proto import P
    if P.get("engine"):
        return record_engine(world, games)
    frames = {}
    play = type(world).blotto_phase

    def game(self, record_decisions):
        G = self.G
        t = self.iteration
        targets = {u: [u] + sorted(G.neighbors(u)) for u in G.nodes()}
        tokens = {u: int(self.tokens.get(u, 0)) for u in G.nodes()}
        frames[(t, 1)] = {"ids": list(G.nodes()), "brain_ids": [int(self.brains[u].brain_id) for u in G.nodes()]}
        out = play(self, record_decisions)
        sent = self.last_alloc
        frames[(t, 2)] = {
            "ids": list(G.nodes()), "brain_ids": [int(self.brains[u].brain_id) for u in G.nodes()],
            "edges": [[int(a), int(b)] for a, b in G.edges()],
            "decisions": {
                "allocations": [{"agent": int(u), "tokens": tokens[u], "targets": tg,
                                 "alloc": [int(sent.get(u, {}).get(v, 0)) for v in tg]}
                                for u, tg in targets.items()],
                "winners": [{"node": int(v), "winner": int(w)} for v, w in self.last_winners.items()]},
            "allies": [[int(u), int(v)] for u, vs in getattr(self, "last_allies", {}).items() for v in vs if u < v]}
        return out

    world.blotto_phase = game.__get__(world)
    first = world.iteration
    for _ in range(games):
        world.step(record_decisions=False)
        if world.G.number_of_nodes() == 0:
            break
    return first, world.iteration, frames


def record_engine(world, games):
    """The engine records its own games, in the lab's form: keep the frames it hands back."""
    frames = {}
    first = world.iteration
    for _ in range(games):
        for frame in world.step(record_decisions=True):
            frames[(frame["iteration"], frame["phase"])] = frame
        if world.G.number_of_nodes() == 0:
            break
    return first, world.iteration, frames


def rules(pilot):
    """How the pilot's game decides a node, for decisive.py: the engine's rule or the prototype's."""
    meta = json.load(open(os.path.join(pilot, "config.json")))
    proto, cfg = meta.get("proto", {}), meta["cfg"]
    if proto.get("engine"):
        return {"pooled": False, "incumbency": float(cfg.get("incumbency") or 1.0),
                "revolutions": bool(cfg.get("allow_revolutions"))}
    return {"pooled": bool(proto.get("allies")), "incumbency": float(proto.get("incumbency", 1.0)),
            "revolutions": bool(cfg.get("allow_revolutions"))}


def alliances(frames, t0, t1):
    """What the alliance labels say, to check the label-free measures against (alliance pilots only)."""
    staked = on_allies = gifts_to_allies = 0
    kept_allied, kept_alone = [], []
    pairs_run, mutual_long, mutual_long_allied = {}, 0, 0
    for t in range(t0, t1):
        f = frames.get((t, 2))
        if not f:
            continue
        allied = {(u, v) for u, v in f.get("allies", [])} | {(v, u) for u, v in f.get("allies", [])}
        has_ally = {u for u, v in allied}
        winners = {w["node"]: w["winner"] for w in f["decisions"]["winners"]}
        stakes = {}
        for a in f["decisions"]["allocations"]:
            u = a["agent"]
            for v, x in zip(a["targets"][1:], a["alloc"][1:]):
                if x > 0:
                    stakes[(u, v)] = x
                    staked += x
                    if (u, v) in allied:
                        on_allies += x
                        gifts_to_allies += x * (winners.get(v) != u)
            if a["tokens"] > 0:
                (kept_allied if u in has_ally else kept_alone).append(winners.get(u) == u)
        mutual = {(u, v) for (u, v) in stakes if u < v and (v, u) in stakes}
        pairs_run = {pr: pairs_run.get(pr, 0) + 1 for pr in mutual}
        for pr, L in pairs_run.items():
            if L == 3:
                mutual_long += 1
                mutual_long_allied += pr in allied
    return {"stakes_on_allies": on_allies / max(1, staked),
            "of_those_gifts": gifts_to_allies / max(1, on_allies),
            "kept_with_ally": float(np.mean(kept_allied)) if kept_allied else None,
            "kept_without": float(np.mean(kept_alone)) if kept_alone else None,
            "agents_with_ally": len(kept_allied) / max(1, len(kept_allied) + len(kept_alone)),
            "partnerships_allied": mutual_long_allied / max(1, mutual_long), "partnerships": mutual_long}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot")
    ap.add_argument("--at", type=int, required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    world = load(a.pilot, a.at)
    t0, t1, frames = record(world, a.games)
    D.frame_at = lambda run, t, phase: frames.get((t, phase))
    name = os.path.relpath(a.pilot, os.path.join(ROOT, "GraphOfLifeRuns", ".pilots"))
    res = {"pilot": name, "from": t0, "to": t1,
           "structures": check("structures").analyse(name, t0, t1),
           "gliders": check("gliders").analyse(name, t0, t1),
           "circulation": check("circulation").analyse(name, t0, t1, a.window),
           "kin": check("kin_flows").analyse(name, t0, t1),
           "alliances": alliances(frames, t0, t1),
           "decisive": check("decisive").analyse(name, t0, t1, rules(a.pilot))}
    res["gliders"].pop("examined_rows", None)
    for k in ("all", "gift", "take"):
        res["circulation"].pop(k + "_rows", None)
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
