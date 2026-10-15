"""
Run one pilot world and record what the OEE yardstick needs.

    python3 run.py OUTDIR --iters 1000 --ckpt 100 --set total_tokens=5000 hidden_layers=16 --proto mor=all

Writes OUTDIR/series.jsonl (one row per iteration, plus network health every
`--health` iterations), OUTDIR/ckpt_<it>.npz (whole-world checkpoints with the
founder root of every brain) and OUTDIR/config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx

from gol_config import SimConfig
from gol_series import _structure
from proto import make_world, make_config


def parse_value(v: str):
    if v in ("true", "on"):
        return True
    if v in ("false", "off"):
        return False
    if "," in v or v.startswith("["):
        return [int(x) for x in v.strip("[]").split(",") if x]
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def gini(values):
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def domains(world, same):
    """Connected groups of agents joined by edges whose ends satisfy `same`: (count, largest, node-weighted mean size)."""
    G = world.G
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((a, b) for a, b in G.edges() if same(a, b))
    sizes = [len(c) for c in nx.connected_components(H)]
    n = sum(sizes)
    return len(sizes), max(sizes), (sum(s * s for s in sizes) / n if n else 0.0)


def P_tags():
    from proto import P
    return P.get("tag_rate", 0) > 0


def organisation(world):
    out = {}
    bid = {u: world.brains[u].brain_id for u in world.G.nodes()}
    c, big, mean = domains(world, lambda a, b: bid[a] == bid[b])
    out.update(cloneDomains=c, cloneLargest=big, cloneMean=mean)
    c, big, mean = domains(world, lambda a, b: world.kin_distance(bid[a], bid[b], 4) is not None)
    out.update(kin4Domains=c, kin4Largest=big, kin4Mean=mean)
    if P_tags():
        c, big, mean = domains(world, lambda a, b: world.family(bid[a]) == world.family(bid[b]))
        out.update(familyDomains=c, familyLargest=big, familyMean=mean)
    from proto import pooled
    if pooled():
        c, big, mean = domains(world, lambda a, b: world.same_side(a, b))
        out.update(sideDomains=c, sideLargest=big, sideMean=mean)
    alloc = getattr(world, "last_alloc", {}) or {}
    allocations = [{"agent": u, "targets": list(t.keys()), "alloc": list(t.values())}
                   for u, t in alloc.items()]
    if allocations:
        from gol_lightning import lightning
        lt = lightning({"decisions": {"allocations": allocations}})
        out.update({k: lt[k] for k in ("lightningScore", "cyclingShare", "lightningLongest",
                                         "netLightningScore", "netCyclingShare", "netLightningLongest")})
    return out


def health(world):
    G = world.G
    ids = list(G.nodes())
    n = len(ids)
    s = _structure(ids, [list(e) for e in G.edges()])
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    core = nx.core_number(G) if n else {}
    out["core3Share"] = (sum(1 for v in core.values() if v >= 3) / n) if n else 0.0
    out["articulationShare"] = (len(list(nx.articulation_points(G))) / n) if n else 0.0
    from cuts import node_cut_risk
    out["nodeCutRisk"] = node_cut_risk(G)
    deg = np.array([d for _, d in G.degree()]) if n else np.zeros(1)
    out["meanDegree"] = float(deg.mean())
    out["maxDegreeShare"] = float(deg.max() / max(1, n - 1))
    out["leafShare"] = float(np.mean(deg <= 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--ckpt", type=int, default=100)
    ap.add_argument("--health", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--proto", nargs="*", default=[])
    ap.add_argument("--from", dest="start", default="",
                    help="RUN:ITERATION: go on from a lab run's archive instead of new founders "
                         "(its own settings; --set is ignored)")
    ap.add_argument("--copy-eras", default="",
                    help="with --from: copy these archived iterations of the lab run in as checkpoints")
    ap.add_argument("--resume", default="",
                    help="ITERATION or 'last': go on in OUT from its own checkpoint, with its own settings "
                         "(--set and --proto are ignored); the series is appended to")
    a = ap.parse_args()
    if a.resume:
        return resume(a)

    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
    if "hidden_layers" in overrides and isinstance(overrides["hidden_layers"], int):
        overrides["hidden_layers"] = [overrides["hidden_layers"]]
    proto = {}
    for kv in a.proto:
        k, v = kv.split("=", 1)
        proto[k] = parse_value(v)

    from proto import set_proto
    set_proto(proto)
    if overrides.get("brain_kind") == "grow":
        import grow
        grow.install()
    os.makedirs(a.out, exist_ok=True)
    if a.start:
        import gol_store as store
        from proto import ProtoConfig, ProtoWorld
        run, at = a.start.rsplit(":", 1)
        cfg = ProtoConfig.from_dict(store.load_meta(run)["config"])
        cfg.extinction_threshold = 0
        world = ProtoWorld.from_checkpoint(store.load_archive(run, int(at)), cfg)
        for era in [int(e) for e in a.copy_eras.split(",") if e]:
            import shutil
            shutil.copy(store.archive_path(run, era), os.path.join(a.out, f"ckpt_{era:06d}.npz"))
    else:
        cfg = make_config(**overrides)
        cfg.extinction_threshold = 0
        world = make_world(cfg, proto)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"cfg": cfg.to_dict(), "proto": proto, "strain": cfg.strain_id(),
                   "iters": a.iters, "from": a.start or None}, f, indent=1)

    run_loop(world, a, os.path.join(a.out, "series.jsonl"), mode="w", first_ckpt_done=bool(a.start))


def run_loop(world, a, path, mode, first_ckpt_done):
    """Step the world a.iters times: a row of the series each iteration, a checkpoint every a.ckpt."""
    rows = open(path, mode)
    t0 = time.time()
    first = world.iteration
    for it in range(first, first + a.iters):
        if a.ckpt and it % a.ckpt == 0 and not (first_ckpt_done and it == first):
            save_ckpt(world, a.out, it)
        world.c = {}
        world.step(record_decisions=False)
        G = world.G
        n = G.number_of_nodes()
        ages = np.array([world._age_of(u) for u in G.nodes()]) if n else np.zeros(1)
        roots = {}
        genos = set()
        for u in G.nodes():
            r = world.root(u)
            roots[r] = roots.get(r, 0) + 1
            genos.add(world.brains[u].brain_id)
        row = {"it": world.iteration, "nodes": n, "edges": G.number_of_edges(),
               "meanAge": float(ages.mean()), "maxAge": int(ages.max()),
               "medianAge": float(np.median(ages)),
               "roots": len(roots), "topRootShare": max(roots.values()) / n if n else 0,
               "genotypes": len(genos),
               "gini": gini([world.tokens[u] for u in G.nodes()]),
               "secs": round(time.time() - t0, 2), **world.c}
        if n and hasattr(world.brains[next(iter(G.nodes()))], "active"):
            act = [world.brains[u].active for u in G.nodes()]
            row["activeMean"] = float(np.mean(act))
            row["activeMax"] = int(max(act))
            row["shadowMean"] = float(np.mean([world.brains[u].shadow for u in G.nodes()]))
            row["shadowMax"] = int(max(world.brains[u].shadow for u in G.nodes()))
            row["memMean"] = float(np.mean([world.brains[u].mem for u in G.nodes()]))
            row["memShadowMean"] = float(np.mean([world.brains[u].memshadow for u in G.nodes()]))
        deaths = world.death_ages
        row["deaths"] = len(deaths)
        row["deathAgeSum"] = int(sum(deaths))
        world.death_ages = []
        if a.health and world.iteration % a.health == 0 and n > 1:
            row.update(health(world))
            row.update(organisation(world))
        rows.write(json.dumps(row) + "\n")
        rows.flush()
        if n == 0:
            break
    if a.ckpt:
        save_ckpt(world, a.out, world.iteration)
    rows.close()


def resume(a):
    """Go on with a pilot world from one of its own checkpoints, for a.iters more iterations."""
    import glob
    from proto import ProtoConfig, ProtoWorld, restore_tags, set_proto
    meta = json.load(open(os.path.join(a.out, "config.json")))
    set_proto(meta.get("proto", {}))
    if meta["cfg"].get("brain_kind") == "grow":
        import grow
        grow.install()
    cfg = ProtoConfig.from_dict(meta["cfg"], stored=False)
    cfg.extinction_threshold = 0
    if a.resume == "last":
        at = max(int(os.path.basename(f)[5:11]) for f in glob.glob(os.path.join(a.out, "ckpt_*.npz")))
    else:
        at = int(a.resume)
    with np.load(os.path.join(a.out, f"ckpt_{at:06d}.npz"), allow_pickle=False) as z:
        blob = {k: z[k] for k in z.files}
    world = ProtoWorld.from_checkpoint(blob, cfg)
    restore_tags(world, blob)
    import grow
    grow.restore_active(world, blob)
    meta.setdefault("resumes", []).append({"at": at, "iters": a.iters})
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(meta, f, indent=1)
    # The series goes on from the checkpoint: rows after it, from a run that went further
    # before, would be read twice.
    path = os.path.join(a.out, "series.jsonl")
    kept = [l for l in open(path) if json.loads(l)["it"] <= at] if os.path.exists(path) else []
    with open(path, "w") as f:
        f.writelines(kept)
    run_loop(world, a, path, mode="a", first_ckpt_done=True)


def save_ckpt(world, out, it):
    blob = world.to_checkpoint()
    nodes = sorted(world.G.nodes())
    blob["roots"] = np.array([world.root(u) for u in nodes], dtype=np.int64)
    from proto import P
    if P.get("tag_rate", 0) > 0:
        blob["tags"] = np.array([world.family(world.brains[u].brain_id) for u in nodes], dtype=np.int64)
    import grow
    grow.save_active(world, blob)
    np.savez_compressed(os.path.join(out, f"ckpt_{it:06d}.npz"), **blob)


if __name__ == "__main__":
    main()
