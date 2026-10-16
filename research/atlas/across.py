"""
Every world the book has made, side by side: one row of statistics for each.

A lab run records about ninety-five statistics after every phase
(stats.jsonl); a pilot world keeps a few of them and its checkpoints. To put
them in one table, each world is taken over the same stretches of iterations:

    500   iterations 400–499        1000  900–999        2000  1900–1999
    end   the last fifth of what the world lived

A lab run's numbers are the means of its record over each stretch. A pilot's
are made the same way from frames: from its checkpoint at 1,000 and from its
last one, the world is played on for GAMES games, recorded as the lab records
its own, and each game's row is the one gol_series.frame_stats gives a lab run
(the heavy statistics every fifth game, as a lab run has them every 25th
iteration). A pilot played in the prototype's own game, not the engine's,
records no decisions, so it keeps the statistics its series has.

    python3 research/atlas/across.py lab                 # every lab run, from its record
    python3 research/atlas/across.py pilots [--jobs 2]   # every pilot not yet measured
    python3 research/atlas/across.py write               # book/results/worlds.json

The lab and pilot rows are kept in GraphOfLifeRuns/.pilots/atlas/across/, one
file per world, so a new batch adds its worlds without measuring the rest again.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "checks"))

import numpy as np

RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS, ".pilots")
OUT = os.path.join(PILOTS, "atlas", "across")
TABLE = os.path.join(ROOT, "book", "results", "worlds.json")
WINDOWS = {"500": (400, 500), "1000": (900, 1000), "2000": (1900, 2000)}
ORDER = ["500", "1000", "2000", "end"]
#: Games a pilot is played on for from a checkpoint, and how often the heavy statistics are taken.
GAMES = 20
HEAVY_EVERY = 5
#: Not statistics of a world: bookkeeping, or the same in every frame.
NOT_STATISTICS = {"iteration", "phase", "tokens", "nodes_before", "cladesInWindow", "it", "secs"}
#: The pilot series' names for what the lab records under the same name; the
#: rest of a series is kept under its own name.
SERIES_SKIP = {"it", "secs", "mixed", "mutations"}


def statistic(name, value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and not name.startswith("_") and name not in NOT_STATISTICS)


def means(rows):
    """The mean of every statistic over rows, each over the rows that have it."""
    sums, counts = {}, {}
    for r in rows:
        for k, v in r.items():
            if statistic(k, v) and np.isfinite(v):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def games(rows):
    """One row per game: its reproduction phase's statistics, and its game phase's where both have one."""
    by = {}
    for r in rows:
        t, p = r.get("iteration"), r.get("phase")
        if t is not None:
            by.setdefault(int(t), {})[int(p or 0)] = r
    out = {}
    for t, phases in by.items():
        row = {k: v for k, v in (phases.get(1) or {}).items() if v is not None}
        row.update({k: v for k, v in (phases.get(2) or {}).items() if v is not None})
        out[t] = row
    return out


def stretches(last):
    """The stretches of iterations a world that lived to `last` has, as (name, first, stop)."""
    out = [(name, a, b) for name, (a, b) in WINDOWS.items() if last >= b - 1]
    if last >= 50:
        out.append(("end", int(last * 0.8), last + 1))
    return out


# ---------------------------------------------------------------------------
# Lab runs
# ---------------------------------------------------------------------------

def conditions():
    """Run id -> the experiments and conditions it belongs to, from every plan."""
    import gol_plan
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "book", "experiments", "E*.json"))):
        name = os.path.basename(path)[:-5]
        try:
            specs = gol_plan.experiment_runs(name)
        except Exception:
            continue
        for s in specs:
            out.setdefault(s.run_id, []).append(f"{name} · {s.condition}")
    return out


def lab_world(run, named):
    meta = json.load(open(os.path.join(RUNS, run, "meta.json")))
    path = os.path.join(RUNS, run, "stats.jsonl")
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 200000))
        tail = f.read().decode(errors="ignore").splitlines()
    last = max((json.loads(line).get("iteration", 0) for line in tail[1:] if line.strip()), default=0)
    wanted = stretches(last)
    rows = []
    pattern = re.compile(r'"iteration": ?(\d+)')
    lo = min((a for _, a, _ in wanted), default=0)
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m and int(m.group(1)) >= lo:
                rows.append(json.loads(line))
    by_game = games(rows)
    at = {}
    for name, a, b in wanted:
        chosen = [by_game[t] for t in range(a, b) if t in by_game]
        if chosen:
            at[name] = means(chosen)
    lab = meta.get("lab") or {}
    cfg = meta.get("config") or {}
    family = re.sub(r"-s\d+$", "", run)
    return {
        "id": run, "family": family, "source": "lab", "baseline": lab.get("baseline", ""),
        "tokens": cfg.get("total_tokens"), "settings": lab.get("set") or {},
        "conditions": named.get(run, []), "lived": meta.get("status") != "extinct", "end": last, "at": at,
    }


def measure_lab():
    os.makedirs(OUT, exist_ok=True)
    named = conditions()
    runs = sorted(d for d in os.listdir(RUNS)
                  if os.path.exists(os.path.join(RUNS, d, "stats.jsonl"))
                  and os.path.exists(os.path.join(RUNS, d, "meta.json")))
    for i, run in enumerate(runs):
        world = lab_world(run, named)
        json.dump(world, open(os.path.join(OUT, f"lab_{run}.json"), "w"))
        print(f"{i + 1}/{len(runs)} {run} {sorted(world['at'])}", flush=True)


# ---------------------------------------------------------------------------
# Pilots
# ---------------------------------------------------------------------------

def pilots():
    out = []
    for batch in sorted(os.listdir(PILOTS)):
        root = os.path.join(PILOTS, batch)
        if not os.path.isdir(root) or batch in ("atlas", "png"):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.exists(os.path.join(d, "config.json")) and glob.glob(os.path.join(d, "ckpt_*.npz")):
                out.append((batch, name, d))
    return out


def checkpoints(d):
    return sorted(int(re.search(r"ckpt_(\d+)", p).group(1)) for p in glob.glob(os.path.join(d, "ckpt_*.npz")))


def recorded_means(d, at):
    """Play a pilot on from its checkpoint `at` for GAMES games and take the mean of their rows."""
    import gol_series
    import pilot_frames
    world = pilot_frames.load(d, at)
    _, _, frames = pilot_frames.record_engine(world, GAMES)
    rows, previous = [], None
    for key in sorted(frames):
        frame = frames[key]
        heavy = frame["phase"] == 2 and (frame["iteration"] - at) % HEAVY_EVERY == 0
        row = gol_series.frame_stats(frame, previous if not frame.get("delta") else None, heavy=heavy)
        row["iteration"], row["phase"] = frame["iteration"], frame["phase"]
        rows.append(row)
        previous = frame
    return means(games(rows).values())


def series_means(d, a, b):
    rows = []
    with open(os.path.join(d, "series.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if a <= r.get("it", -1) < b:
                rows.append({k: v for k, v in r.items() if k not in SERIES_SKIP})
    out = means(rows)
    if out.get("deaths"):
        out["deathAge"] = out.pop("deathAgeSum", 0.0) / out["deaths"]
    return out


def pilot_world(item):
    batch, name, d = item
    meta = json.load(open(os.path.join(d, "config.json")))
    cfg, proto = meta.get("cfg", {}), meta.get("proto", {})
    have = checkpoints(d)
    last = have[-1]
    engine = bool(proto.get("engine"))
    # Each stretch: the series' means over it, and for a pilot that plays the
    # engine's game the full row from games recorded at the checkpoint that
    # starts it (each checkpoint played once, the last one for the end).
    starts = {"1000": 1000, "2000": 2000, "end": last}
    recorded, at = {}, {}
    for window, a, b in stretches(last):
        row = series_means(d, a, b)
        start = starts.get(window)
        if engine and start in have:
            if start not in recorded:
                recorded[start] = recorded_means(d, start)
            row.update(recorded[start])
        at[window] = row
    family = re.sub(r"-s\d+$", "", name)
    return {
        "id": f"{batch}/{name}", "family": f"{batch}/{family}", "source": "pilot",
        "batch": batch, "baseline": "B2" if cfg.get("mend_on_death") else "B1",
        "tokens": cfg.get("total_tokens"), "settings": {k: v for k, v in proto.items() if k != "engine"},
        "engine": engine, "lived": True, "end": last, "at": at,
    }


def running():
    """The pilot worlds a run.py is still advancing: measured later, when they are done."""
    import subprocess
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return {os.path.realpath(os.path.join(ROOT, "research", "oee", "pilots", line.split()[2]))
            for line in out.splitlines() if line.startswith("python3 run.py ") and len(line.split()) > 2}


def measure_pilots(jobs, redo=False):
    os.makedirs(OUT, exist_ok=True)
    busy = running()
    todo = [p for p in pilots() if os.path.realpath(p[2]) not in busy
            and (redo or not os.path.exists(os.path.join(OUT, f"pilot_{p[0]}_{p[1]}.json")))]
    print(f"{len(todo)} pilots to measure", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for item, world in zip(todo, pool.map(pilot_world, todo)):
            json.dump(world, open(os.path.join(OUT, f"pilot_{item[0]}_{item[1]}.json"), "w"))
            print(item[0], item[1], sorted(world["at"]), flush=True)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def rounded(v):
    return float(f"{v:.4g}") if isinstance(v, float) and np.isfinite(v) else v


def write():
    from book_chapters.everyworld import replicas
    repeated = replicas()
    worlds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    for w in worlds:
        w["at"] = {name: {k: rounded(v) for k, v in stats.items()} for name, stats in w["at"].items()}
        # A pilot that replays a lab world is that world again: say which, so a comparison counts it once.
        if w["source"] == "pilot" and w["id"] in repeated:
            w["replica_of"] = repeated[w["id"]]
    stats = sorted({k for w in worlds for s in w["at"].values() for k in s})
    table = {
        "what": "Every world the book has made, each over the same stretches of iterations "
                 "(research/atlas/across.py): the means of the statistics a lab run records, "
                 "and the same for a pilot from games recorded at its checkpoints.",
        "windows": [w for w in ORDER if any(w in x["at"] for x in worlds)],
        "stats": stats,
        "worlds": worlds,
    }
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    json.dump(table, open(TABLE, "w"), separators=(",", ":"))
    print(f"{len(worlds)} worlds, {len(stats)} statistics -> {TABLE} ({os.path.getsize(TABLE) // 1024} kB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("lab", "pilots", "write"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    if a.what == "lab":
        measure_lab()
    elif a.what == "pilots":
        measure_pilots(a.jobs, a.redo)
    else:
        write()


if __name__ == "__main__":
    main()
