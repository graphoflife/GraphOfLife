"""
Flow modules in every world, by the page's own arithmetic.

A flow module is a group of agents a token wandering the flow network tends to
stay inside (the map equation; web/js/flowmodules.js, run through
tools/flowmodules_cli.js). For each world, GAMES consecutive games are read —
a lab run's from its record, from iteration 950 when it lived that long; a
pilot's played on from its checkpoint at 1,000 — and the modules are followed
from game to game three ways: matched best-first above the view's floor of
0.3, matched only when each is the other's best overlap, and the latter again
on the same flow rewired at random with every agent's amount kept (the null).

    python3 research/atlas/flowmodules.py lab [--per-family 10] [--jobs 2]
    python3 research/atlas/flowmodules.py pilots [--jobs 2]

Writes GraphOfLifeRuns/.pilots/atlas/flowmodules/<world>.json; across.py write
adds the numbers to the table of every world.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "checks"))

RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
OUT = os.path.join(RUNS, ".pilots", "atlas", "flowmodules")
CLI = os.path.join(ROOT, "tools", "flowmodules_cli.js")
GAMES = 50
START = 950
FIELDS = ("iteration", "phase", "ids")


def cut(frame):
    out = {k: frame[k] for k in FIELDS if k in frame}
    out["decisions"] = {"allocations": (frame.get("decisions") or {}).get("allocations") or []}
    return out


def modules(frames):
    """The three readings of a window: floor 0.3, mutual best, and mutual best on rewired flow."""
    text = "\n".join(json.dumps(cut(f), separators=(",", ":")) for f in frames)
    out = {}
    for name, args in (("floor", []), ("mutual", ["--mutual"]), ("null", ["--mutual", "--null", "1"])):
        done = subprocess.run(["node", CLI] + args, input=text, capture_output=True, text=True, check=True)
        out[name] = json.loads(done.stdout)
    return out


def lab_world(run):
    import book_data as D
    meta = json.load(open(os.path.join(RUNS, run, "meta.json")))
    last = int(meta.get("iteration", 0))
    start = START if last >= START + GAMES else max(0, last - GAMES)
    frames = []
    for t in range(start, start + GAMES):
        try:
            frames.append(D.frame_at(run, t, 2))
        except Exception:
            break
    result = {"id": run, "family": re.sub(r"-s\d+$", "", run), "source": "lab", "from": start,
              "games": len(frames), **modules(frames)}
    json.dump(result, open(os.path.join(OUT, f"lab_{run}.json"), "w"))
    return run, result["mutual"]["summary"]


def pilot_world(item):
    import pilot_frames
    batch, name, d = item
    have = sorted(int(re.search(r"ckpt_(\d+)", p).group(1)) for p in glob.glob(os.path.join(d, "ckpt_*.npz")))
    at = 1000 if 1000 in have else have[-1]
    world = pilot_frames.load(d, at)
    _, _, recorded = pilot_frames.record_engine(world, GAMES)
    frames = [recorded[k] for k in sorted(recorded) if k[1] == 2]
    result = {"id": f"{batch}/{name}", "family": f"{batch}/{re.sub(r'-s[0-9]+$', '', name)}",
              "source": "pilot", "from": at, "games": len(frames), **modules(frames)}
    json.dump(result, open(os.path.join(OUT, f"pilot_{batch}_{name}.json"), "w"))
    return f"{batch}/{name}", result["mutual"]["summary"]


def done(path):
    """Whether a world's result is there and holds what the chapter draws (module lives came later)."""
    if not os.path.exists(path):
        return False
    try:
        return "lives" in json.load(open(path))["mutual"]
    except (ValueError, KeyError):
        return False


def lab_runs(per_family):
    runs = sorted(d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "stats.jsonl")))
    seen = {}
    chosen = []
    for run in runs:
        family = re.sub(r"-s\d+$", "", run)
        if seen.get(family, 0) >= per_family:
            continue
        meta = json.load(open(os.path.join(RUNS, run, "meta.json")))
        if not (meta.get("config") or {}).get("export_decisions", True) or int(meta.get("iteration", 0)) < GAMES:
            continue
        # A world measured already counts towards its family's share too.
        seen[family] = seen.get(family, 0) + 1
        if not done(os.path.join(OUT, f"lab_{run}.json")):
            chosen.append(run)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("lab", "pilots"))
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "lab":
        todo, work = lab_runs(a.per_family), lab_world
    else:
        import across
        busy = across.running()
        todo = [p for p in across.pilots()
                if os.path.realpath(p[2]) not in busy
                and json.load(open(os.path.join(p[2], "config.json"))).get("proto", {}).get("engine")
                and not done(os.path.join(OUT, f"pilot_{p[0]}_{p[1]}.json"))]
        work = pilot_world
    print(f"{len(todo)} worlds", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for name, summary in pool.map(work, todo):
            if summary:
                print(f"{name}: {summary['meanModules']:.0f} modules, compression {summary['compression']:.2f}, "
                      f"longest life {summary['longestLife']}, turnover {summary['meanTurnover']:.2f}", flush=True)
            else:
                print(f"{name}: no flow", flush=True)


if __name__ == "__main__":
    main()
