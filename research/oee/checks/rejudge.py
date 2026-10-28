"""
Re-judge worlds with contests a lifetime long: the later era against earlier ones, T = 100.

The tournaments of Part V lasted 20 iterations, too short for what pays off over
an agent's life (Chapter 50, "Would we notice if it went on?"). This plays a few
pairs of eras again with T = 100, once in each era's world, for pilot worlds
(research/oee/pilots/tourney.contest, the prototype's own rules) and lab runs
(gol_tournament.contest).

    python3 research/oee/checks/rejudge.py --worlds worlds.txt --T 100 --jobs 2 --out out.json

worlds.txt: one world per line, `pilot <dir> <a>:<b>,<a>:<b>` or `lab <run> <a>:<b>,...`.
Results are appended to --out as they come, so the command can be stopped and
run again: contests already in the file are not played twice.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))


def play(task):
    kind, world, home, a, b, T, seed = task
    if kind == "pilot":
        import tourney
        state_it, a_it, b_it, s, share_a, tok_a, n = tourney.contest((world, home, a, b, T, seed))
        share = None if share_a is None else 1.0 - share_a
        tokens = None if tok_a is None else 1.0 - tok_a
        agents = n
    else:
        import gol_store as store
        import gol_tournament as tour
        config = store.load_meta(world)["config"]
        out = tour.contest(config, store.load_archive(world, home), store.load_archive(world, a),
                           store.load_archive(world, b), T, seed)
        share, tokens, agents = out["share"], out["tokens"], out["agents"]
    return {"kind": kind, "world": world, "home": home, "a": a, "b": b, "T": T, "seed": seed,
            "share": share, "tokens": tokens, "agents": agents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="17", help="contest seeds, comma-separated: an era against itself "
                                                  "(a pair a:a) needs more than one to be a control")
    a = ap.parse_args()
    done = []
    if os.path.exists(a.out):
        done = json.load(open(a.out))
    have = {(r["world"], r["home"], r["a"], r["b"], r["T"], r["seed"]) for r in done}
    seeds = [int(x) for x in a.seeds.split(",")]
    tasks = []
    for line in open(a.worlds):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, world, pairs = line.split()
        for pair in pairs.split(","):
            x, y = (int(v) for v in pair.split(":"))
            for home in sorted({x, y}):
                for seed in seeds:
                    if (world, home, x, y, a.T, seed) not in have:
                        tasks.append((kind, world, home, x, y, a.T, seed))
    print(f"{len(tasks)} contests to play", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        futures = [pool.submit(play, t) for t in tasks]
        for f in as_completed(futures):
            done.append(f.result())
            json.dump(done, open(a.out + ".tmp", "w"))
            os.replace(a.out + ".tmp", a.out)
    print("all played", flush=True)


if __name__ == "__main__":
    main()
