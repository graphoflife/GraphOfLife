"""
Play each pilot world's tournament as soon as the world has finished, one world at a time.

    python3 when_done.py DIR [DIR ...] --last 1000 --eras 0,100,200,500,1000 --jobs 3 [--layout halves --T 50]

A world has finished when its checkpoint of iteration --last exists, or when its
series stopped short of it because it died out (no tournament then). Worlds whose
tournament file already exists are skipped, so the command can be run again.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def finished(d, last):
    if os.path.exists(os.path.join(d, f"ckpt_{last:06d}.npz")):
        return "done"
    series = os.path.join(d, "series.jsonl")
    if os.path.exists(series):
        with open(series) as f:
            rows = f.readlines()
        if rows and json.loads(rows[-1])["nodes"] <= 1:
            return "extinct"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--last", type=int, default=1000)
    ap.add_argument("--eras", default="0,100,200,500,1000")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--layout", default="coins")
    a = ap.parse_args()
    out_name = f"tourney_T{a.T}_all{'' if a.layout == 'coins' else '_' + a.layout}.json"
    todo = list(a.dirs)
    while todo:
        for d in list(todo):
            state = finished(d, a.last)
            if state is None:
                continue
            todo.remove(d)
            if state == "extinct" or os.path.exists(os.path.join(d, out_name)):
                print(f"{d}: {state}, no tournament needed", flush=True)
                continue
            cmd = [sys.executable, os.path.join(HERE, "tourney.py"), d, "--eras", a.eras, "--T", str(a.T),
                   "--reps", str(a.reps), "--jobs", str(a.jobs), "--layout", a.layout]
            print(f"{d}: playing", flush=True)
            subprocess.run(cmd, check=False)
        if todo:
            time.sleep(60)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
