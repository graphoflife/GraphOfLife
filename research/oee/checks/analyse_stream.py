"""
Analyse pilot worlds as they finish, so the measures keep pace with the runs.

For every world of the named batches whose checkpoint AT is written and whose
run has ended, once:
  1. AT's games played on and recorded, GAMES of them (pilot_frames.py →
     .pilots/frames_<batch>_<name>_<AT>.json): structures, gliders, circulation,
     kin, help, and the gifts;
  2. the lifetime contests between eras AT/2 and AT, T = 100 (rejudge.py →
     .pilots/batch_<label>_T100.json);
  3. how different the agents are at AT (diversity.py → .pilots/diversity/<label>.json).
After each world the tools that sweep every pilot pick it up themselves:
research/atlas/across.py pilots (the table of every world) and
research/atlas/flowmodules.py pilots (flow modules).

    python3 research/oee/checks/analyse_stream.py T U V S P Q R --at 1000 --label TUV --queue PID

(S, P, Q and R are the controls of T, measured at the same iteration.)

It scans every few minutes and ends when every queue named has ended and every
world is analysed. A world whose analysis fails twice is reported and left.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
DONE = os.path.join(PILOTS, "analysis")
GAMES = 200
NICE = ["nice", "-n", "10", "python3"]


def running() -> set:
    """The world directories a run.py is still advancing."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    here = os.path.join(ROOT, "research", "oee", "pilots")
    return {os.path.realpath(os.path.join(here, line.split()[2]))
            for line in out.splitlines() if line.startswith("python3 run.py ") and len(line.split()) > 2}


def finished(batches, at):
    busy = running()
    for batch in batches:
        for d in sorted(glob.glob(os.path.join(PILOTS, batch, "*"))):
            if (os.path.exists(os.path.join(d, f"ckpt_{at:06d}.npz")) and os.path.realpath(d) not in busy
                    and os.path.exists(os.path.join(d, "config.json"))):
                yield batch, os.path.basename(d), d


def marker(batch, name):
    return os.path.join(DONE, f"{batch}_{name}.json")


def call(args, log):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(args)}\n")
        f.flush()
        return subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def judged() -> set:
    """Every lifetime contest already played, in any file: (world, a, b, T)."""
    out = set()
    for path in glob.glob(os.path.join(PILOTS, "*T100*.json")):
        try:
            rows = json.load(open(path))
        except ValueError:
            continue
        for c in rows if isinstance(rows, list) else []:
            if isinstance(c, dict) and "world" in c:
                out.add((c["world"].split(".pilots/")[-1], int(c["a"]), int(c["b"]), int(c.get("T", 0))))
    return out


def analyse(item):
    batch, name, d, at, label = item
    log = os.path.join(DONE, f"{batch}_{name}.log")
    steps = {}
    frames = os.path.join(PILOTS, f"frames_{batch}_{name}_{at}.json")
    if not os.path.exists(frames):
        steps["frames"] = call(NICE + ["research/oee/checks/pilot_frames.py", d, "--at", str(at), "--games", str(GAMES),
                                       "--out", frames], log)
    # Contests already played for an earlier batch are not played again. The list
    # of worlds is split on spaces, so it names the world relative to the project.
    if (f"{batch}/{name}", at // 2, at, 100) not in judged():
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"pilot {os.path.relpath(d, ROOT)} {at // 2}:{at}\n")
            worlds = f.name
        steps["contests"] = call(NICE + ["research/oee/checks/rejudge.py", "--worlds", worlds, "--T", "100",
                                         "--jobs", "2", "--out", os.path.join(PILOTS, f"batch_{label}_T100.json")], log)
        os.unlink(worlds)
    steps["diversity"] = call(NICE + ["research/oee/checks/diversity.py", f"pilot:{d}:{at}", "--label", label], log)
    return batch, name, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--at", type=int, default=1000)
    ap.add_argument("--label", required=True)
    ap.add_argument("--queue", type=int, action="append", default=[],
                    help="a queue whose worlds are still to come (may be given more than once)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(DONE, exist_ok=True)
    failures = {}
    while True:
        todo = [(b, n, d, a.at, a.label) for b, n, d in finished(a.batches, a.at)
                if not os.path.exists(marker(b, n)) and failures.get((b, n), 0) < 2]
        if todo:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                for batch, name, steps in pool.map(analyse, todo):
                    if all(code == 0 for code in steps.values()):
                        json.dump({"steps": steps, "at": a.at, "when": time.strftime("%Y-%m-%d %H:%M")},
                                  open(marker(batch, name), "w"))
                        print(f"{batch}/{name}: analysed", flush=True)
                    else:
                        failures[(batch, name)] = failures.get((batch, name), 0) + 1
                        print(f"{batch}/{name}: failed {steps}", flush=True)
            # The sweeps over every pilot take up the new worlds.
            for tool in (["research/atlas/across.py", "pilots", "--jobs", "1"],
                         ["research/atlas/flowmodules.py", "pilots", "--jobs", "1"]):
                call(NICE + tool, os.path.join(DONE, "sweeps.log"))
            continue
        queue_alive = any(subprocess.run(["kill", "-0", str(q)], capture_output=True).returncode == 0 for q in a.queue)
        if not queue_alive and not running() & {os.path.realpath(d) for b in a.batches
                                                 for d in glob.glob(os.path.join(PILOTS, b, "*"))}:
            print("stream analysed", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
