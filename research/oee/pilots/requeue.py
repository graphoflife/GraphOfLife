"""
What is left of a list of jobs, after a queue was stopped (a reboot, say).

For every job of the list: a world that reached its last iteration is left out;
a world with a checkpoint goes on from its last one (`run.py --resume last`),
for the iterations it still lacks; a world never started runs as written. The
same world named by two lists, or twice, is taken once.

    python3 requeue.py jobs_K.txt [jobs_Y.txt ...] > jobs_left.txt
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def last_iteration(d: str) -> int:
    with open(os.path.join(d, "series.jsonl")) as f:
        lines = f.read().splitlines()
    return json.loads(lines[-1])["it"] if lines else 0


def target(cmd: str, d: str) -> int:
    """The iteration a job's world is to reach."""
    iters = int(re.search(r"--iters (\d+)", cmd).group(1))
    if "--resume" in cmd:
        # A job that went on from a checkpoint names only the iterations to add.
        meta = json.load(open(os.path.join(d, "config.json")))
        if meta.get("resumes"):
            return max(r["at"] + r["iters"] for r in meta["resumes"])
        # Not gone on yet (a candidate's fresh copy): from its last checkpoint.
        return iters + max(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(d, "ckpt_*.npz")))
    start = re.search(r"--from \S+:(\d+)", cmd)
    return iters + (int(start.group(1)) if start else 0)


def left(cmd: str) -> str | None:
    d = os.path.normpath(os.path.join(HERE, cmd.split()[2]))
    if not os.path.exists(os.path.join(d, "series.jsonl")):
        return cmd
    goal, now = target(cmd, d), last_iteration(d)
    if now >= goal:
        return None
    checkpoints = glob.glob(os.path.join(d, "ckpt_*.npz"))
    if not checkpoints:
        return cmd
    at = max(int(os.path.basename(p)[5:11]) for p in checkpoints)
    ckpt = re.search(r"--ckpt (\d+)", cmd)
    health = re.search(r"--health (\d+)", cmd)
    return (f"python3 run.py {cmd.split()[2]} --resume last --iters {goal - at}"
            + (f" --ckpt {ckpt.group(1)}" if ckpt else "") + (f" --health {health.group(1)}" if health else ""))


def main():
    seen, out = set(), []
    for path in sys.argv[1:]:
        for line in open(os.path.join(HERE, path)):
            if not line.startswith("python3 run.py"):
                continue
            world = os.path.normpath(line.split()[2])
            if world in seen:
                continue
            seen.add(world)
            job = left(line.strip())
            if job:
                out.append(job)
    print(f"# What was left of {', '.join(sys.argv[1:])} (requeue.py).")
    print("\n".join(out))


if __name__ == "__main__":
    main()
