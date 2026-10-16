"""
The most promising worlds so far, and their long runs.

A candidate is a world that comes closer than B2 itself to the author's picture
of one that might keep evolving (Meta X, 2026-10-16):
  - it meets every mark of the health bar (Chapter 46): one piece, a 2-core of
    0.8, no agent or link whose loss cuts off more than 2% / 1%, no hub over
    5%, 3-12 links per agent, a dimension of 2.3-4;
  - its agents are more diverse than B2's own: its diversity score (the mean
    percentile of the spreads of tokens, links, ages and genotypes among all
    worlds) above the median of Experiment 16's B2 worlds;
  - it has at least 10,000 tokens, the book's rule for new runs.
No other line is drawn. Every candidate is played on to 10,000 iterations in a
folder of its own (GraphOfLifeRuns/.pilots/LC/<batch>-<world>), copied from the
original so that nothing a chapter reads changes (run.py --resume last).

    python3 candidates.py      writes research/oee/candidates.md and jobs_LC.txt (new candidates only)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
HERE = os.path.dirname(os.path.abspath(__file__))
LONG = 10000
KEEP = {"seed", "total_tokens", "extinction_threshold", "archive_every", "checkpoint_every", "export_every",
        "export_decisions"}


def settings_of(wid: str):
    """(tokens, what the world changes against the rule set it starts from) for a pilot or a lab run."""
    if "/" in wid:
        meta = json.load(open(os.path.join(PILOTS, wid, "config.json")))
        cfg, proto = meta["cfg"], {k: v for k, v in (meta.get("proto") or {}).items() if k != "engine"}
    else:
        meta = json.load(open(os.path.join(ROOT, "GraphOfLifeRuns", wid, "meta.json")))
        cfg, proto = meta["config"], {}
    base = "B2" if cfg.get("mend_on_death") else "B1"
    settings = json.load(open(os.path.join(ROOT, "book", "experiments", f"{base}.json")))["settings"]
    changed = [f"{k} = {v}" for k, v in sorted(cfg.items())
               if k not in KEEP and k in settings and v != settings[k]]
    changed += [f"{k} = {v}" for k, v in sorted(proto.items())]
    return int(cfg.get("total_tokens") or 0), base + (" with " + ", ".join(changed) if changed else "")


def main():
    from book_chapters.healthy import _load
    import numpy as np
    worlds = _load()
    b2 = float(np.median([w["diversity"] for w in worlds if w["group"] == "B2 10k"]))
    chosen = []
    for w in sorted(worlds, key=lambda w: -w["diversity"]):
        if w["marks"] < 7 or w["diversity"] <= b2 or w["id"].startswith(("L/", "LC/")):
            continue
        tokens, what = settings_of(w["id"])
        if tokens < 10000:
            continue
        chosen.append((w, what))
    lines = ["# Candidates for long runs", "",
             "Made by `research/oee/pilots/candidates.py` from the profile of every world (Meta X). A candidate",
             "meets every mark of Chapter 46's health bar, its agents are more diverse than B2's own (a diversity",
             f"score above {b2:.2f}, the median of Experiment 16's B2 worlds), and it has at least 10,000 tokens.",
             f"Each is played on to {LONG:,} iterations in `GraphOfLifeRuns/.pilots/LC/`.", "",
             "| world | what it is | diversity | median tokens | 3 tokens or fewer | dimension | agents | long run |",
             "|---|---|---|---|---|---|---|---|"]
    jobs = []
    for w, what in chosen:
        at = int(w["at"])
        name = w["id"].replace("/", "-")
        long_dir = os.path.join(PILOTS, "LC", name)
        status = "running or done" if os.path.exists(long_dir) else "queued"
        lines.append(f"| `{w['id']}` | {what} | {w['diversity']:.2f} | {w['median_tokens']:.0f} | "
                     f"{w['share_at_most_three']:.0%} | {w.get('dimension') or 0:.1f} | {w['agents']:,} | "
                     f"from {at:,} to {LONG:,}: {status} |")
        if os.path.exists(long_dir) or at >= LONG:
            continue
        if "/" in w["id"]:
            shutil.copytree(os.path.join(PILOTS, w["id"]), long_dir)
            jobs.append(f"python3 run.py ../../../GraphOfLifeRuns/.pilots/LC/{name} --resume last "
                        f"--iters {LONG - at} --ckpt 500 --health 50")
        else:
            jobs.append(f"python3 run.py ../../../GraphOfLifeRuns/.pilots/LC/{name} --from {w['id']}:{at} "
                        f"--iters {LONG - at} --ckpt 500 --health 50 --proto engine=true")
    open(os.path.join(ROOT, "research", "oee", "candidates.md"), "w").write("\n".join(lines) + "\n")
    if jobs:
        open(os.path.join(HERE, "jobs_LC.txt"), "a").write("\n".join(jobs) + "\n")
    print(f"{len(chosen)} candidates (B2's diversity {b2:.2f}); {len(jobs)} new long runs in jobs_LC.txt")


if __name__ == "__main__":
    main()
