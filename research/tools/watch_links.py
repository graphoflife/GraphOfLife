#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watches the links of a follow-up's runs while they play (the author, 2026-11-09:
three of Experiment 43's settings keep their links with no hard limit, and the
answer was "keep, but watch"). For every run of the plan past iteration 100,
its links an agent over its last 25 games are set against what its world held
in round 3 (research/screen/health.json: links an agent averaged from iteration
100 on). A run past twice that is stopped the way the author would stop it: the
lab is paused, saying which run and why, for the author to decide whether the
world is taken out. Every check writes a line.

    python3 research/tools/watch_links.py E43            check, and pause the lab if a run's links run away
    python3 research/tools/watch_links.py E43 --dry-run  only say what it sees

Run from the checkout the lab runs from (cron, every 30 minutes, beside
move_finished_runs.py).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Dict, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import gol_lab  # noqa: E402
import gol_plan  # noqa: E402
import gol_record  # noqa: E402
import gol_store as store  # noqa: E402

HEALTH = os.path.join(ROOT, "research", "screen", "health.json")
#: From when a world's links are compared: round 3's averages start there too, after the founders' ring.
SETTLED = 100
#: How many games' links are averaged, so that one game's births do not stop a run.
GAMES = 25


def round3_links() -> Dict[Tuple[str, int], float]:
    """Links an agent of every living world of round 3, by (round, seed)."""
    with open(HEALTH, encoding="utf-8") as f:
        worlds = json.load(f)["worlds"]
    return {(w["round"], w["seed"]): w["averages"]["links"] for w in worlds if w.get("averages", {}).get("links")}


def world_of(condition: str) -> Tuple[str, int]:
    """The round-3 world a condition plays again: 's3042', 's4002 weakest first', 's4006 strongest first'."""
    seed = int(condition[1:5])
    if "weakest" in condition:
        return "E41", seed
    if "strongest" in condition:
        return "E42", seed
    return "E40", seed


def links_now(run_id: str) -> Optional[Tuple[int, float]]:
    """A run's last iteration and its links an agent over its last GAMES games, once it is past SETTLED."""
    if not os.path.exists(os.path.join(store.run_dir(run_id), "stats.jsonl")):
        return None
    games = [r for r in gol_record.read_stats(run_id) if r.get("phase") == 2 and r.get("nodes")]
    games = [r for r in games if r["iteration"] >= SETTLED][-GAMES:]
    if not games:
        return None
    return games[-1]["iteration"], sum(2 * r["edges"] / r["nodes"] for r in games) / len(games)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    before = round3_links()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    runaway = []
    for spec in gol_plan.experiment_runs(a.experiment):
        now = links_now(spec.run_id)
        world = world_of(spec.condition)
        if now is None or world not in before:
            continue
        iteration, links = now
        ratio = links / before[world]
        flag = " RUNAWAY" if ratio > 2 else ""
        print(f"{stamp}  {spec.run_id} ({spec.condition}) at {iteration}: {links:.1f} links an agent, "
              f"{ratio:.2f} times round 3's {before[world]:.1f}{flag}")
        if flag:
            runaway.append(f"{spec.condition} ({spec.run_id}) holds {links:.1f} links an agent at iteration "
                           f"{iteration}, {ratio:.1f} times what its world held in round 3")
    if runaway and not a.dry_run:
        control = gol_lab.read_control()
        if not control.get("paused"):
            gol_lab._pause("links run away: " + "; ".join(runaway) + ". Take the world out of the plan, or "
                           "press play to go on.")
            print(f"{stamp}  paused the lab")


if __name__ == "__main__":
    main()
