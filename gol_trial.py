#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One trial, run by one process of the lab, on the frozen engine beside it.

    python3 -B <snapshot>/gol_trial.py TRIAL_ID --until T [--spec JSON]

A trial is short, and starts from a world another simulation kept: a lab run
in its archive, or a pilot of the engine's own rules in its checkpoint
(`pilot:<batch>/<world>`, gol_store.kept_world):

  contest    the world of one era, its agents given brains drawn by coin from
             two eras, with mutation off (gol_tournament.contest); its result
             is the later era's share of the agents after T iterations
  invasion   the world of one era, a patch of it given a strategy written by
             hand (gol_strategies.plant), after the whole world was given
             another strategy and left to settle if the trial says so; its
             result is the strategy's agents and tokens after every iteration

The command line is the worker's (gol_worker.py), so the lab starts either the
same way. A trial keeps no frames. Its run folder holds its settings, how far
it has got, and at the end `result.json`, and its metadata reaches its last
iteration only once that is written, so the lab counts a trial as done exactly
when it has a result. Stopped on the way — the lab pausing — it leaves no
result and starts again from its beginning next time: a trial is minutes long.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np            # noqa: E402

import gol_store as store     # noqa: E402
import gol_strategies         # noqa: E402
import gol_tournament         # noqa: E402
import GraphOfLifeSimple      # noqa: E402
from gol_config import SimConfig   # noqa: E402
from gol_worker import EXIT_REFUSED, environment   # noqa: E402

#: How often a trial says how far it has got.
EVERY = 10


def contest(trial: Dict[str, Any], stop: Callable[[Any, int], bool]) -> Optional[Dict[str, Any]]:
    source = trial["run"]
    return gol_tournament.contest(store.source_config(source), store.kept_world(source, trial["home"]),
                                  store.kept_world(source, trial["a"]), store.kept_world(source, trial["b"]),
                                  int(trial["T"]), int(trial["seed"]), on_step=stop)


def invasion(trial: Dict[str, Any], stop: Callable[[Any, int], bool]) -> Optional[Dict[str, Any]]:
    """Chapter 51's invasion test, with the patch's strategy counted after every iteration."""
    cfg = SimConfig.from_dict(store.source_config(trial["run"]))
    cfg.extinction_threshold = 0
    world = GraphOfLifeSimple.GraphOfLife.from_checkpoint(store.kept_world(trial["run"], int(trial["era"])), cfg)
    world.rng = np.random.RandomState(int(trial["place"]))
    background = trial.get("background")
    if background:
        gol_strategies.fill(world, background)
        for t in range(int(trial.get("settle", 0))):
            world.step(record_decisions=False)
            if world.G.number_of_nodes() == 0:
                return {"series": [], "patch": 0, "died_settling": t + 1}
    patch = gol_strategies.plant(world, trial["strategy"], float(trial["size"]), int(trial["place"]))
    mark = gol_strategies.STRATEGY_IDS[trial["strategy"]]
    series: List[Dict[str, int]] = []
    for t in range(int(trial["T"]) + 1):
        if t:
            world.step(record_decisions=False)
            if stop(world, t):
                return None
        ids = list(world.G.nodes())
        mine = [u for u in ids if world.brains[u].brain_id == mark]
        series.append({"t": t, "agents": len(ids), "mine": len(mine),
                       "tokens": int(sum(world.tokens[u] for u in mine))})
        if not mine or not ids:
            break
    return {"series": series, "patch": len(patch)}


PLAY = {"contest": contest, "invasion": invasion}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--until", type=int, required=True)
    parser.add_argument("--spec", help="how to make the trial, as JSON, if it does not exist")
    parser.add_argument("--fault-at", type=int, help="unused: a trial has no checkpoint to cut")
    args = parser.parse_args()

    # The engine has to be the copy this file sits in, as for the worker.
    for module in (GraphOfLifeSimple, gol_strategies, gol_tournament, store):
        if os.path.dirname(os.path.abspath(module.__file__)) != HERE:
            print(f"refusing: {module.__name__} comes from {module.__file__}, not {HERE}",
                  file=sys.stderr)
            return EXIT_REFUSED

    run_id = args.run_id
    if not os.path.exists(os.path.join(store.run_dir(run_id), "meta.json")):
        spec = json.loads(args.spec)
        gol_strategies.check_trial(spec["trial"])
        cfg = SimConfig.from_dict(spec["config"], stored=False)
        store.create_run(spec["name"], cfg, run_id=run_id, record=spec["record"], lab=spec["lab"],
                         trial=spec["trial"])
    meta = store.load_meta(run_id)
    trial = meta["trial"]
    try:
        gol_strategies.check_trial(trial)
    except ValueError as exc:
        store.update_meta(run_id, lab_blocked=str(exc))
        print(f"refusing {run_id}: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    asked = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: asked.update(stop=True))
    signal.signal(signal.SIGINT, lambda *_: asked.update(stop=True))
    parent = os.getppid()

    def stop(world, t: int) -> bool:
        if t % EVERY == 0:
            store.update_meta(run_id, iteration=t, status="running")
        return asked["stop"] or os.getppid() != parent

    started = time.time()
    with store.hold(run_id):
        store.update_meta(run_id, iteration=0, status="running", checkpoint_iteration=0)
        try:
            result = PLAY[trial["kind"]](trial, stop)
        except ValueError as exc:
            store.update_meta(run_id, lab_blocked=str(exc), status="idle")
            print(f"refusing {run_id}: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        if result is None:
            store.update_meta(run_id, iteration=0, status="idle")
            return 0
        engine = {}
        if os.path.exists(os.path.join(HERE, "ENGINE.json")):
            with open(os.path.join(HERE, "ENGINE.json")) as f:
                engine = json.load(f)
        store.write_json(os.path.join(store.run_dir(run_id), "result.json"),
                         {"trial": trial, "result": result, "engine": engine,
                          "seconds": round(time.time() - started, 1), "environment": environment()},
                         indent=1)
        store.update_meta(run_id, iteration=args.until, checkpoint_iteration=args.until, status="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
