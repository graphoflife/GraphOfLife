#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A world of the pilots' rules, advanced by one process of the lab, on the
frozen engine and prototype beside it.

    python3 -B <snapshot>/gol_pilot.py RUN_ID --until N [--spec JSON]

A run whose settings ask for rules only the research prototype plays
(SimConfig.proto: came_from, contest = lottery, pool, mate_pays, grow, ...) is
not the engine's to make: GraphOfLife refuses it. The lab hands it here
instead, and this plays it exactly as the prototype's own runner does
(research/oee/pilots/run.py, proto.py, grow.py, with the game's rules from
research/oee/checks/rules.py, all frozen into the snapshot beside the
engine): the same world, draw for draw, as a pilot started by hand with the
same settings and seed.

A pilot keeps no frames. Its run folder holds its settings, the prototype's
config.json and series.jsonl — one row of the world's statistics each
iteration — and a whole world at every checkpoint (ckpt_<iteration>.npz).
Stopped on the way, it saves the world where it stands and goes on from
there next time, as the prototype's resume does.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np            # noqa: E402

import gol_store as store     # noqa: E402
import GraphOfLifeSimple      # noqa: E402
from gol_config import SimConfig   # noqa: E402
from gol_worker import EXIT_REFUSED, environment   # noqa: E402

import grow                   # noqa: E402  (research/oee/pilots/grow.py)
import proto                  # noqa: E402  (research/oee/pilots/proto.py)
import run as runner          # noqa: E402  (research/oee/pilots/run.py)

#: How often a pilot health row is written, as the prototype's runner writes it.
HEALTH = 10


def prototype(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    A run's settings as the prototype reads them: its own settings with the proto
    rules taken out, a growing brain named as the prototype names it, and the
    rules as the prototype's switches, in its engine mode.
    """
    rules = dict(config.get("proto") or {})
    cfg = {**config, "proto": {}}
    if rules.pop("grow", False):
        cfg["brain_kind"] = "grow"
    return cfg, {"engine": True, **rules}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--until", type=int, required=True)
    parser.add_argument("--spec", help="how to make the run, as JSON, if it does not exist")
    parser.add_argument("--fault-at", type=int, help="unused: the prototype has no cut of its own")
    args = parser.parse_args()

    for module in (GraphOfLifeSimple, store, proto, grow, runner):
        if os.path.dirname(os.path.abspath(module.__file__)) != HERE:
            print(f"refusing: {module.__name__} comes from {module.__file__}, not {HERE}", file=sys.stderr)
            return EXIT_REFUSED

    run_id = args.run_id
    folder = store.run_dir(run_id)
    if not os.path.exists(os.path.join(folder, "meta.json")):
        spec = json.loads(args.spec)
        made = SimConfig.from_dict(spec["config"], stored=False)
        store.create_run(spec["name"], made, run_id=run_id, record=spec["record"], lab=spec["lab"],
                         pilot=True)
    meta = store.load_meta(run_id)
    config = meta["config"]
    cfg_dict, rules = prototype(config)
    proto.set_proto(rules)
    if cfg_dict.get("brain_kind") == "grow":
        grow.install()

    asked = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: asked.update(stop=True))
    signal.signal(signal.SIGINT, lambda *_: asked.update(stop=True))
    parent = os.getppid()
    every = int(config.get("checkpoint_every") or 100)
    series = os.path.join(folder, "series.jsonl")

    def stop(world) -> bool:
        if world.iteration % HEALTH == 0:
            store.update_meta(run_id, iteration=int(world.iteration), status="running")
        return asked["stop"] or os.getppid() != parent

    started, clock = time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.time()
    with store.hold(run_id):
        have = sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(folder, "ckpt_*.npz")))
        at = have[-1] if have else 0
        if at >= args.until:
            store.update_meta(run_id, iteration=at, checkpoint_iteration=at, status="done")
            return 0
        a = SimpleNamespace(out=folder, iters=args.until - at, ckpt=every, health=HEALTH)
        if have:
            # The prototype's own resume: the world of the last checkpoint, its rows up to it.
            cfg = proto.ProtoConfig.from_dict(cfg_dict, stored=False)
            cfg.extinction_threshold = 0
            with np.load(os.path.join(folder, f"ckpt_{at:06d}.npz"), allow_pickle=False) as z:
                blob = {k: z[k] for k in z.files}
            world = proto.ProtoWorld.from_checkpoint(blob, cfg)
            proto.restore_state(world, blob)
            grow.restore_active(world, blob)
            kept = [line for line in open(series) if json.loads(line)["it"] <= at] if os.path.exists(series) else []
            with open(series, "w") as f:
                f.writelines(kept)
            runner.run_loop(world, a, series, mode="a", first_ckpt_done=True, stop=stop)
        else:
            # Every value as the run's settings give it, as the prototype's runner made a pilot's.
            cfg = proto.ProtoConfig.from_dict(cfg_dict, stored=False)
            cfg.extinction_threshold = 0
            world = proto.make_world(cfg, rules)
            with open(os.path.join(folder, "config.json"), "w") as f:
                json.dump({"cfg": cfg.to_dict(), "proto": rules, "strain": cfg.strain_id(),
                           "iters": args.until, "from": None}, f, indent=1)
            runner.run_loop(world, a, series, mode="w", first_ckpt_done=False, stop=stop)

        reached = int(world.iteration)
        empty = world.G.number_of_nodes() == 0
        sessions = meta.get("sessions", []) + [{
            "started": started, "from": at, "to": reached, "seconds": round(time.time() - clock, 1),
            "environment": environment(), "engine": _engine()}]
        store.update_meta(run_id, iteration=reached, checkpoint_iteration=reached, sessions=sessions,
                          status="extinct" if empty else "done" if reached >= args.until else "paused")
    return 0


def _engine() -> Dict[str, Any]:
    try:
        with open(os.path.join(HERE, "ENGINE.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"hash": None, "commit": None}


if __name__ == "__main__":
    sys.exit(main())
