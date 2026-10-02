#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advancing a stored run: the one loop everything that runs a simulation shares.

The server steps a run from a thread when you press Start; the lab steps one
from a process of its own when an experiment needs it. Both used to be the
server's Worker, and the lab would have been a second copy of it — with its
own idea of what a resume truncates, when a checkpoint is due, and what a run
that died out is called. Two loops that agree today are two loops that
disagree after the next change to either.

The run is held for as long as it is advanced (gol_store.hold), so nobody else
can step it at the same time, and anybody can ask whether it is going.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

import gol_store as store


def advance(run_id: str, should_stop: Callable[[Any], bool] = lambda world: False,
            until: Optional[int] = None) -> str:
    """
    Step a run from wherever it was left until it is told to stop, reaches
    `until`, or dies out, and return how it ended: "stopped", "extinct" or
    "error".

    A run with no checkpoint starts from nothing, and anything it recorded
    before is thrown away. A run with one resumes from it, and every frame
    recorded past it is deleted first: those describe a future the resumed
    world never lived through, and it is about to live through its own.
    `should_stop` is asked before every iteration and handed the world.

    However it ends, a checkpoint is left at the iteration it ended on, so the
    run can always be picked up again from exactly there. Raises RunBusy if
    someone else is advancing the run; anything else that goes wrong is
    written into the run's metadata before it is raised.
    """
    with store.hold(run_id):
        try:
            return _advance(run_id, should_stop, until)
        except Exception:
            store.update_meta(run_id, status="error", error=traceback.format_exc(limit=4))
            raise


def _advance(run_id: str, should_stop: Callable[[Any], bool], until: Optional[int]) -> str:
    from GraphOfLifeSimple import new_world

    cfg = store.load_config(run_id)
    store.clear_leftovers(run_id)

    world = store.load_checkpoint(run_id, cfg)
    if world is None:
        store.truncate_frames_from(run_id, 0)
        world = new_world(cfg)
        cursor = 0
    else:
        cursor = cfg.frames_before(world.iteration)
        store.truncate_frames_from(run_id, cursor)

    store.update_meta(run_id, status="running", error=None,
                      iteration=world.iteration, frame_count=cursor)

    status = "stopped"
    # No ceiling of its own: a run ends when it is stopped, reaches what it
    # was asked to, or dies out, and nothing else.
    while not should_stop(world) and (until is None or world.iteration < until):
        record = cfg.records(world.iteration)
        frames = world.step(record_decisions=cfg.export_decisions and record)

        if record:
            for frame in frames:
                store.write_frame(run_id, cursor, frame)
                cursor += 1

        if cfg.checkpoint_every and world.iteration % cfg.checkpoint_every == 0:
            store.save_checkpoint(run_id, world)

        store.update_meta(run_id, iteration=world.iteration, frame_count=cursor)

        if world.is_extinct():
            status = "extinct"
            break

    if cfg.checkpoint_every and store.load_meta(run_id).get("checkpoint_iteration") != world.iteration:
        store.save_checkpoint(run_id, world)
    store.update_meta(run_id, status=status, iteration=world.iteration, frame_count=cursor)
    return status
