#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A run's statistics, one line per frame, written while the frame is still in
memory.

The charts summarise a run from its stored frames when they are asked to, into
series.json: a cache that keeps at most a few hundred samples, thins itself as
the run grows, and is thrown away when its format changes. An experiment needs
the opposite — every frame of every run, kept, and ready the moment the run
ends rather than after a pass over tens of gigabytes. So a run advanced with a
recording policy (its metadata's "record") writes stats.jsonl as it goes:

    {"_frame": 0, "_heavy": true, "iteration": 0, "phase": 1, "nodes": 99, ...}

Each row is exactly what gol_series.frame_stats makes of that frame, handed the
frame before it, as the charts' rows are; the graph statistics, which cost many
times the rest, every `heavy_every` iterations; how many families the living
form, which needs every iteration in order; and what the iteration cost, on
its second frame: wall seconds since the one before, processor seconds, and
the most memory the process has held. Those last three are what the lab's
estimates are fitted to.

The same code summarises a run's stored frames afterwards (`record_stored`),
so a run that was never recorded live — any run made before this existed — can
be given the same file.

The file never shares anything with series.json: one is a cache, the other a
record.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gol_series
import gol_store as store

try:
    import resource
except ImportError:          # not POSIX: memory goes unrecorded
    resource = None

STATS_FILE = "stats.jsonl"

#: Iterations between rows that carry the graph statistics.
HEAVY_EVERY = 25


def stats_path(run_id: str) -> str:
    return os.path.join(store.run_dir(run_id), STATS_FILE)


def read_stats(run_id: str) -> List[Dict[str, Any]]:
    """Every complete row recorded, in frame order. A half-written last line is not one."""
    rows: List[Dict[str, Any]] = []
    try:
        with open(stats_path(run_id)) as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def _cut_back(path: str, cursor: int) -> None:
    """Cut the file back to its complete rows from frames before `cursor`. Rows are in frame order."""
    keep = 0
    try:
        with open(path, "rb") as f:
            for line in f:
                if not line.endswith(b"\n") or json.loads(line)["_frame"] >= cursor:
                    break
                keep += len(line)
    except FileNotFoundError:
        return
    os.truncate(path, keep)


def _finite(value: Any) -> Any:
    """NaN and infinity as None, so every line is JSON any reader can parse."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _peak_megabytes() -> Optional[float]:
    if resource is None:
        return None
    # Kilobytes on Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class Recorder:
    """Writes a run's stats.jsonl as its frames are made."""

    def __init__(self, run_id: str, heavy_every: int = HEAVY_EVERY, timed: bool = True) -> None:
        self.run_id = run_id
        self.heavy_every = max(1, int(heavy_every))
        # Off when summarising afterwards: the time it takes to read a frame
        # back says nothing about what it cost to make.
        self.timed = timed
        self.previous: Optional[Dict[str, Any]] = None
        self.families = gol_series._CladeWindow()
        self.file = None
        self.clock = (time.perf_counter(), time.process_time())

    def start(self, cursor: int) -> None:
        """
        Pick up at frame `cursor`, which is where the run's frames now end.

        Rows from frames past it describe iterations that are about to be lived
        again, and a half-written line is the remains of one, so the file is
        cut back to the rows before it. The frame before it and the ancestry
        leading up to it are read back, so the next row is the one a run that
        was never interrupted would have written.
        """
        _cut_back(stats_path(self.run_id), cursor)

        window = gol_series.CLADE_WINDOW * 2
        warm = list(gol_series.stored_frames(self.run_id, max(0, cursor - window), cursor))
        self.families.warm(warm)
        self.previous = warm[-1][1] if warm else None
        self.file = open(stats_path(self.run_id), "a")
        self.clock = (time.perf_counter(), time.process_time())

    def observe(self, frames: Iterable[Tuple[int, Dict[str, Any]]]) -> None:
        """Summarise one iteration's frames, given with their indices, and write them out."""
        lines = []
        for index, frame in frames:
            iteration = int(frame.get("iteration", index // 2))
            heavy = iteration % self.heavy_every == 0
            row = gol_series.frame_stats(frame, self.previous, heavy)
            row["cladesInWindow"] = self.families.count(frame, index)
            row["_frame"] = index
            row["_heavy"] = heavy
            if self.timed and frame.get("phase") == 2:
                wall, cpu = time.perf_counter(), time.process_time()
                row["_seconds"] = round(wall - self.clock[0], 4)
                row["_cpu"] = round(cpu - self.clock[1], 4)
                row["_peakMB"] = _peak_megabytes()
                self.clock = (wall, cpu)
            lines.append(json.dumps(_finite(row), separators=(",", ":")) + "\n")
            self.previous = frame
        self.file.writelines(lines)
        # Into the operating system now; onto the disk with the next
        # checkpoint, which syncs everything before it is written.
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def record_stored(run_id: str, heavy_every: int = HEAVY_EVERY,
                  cancelled: Optional[Callable[[], bool]] = None) -> int:
    """
    Summarise a run's stored frames into stats.jsonl, from wherever the file
    ends, and return how many rows it holds. For runs that were not recorded
    as they ran. Stops at the first frame that will not read, or when asked.
    """
    have = len(read_stats(run_id))
    recorder = Recorder(run_id, heavy_every, timed=False)
    recorder.start(have)
    try:
        frames = gol_series.stored_frames(run_id, have, store.count_frames(run_id))
        for index, frame in frames:
            if cancelled is not None and cancelled():
                break
            recorder.observe([(index, frame)])
    finally:
        recorder.close()
    return len(read_stats(run_id))
