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
record. Both are kept here; gol_series only computes rows, and touches no disk.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

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
        self.families = gol_series.CladeWindow()
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
        warm = list(stored_frames(self.run_id, max(0, cursor - window), cursor))
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
        frames = stored_frames(run_id, have, store.count_frames(run_id))
        for index, frame in frames:
            if cancelled is not None and cancelled():
                break
            recorder.observe([(index, frame)])
    finally:
        recorder.close()
    return len(read_stats(run_id))


# ----------------------------------------------------------------------------
# The charts' cache: series.json
# ----------------------------------------------------------------------------
#
# The viewer charts a run from a summary of its stored frames, kept in the run
# directory as series.json and extended incrementally: only frames added since
# the last request are read. It is keyed by gol_series.SERIES_VERSION, so a
# changed formula invalidates it rather than mixing old numbers with new.

# Progress of in-flight builds, so the browser can show how far along a rebuild
# is instead of sitting on a blank wait. Reads happen on a different thread from
# the build, since the server handles each request in its own.
_PROGRESS: Dict[str, Dict[str, Any]] = {}
_PROGRESS_LOCK = threading.Lock()
_BUILD_LOCKS: Dict[str, threading.Lock] = {}


def _build_lock(run_id: str) -> threading.Lock:
    """One lock per run, so two callers do not rebuild the same series twice."""
    with _PROGRESS_LOCK:
        lock = _BUILD_LOCKS.get(run_id)
        if lock is None:
            lock = _BUILD_LOCKS[run_id] = threading.Lock()
        return lock


def _set_progress(run_id: str, done: int, total: int, building: bool = True) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS[run_id] = {"building": building, "done": done, "total": total}


def series_progress(run_id: str) -> Dict[str, Any]:
    """How far a build has got, for the progress bar."""
    with _PROGRESS_LOCK:
        state = _PROGRESS.get(run_id)
    return dict(state) if state else {"building": False, "done": 0, "total": 0}


def stored_frames(run_id: str, start: int, stop: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Stored frames `start` to `stop` with their indices, ending at the first that will not read."""
    for index in range(start, stop):
        try:
            yield index, store.read_frame(run_id, index)
        except (OSError, json.JSONDecodeError, KeyError):
            return


def _cache_path(run_id: str) -> str:
    return os.path.join(store.run_dir(run_id), "series.json")


def _load_cache(run_id: str) -> Dict[str, Any]:
    try:
        with open(_cache_path(run_id), "r") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": gol_series.SERIES_VERSION, "rows": []}

    if cache.get("version") != gol_series.SERIES_VERSION:
        return {"version": gol_series.SERIES_VERSION, "rows": []}
    return cache


def _save_cache(run_id: str, cache: Dict[str, Any]) -> None:
    try:
        store.write_json(_cache_path(run_id), cache)
    except OSError:
        pass  # a missing cache only costs time, never correctness


def build_series(run_id: str, points: Optional[int] = None,
                 heavy: bool = True,
                 cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """
    Statistics for a run's history, as parallel arrays.

    Frames already summarised are reused; only new ones are read. If the run was
    resumed and its history truncated, the cache is trimmed to match rather than
    describing frames that no longer exist.

    `points` asks for a coarser answer: the first `points` samples in bisection
    order, spread across the whole run. None means all of them. Summarising one
    large frame costs well over a second, so a full history is minutes of work,
    and a caller that waits for it has nothing to show for that whole time. A
    caller that climbs — two points, three, five, nine — has a chart of the
    entire run within a second and refines it, and pays no more in total,
    because each request only computes what the last one did not.

    `heavy` is the other axis of the same idea. Five sixths of what a frame
    costs to summarise goes on statistics that walk the graph, and most charts
    plot none of them — so a caller that plots none gets a complete chart of
    the run in a sixth of the time. A row already stored light is recomputed
    when a heavy request reaches it; one already heavy is never recomputed.

    Whatever was asked, the reply is everything known of the run — see
    History.reply.
    """
    with _build_lock(run_id):
        try:
            return _build_series_locked(run_id, points, heavy, cancelled)
        finally:
            _set_progress(run_id, 0, 0, building=False)


def _build_series_locked(run_id: str, points: Optional[int] = None,
                         heavy: bool = True,
                         cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    cache = _load_cache(run_id)
    history = gol_series.History(cache.get("rows", []), cache.get("stride", 1))
    wanted = history.plan(store.count_frames(run_id), points, heavy)

    every = 1
    try:
        every = max(1, int(store.load_meta(run_id).get("config", {}).get("export_every", 1)))
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    if wanted:
        _set_progress(run_id, 0, len(wanted), building=True)

    # How many families the living divide into needs ancestry, and ancestry is
    # a chain: it cannot be read off one frame and it cannot be sampled. So it
    # is computed here rather than in frame_stats, and only where the chain is
    # whole — every iteration recorded, none of them thinned away, and a
    # request for the whole grid. A coarse request simply leaves the key off
    # rather than filling it from a broken chain.
    families = gol_series.CladeWindow() if (every == 1 and history.stride == 1 and history.whole) else None
    if families is not None and wanted:
        # Resuming mid-run leaves the window empty, so the frames just before
        # the first new one are read to fill it. Their statistics are already
        # cached; only their ancestry is wanted.
        families.warm(stored_frames(run_id, max(0, wanted[0] - gol_series.CLADE_WINDOW * 2), wanted[0]))

    def frames():
        for index in wanted:
            # A build nobody is waiting for any more stops here, and the rows
            # already computed are still saved below. A summary is incremental:
            # the next request builds on whatever this one finished, so
            # stopping loses nothing but the frame in hand. Asked before every
            # frame: the question is a peek at a socket, and a frame of a large
            # run costs seconds once the graph statistics are on.
            if cancelled is not None and cancelled():
                return
            try:
                frame = store.read_frame(run_id, index)
            except (OSError, json.JSONDecodeError, KeyError):
                return
            yield index, frame

    summarised = 0

    def each(index: int, frame: Dict[str, Any], row: Dict[str, Any]) -> None:
        nonlocal summarised
        if families is not None:
            row["cladesInWindow"] = families.count(frame, index)
        # Every frame. Throttling this to every tenth was sized for the cheap
        # statistics; with the graph statistics a frame of a large run takes
        # two seconds, and a step of sixteen frames then reported twice.
        summarised += 1
        _set_progress(run_id, summarised, len(wanted), building=True)

    history.summarise(frames(), heavy, each)

    # The strain travels with the summary as well as with the run, because a
    # series.json is the file most likely to be read on its own — it is the one
    # an analysis loads, and a chart made from it should not have to go back to
    # the run directory to find out which algorithm it is of.
    if history.changed:
        _save_cache(run_id, {"version": gol_series.SERIES_VERSION, "stride": history.stride,
                             "strain": store.load_meta(run_id).get("strain"),
                             "rows": history.rows})
    return history.reply(heavy)
