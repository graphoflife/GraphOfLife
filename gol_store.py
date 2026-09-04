#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-disk layout for GraphOfLife runs.

Everything lives under GraphOfLifeRuns/, one directory per run:

    GraphOfLifeRuns/<run_id>/
        meta.json           run name, status, config, progress
        checkpoint.npz      the single rolling resume point (overwritten)
        frames/
            frame_00000.json.gz     iteration 0, phase 1
            frame_00001.json.gz     iteration 0, phase 2
            frame_00002.json.gz     iteration 1, phase 1
            ...

Frames are numbered sequentially from 0 as they are recorded, so the viewer can
walk them with plain index arithmetic. The iteration and phase a frame belongs
to are stored inside the frame itself. This keeps numbering contiguous even
when `export_every > 1` records only some iterations.

Only ONE checkpoint is kept per run. Resuming from it truncates every frame
recorded after that point, so a run's history always matches its saved state
rather than describing a future the resumed world never lived through.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import threading
import time
from typing import Any, Dict, List

import numpy as np

from gol_config import SimConfig

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GraphOfLifeRuns")

# Allocating a run id means reading the directory and then creating one, and the
# server answers on several threads at once. Two creations landing together
# would pick the same name and the second would move into the first's directory.
_CREATE_LOCK = threading.Lock()

# Run ids are generated, but they end up in filesystem paths, so they are
# validated on every lookup rather than trusted.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _ensure_base() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)


def frames_recorded_before(iteration: int, export_every: int) -> int:
    """
    How many frames exist for iterations strictly below `iteration`.

    Recording happens on iterations where `t % export_every == 0`, two frames
    each. Used on resume to find where the checkpoint's timeline ends.
    """
    exported_iterations = -(-max(0, iteration) // max(1, export_every))  # ceil
    return exported_iterations * 2


def run_dir(run_id: str) -> str:
    """Resolve a run directory, refusing anything that could escape BASE_DIR."""
    if not RUN_ID_RE.match(run_id or ""):
        raise ValueError("invalid run id")
    path = os.path.join(BASE_DIR, run_id)
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(BASE_DIR):
        raise ValueError("invalid run id")
    return path


def frames_dir(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "frames")


FRAME_PREFIX = "frame_"
FRAME_SUFFIX = ".json.gz"


def frame_path(run_id: str, index: int) -> str:
    return os.path.join(frames_dir(run_id),
                        f"{FRAME_PREFIX}{index:05d}{FRAME_SUFFIX}")


def frame_index(name: str) -> int | None:
    """
    The index a frame file name carries, or None if it is not one.

    Read between the fixed prefix and suffix rather than from a fixed slice.
    The name is zero-padded to five digits, but padding is a minimum, not a
    limit: at index 100000 the name grows a digit, and a five-character slice
    read it as 10000. Frames past that point then looked like frames from far
    earlier in the run, so a resume that truncated the discarded timeline
    stepped straight over them and left them in place.
    """
    if not (name.startswith(FRAME_PREFIX) and name.endswith(FRAME_SUFFIX)):
        return None
    digits = name[len(FRAME_PREFIX):-len(FRAME_SUFFIX)]
    return int(digits) if digits.isdigit() else None


def checkpoint_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "checkpoint.npz")


def meta_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "meta.json")


# ----------------------------------------------------------------------------
# Run lifecycle
# ----------------------------------------------------------------------------

def next_run_id() -> str:
    """
    Allocate the next run id of the form GOL_YY_MM_DD_nNNN.

    The counter restarts each day, so ids stay short and sort naturally. It is
    derived from what is already on disk rather than stored, so deleting the
    newest run of the day frees its number again.
    """
    _ensure_base()
    prefix = time.strftime("GOL_%y_%m_%d_n")

    used = []
    for name in os.listdir(BASE_DIR):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            used.append(int(name[len(prefix):]))

    counter = (max(used) + 1) if used else 1
    run_id = f"{prefix}{counter:03d}"
    while os.path.exists(os.path.join(BASE_DIR, run_id)):
        counter += 1
        run_id = f"{prefix}{counter:03d}"
    return run_id


def create_run(name: str, cfg: SimConfig) -> Dict[str, Any]:
    """Allocate a new run directory and write its initial metadata."""
    _ensure_base()

    with _CREATE_LOCK:
        run_id = next_run_id()
        os.makedirs(frames_dir(run_id), exist_ok=False)

    meta = {
        "id": run_id,
        # The id doubles as the default name; a typed name overrides it.
        "name": (name or "").strip() or run_id,
        "created_at": time.time(),
        "status": "idle",
        "iteration": 0,
        "frame_count": 0,
        "checkpoint_iteration": None,
        "error": None,
        "config": cfg.to_dict(),
    }
    save_meta(run_id, meta)
    return meta


def list_runs() -> List[Dict[str, Any]]:
    """Every run on this machine, newest first."""
    _ensure_base()
    runs: List[Dict[str, Any]] = []
    for name in os.listdir(BASE_DIR):
        if not RUN_ID_RE.match(name):
            continue
        if not os.path.isdir(os.path.join(BASE_DIR, name)):
            continue
        try:
            runs.append(load_meta(name))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return runs


def load_meta(run_id: str) -> Dict[str, Any]:
    with open(meta_path(run_id), "r") as f:
        return json.load(f)


def save_meta(run_id: str, meta: Dict[str, Any]) -> None:
    """Write metadata atomically, so a crash mid-write cannot corrupt a run."""
    path = meta_path(run_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)


def update_meta(run_id: str, **changes: Any) -> Dict[str, Any]:
    meta = load_meta(run_id)
    meta.update(changes)
    save_meta(run_id, meta)
    return meta


def load_config(run_id: str) -> SimConfig:
    """A run always continues under the settings it was created with."""
    return SimConfig.from_dict(load_meta(run_id).get("config", {}))


def delete_run(run_id: str) -> None:
    shutil.rmtree(run_dir(run_id), ignore_errors=True)


def run_size_bytes(run_id: str) -> int:
    total = 0
    for root, _, files in os.walk(run_dir(run_id)):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------

def write_frame(run_id: str, index: int, frame: Dict[str, Any]) -> int:
    """Persist one phase frame at a sequential index. Returns that index."""
    path = frame_path(run_id, index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(frame, f, separators=(",", ":"))
    os.replace(tmp, path)
    return index


def read_frame(run_id: str, index: int) -> Dict[str, Any]:
    with gzip.open(frame_path(run_id, index), "rt") as f:
        return json.load(f)


def has_frame(run_id: str, index: int) -> bool:
    return os.path.exists(frame_path(run_id, index))


def count_frames(run_id: str) -> int:
    """Number of contiguous frames starting at index 0."""
    directory = frames_dir(run_id)
    if not os.path.isdir(directory):
        return 0
    indices = set()
    for name in os.listdir(directory):
        index = frame_index(name)
        if index is not None:
            indices.add(index)
    count = 0
    while count in indices:
        count += 1
    return count


def truncate_frames_from(run_id: str, first_index: int) -> int:
    """
    Delete every frame at or after `first_index`.

    Used when resuming: the checkpoint describes the world at some iteration,
    and any frame recorded past that point belongs to a timeline the resumed
    run will now replace.
    """
    directory = frames_dir(run_id)
    if not os.path.isdir(directory):
        return 0

    removed = 0
    for name in os.listdir(directory):
        index = frame_index(name)
        if index is None:
            continue
        if index >= first_index:
            try:
                os.remove(os.path.join(directory, name))
                removed += 1
            except OSError:
                pass
    return removed


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------

def save_checkpoint(run_id: str, world: Any) -> None:
    """
    Overwrite the run's single checkpoint with the current world.

    Written to a temp file and renamed, so an interrupted save never leaves a
    half-written checkpoint that would fail to resume.
    """
    path = checkpoint_path(run_id)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **world.to_checkpoint())
    os.replace(tmp, path)
    update_meta(run_id, checkpoint_iteration=world.iteration)


def load_checkpoint(run_id: str, cfg: SimConfig):
    """Rebuild the world from this run's checkpoint, or None if there is none."""
    from GraphOfLifeSimple import GraphOfLife

    path = checkpoint_path(run_id)
    if not os.path.exists(path):
        return None
    with np.load(path) as blob:
        return GraphOfLife.from_checkpoint(blob, cfg)


def has_checkpoint(run_id: str) -> bool:
    return os.path.exists(checkpoint_path(run_id))
