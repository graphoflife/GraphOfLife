"""
Move the finished runs of the experiments named to another disk while the lab
plays (the author, 2026-11-09: "set up the automatic mover to Speicher2"),
leaving a link in each one's place: what research/tools/move_runs.sh does with
the lab stopped, done for the runs the lab is finished with, while it goes on.

A run is moved only when nothing will write to it again: the lab's own test
says its plan has nothing left for it (it died out, was stopped as too slow, or
reached its last target), and the mover holds the run's own lock, which every
worker holds while it advances a run. The run is copied, the copy checked file
by file (the same files, the same bytes), and then the folder and a link to the
copy trade places in one step (renameat2 with RENAME_EXCHANGE). So the run's
path is never empty: the lab, which may look at any run at any moment, finds the
folder or the link, and never takes the run for one still to be made. The folder
left behind is then removed, and where the run stands is noted in the lab's own
folder (gol_lab.remember_moved), so that it counts as done while the drive is
away. A move cut short is finished or undone the next time (`tidy`).

Nothing is copied unless the destination is a folder on another disk that is
there now: an unplugged drive leaves an empty folder on this disk, and copying
into it would only fill this one.

    python3 research/tools/move_finished_runs.py DEST E43 [E44 ...] [--dry-run]
        moves every finished run of the experiments named still on this disk; cron
        runs it every ten minutes for Experiment 43, writing what it moved to
        GraphOfLifeRuns/.lab/moved.log (`crontab -l` shows the line)
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import filecmp
import os
import shutil
import sys
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import gol_lab  # noqa: E402
import gol_plan  # noqa: E402
import gol_store as store  # noqa: E402

#: Room the destination keeps free beyond the run being copied to it.
MARGIN = 2 * 2**30

_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_AT_FDCWD, _RENAME_EXCHANGE = -100, 2


def exchange(a: str, b: str) -> None:
    """Swap two paths of one filesystem in one step, so that neither is ever missing."""
    if _LIBC.renameat2(_AT_FDCWD, os.fsencode(a), _AT_FDCWD, os.fsencode(b), _RENAME_EXCHANGE) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), a)


def _files(top: str) -> List[str]:
    return sorted(os.path.relpath(os.path.join(d, f), top) for d, _, names in os.walk(top) for f in names)


def same(a: str, b: str) -> bool:
    """Whether two folders hold the same files, byte for byte."""
    have = _files(a)
    return have == _files(b) and all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)
                                     for f in have)


def _aside(folder: str) -> str:
    """Where a run's folder stands while it and its link trade places, and after, until it is removed."""
    return folder + ".moving"


def tidy(folder: str) -> None:
    """Finish or undo a move cut short: a link never swapped in, or a folder swapped out but not removed."""
    aside = _aside(folder)
    if os.path.islink(aside):
        os.remove(aside)
    elif os.path.isdir(aside) and os.path.islink(folder) and os.path.isdir(folder):
        shutil.rmtree(aside)


def move(folder: str, copy: str) -> None:
    """
    Copy a run's folder to `copy`, check the copy, and put a link to it in the folder's place, the folder and the
    link trading places in one step; the folder, now aside, is removed.
    """
    shutil.copytree(folder, copy, symlinks=True, dirs_exist_ok=True)
    if not same(folder, copy):
        raise OSError(f"the copy of {folder} at {copy} differs from it")
    aside = _aside(folder)
    os.symlink(copy, aside)
    exchange(aside, folder)
    if not (os.path.islink(folder) and os.path.realpath(folder) == os.path.realpath(copy)) or os.path.islink(aside):
        raise OSError(f"{folder} did not trade places with its link")
    shutil.rmtree(aside)


def finished(spec: gol_plan.RunSpec) -> bool:
    """Whether a run exists on this disk and its plan has nothing left for it: the lab's own test."""
    folder = store.run_dir(spec.run_id)
    if os.path.islink(folder) or not os.path.exists(os.path.join(folder, "meta.json")):
        return False
    meta = store.load_meta(spec.run_id)
    return gol_lab._done({"status": meta.get("status"), "checkpoint": meta.get("checkpoint_iteration") or 0}, spec)


def size(folder: str) -> int:
    return sum(os.path.getsize(os.path.join(folder, f)) for f in _files(folder))


def move_finished(spec: gol_plan.RunSpec, dest: str) -> Optional[str]:
    """Move one run if it is finished and nobody is advancing it; say what became of it, or None if nothing did."""
    folder = store.run_dir(spec.run_id)
    tidy(folder)
    if not finished(spec):
        return None
    try:
        with store.hold(spec.run_id):
            need = size(folder)
            if shutil.disk_usage(dest).free < need + MARGIN:
                return f"{spec.run_id}: no room for its {need / 2**30:.1f} GB on {dest}, left in place"
            move(folder, os.path.join(dest, spec.run_id))
    except store.RunBusy:
        return f"{spec.run_id}: being played, left in place"
    except OSError as exc:
        return f"{spec.run_id}: {exc}, left in place"
    gol_lab.remember_moved([spec.run_id])
    return f"{spec.run_id}: moved ({need / 2**30:.1f} GB)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dest", help="the runs folder on the other disk, e.g. /media/stefan/Speicher2/GraphOfLifeRuns")
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="only say which runs would be moved")
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    if not os.path.isdir(dest) or os.stat(dest).st_dev == os.stat(store.BASE_DIR).st_dev:
        sys.exit(f"{dest} is not a folder on another disk now: is the drive plugged in?")
    os.makedirs(gol_lab.lab_dir(), exist_ok=True)
    with open(os.path.join(gol_lab.lab_dir(), "mover.lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return                      # the mover started ten minutes ago is still at work
        for name in a.experiments:
            for spec in gol_plan.experiment_runs(name):
                said = (f"{spec.run_id}: would be moved" if finished(spec) else None) if a.dry_run \
                    else move_finished(spec, dest)
                if said:
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), said, flush=True)


if __name__ == "__main__":
    main()
