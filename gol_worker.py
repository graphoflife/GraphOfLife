#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One run, advanced by one process of the lab, on the frozen engine beside it.

    python3 -B <snapshot>/gol_worker.py RUN_ID --until N [--spec JSON] [--fault-at N]

The lab copies this file into each snapshot of the engine it takes
(gol_lab.ENGINE_FILES), and starts it from there: so a run is only ever made
and continued by the code it was started on, whatever has been changed in the
repository since. The command line above is what every snapshot ever taken is
started with, so it does not change.

A run that does not exist yet is made from `--spec`. Each session — one start
of this process — is written into the run's provenance.json: the Python and
libraries, the CPU and how numpy dispatches on it, the threads, the engine and
commit, which iterations it ran, what that cost, and why it ended. A run will
not continue under libraries or a processor other than the ones it started
on, since a last-bit difference in a sum becomes a different history.

SIGTERM or SIGINT ends the session after the iteration in hand, with a
checkpoint, which is how the lab pauses. So does the lab itself dying.
`--fault-at` ends it at that iteration without a checkpoint, the way a power
cut would; that is how the reproducibility experiment cuts a run.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import sys
import time
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))

import networkx as nx          # noqa: E402
import numpy as np             # noqa: E402

import GraphOfLifeSimple       # noqa: E402
import gol_record              # noqa: E402
import gol_run                 # noqa: E402
import gol_store as store      # noqa: E402
from gol_config import SimConfig   # noqa: E402

#: How a session can end, as its exit code.
EXIT_FAULT = 3
EXIT_REFUSED = 4


def _blas() -> str:
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{blas.get('name')} {blas.get('version')}"
    except Exception:          # noqa: BLE001 - a numpy without the dict form
        return "unknown"


def _cpu() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _dispatch() -> str:
    """The highest instruction set numpy picks its kernels from on this processor."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:
        return "unknown"
    found = [name for name in ("AVX512_SPR", "AVX512_ICL", "AVX512_SKX", "AVX512F",
                               "AVX2", "AVX", "SSE42", "ASIMD", "NEON") if features.get(name)]
    return found[0] if found else "baseline"


def environment() -> Dict[str, Any]:
    """Everything about this machine and its libraries a run's history could depend on."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "blas": _blas(),
        "dispatch": _dispatch(),
        "cpu": _cpu(),
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "threads": {name: os.environ.get(name) for name in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


#: What has to match for a run to continue: the things that decide its arithmetic.
FINGERPRINT = ("python", "numpy", "networkx", "blas", "dispatch", "cpu")


def provenance_path(run_id: str) -> str:
    return os.path.join(store.run_dir(run_id), "provenance.json")


def read_provenance(run_id: str) -> Dict[str, Any]:
    try:
        with open(provenance_path(run_id)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"run": run_id, "sessions": []}


def _engine() -> Dict[str, Any]:
    try:
        with open(os.path.join(HERE, "ENGINE.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"hash": None, "commit": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--until", type=int, required=True)
    parser.add_argument("--spec", help="how to make the run, as JSON, if it does not exist")
    parser.add_argument("--fault-at", type=int, help="end at this iteration with no checkpoint")
    args = parser.parse_args()

    # The engine has to be the copy this file sits in. A path that led back to
    # the repository would run whatever is there today under the name of a
    # snapshot taken weeks ago.
    for module in (GraphOfLifeSimple, gol_record, gol_run, store):
        if os.path.dirname(os.path.abspath(module.__file__)) != HERE:
            print(f"refusing: {module.__name__} comes from {module.__file__}, not {HERE}",
                  file=sys.stderr)
            return EXIT_REFUSED

    run_id = args.run_id
    if not os.path.exists(os.path.join(store.run_dir(run_id), "meta.json")):
        spec = json.loads(args.spec)
        cfg = SimConfig.from_dict(spec["config"], stored=False)
        store.create_run(spec["name"], cfg, run_id=run_id,
                         record=spec["record"], lab=spec["lab"])

    env = environment()
    provenance = read_provenance(run_id)
    sessions = provenance["sessions"]
    if sessions:
        first = sessions[0]["environment"]
        changed = {k: (first.get(k), env[k]) for k in FINGERPRINT if first.get(k) != env[k]}
        if changed:
            reason = "; ".join(f"{k} {a} → {b}" for k, (a, b) in changed.items())
            store.update_meta(run_id, lab_blocked=f"the environment changed: {reason}")
            print(f"refusing to continue {run_id}: {reason}", file=sys.stderr)
            return EXIT_REFUSED

    started_at = store.load_meta(run_id).get("iteration", 0)
    session = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(),
               "from": started_at, "until": args.until, "engine": _engine(),
               "environment": env}
    sessions.append(session)
    store.write_json(provenance_path(run_id), provenance, indent=1)
    clock = (time.time(), time.process_time())

    def close(ending: str) -> None:
        session.update(ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"), exit=ending,
                       to=store.load_meta(run_id).get("iteration"),
                       seconds=round(time.time() - clock[0], 1),
                       cpu=round(time.process_time() - clock[1], 1),
                       peakMB=gol_record._peak_megabytes())
        store.write_json(provenance_path(run_id), provenance, indent=1)

    asked = {"stop": False}

    def stop(_signum, _frame) -> None:
        asked["stop"] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parent = os.getppid()

    def should_stop(world) -> bool:
        if args.fault_at is not None and world.iteration >= args.fault_at:
            close("fault")
            os._exit(EXIT_FAULT)
        return asked["stop"] or os.getppid() != parent

    try:
        status = gol_run.advance(run_id, should_stop, args.until)
    except Exception:
        close("error")
        raise
    reached = store.load_meta(run_id).get("iteration", 0) >= args.until
    close("extinct" if status == "extinct" else "target" if reached
          else "orphaned" if os.getppid() != parent else "paused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
