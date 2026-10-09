#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The lab: runs the simulations the book's experiments ask for, unattended.

An experiment is a plan in book/experiments/ — a baseline, a world size, the
conditions that differ from the baseline one setting at a time, seeds and a
length. The lab works out which runs that means, makes the ones that do not
exist yet, and advances them, several at once, each in a process of its own,
until every queued experiment has what it asked for. You can pause it and
start it again at any point, from the Book tab or from here:

    python3 gol_lab.py plan E02       what E02 needs, what exists, what it will cost
    python3 gol_lab.py run E02        queue it, and run the lab in this terminal
    python3 gol_lab.py pause          ask a running lab to pause
    python3 gol_lab.py status         where everything stands
    python3 gol_lab.py verify E02     re-run a few of its runs and compare
    python3 gol_lab.py analyse E02    write its results and figures into the book
    python3 gol_lab.py costs          refit what a simulation costs, for the book too

Three things about it are deliberate.

A run is named for what it is — baseline, world size, what differs, seed — and
not for who asked for it, so two experiments that need the same run share it,
and a run is taken as far as the furthest of them asks.

A run is only ever made and advanced by a frozen copy of the engine
(.lab/engines/<hash>/, taken when it was first needed), so nothing changed in
the repository afterwards can mix into a run that has started. Each copy notes
the commit it was taken from, and none is taken from uncommitted engine code.
An experiment whose runs come from more than one copy says so, and its analysis
first checks that the newest copy reproduces the older ones' runs.

What it costs is measured, not guessed: every recorded iteration carries its
seconds and the process's peak memory (gol_record), and the estimates — time,
memory, disk — are fitted to those.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gol_store as store
import gol_worker
# How a worker's session can end: the worker owns these, and the command line
# that starts it, because every snapshot of it ever taken answers to them.
from gol_worker import EXIT_FAULT, EXIT_REFUSED
import gol_plan
from gol_config import SimConfig
# The plans, which the lab carries out and the analysis and the book read too.
from gol_plan import (WORLD, LabError, RunSpec, baseline, experiment_runs, experiments, read_plan,
                      wanted, what_it_is, workers_cap)

HERE = os.path.dirname(os.path.abspath(__file__))

#: What a run is made and advanced by, copied whole into each snapshot.
ENGINE_FILES = ("GraphOfLifeSimple.py", "gol_config.py", "gol_series.py", "gol_spectral.py",
                "gol_lightning.py", "gol_store.py", "gol_run.py", "gol_record.py",
                "gol_worker.py")

DEFAULT_WORKERS = 4
SETTLED = 100                     # iterations before a world's size says anything
CHECKPOINT_SECONDS = 600          # about how often a run saves itself
DISK_MARGIN = 10 * 2**30          # never fill the disk closer than this
MEMORY_SHARE = 0.75               # of the machine's memory, for all workers together

#: What a simulation costs before anything has been measured: the calibration
#: of 2026-10-02 on the baseline B1 (Core Ultra 7 258V, numpy 2.5.1) — about
#: 0.95 ms an agent in memory with 15,795 weights, and more in the lab, which
#: also writes every frame and its statistics.
CALIBRATION = {
    "secondsPerAgentIteration": 0.85e-3,    # per 10,000 weights in a brain
    "agentsPerToken": 0.18,
    "bytesPerAgentIteration": 70.0,
    "peakBytesPerWeightByte": 3.0,          # the Blotto copy and the checkpoint stack
    "baseMB": 300.0,
}


def lab_dir() -> str:
    return os.path.join(store.BASE_DIR, ".lab")


# ----------------------------------------------------------------------------
# Frozen engines
# ----------------------------------------------------------------------------

def engine_hash(directory: str = HERE) -> str:
    digest = hashlib.sha256()
    for name in sorted(ENGINE_FILES):
        with open(os.path.join(directory, name), "rb") as f:
            digest.update(name.encode() + b"\0" + f.read() + b"\0")
    return digest.hexdigest()[:16]


def engine_dir(engine: str) -> str:
    return os.path.join(lab_dir(), "engines", engine)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                          check=True).stdout.strip()


def snapshot() -> str:
    """
    The engine as it is now, frozen where the lab's workers will run it, and
    its hash. Refused while any of its files differs from the last commit,
    so the commit a snapshot names is the code it holds.
    """
    engine = engine_hash()
    target = engine_dir(engine)
    if os.path.isdir(target):
        return engine
    dirty = _git("status", "--porcelain", "--", *ENGINE_FILES)
    if dirty:
        raise LabError("new runs need the engine committed, so that a run can be traced to "
                       "a commit, and these differ from it: "
                       + ", ".join(line[3:] for line in dirty.splitlines()))
    staging = tempfile.mkdtemp(prefix=f"{engine}.", dir=_made(os.path.dirname(target)))
    for name in ENGINE_FILES:
        shutil.copy2(os.path.join(HERE, name), staging)
    store.write_json(os.path.join(staging, "ENGINE.json"),
                     {"hash": engine, "commit": _git("rev-parse", "HEAD"),
                      "taken": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "files": list(ENGINE_FILES)}, indent=1)
    os.replace(staging, target)
    return engine


def _made(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------------
# Comparing what two runs recorded
# ----------------------------------------------------------------------------

def _canonical_frames(run_id: str, count: Optional[int] = None) -> List[str]:
    total = store.count_frames(run_id) if count is None else count
    return [json.dumps(store.read_frame(run_id, i), sort_keys=True) for i in range(total)]


def _stats_without_costs(run_id: str, count: Optional[int] = None) -> List[Dict[str, Any]]:
    import gol_record
    rows = gol_record.read_stats(run_id)[:count]
    return [{k: v for k, v in r.items() if k not in ("_seconds", "_cpu", "_peakMB")}
            for r in rows]


def _checkpoint_arrays(run_id: str) -> Dict[str, bytes]:
    import numpy as np
    with np.load(store.checkpoint_path(run_id)) as blob:
        return {key: blob[key].tobytes() for key in blob.files}


def compare(a: Tuple[str, str], b: Tuple[str, str], frames: Optional[int] = None,
            checkpoints: bool = True) -> Dict[str, Any]:
    """
    Whether two runs recorded the same history: every frame as canonical JSON,
    every statistics row without what it cost, and the final checkpoints array
    by array, the random stream included. Each run is (runs folder, run id).
    The files themselves cannot be compared: gzip and zip stamp them with the
    time. Says where they first part, if they do.
    """
    answer: Dict[str, Any] = {"same": True}
    original = store.BASE_DIR
    try:
        seen = []
        for folder, run_id in (a, b):
            store.BASE_DIR = folder
            seen.append((_canonical_frames(run_id, frames), _stats_without_costs(run_id, frames),
                         _checkpoint_arrays(run_id) if checkpoints else {}))
    finally:
        store.BASE_DIR = original
    (fa, sa, ca), (fb, sb, cb) = seen
    answer["frames"] = len(fa)
    if fa != fb:
        answer["same"] = False
        answer["firstFrame"] = next((i for i, (x, y) in enumerate(zip(fa, fb)) if x != y),
                                    min(len(fa), len(fb)))
    if sa != sb:
        answer["same"] = False
        answer["firstRow"] = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y),
                                  min(len(sa), len(sb)))
    if ca != cb:
        answer["same"] = False
        answer["checkpointArrays"] = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
    return answer


def reproduce(run_id: str, engine: str, iterations: int) -> Dict[str, Any]:
    """
    Make a run again from its seed, on `engine`, in a folder of its own and a
    fresh process, for its first `iterations`, and compare it with the stored
    one. The question a run asks of an engine before trusting it with more.
    """
    meta = store.load_meta(run_id)
    spec = {"name": meta["name"], "config": meta["config"], "record": meta.get("record"),
            "lab": meta.get("lab")}
    with tempfile.TemporaryDirectory(prefix="gol-reproduce-") as folder:
        env = {**os.environ, "GOL_RUNS_DIR": folder, **_thread_settings(None)}
        subprocess.run(gol_worker.command(engine_dir(engine), run_id, iterations, spec=spec),
                       cwd=engine_dir(engine), env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return compare((store.BASE_DIR, run_id), (folder, run_id),
                       frames=2 * iterations, checkpoints=False)


# ----------------------------------------------------------------------------
# What a simulation costs
# ----------------------------------------------------------------------------

def _tail_rows(path: str, count: int = 400) -> List[Dict[str, Any]]:
    """The last complete rows of a stats file, read from its end."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - count * 2500))
            lines = f.read().split(b"\n")
    except FileNotFoundError:
        return []
    rows = []
    for line in lines[1 if size > count * 2500 else 0:]:
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _weights(config: Dict[str, Any]) -> Tuple[int, int]:
    from GraphOfLifeSimple import brain_shape
    shape = brain_shape(SimConfig.from_dict(config))
    return shape["weights"], shape["bytesPerWeight"]


def _median(values: List[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def kind_of(config: Dict[str, Any]) -> str:
    """
    Which kind of run a configuration makes, for what it costs: everything but
    the size of the world, the seed and what is recorded. Kinds differ in
    more than the size of their brains — B1's brains are half as big again as
    the ones before it and cost an agent hardly any more, because its agents
    have fewer neighbours to look at — so each is fitted on its own.
    """
    what = {k: v for k, v in what_it_is(config).items() if k not in WORLD and k != "seed"}
    return hashlib.sha1(json.dumps(what, sort_keys=True).encode()).hexdigest()[:10]


def fit_costs() -> Dict[str, Any]:
    """
    Fit what a simulation costs to every run the lab has recorded, and keep it
    in .lab/costs.json: overall, and for each kind of run. Each measure falls
    back from the kind to all runs to the calibration, until something has
    measured it.
    """
    seconds, agent_iterations, per_token, disk, peak = 0.0, 0.0, [], [], []
    kinds: Dict[str, Dict[str, Any]] = {}
    runs = 0
    for meta in store.list_runs():
        if "lab" not in meta:
            continue
        import gol_record
        rows = _tail_rows(gol_record.stats_path(meta["id"]))
        timed = [r for r in rows if r.get("_seconds") and r.get("nodes")]
        if not timed:
            continue
        runs += 1
        weights, width = _weights(meta["config"])
        # Total time over total work, not the typical iteration: a run spends
        # most of its time when it is biggest, and that is what an estimate of
        # its length has to get right.
        spent = sum(r["_seconds"] for r in timed)
        seconds += spent
        agent_iterations += sum(r["nodes"] * weights / 1e4 for r in timed)
        # Only after the founding boom: a world's first iterations are
        # nothing like the population it settles at, and short runs would
        # otherwise teach the estimates that worlds stay small.
        tokens = meta["config"]["total_tokens"]
        settled = [r["nodes"] / tokens for r in rows if r.get("iteration", 0) >= SETTLED]
        per_token += settled
        config = meta["config"]
        kind = kinds.setdefault(kind_of(config), {
            "seconds": 0.0, "agentIterations": 0, "perToken": [], "runs": 0,
            "label": (f"{meta.get('strain')}, {config['message_amount']}-number messages, "
                      f"mutation {config['mutation_probability']}, {weights:,} weights")})
        kind["seconds"] += spent
        kind["agentIterations"] += sum(r["nodes"] for r in timed)
        kind["perToken"] += settled
        kind["runs"] += 1
        biggest = max(r["nodes"] for r in timed)
        top = max((r.get("_peakMB") or 0) for r in timed)
        if top and biggest:
            peak.append((biggest * weights * width / 2**20, top))
        frames = os.path.join(store.run_dir(meta["id"]), "frames")
        stored = sum(os.path.getsize(os.path.join(frames, n)) for n in os.listdir(frames))
        recorded = (meta.get("iteration") or 0) * (_median([r["nodes"] for r in rows]) or 0)
        if recorded:
            disk.append(stored / recorded)

    slope, base = _memory_line(peak)
    fitted = {
        "secondsPerAgentIteration": seconds / agent_iterations if agent_iterations else None,
        "agentsPerToken": _median(per_token),
        "bytesPerAgentIteration": _median(disk),
        "peakBytesPerWeightByte": slope,
        "baseMB": base,
    }
    costs = {**CALIBRATION, **{k: v for k, v in fitted.items() if v is not None},
             "measured": {k: v is not None for k, v in fitted.items()},
             # Seconds per agent as each kind of run measured them, unscaled,
             # and the population it settles at, where it has been seen to.
             "kinds": {key: {"label": k["label"],
                             "secondsPerAgent": k["seconds"] / k["agentIterations"],
                             "agentsPerToken": _median(k["perToken"]), "runs": k["runs"]}
                       for key, k in kinds.items() if k["agentIterations"]},
             "runs": runs, "fitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    store.write_json(os.path.join(_made(lab_dir()), "costs.json"), costs, indent=1)
    return costs


def _memory_line(points: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    A run's peak memory as a base plus a multiple of its brains' bytes, from
    (brain MB, peak MB) of the recorded runs: the multiple by least squares, the
    base then raised until no run lies above the line, so that an estimate
    bounds a run rather than averaging it. Small runs are nearly all base and
    big ones nearly all brains; until runs of very different sizes have been
    measured the two cannot be told apart, and the calibration's multiple stands.
    """
    if not points:
        return None, None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    slope = CALIBRATION["peakBytesPerWeightByte"]
    if len(points) >= 3 and max(xs) >= 10 * max(min(xs), 1e-9):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slope = max(1.0, sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx)
    return slope, max(y - slope * x for x, y in points)


def costs() -> Dict[str, Any]:
    try:
        with open(os.path.join(lab_dir(), "costs.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {**CALIBRATION, "measured": {}, "runs": 0}


def predict(config: Dict[str, Any], iterations: int,
            known: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    What `iterations` more of a run with this configuration will cost: from
    runs of its own kind where some have been measured, otherwise from all
    runs scaled by the size of its brains, otherwise from the calibration.
    """
    known = known or costs()
    weights, width = _weights(config)
    kind = known.get("kinds", {}).get(kind_of(config), {})
    agents = (kind.get("agentsPerToken") or known["agentsPerToken"]) * config["total_tokens"]
    per_agent = kind.get("secondsPerAgent") or known["secondsPerAgentIteration"] * weights / 1e4
    return {
        "agents": agents,
        "seconds": iterations * agents * per_agent,
        # The frames, and the checkpoint beside them, which holds every brain.
        "diskBytes": iterations * agents * known["bytesPerAgentIteration"] + agents * weights * width,
        "peakMB": known["baseMB"] + agents * weights * width
                  * known["peakBytesPerWeightByte"] / 2**20,
    }


def _planned_checkpoints(config: Dict[str, Any], known: Dict[str, Any]) -> int:
    """Iterations between checkpoints so that one falls about every ten minutes."""
    seconds = max(1e-3, predict(config, 1, known)["seconds"])
    every = max(10, int(CHECKPOINT_SECONDS / seconds))
    return int(round(every, -len(str(every)) + 1))      # one significant figure


def schedule(durations: List[float], workers: int) -> float:
    """How long jobs of these lengths take on `workers`, longest first."""
    loads = [0.0] * max(1, workers)
    for seconds in sorted(durations, reverse=True):
        loads[loads.index(min(loads))] += seconds
    return max(loads) if durations else 0.0


# ----------------------------------------------------------------------------
# The control file and the lab's own state
# ----------------------------------------------------------------------------

def _control_defaults() -> Dict[str, Any]:
    return {"queue": [], "paused": False, "workers": DEFAULT_WORKERS, "reason": None}


def read_control() -> Dict[str, Any]:
    try:
        with open(os.path.join(lab_dir(), "control.json")) as f:
            control = json.load(f)
    except FileNotFoundError:
        control = {}
    return {**_control_defaults(), **control}


def _update_control(change) -> Dict[str, Any]:
    """
    Change the control file under its lock. The server writes it when someone
    presses a button and the lab when it pauses itself, from two processes, and
    a read-modify-write of one could otherwise land in the middle of the
    other's and undo it.
    """
    def merged(control: Dict[str, Any]) -> None:
        for key, value in _control_defaults().items():
            control.setdefault(key, value)
        change(control)
    return store.update_json(os.path.join(_made(lab_dir()), "control.json"), merged, indent=1)


def request(run: Optional[str] = None, pause: Optional[bool] = None,
            workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Say what the lab should do: queue an experiment (and unpause), pause, or
    use so many workers. Written to the control file, which the lab reads
    every few seconds; nothing signals it.
    """
    if run is not None:
        experiment_runs(run)                       # refuses a plan it cannot carry out

    def change(control: Dict[str, Any]) -> None:
        if run is not None:
            if run not in control["queue"]:
                control["queue"].append(run)
            control["paused"], control["reason"] = False, None
        if pause is not None:
            control["paused"], control["reason"] = bool(pause), None
        if workers is not None:
            control["workers"] = max(1, min(int(workers), os.cpu_count() or 1))
    return _update_control(change)


def _pause(reason: Optional[str]) -> None:
    """Pause, and say why, in one write: a reader never sees the one without the other."""
    _update_control(lambda control: control.update(paused=True, reason=reason))
    _log(f"paused: {reason}")


def lab_alive() -> bool:
    return store.is_locked(os.path.join(lab_dir(), "lab.lock"))


def spawn_lab() -> bool:
    """
    Start the lab in the background, unless one is running, and say whether it
    did. In a session of its own, so it outlives whatever started it: closing
    the page or restarting the server leaves an experiment running.
    """
    if lab_alive():
        return False
    log = open(os.path.join(_made(lab_dir()), "lab.log"), "a")
    lab = subprocess.Popen([sys.executable, "-B", os.path.join(HERE, "gol_lab.py"), "run"],
                           cwd=HERE, env={**os.environ, "GOL_RUNS_DIR": store.BASE_DIR},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    log.close()
    # Collected when it ends, so a starter that lives on is not left holding a zombie.
    threading.Thread(target=lab.wait, daemon=True).start()
    return True


# ----------------------------------------------------------------------------
# What there is to do
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class Job:
    spec: RunSpec
    until: int
    engine: str
    exists: bool
    fault_at: Optional[int]
    progress: int


def _faulted(run_id: str) -> bool:
    return any(s.get("exit") == "fault" for s in gol_worker.read_provenance(run_id)["sessions"])


def run_state(spec: RunSpec, engine: Optional[str]) -> Dict[str, Any]:
    """
    Where one run stands: done, blocked (and why), being advanced, or waiting
    — and if waiting, the job that takes it to its next target. A run that
    exists is continued by the engine that made it; one that does not is made
    by `engine`, the snapshot of today's, which is None while there is none.
    """
    path = os.path.join(store.run_dir(spec.run_id), "meta.json")
    if not os.path.exists(path):
        return {"state": "waiting", "iteration": 0, "engine": engine,
                "job": Job(spec, spec.targets[0], engine, False, spec.fault_at, 0)}
    meta = store.load_meta(spec.run_id)
    state = {"iteration": meta.get("iteration", 0), "status": meta.get("status"),
             "checkpoint": meta.get("checkpoint_iteration") or 0}
    if what_it_is(meta["config"]) != what_it_is(spec.config):
        return {**state, "state": "blocked",
                "reason": f"{spec.run_id} exists with other settings than the plan gives it"}
    # Blocked by its worker, which says so in the run's own metadata when the
    # environment it was made in has changed; or by the lab, in its own record.
    blocked = meta.get("lab_blocked") or _blocked().get(spec.run_id)
    if blocked:
        return {**state, "state": "blocked", "reason": blocked}
    if meta.get("status") == "extinct":
        return {**state, "state": "done"}
    made_by = (meta.get("lab") or {}).get("engine")
    state["engine"] = made_by
    if store.held(spec.run_id):
        return {**state, "state": "running"}
    progress = state["checkpoint"]
    next_target = next((t for t in spec.targets if t > progress), None)
    if next_target is None:
        return {**state, "state": "done"}
    fault = None if _faulted(spec.run_id) else spec.fault_at
    return {**state, "state": "waiting",
            "job": Job(spec, next_target, made_by, True, fault, progress)}


def jobs(queue: Iterable[str], engine: Optional[str]) -> List[Job]:
    """
    What is waiting to be done for the queued experiments, longest first, so
    that the workers finish together. An experiment kept to one worker takes
    its turns in the same places but starts with its quickest runs: on one
    worker the order cannot change how long it takes, and the quick runs
    answer first.
    """
    queue = list(queue)
    known = costs()
    found = []
    for spec in wanted(queue).values():
        state = run_state(spec, engine)
        if state.get("job"):
            found.append(state["job"])

    def seconds(job: Job) -> float:
        return predict(job.spec.config, job.until - job.progress, known)["seconds"]

    ordered = sorted(found, key=lambda j: -seconds(j))
    for name, cap in _caps(queue).items():
        if cap == 1:
            places = [i for i, job in enumerate(ordered) if name in job.spec.experiments]
            for i, job in zip(places, sorted((ordered[i] for i in places), key=seconds)):
                ordered[i] = job
    return ordered


# ----------------------------------------------------------------------------
# The lab itself
# ----------------------------------------------------------------------------

def _thread_settings(threads: Optional[str]) -> Dict[str, str]:
    """One thread per worker, unless a condition asks for the library's default."""
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    return {} if threads == "default" else {n: "1" for n in names}


def _memory_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 * 2**30


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}", flush=True)


def run_lab() -> int:
    """
    Work through the queue until it is done or the lab is paused. One process
    per run, at most as many as the control file says and the memory allows;
    a run that would not fit on the disk pauses everything instead of failing
    halfway.
    """
    os.makedirs(lab_dir(), exist_ok=True)
    try:
        lock = store.locked(os.path.join(lab_dir(), "lab.lock"), wait=False)
        lock.__enter__()
    except store.RunBusy:
        print("a lab is already running", file=sys.stderr)
        return 1

    asked = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: asked.update(stop=True))
    signal.signal(signal.SIGINT, lambda *_: asked.update(stop=True))

    running: Dict[str, Tuple[subprocess.Popen, Job, Dict[str, Any]]] = {}
    crashes: Dict[str, int] = {}
    budget = MEMORY_SHARE * _memory_bytes() / 2**20
    known = fit_costs()
    _log(f"lab started, pid {os.getpid()}, {budget:,.0f} MB for workers")

    try:
        while True:
            with gol_plan.reading():
                stopping, queue = _tick(running, crashes, asked, budget, known)
                done = not running and (stopping or not _anything_left(queue))
            _write_state(running)
            if done:
                break
            time.sleep(2)
    finally:
        _write_state({})
        lock.__exit__(None, None, None)
        _log("lab stopped")
    return 0


def _tick(running: Dict[str, Tuple[subprocess.Popen, Job, Dict[str, Any]]], crashes: Dict[str, int],
          asked: Dict[str, bool], budget: float, known: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    One turn of the lab: start what can be started and see to what has ended.
    Whether it is stopping, and the queue it worked from.
    """
    control = read_control()
    stopping = asked["stop"] or control["paused"]
    if stopping:
        for proc, _job, _ in running.values():
            if proc.poll() is None and not getattr(proc, "_asked", False):
                proc.send_signal(signal.SIGTERM)
                proc._asked = True          # type: ignore[attr-defined]
    else:
        engine, held_back = None, None
        try:
            engine = snapshot()
        except LabError as exc:
            held_back = str(exc)
        try:
            waiting = [j for j in jobs(control["queue"], engine)
                       if j.spec.run_id not in running]
        except LabError as exc:
            _pause(str(exc))
            waiting = []
        # Runs that exist go on, on their own engines. Runs still to
        # be made wait for a snapshot to be made with.
        startable = [j for j in waiting if j.engine is not None]
        if waiting and not startable and not running:
            _pause(held_back)
        start, short = decide(startable, [(job, cost) for _, job, cost in running.values()],
                              control["workers"], _caps(control["queue"]), budget,
                              shutil.disk_usage(_made(store.BASE_DIR)).free, known)
        for job, cost in start:
            running[job.spec.run_id] = (_start(job), job, cost)
            _log(f"started {job.spec.run_id} at {job.progress} → {job.until}")
        if short:
            _pause(short)

    for run_id, (proc, job, _cost) in list(running.items()):
        code = proc.poll()
        if code is None:
            continue
        del running[run_id]
        _log(f"{run_id} ended with {code}")
        if code not in (0, EXIT_FAULT, EXIT_REFUSED) and not stopping:
            meta = store.load_meta(run_id) if os.path.isdir(store.run_dir(run_id)) else {}
            crashes[run_id] = crashes.get(run_id, 0) + 1
            if crashes[run_id] >= 2 and (meta.get("checkpoint_iteration") or 0) <= job.progress:
                _block(run_id, f"it crashed twice without getting past iteration {job.progress}; "
                               f"its lab.log says why")
    return stopping, control["queue"]


def decide(waiting: List[Job], running: List[Tuple[Job, Dict[str, Any]]], workers: int,
           caps: Dict[str, int], budget: float, free: int, known: Dict[str, Any]
           ) -> Tuple[List[Tuple[Job, Dict[str, Any]]], Optional[str]]:
    """
    Which of the waiting jobs to start now, in their order, each with what it
    is expected to cost — and, if one would not fit on the disk, why the lab
    has to pause instead of starting it or anything after it.

    No more than `workers` at once, nor more of an experiment than its plan's
    cap; within `budget` megabytes of memory, except that a job always starts
    when nothing else is running, or the lab could never make a run larger
    than its budget; and within the `free` disk, counting what the jobs chosen
    before it will write — each used to be held against the whole disk, so
    jobs that fitted one by one could fill it together. Everything is handed
    in, so the choice can be tested without starting a process or asking the
    disk.
    """
    start: List[Tuple[Job, Dict[str, Any]]] = []
    busy = [job for job, _ in running]
    used = sum(cost["peakMB"] for _, cost in running)
    for job in waiting:
        if len(busy) >= workers:
            break
        if _at_cap(job, busy, caps):
            continue
        cost = predict(job.spec.config, job.until - job.progress, known)
        if busy and used + cost["peakMB"] > budget:
            continue
        if cost["diskBytes"] > free - DISK_MARGIN:
            return start, (f"{job.spec.run_id} needs about {cost['diskBytes'] / 2**30:.1f} GB and "
                           f"the disk has {(free - DISK_MARGIN) / 2**30:.1f} GB to spare")
        start.append((job, cost))
        busy.append(job)
        used += cost["peakMB"]
        free -= cost["diskBytes"]          # what the jobs chosen so far will write
    return start, None


def _blocked() -> Dict[str, str]:
    """
    The runs the lab has given up on, and why. Kept in the lab's own state
    rather than in a run's metadata, which a run deleted while the lab ran
    would not have. Taking a run off this list lets the lab try it again.
    """
    try:
        with open(os.path.join(lab_dir(), "blocked.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _block(run_id: str, reason: str) -> None:
    store.update_json(os.path.join(_made(lab_dir()), "blocked.json"),
                      lambda blocked: blocked.update({run_id: reason}), indent=1)


def _caps(queue: List[str]) -> Dict[str, int]:
    """The worker caps the queued experiments' plans set."""
    caps = {}
    for name in queue:
        try:
            cap = workers_cap(name)
        except LabError:
            continue
        if cap:
            caps[name] = cap
    return caps


def _at_cap(job: Job, running: List[Job], caps: Dict[str, int]) -> bool:
    """Whether starting this job would run more of some experiment's runs than its plan allows."""
    return any(sum(name in other.spec.experiments for other in running) >= caps[name]
               for name in job.spec.experiments if name in caps)


def _anything_left(queue: List[str]) -> bool:
    try:
        return any(run_state(spec, None)["state"] in ("waiting", "running")
                   for spec in wanted(queue).values())
    except LabError:
        return False


def _start(job: Job) -> subprocess.Popen:
    """One worker for one job, on the run's own engine, logging beside the run."""
    spec = job.spec
    made = None
    if not job.exists:
        config = dict(spec.config)
        if not config["checkpoint_every"]:
            config["checkpoint_every"] = _planned_checkpoints(config, costs())
        made = {"name": spec.name, "config": config, "record": spec.record,
                "lab": {**spec.lab, "engine": job.engine, "commit": _engine_commit(job.engine)}}
    command = gol_worker.command(engine_dir(job.engine), spec.run_id, job.until,
                                 spec=made, fault_at=job.fault_at)
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env.update(GOL_RUNS_DIR=store.BASE_DIR, **_thread_settings(spec.threads))
    log = open(os.path.join(_made(os.path.join(lab_dir(), "logs")), f"{spec.run_id}.log"), "a")
    proc = subprocess.Popen(command, cwd=engine_dir(job.engine), env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return proc


def _engine_commit(engine: str) -> Optional[str]:
    try:
        with open(os.path.join(engine_dir(engine), "ENGINE.json")) as f:
            return json.load(f).get("commit")
    except FileNotFoundError:
        return None


def _write_state(running: Dict[str, Any]) -> None:
    store.write_json(os.path.join(lab_dir(), "state.json"), {
        "pid": os.getpid(), "updated": time.time(),
        "running": {run_id: {"pid": proc.pid, "until": job.until}
                    for run_id, (proc, job, _) in running.items()}}, indent=1)


# ----------------------------------------------------------------------------
# Where everything stands
# ----------------------------------------------------------------------------

def _recent_rate(run_id: str) -> Optional[float]:
    """Seconds per iteration lately, from the run's own record."""
    import gol_record
    rows = [r["_seconds"] for r in _tail_rows(gol_record.stats_path(run_id), 40)
            if r.get("_seconds")]
    return _median(rows[-20:])


def experiment_status(name: str, control: Dict[str, Any], known: Dict[str, Any],
                      alive: bool) -> Dict[str, Any]:
    """
    One experiment: its runs, how far each is, what is left, and how long that
    will take. Its state is one of ready (not asked for), queued (asked for,
    waiting its turn), running, paused (by you, or by the lab with a reason),
    stopped (asked for, but no lab is running — after a restart, say),
    finished, blocked or invalid.
    """
    try:
        specs = experiment_runs(name)
        workers = min(control["workers"], workers_cap(name) or control["workers"])
    except LabError as exc:
        return {"state": "invalid", "reason": str(exc), "runs": []}
    shared = {}
    for other in experiments():
        if other != name:
            try:
                for spec in experiment_runs(other):
                    shared.setdefault(spec.run_id, []).append(other)
            except LabError:
                pass

    runs, left, total_iterations, done_iterations = [], [], 0, 0
    whole, engines = [], {}
    for spec in specs:
        state = run_state(spec, None)
        state.pop("job", None)
        if state.get("engine"):
            engines[state["engine"]] = engines.get(state["engine"], 0) + 1
        iteration = min(state.get("iteration", 0), spec.until)
        if state["state"] == "done":
            iteration = spec.until
        total_iterations += spec.until
        done_iterations += iteration
        rate = _recent_rate(spec.run_id) if state["state"] == "running" else None
        guess = predict(spec.config, 1, known)["seconds"]
        remaining = 0 if state["state"] in ("done", "blocked") else spec.until - iteration
        left.append(remaining * (rate or guess))
        whole.append(predict(spec.config, spec.until, known))
        runs.append({"id": spec.run_id, "name": spec.name, "condition": spec.condition,
                     "seed": spec.lab["seed"], "iteration": iteration, "until": spec.until,
                     "state": state["state"], "reason": state.get("reason"),
                     "diedAt": state["iteration"] if state.get("status") == "extinct" else None,
                     "secondsPerIteration": rate, "sharedWith": shared.get(spec.run_id, [])})

    states = {r["state"] for r in runs}
    if "blocked" in states:
        overall = "blocked"
    elif states == {"done"}:
        overall = "finished"
    elif "running" in states:
        overall = "running"
    elif name in control["queue"]:
        overall = "paused" if control["paused"] else "queued" if alive else "stopped"
    else:
        overall = "ready"
    return {
        "state": overall, "runs": runs, "engines": engines, "workers": workers,
        "done": done_iterations, "total": total_iterations,
        "secondsLeft": schedule([s for s in left if s], workers),
        "estimate": {"seconds": schedule([w["seconds"] for w in whole], workers),
                     "diskBytes": sum(w["diskBytes"] for w in whole),
                     "peakMB": max((w["peakMB"] for w in whole), default=0)},
    }


def status() -> Dict[str, Any]:
    with gol_plan.reading():
        return _status()


def _status() -> Dict[str, Any]:
    control = read_control()
    known = costs()
    alive = lab_alive()
    disk = shutil.disk_usage(_made(store.BASE_DIR))
    return {
        "lab": {"alive": alive, "paused": control["paused"], "reason": control["reason"],
                "queue": control["queue"], "workers": control["workers"],
                "cores": os.cpu_count()},
        "disk": {"free": disk.free, "total": disk.total},
        "memory": {"total": _memory_bytes()},
        "costs": known,
        "experiments": {name: experiment_status(name, control, known, alive)
                        for name in experiments()},
    }


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="what an experiment needs and costs")
    plan.add_argument("experiment")
    run = commands.add_parser("run", help="queue experiments and run the lab here")
    run.add_argument("experiment", nargs="*")
    commands.add_parser("pause", help="ask a running lab to pause")
    commands.add_parser("status", help="where everything stands")
    verify = commands.add_parser("verify", help="re-run some of an experiment's runs")
    verify.add_argument("experiment")
    verify.add_argument("--runs", type=int, default=3)
    verify.add_argument("--iterations", type=int, default=20)
    analyse = commands.add_parser("analyse", help="write an experiment's results into the book")
    analyse.add_argument("experiment")
    commands.add_parser("costs", help="refit what a simulation costs, and write it for the book")
    args = parser.parse_args()

    if args.command == "plan":
        control = read_control()
        state = experiment_status(args.experiment, control, costs(), lab_alive())
        for r in state["runs"]:
            shared = f"  (also {', '.join(r['sharedWith'])})" if r["sharedWith"] else ""
            where = (f"died out at {r['diedAt']}" if r["diedAt"] is not None
                     else f"{r['iteration']:>6} / {r['until']:<6}")
            print(f"  {r['id']:<32} {r['state']:<8} {where}{shared}")
        e = state["estimate"]
        print(f"\n{len(state['runs'])} runs. On {state['workers']} "
              f"worker{'s' if state['workers'] != 1 else ''}: about "
              f"{_hours(e['seconds'])} from nothing, {_hours(state['secondsLeft'])} from here; "
              f"{e['diskBytes'] / 2**30:.1f} GB of disk; up to {e['peakMB']:,.0f} MB a run.")
        return 0
    if args.command == "run":
        for name in args.experiment:
            request(run=name)
        return run_lab()
    if args.command == "pause":
        request(pause=True)
        print("asked the lab to pause; its runs finish their iteration and save")
        return 0
    if args.command == "status":
        print(json.dumps(status(), indent=1))
        return 0
    if args.command == "verify":
        engine = snapshot()
        specs = [s for s in experiment_runs(args.experiment)
                 if os.path.exists(os.path.join(store.run_dir(s.run_id), "meta.json"))]
        for spec in specs[:args.runs]:
            made_by = store.load_meta(spec.run_id)["lab"]["engine"]
            same = reproduce(spec.run_id, made_by, args.iterations)
            print(f"  {spec.run_id}: {'reproduced' if same['same'] else 'DIFFERS'} "
                  f"over {args.iterations} iterations on its own engine {made_by}")
            if made_by != engine:
                same = reproduce(spec.run_id, engine, args.iterations)
                print(f"  {'':<{len(spec.run_id)}}  {'reproduced' if same['same'] else 'DIFFERS'}"
                      f" on the engine now, {engine}")
        return 0
    if args.command == "analyse":
        import gol_analysis
        results = gol_analysis.analyse(args.experiment)
        print(f"wrote book/results/{args.experiment}.json"
              + "".join(f" and book/figures/{args.experiment}/{f}.svg"
                        for f in results.get("figures", [])))
        return 0
    if args.command == "costs":
        import gol_analysis
        print(json.dumps(gol_analysis.costs_report()["fitted"], indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
