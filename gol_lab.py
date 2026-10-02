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
from gol_config import INFRASTRUCTURE, MECHANICS, PARAMETERS, SPEC, SimConfig

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS = os.path.join(HERE, "book", "experiments")
STRAINS = os.path.join(HERE, "research", "strains.md")

#: What a run is made and advanced by, copied whole into each snapshot.
ENGINE_FILES = ("GraphOfLifeSimple.py", "gol_config.py", "gol_series.py", "gol_spectral.py",
                "gol_lightning.py", "gol_store.py", "gol_run.py", "gol_record.py",
                "gol_worker.py")

#: A baseline names every one of these; a condition may change any of them.
SETTABLE = (set(MECHANICS) | set(PARAMETERS)) - {"seed"}
WORLD = ("total_tokens", "n_nodes", "k_neighbors")

DEFAULT_WORKERS = 4
SETTLED = 100                     # iterations before a world's size says anything
HEAVY_EVERY = 25
CHECKPOINT_SECONDS = 600          # about how often a run saves itself
DISK_MARGIN = 10 * 2**30          # never fill the disk closer than this
MEMORY_SHARE = 0.75               # of the machine's memory, for all workers together

#: What a simulation costs before anything has been measured: the calibration
#: of 2026-10-02 on the new-run defaults (Core Ultra 7 258V, numpy 2.5.1).
CALIBRATION = {
    "secondsPerAgentIteration": 1.0e-3,     # per 10,000 weights in a brain
    "agentsPerToken": 0.4,
    "bytesPerAgentIteration": 100.0,
    "peakBytesPerWeightByte": 3.0,          # the Blotto copy and the checkpoint stack
    "baseMB": 300.0,
}

EXIT_FAULT, EXIT_REFUSED = 3, 4


class LabError(ValueError):
    """A plan that cannot be carried out as written, with the reason."""


def lab_dir() -> str:
    return os.path.join(store.BASE_DIR, ".lab")


# ----------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------

def read_plan(name: str) -> Dict[str, Any]:
    if not re.fullmatch(r"[A-Z]\d+", name or ""):
        raise LabError(f"{name!r} is not the name of a plan")
    try:
        with open(os.path.join(PLANS, f"{name}.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        raise LabError(f"there is no plan {name}") from None


def experiments() -> List[str]:
    """Every experiment the book has a plan for, in order."""
    names = (n[:-5] for n in os.listdir(PLANS) if re.fullmatch(r"E\d+\.json", n))
    return sorted(names, key=lambda n: int(n[1:]))


def baseline(name: str) -> Dict[str, Any]:
    """A baseline's settings, which have to name every mechanic and parameter but the seed."""
    settings = read_plan(name)["settings"]
    missing = SETTABLE - set(settings)
    unknown = set(settings) - SETTABLE
    if missing or unknown:
        raise LabError(f"{name} has to name every mechanic and parameter but the seed: "
                       f"missing {sorted(missing)}, not settings {sorted(unknown)}")
    return settings


def parse_seeds(seeds: Any) -> List[int]:
    """Seeds as `[1, 2, 3]`, `"1..30"` or `"1..10,20"`."""
    if isinstance(seeds, list):
        return [int(s) for s in seeds]
    out: List[int] = []
    for part in str(seeds).split(","):
        if ".." in part:
            low, high = part.split("..")
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return out


@dataclasses.dataclass
class RunSpec:
    """One run an experiment needs, and how far."""
    run_id: str
    name: str
    config: Dict[str, Any]
    lab: Dict[str, Any]
    record: Dict[str, Any]
    targets: List[int]
    condition: str
    threads: Optional[str] = None
    fault_at: Optional[int] = None
    experiments: List[str] = dataclasses.field(default_factory=list)

    @property
    def until(self) -> int:
        return self.targets[-1]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:20]


def run_id_for(base: str, world: Dict[str, Any], changed: Dict[str, Any], seed: int,
               replicate: Optional[str] = None) -> str:
    """
    A run's name, from what it is: `B1-5000-s007`, or with a short hash of
    whatever else differs from the baseline, `B1-5000-3fa2c1-s007`. A
    replicate — a run made again on purpose, to stop it or cut it — carries
    its condition's name as well. Once the algorithm's SPEC moves past 1
    (gol_config), runs carry it too, since the same settings then name a
    different algorithm.
    """
    differs = {**{k: v for k, v in world.items() if k != "total_tokens"}, **changed}
    mark = ("-" + hashlib.sha1(json.dumps(differs, sort_keys=True).encode()).hexdigest()[:6]
            if differs else "")
    run_id = f"{base}-{int(world['total_tokens'])}{mark}-s{int(seed):03d}"
    run_id += f"-g{SPEC}" if SPEC != 1 else ""
    return f"{run_id}-{_slug(replicate)}" if replicate else run_id


def _check_keys(where: str, given: Iterable[str], allowed: Iterable[str]) -> None:
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise LabError(f"{where}: {', '.join(unknown)} is not something a plan can say "
                       f"(it can say {', '.join(sorted(allowed))})")


def experiment_runs(name: str) -> List[RunSpec]:
    """Every run an experiment needs, made from its plan and checked on the way."""
    plan = read_plan(name)
    runs = plan.get("runs")
    if isinstance(runs, str):
        other = re.fullmatch(r"same as (E\d+)", runs.strip())
        if not other:
            raise LabError(f"{name}: runs is either a plan or 'same as E<n>', not {runs!r}")
        specs = experiment_runs(other.group(1))
        for spec in specs:
            spec.experiments = [name]
        return specs

    _check_keys(f"{name} runs", runs,
                ("baseline", "world", "seeds", "iterations", "conditions", "record"))
    base_name = runs["baseline"]
    base = baseline(base_name)
    world = dict(runs["world"])
    _check_keys(f"{name} world", world, WORLD)
    record = dict(runs.get("record", {}))
    _check_keys(f"{name} record", record, ("heavy_every", "checkpoint_every"))
    iterations = int(runs["iterations"])

    specs: List[RunSpec] = []
    for condition in runs["conditions"]:
        label = condition.get("name")
        _check_keys(f"{name} condition {label!r}", condition,
                    ("name", "set", "stops", "fault_at", "threads"))
        changed = dict(condition.get("set", {}))
        forbidden = sorted(set(changed) & ({"seed"} | set(INFRASTRUCTURE)))
        if forbidden:
            raise LabError(f"{name} condition {label!r} sets {', '.join(forbidden)}, which a "
                           f"condition cannot: the seed belongs to the seeds, and what is "
                           f"recorded to the lab")
        _check_keys(f"{name} condition {label!r} set", changed, SETTABLE - set(WORLD))
        replicate = (label if any(k in condition for k in ("stops", "fault_at", "threads"))
                     else None)

        for seed in parse_seeds(runs["seeds"]):
            config = {**base, **world, **changed, "seed": seed, "export_every": 1,
                      "export_decisions": True,
                      "checkpoint_every": int(record.get("checkpoint_every", 0))}
            SimConfig.from_dict(config, stored=False)            # refuses what it cannot run
            run_id = run_id_for(base_name, world, changed, seed, replicate)
            differs = "".join(f" · {k}={v}" for k, v in {**world, **changed}.items()
                              if k != "total_tokens")
            specs.append(RunSpec(
                run_id=run_id,
                name=(f"{base_name} · {int(world['total_tokens']):,} tokens{differs}"
                      f" · seed {seed}" + (f" · {replicate}" if replicate else "")),
                config=config,
                lab={"baseline": base_name, "world": world, "set": changed, "seed": seed,
                     "replicate": replicate},
                record={"heavy_every": int(record.get("heavy_every", HEAVY_EVERY))},
                targets=sorted(set(int(s) for s in condition.get("stops", [])) | {iterations}),
                condition=label,
                threads=condition.get("threads"),
                fault_at=condition.get("fault_at"),
                experiments=[name]))
    return specs


def _what_it_is(config: Dict[str, Any]) -> Dict[str, Any]:
    """A run's configuration without what only decides how it is recorded."""
    return {k: v for k, v in SimConfig.from_dict(config).to_dict().items()
            if k not in INFRASTRUCTURE}


def wanted(queue: Iterable[str]) -> Dict[str, RunSpec]:
    """
    Every run the queued experiments need, each once, taken as far as the
    furthest of them asks. Two plans that say different things about the same
    run are refused rather than reconciled.
    """
    merged: Dict[str, RunSpec] = {}
    for name in queue:
        for spec in experiment_runs(name):
            have = merged.get(spec.run_id)
            if have is None:
                merged[spec.run_id] = spec
                continue
            if _what_it_is(have.config) != _what_it_is(spec.config):
                raise LabError(f"{have.experiments[0]} and {name} describe {spec.run_id} "
                               f"differently")
            have.targets = sorted(set(have.targets) | set(spec.targets))
            have.experiments.append(name)
    return merged


def registered_strains() -> List[str]:
    """The strains research/strains.md says have been used."""
    with open(STRAINS) as f:
        return re.findall(r"^\| `(gol-[^`]+)` \|", f.read(), flags=re.M)


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
        subprocess.run([sys.executable, "-B", os.path.join(engine_dir(engine), "gol_worker.py"),
                        run_id, "--until", str(iterations), "--spec", json.dumps(spec)],
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


def fit_costs() -> Dict[str, Any]:
    """
    Fit what a simulation costs to every run the lab has recorded, and keep it
    in .lab/costs.json. Each measure falls back to the calibration until
    something has measured it.
    """
    seconds, agent_iterations, per_token, disk, peak = 0.0, 0.0, [], [], []
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
        seconds += sum(r["_seconds"] for r in timed)
        agent_iterations += sum(r["nodes"] * weights / 1e4 for r in timed)
        # Only after the founding boom: a world's first iterations are
        # nothing like the population it settles at, and short runs would
        # otherwise teach the estimates that worlds stay small.
        tokens = meta["config"]["total_tokens"]
        per_token += [r["nodes"] / tokens for r in rows if r.get("iteration", 0) >= SETTLED]
        biggest = max(r["nodes"] for r in timed)
        top = max((r.get("_peakMB") or 0) for r in timed)
        if top and biggest:
            peak.append((top - CALIBRATION["baseMB"]) * 2**20 / (biggest * weights * width))
        frames = os.path.join(store.run_dir(meta["id"]), "frames")
        stored = sum(os.path.getsize(os.path.join(frames, n)) for n in os.listdir(frames))
        recorded = (meta.get("iteration") or 0) * (_median([r["nodes"] for r in rows]) or 0)
        if recorded:
            disk.append(stored / recorded)

    fitted = {
        "secondsPerAgentIteration": seconds / agent_iterations if agent_iterations else None,
        "agentsPerToken": _median(per_token),
        "bytesPerAgentIteration": _median(disk),
        "peakBytesPerWeightByte": max(peak) if peak else None,
    }
    costs = {**CALIBRATION, **{k: v for k, v in fitted.items() if v is not None},
             "measured": {k: v is not None for k, v in fitted.items()},
             "runs": runs, "fitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    store.write_json(os.path.join(_made(lab_dir()), "costs.json"), costs, indent=1)
    return costs


def costs() -> Dict[str, Any]:
    try:
        with open(os.path.join(lab_dir(), "costs.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {**CALIBRATION, "measured": {}, "runs": 0}


def predict(config: Dict[str, Any], iterations: int,
            known: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """What `iterations` more of a run with this configuration will cost."""
    known = known or costs()
    weights, width = _weights(config)
    agents = known["agentsPerToken"] * config["total_tokens"]
    return {
        "agents": agents,
        "seconds": iterations * agents * known["secondsPerAgentIteration"] * weights / 1e4,
        "diskBytes": iterations * agents * known["bytesPerAgentIteration"],
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

def read_control() -> Dict[str, Any]:
    try:
        with open(os.path.join(lab_dir(), "control.json")) as f:
            control = json.load(f)
    except FileNotFoundError:
        control = {}
    return {"queue": [], "paused": False, "workers": DEFAULT_WORKERS, "reason": None, **control}


def request(run: Optional[str] = None, pause: Optional[bool] = None,
            workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Say what the lab should do: queue an experiment (and unpause), pause, or
    use so many workers. Written to the control file, which the lab reads
    every few seconds; nothing signals it.
    """
    control = read_control()
    if run is not None:
        experiment_runs(run)                       # refuses a plan it cannot carry out
        if run not in control["queue"]:
            control["queue"].append(run)
        control["paused"], control["reason"] = False, None
    if pause is not None:
        control["paused"], control["reason"] = bool(pause), None
    if workers is not None:
        control["workers"] = max(1, min(int(workers), os.cpu_count() or 1))
    store.write_json(os.path.join(_made(lab_dir()), "control.json"), control, indent=1)
    return control


def lab_alive() -> bool:
    lock = os.path.join(lab_dir(), "lab.lock")
    if not os.path.exists(lock):
        return False
    try:
        with store._locked(lock, wait=False):
            return False
    except store.RunBusy:
        return True


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
    try:
        with open(os.path.join(store.run_dir(run_id), "provenance.json")) as f:
            return any(s.get("exit") == "fault" for s in json.load(f)["sessions"])
    except FileNotFoundError:
        return False


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
    if _what_it_is(meta["config"]) != _what_it_is(spec.config):
        return {**state, "state": "blocked",
                "reason": f"{spec.run_id} exists with other settings than the plan gives it"}
    if meta.get("lab_blocked"):
        return {**state, "state": "blocked", "reason": meta["lab_blocked"]}
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
    """What is waiting to be done for the queued experiments, longest first."""
    known = costs()
    found = []
    for spec in wanted(queue).values():
        state = run_state(spec, engine)
        if state.get("job"):
            found.append(state["job"])
    return sorted(found, key=lambda j: -predict(j.spec.config, j.until - j.progress,
                                                 known)["seconds"])


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
        lock = store._locked(os.path.join(lab_dir(), "lab.lock"), wait=False)
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
                    request(pause=True)
                    _set_reason(str(exc))
                    _log(f"paused: {exc}")
                    waiting = []
                # Runs that exist go on, on their own engines. Runs still to
                # be made wait for a snapshot to be made with.
                startable = [j for j in waiting if j.engine is not None]
                if waiting and not startable and not running:
                    request(pause=True)
                    _set_reason(held_back)
                    _log(f"paused: {held_back}")
                waiting = startable
                used = sum(info["peakMB"] for _, _, info in running.values())
                for job in waiting:
                    if len(running) >= control["workers"]:
                        break
                    cost = predict(job.spec.config, job.until - job.progress, known)
                    if running and used + cost["peakMB"] > budget:
                        continue
                    free = shutil.disk_usage(_made(store.BASE_DIR)).free
                    if cost["diskBytes"] > free - DISK_MARGIN:
                        request(pause=True)
                        _set_reason(f"{job.spec.run_id} needs about "
                                    f"{cost['diskBytes'] / 2**30:.1f} GB and the disk has "
                                    f"{(free - DISK_MARGIN) / 2**30:.1f} GB to spare")
                        _log("paused: the disk would fill")
                        break
                    running[job.spec.run_id] = (_start(job), job, cost)
                    used += cost["peakMB"]
                    _log(f"started {job.spec.run_id} at {job.progress} → {job.until}")

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
                        store.update_meta(run_id, lab_blocked=(
                            f"it crashed twice without getting past iteration {job.progress}; "
                            f"its lab.log says why"))

            _write_state(running)
            if not running and (stopping or not _anything_left(control["queue"])):
                break
            time.sleep(2)
    finally:
        _write_state({})
        lock.__exit__(None, None, None)
        _log("lab stopped")
    return 0


def _set_reason(reason: str) -> None:
    control = read_control()
    control["reason"] = reason
    store.write_json(os.path.join(lab_dir(), "control.json"), control, indent=1)


def _anything_left(queue: List[str]) -> bool:
    try:
        return any(run_state(spec, None)["state"] in ("waiting", "running")
                   for spec in wanted(queue).values())
    except LabError:
        return False


def _start(job: Job) -> subprocess.Popen:
    """One worker for one job, on the run's own engine, logging beside the run."""
    spec = job.spec
    command = [sys.executable, "-B", os.path.join(engine_dir(job.engine), "gol_worker.py"),
               spec.run_id, "--until", str(job.until)]
    if not job.exists:
        config = dict(spec.config)
        if not config["checkpoint_every"]:
            config["checkpoint_every"] = _planned_checkpoints(config, costs())
        command += ["--spec", json.dumps({
            "name": spec.name, "config": config, "record": spec.record,
            "lab": {**spec.lab, "engine": job.engine,
                    "commit": _engine_commit(job.engine)}})]
    if job.fault_at is not None:
        command += ["--fault-at", str(job.fault_at)]
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
    workers = control["workers"]
    try:
        specs = experiment_runs(name)
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
        "state": overall, "runs": runs, "engines": engines,
        "done": done_iterations, "total": total_iterations,
        "secondsLeft": schedule([s for s in left if s], workers),
        "estimate": {"seconds": schedule([w["seconds"] for w in whole], workers),
                     "diskBytes": sum(w["diskBytes"] for w in whole),
                     "peakMB": max((w["peakMB"] for w in whole), default=0)},
    }


def status() -> Dict[str, Any]:
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
            print(f"  {r['id']:<32} {r['state']:<8} {r['iteration']:>6} / {r['until']:<6}{shared}")
        e = state["estimate"]
        print(f"\n{len(state['runs'])} runs. On {control['workers']} workers: about "
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
              + "".join(f" and book/figures/{args.experiment}/{f}.json"
                        for f in results.get("figures", [])))
        return 0
    if args.command == "costs":
        import gol_analysis
        print(json.dumps(gol_analysis.costs_report()["fitted"], indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
