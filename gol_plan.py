#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The book's experiments as the lab and the analysis read them.

A plan is book/experiments/<name>.json. Its `runs` say which runs the
experiment needs — a baseline, a world, the conditions that differ from the
baseline one setting at a time, seeds and a length — and this module turns
them into RunSpecs, named for what each run is, so that two experiments
needing the same run share it. Its `analyse` says what the analysis will look
at; check_analyse holds that to what gol_analysis reads, so a misspelt key is
refused when the plan is written rather than ignored when it is analysed.

Both halves used to live in gol_lab, where the lab loop, its costs and its
status sat beside them; the analysis and the book reached into the lab for a
plan.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from gol_config import INFRASTRUCTURE, MECHANICS, PARAMETERS, SPEC, SimConfig

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS = os.path.join(HERE, "book", "experiments")
STRAINS = os.path.join(HERE, "research", "strains.md")

#: A baseline names every one of these; a condition may change any of them.
SETTABLE = (set(MECHANICS) | set(PARAMETERS)) - {"seed"}
WORLD = ("total_tokens", "n_nodes", "k_neighbors")
#: How often the graph statistics are recorded, unless a plan says otherwise.
HEAVY_EVERY = 25


class LabError(ValueError):
    """A plan that cannot be carried out as written, with the reason."""


# ----------------------------------------------------------------------------
# Reading each plan once
# ----------------------------------------------------------------------------
#
# A lab tick and a status both ask after every queued plan several times over —
# its runs, its cap, the runs of every plan merged — and each ask read and
# checked the files again. Inside reading() each is read once, so one tick sees
# one version of every plan even if a file is saved half-way through it, and
# every caller gets a copy of its own: wanted() changes the specs it merges.

_SCOPE: Optional[Dict[Any, Any]] = None


@contextlib.contextmanager
def reading():
    """Read each plan once for the length of the block, and hand out copies."""
    global _SCOPE
    outer = _SCOPE
    if outer is None:
        _SCOPE = {}
    try:
        yield
    finally:
        if outer is None:
            _SCOPE = None


def _once(key: Any, make: Callable[[], Any]) -> Any:
    if _SCOPE is None:
        return make()
    if key not in _SCOPE:
        _SCOPE[key] = make()
    return copy.deepcopy(_SCOPE[key])


def read_plan(name: str) -> Dict[str, Any]:
    if not re.fullmatch(r"[A-Z]\d+", name or ""):
        raise LabError(f"{name!r} is not the name of a plan")
    return _once(("plan", name), lambda: _read(name))


def _read(name: str) -> Dict[str, Any]:
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
    return _once(("runs", name), lambda: _experiment_runs(name))


def _experiment_runs(name: str) -> List[RunSpec]:
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
                ("baseline", "world", "seeds", "iterations", "conditions", "record", "workers"))
    base_name = runs["baseline"]
    base = baseline(base_name)
    world = dict(runs["world"])
    _check_keys(f"{name} world", world, WORLD)
    record = dict(runs.get("record", {}))
    _check_keys(f"{name} record", record, ("heavy_every", "checkpoint_every"))
    iterations = int(runs["iterations"])
    workers_cap(name)                                       # refuses one that is not a count

    specs: List[RunSpec] = []
    for condition in runs["conditions"]:
        label = condition.get("name")
        _check_keys(f"{name} condition {label!r}", condition,
                    ("name", "set", "sizes", "stops", "fault_at", "threads", "replicate"))
        changed = dict(condition.get("set", {}))
        forbidden = sorted(set(changed) & ({"seed"} | set(INFRASTRUCTURE)))
        if forbidden:
            raise LabError(f"{name} condition {label!r} sets {', '.join(forbidden)}, which a "
                           f"condition cannot: the seed belongs to the seeds, and what is "
                           f"recorded to the lab")
        _check_keys(f"{name} condition {label!r} set", changed, SETTABLE - set(WORLD))
        # A run made again on purpose — to be stopped, cut off, run on other
        # threads, or simply kept apart from anyone else's — has a run of its
        # own, named for its condition, rather than sharing one.
        replicate = (label if condition.get("replicate")
                     or any(k in condition for k in ("stops", "fault_at", "threads")) else None)

        for size, seed in ((size, seed) for size in _sizes(name, label, condition, world)
                           for seed in parse_seeds(runs["seeds"])):
            here = {**world, "total_tokens": size}
            config = {**base, **here, **changed, "seed": seed, "export_every": 1,
                      "export_decisions": True,
                      "checkpoint_every": int(record.get("checkpoint_every", 0))}
            try:
                SimConfig.from_dict(config, stored=False)        # refuses what it cannot run
            except ValueError as exc:
                raise LabError(f"{name} condition {label!r} at {size:,} tokens cannot be run: "
                               f"{exc}") from None
            run_id = run_id_for(base_name, here, changed, seed, replicate)
            differs = "".join(f" · {k}={v}" for k, v in {**here, **changed}.items()
                              if k != "total_tokens")
            specs.append(RunSpec(
                run_id=run_id,
                name=(f"{base_name} · {size:,} tokens{differs}"
                      f" · seed {seed}" + (f" · {replicate}" if replicate else "")),
                config=config,
                lab={"baseline": base_name, "world": here, "set": changed, "seed": seed,
                     "replicate": replicate},
                record={"heavy_every": int(record.get("heavy_every", HEAVY_EVERY))},
                targets=sorted(set(int(s) for s in condition.get("stops", [])) | {iterations}),
                condition=label,
                threads=condition.get("threads"),
                fault_at=condition.get("fault_at"),
                experiments=[name]))
    return specs


def _sizes(name: str, label: str, condition: Dict[str, Any], world: Dict[str, Any]) -> List[int]:
    """
    The token supplies a condition runs at: the plan's world, which may list
    several, unless the condition names its own.
    """
    sizes = condition.get("sizes", world["total_tokens"])
    sizes = sizes if isinstance(sizes, list) else [sizes]
    if not sizes or not all(isinstance(t, int) and not isinstance(t, bool) and t > 0
                            for t in sizes):
        raise LabError(f"{name} condition {label!r}: a world's tokens are whole numbers above "
                       f"nothing, not {sizes!r}")
    return sizes


def workers_cap(name: str) -> Optional[int]:
    """The most of an experiment's runs its plan lets run at once, if it says."""
    runs = read_plan(name).get("runs")
    cap = runs.get("workers") if isinstance(runs, dict) else None
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < 1):
        raise LabError(f"{name}: workers is how many of its runs may run at once, not {cap!r}")
    return cap


def what_it_is(config: Dict[str, Any]) -> Dict[str, Any]:
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
            if what_it_is(have.config) != what_it_is(spec.config):
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
# What an analysis reads
# ----------------------------------------------------------------------------
#
# Everything gol_analysis reads of a plan's `analyse`, kind by kind, and
# nothing more. It reads with .get(), so a key it does not know — `figure`
# for `figures`, `settledfrom` — was simply never read, and an analysis ran
# without what its plan asked for. check_analyse refuses it instead: at the
# start of an analysis, and in the tests, for every plan in the book.

ANALYSE = {
    "identity": {"kind", "reference"},
    "series": {"kind", "reference", "figures", "endpoints", "settledFrom", "windows",
               "seedsNeeded", "wandering", "lineage"},
    "scaling": {"kind", "reference", "stats", "window", "check", "fitFrom"},
}
FIGURE = {"name", "stat", "phase", "title", "y", "log", "min", "seeds", "seedsOf", "guides"}
ENDPOINT = {"stat", "phase"}
WINDOW = {"name", "from", "to", "of"}
#: How a window may sum a run up: gol_analysis.WINDOW_OF, which a test holds to this.
WINDOW_SUMMARIES = ("mean", "median", "min", "max")
WANDERING = {"from", "stretch"}
STRETCH = {"from", "to"}
SCALED = {"name", "stat", "title", "y", "phase"}


def check_analyse(name: str, analyse: Dict[str, Any]) -> None:
    """Refuse an analyse block that says anything the analysis does not read, or leaves out what it needs."""
    kind = analyse.get("kind", "series")
    if kind not in ANALYSE:
        raise LabError(f"{name} analyse: kind {kind!r} is not one of {', '.join(ANALYSE)}")
    _check_keys(f"{name} analyse", analyse, ANALYSE[kind])

    def needs(where: str, given: Dict[str, Any], *keys: str) -> None:
        missing = [k for k in keys if k not in given]
        if missing:
            raise LabError(f"{where} needs {', '.join(missing)}")

    for i, figure in enumerate(analyse.get("figures", [])):
        where = f"{name} figure {figure.get('name', i)!r}"
        _check_keys(where, figure, FIGURE)
        needs(where, figure, "name", "stat")
    for endpoint in analyse.get("endpoints", []):
        if not isinstance(endpoint, str):
            _check_keys(f"{name} endpoint", endpoint, ENDPOINT)
            needs(f"{name} endpoint", endpoint, "stat")
    for window in analyse.get("windows", []):
        where = f"{name} window {window.get('name')!r}"
        _check_keys(where, window, WINDOW)
        needs(where, window, "name")
        if window.get("of", "mean") not in WINDOW_SUMMARIES:
            raise LabError(f"{where}: of is one of {', '.join(WINDOW_SUMMARIES)}, not {window['of']!r}")
    if "wandering" in analyse:
        _check_keys(f"{name} wandering", analyse["wandering"], WANDERING)
        needs(f"{name} wandering", analyse["wandering"], "stretch")
    if kind == "scaling":
        needs(f"{name} analyse", analyse, "window", "stats")
        for key in ("window", "check"):
            if key in analyse:
                _check_keys(f"{name} {key}", analyse[key], STRETCH)
                needs(f"{name} {key}", analyse[key], "from", "to")
        for item in analyse["stats"]:
            _check_keys(f"{name} scaled statistic", item, SCALED)
            needs(f"{name} scaled statistic", item, "stat")


@dataclasses.dataclass
class Plan:
    """An experiment's plan, read and checked whole: its runs, how many may run at once, its analysis."""
    name: str
    specs: List[RunSpec]
    cap: Optional[int]
    analyse: Dict[str, Any]


def plan(name: str) -> Plan:
    analyse = read_plan(name).get("analyse") or {}
    if analyse:
        check_analyse(name, analyse)
    return Plan(name, experiment_runs(name), workers_cap(name), analyse)
