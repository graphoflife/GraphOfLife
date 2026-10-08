#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What an experiment's runs say, for its chapter.

    python3 gol_lab.py analyse E02

reads the runs an experiment's plan names (their stats.jsonl and
provenance.json), and writes two things into the book:

    book/results/E02.json        every number the chapter quotes, the
                                 comparisons with their intervals, and where
                                 every run came from
    book/figures/E02/<name>.svg  one per figure the plan asks for, drawn by
                                 book_svg.py, for a chapter to show

An experiment's plan says what to look at under "analyse":

    "analyse": {
      "kind": "series",                     or "identity" (Experiment 1), or
                                            "scaling" (a world's size against its
                                            tokens; see analyse_scaling)
      "reference": "baseline",              the condition the others are compared with
      "figures": [{"name": "nodes", "stat": "nodes", "phase": 2,
                   "title": "Agents", "y": "agents", "log": false,
                   "seeds": [1, 2], "seedsOf": "baseline"}],
                                            single worlds drawn as lines, of
                                            one condition or of every one
      "endpoints": ["nodes", "gini"],       compared where the runs ended up
      "settledFrom": 500,                   ... their mean from here to the end,
                                            rather than over their last fifth
      "windows": [{"name": "from 100", "from": 100},
                  {"name": "lowest", "from": 100, "of": "min"}],
                                            the endpoints over other stretches
      "seedsNeeded": true,                  how many seeds an effect needs, at
                                            the end and over every window
      "wandering": {"from": 100, "stretch": 100},
                                            how much is the seed, how much is time
      "lineage": true                       genotypes and lifetimes, from the frames
    }

Numpy only, and every resampling is seeded, so the same runs always give the
same numbers. A difference seen in fewer than thirty seeds per condition is
labelled indicative, as the book's rules ask.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gol_lab
import gol_record
import gol_store as store
from gol_series import NO_PARENT

BOOK = os.path.join(gol_lab.HERE, "book")

#: The most points a figure's line holds; longer runs are averaged into bins.
FIGURE_POINTS = 600
#: Where a run ended up is its level over this last share of its iterations.
ENDING = 0.2
RESAMPLES = 10_000
EFFECT_SEEDS = 30           # below this a comparison is indicative
Z_ALPHA, Z_POWER = 1.959964, 0.841621   # two-sided 5%, 80% power


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _finite(value.item())
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _draw(path: str, chart: Dict[str, Any]) -> None:
    """A chart for the book, as an SVG image (book_svg.py) at `path`, minus its extension."""
    import book_svg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}.svg", "w", encoding="utf-8") as f:
        f.write(book_svg.render([_finite(chart)], name=os.path.relpath(path, BOOK)))


def _write(path: str, value: Any, compact: bool = False) -> None:
    """
    Write JSON the page can read: no NaN anywhere. Results are indented for
    people to read; figures, which are thousands of points for the page to
    draw, are written compactly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_finite(value), f, allow_nan=False,
                  **({"separators": (",", ":")} if compact else {"indent": 1}))
        f.write("\n")


# ----------------------------------------------------------------------------
# Reading the runs
# ----------------------------------------------------------------------------

def _provenance(run_id: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(store.run_dir(run_id), "provenance.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"sessions": []}


def series(rows: List[Dict[str, Any]], stat: str, phase: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One statistic of one run, by iteration, from the frames of one phase.
    `a/b` is one statistic over another, frame by frame: bridges as a share of
    edges, leaves as a share of agents.
    """
    top, _, bottom = stat.partition("/")

    def value(row):
        a = row.get(top)
        if not bottom:
            return a
        b = row.get(bottom)
        return None if a is None or not b else a / b

    picked = [(r["iteration"], value(r)) for r in rows if r.get("phase") == phase]
    its = np.array([i for i, _ in picked], dtype=float)
    values = np.array([np.nan if v is None else float(v) for _, v in picked], dtype=float)
    return its, values


def bands(runs: List[Tuple[np.ndarray, np.ndarray]], points: int = FIGURE_POINTS) -> Dict[str, Any]:
    """
    The spread of a statistic over a set of runs, through time: in each of up
    to `points` stretches of iterations, every run's mean over the stretch,
    and across runs the median, the middle half, nine in ten, and how many
    runs were still there to count.

    The stretches follow the iterations the statistic was measured on. The
    graph statistics are measured every so many iterations, and stretches cut
    finer than that would leave most of them empty and the line in pieces.
    """
    measured = [its[np.isfinite(values)] for its, values in runs]
    measured = [m for m in measured if m.size]
    if not measured:
        return {"x": [], "y": [], "lo": [], "hi": [], "outerLo": [], "outerHi": [], "alive": []}
    # Stretches no shorter than the sparsest run's spacing between
    # measurements, so a run measured more often than the rest cannot leave
    # stretches only it reaches, with a band drawn over one run. A run that
    # ends early still leaves the rest at full detail.
    every = np.unique(np.concatenate(measured))
    spacing = max(float(np.median(np.diff(m))) if m.size > 1 else 1.0 for m in measured)
    span = every.max() + 1.0 - every.min()
    edges = np.linspace(every.min(), every.max() + 1.0,
                        int(max(1, min(points, round(span / spacing)))) + 1)
    middle = (edges[:-1] + edges[1:]) / 2
    means = np.full((len(runs), len(middle)), np.nan)
    for k, (its, values) in enumerate(runs):
        which = np.digitize(its, edges) - 1
        for b in range(len(middle)):
            inside = values[(which == b) & np.isfinite(values)]
            if inside.size:
                means[k, b] = inside.mean()
    alive = np.isfinite(means).sum(axis=0)

    def quantile(q):
        out = np.full(len(middle), np.nan)
        for b in np.nonzero(alive)[0]:
            out[b] = np.quantile(means[np.isfinite(means[:, b]), b], q)
        return out

    return {"x": middle.round(2).tolist(), "y": quantile(0.5).tolist(),
            "lo": quantile(0.25).tolist(), "hi": quantile(0.75).tolist(),
            "outerLo": quantile(0.05).tolist(), "outerHi": quantile(0.95).tolist(),
            "alive": alive.tolist()}


WINDOW_OF = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max}


def window(its: np.ndarray, values: np.ndarray, start: float, stop: float,
           of: str = "mean") -> Optional[float]:
    """A run's level over iterations `start` to `stop`: its mean, median, lowest or highest."""
    inside = values[(its >= start) & (its <= stop) & np.isfinite(values)]
    return float(WINDOW_OF[of](inside)) if inside.size else None


# ----------------------------------------------------------------------------
# Comparing conditions
# ----------------------------------------------------------------------------

def bootstrap(values: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    """A 95% interval for the mean, by resampling."""
    picks = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare_conditions(reference: Dict[int, float], other: Dict[int, float],
                       seed: int = 0) -> Dict[str, Any]:
    """
    How a condition differs from the reference at the end of the runs. Paired
    by seed where both have it, since runs with the same seed start from the
    same graph; otherwise as two independent samples. The null is that the
    condition makes no difference: signs flipped at random for pairs, labels
    shuffled otherwise.
    """
    rng = np.random.default_rng(seed)
    paired = sorted(set(reference) & set(other))
    if len(paired) >= 3:
        d = np.array([other[s] - reference[s] for s in paired])
        low, high = bootstrap(d, rng)
        flips = rng.choice([-1.0, 1.0], size=(RESAMPLES, len(d)))
        null = np.abs((flips * d).mean(axis=1))
        p = float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (RESAMPLES + 1))
        spread = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        return {"paired": True, "n": len(d), "difference": float(d.mean()),
                "interval": [low, high], "p": p,
                "effect": float(d.mean() / spread) if spread else None,
                "indicative": len(d) < EFFECT_SEEDS}
    a, b = np.array(list(reference.values())), np.array(list(other.values()))
    if len(a) < 2 or len(b) < 2:
        return {"paired": False, "n": min(len(a), len(b)), "difference": None}
    pooled = np.concatenate([a, b])
    observed = b.mean() - a.mean()
    null = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        rng.shuffle(pooled)
        null[k] = pooled[len(a):].mean() - pooled[:len(a)].mean()
    p = float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (RESAMPLES + 1))
    diffs = (rng.choice(b, size=(RESAMPLES, len(b))).mean(axis=1)
             - rng.choice(a, size=(RESAMPLES, len(a))).mean(axis=1))
    spread = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"paired": False, "n": min(len(a), len(b)), "difference": float(observed),
            "interval": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p": p, "effect": float(observed / spread) if spread else None,
            "indicative": min(len(a), len(b)) < EFFECT_SEEDS}


def seeds_needed(values: List[float], shares=(0.05, 0.1, 0.2)) -> Optional[Dict[str, int]]:
    """
    How many seeds a condition would need for a change of 5, 10 or 20% of the
    mean to be found four times in five, given the spread these runs show
    (two groups, normal approximation, 5% two-sided). None below three runs.
    """
    v = np.array([x for x in values if x is not None and math.isfinite(x)])
    if len(v) < 3 or v.mean() == 0:
        return None
    cv = float(v.std(ddof=1)) / abs(float(v.mean()))
    return {f"{int(100 * s)}%": math.ceil(2 * ((Z_ALPHA + Z_POWER) * cv / s) ** 2)
            for s in shares}


def wandering(runs: Dict[int, Tuple[np.ndarray, np.ndarray]], start: int, stop: int,
              stretch: int, seed: int = 0) -> Dict[str, Any]:
    """
    How much of a statistic's variation is between seeds and how much is each
    world wandering over time. Every run's level in stretches of `stretch`
    iterations from `start` to `stop`; the share of all the variance that
    lies between the runs' own averages; how much a run's highest stretch
    exceeds its lowest; how alike two stretches of one run are, 1, 2, 3, 5
    and 10 stretches apart — how long a world remembers its level; and
    whether a run's first half says anything about its second, across runs,
    against the null that it does not (halves paired at random). `runs` is
    keyed by seed.
    """
    rows = {k: [window(its, values, a, a + stretch - 1) for a in range(start, stop, stretch)]
            for k, (its, values) in runs.items()}
    rows = {k: row for k, row in rows.items() if None not in row}
    levels = np.array(list(rows.values()), dtype=float)
    if len(levels) < 3 or levels.shape[1] < 3:
        return {"runs": len(levels)}
    between = float(levels.mean(axis=1).var(ddof=1))
    within = float(levels.var(axis=1, ddof=1).mean())
    centred = levels - levels.mean(axis=1, keepdims=True)
    memory = {}
    for lag in (1, 2, 3, 5, 10):
        if lag < levels.shape[1] - 1:
            a, b = centred[:, :-lag].ravel(), centred[:, lag:].ravel()
            memory[str(lag * stretch)] = float(np.corrcoef(a, b)[0, 1])
    low, high = levels.min(axis=1), levels.max(axis=1)
    spread = dict(zip(rows, (high / low).tolist())) if (low > 0).all() else {}
    half = levels.shape[1] // 2
    first, second = levels[:, :half].mean(axis=1), levels[:, half:].mean(axis=1)
    r = float(np.corrcoef(first, second)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(first, rng.permutation(second))[0, 1] for _ in range(RESAMPLES)])
    return {"runs": len(levels), "stretches": levels.shape[1], "stretch": stretch,
            "between": between, "within": within,
            "betweenShare": between / (between + within) if between + within else None,
            "highOverLow": float(np.median(list(spread.values()))) if spread else None,
            "highOverLowBySeed": spread, "memory": memory,
            "halves": {"r": r, "p": float((np.sum(null >= r - 1e-12) + 1) / (RESAMPLES + 1))}}


# ----------------------------------------------------------------------------
# An experiment
# ----------------------------------------------------------------------------

def _runs_by_condition(name: str) -> Dict[str, List[gol_lab.RunSpec]]:
    found: Dict[str, List[gol_lab.RunSpec]] = {}
    for spec in gol_lab.experiment_runs(name):
        if os.path.exists(os.path.join(store.run_dir(spec.run_id), "meta.json")):
            found.setdefault(spec.condition, []).append(spec)
    return found


def _citation(name: str, by_condition: Dict[str, List[gol_lab.RunSpec]]) -> Dict[str, Any]:
    """Where every run came from, and the experiment as the strain registry cites one."""
    from gol_config import SimConfig
    runs, strains, commits, engines, environments = [], set(), set(), set(), set()
    for condition, specs in by_condition.items():
        for spec in specs:
            meta = store.load_meta(spec.run_id)
            sessions = _provenance(spec.run_id)["sessions"]
            first = sessions[0]["environment"] if sessions else {}
            rows = gol_record.read_stats(spec.run_id)
            seconds = sum(r.get("_seconds") or 0 for r in rows)
            peak = max((r.get("_peakMB") or 0 for r in rows), default=0)
            strain = SimConfig.from_dict(meta["config"]).strain_id()
            strains.add(strain)
            lab = meta.get("lab") or {}
            commits.add(lab.get("commit"))
            engines.add(lab.get("engine"))
            environments.add(tuple(first.get(k) for k in ("python", "numpy", "networkx",
                                                         "blas", "cpu", "dispatch")))
            runs.append({"id": spec.run_id, "condition": condition, "seed": spec.lab["seed"],
                         "iterations": meta.get("iteration"), "status": meta.get("status"),
                         "strain": strain, "engine": lab.get("engine"),
                         "commit": lab.get("commit"), "seconds": round(seconds, 1),
                         "peakMB": peak, "sessions": len(sessions),
                         "environment": {k: first.get(k) for k in
                                         ("python", "numpy", "networkx", "blas", "cpu",
                                          "dispatch")}})
    plan = gol_lab.read_plan(name)
    return {"strains": sorted(s for s in strains if s), "commits": sorted(c for c in commits if c),
            "engines": sorted(e for e in engines if e),
            "environments": [dict(zip(("python", "numpy", "networkx", "blas", "cpu",
                                       "dispatch"), e)) for e in sorted(environments,
                                                                       key=str)],
            "setup": plan.get("runs"), "runs": runs}


def _engines_agree(citation: Dict[str, Any]) -> Dict[str, Any]:
    """
    An experiment whose runs come from more than one frozen engine can only be
    read as one if the newest engine makes the others' runs again exactly.
    One run per older engine is made again on the newest and compared.
    """
    by_engine: Dict[str, str] = {}
    for run in citation["runs"]:
        by_engine.setdefault(run["engine"], run["id"])
    if len(by_engine) <= 1:
        return {"engines": list(by_engine), "agree": True, "checked": []}
    newest = max(by_engine, key=lambda e: os.path.getmtime(gol_lab.engine_dir(e)))
    checked = []
    for engine, run_id in by_engine.items():
        if engine == newest:
            continue
        same = gol_lab.reproduce(run_id, newest, 5)["same"]
        checked.append({"run": run_id, "madeBy": engine, "remadeBy": newest, "same": same})
    return {"engines": list(by_engine), "agree": all(c["same"] for c in checked),
            "checked": checked}


def analyse_identity(name: str, plan: Dict[str, Any],
                     by_condition: Dict[str, List[gol_lab.RunSpec]]) -> Dict[str, Any]:
    """Each condition's runs against the reference condition's, seed by seed."""
    reference = plan["analyse"].get("reference", "straight")
    base = {spec.lab["seed"]: spec for spec in by_condition.get(reference, [])}
    folder = store.BASE_DIR
    comparisons = []
    for condition, specs in by_condition.items():
        if condition == reference:
            continue
        for spec in specs:
            other = base.get(spec.lab["seed"])
            if other is None:
                continue
            answer = gol_lab.compare((folder, other.run_id), (folder, spec.run_id))
            comparisons.append({"condition": condition, "seed": spec.lab["seed"],
                                "reference": other.run_id, "run": spec.run_id,
                                "frames": answer["frames"],
                                "rows": len(gol_record.read_stats(spec.run_id)), **answer})
    return {"reference": reference, "comparisons": comparisons,
            "allSame": bool(comparisons) and all(c["same"] for c in comparisons)}


class _Runs:
    """
    An experiment's runs by condition, each statistic read from their
    recorded rows once, and how far each run lived.
    """

    def __init__(self, by_condition: Dict[str, List[gol_lab.RunSpec]]) -> None:
        self.by_condition = by_condition
        everyone = [s for specs in by_condition.values() for s in specs]
        self.rows = {s.run_id: gol_record.read_stats(s.run_id) for s in everyone}
        self.lived = {s.run_id: store.load_meta(s.run_id).get("iteration", 0) for s in everyone}
        self._measured: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}

    def measure(self, run_id: str, stat: str, phase: int) -> Tuple[np.ndarray, np.ndarray]:
        if (run_id, stat, phase) not in self._measured:
            self._measured[run_id, stat, phase] = series(self.rows[run_id], stat, phase)
        return self._measured[run_id, stat, phase]

    def reached_end(self, spec: gol_lab.RunSpec) -> bool:
        return self.lived[spec.run_id] >= spec.until

    def levels(self, endpoints: List[Tuple[str, int]], start, stop,
               of: str = "mean") -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Each endpoint's level over iterations `start(run)` to `stop(run)`, by
        condition and seed, in every run that lived through them.
        """
        table: Dict[str, Dict[str, Dict[int, float]]] = {}
        for stat, phase in endpoints:
            table[stat] = {}
            for condition, specs in self.by_condition.items():
                values = {}
                for s in specs:
                    if self.lived[s.run_id] >= stop(s):
                        value = window(*self.measure(s.run_id, stat, phase), start(s), stop(s), of)
                        if value is not None:
                            values[s.lab["seed"]] = value
                table[stat][condition] = values
        return table

    def figure(self, name: str, figure: Dict[str, Any]) -> None:
        """One figure the plan asks for: a band per condition, and any single seeds as lines."""
        stat, phase, several = figure["stat"], figure.get("phase", 2), len(self.by_condition) > 1
        drawn = []
        for condition, specs in self.by_condition.items():
            band = bands([self.measure(s.run_id, stat, phase) for s in specs])
            band["label"] = ((f"{condition} — " if several else "")
                             + f"median, middle half, nine in ten of {len(specs)} runs")
            drawn.append(band)
        for seed in figure.get("seeds", []):
            for condition, specs in self.by_condition.items():
                if figure.get("seedsOf", condition) != condition:
                    continue
                for s in specs:
                    if s.lab["seed"] == seed:
                        its, values = self.measure(s.run_id, stat, phase)
                        drawn.append({"label": f"{condition}, seed {seed}" if several else f"seed {seed}",
                                      "x": its.tolist(), "y": values.tolist(), "width": 1.0})
        _draw(os.path.join(BOOK, "figures", name, figure["name"]), {
            "title": figure.get("title"), "caption": figure.get("caption"),
            "x": {"label": "iteration"},
            "y": {"label": figure.get("y", stat), "log": bool(figure.get("log")),
                  **({"min": figure["min"]} if "min" in figure else {})},
            "guides": figure.get("guides", []), "series": drawn})


def _compared(name: str, label: str, reference: str,
              by_condition: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    """Every condition against the reference, on values keyed by seed."""
    return {condition: compare_conditions(by_condition[reference], values,
                                          seed=hash_seed(name, label, condition))
            for condition, values in by_condition.items() if condition != reference}


def analyse_series(name: str, plan: Dict[str, Any],
                   by_condition: Dict[str, List[gol_lab.RunSpec]]) -> Dict[str, Any]:
    """
    Figures; where the runs ended up, and the endpoints over any other
    stretches the plan names; each condition against the reference; how
    much is the seed and how much is time; dying out; and lineages.

    Where a run ended up is measured over the runs that reached the end. A
    world that died out did not end up anywhere: its last fifth is the
    record of its dying, and averaging it in with the living made the seeds
    look several times more different than the living worlds are. Dying out
    is an outcome of its own, counted separately, and a condition is compared
    on both.
    """
    spec = plan["analyse"]
    reference = spec.get("reference")
    runs = _Runs(by_condition)
    for figure in spec.get("figures", []):
        runs.figure(name, figure)

    endpoints = [(e["stat"], e.get("phase", 2)) if isinstance(e, dict) else (e, 2)
                 for e in spec.get("endpoints", [])]

    def described(table):
        return {stat: {condition: {**_describe(list(values.values())), "bySeed": values,
                                   **({"seedsNeeded": seeds_needed(list(values.values()))}
                                      if spec.get("seedsNeeded") else {})}
                       for condition, values in by_condition_values.items()}
                for stat, by_condition_values in table.items()}

    settled = spec.get("settledFrom")
    endings = runs.levels(endpoints,
                          lambda s: (1 - ENDING) * s.until if settled is None else settled,
                          lambda s: s.until)
    body = {"reference": reference, "figures": [f["name"] for f in spec.get("figures", [])],
            "endingsFrom": settled if settled is not None else f"the last {ENDING:.0%}",
            "endings": described(endings),
            "windows": {w["name"]: described(runs.levels(endpoints,
                                                         lambda s, w=w: w.get("from", 0),
                                                         lambda s, w=w: w.get("to", s.until),
                                                         w.get("of", "mean")))
                        for w in spec.get("windows", [])},
            "comparisons": ({stat: _compared(name, stat, reference, values)
                             for stat, values in endings.items()} if reference else {}),
            "extinct": {condition: [{"seed": s.lab["seed"], "at": runs.lived[s.run_id]}
                                    for s in specs
                                    if store.load_meta(s.run_id).get("status") == "extinct"]
                        for condition, specs in by_condition.items()},
            "reached": {condition: sum(map(runs.reached_end, specs))
                        for condition, specs in by_condition.items()}}

    wander = spec.get("wandering")
    if wander:
        body["wandering"] = wanders = {
            stat: {condition: wandering({s.lab["seed"]: runs.measure(s.run_id, stat, phase)
                                         for s in specs if runs.reached_end(s)},
                                        wander.get("from", 0), max(s.until for s in specs),
                                        wander["stretch"], seed=hash_seed(name, stat, condition))
                   for condition, specs in by_condition.items()}
            for stat, phase in endpoints}
        if reference:
            body["wanderingComparisons"] = {
                stat: _compared(name, f"wandering {stat}", reference,
                                {condition: w.get("highOverLowBySeed", {})
                                 for condition, w in by_condition_values.items()})
                for stat, by_condition_values in wanders.items()}

    if spec.get("lineage"):
        reached = {s.run_id for specs in by_condition.values() for s in specs
                   if runs.reached_end(s)}
        body["lineage"] = lineages = analyse_lineage(name, by_condition, reached)
        if reference:
            def per_seed(condition, part, key):
                found = lineages[condition]["runs"]
                return {s.lab["seed"]: found[s.run_id][part][key] for s in by_condition[condition]
                        if s.run_id in reached and found[s.run_id][part][key] is not None}
            body["lineageComparisons"] = {
                f"{part}.{key}": _compared(name, f"{part}.{key}", reference,
                                           {condition: per_seed(condition, part, key)
                                            for condition in by_condition})
                for part, key in LINEAGE_COMPARED}
    return body


# ----------------------------------------------------------------------------
# Lineages, read from the frames
# ----------------------------------------------------------------------------

#: How often, in iterations, the living are traced back to their common ancestors.
ANCESTRY_EVERY = 25
#: What a condition's lineages are compared with the reference condition's on.
LINEAGE_COMPARED = (("ancestor", "moves"), ("ancestor", "allFrom500"),
                    ("ancestor", "ninetyFrom500"), ("ancestor", "halfFrom500"),
                    ("ancestor", "oneFounder"), ("topShare", "maxFrom100"))


def shared_ancestors(parent: Dict[int, int], counts: Dict[int, int],
                     shares: Tuple[float, ...] = (1.0, 0.9, 0.5)) -> List[Optional[int]]:
    """
    The newest genotype that each share of the living agents descends from,
    or None while they still descend from more than one founder. `counts` is
    how many agents carry each living genotype. The tree is climbed from the
    living, newest genotype first: ids are handed out in order and a parent
    is always older than its child, so a genotype is reached only after all
    of its descendants have been counted into it, and the first to hold a
    share is the newest that does.
    """
    import heapq
    total = sum(counts.values())
    below = dict(counts)
    waiting = [-g for g in below]
    heapq.heapify(waiting)
    found: Dict[float, int] = {}
    while waiting and len(found) < len(shares):
        g = -heapq.heappop(waiting)
        for share in shares:
            if share not in found and below[g] >= share * total:
                found[share] = g
        up = parent.get(g, NO_PARENT)
        if up == NO_PARENT:
            continue
        if up not in below:
            below[up] = 0
            heapq.heappush(waiting, -up)
        below[up] += below[g]
    return [found.get(share) for share in shares]


def lineage(run_id: str) -> Dict[str, Any]:
    """
    Genotypes and agents through a run, from its frames: after every game,
    how much of the world the most common genotype holds and how old the
    living are; how long genotypes and agents last, from the first game
    frame they are in to the last (a life still going at the run's end is
    not counted, so the lengths are of lives that ended); and every
    ANCESTRY_EVERY iterations, how many iterations back all, nine in ten and
    half of the living share a single ancestor — None while they still
    descend from more than one founder, and counted as the age of the world
    when summed up. Needs every frame.
    """
    from collections import Counter

    first_g: Dict[int, int] = {}
    last_g: Dict[int, int] = {}
    first_a: Dict[int, int] = {}
    last_a: Dict[int, int] = {}
    born: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    top, ages, ancestry, end = [], [], [], None
    for index in range(store.count_frames(run_id)):
        frame = store.read_frame(run_id, index)
        it = frame["iteration"]
        for b, p in zip(frame["brain_ids"], frame["parent_brain_ids"]):
            if b not in born:
                born[b], parent[b] = it, p
        ids = frame["ids"]
        if frame.get("phase") != 2 or not ids:
            continue
        counts = Counter(frame["brain_ids"])
        top.append((it, max(counts.values()) / len(ids)))
        known = [a for a in frame["ages"] if a >= 0]
        if known:
            ages.append((it, float(np.median(known)), max(known)))
        if it % ANCESTRY_EVERY == 0:
            ancestry.append((it, *(None if g is None else it - born[g]
                                   for g in shared_ancestors(parent, counts))))
        for b in counts:
            first_g.setdefault(b, it)
            last_g[b] = it
        for a in ids:
            first_a.setdefault(a, it)
            last_a[a] = it
        end = it

    def lives(first, last):
        span = np.array([last[k] - first[k] + 1 for k in first if last[k] < end], dtype=float)
        if not span.size:
            return {"n": 0}
        return {"n": int(span.size), "median": float(np.median(span)),
                "p90": float(np.quantile(span, 0.9)), "longest": float(span.max()),
                "over5": float((span > 5).mean()), "over50": float((span > 50).mean())}

    shares = np.array([share for _, share in top])
    later = np.array([share for it, share in top if it >= 100])
    # How long one genotype held more than a tenth of the world at a stretch.
    longest = stretch = 0
    for share in shares:
        stretch = stretch + 1 if share > 0.1 else 0
        longest = max(longest, stretch)
    # Back to the founders, while there is no common ancestor yet.
    depths = [(it, *(it if d is None else d for d in rest)) for it, *rest in ancestry]
    settled = [row for row in depths if row[0] >= 500]
    # The common ancestor of everyone moves forward each time one branch of
    # the family tree has outlived all the others: one lineage replacing the rest.
    moves = sum(1 for before, after in zip(depths, depths[1:])
                if after[0] >= 100 and after[1] < before[1] + (after[0] - before[0]))
    return {"end": end,
            "topShare": {"max": float(shares.max()), "maxFrom100": float(later.max()) if later.size else None,
                         "medianFrom100": float(np.median(later)) if later.size else None,
                         "longestOverATenth": longest},
            "genotypeLife": lives(first_g, last_g), "agentLife": lives(first_a, last_a),
            "ancestor": {"allFrom500": float(np.median([r[1] for r in settled])) if settled else None,
                         "ninetyFrom500": float(np.median([r[2] for r in settled])) if settled else None,
                         "halfFrom500": float(np.median([r[3] for r in settled])) if settled else None,
                         "moves": moves,
                         "oneFounder": next((it for it, d, *_ in ancestry if d is not None), None)},
            "top": top, "ages": ages, "ancestry": depths}


def _lineage_job(job: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    folder, run_id = job
    store.BASE_DIR = folder
    return run_id, lineage(run_id)


def analyse_lineage(name: str, by_condition: Dict[str, List[gol_lab.RunSpec]],
                    reached: set) -> Dict[str, Any]:
    """
    Every run's lineage, read in parallel; figures of the largest genotype's
    share, of the age of the living and of how far back they share one
    ancestor; and, over the runs that reached the end, how many a single
    genotype ever held a tenth, a fifth, a third or half of after iteration
    100, how long genotypes and agents typically last, and how far back the
    living typically share an ancestor once a world has settled.
    """
    from multiprocessing import Pool

    jobs = [(store.BASE_DIR, s.run_id) for specs in by_condition.values() for s in specs]
    with Pool(min(4, len(jobs))) as pool:
        found = dict(pool.map(_lineage_job, jobs))
    for key, label, title, y in (
            ("top", "share", "Share of agents carrying the most common genotype", "share"),
            ("ages", "age", "Median age of the living, in iterations", "iterations"),
            ("ancestry", "ancestor",
             "How many iterations back all the living share one ancestor", "iterations")):
        drawn = []
        for condition, specs in by_condition.items():
            band = bands([(np.array([p[0] for p in found[s.run_id][key]], dtype=float),
                           np.array([p[1] for p in found[s.run_id][key]], dtype=float))
                          for s in specs])
            band["label"] = (f"{condition} — " if len(by_condition) > 1 else "") + \
                f"median, middle half, nine in ten of {len(specs)} runs"
            drawn.append(band)
        _draw(os.path.join(BOOK, "figures", name, label), {
            "title": title, "x": {"label": "iteration"},
            "y": {"label": y, "log": False, "min": 0}, "series": drawn})
    summaries = {}
    for condition, specs in by_condition.items():
        lived = [found[s.run_id] for s in specs if s.run_id in reached]

        def median(values):
            values = [v for v in values if v is not None]
            return float(np.median(values)) if values else None

        summaries[condition] = {
            "runs": {s.run_id: {k: v for k, v in found[s.run_id].items()
                                if k not in ("top", "ages", "ancestry")} for s in specs},
            "reached": len(lived),
            "largestFrom100": {str(share): sum((r["topShare"]["maxFrom100"] or 0) > share
                                               for r in lived) for share in (0.1, 0.2, 0.33, 0.5)},
            "genotypeLife": {k: median([r["genotypeLife"].get(k) for r in lived])
                             for k in ("median", "p90", "longest", "over5")},
            "agentLife": {k: median([r["agentLife"].get(k) for r in lived])
                          for k in ("median", "p90", "longest", "over5", "over50")},
            "ancestor": {k: median([r["ancestor"][k] for r in lived])
                         for k in ("allFrom500", "ninetyFrom500", "halfFrom500", "moves",
                                   "oneFounder")}}
    return summaries


def hash_seed(*parts: str) -> int:
    """A resampling seed that depends only on what is being resampled."""
    import hashlib
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _describe(values: List[float]) -> Dict[str, Any]:
    v = np.array(values, dtype=float)
    if not v.size:
        return {"n": 0}
    sd = float(v.std(ddof=1)) if v.size > 1 else None
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": sd, "cv": sd / abs(float(v.mean())) if sd is not None and v.mean() else None,
            "min": float(v.min()), "max": float(v.max()),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


def analyse(name: str) -> Dict[str, Any]:
    """Analyse an experiment and write its results and figures into the book."""
    plan = gol_lab.read_plan(name)
    if "analyse" not in plan:
        raise gol_lab.LabError(f"{name}'s plan does not say what to analyse")
    by_condition = _runs_by_condition(name)
    if not by_condition:
        raise gol_lab.LabError(f"none of {name}'s runs exist yet")
    citation = _citation(name, by_condition)
    engines = _engines_agree(citation)
    if not engines["agree"]:
        raise gol_lab.LabError(f"{name}'s runs come from engines that do not make each other's "
                               f"runs again: {engines['checked']}")
    kind = plan["analyse"].get("kind", "series")
    body = {"identity": analyse_identity, "scaling": analyse_scaling,
            "series": analyse_series}[kind](name, plan, by_condition)
    results = {"experiment": name, "analysed": time.strftime("%Y-%m-%d"), "kind": kind,
               "engines": engines, **body, "citation": citation}
    _write(os.path.join(BOOK, "results", f"{name}.json"), results)
    return results


# ----------------------------------------------------------------------------
# How a world's size follows its tokens
# ----------------------------------------------------------------------------

def power_fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    """
    A straight line through log10(y) against log10(x): the exponent b and the
    prefactor a of y = a · x^b, with a 95% interval for b from resampling the
    points within each group (a group is one world size, so every resample
    keeps every size), how much of the scatter the line explains, and how far
    the points bend away from it — the coefficient of a squared term in
    log10(x), with its own interval. A power law is a straight line on these
    axes; a bend whose interval stays clear of 0 says the points are not one.
    """
    keep = (x > 0) & (y > 0)
    u, v, groups = np.log10(x[keep]), np.log10(y[keep]), groups[keep]
    sizes = np.unique(groups)
    if len(sizes) < 3:
        return {"n": int(keep.sum()), "sizes": len(sizes)}
    centre = u.mean()

    def fitted(u_, v_):
        slope, intercept = np.polyfit(u_, v_, 1)
        bend = np.polyfit(u_ - centre, v_, 2)[0]
        return slope, intercept, bend

    slope, intercept, bend = fitted(u, v)
    residual = v - (intercept + slope * u)
    r2 = 1 - residual.var() / v.var() if v.var() else None
    rng = np.random.default_rng(seed)
    members = [np.nonzero(groups == g)[0] for g in sizes]
    slopes, bends = np.empty(RESAMPLES), np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        pick = np.concatenate([rng.choice(m, size=len(m)) for m in members])
        slopes[k], _, bends[k] = fitted(u[pick], v[pick])
    return {"n": int(keep.sum()), "sizes": len(sizes),
            "exponent": float(slope), "interval": [float(np.quantile(slopes, 0.025)),
                                                   float(np.quantile(slopes, 0.975))],
            "prefactor": float(10 ** intercept), "r2": float(r2) if r2 is not None else None,
            "bend": float(bend), "bendInterval": [float(np.quantile(bends, 0.025)),
                                                  float(np.quantile(bends, 0.975))]}


def analyse_scaling(name: str, plan: Dict[str, Any],
                    by_condition: Dict[str, List[gol_lab.RunSpec]]) -> Dict[str, Any]:
    """
    How a world's size follows its token supply. Every run's mean of each
    statistic over the plan's window of iterations, against the tokens of its
    world, on logarithmic axes; a power law fitted to them for each condition
    — over the sizes from `fitFrom` on, and over every size with a world
    alive — and the exponent between each size and the next, which a power
    law keeps constant. Each run's mean over an earlier window says whether
    it had settled. A world that did not live through the window is counted
    apart, as dying out always is.
    """
    spec = plan["analyse"]
    window, check = spec["window"], spec.get("check")
    fit_from = spec.get("fitFrom", 0)
    runs = _Runs(by_condition)

    def tokens(s: gol_lab.RunSpec) -> int:
        return int(s.lab["world"]["total_tokens"])

    results: Dict[str, Any] = {}
    figures = []
    for item in spec["stats"]:
        stat, phase = item["stat"], item.get("phase", 2)
        per_condition = {}
        drawn = []
        for condition, specs in by_condition.items():
            points = []
            for s in sorted(specs, key=lambda s: (tokens(s), s.lab["seed"])):
                if runs.lived[s.run_id] < window["to"]:
                    continue
                its, values = runs.measure(s.run_id, stat, phase)
                value = window_of(its, values, window)
                earlier = window_of(its, values, check) if check else None
                if value is not None:
                    points.append({"tokens": tokens(s), "seed": s.lab["seed"], "value": value,
                                   "settling": value / earlier if earlier else None})
            x = np.array([p["tokens"] for p in points], dtype=float)
            y = np.array([p["value"] for p in points], dtype=float)
            by_size = {}
            for size in sorted(set(int(t) for t in x)):
                here = y[x == size]
                settling = [p["settling"] for p in points
                            if p["tokens"] == size and p["settling"] is not None]
                by_size[str(size)] = {"n": int(here.size), "mean": float(here.mean()),
                                      "perToken": float(here.mean() / size),
                                      "cv": float(here.std(ddof=1) / here.mean())
                                      if here.size > 1 and here.mean() else None,
                                      "settling": float(np.median(settling)) if settling else None}
            ordered = sorted(by_size.items(), key=lambda kv: int(kv[0]))
            local = [{"from": int(a), "to": int(b),
                      "exponent": float(np.log10(vb["mean"] / va["mean"]) / np.log10(int(b) / int(a)))}
                     for (a, va), (b, vb) in zip(ordered, ordered[1:]) if va["mean"] > 0 and vb["mean"] > 0]
            inside = x >= fit_from
            fit = power_fit(x[inside], y[inside], x[inside], seed=hash_seed(name, stat, condition))
            per_condition[condition] = {
                "points": points, "bySize": by_size, "local": local, "fit": fit,
                "fitAll": power_fit(x, y, x, seed=hash_seed(name, stat, condition, "all"))}
            prefix = f"{condition} — " if len(by_condition) > 1 else ""
            drawn.append({"label": f"{prefix}each run, mean over iterations "
                                   f"{window['from']}–{window['to']}",
                          "x": x.tolist(), "y": y.tolist(), "points": True})
            if fit.get("exponent") is not None:
                ends = np.array([x[inside].min(), x[inside].max()])
                drawn.append({"label": f"{prefix}power law, exponent {fit['exponent']:.2f}",
                              "x": ends.tolist(),
                              "y": (fit["prefactor"] * ends ** fit["exponent"]).tolist()})
        results[item.get("name", stat)] = per_condition
        reference = per_condition.get(spec.get("reference")) or next(iter(per_condition.values()))
        if reference["points"]:
            # What exact proportion would look like: a line of exponent 1
            # through the middle of the reference condition's points.
            x = np.array([p["tokens"] for p in reference["points"]], dtype=float)
            y = np.array([p["value"] for p in reference["points"]], dtype=float)
            keep = y > 0
            middle = 10 ** np.mean(np.log10(y[keep]) - np.log10(x[keep]))
            ends = np.array([x.min(), x.max()])
            drawn.append({"label": "in proportion to the tokens (exponent 1)",
                          "x": ends.tolist(), "y": (middle * ends).tolist(), "width": 1.0})
        _draw(os.path.join(BOOK, "figures", name, item.get("name", stat)), {
            "title": item.get("title"), "caption": item.get("caption"),
            "x": {"label": "tokens in the world", "log": True},
            "y": {"label": item.get("y", stat), "log": True}, "series": drawn})
        figures.append(item.get("name", stat))

    return {"reference": spec.get("reference"), "figures": figures, "window": window,
            "check": check, "fitFrom": fit_from, "scaling": results,
            "extinct": {condition: [{"tokens": tokens(s), "seed": s.lab["seed"],
                                     "at": runs.lived[s.run_id]}
                                    for s in specs
                                    if store.load_meta(s.run_id).get("status") == "extinct"]
                        for condition, specs in by_condition.items()},
            "reached": {condition: sum(runs.lived[s.run_id] >= window["to"] for s in specs)
                        for condition, specs in by_condition.items()}}


def window_of(its: np.ndarray, values: np.ndarray, stretch: Dict[str, Any]) -> Optional[float]:
    """A run's mean over a stretch a plan names as {"from": …, "to": …}, both included."""
    return window(its, values, stretch["from"], stretch["to"])


# ----------------------------------------------------------------------------
# What a simulation costs
# ----------------------------------------------------------------------------

def costs_report() -> Dict[str, Any]:
    """
    Refit what a simulation costs, and write it for the book's appendix: the
    fitted numbers, and every recorded iteration's seconds against how many
    agents it had, with the line the estimates use.
    """
    from GraphOfLifeSimple import brain_shape
    from gol_config import SimConfig

    known = gol_lab.fit_costs()
    points, runs = [], []
    for meta in store.list_runs():
        if "lab" not in meta:
            continue
        rows = [r for r in gol_lab._tail_rows(gol_record.stats_path(meta["id"]), 2000)
                if r.get("_seconds") and r.get("nodes")]
        if not rows:
            continue
        weights = brain_shape(SimConfig.from_dict(meta["config"]))["weights"]
        step = max(1, len(rows) // 100)
        points += [(r["nodes"], r["_seconds"] / (weights / 1e4)) for r in rows[::step]]
        runs.append({"id": meta["id"], "tokens": meta["config"]["total_tokens"],
                     "weights": weights, "iterations": meta.get("iteration"),
                     "peakMB": max((r.get("_peakMB") or 0) for r in rows)})
    points.sort()
    top = points[-1][0] if points else 1
    _draw(os.path.join(BOOK, "figures", "costs", "time"), {
        "title": "Seconds per iteration, against the agents alive",
        "x": {"label": "agents"}, "y": {"label": "seconds per iteration, per 10,000 weights"},
        "series": [
            {"label": f"{len(points):,} recorded iterations of {len(runs)} runs",
             "x": [p[0] for p in points], "y": [p[1] for p in points], "points": True},
            {"label": "what the estimates assume", "x": [0, top],
             "y": [0, top * known["secondsPerAgentIteration"]]}]})
    report = {"fitted": known, "runs": runs, "updated": time.strftime("%Y-%m-%d")}
    _write(os.path.join(BOOK, "results", "costs.json"), report)
    return report
