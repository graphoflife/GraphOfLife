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
    book/figures/E02/<name>.json one per figure the plan asks for, drawn by
                                 the Book tab

An experiment's plan says what to look at under "analyse":

    "analyse": {
      "kind": "series",                     or "identity" (Experiment 1)
      "reference": "baseline",              the condition the others are compared with
      "figures": [{"name": "nodes", "stat": "nodes", "phase": 2,
                   "title": "Agents", "y": "agents", "log": false}],
      "endpoints": ["nodes", "gini"],       compared at the end of the runs
      "seedsNeeded": true                   how many seeds an effect needs
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

BOOK = os.path.join(gol_lab.HERE, "book")

#: The most points a figure's line holds; longer runs are averaged into bins.
FIGURE_POINTS = 600
#: The share of a run's end that stands for where it ended up.
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


def _write(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_finite(value), f, indent=1, allow_nan=False)
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


def ending(its: np.ndarray, values: np.ndarray, share: float = ENDING) -> Optional[float]:
    """Where a run ended up: its mean over the last `share` of its iterations."""
    keep = np.isfinite(values)
    its, values = its[keep], values[keep]
    if not its.size:
        return None
    return float(values[its >= its.max() - share * (its.max() - its.min())].mean())


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


def seeds_needed(values: List[float], shares=(0.05, 0.1, 0.2)) -> Dict[str, Any]:
    """
    How many seeds a condition would need for a change of 5, 10 or 20% of the
    mean to be found four times in five, given the spread these runs show
    (two groups, normal approximation, 5% two-sided).
    """
    v = np.array([x for x in values if x is not None and math.isfinite(x)])
    if len(v) < 3 or v.mean() == 0:
        return {"n": len(v)}
    sd, mean = float(v.std(ddof=1)), float(v.mean())
    return {"n": len(v), "mean": mean, "sd": sd, "cv": sd / abs(mean),
            "seeds": {f"{int(100 * s)}%": math.ceil(2 * ((Z_ALPHA + Z_POWER) * sd
                                                         / (s * abs(mean))) ** 2)
                      for s in shares}}


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


def analyse_series(name: str, plan: Dict[str, Any],
                   by_condition: Dict[str, List[gol_lab.RunSpec]]) -> Dict[str, Any]:
    """Bands, endings, comparisons with the reference condition, and seeds needed."""
    spec = plan["analyse"]
    reference = spec.get("reference")
    rows = {s.run_id: gol_record.read_stats(s.run_id)
            for specs in by_condition.values() for s in specs}

    figures = []
    for figure in spec.get("figures", []):
        drawn = []
        for condition, specs in by_condition.items():
            runs = [series(rows[s.run_id], figure["stat"], figure.get("phase", 2)) for s in specs]
            band = bands(runs)
            band["label"] = (f"{condition} — median, middle half, nine in ten of {len(specs)} runs"
                             if len(by_condition) > 1 else
                             f"median, middle half, nine in ten of {len(specs)} runs")
            drawn.append(band)
        for k, seed_spec in enumerate(figure.get("seeds", [])):
            for specs in by_condition.values():
                for s in specs:
                    if s.lab["seed"] == seed_spec:
                        its, values = series(rows[s.run_id], figure["stat"], figure.get("phase", 2))
                        drawn.append({"label": f"seed {seed_spec}", "x": its.tolist(),
                                      "y": values.tolist(), "width": 1.0})
        _write(os.path.join(BOOK, "figures", name, f"{figure['name']}.json"), {
            "title": figure.get("title"), "caption": figure.get("caption"),
            "x": {"label": "iteration"},
            "y": {"label": figure.get("y", figure["stat"]), "log": bool(figure.get("log")),
                  **({"min": figure["min"]} if "min" in figure else {})},
            "guides": figure.get("guides", []), "series": drawn})
        figures.append(figure["name"])

    endings: Dict[str, Dict[str, Dict[int, float]]] = {}
    for stat in spec.get("endpoints", []):
        phase = 2
        if isinstance(stat, dict):
            stat, phase = stat["stat"], stat.get("phase", 2)
        endings[stat] = {}
        for condition, specs in by_condition.items():
            values = {}
            for s in specs:
                value = ending(*series(rows[s.run_id], stat, phase))
                if value is not None:
                    values[s.lab["seed"]] = value
            endings[stat][condition] = values

    summary = {stat: {condition: _describe(list(values.values()))
                      for condition, values in by_condition_values.items()}
               for stat, by_condition_values in endings.items()}
    comparisons = {}
    if reference:
        for stat, by_condition_values in endings.items():
            comparisons[stat] = {
                condition: compare_conditions(by_condition_values[reference], values,
                                              seed=hash_seed(name, stat, condition))
                for condition, values in by_condition_values.items() if condition != reference}
    needed = ({stat: {condition: seeds_needed(list(values.values()))
                      for condition, values in by_condition_values.items()}
               for stat, by_condition_values in endings.items()}
              if spec.get("seedsNeeded") else {})

    extinct = {condition: [{"seed": s.lab["seed"],
                            "at": store.load_meta(s.run_id).get("iteration")}
                           for s in specs if store.load_meta(s.run_id).get("status") == "extinct"]
               for condition, specs in by_condition.items()}
    return {"reference": reference, "figures": figures, "endings": summary,
            "comparisons": comparisons, "seedsNeeded": needed, "extinct": extinct}


def hash_seed(*parts: str) -> int:
    """A resampling seed that depends only on what is being resampled."""
    import hashlib
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _describe(values: List[float]) -> Dict[str, Any]:
    v = np.array(values, dtype=float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else None,
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
    body = (analyse_identity(name, plan, by_condition) if kind == "identity"
            else analyse_series(name, plan, by_condition))
    results = {"experiment": name, "analysed": time.strftime("%Y-%m-%d"), "kind": kind,
               "engines": engines, **body, "citation": citation}
    _write(os.path.join(BOOK, "results", f"{name}.json"), results)
    return results


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
        step = max(1, len(rows) // 200)
        points += [(r["nodes"], r["_seconds"] / (weights / 1e4)) for r in rows[::step]]
        runs.append({"id": meta["id"], "tokens": meta["config"]["total_tokens"],
                     "weights": weights, "iterations": meta.get("iteration"),
                     "peakMB": max((r.get("_peakMB") or 0) for r in rows)})
    points.sort()
    top = points[-1][0] if points else 1
    _write(os.path.join(BOOK, "figures", "costs", "time.json"), {
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
