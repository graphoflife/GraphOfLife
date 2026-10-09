#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What a simulation costs: fitted to what the lab's runs recorded, and
predicted for the runs still to come.

Every recorded iteration carries its seconds and the process's peak memory
(gol_record), and a run's frames on disk say what it writes. fit_costs fits
seconds per agent, agents per token, bytes per agent-iteration and memory per
byte of brain to those, overall and for each kind of run, and keeps them in
.lab/costs.json; predict says what a run of a configuration will need. The lab
decides from it which runs fit now, and the book's Appendix A shows the same
samples the fit was made from (gol_analysis.costs_report).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gol_store as store
from gol_config import SimConfig
from gol_plan import WORLD, what_it_is

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
SETTLED = 100                     # iterations before a world's size says anything


def _costs_file() -> str:
    return os.path.join(store.BASE_DIR, ".lab", "costs.json")


def samples(count: int = 400) -> Iterator[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Every run the lab made that recorded what its iterations cost: its
    metadata, the last `count` rows of its record, and those of them that were
    timed. What the costs are fitted to, and what Appendix A shows.
    """
    import gol_record
    for meta in store.list_runs():
        if "lab" not in meta:
            continue
        rows = tail_rows(gol_record.stats_path(meta["id"]), count)
        timed = [r for r in rows if r.get("_seconds") and r.get("nodes")]
        if timed:
            yield meta, rows, timed


def tail_rows(path: str, count: int = 400) -> List[Dict[str, Any]]:
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


def weights_of(config: Dict[str, Any]) -> Tuple[int, int]:
    from GraphOfLifeSimple import brain_shape
    shape = brain_shape(SimConfig.from_dict(config))
    return shape["weights"], shape["bytesPerWeight"]


def median(values: List[float]) -> Optional[float]:
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
    for meta, rows, timed in samples():
        runs += 1
        weights, width = weights_of(meta["config"])
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
        recorded = (meta.get("iteration") or 0) * (median([r["nodes"] for r in rows]) or 0)
        if recorded:
            disk.append(stored / recorded)

    slope, base = _memory_line(peak)
    fitted = {
        "secondsPerAgentIteration": seconds / agent_iterations if agent_iterations else None,
        "agentsPerToken": median(per_token),
        "bytesPerAgentIteration": median(disk),
        "peakBytesPerWeightByte": slope,
        "baseMB": base,
    }
    costs = {**CALIBRATION, **{k: v for k, v in fitted.items() if v is not None},
             "measured": {k: v is not None for k, v in fitted.items()},
             # Seconds per agent as each kind of run measured them, unscaled,
             # and the population it settles at, where it has been seen to.
             "kinds": {key: {"label": k["label"],
                             "secondsPerAgent": k["seconds"] / k["agentIterations"],
                             "agentsPerToken": median(k["perToken"]), "runs": k["runs"]}
                       for key, k in kinds.items() if k["agentIterations"]},
             "runs": runs, "fitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    os.makedirs(os.path.dirname(_costs_file()), exist_ok=True)
    store.write_json(_costs_file(), costs, indent=1)
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
    """
    What the lab last fitted, over the calibration: a file written before a
    measure existed still answers for it, rather than lacking the key.
    """
    try:
        with open(_costs_file()) as f:
            fitted = json.load(f)
    except FileNotFoundError:
        fitted = {}
    return {**CALIBRATION, "measured": {}, "runs": 0, **fitted}


def predict(config: Dict[str, Any], iterations: int,
            known: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    What `iterations` more of a run with this configuration will cost: from
    runs of its own kind where some have been measured, otherwise from all
    runs scaled by the size of its brains, otherwise from the calibration.
    """
    known = known or costs()
    weights, width = weights_of(config)
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
