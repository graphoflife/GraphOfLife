"""
Judging a round of the screen of random worlds (the author's plan of
2026-10-22 and rule of 2026-10-23): every world read at 1,000 and 2,000 by the
criteria of research/screen/criteria.py, and the drawn values marked bad. No
world is played: every reading comes from the worlds' archives.

A world is healthy when it meets every criterion, and interesting when it is
alive and misses at most one of the others. The criteria are listed, with what
each asks, in research/screen/criteria.py; the judge supplies the readings: the
health bar, width, diversity score and Gini from research/atlas/readings.py, and
the effective kinds of staking (research/oee/checks/diversity.py, read from the
archive with no game played) as a mean over the moment and the one 500 before
it, since the kinds swing from one moment to the next.

B2's lines are Experiment 16's ten worlds, read the same way (--reference).

Marking a value bad: the author's rule. A value is marked bad when it was drawn
in at least ten worlds where it could matter, and every one of them died, or
none of them was interesting. Beside each, the report gives how likely that
would be by chance alone (the exact chance were the interesting worlds spread
at random): a warning when a marking may be luck, not a gate. A setting keeps
at least one value.

    python3 research/screen/judge.py --reference                 B2's lines (research/screen/reference.json)
    python3 research/screen/judge.py E38 [E39 ...] [--jobs 4]    every round so far, read and judged

Writes GraphOfLifeRuns/.pilots/screen/readings.json (every reading, kept between calls),
research/screen/<last round>.json (the judgement) and prints the next round's `forced`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_screen  # noqa: E402
import gol_store  # noqa: E402

sys.path.insert(0, HERE)
import criteria as C  # noqa: E402

READINGS = os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "readings.json")
REFERENCE = os.path.join(HERE, "reference.json")
REFERENCE_RUNS = [f"B2-10000-s{s:03d}" for s in range(1, 11)]
MOMENTS = (1000, 2000)
EARLIER = 500                       # the decisions are a mean over the moment and the one this much before
SMALL = 30                          # fewer agents than this: no decisions are read


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Reading a world
# ---------------------------------------------------------------------------

def read(item: Tuple[str, int]) -> Dict[str, Any]:
    """One world at one archived iteration: the health bar, diversity, width, Gini and kinds of staking."""
    run_id, at = item
    readings = _module("readings_atlas", os.path.join(ROOT, "research", "atlas", "readings.py"))
    r = readings.read((run_id, at, "screen"))
    if r.get("dead") or r.get("error"):
        return {"id": run_id, "at": at, **r}
    if r.get("agents", 0) >= SMALL:
        diversity = _module("diversity_checks", os.path.join(ROOT, "research", "oee", "checks", "diversity.py"))
        r["kinds"] = float(diversity.measure(f"{run_id}:{at}", game=False)["stake_kinds"]["effective"])
    return r


def fate(run_id: str, until: int) -> Tuple[str, int]:
    """How a world ended, and the iteration it ended at: lived, died, too slow, or not yet."""
    meta = gol_store.load_meta(run_id)
    iteration = int(meta.get("iteration") or 0)
    status = meta.get("status")
    if status == "extinct":
        return "died", iteration
    if status == "too_slow":
        return "too slow", iteration
    return ("lived" if iteration >= until else "unfinished"), iteration


def wanted_readings(run_id: str, until: int) -> List[Tuple[str, int]]:
    """The archives a world is read at: each moment and the one before it, where they were kept."""
    kept = set(gol_store.archived(run_id))
    out = []
    for at in MOMENTS:
        for it in (at - EARLIER, at):
            if it in kept and it <= until:
                out.append((run_id, it))
    return sorted(set(out))


def gather(items: List[Tuple[str, int]], jobs: int) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Every reading asked for, the ones read before kept."""
    have = {}
    if os.path.exists(READINGS):
        for r in json.load(open(READINGS)):
            have[(r["id"], r["at"])] = r
    todo = [i for i in items if i not in have]
    if todo:
        with ProcessPoolExecutor(jobs) as pool:
            for item, r in zip(todo, pool.map(read, todo)):
                have[item] = r
                print(f"read {item[0]} at {item[1]}", flush=True)
        os.makedirs(os.path.dirname(READINGS), exist_ok=True)
        with open(READINGS + ".tmp", "w") as f:
            json.dump(list(have.values()), f)
        os.replace(READINGS + ".tmp", READINGS)
    return have


# ---------------------------------------------------------------------------
# The lines
# ---------------------------------------------------------------------------

def kinds_at(have: Dict[Tuple[str, int], Dict[str, Any]], run_id: str, at: int) -> Optional[float]:
    values = [have[(run_id, it)].get("kinds") for it in (at - EARLIER, at) if (run_id, it) in have]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def reference(jobs: int) -> Dict[str, Dict[str, float]]:
    """B2's lines at each moment: Experiment 16's ten worlds read as every screened world is."""
    items = [(run, it) for run in REFERENCE_RUNS for at in MOMENTS for it in (at - EARLIER, at)]
    have = gather(items, jobs)
    out = {}
    for at in MOMENTS:
        rows = [have[(run, at)] for run in REFERENCE_RUNS]
        kinds = [kinds_at(have, run, at) for run in REFERENCE_RUNS]
        out[str(at)] = {"diversity": float(np.median([r["diversity"] for r in rows])),
                        "gini": float(min(r["gini_tokens"] for r in rows)),
                        "kinds": float(np.median([k for k in kinds if k is not None])),
                        "worlds": REFERENCE_RUNS}
    with open(REFERENCE, "w") as f:
        json.dump(out, f, indent=1)
    return out


# ---------------------------------------------------------------------------
# A round's worlds, judged
# ---------------------------------------------------------------------------

def worlds_of(rounds: List[str]) -> List[Dict[str, Any]]:
    """Every world of the rounds named, with its draw and how it ended."""
    out = []
    for name in rounds:
        version = int(gol_plan.read_plan(name)["screen"]["sampler"])
        for spec in gol_plan.experiment_runs(name):
            exists = os.path.exists(os.path.join(gol_store.run_dir(spec.run_id), "meta.json"))
            ended, at = fate(spec.run_id, spec.until) if exists else ("not made", 0)
            out.append({"round": name, "id": spec.run_id, "baseline": spec.lab["baseline"],
                        "seed": spec.lab["seed"], "set": spec.lab["set"], "fate": ended, "ended_at": at,
                        "until": spec.until, "sampler": version,
                        "drawn": spec.lab["baseline"] == "B1"})
    return out


def judge_worlds(worlds: List[Dict[str, Any]], ref: Dict[str, Dict[str, float]], jobs: int) -> None:
    """Read every finished world and give it its lines at each moment, in place."""
    items = []
    for w in worlds:
        if w["fate"] in ("lived", "too slow", "died"):
            items += wanted_readings(w["id"], w["ended_at"])
    have = gather(items, jobs)
    for w in worlds:
        w["moments"] = {}
        for at in MOMENTS:
            r = have.get((w["id"], at))
            alive = (w["fate"] == "lived" or w["ended_at"] > at) and r is not None and r.get("agents", 0) > 20
            if at == MOMENTS[-1]:
                alive = alive and w["fate"] == "lived"
            got = C.meets(r, kinds_at(have, w["id"], at), alive, ref[str(at)])
            w["moments"][str(at)] = {"criteria": got, "healthy": C.healthy(got),
                                     "interesting": C.interesting(got),
                                     "score": sum(got.values()) / len(C.KEYS),
                                     "reading": {k: (r or {}).get(k) for k in
                                                 ("agents", "marks", "dimension", "wide", "diversity",
                                                  "gini_tokens", "tokens_per_agent", "hub", "agent_cut")},
                                     "kinds": kinds_at(have, w["id"], at)}


def exact_chance(total: int, good: int, with_value: int) -> float:
    """The chance that none of `good` worlds among `total` falls among `with_value` of them, by chance."""
    if with_value > total - good:
        return 0.0
    return math.comb(total - good, with_value) / math.comb(total, with_value)


def applies(setting: gol_screen.Setting, full: Dict[str, Any]) -> bool:
    return gol_screen._applies(setting[3], full)


def marking(worlds: List[Dict[str, Any]], forced: Dict[str, List[Any]], at: int = MOMENTS[-1]) -> Dict[str, Any]:
    """
    Every drawn value's record, and the values marked bad by the author's rule: tried in at least
    C.TRIED_AT_LEAST worlds where it could matter, and every one died, or none was interesting.
    Only drawn worlds that finished count, over every round judged.
    """
    judged = [w for w in worlds if w["drawn"] and w["fate"] in ("lived", "died", "too slow")]
    version = judged[0]["sampler"] if judged else 1
    space = gol_screen.space(version)
    fulls = [gol_screen.full(w["set"], version) for w in judged]
    records = []
    for setting in space:
        name = setting[0]
        eligible = [(w, f) for w, f in zip(judged, fulls) if applies(setting, f)]
        died_all = sum(1 for w, _ in eligible if w["fate"] == "died")
        good_all = sum(1 for w, _ in eligible if w["moments"][str(at)]["interesting"])
        for value in gol_screen.values_of(setting):
            if value in forced.get(name, []):
                continue
            mine = [w for w, f in eligible if f[name] == value]
            if not mine:
                continue
            died = sum(1 for w in mine if w["fate"] == "died")
            good = sum(1 for w in mine if w["moments"][str(at)]["interesting"])
            healthy = sum(1 for w in mine if w["moments"][str(at)]["healthy"])
            reasons = []
            if len(mine) >= C.TRIED_AT_LEAST:
                if died == len(mine):
                    reasons.append(f"all {len(mine)} worlds died (chance {exact_chance(len(eligible), len(eligible) - died_all, len(mine)):.2g})")
                elif good == 0:
                    reasons.append(f"none of {len(mine)} worlds was interesting (chance {exact_chance(len(eligible), good_all, len(mine)):.2g})")
            records.append({"setting": name, "value": value, "worlds": len(mine), "died": died,
                            "interesting": good, "healthy": healthy, "bad": reasons})
    proposal: Dict[str, List[Any]] = {k: list(v) for k, v in forced.items()}
    for setting in space:
        name = setting[0]
        bad = [r["value"] for r in records if r["setting"] == name and r["bad"]]
        left = [v for v in gol_screen.values_of(setting) if v not in proposal.get(name, []) and v not in bad]
        if bad and left:
            proposal[name] = proposal.get(name, []) + bad
    return {"values": records, "judged": len(judged), "forced_next": proposal}


def judge(rounds: List[str], jobs: int) -> Dict[str, Any]:
    if not os.path.exists(REFERENCE):
        reference(jobs)
    ref = json.load(open(REFERENCE))
    worlds = worlds_of(rounds)
    judge_worlds(worlds, ref, jobs)
    forced = gol_plan.read_plan(rounds[-1])["screen"].get("forced") or {}
    verdict = marking(worlds, forced)
    at = str(MOMENTS[-1])
    healthy = [w["id"] for w in worlds if w["moments"][at]["healthy"]]
    interesting = [w["id"] for w in worlds if w["moments"][at]["interesting"]]
    out = {"rounds": rounds, "reference": ref, "criteria": C.CRITERIA, "worlds": worlds, "healthy": healthy,
           "interesting": interesting, **verdict}
    with open(os.path.join(HERE, f"{rounds[-1]}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rounds", nargs="*")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.reference:
        print(json.dumps(reference(a.jobs), indent=1))
    if a.rounds:
        out = judge(a.rounds, a.jobs)
        at = str(MOMENTS[-1])
        fates = {}
        for w in out["worlds"]:
            fates[w["fate"]] = fates.get(w["fate"], 0) + 1
        print("worlds:", len(out["worlds"]), fates, "healthy at 2,000:", out["healthy"],
              "interesting:", len(out["interesting"]))
        for c in C.CRITERIA:
            met = sum(1 for w in out["worlds"] if w["moments"][at]["criteria"][c["key"]])
            print(f"  {c['name']}: {met} worlds")
        for v in out["values"]:
            if v["bad"]:
                print("bad:", v["setting"], v["value"], "—", "; ".join(v["bad"]))
        print("forced next round:", json.dumps(out["forced_next"]))


if __name__ == "__main__":
    main()
