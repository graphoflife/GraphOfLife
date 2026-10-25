"""
Judging a round of the screen of random worlds, and marking the drawn values
bad (the author's plan of 2026-10-22 and rule of 2026-10-23). No world is
played: every number comes from what the worlds recorded.

From round 3 on (the author's walkthrough of 2026-10-24) a world is judged by
the lines of research/screen/criteria.py, every one drawn on an average over
iterations 100 to the end of its run (research/screen/measures.py reads them).
Each world's value is shown beside every line, and the sensitivity table moves
each line alone and counts the worlds that would pass. Interesting is to be
defined with the author once the round has run: until then a world counts as
interesting, provisionally, when it is alive and misses at most two criteria.

Rounds 1 and 2 were judged by the criteria before these
(research/screen/criteria_1.py), read at iterations 1,000 and 2,000 from the
worlds' archives: the health bar, width, diversity score and Gini from
research/atlas/readings.py, and the effective kinds of staking
(research/oee/checks/diversity.py) as a mean over the moment and the one 500
before it, against B2's lines from Experiment 16's ten worlds (--reference).
A round is judged by the criteria of its sampler, and rounds judged together
must share one.

Marking a value bad: the author's rule. A value is marked bad when it was drawn
in at least ten worlds where it could matter, and every one of them died, or
none of them was interesting. Beside each, the report gives how likely that
would be by chance alone (the exact chance were the interesting worlds spread
at random): a warning when a marking may be luck, not a gate. A setting keeps
at least one value.

    python3 research/screen/judge.py --reference                 B2's lines for rounds 1 and 2 (reference.json)
    python3 research/screen/judge.py E38 [E39 ...] [--jobs 4]    rounds, read and judged

Keeps every reading it makes (rounds 1 and 2 in GraphOfLifeRuns/.pilots/screen/readings.json, later rounds
one file a world in .pilots/screen/worlds/), writes research/screen/<last round>.json (the judgement) and
prints the report and the next round's `forced`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

# One thread a process: the judge's processes are its parallelism, and it reads beside the lab's workers.
# An explicit setting in the environment still wins.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_screen  # noqa: E402
import gol_store  # noqa: E402

sys.path.insert(0, HERE)
import criteria as C  # noqa: E402
import criteria_1 as C1  # noqa: E402
import measures as M  # noqa: E402

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


def gather(items: List[Any], jobs: int, read: Callable[[Any], Dict[str, Any]], have: Dict[Any, Dict[str, Any]],
           keep: Callable[[Any, Dict[Any, Dict[str, Any]]], None]) -> Dict[Any, Dict[str, Any]]:
    """
    Every reading asked for, the ones in `have` kept. Each new reading is kept
    (`keep`) as soon as it is made, so a reading that fails loses no other; the
    failures are raised once the rest are in.
    """
    todo = [i for i in items if i not in have]
    failed = []
    if todo:
        with ProcessPoolExecutor(jobs) as pool:
            futures = {pool.submit(read, item): item for item in todo}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    have[item] = future.result()
                except Exception as e:  # noqa: BLE001 - raised below, after the others are kept
                    failed.append((item, e))
                    continue
                keep(item, have)
                print("read", *item, flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} readings failed, the first {failed[0][0]}") from failed[0][1]
    return have


# ---------------------------------------------------------------------------
# A round's worlds
# ---------------------------------------------------------------------------

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


def worlds_of(rounds: List[str]) -> List[Dict[str, Any]]:
    """Every world of the rounds named, with its draw and how it ended."""
    out = []
    for name in rounds:
        version = int(gol_plan.read_plan(name)["screen"]["sampler"])
        drawn = {setting[0] for setting in gol_screen.space(version)}
        for spec in gol_plan.experiment_runs(name):
            exists = os.path.exists(os.path.join(gol_store.run_dir(spec.run_id), "meta.json"))
            ended, at = fate(spec.run_id, spec.until) if exists else ("not made", 0)
            # A drawn setting of the world itself (its founders) is kept with the world, not the condition.
            changed = {**spec.lab["set"], **{k: v for k, v in spec.lab["world"].items() if k in drawn}}
            out.append({"round": name, "id": spec.run_id, "baseline": spec.lab["baseline"],
                        "seed": spec.lab["seed"], "set": changed, "fate": ended, "ended_at": at,
                        "until": spec.until, "sampler": version,
                        "drawn": spec.lab["baseline"] == "B1"})
    return out


def finished(w: Dict[str, Any]) -> bool:
    return w["fate"] in ("lived", "too slow", "died")


# ---------------------------------------------------------------------------
# From round 3: averages over the whole run
# ---------------------------------------------------------------------------

def judge_averages(worlds: List[Dict[str, Any]], jobs: int) -> Dict[str, Any]:
    """Read every finished world, average its numbers from iteration C.SETTLE on, and judge it, in place."""
    wanted = {w["id"]: M.items(w["id"], w["ended_at"], C.SETTLE) for w in worlds if finished(w)}
    have = gather([i for items in wanted.values() for i in items], jobs, M.read, M.kept(list(wanted)), M.keep)
    for w in worlds:
        w["alive"] = w["fate"] == "lived"
        w["averages"] = M.averages(have, wanted[w["id"]], C.SETTLE) if w["id"] in wanted else {}
        w["verdict"] = C.judged(w["averages"], w["alive"])
    judged = [w for w in worlds if finished(w)]
    return {"settle": C.SETTLE, "lines": C.LINES, "reported": C.REPORTED, "criteria": C.CRITERIA,
            "misses_allowed": C.MISSES_ALLOWED, "sensitivity": C.sensitivity(judged)}


# ---------------------------------------------------------------------------
# Rounds 1 and 2: readings at moments, against B2's lines
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
        r["kinds"] = float(diversity.stake_kinds(f"{run_id}:{at}", game=False)["effective"])
    return r


def wanted_readings(run_id: str, until: int) -> List[Tuple[str, int]]:
    """The archives a world is read at: each moment and the one before it, where they were kept."""
    kept = set(gol_store.archived(run_id))
    out = []
    for at in MOMENTS:
        for it in (at - EARLIER, at):
            if it in kept and it <= until:
                out.append((run_id, it))
    return sorted(set(out))


def kept_readings() -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not os.path.exists(READINGS):
        return {}
    return {(r["id"], r["at"]): r for r in json.load(open(READINGS))}


def keep_readings(item: Tuple[str, int], have: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(READINGS), exist_ok=True)
    with open(READINGS + ".tmp", "w") as f:
        json.dump(list(have.values()), f)
    os.replace(READINGS + ".tmp", READINGS)


def gather_readings(items: List[Tuple[str, int]], jobs: int) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return gather(items, jobs, read, kept_readings(), keep_readings)


def kinds_at(have: Dict[Tuple[str, int], Dict[str, Any]], run_id: str, at: int) -> Optional[float]:
    values = [have[(run_id, it)].get("kinds") for it in (at - EARLIER, at) if (run_id, it) in have]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def reference(jobs: int) -> Dict[str, Dict[str, float]]:
    """B2's lines at each moment: Experiment 16's ten worlds read as every screened world is."""
    items = [(run, it) for run in REFERENCE_RUNS for at in MOMENTS for it in (at - EARLIER, at)]
    have = gather_readings(items, jobs)
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


def judge_worlds(worlds: List[Dict[str, Any]], ref: Dict[str, Dict[str, float]], jobs: int) -> None:
    """Read every finished world and give it its lines at each moment, in place; its verdict is the last's."""
    items = []
    for w in worlds:
        if finished(w):
            items += wanted_readings(w["id"], w["ended_at"])
    have = gather_readings(items, jobs)
    for w in worlds:
        w["moments"] = {}
        for at in MOMENTS:
            r = have.get((w["id"], at))
            alive = (w["fate"] == "lived" or w["ended_at"] > at) and r is not None and r.get("agents", 0) > 20
            if at == MOMENTS[-1]:
                alive = alive and w["fate"] == "lived"
            got = C1.meets(r, kinds_at(have, w["id"], at), alive, ref[str(at)])
            w["moments"][str(at)] = {"criteria": got, "healthy": C1.healthy(got),
                                     "interesting": C1.interesting(got),
                                     "score": sum(got.values()) / len(C1.KEYS),
                                     "reading": {k: (r or {}).get(k) for k in
                                                 ("agents", "marks", "dimension", "wide", "diversity",
                                                  "gini_tokens", "tokens_per_agent", "hub", "agent_cut")},
                                     "kinds": kinds_at(have, w["id"], at)}
        w["verdict"] = w["moments"][str(MOMENTS[-1])]


def judge_moments(worlds: List[Dict[str, Any]], jobs: int) -> Dict[str, Any]:
    if not os.path.exists(REFERENCE):
        reference(jobs)
    ref = json.load(open(REFERENCE))
    judge_worlds(worlds, ref, jobs)
    return {"reference": ref, "criteria": C1.CRITERIA}


# ---------------------------------------------------------------------------
# Marking values bad
# ---------------------------------------------------------------------------

def exact_chance(total: int, good: int, with_value: int) -> float:
    """The chance that none of `good` worlds among `total` falls among `with_value` of them, by chance."""
    if with_value > total - good:
        return 0.0
    return math.comb(total - good, with_value) / math.comb(total, with_value)


def applies(setting: gol_screen.Setting, full: Dict[str, Any]) -> bool:
    return gol_screen._applies(setting[3], full)


def marking(worlds: List[Dict[str, Any]], forced: Dict[str, List[Any]], tried_at_least: int = C.TRIED_AT_LEAST
            ) -> Dict[str, Any]:
    """
    Every drawn value's record, and the values marked bad by the author's rule: tried in at least
    `tried_at_least` worlds where it could matter, and every one died, or none was interesting (each
    world's "verdict"). Only drawn worlds that finished count, over every round judged.
    """
    judged = [w for w in worlds if w["drawn"] and finished(w)]
    version = judged[0]["sampler"] if judged else 1
    space = gol_screen.space(version)
    fulls = [gol_screen.full(w["set"], version) for w in judged]
    records = []
    for setting in space:
        name = setting[0]
        eligible = [(w, f) for w, f in zip(judged, fulls) if applies(setting, f)]
        died_all = sum(1 for w, _ in eligible if w["fate"] == "died")
        good_all = sum(1 for w, _ in eligible if w["verdict"]["interesting"])
        for value in gol_screen.values_of(setting):
            if value in forced.get(name, []):
                continue
            mine = [w for w, f in eligible if f[name] == value]
            if not mine:
                continue
            died = sum(1 for w in mine if w["fate"] == "died")
            good = sum(1 for w in mine if w["verdict"]["interesting"])
            healthy = sum(1 for w in mine if w["verdict"]["healthy"])
            reasons = []
            if len(mine) >= tried_at_least:
                if died == len(mine):
                    reasons.append(f"all {len(mine)} worlds died (chance {exact_chance(len(eligible), len(eligible) - died_all, len(mine)):.2g})")
                elif good == 0:
                    reasons.append(f"none of {len(mine)} worlds was interesting (chance {exact_chance(len(eligible), good_all, len(mine)):.2g})")
            met = {key: sum(1 for w in mine if w["verdict"]["criteria"].get(key)) for key in mine[0]["verdict"]["criteria"]}
            records.append({"setting": name, "value": value, "worlds": len(mine), "died": died,
                            "interesting": good, "healthy": healthy, "met": met, "bad": reasons})
    proposal: Dict[str, List[Any]] = {k: list(v) for k, v in forced.items()}
    for setting in space:
        name = setting[0]
        bad = [r["value"] for r in records if r["setting"] == name and r["bad"]]
        left = [v for v in gol_screen.values_of(setting) if v not in proposal.get(name, []) and v not in bad]
        if bad and left:
            proposal[name] = proposal.get(name, []) + bad
    return {"values": records, "judged": len(judged), "forced_next": proposal}


# ---------------------------------------------------------------------------
# The judgement
# ---------------------------------------------------------------------------

def judge(rounds: List[str], jobs: int) -> Dict[str, Any]:
    versions = {int(gol_plan.read_plan(name)["screen"]["sampler"]) for name in rounds}
    if len(versions) != 1:
        raise SystemExit(f"rounds judged together share one sampler, and these have {sorted(versions)}")
    version = versions.pop()
    worlds = worlds_of(rounds)
    parts = judge_moments(worlds, jobs) if version == 1 else judge_averages(worlds, jobs)
    forced = gol_plan.read_plan(rounds[-1])["screen"].get("forced") or {}
    verdict = marking(worlds, forced)
    judged = [w for w in worlds if finished(w)]
    out = {"rounds": rounds, "version": version, **parts, "worlds": worlds,
           "healthy": [w["id"] for w in judged if w["verdict"]["healthy"]],
           "interesting": [w["id"] for w in judged if w["verdict"]["interesting"]], **verdict}
    with open(os.path.join(HERE, f"{rounds[-1]}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def _value(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3g}"


def report(out: Dict[str, Any]) -> None:
    """The judgement, in words and tables: every world, every line, the sensitivity, and the drawn values."""
    worlds = out["worlds"]
    fates: Dict[str, int] = {}
    for w in worlds:
        fates[w["fate"]] = fates.get(w["fate"], 0) + 1
    judged = [w for w in worlds if finished(w)]
    living = [w for w in judged if w["verdict"]["criteria"]["alive"]]
    print(f"worlds: {len(worlds)} {fates}; healthy {len(out['healthy'])}, interesting {len(out['interesting'])}"
          + (" (provisional: at most %d misses)" % out["misses_allowed"] if out["version"] > 1 else ""))
    for c in out["criteria"]:
        met = sum(1 for w in judged if w["verdict"]["criteria"][c["key"]])
        print(f"  {c['name']}: {met} of {len(judged)} worlds")
    if out["version"] > 1:
        print(f"\nevery line, against the {len(living)} living worlds' averages from iteration {out['settle']} on:")
        for line in out["lines"]:
            values = [w["averages"].get(line["measure"]) for w in living]
            have = [v for v in values if v is not None]
            spread = (f"{min(have):.3g} / {float(np.median(have)):.3g} / {max(have):.3g}") if have else "-"
            print(f"  {line['key']:17s} {line['passes']:8s} {line['value']:<5g} passed by "
                  f"{sum(w['verdict']['lines'][line['key']] for w in living):3d}   min/median/max {spread}")
        print("\nreported, with no line:")
        for r in out["reported"]:
            have = [w["averages"][r["measure"]] for w in living if w["averages"].get(r["measure"]) is not None]
            spread = (f"{min(have):.3g} / {float(np.median(have)):.3g} / {max(have):.3g}") if have else "-"
            print(f"  {r['name']:37s} min/median/max {spread}")
        print("\nsensitivity, each line moved alone (worlds passing it / meeting its criterion / healthy / interesting):")
        for row in out["sensitivity"]:
            mark = "*" if row["drawn"] else " "
            print(f"  {row['line']:17s} {row['value']:<5g}{mark} {row['pass']:3d} {row['criterion']:3d} "
                  f"{row['healthy']:3d} {row['interesting']:3d}")
        misses: Dict[int, int] = {}
        for w in living:
            misses[w["verdict"]["misses"]] = misses.get(w["verdict"]["misses"], 0) + 1
        print("\nliving worlds by criteria missed:", dict(sorted(misses.items())))
        print("\nevery world (lines failed):")
        for w in sorted(judged, key=lambda w: (not w["verdict"]["criteria"]["alive"], w["verdict"]["misses"])):
            failed = [k for k, ok in w["verdict"]["lines"].items() if not ok]
            a = w["averages"]
            print(f"  {w['id']:26s} {w['fate']:5s} misses {w['verdict']['misses']}  agents {_value(a.get('agents'))}"
                  f"  half-cut {_value(a.get('half_cut'))} (twin {_value(a.get('half_cut_twin'))})"
                  f"  apart {_value(a.get('apart'))}  failed: {', '.join(failed) or '-'}")
    for v in out["values"]:
        if v["bad"]:
            print("bad:", v["setting"], v["value"], "-", "; ".join(v["bad"]))
    print("forced next round" + (" (provisional)" if out["version"] > 1 else "") + ":", json.dumps(out["forced_next"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rounds", nargs="*")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.reference:
        print(json.dumps(reference(a.jobs), indent=1))
    if a.rounds:
        report(judge(a.rounds, a.jobs))


if __name__ == "__main__":
    main()
