"""
Judging a round of the screen of random worlds (the author's plan of
2026-10-22): every world read at 1,000 and 2,000 by the five lines of health,
how often each drawn value came with a healthy world, and the values the next
round forces off. No world is played: every reading comes from the worlds'
archives.

The five lines, all of which a healthy world meets:

  alive      more than 20 agents at 2,000, and not stopped as too slow
  network    the health bar's marks besides dimension: one piece, at least 80%
             in the 2-core, no agent's loss cutting off more than 2% and no
             link's more than 1%, no agent linked to more than 5% of the
             others, 3 to 12 links per agent
  space      a dimension of 2 to 4 by ball growth, and a typical distance at
             least 1.5 times its random twin's: a true space, not a small world
  diverse    a diversity score (tokens, links, ages, genotypes) above B2's
             median world at the same iteration, and decisions as varied: the
             effective kinds of staking (research/oee/checks/diversity.py, read
             from the archive with no game played) above B2's median, both as a
             mean over the iteration and the one 500 before it, since the kinds
             swing from one moment to the next
  uneven     a Gini of the tokens no lower than B2's least even world

B2's lines are Experiment 16's ten worlds, read the same way (--reference).

Forcing values off: the author's rule, that a value which made every world it
appeared in unhealthy is drawn no more, kept from chance. A value is forced off
when every world it appeared in was unhealthy and the chance of that, were the
round's healthy worlds spread among its worlds at random, is below 0.05 divided
by the number of values judged (the chance is exact, a ratio of binomial
coefficients, and rounds multiply). The same is asked line by line, among the
worlds alive for every line but the first. A value is judged only among the
worlds where it could vary; a setting keeps at least one value.

    python3 research/screen/judge.py --reference                 B2's lines (research/screen/reference.json)
    python3 research/screen/judge.py E38 [E39 ...] [--jobs 4]    every round so far, read and judged
    python3 research/screen/judge.py --null                      how often chance alone would force a value off

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

READINGS = os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "readings.json")
REFERENCE = os.path.join(HERE, "reference.json")
REFERENCE_RUNS = [f"B2-10000-s{s:03d}" for s in range(1, 11)]
MOMENTS = (1000, 2000)
EARLIER = 500                       # the decisions are a mean over the moment and the one this much before
LINES = ("alive", "network", "space", "diverse", "uneven")
NETWORK_MARKS = ("one piece", "2-core", "agent cut", "link cut", "no hub", "links")
ALPHA = 0.05
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


def lines(r: Optional[Dict[str, Any]], kinds: Optional[float], alive: bool, ref: Dict[str, float]) -> Dict[str, bool]:
    """Which of the five lines a world meets at one moment."""
    if not alive or r is None or r.get("dead") or r.get("error"):
        return {line: False for line in LINES}
    bar = r.get("bar") or {}
    dim = r.get("dimension")
    return {
        "alive": True,
        "network": all(bool(bar.get(mark)) for mark in NETWORK_MARKS),
        "space": dim is not None and 2.0 <= dim <= 4.0 and (r.get("wide") or 0.0) >= 1.5,
        "diverse": (r.get("diversity") or 0.0) > ref["diversity"] and kinds is not None and kinds > ref["kinds"],
        "uneven": (r.get("gini_tokens") or 0.0) >= ref["gini"],
    }


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
            got = lines(r, kinds_at(have, w["id"], at), alive, ref[str(at)])
            w["moments"][str(at)] = {"lines": got, "healthy": all(got.values()),
                                     "score": sum(got.values()) / len(LINES),
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


def forcing(worlds: List[Dict[str, Any]], forced: Dict[str, List[Any]], at: int = MOMENTS[-1]) -> Dict[str, Any]:
    """
    Every drawn value's record, and the values chance cannot explain: healthy never, or one line never,
    among enough worlds. Only drawn worlds that finished count; a value only where its setting applies.
    """
    judged = [w for w in worlds if w["drawn"] and w["fate"] in ("lived", "died", "too slow")]
    version = judged[0]["sampler"] if judged else 1
    space = gol_screen.space(version)
    fulls = [gol_screen.full(w["set"], version) for w in judged]
    rounds = sorted({w["round"] for w in judged})
    records = []
    for setting in space:
        name = setting[0]
        for value in gol_screen.values_of(setting):
            if value in forced.get(name, []):
                continue
            eligible = [(w, f) for w, f in zip(judged, fulls) if applies(setting, f)]
            if not any(f[name] == value for _, f in eligible):
                continue
            rec = {"setting": name, "value": value, "worlds": 0, "healthy": 0, "lines": {}}
            for test in ("healthy",) + LINES:
                chance, seen, good_with = 1.0, 0, 0
                for rnd in rounds:
                    pool = [(w, f) for w, f in eligible if w["round"] == rnd]
                    if test not in ("healthy", "alive"):
                        pool = [(w, f) for w, f in pool if w["moments"][str(at)]["lines"]["alive"]]
                    meets = [w["moments"][str(at)]["healthy"] if test == "healthy"
                             else w["moments"][str(at)]["lines"][test] for w, _ in pool]
                    with_value = [m for (w, f), m in zip(pool, meets) if f[name] == value]
                    seen += len(with_value)
                    good_with += sum(with_value)
                    chance *= exact_chance(len(pool), sum(meets), len(with_value)) if not any(with_value) else 1.0
                if test == "healthy":
                    rec["worlds"], rec["healthy"], rec["chance"] = seen, good_with, chance if good_with == 0 else 1.0
                else:
                    rec["lines"][test] = {"worlds": seen, "met": good_with,
                                          "chance": chance if good_with == 0 else 1.0}
            records.append(rec)
    m = len(records)
    line_tests = m * len(LINES)
    out = []
    for rec in records:
        reasons = []
        if rec["healthy"] == 0 and rec["chance"] < ALPHA / max(1, m):
            reasons.append(f"no healthy world among {rec['worlds']} (chance {rec['chance']:.2g})")
        for line, got in rec["lines"].items():
            if got["met"] == 0 and got["worlds"] and got["chance"] < ALPHA / max(1, line_tests):
                reasons.append(f"line {line} met by none of {got['worlds']} (chance {got['chance']:.2g})")
        rec["force"] = reasons
        out.append(rec)
    # A setting keeps at least one value.
    proposal: Dict[str, List[Any]] = {k: list(v) for k, v in forced.items()}
    for setting in space:
        name = setting[0]
        to_force = [r["value"] for r in out if r["setting"] == name and r["force"]]
        left = [v for v in gol_screen.values_of(setting) if v not in proposal.get(name, []) and v not in to_force]
        if to_force and left:
            proposal[name] = proposal.get(name, []) + to_force
    return {"values": out, "judged": len(judged), "values_judged": m, "forced_next": proposal}


def judge(rounds: List[str], jobs: int) -> Dict[str, Any]:
    if not os.path.exists(REFERENCE):
        reference(jobs)
    ref = json.load(open(REFERENCE))
    worlds = worlds_of(rounds)
    judge_worlds(worlds, ref, jobs)
    forced = gol_plan.read_plan(rounds[-1])["screen"].get("forced") or {}
    verdict = forcing(worlds, forced)
    healthy = [w["id"] for w in worlds if w["moments"][str(MOMENTS[-1])]["healthy"]]
    out = {"rounds": rounds, "reference": ref, "worlds": worlds, "healthy": healthy, **verdict}
    with open(os.path.join(HERE, f"{rounds[-1]}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


# ---------------------------------------------------------------------------
# How often chance alone forces a value off
# ---------------------------------------------------------------------------

def null(rounds: int = 200, worlds: int = 50, share: float = 0.2, seed: int = 1) -> Dict[str, float]:
    """Rounds of drawn worlds made healthy at random, whatever their values: how often a value is forced."""
    rng = np.random.default_rng(seed)
    forced_rounds, forcings = 0, 0
    for r in range(rounds):
        fake = []
        for i in range(1, worlds + 1):
            changed, _ = gol_screen.draw(1, 10_000 + r, i)
            lines_ok = {line: bool(rng.random() < (1.0 if line == "alive" else 0.6)) for line in LINES}
            healthy = bool(rng.random() < share)
            if healthy:
                lines_ok = {line: True for line in LINES}
            fake.append({"round": "N", "drawn": True, "fate": "lived", "set": changed, "sampler": 1,
                         "moments": {str(MOMENTS[-1]): {"lines": lines_ok, "healthy": healthy}}})
        got = forcing(fake, {})
        n = sum(1 for v in got["values"] if v["force"])
        forcings += n
        forced_rounds += n > 0
    return {"rounds": rounds, "rounds_with_a_forcing": forced_rounds / rounds, "forcings_per_round": forcings / rounds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rounds", nargs="*")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.reference:
        print(json.dumps(reference(a.jobs), indent=1))
    if a.null:
        print(json.dumps(null(), indent=1))
    if a.rounds:
        out = judge(a.rounds, a.jobs)
        at = str(MOMENTS[-1])
        fates = {}
        for w in out["worlds"]:
            fates[w["fate"]] = fates.get(w["fate"], 0) + 1
        print("worlds:", len(out["worlds"]), fates, "healthy at 2,000:", out["healthy"])
        for line in LINES:
            met = sum(1 for w in out["worlds"] if w["moments"][at]["lines"][line])
            print(f"  line {line}: {met} worlds")
        for v in out["values"]:
            if v["force"]:
                print("force off:", v["setting"], v["value"], "—", "; ".join(v["force"]))
        print("forced next round:", json.dumps(out["forced_next"]))


if __name__ == "__main__":
    main()
