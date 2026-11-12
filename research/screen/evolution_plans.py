"""
The plans that follow round 3's evolution rank (2026-11-12). Both read the rank
(research/screen/evolution.json, made by research/screen/braintree.py) and
health of 9 November (research/screen/health.json), which now only filters:

- Experiment 45, round 3's eras against each other with more coins: in the
  ten worlds the rank puts first, and in the five healthiest and the five least
  healthy that lived, the brains of iterations 500 to 3,000 against those of
  250 and of 1,500, three coins on each era's world, and three eras each
  against itself with four coins for the noise. It replaces Experiment 44,
  withdrawn unplayed.
- Experiment 46, why evolution levels off: the rules of the best world by the
  rank that has revolutions on, at 10,000 tokens, and arms that each change
  one thing — four times the tokens, faithful copies, the other kind of message
  to oneself, and revolutionary tokens that count for the coalition alone.

A plan is written with an empty thesis, to be written into it by hand before
any run, and a plan already written is never written over. Experiment 47, the
contests that read Experiment 46, is written with it, so that how it is read
is fixed before any run.

    python3 research/screen/evolution_plans.py           writes book/experiments/E45.json, E46.json and E47.json
    python3 research/screen/evolution_plans.py --show    says what it would write, and writes nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
from followup import ROUNDS, condition  # noqa: E402

EVOLUTION = os.path.join(HERE, "evolution.json")
HEALTH = os.path.join(HERE, "health.json")

#: Experiment 45: how many worlds of each kind, the eras, the fixed eras, the coins.
TOP, ENDS = 10, 5
ERAS = list(range(500, 3001, 500))
FIRST, MIDDLE, LAST = 250, 1500, 3000
COINS, NOISE_COINS, T = "17..19", "17..20", 100

#: Experiment 46: its size, how much bigger the bigger arm is, how long, its seeds.
TOKENS, BIGGER, ITERATIONS, SEEDS = 10000, 4, 20000, "4601..4603"
#: The most worker-hours, by the lab's cost model, Experiment 46 may take: about
#: three days on the lab's two workers at the pace the lab has measured, which
#: has run well behind the model. A world whose rules would take longer is passed over.
HOURS = 100


def lived_by_health() -> List[Dict[str, Any]]:
    """Round 3's worlds that lived, healthiest first (health's first order, the product)."""
    with open(HEALTH) as f:
        worlds = json.load(f)["worlds"]
    return sorted((w for w in worlds if w["fate"] == "lived"), key=lambda w: -w["health"]["product"])


def ranked() -> List[Dict[str, Any]]:
    """Round 3's worlds that lived, most evolution first."""
    with open(EVOLUTION) as f:
        rank = json.load(f)
    if rank.get("provisional"):
        raise SystemExit(f"{EVOLUTION} is provisional: wait for the whole rank")
    return sorted(rank["worlds"], key=lambda w: w["rank"])


def chosen_worlds() -> Tuple[List[str], Dict[str, List[str]]]:
    """Experiment 45's worlds, the rank's first, then the healthiest and least healthy not among them."""
    by_health = lived_by_health()
    groups = {"evolution": [w["id"] for w in ranked()[:TOP]],
              "healthiest": [w["id"] for w in by_health[:ENDS]],
              "least healthy": [w["id"] for w in by_health[-ENDS:]]}
    worlds: List[str] = []
    for ids in groups.values():
        worlds.extend(i for i in ids if i not in worlds)
    return worlds, groups


def e45() -> Dict[str, Any]:
    worlds, groups = chosen_worlds()
    after = [e for e in ERAS if e != MIDDLE]
    return {
        "id": "E45", "title": "Did the brains get better? Round 3's eras, with more coins",
        "thesis": {"claim": "", "why": "", "confirm": "", "refute": ""},
        "chosen": groups,
        "trials": {"kind": "contest", "T": T, "sets": [
            {"name": f"against {FIRST:,}",
             "grid": {"run": worlds, "pair": [[FIRST, e] for e in ERAS], "home": ["a", "b"], "seed": COINS}},
            {"name": f"against {MIDDLE:,}",
             "grid": {"run": worlds, "pair": [sorted((e, MIDDLE)) for e in after], "home": ["a", "b"], "seed": COINS}},
            {"name": "each against itself",
             "grid": {"run": worlds, "pair": [[e, e] for e in (FIRST, MIDDLE, LAST)], "home": ["a"],
                      "seed": NOISE_COINS}}]}}


def contests_a_world(plan: Dict[str, Any]) -> int:
    """How many contests a trials plan plays in each of its worlds."""
    return sum(len(s["grid"]["pair"]) * len(s["grid"]["home"]) * len(gol_plan.parse_seeds(s["grid"]["seed"]))
               for s in plan["trials"]["sets"])


def hours(plan: Dict[str, Any]) -> float:
    """
    A trials plan's worker-hours, from each world's median seconds an iteration
    over its last thousand, as the world recorded them (as Experiment 44's page
    reckoned its own).
    """
    import gol_costs
    import gol_store
    seconds = 0.0
    for run_id in plan["trials"]["sets"][0]["grid"]["run"]:
        rows = gol_costs.tail_rows(os.path.join(gol_store.run_dir(run_id), "stats.jsonl"), 1000)
        seconds += float(np.median([r["_cpu"] for r in rows if r.get("_cpu")]))
    return contests_a_world(plan) * plan["trials"]["T"] * seconds / 3600


def round3_spec(run_id: str) -> gol_plan.RunSpec:
    """The plan's run of round 3 that a run id names."""
    for name in ROUNDS.values():
        for spec in gol_plan.experiment_runs(name):
            if spec.run_id == run_id:
                return spec
    raise SystemExit(f"{run_id} is not a run of round 3")


def revolts(drawn: Dict[str, Any], base: Dict[str, Any]) -> bool:
    """Whether a world drawn with these settings plays with revolutions."""
    return bool(drawn.get("allow_revolutions", base["allow_revolutions"]))


def arm(name: str, start: Dict[str, Any], change: Dict[str, Any], base: Dict[str, Any],
        sizes: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    One arm: the chosen world's condition with `change` made, every setting at the
    baseline's own value left out (a plan refuses one, since it changes nothing).
    """
    drawn = {**start["set"], **change}
    out = {"name": name, "set": {k: v for k, v in drawn.items() if v != base.get(k)}}
    if sizes:
        out["sizes"] = sizes
    if "world" in start:
        out["world"] = dict(start["world"])
    return out


#: How a brain changes: what the faithful-copies arm takes from the world of round 3 whose copies are most faithful.
MUTATION = ("mutate_on_replication", "mutation_probability", "mutation_noise_std", "mutation_sparsity")
#: How often a mutated copy starts a family of its own, as the family-pool pilots had it (another-game.md).
TAG_RATE = 0.05


def faithful(drawn: Dict[str, Any], base: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    The faithful-copies arm's change: how brains change in the world of round 3
    whose brains are copied most faithfully (heredity on the brain tree), the
    first such world that changes them otherwise than `drawn` does.
    """
    for world in sorted(ranked(), key=lambda w: -w["values"]["heredity"]):
        theirs = {**base, **round3_spec(world["id"]).lab["set"]}
        change = {k: theirs[k] for k in MUTATION if theirs[k] != drawn[k]}
        if change:
            return world["id"], change
    raise SystemExit("every world of round 3 changes its brains as the chosen one does")


def cost(spec: gol_plan.RunSpec, runs: int) -> float:
    """Worker-hours, by the lab's cost model, of `runs` runs of a world's rules at TOKENS, three of them BIGGER."""
    import gol_costs
    known = gol_costs.costs()

    def hours(tokens: int) -> float:
        config = {**spec.config, **condition(spec, tokens).get("world", {}), "total_tokens": tokens}
        return gol_costs.predict(config, ITERATIONS, known)["seconds"] / 3600
    seeds = len(gol_plan.parse_seeds(SEEDS))
    return (runs - seeds) * hours(TOKENS) + seeds * hours(TOKENS * BIGGER)


def e46() -> Dict[str, Any]:
    base = gol_plan.baseline("B1")
    passed = []
    for world in ranked():
        spec = round3_spec(world["id"])
        if not revolts(spec.lab["set"], base):
            continue
        if cost(spec, 6 * len(gol_plan.parse_seeds(SEEDS))) <= HOURS:
            break
        passed.append(world["id"])
    else:
        raise SystemExit(f"no world of the rank plays with revolutions within {HOURS} worker-hours")
    start = condition(spec, TOKENS)
    drawn = {**base, **start["set"]}
    copies_from, copies = faithful(drawn, base)
    other = {"own": "largest", "largest": "own"}[drawn["self_message"]]
    bigger = condition(spec, TOKENS * BIGGER)
    arms = [
        arm("control", start, {}, base),
        arm(f"{BIGGER} times the tokens", bigger, {}, base, [TOKENS * BIGGER]),
        arm("faithful copies", start, copies, base),
        arm(f"message to oneself: {other}", start, {"self_message": other}, base),
        arm("revolutionary tokens for the coalition alone", start, {"revolt_tokens": "separate"}, base),
    ]
    # Stakes pooled by family need every token staked at once (allocation full).
    if drawn["allocation"] == "full":
        arms.append(arm("stakes pooled by family", start, {"pool": "family", "tag_rate": TAG_RATE}, base))
    return {
        "id": "E46", "title": "Why does evolution level off?",
        "thesis": {"claim": "", "why": "", "confirm": "", "refute": ""},
        "chosen": {"world": world["id"], "rank": world["rank"], "passed over as too slow": passed,
                   "faithful copies from": copies_from},
        "runs": {"baseline": "B1", "world": {"total_tokens": TOKENS}, "seeds": SEEDS, "iterations": ITERATIONS,
                 "record": {"heavy_every": 25, "checkpoint_every": 100, "archive_every": 500},
                 "conditions": arms}}


#: Experiment 47, the contests that read Experiment 46: its eras, each played
#: against the one before it, and the eras played against themselves for the noise.
STEPS = [500, 1000, 2500, 5000, 10000, 20000]
STILL = [1000, 5000, 20000]


def e47(runs: List[str]) -> Dict[str, Any]:
    """
    The contests in Experiment 46's runs, written with it so that how it is read
    is fixed before any run: each era against the era before it, three coins on
    each era's world, the last step also dealt in patches (a family whose kin
    stake as one stays whole), and three eras against themselves for the noise.
    """
    steps = [[a, b] for a, b in zip(STEPS, STEPS[1:])]
    return {
        "id": "E47", "title": "Why does evolution level off? The contests that read it",
        "trials": {"kind": "contest", "T": T, "sets": [
            {"name": "each era against the one before",
             "grid": {"run": runs, "pair": steps, "home": ["a", "b"], "seed": COINS}},
            {"name": "the last step, in patches",
             "grid": {"run": runs, "pair": steps[-1:], "home": ["a", "b"], "seed": COINS, "plant": "patch"}},
            {"name": "each against itself",
             "grid": {"run": runs, "pair": [[e, e] for e in STILL], "home": ["a"], "seed": NOISE_COINS}}]}}



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    first = e45()
    worlds = len(first["trials"]["sets"][0]["grid"]["run"])
    print(f"E45: {worlds} worlds, {contests_a_world(first)} contests each, "
          f"about {hours(first):.0f} worker-hours at the worlds' own pace")
    for plan in (first, e46(), None):
        if plan is None:
            # Experiment 47 plays in Experiment 46's runs, named by its plan as written.
            if not os.path.exists(os.path.join(ROOT, "book", "experiments", "E46.json")):
                print("E47 is written once E46 is")
                continue
            plan = e47([spec.run_id for spec in gol_plan.experiment_runs("E46")])
        path = os.path.join(ROOT, "book", "experiments", f"{plan['id']}.json")
        text = json.dumps(plan, indent=1) + "\n"
        if a.show:
            print(text)
        elif os.path.exists(path):
            print(f"{path} is written already: left as it is")
        else:
            with open(path, "w") as f:
                f.write(text)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
