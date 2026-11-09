"""
How the drawn settings shape the screen's worlds (the author, 2026-11-07:
"analyze and compare the influence of the different simulation settings on the
results. Maybe you can exclude some specific setting combination").

Every world's settings were drawn independently of each other, so the worlds
drawn with one value of a setting and those drawn with another differ, in
everything else, by chance alone. A setting's values can therefore be compared
directly, one setting at a time, with no model of the others: what differs
beyond chance is the setting's doing.

For every setting and every outcome:
  - an outcome that happens or not (a world died): how often it happened under
    each value, and the chance of a spread across the values at least this
    uneven were the outcome shuffled at random among the worlds (a permutation
    test of the chi-square statistic);
  - an outcome that is a number (health, size, how differently agents act): its
    median under each value, and the chance of differences among the values at
    least this large were the numbers shuffled among the worlds (the
    Kruskal-Wallis statistic, by permutation).
A setting is compared only among the worlds where it could matter (the mate's
kind only where brains mix, as the sampler draws it). Many settings and
outcomes are tested at once, so some small chances arise by luck alone: beside
each is its false discovery rate (Benjamini-Hochberg) among all the settings
tested on that outcome.

Combinations: every pair of drawn values, for how often the worlds that had
both died, against the chance of that many deaths among that many worlds drawn
at random; reported only where the pair says more than either value alone.

Simplicity (the author: "simpler settings with clear mechanisms are preferred
in general"): how many of a world's mechanisms are switched on beyond the
simplest rules (SIMPLEST), against its health.

    python3 research/screen/effects.py      reads research/screen/health.json and the dossiers,
                                            writes research/screen/effects.json and prints the tables
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_screen  # noqa: E402

import candidates as K  # noqa: E402
import families as FA  # noqa: E402
import funnel as FN  # noqa: E402

#: Shuffles per test, and their seed.
SHUFFLES = 20000
SEED = 11
#: A pair of values is tried only where at least this many worlds had both.
PAIR_AT_LEAST = 4

#: The simplest value of each mechanism the sampler draws: a world's other values switch something on.
SIMPLEST = {"message_amount": 0, "random_input_amount": 0, "contest": "largest", "allow_revolutions": False,
            "allow_handover": False, "allow_gifting": False, "mend_on_death": False, "mix_at_birth": False,
            "redistribution": "uniform", "link_search": "off", "valence": 0, "link_token": False,
            "link_upkeep": "off", "allocation": "full"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _groups(values: Sequence[Any]) -> Tuple[List[str], np.ndarray]:
    labels = [json.dumps(v) for v in values]
    names = sorted(set(labels))
    onehot = np.array([[l == n for n in names] for l in labels], dtype=float)
    return names, onehot


def chi_square(outcome: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """The chi-square statistic of an outcome (0/1, one row per shuffle) across the groups."""
    n = onehot.sum(axis=0)
    p = outcome.mean(axis=-1, keepdims=True)
    hits = outcome @ onehot
    expected = n * p
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ((hits - expected) ** 2 / (expected * (1 - p))).sum(axis=-1)
    return np.nan_to_num(stat)


def kruskal(ranks: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """The Kruskal-Wallis statistic of ranks (one row per shuffle) across the groups (ties averaged)."""
    n = onehot.sum(axis=0)
    N = ranks.shape[-1]
    sums = ranks @ onehot
    return 12.0 / (N * (N + 1)) * (sums ** 2 / n).sum(axis=-1) - 3 * (N + 1)


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def permutation_test(y: np.ndarray, groups: Sequence[Any], binary: bool,
                     rng: np.random.RandomState) -> Tuple[float, float]:
    """
    The chance, over shuffles of y among the worlds, of a statistic at least as large as the one observed;
    and the share of y's variation the groups account for: for a number, epsilon squared, the Kruskal-Wallis
    statistic over n - 1; for a yes or no, phi squared, the chi-square statistic over n.
    """
    _, onehot = _groups(groups)
    if onehot.shape[1] < 2:
        return 1.0, 0.0
    base = y.astype(float) if binary else average_ranks(y)
    stat = chi_square if binary else kruskal
    observed = stat(base[None, :], onehot)[0]
    shuffled = base[np.argsort(rng.rand(SHUFFLES, base.size), axis=1)]
    null = stat(shuffled, onehot)
    size = observed / base.size if binary else observed / (base.size - 1)
    return float((1 + (null >= observed - 1e-12).sum()) / (1 + SHUFFLES)), float(size)


def false_discovery(chances: List[float]) -> List[float]:
    """Benjamini-Hochberg: the false discovery rate at which each chance would be called a finding."""
    m = len(chances)
    order = np.argsort(chances)
    q = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        k = order[i]
        running = min(running, chances[k] * m / (i + 1))
        q[k] = running
    return q.tolist()


def tail(total: int, hits: int, drawn: int, got: int) -> float:
    """The chance of `got` or more hits among `drawn` worlds taken at random from `total` holding `hits`."""
    out = 0.0
    for k in range(got, min(drawn, hits) + 1):
        out += math.comb(hits, k) * math.comb(total - hits, drawn - k)
    return out / math.comb(total, drawn)


# ---------------------------------------------------------------------------
# The worlds
# ---------------------------------------------------------------------------

def settings_of(rounds: List[str]) -> Tuple[List[gol_screen.Setting], Dict[str, Dict[str, Any]]]:
    """The sampler's settings, with the allocation of the step rounds as one more, and each world's values."""
    version = {int(gol_plan.read_plan(r)["screen"]["sampler"]) for r in rounds}.pop()
    space = list(gol_screen.space(version)) + [("allocation", "choice", ("full", "weakest_first", "strongest_first"),
                                                None)]
    out = {}
    for name in rounds:
        fixed = gol_plan.read_plan(name)["screen"].get("fixed") or {}
        for spec in gol_plan.experiment_runs(name):
            full = gol_screen.full({**spec.lab["set"], **{k: v for k, v in spec.lab["world"].items()
                                                         if k in {s[0] for s in space}}}, version)
            full["allocation"] = fixed.get("allocation", "full")
            out[spec.run_id] = full
    return space, out


def outcomes(health: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, bool, str]]]:
    """Every world's outcomes, and the outcomes: key, name, whether yes/no, and which worlds it is read in."""
    worlds = []
    for w in health["worlds"]:
        row = {"id": w["id"], "round": w["round"], "seed": w["seed"], "died": not w["alive"]}
        if w.get("health"):
            a = w["averages"]
            row.update(health=w["health"]["product"], **{f"c_{c}": v for c, v in w["criteria"].items()},
                       agents=math.log10(a["median_agents"]) if a.get("median_agents") else None,
                       apart=a.get("apart"), gini=a.get("gini"), core=a.get("core"), width=a.get("width"),
                       culls=a.get("culled_tokens"))
            dossier = K.kept(w["id"])
            if dossier:
                n = FN.numbers({"dossier": dossier, "families": FA.kept(w["id"]), "tokens": 30000,
                                "until": w["until"]})
                row.update(heredity=n.get("heredity"), generations=n.get("generations"),
                           population=(math.log10(n["effective_population"]) if n.get("effective_population")
                                       else None),
                           novelty=n.get("late_novelty"), fertility=n.get("fertility"))
        worlds.append(row)
    keys = [("died", "died", True, "all"), ("died_rest", "died, valence 2 aside", True, "all"),
            ("health", "health (the six criteria multiplied, as their geometric average)", False, "living"),
            ("c_decentralised", "decentralised", False, "living"),
            ("c_resilient", "resilient", False, "living"), ("c_space", "almost three-dimensional", False, "living"),
            ("c_diverse", "diverse", False, "living"), ("c_uneven", "not too even", False, "living"),
            ("c_culls", "few tokens culled", False, "living"),
            ("agents", "agents (log10 of the median)", False, "living"),
            ("apart", "same-position share", False, "living"), ("gini", "Gini of the tokens", False, "living"),
            ("culls", "share of all tokens culled a game", False, "living"),
            ("heredity", "heredity index", False, "living"), ("generations", "generations", False, "living"),
            ("population", "effective population (log10)", False, "living"),
            ("novelty", "late novelty of the line of descent", False, "living"),
            ("fertility", "fertility passed on", False, "living")]
    return worlds, keys


def effects(rounds: List[str], health: Dict[str, Any]) -> Dict[str, Any]:
    """
    Every setting against every outcome, over the worlds of the rounds named. The step rounds play
    eight draws twice each, so the drawn settings are compared over Experiment 40's own worlds, where
    every world is a draw of its own, and the allocation apart (`allocation`).
    """
    rng = np.random.RandomState(SEED)
    space, values = settings_of(rounds)
    worlds, keys = outcomes(health)
    # In the order of their names, so that the shuffles, and the chances, are the same however the worlds
    # were listed.
    worlds = sorted((w for w in worlds if w["round"] in rounds), key=lambda w: w["id"])
    # The deaths a cap of two links does not explain: every world with that cap died (see the page).
    for w in worlds:
        w["died_rest"] = None if values[w["id"]].get("valence") == 2 else w["died"]
    tests = []
    for key, name, binary, among in keys:
        rows = []
        for setting in space:
            s = setting[0]
            mine = [w for w in worlds if w["id"] in values and gol_screen._applies(setting[3], values[w["id"]])
                    and (among == "all" or not w["died"]) and w.get(key) is not None]
            if len(mine) < 6:
                continue
            groups = [values[w["id"]][s] for w in mine]
            if len({json.dumps(g) for g in groups}) < 2:
                continue
            y = np.array([float(w[key]) for w in mine])
            chance, size = permutation_test(y, groups, binary, rng)
            by = {}
            for g in sorted({json.dumps(g) for g in groups}):
                ys = [v for v, gg in zip(y, groups) if json.dumps(gg) == g]
                by[g] = {"worlds": len(ys), ("rate" if binary else "median"): float(np.mean(ys) if binary
                                                                                    else np.median(ys))}
            rows.append({"setting": s, "worlds": len(mine), "chance": chance, "size": size, "by": by})
        for row, q in zip(rows, false_discovery([r["chance"] for r in rows])):
            row["fdr"] = q
        tests.append({"outcome": key, "name": name, "binary": binary, "among": among,
                      "rows": sorted(rows, key=lambda r: r["chance"])})
    living = [w for w in worlds if not w["died"]]
    return {"tests": tests, "quiet": quiet(tests), "pairs": pairs_for_death(space, values, worlds),
            "simplicity": simplicity(values, worlds), "tree": tree(space, values, living), "worlds": worlds}


#: A setting is quiet when no test of it on any outcome had a chance below this.
QUIET = 0.05


def quiet(tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The settings no outcome showed any sign of: no test of them, on any outcome, with a chance below QUIET.
    Candidates to fix at their simplest value: in 84 worlds a small effect can hide, so quiet is not proof
    of none.
    """
    smallest: Dict[str, Tuple[float, str]] = {}
    for t in tests:
        for r in t["rows"]:
            if r["setting"] not in smallest or r["chance"] < smallest[r["setting"]][0]:
                smallest[r["setting"]] = (r["chance"], t["outcome"])
    ranked = sorted(smallest.items(), key=lambda x: -x[1][0])
    return [{"setting": s, "smallest_chance": c, "on": o} for s, (c, o) in ranked if c >= QUIET]


def pairs_for_death(space, values, worlds) -> List[Dict[str, Any]]:
    """
    Every pair of drawn values whose worlds died more often than either value's alone would say, among
    the deaths a cap of two links does not explain (every world with that cap died).
    """
    have = [w for w in worlds if w["id"] in values and w.get("died_rest") is not None]
    total, dead = len(have), sum(w["died_rest"] for w in have)
    conditions = []
    for setting in space:
        s = setting[0]
        for v in {json.dumps(values[w["id"]][s]) for w in have if gol_screen._applies(setting[3], values[w["id"]])}:
            ids = {w["id"] for w in have if gol_screen._applies(setting[3], values[w["id"]])
                   and json.dumps(values[w["id"]][s]) == v}
            conditions.append((s, v, ids))
    died = {w["id"] for w in have if w["died_rest"]}
    rate = lambda ids: len(ids & died) / len(ids) if ids else 0.0
    found = []
    for (s1, v1, a), (s2, v2, b) in itertools.combinations(conditions, 2):
        if s1 == s2:
            continue
        both = a & b
        if len(both) < PAIR_AT_LEAST:
            continue
        got = len(both & died)
        if got <= max(rate(a), rate(b)) * len(both):
            continue
        found.append({"pair": [[s1, json.loads(v1)], [s2, json.loads(v2)]], "worlds": len(both), "died": got,
                      "alone": [round(rate(a), 3), round(rate(b), 3)], "chance": tail(total, dead, len(both), got)})
    for row, q in zip(found, false_discovery([f["chance"] for f in found])):
        row["fdr"] = q
    return sorted(found, key=lambda f: f["chance"])[:40]


#: The tree of combinations: no leaf with fewer worlds than this, and no more than this many splits deep.
LEAF, DEPTH = 6, 3


def tree(space, values, worlds, key: str = "health") -> Dict[str, Any]:
    """
    A small regression tree of an outcome over the living worlds: at each step the one drawn value
    (a setting equal to a value, or not) that best divides the worlds into two groups as unlike in the
    outcome as possible (the largest drop in the squared differences from each group's mean), so long
    as each group keeps LEAF worlds; at most DEPTH steps deep. Combinations of values show as paths.
    """
    have = [w for w in worlds if w["id"] in values and w.get(key) is not None]
    tests = []
    for setting in space:
        s = setting[0]
        for v in sorted({json.dumps(values[w["id"]][s]) for w in have}):
            tests.append((s, v))

    def grow(group: List[Dict[str, Any]], depth: int) -> Dict[str, Any]:
        y = np.array([w[key] for w in group], dtype=float)
        node = {"worlds": len(group), "mean": float(y.mean()), "median": float(np.median(y))}
        if depth == DEPTH or len(group) < 2 * LEAF:
            return node
        best, gain = None, 0.0
        base = ((y - y.mean()) ** 2).sum()
        for s, v in tests:
            inside = np.array([json.dumps(values[w["id"]][s]) == v for w in group])
            if inside.sum() < LEAF or (~inside).sum() < LEAF:
                continue
            a, b = y[inside], y[~inside]
            drop = base - ((a - a.mean()) ** 2).sum() - ((b - b.mean()) ** 2).sum()
            if drop > gain:
                best, gain = (s, v, inside), drop
        if best is None:
            return node
        s, v, inside = best
        node.update(split=[s, json.loads(v)], explained=float(gain / base) if base else 0.0,
                    yes=grow([w for w, i in zip(group, inside) if i], depth + 1),
                    no=grow([w for w, i in zip(group, inside) if not i], depth + 1))
        return node

    return grow(have, 0)


def show_tree(node: Dict[str, Any], indent: str = "") -> List[str]:
    out = [f"{indent}{node['worlds']} worlds, mean {node['mean']:.2f}"]
    if "split" in node:
        s, v = node["split"]
        out += [f"{indent}  if {s} = {v} (explains {node['explained']:.0%}):"] + show_tree(node["yes"], indent + "    ")
        out += [f"{indent}  else:"] + show_tree(node["no"], indent + "    ")
    return out


def simplicity(values, worlds) -> Dict[str, Any]:
    """How many mechanisms each world switches on beyond SIMPLEST, against its health."""
    rows = []
    for w in worlds:
        if w["id"] not in values:
            continue
        full = values[w["id"]]
        on = sorted(k for k, v in SIMPLEST.items() if full.get(k) != v)
        rows.append({"id": w["id"], "seed": w["seed"], "round": w["round"], "on": on, "count": len(on),
                     "died": w["died"], "health": w.get("health")})
    living = [r for r in rows if r["health"] is not None]
    x = np.array([r["count"] for r in living], dtype=float)
    y = np.array([r["health"] for r in living], dtype=float)
    rho = float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]) if len(living) > 2 else None
    return {"simplest": SIMPLEST, "worlds": rows, "spearman": rho}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--health", default=os.path.join(HERE, "health.json"))
    ap.add_argument("--rounds", nargs="+", default=["E40"])
    a = ap.parse_args()
    health = json.load(open(a.health))
    out = effects(a.rounds, health)
    with open(os.path.join(HERE, "effects.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for t in out["tests"]:
        print(f"\n{t['name']} ({t['among']} worlds):")
        for r in t["rows"][:8]:
            key = "rate" if t["binary"] else "median"
            by = "  ".join(f"{g}: {d[key]:.2f} ({d['worlds']})" for g, d in r["by"].items())
            print(f"  {r['setting']:22s} chance {r['chance']:.4f}  fdr {r['fdr']:.3f}  size {r['size']:.2f}   {by}")
    print("\npairs that die more than either value alone (valence 2 aside):")
    for p in out["pairs"][:15]:
        print(f"  {p['pair']}  {p['died']}/{p['worlds']} died (alone {p['alone']})  chance {p['chance']:.2g}  "
              f"fdr {p['fdr']:.2f}")
    s = out["simplicity"]
    print(f"\nmechanisms on vs health: Spearman {s['spearman']}")
    print("\nquiet settings (no chance below %.2f on any outcome):" % QUIET,
          ", ".join(f"{q['setting']} ({q['smallest_chance']:.2f})" for q in out["quiet"]))
    print("\nthe tree of health:")
    print("\n".join(show_tree(out["tree"])))


if __name__ == "__main__":
    main()
