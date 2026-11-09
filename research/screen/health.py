"""
Which of the screen's worlds are healthy? Three orders, for the author to check
(the author, 2026-11-07: "sort the simulations according to what you think
healthy could mean, most healthy to least healthy, and I will check the order";
2026-11-09: no lines, "a continuous scoring ... always the bigger the better and
always positive", combined "by adding scores, multiplying scores, or just by the
worst score", and "the union of the best of these rankings are candidates").

A world is read by the measures of the author's six criteria (MEASURES), each
averaged over iterations 100 to the end of its run, as the judge reads them
(research/screen/measures.py). A measure becomes a score by asking how many of
the living worlds scored with it the world does better than (`shares`): the best
scores 1, the worst 1/n of n worlds, and worlds that tie share their place. So
every score is positive and a bigger one is better; every difference counts, not
only one past a line; and every measure counts alike, whatever its units, which
is what lets the criteria be added, multiplied or compared at all. A criterion is
the geometric mean of its measures' scores, so that one poor measure pulls it
down more than one good one lifts it: a star is not decentralised, however even
its tokens.

A world's health comes in three orders over its six criteria (ORDERS):

    product  the six multiplied: a strong criterion makes up for a weak one in part
    sum      the six added: a strong criterion makes up for a weak one in full
    worst    the weakest of the six: nothing makes up for it

Each order's number is put on the criteria's own scale, from 0 to 1, without
changing the order: the sum as the six's average, the product as their
geometric average (its sixth root), so that a world's three numbers can be read
side by side, the worst never above the geometric average, nor that above the
average.

The sixth criterion is the author's (2026-11-08, 2026-11-09): the share of all
tokens culled in a game, on average. Culled are the agents the engine removes for
being cut off from the largest piece of the network ("Culled" in the web app's
statistics). The starved hold nothing, so every token a game's cleanup shares out
was a culled agent's, and the run's statistics record it (`redistributed`;
under link_token it also holds the token of a link between two of the dead, a
token or two in a game). Counted in tokens, an agent cut off alone at the edge,
a leaf, weighs little, and a whole branch weighs what it held.

A few of the author's measures are left out, each for its reason (DROPPED). A
world that died has no health: the dead come last, the longest-lived first. The
two halves of every run are also scored, together with every other half, to say
whether a world grew healthier or sicker.

    python3 research/screen/health.py E41 E42 E40 [--read 3]
        prints the orders and writes research/screen/health.json; with --read, first makes every reading a
        world still lacks, on that many processes

2026-11-09, later: links an agent come back, as a measure that is low at both ends (the author: "fewer is
not exactly better, because at some point it becomes a tree; is there a measure that is high when not tree
like and not fully connected?"): agents on loops per link, the share of the agents in the 2-core over half the
links an agent holds, in the resilient criterion beside the cuts. And what each world's statistics say of its
games is kept beside the readings (GAMES), so that the worlds can be scored again while their runs are on a
drive that is not plugged in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gol_record  # noqa: E402
import gol_store  # noqa: E402

import criteria as C  # noqa: E402
import judge as J  # noqa: E402
import measures as M  # noqa: E402

#: The criteria: the author's after "alive", which every world in the orders meets, and the author's sixth.
CRITERIA = ["decentralised", "resilient", "space", "diverse", "uneven", "culls"]

#: Each criterion's measures, and which way is better: more, less, or nearer three dimensions.
MEASURES: List[Dict[str, Any]] = [
    {"key": "hub", "criterion": "decentralised", "better": "less",
     "name": "links of the best-linked agent, as a share of the others"},
    {"key": "richest", "criterion": "decentralised", "better": "less",
     "name": "tokens of the richest agent, as a share of all tokens"},
    {"key": "core", "criterion": "resilient", "better": "more", "name": "share of the agents in the 2-core"},
    {"key": "agent_cut", "criterion": "resilient", "better": "less",
     "name": "most of the world one agent's loss cuts off"},
    {"key": "link_cut", "criterion": "resilient", "better": "less",
     "name": "most of the world one link's loss cuts off"},
    # The author, 2026-11-09: links an agent should not be too many, but fewer is not simply better, since
    # at some point the world is a tree. The share of agents on loops, per link an agent holds, is low at
    # both ends: a tree has no loops, and a world crowded with links spends many on each agent's loops.
    {"key": "lean", "criterion": "resilient", "better": "more",
     "name": "agents on loops per link: the share of the agents in the 2-core, over half the links an agent "
             "holds (a ring scores 1, a tree 0, a world of 28 links an agent 0.07)"},
    {"key": "width", "criterion": "space", "better": "more", "name": "typical distance, over the random twin's"},
    {"key": "dimension", "criterion": "space", "better": "near 3", "name": "dimension, by how balls grow"},
    {"key": "apart", "criterion": "diverse", "better": "more",
     "name": "same-position share: of two agents in the same situation, how often they act measurably "
             "differently"},
    {"key": "tokens_alike", "criterion": "diverse", "better": "less",
     "name": "share of the agents holding the most common number of tokens"},
    {"key": "core_links_alike", "criterion": "diverse", "better": "less",
     "name": "share of the 2-core's agents with its most common number of links"},
    {"key": "ages_alike", "criterion": "diverse", "better": "less", "name": "share of the agents of the most common age"},
    {"key": "children_alike", "criterion": "diverse", "better": "less",
     "name": "share of the agents making the more common choice, to have a child or not"},
    {"key": "gini_random", "criterion": "uneven", "better": "more",
     "name": "Gini of the tokens, over the Gini of tokens dealt out at random at the same mean holding"},
    {"key": "culled_tokens", "criterion": "culls", "better": "less",
     "name": "share of all tokens culled in a game, with the agents cut off from the largest piece"},
]
#: The author's measures left out, with the reason.
DROPPED = {"links": "links per agent asked for a range, 3 to 12, and neither more nor fewer is simply better: "
                    "agents on loops per link (`lean`) reads it against the loops the links make",
           "max_tokens": "the largest holding is the richest agent's share of all tokens, counted the other way "
                         "round, so the two would cancel; its line kept out a world of paupers, which no living "
                         "world is",
           "genotypes_alike": "a genotype is renewed by every mutation, so how many agents share one says how "
                              "seldom brains mutate, not how differently agents act; the same-position share "
                              "measures that",
           "gini": "drawn at B2's most even world, a baseline the screen no longer compares with; the Gini over a "
                   "random deal's at the same mean holding reads the same thing against chance"}
#: The three ways to put the criteria together, each an order of its own.
ORDERS = {"product": "the six multiplied (read as their geometric average): a strong criterion makes up for a "
                      "weak one in part",
          "sum": "the six added (read as their average): a strong criterion makes up for a weak one in full",
          "worst": "the weakest of the six: nothing makes up for it"}


def oriented(measure: Dict[str, Any], value: Optional[float]) -> float:
    """A measure's value turned so that more is better; a measure not read counts as the worst."""
    if value is None or not math.isfinite(value) or (measure["better"] == "near 3" and value <= 0):
        return -math.inf
    if measure["better"] == "more":
        return value
    if measure["better"] == "less":
        return -value
    return -abs(math.log(value / 3.0))      # as far from three dimensions at two as at four and a half


def shares(values: List[float]) -> List[float]:
    """Each value's place among them all, as the share of them it does better than: the best 1, the worst 1/n."""
    n = len(values)
    return [(sum(v < x for v in values) + (sum(v == x for v in values) + 1) / 2) / n for x in values]


def score(table: List[Dict[str, Optional[float]]]) -> List[Dict[str, Any]]:
    """
    Every world's scores, from the averages of all the worlds scored together: each measure's share, each
    criterion's geometric mean of them, and the health of each of the three orders, on the criteria's scale.
    """
    each = {m["key"]: shares([oriented(m, a.get(m["key"])) for a in table]) for m in MEASURES}
    out = []
    for i in range(len(table)):
        criteria = {c: float(math.exp(np.mean([math.log(each[m["key"]][i]) for m in MEASURES
                                                if m["criterion"] == c]))) for c in CRITERIA}
        out.append({"measures": {k: v[i] for k, v in each.items()}, "criteria": criteria,
                    "health": {"product": float(math.prod(criteria.values()) ** (1 / len(CRITERIA))),
                               "sum": float(np.mean(list(criteria.values()))), "worst": float(min(criteria.values()))}})
    return out


# ---------------------------------------------------------------------------
# What the half-cut exponent reads for networks of known shape
# ---------------------------------------------------------------------------

#: Random networks in a plane and in a cube, at the densities of the screen's worlds and above: (dimensions,
#: links an agent asked for). The half-cut exponent of a lattice is (d - 1) / d, but a sparse network halves
#: more cheaply: at five links an agent a plane reads about 0.38 and a cube 0.56, and only at eight do they
#: reach a lattice's 1/2 and 2/3. That is why the exponent is not one of space's measures.
CALIBRATION = [(2, 3), (2, 5), (2, 8), (3, 3), (3, 5), (3, 8)]
CALIBRATION_AGENTS, CALIBRATION_SEED = 4000, 1


def geometric(n: int, dim: int, links: float, seed: int) -> np.ndarray:
    """
    A random network in space: n agents put at random in a unit square (dim 2) or cube (dim 3), each joined to
    every agent nearer than the distance at which an agent has `links` others on average. Its links as pairs.
    """
    rng = np.random.default_rng(seed)
    points = rng.random((n, dim))
    reach = (links / (n * math.pi ** (dim / 2) / math.gamma(dim / 2 + 1))) ** (1 / dim)
    cells = np.floor(points / reach).astype(np.int64)
    side = int(cells.max()) + 2
    key = sum(cells[:, i] * side ** i for i in range(dim))
    order = np.argsort(key, kind="stable")
    first = {}
    for i, k in zip(order, key[order]):
        first.setdefault(int(k), []).append(int(i))
    pairs = []
    steps = np.array(np.meshgrid(*[(-1, 0, 1)] * dim)).reshape(dim, -1).T
    for k, members in first.items():
        cell = np.array([(k // side ** i) % side for i in range(dim)])
        a = np.array(members)
        for step in steps:
            other = first.get(int(sum((cell[i] + step[i]) * side ** i for i in range(dim))))
            if not other:
                continue
            b = np.array(other)
            near = ((points[a][:, None, :] - points[b][None, :, :]) ** 2).sum(-1) <= reach * reach
            i, j = np.nonzero(near)
            keep = a[i] < b[j]
            pairs.append(np.stack([a[i][keep], b[j][keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)


def calibration() -> List[Dict[str, Any]]:
    """The half-cut exponent of each network of CALIBRATION, with its links an agent and its largest piece."""
    out = []
    for dim, links in CALIBRATION:
        pairs = geometric(CALIBRATION_AGENTS, dim, links, CALIBRATION_SEED)
        cut = M.half_cut(CALIBRATION_AGENTS, pairs)
        out.append({"dimensions": dim, "links_asked": links, "links": 2 * len(pairs) / CALIBRATION_AGENTS,
                    "agents": cut["agents"], "crossing": cut["crossing"], "exponent": cut["exponent"],
                    "lattice": (dim - 1) / dim})
    return out


# ---------------------------------------------------------------------------
# A world's numbers, from the judge's readings
# ---------------------------------------------------------------------------

def rows_of(have: Dict[M.Item, Dict[str, Any]], wanted: List[M.Item]) -> List[Dict[str, Any]]:
    """Every row a world's readings hold: one an iteration, one a heavy frame, one a kept world."""
    rows: List[Dict[str, Any]] = []
    for item in wanted:
        reading = have[item]
        rows += reading["rows"] + reading["frames"] if item[1] == "series" else [reading]
    return rows


def random_gini(mean: float) -> float:
    """The Gini of whole tokens dealt out at random, one at least, at a mean of `mean` an agent (geometric)."""
    return (mean - 1.0) / (2.0 * mean - 1.0) if mean > 1.0 else 0.0


#: What each world's statistics say of its games, kept beside the readings: the runs may be on a drive that is not
#: plugged in when the worlds are scored again.
GAMES = os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "games")


def games(run_id: str, tokens: int) -> Dict[int, Dict[str, float]]:
    """
    Every game of a world, from its statistics: the mean holding after it (tokens a link locks are held by
    none), the share of all tokens its cleanup culled (`redistributed`: the starved hold nothing), and the share
    of its agents culled (`nodes` are those left after the cleanup). Read from the run when it can be, and
    kept; from what was kept when the run cannot be read.
    """
    kept = os.path.join(GAMES, f"{run_id}.json")
    try:
        out = _games(run_id, tokens)
    except OSError:
        if not os.path.exists(kept):
            raise
        with open(kept) as f:
            return {int(t): g for t, g in json.load(f).items()}
    os.makedirs(GAMES, exist_ok=True)
    with open(kept, "w") as f:
        json.dump(out, f)
    return out


def _games(run_id: str, tokens: int) -> Dict[int, Dict[str, float]]:
    out = {}
    for r in gol_record.read_stats(run_id):
        if r["phase"] == 2 and r.get("meanTokens") is not None:
            before = (r.get("nodes") or 0) + (r.get("starved") or 0) + (r.get("orphaned") or 0) + (r.get("aged") or 0)
            out[int(r["iteration"])] = {"mean": float(r["meanTokens"]),
                                        "culled": float(r.get("redistributed") or 0) / tokens,
                                        "culled_agents": float(r.get("orphaned") or 0) / before if before else 0.0}
    return out


def averages(rows: List[Dict[str, Any]], tokens: int, start: int, stop: int,
             game: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[str, Optional[float]]:
    """The judge's averages over iterations start to stop, with the measures read beside them."""
    values: Dict[str, List[float]] = {}
    agents: List[float] = []
    for row in rows:
        if not start <= row["t"] < stop:
            continue
        for key, v in row.items():
            if key != "t" and v is not None and math.isfinite(v):
                values.setdefault(key, []).append(float(v))
        if row.get("max_tokens") is not None:
            values.setdefault("richest", []).append(row["max_tokens"] / tokens)
        if row.get("agents") is not None:
            agents.append(float(row["agents"]))
        if row.get("core") is not None and row.get("links"):
            values.setdefault("lean", []).append(2.0 * row["core"] / row["links"])
        m = (game or {}).get(row["t"], {}).get("mean")
        if row.get("gini") is not None and m is not None and random_gini(m) > 0:
            values.setdefault("gini_random", []).append(row["gini"] / random_gini(m))
    out: Dict[str, Optional[float]] = {key: float(np.mean(v)) for key, v in sorted(values.items())}
    if agents:
        out["low"] = float(np.quantile(agents, 0.05))
        out["median_agents"] = float(np.median(agents))
    played = [g for t, g in (game or {}).items() if start <= t < stop]
    out["culled_tokens"] = float(np.mean([g["culled"] for g in played])) if played else None
    out["culled_agents"] = float(np.mean([g["culled_agents"] for g in played])) if played else None
    return out


def world(w: Dict[str, Any], have: Dict[M.Item, Dict[str, Any]]) -> Dict[str, Any]:
    """One world's averages: over its whole run, and over each half of it."""
    wanted = M.items(w["id"], w["ended_at"], C.SETTLE)
    missing = [i for i in wanted if i not in have]
    out = {k: w[k] for k in ("round", "id", "seed", "fate", "ended_at", "until", "set")}
    out["alive"] = w["fate"] == "lived"
    out["stopped"] = w["fate"] == "too slow"         # a round may stop its slowest worlds: neither lived nor died
    out["read"] = not missing
    if missing or not out["alive"]:
        return out
    rows = rows_of(have, wanted)
    tokens = int(w["tokens"])
    end = w["ended_at"] + 1
    half = (C.SETTLE + end) // 2            # the two halves of a run, scored apart: 100 to 1,550 and on, in round 3
    game = games(w["id"], tokens)
    out.update(averages=averages(rows, tokens, C.SETTLE, end, game),
               half_averages=[averages(rows, tokens, C.SETTLE, half, game), averages(rows, tokens, half, end, game)])
    return out


def order(worlds: List[Dict[str, Any]], key: str = "product") -> List[Dict[str, Any]]:
    """
    Most healthy first, by one of the three orders: the living by health, ties by their weakest criterion;
    then any living world not yet read; then the worlds stopped as too slow, which neither lived nor died; then
    the dead, longest-lived first.
    """
    living = [w for w in worlds if w["alive"] and w.get("health")]
    unread = [w for w in worlds if w["alive"] and not w.get("health")]
    stopped = [w for w in worlds if w.get("stopped")]
    dead = [w for w in worlds if not w["alive"] and not w.get("stopped")]
    living.sort(key=lambda w: (-w["health"][key], -w["health"]["worst"], w["id"]))
    dead.sort(key=lambda w: (-w["ended_at"], w["id"]))
    return (living + sorted(unread, key=lambda w: w["id"]) + sorted(stopped, key=lambda w: w["id"]) + dead)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rounds", nargs="+")
    ap.add_argument("--read", type=int, default=0, metavar="JOBS",
                    help="first make every reading the worlds lack, on this many processes (the judge's own "
                         "readings, research/screen/measures.py), whatever criteria their round was judged by")
    a = ap.parse_args()
    # The worlds drawn at random only: a round's reference worlds are not candidates.
    worlds = [w for w in J.worlds_of(a.rounds) if w["drawn"] and J.finished(w)]
    tokens = {name: int(J.gol_plan.read_plan(name)["screen"]["tokens"]) for name in a.rounds}
    for w in worlds:
        w["tokens"] = tokens[w["round"]]
    have = M.kept([w["id"] for w in worlds])
    if a.read:
        wanted = [i for w in worlds for i in M.items(w["id"], w["ended_at"], C.SETTLE)]
        have = J.gather(wanted, a.read, M.read, have, M.keep)
    scored = [world(w, have) for w in worlds]
    # Every living world is scored against all the others; each half of a run against every half of every run.
    living = [w for w in scored if w["alive"] and w["read"]]
    for w, s in zip(living, score([w["averages"] for w in living])):
        w.update(s)
    halves = score([w["half_averages"][0] for w in living] + [w["half_averages"][1] for w in living])
    for i, w in enumerate(living):
        w["halves"] = [halves[i]["health"], halves[len(living) + i]["health"]]
        del w["half_averages"]
    ranked = order(scored)
    for key in ORDERS:
        for i, w in enumerate(order(scored, key), 1):
            w.setdefault("ranks", {})[key] = i
    out = {"rounds": a.rounds, "criteria": CRITERIA, "measures": MEASURES, "dropped": DROPPED, "orders": ORDERS,
           "calibration": calibration(), "worlds": ranked}
    with open(os.path.join(HERE, "health.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    unread = [w["id"] for w in ranked if not w["read"]]
    print(f"{len(ranked)} worlds; {len(unread)} not yet read" + (f" ({', '.join(unread[:5])} ...)" if unread else ""))
    print("rank: product  sum  worst   world    product    sum  worst  " + "  ".join(f"{c[:6]:>6s}" for c in CRITERIA))
    for w in ranked:
        if not w["alive"]:
            how = "stopped as too slow" if w["stopped"] else "died"
            print(f"                       s{w['seed']}  {how} at {w['ended_at']}")
            continue
        if not w.get("health"):
            print(f"                       s{w['seed']}  not read")
            continue
        h, r = w["health"], w["ranks"]
        print(f"      {r['product']:7d} {r['sum']:4d} {r['worst']:6d}   s{w['seed']}  {h['product']:9.3f} {h['sum']:6.3f} "
              f"{h['worst']:6.3f}  " + "  ".join(f"{w['criteria'][c]:6.2f}" for c in CRITERIA))


if __name__ == "__main__":
    main()
