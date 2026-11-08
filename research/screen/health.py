"""
Which of the screen's worlds are healthy? A proposed order, for the author to
check and question (the author, 2026-11-07: "sort the simulations according to
what you think healthy could mean, most healthy to least healthy, and I will
check the order").

The author's criteria (research/screen/criteria.py) say yes or no: a world
meets a line or it does not. To put worlds in order, each line is read instead
as how much of what it asks a world has, from nothing to all of it:

    progress = what the world has / what the line asks for, at most 1

Both are counted from the line's floor: the value a world has when it has none
of what the line asks for. That is 0 for most numbers; for links per agent it
is a tree's 2 (a network in one piece with fewer links has no loop at all); for
width it is the random twin's 1; for the largest holding it is one token; for
how alike the agents are it is "all alike", 1, from which the share unlike is
counted. A line that asks for at most some amount (a hub, a cut) is met to the
line's amount over the world's: a world whose best-linked agent reaches 30% of
the others, against a line of 5%, is met to 5/30 = 0.17. A missing number (no
width under a hundred agents) meets nothing.

A criterion is met as far as its weakest line, as a criterion is met only when
all its lines are; a world's health is the sum of its five criteria's scores
(the author's after "alive"), so it counts a criterion met as one point and one
missed by a little as nearly one. A world's low points (the fewest agents it
had in 5% of its iterations) are kept beside its averages: a world that kept
dipping towards extinction would be less healthy, but no world that lived to
its end came below a few hundred agents, so a line for it would order nothing
yet. A world that died has no health: the dead come last, the longest-lived
first.

Beside the author's lines, the proposal (PROPOSED) adds lines of its own and
drops two, each with its reason, all for the author to accept, move or refuse:
a richest agent holding at most 5% of all tokens (no ruler of tokens, as the hub
line says of links), a line for the author's same-position share of
2026-10-24 (at least a half), and a line for "not too even" drawn by chance
instead of by B2: at least as unequal as tokens dealt out at random at the
world's mean holding, whose Gini is (m - 1) / (2m - 1) for a mean of m tokens an
agent (the most random spread of whole tokens, one at least, is geometric).
The author's other new number, the half-cut exponent, gets no line: it reads a
network's density as much as its dimension (CALIBRATION), so a fixed line would
call a sparse world of three dimensions thin.
The genotype line is dropped (a genotype is renewed by every mutation, so
sharing one says how seldom brains mutate, not how differently agents act,
which the same-position share measures), and so is B2's Gini line. The author's lines
alone give the order `author`.

Every number is an average over iterations 100 to the end of a world's run, as
the judge reads it (research/screen/measures.py); the two halves of a run are
also scored apart, to say whether a world grew healthier or sicker.

    python3 research/screen/health.py E41 E42 E40 [--read 3]
        prints the order and writes research/screen/health.json; with --read, first makes every reading a
        world still lacks, on that many processes
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

import criteria as C  # noqa: E402
import judge as J  # noqa: E402
import measures as M  # noqa: E402

#: Where each line's quantity counts from: the value of a world with none of what the line asks for.
FLOOR = {"hub": 0.0, "links_min": 2.0, "links_max": 2.0, "core": 0.0, "agent_cut": 0.0, "link_cut": 0.0,
         "width": 1.0, "dimension_min": 1.0, "dimension_max": 1.0, "tokens_alike": 1.0, "max_tokens": 1.0,
         "core_links_alike": 1.0, "ages_alike": 1.0, "genotypes_alike": 1.0, "children_alike": 1.0,
         "gini": 0.0, "richest": 0.0, "apart": 0.0, "gini_random": 0.0}

#: The lines this proposal adds to the author's, each with its reason.
PROPOSED: List[Dict[str, Any]] = [
    {"key": "richest", "name": "tokens of the richest agent, as a share of all tokens", "criterion": "decentralised",
     "measure": "richest", "passes": "at most", "value": 0.05,
     "why": "no ruler: the hub line asks that no agent hold more than 5% of the links, this asks the same of "
            "tokens; a world unequal everywhere is healthy, one agent holding the world's fortune is not"},
    {"key": "apart", "name": "same-position share: of two agents in the same situation, how often they act "
                             "measurably differently", "criterion": "diverse", "measure": "apart",
     "passes": "at least", "value": 0.5,
     "why": "the author's own number for very different behaviours surviving (2026-10-24), with its line at "
            "half: two agents more often act differently than alike"},
    {"key": "gini_random", "name": "Gini of the tokens, over the Gini of tokens dealt out at random at the same "
                                   "mean holding", "criterion": "uneven", "measure": "gini_random",
     "passes": "at least", "value": 1.0,
     "why": "with no baseline, chance draws the line: a world more even than a random deal has something "
            "flattening it; tokens dealt out at random, one at least, at a mean of m an agent have a Gini of "
            "(m - 1) / (2m - 1), so the line moves with the world's own riches (0.40 at 3 tokens an agent, "
            "0.46 at 7, 0.47 at 10)"},
]
#: The author's lines this proposal leaves out, with the reason.
DROPPED = {"genotypes_alike": "a genotype is renewed by every mutation, so how many agents share one says how "
                              "seldom brains mutate, not how differently agents act; the same-position share "
                              "measures that (in round 3 no world came near the line: at most 26% of a world's "
                              "agents shared a genotype)",
           "gini": "drawn at B2's most even world of Experiment 16, a baseline the screen no longer compares with; "
                   "the proposal's line for not too even is drawn by chance instead"}
#: The criteria in the proposal's order: the author's after "alive", which every world in the order meets.
CRITERIA = ["decentralised", "resilient", "space", "diverse", "uneven"]


def lines(proposed: bool = True) -> List[Dict[str, Any]]:
    """The lines a world is scored on: the author's, or the proposal's."""
    if not proposed:
        return list(C.LINES)
    return [line for line in C.LINES if line["key"] not in DROPPED] + PROPOSED


def progress(line: Dict[str, Any], value: Optional[float]) -> float:
    """How much of what a line asks a world has: 0 none, 1 all of it (meeting the line)."""
    if value is None or not math.isfinite(value):
        return 0.0
    floor, need = FLOOR[line["key"]], line["value"]
    if C.passes(line, value):
        return 1.0
    if floor == need:
        return 0.0
    if line["passes"] in ("at least", "above"):
        got = (value - floor) / (need - floor)
    elif floor > need:              # "below" a share of agents alike: the share unlike, against the one asked
        got = (floor - value) / (floor - need)
    else:                           # "at most" an amount: the amount allowed over the world's
        got = (need - floor) / (value - floor)
    return float(min(1.0, max(0.0, got)))


def score(averages: Dict[str, Optional[float]], proposed: bool = True) -> Dict[str, Any]:
    """Every line's progress, every criterion's score (its weakest line's), and the health, their sum."""
    each = lines(proposed)
    got = {line["key"]: progress(line, averages.get(line["measure"])) for line in each}
    criteria = {}
    for c in (CRITERIA if proposed else C.KEYS[1:]):
        mine = [got[line["key"]] for line in each if line["criterion"] == c]
        if mine:
            criteria[c] = min(mine)
    return {"lines": got, "criteria": criteria, "health": float(sum(criteria.values())),
            "of": len(criteria), "met": sum(v >= 1.0 for v in criteria.values())}


# ---------------------------------------------------------------------------
# What the half-cut exponent reads for networks of known shape
# ---------------------------------------------------------------------------

#: Random networks in a plane and in a cube, at the densities of the screen's worlds and above: (dimensions,
#: links an agent asked for). The half-cut exponent of a lattice is (d - 1) / d, but a sparse network halves
#: more cheaply: at five links an agent a plane reads about 0.38 and a cube 0.56, and only at eight do they
#: reach a lattice's 1/2 and 2/3. That is why the exponent gets no fixed line.
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


def held(run_id: str) -> Dict[int, float]:
    """A world's mean holding after every game, from its statistics (tokens a link locks are held by none)."""
    return {int(r["iteration"]): float(r["meanTokens"]) for r in gol_record.read_stats(run_id)
            if r["phase"] == 2 and r.get("meanTokens") is not None}


def averages(rows: List[Dict[str, Any]], tokens: int, start: int, stop: int,
             mean_held: Optional[Dict[int, float]] = None) -> Dict[str, Optional[float]]:
    """The judge's averages over iterations start to stop, with the proposal's numbers beside them."""
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
        m = (mean_held or {}).get(row["t"])
        if row.get("gini") is not None and m is not None and random_gini(m) > 0:
            values.setdefault("gini_random", []).append(row["gini"] / random_gini(m))
    out: Dict[str, Optional[float]] = {key: float(np.mean(v)) for key, v in sorted(values.items())}
    if agents:
        out["low"] = float(np.quantile(agents, 0.05))
        out["median_agents"] = float(np.median(agents))
    return out


def world(w: Dict[str, Any], have: Dict[M.Item, Dict[str, Any]]) -> Dict[str, Any]:
    """One world scored: over its whole run, by the proposal and by the author's lines, and in two halves."""
    wanted = M.items(w["id"], w["ended_at"], C.SETTLE)
    missing = [i for i in wanted if i not in have]
    out = {k: w[k] for k in ("round", "id", "seed", "fate", "ended_at", "until", "set")}
    out["alive"] = w["fate"] == "lived"
    out["stopped"] = w["fate"] == "too slow"         # a round may stop its slowest worlds: neither lived nor died
    if missing or not out["alive"]:
        out["read"] = not missing
        return out
    rows = rows_of(have, wanted)
    tokens = int(w["tokens"])
    end = w["ended_at"] + 1
    half = (C.SETTLE + end) // 2            # the two halves of a run, scored apart: 100 to 1,550 and on, in round 3
    mean_held = held(w["id"])
    whole = averages(rows, tokens, C.SETTLE, end, mean_held)
    out.update(read=True, averages=whole, proposal=score(whole), author=score(whole, proposed=False),
               halves=[score(averages(rows, tokens, C.SETTLE, half, mean_held))["health"],
                       score(averages(rows, tokens, half, end, mean_held))["health"]])
    return out


def order(worlds: List[Dict[str, Any]], key: str = "proposal") -> List[Dict[str, Any]]:
    """
    Most healthy first: the living by health, ties by their weakest criterion; then any living world not
    yet read; then the worlds stopped as too slow, which neither lived nor died; then the dead,
    longest-lived first.
    """
    living = [w for w in worlds if w["alive"] and w.get(key)]
    unread = [w for w in worlds if w["alive"] and not w.get(key)]
    stopped = [w for w in worlds if w.get("stopped")]
    dead = [w for w in worlds if not w["alive"] and not w.get("stopped")]
    living.sort(key=lambda w: (-w[key]["health"], -min(w[key]["criteria"].values()), w["id"]))
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
    ranked = order(scored)
    for i, w in enumerate(ranked, 1):
        w["rank"] = i
    author = order(scored, "author")
    for i, w in enumerate(author, 1):
        w["rank_author"] = i
    # The other way to combine the criteria, for the author to compare: a world as healthy as its weakest.
    weakest = sorted((w for w in scored if w["alive"] and w.get("proposal")),
                     key=lambda w: (-min(w["proposal"]["criteria"].values()), -w["proposal"]["health"], w["id"]))
    for i, w in enumerate(weakest, 1):
        w["rank_weakest"] = i
    out = {"rounds": a.rounds, "criteria": CRITERIA, "lines": lines(), "author_lines": C.LINES,
           "floor": FLOOR, "proposed": PROPOSED, "dropped": DROPPED, "calibration": calibration(),
           "worlds": ranked}
    with open(os.path.join(HERE, "health.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    unread = [w["id"] for w in ranked if not w.get("read")]
    print(f"{len(ranked)} worlds; {len(unread)} not yet read" + (f" ({', '.join(unread[:5])} ...)" if unread else ""))
    print("rank  world   health  (author)  " + "  ".join(f"{c[:6]:>6s}" for c in CRITERIA) + "   halves")
    for w in ranked:
        if not w["alive"]:
            how = "stopped as too slow" if w["stopped"] else "died"
            print(f"{w['rank']:4d}  s{w['seed']}  {how} at {w['ended_at']}")
            continue
        if not w.get("proposal"):
            print(f"{w['rank']:4d}  s{w['seed']}  not read")
            continue
        p = w["proposal"]
        print(f"{w['rank']:4d}  s{w['seed']}  {p['health']:5.2f}  "
              f"({w['author']['health']:4.2f} #{w['rank_author']:<3d})  "
              + "  ".join(f"{p['criteria'][c]:6.2f}" for c in CRITERIA)
              + f"   {w['halves'][0]:.2f} → {w['halves'][1]:.2f}")


if __name__ == "__main__":
    main()
