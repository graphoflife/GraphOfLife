"""
Local rules only, How far does a difference travel? and A source and a sink:
three experiments planned on 9 November 2026, none of them played. This module
makes what the three plans are built on.

- What round 3 of the screen of random worlds (Experiment 40) already says
  about two local rules it drew: mending a death, and estates that flow to the
  nearest living. Experiments 41 and 42 play eight of the same draws twice, so
  their settings are not independent draws and are left out of the
  comparisons, as research/screen/effects.py leaves them out.
- The candidate worlds the plans would be played on, with what each world's
  round-3 run cost.
- Which of the 24 shortlisted worlds would keep a light cone once the three
  rules that reach across a world are made local, and why the others would not.
- The size of each world a source and a sink would be tried on, and the flow
  that would feed it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np

import book_figures as F
import gol_store as store
from book_chapters.common import BLUE, GREEN, ORANGE, RED, VIOLET, YELLOW
from book_figures import chapter, dots, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")
HEALTH = os.path.join(SCREEN, "health.json")
EFFECTS = os.path.join(SCREEN, "effects.json")
CANDIDATES = os.path.join(SCREEN, "candidates.json")

#: Round 3 is read from iteration 100 on, as its health is; sizes and holdings from 500, its settled life.
FROM, SETTLED = 100, 500
#: The worlds Local rules only would be played on: the five the plan names first, and the one that culls least.
WORLDS = (3042, 3018, 3080, 3057, 3026, 3004)
#: The three of them How far does a difference travel? and A source and a sink would be played on.
FEW = (3026, 3004, 3080)
#: Round 3's share-out rules, in the order of the figure, with the page's words and colours.
RULES = [("uniform", "evenly over all", BLUE), ("poorest", "to the poorest", GREEN),
         ("newborns", "to the newborns", YELLOW), ("parents", "to the parents", ORANGE),
         ("children", "to the dead's children", VIOLET), ("local", "to the nearest living", RED)]
RULE_WORDS = {key: words for key, words, _ in RULES}
#: Shuffles per permutation test, and their seed (as research/screen/effects.py).
SHUFFLES, SEED = 20000, 11

ROUND_RUNS = ("the worlds of round 3 of the screen of random worlds that lived to the end of their 3,000 iterations "
              "(Experiment 40: 30,000 tokens, every setting drawn at random)")
ROUND_DATA = ("each world's averages over iterations 100 to the end, `research/screen/health.json` (from the "
              "screen's readings, `research/screen/measures.py`), and each run's own statistics, "
              "`GraphOfLifeRuns/<run>/stats.jsonl` (`nodes_before`, `starved`, `orphaned`)")


# ---------------------------------------------------------------------------
# Reading the worlds
# ---------------------------------------------------------------------------

def _stats(run_id: str) -> Dict[str, Any]:
    """
    A world's deaths and size from its own statistics. Deaths: the starved and
    the culled of each phase over the agents that faced it, averaged over the
    phases from FROM on and added over an iteration's two phases. The worst
    game: the largest share of a world's agents one game's cull removed. Size
    and holdings: medians over the games from SETTLED on. Hours: what the whole
    run took, from each row's `_seconds`.
    """
    sums = {1: [0, 0.0, 0.0, 0.0], 2: [0, 0.0, 0.0, 0.0]}
    worst, seconds, last = 0.0, 0.0, 0
    agents: List[float] = []
    links: List[float] = []
    mean_tokens: List[float] = []
    median_tokens: List[float] = []
    shared: List[float] = []
    for line in open(os.path.join(store.BASE_DIR, run_id, "stats.jsonl")):
        r = json.loads(line)
        seconds += r.get("_seconds") or 0.0
        last = max(last, r["iteration"])
        before = r.get("nodes_before") or 0
        if r["iteration"] < FROM or before <= 0:
            continue
        s = sums[r["phase"]]
        s[0] += 1
        s[1] += (r.get("starved") or 0) / before
        s[2] += (r.get("orphaned") or 0) / before
        s[3] += (r.get("births") or 0) / before
        if r["phase"] == 2:
            worst = max(worst, (r.get("orphaned") or 0) / before)
            if r["iteration"] >= SETTLED and r.get("nodes"):
                agents.append(r["nodes"])
                links.append(2 * r["edges"] / r["nodes"])
                mean_tokens.append(r["tokens"] / r["nodes"])
                median_tokens.append(r["medianTokens"])
                shared.append((r.get("redistributed") or 0) / r["tokens"])
    starved = sum(s[1] / max(1, s[0]) for s in sums.values())
    culled = sum(s[2] / max(1, s[0]) for s in sums.values())
    return {"starved": starved, "culled": culled, "deaths": starved + culled,
            "cull_share": culled / (starved + culled) if starved + culled else 0.0,
            "births": sums[1][3] / max(1, sums[1][0]), "worst": worst, "hours": seconds / 3600,
            "agents": float(np.median(agents)), "links": float(np.median(links)),
            "tokens_each": float(np.median(mean_tokens)), "median_tokens": float(np.median(median_tokens)),
            "shared": float(np.median(shared)), "seconds_per_iteration": seconds / (last + 1)}


def _rule(w: Dict[str, Any]) -> str:
    return w["set"].get("redistribution", "uniform")


def _mends(w: Dict[str, Any]) -> bool:
    return bool(w["set"].get("mend_on_death", False))


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def kruskal_chance(values: Sequence[float], groups: Sequence[Any]) -> float:
    """
    The chance, were the values shuffled at random among the worlds, of
    differences among the groups at least as large as those seen: the
    Kruskal-Wallis statistic by permutation, as research/screen/effects.py
    tests a setting. One test on its own: no false discovery rate.
    """
    names = sorted({json.dumps(g) for g in groups})
    onehot = np.array([[json.dumps(g) == n for n in names] for g in groups], dtype=float)
    ranks = _ranks(np.asarray(values, dtype=float))
    n, N = onehot.sum(axis=0), len(ranks)

    def stat(r: np.ndarray) -> np.ndarray:
        return 12.0 / (N * (N + 1)) * ((r @ onehot) ** 2 / n).sum(axis=-1) - 3 * (N + 1)

    rng = np.random.RandomState(SEED)
    shuffled = ranks[np.argsort(rng.rand(SHUFFLES, N), axis=1)]
    observed = stat(ranks[None, :])[0]
    return float((1 + (stat(shuffled) >= observed - 1e-12).sum()) / (1 + SHUFFLES))


def _effect(effects: Dict[str, Any], outcome: str, setting: str) -> Dict[str, Any]:
    """research/screen/effects.py's permutation test of one setting on one outcome."""
    test = next(t for t in effects["tests"] if t["outcome"] == outcome)
    row = next(r for r in test["rows"] if r["setting"] == setting)
    return {"worlds": row["worlds"], "chance": row["chance"], "fdr": row["fdr"],
            "by": {json.loads(k) if k.startswith('"') else k: v for k, v in row["by"].items()}}


# ---------------------------------------------------------------------------
# The light cone of a world's own rules, once the three global rules are local
# ---------------------------------------------------------------------------

#: Why a world's own rules would carry a difference further than its cone, in the table's words.
BREAKS = {"rounds": "stakes placed in rounds", "jump": "a search that jumps", "reach": "a search that reaches",
          "walk": "a scout that walks", "mates": "a mate's child, wherever it is",
          "estate": "upkeep paid into the estate"}


def cone(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    How far a difference can travel in a phase under a world's drawn settings
    once the cull, the share-out and the one random stream are made local
    (How far does a difference travel?): four steps with a look that only
    talks before the look that acts, three without. No bound at all where a
    staker sees the board again before each token; where a search links an
    agent to one as many steps away as its tokens pay for, or a scout reports
    from wherever it has walked to; where a mate must have a living child,
    wherever that child now is; or where upkeep is paid into the estate, which
    no local rule shares out.
    """
    breaks = []
    if s.get("allocation", "full") != "full":
        breaks.append(BREAKS["rounds"])
    if s.get("link_search", "off") != "off":
        breaks.append(BREAKS[s["link_search"]])
    if s.get("mix_at_birth") and s.get("mix_mates", "any") == "parents":
        breaks.append(BREAKS["mates"])
    if s.get("link_upkeep", "off") != "off" and s.get("upkeep_to", "end") == "estate":
        breaks.append(BREAKS["estate"])
    talks = bool(s.get("message_prepass", True))
    return {"talks": talks, "breaks": breaks, "steps": None if breaks else (4 if talks else 3)}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@chapter
def localrules(ch: F.Chapter) -> None:
    if not (os.path.exists(HEALTH) and os.path.exists(EFFECTS) and os.path.exists(CANDIDATES)):
        return
    health = json.load(open(HEALTH))
    effects = json.load(open(EFFECTS))
    shortlist = json.load(open(CANDIDATES))["shortlist"]

    drawn = [w for w in health["worlds"] if w["round"] == "E40"]
    living = [w for w in drawn if w["fate"] == "lived" and w.get("read") and "health" in w]
    lives = {w["id"]: _stats(w["id"]) for w in living}

    def avg(w: Dict[str, Any], key: str) -> float:
        return lives[w["id"]][key] if key in lives[w["id"]] else w["averages"][key]

    mend = [w for w in living if _mends(w)]
    plain = [w for w in living if not _mends(w)]
    local = [w for w in living if _rule(w) == "local"]
    other = [w for w in living if _rule(w) != "local"]

    def med(group: List[Dict[str, Any]], key: str) -> float:
        return float(np.median([avg(w, key) for w in group]))

    keys = ("culled_tokens", "culled_agents", "worst", "deaths", "culled", "starved", "cull_share", "births",
            "width", "dimension", "agent_cut", "link_cut", "core", "richest", "hub", "gini", "agents")

    def compare(a: List[Dict[str, Any]], b: List[Dict[str, Any]], labels: Sequence[Any]) -> Dict[str, Any]:
        out = {"worlds": [len(a), len(b)]}
        for key in keys:
            values = [avg(w, key) for w in a + b]
            out[key] = {"medians": [med(a, key), med(b, key)],
                        "chance": kruskal_chance(values, [labels[0]] * len(a) + [labels[1]] * len(b))}
        return out

    by_rule = {key: {"worlds": sum(_rule(w) == key for w in living),
                     "drawn": sum(_rule(w) == key for w in drawn),
                     "died": sum(_rule(w) == key and w["fate"] != "lived" for w in drawn),
                     **{k: float(np.median([avg(w, k) for w in living if _rule(w) == key]))
                        for k in ("richest", "gini", "dimension", "width", "agent_cut", "culled_tokens",
                                  "deaths", "births")}}
               for key, _, _ in RULES}
    rules = [_rule(w) for w in living]
    rule_tests = {k: kruskal_chance([avg(w, k) for w in living], rules)
                  for k in ("richest", "dimension", "width", "agent_cut", "deaths", "births")}
    # Five of the nine living worlds with estates to the nearest living also reach for strangers, which folds a
    # world: so the fold is read again within the worlds that search the same way.
    searching = {}
    for how in ("jump", "reach"):
        group = [w for w in living if w["set"].get("link_search", "off") == how]
        searching[how] = {side: {"worlds": len(g), "dimension": med(g, "dimension") if g else None,
                                 "width": med(g, "width") if g else None, "gini": med(g, "gini") if g else None,
                                 "richest": med(g, "richest") if g else None}
                          for side, g in (("local", [w for w in group if _rule(w) == "local"]),
                                          ("other", [w for w in group if _rule(w) != "local"]))}
    outcomes = ("culls", "health", "c_space", "c_diverse", "gini", "agents", "apart", "population", "novelty",
                "fertility", "died")
    ch.number("round", {
        "drawn": len(drawn), "living": len(living),
        "mend": {"drawn": [sum(_mends(w) for w in drawn), sum(not _mends(w) for w in drawn)],
                 "died": [sum(_mends(w) and w["fate"] != "lived" for w in drawn),
                          sum(not _mends(w) and w["fate"] != "lived" for w in drawn)],
                 **compare(mend, plain, (True, False)),
                 "effects": {o: _effect(effects, o, "mend_on_death") for o in outcomes}},
        "local": {"drawn": sum(_rule(w) == "local" for w in drawn),
                  "died": sum(_rule(w) == "local" and w["fate"] != "lived" for w in drawn),
                  **compare(local, other, ("local", "other")),
                  "searching": searching,
                  "effects": {o: _effect(effects, o, "redistribution") for o in outcomes}},
        "by_rule": by_rule, "rule_tests": rule_tests,
        "died_all": sum(w["fate"] != "lived" for w in drawn),
    })

    # What mending and a local share-out did, world by world.
    pair = ["mends its dead", "does not"]
    two = [GREEN, ORANGE]
    rule_names = ["evenly", "poorest", "newborns", "parents", "children", "nearest"]
    charts = [
        {"title": "tokens culled in a game", "x": {"label": "", "categories": pair},
         "y": {"label": "share of all tokens", "log": True, "min": 0.0001, "max": 0.5},
         "series": dots(pair, [[avg(w, "culled_tokens") for w in g] for g in (mend, plain)], two),
         "legend": False, "height": 190},
        {"title": "the worst game: agents culled at once", "x": {"label": "", "categories": pair},
         "y": {"label": "share of the agents", "min": 0, "max": 1},
         "series": dots(pair, [[avg(w, "worst") for w in g] for g in (mend, plain)], two),
         "legend": False, "height": 190},
        {"title": "how wide the world is", "x": {"label": "", "categories": pair},
         "y": {"label": "typical distance over the twin's", "min": 0.8, "max": 5},
         "series": dots(pair, [[avg(w, "width") for w in g] for g in (mend, plain)], two),
         "guides": [{"axis": "y", "at": 1, "label": "as near as a random network"}],
         "legend": False, "height": 190},
        {"title": "the richest agent, by who inherits", "x": {"label": "", "categories": rule_names},
         "y": {"label": "share of all tokens", "log": True, "min": 0.0005, "max": 0.3},
         "series": dots(rule_names, [[avg(w, "richest") for w in living if _rule(w) == key] for key, _, _ in RULES],
                        [c for _, _, c in RULES]),
         "legend": False, "height": 190},
    ]
    ch.grid("roundthree", charts, columns=2,
            title="What round 3 says about a local repair and a local share-out",
            caption=f"Each dot is one of the {len(living)} worlds of Experiment 40 that lived to the end of their "
                    "3,000 iterations, read from iteration 100 on; the short bar is the median of a column. "
                    "Top left, on a logarithmic axis: the share of all the world's tokens a game culls, the "
                    "tokens of agents cut off from the largest piece of the network, for the worlds drawn with "
                    "mending (a dying agent's former neighbours joined up again in a ring) and those without. "
                    "Top right: the most of its agents a single game culled. Bottom left: how wide the world "
                    "is, its typical distance over that of a random network with the same links (on the "
                    "dashed line, no wider than one). Bottom right, on a logarithmic axis: what the richest "
                    "agent holds, by where each world's rule sends the tokens of the dead: evenly over all, to "
                    "the poorest, to this iteration's newborns, to their parents, to the dead agent's own "
                    "children, or to the nearest living. Every other setting of each world was drawn at random "
                    "too, so each column mixes many kinds of world.",
            recipe=recipe(ROUND_RUNS, ROUND_DATA, [
                "Read each living world's averages from `health.json`: `culled_tokens` (the share of all tokens "
                "a game's cleanup shares out, every one a culled agent's), `width` and `richest`.",
                "From its `stats.jsonl`, for every game from iteration 100 on, `orphaned` over `nodes_before`: "
                "the share of the agents the game culled. Keep the largest.",
                "Group the worlds by their drawn `mend_on_death`, and by `redistribution`.",
                "One dot per world; the bar is the median."]))

    # The candidate worlds of Local rules only.
    by_seed = {w["seed"]: w for w in living}
    rows = []
    for seed in WORLDS:
        w, life = by_seed[seed], lives[by_seed[seed]["id"]]
        s = w["set"]
        rows.append([{"text": f"s{seed}", "href": f"run:{w['id']}"}, w["ranks"]["product"],
                     RULE_WORDS[_rule(w)], "yes" if _mends(w) else "no", s.get("link_search", "off"),
                     f"{100 * w['averages']['culled_tokens']:.1f}%", f"{100 * life['worst']:.0f}%",
                     f"{life['agents']:,.0f}", f"{life['hours']:.1f}"])
    ch.number("table:candidates", {
        "head": ["world", "health's place (product)", "estates of the dead go", "mends its dead", "search",
                 "tokens culled a game", "worst game culled", "agents", "hours for 3,000 iterations"],
        "rows": rows})
    ch.number("candidates", {f"s{seed}": {"hours": lives[by_seed[seed]["id"]]["hours"],
                                          "agents": lives[by_seed[seed]["id"]]["agents"],
                                          "worst": lives[by_seed[seed]["id"]]["worst"],
                                          "culled_tokens": by_seed[seed]["averages"]["culled_tokens"],
                                          "rank": by_seed[seed]["ranks"]["product"]}
                             for seed in WORLDS})

    # Which shortlisted worlds would keep a light cone.
    every = {w["id"]: w for w in health["worlds"]}
    rows, kept = [], 0
    for run_id in sorted(shortlist, key=lambda r: every[r]["ranks"]["product"]):
        w = every[run_id]
        s = w["set"]
        c = cone(s)
        kept += c["steps"] is not None
        staking = {"full": "all at once", "weakest_first": "weakest first",
                   "strongest_first": "strongest first"}[s.get("allocation", "full")]
        rows.append([{"text": f"s{w['seed']}", "href": f"run:{run_id}"}, "yes" if c["talks"] else "no",
                     s.get("link_search", "off"), staking, RULE_WORDS[_rule(w)],
                     "yes" if s.get("link_upkeep", "off") != "off" and s.get("upkeep_to", "end") == "estate"
                     else "no",
                     f"{c['steps']} steps" if c["steps"] else "none: " + ", ".join(c["breaks"])])
    ch.number("table:cone", {
        "head": ["world", "talks before it acts", "search", "stakes placed", "estates of the dead go",
                 "upkeep into the estate", "how far a difference can go in a phase, once the three rules are local"],
        "rows": rows})
    ch.number("cone", {"shortlist": len(shortlist), "kept": kept,
                       "by_reason": {key: sum(why in cone(every[r]["set"])["breaks"] for r in shortlist)
                                     for key, why in BREAKS.items()}})

    # The worlds a source and a sink would be tried on, and the flow that would feed each.
    rows, flow = [], {}
    for seed in FEW:
        w, life = by_seed[seed], lives[by_seed[seed]["id"]]
        inflow = int(round(life["agents"] / 2))
        flow[f"s{seed}"] = {"agents": life["agents"], "links": life["links"], "tokens_each": life["tokens_each"],
                            "median_tokens": life["median_tokens"], "shared": life["shared"], "inflow": inflow,
                            "hours": life["hours"], "seconds_per_iteration": life["seconds_per_iteration"],
                            "entropy": float(np.log(life["agents"]))}
        share = 100 * life["shared"]
        rows.append([{"text": f"s{seed}", "href": f"run:{w['id']}"}, f"{life['agents']:,.0f}",
                     f"{life['links']:.2f}", f"{life['tokens_each']:.1f}",
                     f"{share:.2f}%" if share < 1 else f"{share:.1f}%", f"{inflow:,}", f"{life['hours']:.1f}"])
    ch.number("table:flow", {
        "head": ["world", "agents", "links per agent", "tokens per agent", "tokens a game's cleanup shares out",
                 "inflow a phase, to feed as many agents", "hours for 3,000 iterations"],
        "rows": rows})
    ch.number("flow", flow)
