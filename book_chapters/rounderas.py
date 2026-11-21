"""
Experiment 45: round 3's eras against each other (Did the brains get better?).

Everything is read from the contests' own records (`GraphOfLifeRuns/<trial>/result.json` and `meta.json`) and,
to set the worlds side by side, from the two files the thesis names: the evolution rank
(`research/screen/evolution.json`) and health (`research/screen/health.json`).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Sequence

import numpy as np

import book_figures as F
import gol_plan
import gol_store
from book_chapters.common import BLUE, GREY, ORANGE, VIOLET
from book_chapters.dynamics import spearman
from book_figures import chapter, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = "E45"
#: The two fixed eras, every era that plays them, and the eras after the middle.
FIRST, MIDDLE = 250, 1500
ERAS = (500, 1000, 1500, 2000, 2500, 3000)
LATE = (2000, 2500, 3000)
#: The three worlds whose contests were played, and two of them seen, before the second reading was written
#: (15 November 2026): that reading is judged on the other worlds, and these are reported beside them.
SEEN = ("B1-30000-192733-s4002", "B1-30000-9efb1d-s4003", "B1-30000-5379e2-s4002")
#: The rank test's permutations, as the thesis sets them, and the seed of their draws.
PERMUTATIONS, SEED = 10000, 45
#: A contest that ends with one side holding at least this share of the agents ended with that era alone.
ALONE = 0.99
#: The written band is twice a contest's noise; the second reading's twice the noise of a mean of six.
READINGS = {"as written": 1.0, "second reading": 1 / np.sqrt(6)}

SETS = (("against 250", "against the brains of 250", BLUE),
        ("against 1,500", "against the brains of 1,500", ORANGE),
        ("each against itself", "an era against itself", GREY))
CONTESTS = ("The 1,326 contests of Experiment 45 (`python3 gol_lab.py run E45`), 78 in each of 17 worlds of round 3: "
            "the eras 500 to 3,000 against 250, the eras 500 to 3,000 against 1,500, each pair three times (coins "
            "17 to 19) on each era's world, and the eras 250, 1,500 and 3,000 each against itself four times (coins "
            "17 to 20). Each contest plays 100 iterations, every mutation off.")
DATA = "each trial's `GraphOfLifeRuns/<trial>/result.json`: `share`, the later era's share of the agents at the end"


def contests() -> List[Dict[str, Any]]:
    """Every contest of the experiment that has a result: its settings, its set, its result, its cost and start."""
    out = []
    for found in gol_plan.trials_of(EXPERIMENT):
        trial = gol_plan.trial_id_for(found["trial"])
        path = os.path.join(gol_store.run_dir(trial), "result.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            made = json.load(f)
        out.append({**found["trial"], **made["result"], "set": found["set"], "seconds": made.get("seconds") or 0.0,
                    "started": gol_store.load_meta(trial).get("created_at")})
    return out


def score(pairs: Dict[tuple, float], era: int, fixed: int) -> float:
    """An era's score against a fixed era: the share of the agents its brains held, the mean of the pair's contests."""
    return pairs[(fixed, era)] if era > fixed else 1.0 - pairs[(era, fixed)]


def won(contest: Dict[str, Any], era: int) -> bool:
    """Did this era's brains end the contest holding more than half of the agents?"""
    return contest["share"] > 0.5 if era == contest["b"] else contest["share"] < 0.5


def reading(run: str, mine: List[Dict[str, Any]], ranked: Dict[str, Any], health: Dict[str, Any]) -> Dict[str, Any]:
    """One world: its noise, every era's score against the two fixed eras, its gain and its late score."""
    itself = [c for c in mine if c["a"] == c["b"]]
    by_pair: Dict[tuple, List[float]] = defaultdict(list)
    for c in mine:
        if c["a"] != c["b"]:
            by_pair[(c["a"], c["b"])].append(c["share"])
    pairs = {k: float(np.mean(v)) for k, v in by_pair.items()}
    first = {e: score(pairs, e, FIRST) for e in ERAS}
    middle = {e: score(pairs, e, MIDDLE) for e in ERAS if e != MIDDLE}
    w = ranked[run]
    return {"run": run, "label": f"s{w['seed']}" + {"E41": " w", "E42": " s"}.get(w["round"], ""),
            "rank": int(w["rank"]), "score": float(w["score"]), "health": float(health[run]["health"]["product"]),
            "seen": run in SEEN, "sd": float(np.std([c["share"] for c in itself], ddof=1)),
            "first": first, "middle": middle,
            "gain": float(np.median(list(first.values()))), "late": float(np.median([middle[e] for e in LATE])),
            "contests": len(mine), "per_pair": sorted({len(v) for v in by_pair.values()}),
            # How often the later era won: against 250, 1,500 against the eras before it, the eras after 1,500
            # against it, and an era against itself (there the side in the later era's place).
            "won_first": float(np.mean([won(c, c["b"]) for c in mine if c["set"] == "against 250"])),
            # The same on the world of 250, and on the later era's own: two common gardens.
            "won_first_there": float(np.mean([won(c, c["b"]) for c in mine
                                              if c["set"] == "against 250" and c["home"] == FIRST])),
            "won_first_home": float(np.mean([won(c, c["b"]) for c in mine
                                             if c["set"] == "against 250" and c["home"] == c["b"]])),
            "won_middle": float(np.mean([won(c, c["b"]) for c in mine
                                         if c["set"] == "against 1,500" and c["b"] == MIDDLE])),
            "won_late": float(np.mean([won(c, c["b"]) for c in mine if c["a"] == MIDDLE and c["b"] in LATE])),
            "won_itself": float(np.mean([won(c, c["b"]) for c in itself])),
            "itself": float(np.mean([c["share"] for c in itself])),
            "home": float(np.mean([c["share"] for c in mine if c["a"] != c["b"] and c["home"] == c["b"]])),
            "away": float(np.mean([c["share"] for c in mine if c["a"] != c["b"] and c["home"] == c["a"]]))}


def judge(worlds: List[Dict[str, Any]], shrink: float) -> Dict[str, Any]:
    """The thesis's three parts on these worlds, with the band 0.5 ± 2 · shrink · the world's noise."""
    n = len(worlds)
    half = [2 * shrink * w["sd"] for w in worlds]
    above = sum(w["gain"] > 0.5 + h for w, h in zip(worlds, half))
    within = sum(abs(w["late"] - 0.5) <= h for w, h in zip(worlds, half))
    late_above = sum(w["late"] > 0.5 + h for w, h in zip(worlds, half))
    gain = np.array([w["gain"] for w in worlds])
    ranked = np.array([w["score"] for w in worlds])
    rho = spearman(gain, ranked)
    rho_health = spearman(gain, np.array([w["health"] for w in worlds]))
    rng = np.random.default_rng(SEED)
    # The notebook's permutation p: the real order counts as one of the orders tried (Permutation tests).
    p = (1 + sum(spearman(rng.permutation(gain), ranked) >= rho for _ in range(PERMUTATIONS))) / (1 + PERMUTATIONS)
    parts = ["holds" if above >= 2 * n / 3 else "fails" if above < n / 2 else "undecided",
             "holds" if within >= 2 * n / 3 else "fails" if late_above >= n / 2 else "undecided",
             "holds" if rho > 0 and p < 0.05 and rho_health < rho
             else "fails" if rho <= 0 or rho_health > rho else "undecided"]
    return {"worlds": n, "gain_above": above, "late_within": within, "late_above": late_above,
            "rho": rho, "p": p, "rho_health": rho_health, "parts": parts,
            "verdict": "fails" if "fails" in parts else "holds" if parts == ["holds"] * 3 else "undecided",
            "band_half": {"min": min(half), "max": max(half)}}


def sign_flip(d: Sequence[float]) -> float:
    """
    Two-sided p of a mean difference this far from 0, if each world's difference were as likely to have the
    other sign: exact, over every pattern of signs.
    """
    d = np.asarray(d, float)
    signs = 1 - 2 * ((np.arange(2 ** d.size)[:, None] >> np.arange(d.size)) & 1)
    return float(np.mean(np.abs(signs @ d) >= abs(d.sum()) - 1e-12))


def rate(worlds: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    How often the later era won, world by world (every world plays as many contests), against a coin and against
    the rate at which the side in the later era's place won when an era played itself.
    """
    won_ = np.array([w[key] for w in worlds])
    beyond = won_ - [w["won_itself"] for w in worlds]
    return {"contests": float(np.mean(won_)), "median_world": float(np.median(won_)),
            "worlds_above_half": int(np.sum(won_ > 0.5)), "p_coin": sign_flip(won_ - 0.5),
            "beyond_itself": float(np.mean(beyond)), "p_itself": sign_flip(beyond)}


def coin(k: int, n: int) -> float:
    """Two-sided p of k or more heads, or as few, in n tosses of a fair coin."""
    tail = sum(comb(n, j) for j in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@chapter
def rounderas(ch: F.Chapter) -> None:
    found = contests()
    plan = gol_plan.read_plan(EXPERIMENT)
    runs = plan["trials"]["sets"][0]["grid"]["run"]
    if len(found) < len(gol_plan.trials_of(EXPERIMENT)):
        return
    with open(os.path.join(ROOT, "research", "screen", "evolution.json"), encoding="utf-8") as f:
        ranked = {w["id"]: w for w in json.load(f)["worlds"]}
    with open(os.path.join(ROOT, "research", "screen", "health.json"), encoding="utf-8") as f:
        health = {w["id"]: w for w in json.load(f)["worlds"]}
    worlds = sorted((reading(r, [c for c in found if c["run"] == r], ranked, health) for r in runs),
                    key=lambda w: w["rank"])
    later = [w for w in worlds if not w["seen"]]
    chosen = plan["chosen"]

    # ---- The thesis, in both readings
    verdict = {"as written": judge(worlds, READINGS["as written"]),
               "second reading": judge(later, READINGS["second reading"]),
               "second reading, all 17 beside": judge(worlds, READINGS["second reading"])}
    ch.number("verdict", verdict)
    ch.number("worlds", {w["label"]: {k: v for k, v in w.items() if k != "label"} for w in worlds})
    # Where twice the noise is more than 0.5, the band as written reaches past 0 and 1: no score can leave it.
    covers = ch.number("band_as_written_covers_all", sum(2 * w["sd"] > 0.5 for w in worlds))

    def why(run: str) -> str:
        groups = [name for name in ("evolution", "healthiest", "least healthy") if run in chosen[name]]
        return ", ".join("the rank" if g == "evolution" else g for g in groups)

    def band(w: Dict[str, Any], shrink: float) -> str:
        h = 2 * shrink * w["sd"]
        return f"{0.5 - h:.2f} to {0.5 + h:.2f}".replace("-", "−")

    def marked(value: float, w: Dict[str, Any]) -> str:
        return f"**{value:.2f}**" if value > 0.5 + 2 * READINGS["second reading"] * w["sd"] else f"{value:.2f}"

    ch.number("table:worlds", {
        "head": ["world", "rank", "chosen by", "noise", "band as written", "band, second reading",
                 "gain against 250", "late against 1,500"],
        "rows": [[{"text": w["label"] + (" (played first)" if w["seen"] else ""), "href": f"run:{w['run']}"},
                  str(w["rank"]), why(w["run"]), f"{w['sd']:.2f}", band(w, 1.0),
                  band(w, READINGS["second reading"]), marked(w["gain"], w), marked(w["late"], w)]
                 for w in worlds]})

    # ---- Pooled over the worlds: how often the later era won, and how far that is from a coin
    itself = [c for c in found if c["a"] == c["b"]]
    b_won = sum(won(c, c["b"]) for c in itself)
    rates = {"first": {e: float(np.mean([won(c, e) for c in found if c["a"] == FIRST and c["b"] == e]))
                       for e in ERAS},
             "middle": {e: float(np.mean([won(c, e) for c in found if {c["a"], c["b"]} == {e, MIDDLE}]))
                        for e in ERAS if e != MIDDLE}}
    alone = {name: [sum(max(c["share"], 1 - c["share"]) >= ALONE for c in found if c["set"] == name),
                    sum(1 for c in found if c["set"] == name)] for name, _, _ in SETS}
    ch.number("pooled", {
        "rates": rates,
        "won_first": rate(worlds, "won_first"), "won_first_there": rate(worlds, "won_first_there"),
        "won_first_home": rate(worlds, "won_first_home"), "won_middle": rate(worlds, "won_middle"),
        "won_late": rate(worlds, "won_late"),
        "itself": {"later_label_won": b_won, "contests": len(itself), "p_coin": coin(b_won, len(itself)),
                   "mean_share": float(np.mean([c["share"] for c in itself])),
                   "p_worlds": sign_flip(np.array([w["itself"] for w in worlds]) - 0.5)},
        "home": {"home": float(np.mean([w["home"] for w in worlds])),
                 "away": float(np.mean([w["away"] for w in worlds])),
                 "worlds_more_at_home": sum(w["home"] > w["away"] for w in worlds),
                 "p": sign_flip([w["home"] - w["away"] for w in worlds])},
        "alone": alone,
        "tokens_and_agents": float(np.corrcoef([c["share"] for c in found], [c["tokens"] for c in found])[0, 1])})
    hours = sum(c["seconds"] for c in found) / 3600
    day = lambda t: datetime.fromtimestamp(t).strftime("%-d %B %Y, %H:%M")
    ch.number("cost", {"worker_hours": hours, "contests": len(found),
                       "from": day(min(c["started"] for c in found)),
                       "to": day(max(c["started"] + c["seconds"] for c in found))})

    # ---- Every world's scores, with both bands
    panels = []
    for w in worlds:
        series = []
        for name, shrink, alpha in (("as written", 1.0, 0.10), ("second reading", READINGS["second reading"], 0.22)):
            h = 2 * shrink * w["sd"]
            series.append(F.area([FIRST, 3000], [max(0.0, 0.5 - h)] * 2, [min(1.0, 0.5 + h)] * 2, GREY,
                                 alpha=alpha))
        series.append(F.line(ERAS, [w["first"][e] for e in ERAS], None, BLUE, width=1.4))
        series.append(F.line(ERAS, [w["first"][e] for e in ERAS], None, BLUE, points=True, size=5, alpha=0.9))
        mid = [w["middle"].get(e) for e in ERAS]
        series.append(F.line(ERAS, mid, None, ORANGE, width=1.4))
        series.append(F.line(ERAS, mid, None, ORANGE, points=True, size=5, alpha=0.9))
        panels.append({"title": f"{w['label']} · rank {w['rank']}" + (" · first" if w["seen"] else ""),
                       "x": {"label": "era", "min": FIRST, "max": 3000},
                       "y": {"label": "score", "min": 0, "max": 1},
                       "series": series, "guides": [{"axis": "y", "at": 0.5, "label": ""}],
                       "legend": False, "height": 150})
    ch.grid("scores", panels, columns=3, title="Every world's eras against 250 and against 1,500",
            caption="One panel for each of the 17 worlds, in the order of the evolution rank. Each dot is an era's "
                    "score: the share of the agents its brains held after 100 iterations, the mean of six "
                    "contests, three on each era's world. Blue: against the brains of iteration 250. Orange: "
                    "against the brains of 1,500 (none at 1,500 itself). The dashed line is a tie. The wide, faint "
                    "band is the thesis's band as written, 0.5 plus or minus twice the spread of the world's "
                    f"twelve contests of an era against itself, cut at 0 and 1; in {covers} worlds it covers every "
                    "score there can be. The narrower, brighter band is the second reading's, twice the spread "
                    "divided by √6, the noise of a mean of six. The three worlds marked first were played, and two "
                    "of them seen, before the second reading was written.",
            recipe=recipe(CONTESTS, DATA, [
                "For each world and pair of eras, the mean of `share` over the pair's six contests: the later "
                "era's score; the earlier era's score is 1 less that.",
                "The noise: the standard deviation (n − 1) of `share` over the world's twelve contests of an era "
                "against itself.",
                "The band as written: 0.5 ± twice the noise; the second reading: 0.5 ± twice the noise / √6."]))

    # ---- Pooled: how often each era won
    def bars(rates_of: Dict[int, float], colour: int) -> Dict[str, Any]:
        return F.bars([f"{e:,}" for e in rates_of], list(rates_of.values()), colour, alpha=0.85)

    # Labelled on the right only, where the bars stay below the lines.
    guides = [{"axis": "y", "at": 0.5, "label": "a coin"},
              {"axis": "y", "at": b_won / len(itself), "label": "an era against itself"}]
    unlabelled = [{**g, "label": ""} for g in guides]
    ch.grid("won", [
        {"title": "each era against the brains of 250",
         "x": {"label": "era", "categories": [f"{e:,}" for e in rates["first"]]},
         "y": {"label": "share of its contests won", "min": 0, "max": 1},
         "series": [bars(rates["first"], BLUE)], "guides": unlabelled, "legend": False, "height": 220},
        {"title": "each era against the brains of 1,500",
         "x": {"label": "era", "categories": [f"{e:,}" for e in rates["middle"]]},
         "y": {"label": "share of its contests won", "min": 0, "max": 1},
         "series": [bars(rates["middle"], ORANGE)], "guides": guides, "legend": False, "height": 220}],
        columns=2, title="How often an era won, over all the worlds",
        caption="Each bar is one era: of its 102 contests against a fixed era, six in each of the 17 worlds, the "
                "share it won, ending with more than half of the agents. Left: against the brains of 250. Right: "
                "against the brains of 1,500; there the eras 500 and 1,000 are the earlier side. The dashed lines, "
                f"in both panels, are a coin, 0.5, and the share of the contests of an era against itself won by the side in the "
                f"later era's place: {b_won} of {len(itself)}. Both sides carry the same brains there; if the place "
                "itself helps, a later era no better than the fixed one would win about this often. The eras 500 "
                "and 1,000 are in the earlier era's place against 1,500, so for them that line would lie as far "
                "below a coin.",
        recipe=recipe(CONTESTS, DATA, [
            "A contest is won by the later era if `share` is above 0.5, by the earlier if below.",
            "For each era, the share of its contests with the fixed era that it won, over every world.",
            "The second line: the share of the 204 contests of an era against itself with `share` above 0.5."]))

    # ---- Does the rank tell which worlds gained? Does health?
    def scatter(key: str) -> List[Dict[str, Any]]:
        return [F.line([w[key] for w in group], [w["gain"] for w in group], None, colour, points=True,
                       size=7, alpha=0.85)
                for group, colour in (([w for w in worlds if not w["seen"]], BLUE),
                                      ([w for w in worlds if w["seen"]], VIOLET))]
    every = verdict["as written"]
    ch.grid("rank", [
        {"title": f"the evolution score: rho {every['rho']:.2f}".replace("-", "−"),
         "x": {"label": "evolution score", "min": 0, "max": 0.8},
         "y": {"label": "gain against 250", "min": 0, "max": 1.05},
         "series": scatter("score"), "guides": [{"axis": "y", "at": 0.5, "label": ""}], "legend": False,
         "height": 240},
        {"title": f"health: rho {every['rho_health']:.2f}".replace("-", "−"),
         "x": {"label": "health (the product)", "min": 0, "max": 0.7},
         "y": {"label": "gain against 250", "min": 0, "max": 1.05},
         "series": scatter("health"), "guides": [{"axis": "y", "at": 0.5, "label": ""}], "legend": False,
         "height": 240}],
        columns=2, title="Gain against the evolution rank, and against health",
        caption="Each dot is one of the 17 worlds: its gain, the median score of its eras 500 to 3,000 against the "
                "brains of 250, against its evolution score (left) and its health (right). Rho is Spearman's rank "
                "correlation over all 17. Violet: the three worlds played first; blue: the 14 the second reading judges.",
        recipe=recipe(CONTESTS, DATA + "; the evolution score `score` of `research/screen/evolution.json` and the "
                      "product of health of `research/screen/health.json`", [
                          "Each era's score against 250 as in the first figure; a world's gain is the median of "
                          "its six eras' scores.",
                          "Spearman's rho of the gain with each score, ties given the mean of their ranks."]))

    # ---- How the contests ended
    edges = np.linspace(0, 1, 21)
    below = np.zeros(len(edges) - 1)
    stack = []
    for name, label, colour in SETS:
        counts, _ = np.histogram([c["share"] for c in found if c["set"] == name], bins=edges)
        stack.append({"label": label, "kind": "bars", "x0": edges[:-1].tolist(), "x1": edges[1:].tolist(),
                      "y": (below + counts).tolist(), "y0": below.tolist(), "colour": colour, "alpha": 0.85})
        below = below + counts
    ch.figure("ends", title="How the contests ended",
              x={"label": "the later era's share of the agents at the end", "min": 0, "max": 1},
              y={"label": "contests", "min": 0}, series=stack, height=240,
              caption=f"All {len(found)} contests, by the share of the agents the later era's brains held after "
                      "100 iterations, in steps of 0.05, stacked by the fixed era they played. When an era plays "
                      "itself, both sides carry the same brains, and the share is that of the side in the later "
                      "era's place. "
                      f"{sum(n for n, _ in alone.values())} contests ended with one side holding at least 99% "
                      "of the agents.",
              recipe=recipe(CONTESTS, DATA, ["Count the contests of each set by `share`, in twenty steps of "
                                             "0.05 from 0 to 1."]))
