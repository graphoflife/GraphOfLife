"""
Round 3's worlds read with the tools of the Wolfram Physics Project: what they rewrite and how locally, their
space through a life, the past cones of their agents, and whether their riches bend their space. Nothing is
read here but research/atlas/wolfram.json, which research/atlas/wolfram.py makes from the worlds' frames and
statistics.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import book_figures as F
from book_chapters.common import BAND_WORDS, BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW
from book_figures import chapter, dots, line, quantile_band, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READINGS = os.path.join(ROOT, "research", "atlas", "wolfram.json")

RUNS = ("the 82 worlds of round 3 of the screen of random worlds that lived to the end of their 3,000 iterations "
        "(Experiments 40, 41 and 42: 30,000 tokens, every setting drawn at random; the worlds of "
        "`research/screen/evolution.json`)")
DATA = ("each world's readings in `research/atlas/wolfram.json`, made by `python3 research/atlas/wolfram.py` "
        "from the world's frames (`GraphOfLifeRuns/<run>/frames/`) and statistics (`stats.jsonl`)")
SEARCHES = [("off", "no search", GREY), ("walk", "a scout walks", GREEN), ("reach", "reaches", ORANGE),
            ("jump", "jumps", RED)]
KINDS = [("news", "news: the rules' bound", BLUE), ("matter", "matter: a step a phase", ORANGE),
         ("flow", "tokens and brains that moved", GREEN)]


def _per(part: Dict[str, Any], key: str) -> float:
    return part[key] / part["agents"] if part.get("agents") else float("nan")


def census(w: Dict[str, Any]) -> Dict[str, float]:
    """A world's rewrites per agent per iteration, and how local they are."""
    rep, game = w["rewrites"]["reproduction"], w["rewrites"]["game"]
    made_old = np.add(rep["made_old"], game["made_old"])
    made_birth = np.add(rep["made_birth"], game["made_birth"])
    both = lambda key: _per(rep, key) + _per(game, key)
    deaths = sum(rep[k] + game[k] for k in ("starved", "culled", "aged"))
    made = made_old.sum() + made_birth.sum() + rep["made_newborns"] + game["made_newborns"]
    far_old = made_old[3:].sum()
    far_birth = made_birth[2:].sum()
    culled = rep["culled"] + game["culled"]
    local = (made - far_old - far_birth) + rep["cut_lapsed"] + game["cut_lapsed"] + (deaths - culled) \
        + rep["born"] + game["born"]
    every = made + rep["cut_lapsed"] + game["cut_lapsed"] + deaths + rep["born"] + game["born"]
    return {
        "births": _per(rep, "born"),
        "deaths": both("starved") + both("culled") + both("aged"),
        "starved": both("starved"), "culled": both("culled"),
        "made": made / rep["agents"] if rep["agents"] else float("nan"),
        "made_old": made_old.sum() / rep["agents"], "made_birth": made_birth.sum() / rep["agents"],
        "lapsed": both("cut_lapsed"),
        "old_local": made_old[2] / made_old.sum() if made_old.sum() else float("nan"),
        "old_three": made_old[3] / made_old.sum() if made_old.sum() else float("nan"),
        "old_far": made_old[4:].sum() / made_old.sum() if made_old.sum() else float("nan"),
        "old_beyond": made_old[3:].sum() / made_old.sum() if made_old.sum() else float("nan"),
        "old_out": made_old[8] / made_old.sum() if made_old.sum() else float("nan"),
        "birth_far": far_birth / made_birth.sum() if made_birth.sum() else float("nan"),
        "cull_share": culled / deaths if deaths else float("nan"),
        "nonlocal": 1 - local / every if every else float("nan"),
        "links_per_agent": rep["links"] / rep["agents"],
        "cut": both("cut_lapsed") + both("cut_dead"),
        # How many iterations a link lasts, and an agent: what the world holds over what it loses an iteration.
        "link_life": (rep["links"] / rep["agents"]) / (both("cut_lapsed") + both("cut_dead")),
        "agent_life": 1 / (both("starved") + both("culled") + both("aged")),
    }


def cone_curves(w: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """A world's past cones: the median agent's share of each earlier world, and the whole world's, at each
    phase back, the median over the moments read."""
    agent = np.median([np.asarray(c["agent"][kind], float) for c in w["cones"]], axis=0)
    whole = np.median([np.asarray(c["whole"][kind], float) for c in w["cones"]], axis=0)
    return {"agent": agent, "whole": whole}


def first(curve: np.ndarray, level: float) -> Optional[int]:
    """The first phase back at which a cone holds `level` of the world, or None within the phases read."""
    hit = np.flatnonzero(curve >= level)
    return int(hit[0]) + 1 if hit.size else None


def crossing(w: Dict[str, Any]) -> Dict[str, Any]:
    """Phases until the median agent's past news cone holds half the world, at each moment read, and the
    world's width in steps over the rules' speed, the same for a network that stood still."""
    measured, still = [], []
    for c in w["cones"]:
        measured.append(first(np.asarray(c["agent"]["news"], float), 0.5))
        still.append(c["half_reach"] / c["news_speed"] if c["half_reach"] == c["half_reach"] else None)
    ok = [m for m in measured if m is not None]
    st = [s for s in still if s is not None]
    return {"measured": float(np.median(ok)) if ok else None, "still": float(np.median(st)) if st else None,
            "half_reach": _median([c["half_reach"] for c in w["cones"]]),
            "speed": w["cones"][0]["news_speed"] if w["cones"] else None}


def middle(band: Dict[str, Any]) -> Dict[str, Any]:
    """A band with only its middle half, where several overlap."""
    return {k: v for k, v in band.items() if k not in ("outerLo", "outerHi")}


def _rho(a: List[float], b: List[float]) -> Optional[float]:
    """Spearman's rank correlation of two lists, over the places where both have a value."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    ra, rb = (np.argsort(np.argsort(np.asarray(v, float))) for v in zip(*pairs))
    return float(np.corrcoef(ra, rb)[0, 1])


def _median(values: List[Optional[float]]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _q(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not v.size:
        return {"median": None, "q1": None, "q3": None, "min": None, "max": None, "n": 0}
    return {"median": float(np.median(v)), "q1": float(np.quantile(v, 0.25)), "q3": float(np.quantile(v, 0.75)),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


@chapter
def wolfram(ch: F.Chapter) -> None:
    if not os.path.exists(READINGS):
        return
    worlds = [w for w in json.load(open(READINGS))["worlds"] if w.get("cones")]
    if not worlds:
        return
    search = lambda w: w["settings"].get("link_search") or "off"
    cen = {w["id"]: census(w) for w in worlds}

    # ---- What a world rewrites, and how locally
    keys = ("births", "deaths", "starved", "culled", "made", "made_old", "made_birth", "lapsed", "old_local",
            "old_three", "old_far", "old_beyond", "old_out", "birth_far", "cull_share", "nonlocal", "links_per_agent", "cut",
            "link_life", "agent_life")
    ch.number("census", {k: _q([cen[w["id"]][k] for w in worlds]) for k in keys})
    ch.number("census_by_search", {
        how: {"worlds": sum(search(w) == how for w in worlds),
              **{k: _q([cen[w["id"]][k] for w in worlds if search(w) == how]) for k in
                 ("old_local", "old_far", "old_beyond", "old_out", "nonlocal", "made_old", "birth_far",
                  "cull_share")}}
        for how, _, _ in SEARCHES})
    mend = lambda w: bool(w["settings"].get("mend_on_death"))
    ch.number("cull_by_mending", {side: _q([cen[w["id"]]["cull_share"] for w in worlds if mend(w) == side])
                                  for side in (True, False)})
    names = [words for _, words, _ in SEARCHES]
    colours = [c for _, _, c in SEARCHES]
    stranger = {w["id"]: _median([1 - c["within2"] for c in w["cones"]]) for w in worlds}
    ch.number("stranger_far", _q([stranger[w["id"]] for w in worlds]))
    ch.grid("locality", [
        {"title": "new links across three steps or more", "x": {"label": "", "categories": names},
         "y": {"label": "share of links made", "min": 0, "max": 1},
         "series": dots(names, [[cen[w["id"]]["old_beyond"] for w in worlds if search(w) == how]
                                for how, _, _ in SEARCHES], colours),
         "legend": False, "height": 220},
        {"title": "rewrites no local rule makes", "x": {"label": "", "categories": names},
         "y": {"label": "share of all rewrites", "min": 0, "max": 0.6},
         "series": dots(names, [[cen[w["id"]]["nonlocal"] for w in worlds if search(w) == how]
                                for how, _, _ in SEARCHES], colours),
         "legend": False, "height": 220}],
        columns=2, title="How local a world's rewrites are",
        caption=f"Each dot is one of the {len(worlds)} worlds of round 3 that lived, read at every 100th iteration "
                "from 100 on, in both phases; the short bar is the median of a column. The columns are how the "
                "world's agents look for new links (`link_search`): not at all, a scout that walks, a search that "
                "reaches as far as its tokens pay, a search that jumps. Left: of the links made between two agents "
                "that were there before the phase, the share whose ends were three steps apart or more just before "
                "it. A link between agents two steps apart closes a triangle, the most local link a rewriting rule "
                "can make. An agent drawn at random lies three steps away or more in "
                f"{np.median(list(stranger.values())):.1%} of draws, in the median world. Right: the share of all "
                "of a phase's rewrites (a birth, a death, a link made, a link that lapses) that no rule acting within "
                "two steps could make: a link made across three steps or more, a newborn's link to an agent beyond its "
                "parent's neighbours, and a death in the cull.",
        recipe=recipe(RUNS, DATA, [
            "At every 100th iteration from 100 to 2,900, read the frames just before the reproduction phase, "
            "after it and after the game, and build each one's network from `edges`.",
            "A phase's rewrites: agents in the later frame and not the earlier are born, the other way round "
            "dead; the cleanup's `starved`, `orphaned` (culled) and `aged` say how they died. Links in one "
            "frame's `edges` and not the other's are made or cut.",
            "For a link made between two agents of the earlier frame, the number of steps between them in the "
            "earlier frame's network (8 for eight or more, or in another piece). For a newborn's link, the steps "
            "between its other end and the newborn's parent (`parent_ids`).",
            "Sum over the phases read; the shares are of those sums."]))

    # ---- Space through a life
    grid = np.arange(0, 3000, 25)
    def table(col: int) -> np.ndarray:
        out = np.full((len(worlds), len(grid)), np.nan)
        for i, w in enumerate(worlds):
            for row in w["space"]["rows"]:
                j = int(row[0]) // 25
                if row[0] % 25 == 0 and j < len(grid) and row[col] is not None:
                    out[i, j] = row[col]
        return out
    dim, ric = table(1), table(2)
    # Means over stretches of 100 iterations, so that the noise of a reading from 24 agents does not hide the course.
    block = 4
    usable = (len(grid) // block) * block
    thin = lambda t: np.nanmean(t[:, :usable].reshape(len(t), -1, block), axis=2)
    span = grid[:usable].reshape(-1, block).mean(axis=1)
    ch.number("space", {"dimension_start": _q(dim[:, 0].tolist()), "dimension_end": _q(dim[:, -1].tolist()),
                        "ricci_start": _q(ric[:, 0].tolist()), "ricci_end": _q(ric[:, -1].tolist()),
                        "dimension_settled": _q(np.nanmedian(dim[:, grid >= 500], axis=1).tolist()),
                        "ricci_settled": _q(np.nanmedian(ric[:, grid >= 500], axis=1).tolist()),
                        "ricci_positive_settled": float(np.mean(np.nanmedian(ric[:, grid >= 500], axis=1) > 0))})
    ch.grid("space", [
        {"title": "dimension", "x": {"label": "iteration"}, "y": {"label": "dimension", "min": 0, "max": 4},
         "series": [{**quantile_band(thin(dim), span), "label": f"the {len(worlds)} worlds", "colour": BLUE}],
         "guides": [{"axis": "y", "at": d, "label": ""} for d in (1, 2, 3)], "height": 220},
        {"title": "Ricci scalar curvature", "x": {"label": "iteration"}, "y": {"label": "curvature R", "min": -1.5,
                                                                              "max": 1.5},
         "series": [{**quantile_band(thin(ric), span), "label": f"the {len(worlds)} worlds", "colour": VIOLET}],
         "guides": [{"axis": "y", "at": 0, "label": "flat"}], "height": 220}],
        columns=2, title="Space and its curvature through a life, as Wolfram reads them",
        caption=f"The {len(worlds)} worlds of round 3 that lived, read after every 25th game and averaged over "
                "stretches of 100 iterations. Left: the viewer's "
                "`dimension`, from how the shells around about 24 agents grow out to five steps (1 a line, 2 a "
                "sheet, 3 a solid). Right: its `ricciCurvature`, the same fit's correction that grows with the "
                "square of the radius (positive where balls grow more slowly than flat space allows, negative where "
                "they grow faster). " + BAND_WORDS + ".",
        recipe=recipe(RUNS, "each run's `GraphOfLifeRuns/<run>/stats.jsonl`, the rows after the game (`phase` 2) "
                      "that hold `dimension` and `ricciCurvature` (every 25th iteration), gathered by "
                      "`research/atlas/wolfram.py` into `research/atlas/wolfram.json`",
                      ["One line of 120 points per world, one every 25 iterations; the mean of each four, "
                       "a stretch of 100 iterations.",
                       "At each point the median of the worlds, the middle half and nine in ten."]))

    # ---- Past cones
    phases = np.arange(1, len(worlds[0]["cones"][0]["agent"]["news"]) + 1)
    curves = {kind: [cone_curves(w, kind) for w in worlds] for kind, _, _ in KINDS}
    cross = {w["id"]: crossing(w) for w in worlds}
    def over_all(values: List[Optional[int]]) -> Dict[str, Any]:
        """The median over every world, a world whose cone never held half the world in the phases read counted
        as further back than all of them (None if that is where the median falls)."""
        v = sorted(np.inf if x is None else x for x in values)
        m = float(np.median(v))
        return {"median": m if np.isfinite(m) else None, "reached": sum(x is not None for x in values),
                "worlds": len(values)}
    ch.number("cones", {
        kind: {"agent_half": _q([first(c["agent"], 0.5) for c in curves[kind]]),
               "agent_half_all": over_all([first(c["agent"], 0.5) for c in curves[kind]]),
               "agent_at": {str(k): _q([float(c["agent"][k - 1]) for c in curves[kind]]) for k in (2, 4, 8, 16, 40)},
               "whole_at": {str(k): _q([float(c["whole"][k - 1]) for c in curves[kind]]) for k in (2, 4, 8, 16, 40)},
               "whole_min": _q([float(c["whole"].min()) for c in curves[kind]])}
        for kind, _, _ in KINDS})
    ch.number("crossing", {"measured": _q([cross[w["id"]]["measured"] for w in worlds]),
                           "still": _q([cross[w["id"]]["still"] for w in worlds]),
                           "half_reach": _q([cross[w["id"]]["half_reach"] for w in worlds]),
                           "speed4": sum(cross[w["id"]]["speed"] == 4 for w in worlds),
                           "within_one": float(np.mean([abs(cross[w["id"]]["measured"] - cross[w["id"]]["still"]) <= 1
                                                        for w in worlds if cross[w["id"]]["measured"] is not None
                                                        and cross[w["id"]]["still"] is not None]))})
    ch.grid("cones", [
        {"title": "the past of one agent", "x": {"label": "phases back"},
         "y": {"label": "share of that earlier world", "min": 0, "max": 1},
         "series": [{**middle(quantile_band([c["agent"] for c in curves[kind]], phases)), "label": words,
                     "colour": colour}
                    for kind, words, colour in KINDS], "height": 230},
        {"title": "the past of the whole world", "x": {"label": "phases back"},
         "y": {"label": "share of that earlier world", "min": 0, "max": 1},
         "series": [{**middle(quantile_band([c["whole"] for c in curves[kind]], phases)), "label": words,
                     "colour": colour}
                    for kind, words, colour in KINDS], "height": 230}],
        columns=2, title="Past light cones",
        caption=f"The {len(worlds)} worlds of round 3 that lived. An agent's past cone, k phases back, is every agent "
                "of that earlier world that could have had an effect on it by then. Blue: news, as far as the "
                "rules let it travel in a phase (four steps, three in worlds without the look that only talks); "
                "orange: matter, a step a phase, as a stake, a gift or a child moves; green: the tokens and brains "
                "that did move, read from the decisions (stakes, nodes won, gifts, children and a mate's share). "
                "The global rules, the cull, the share-out and the one random stream, are left out, though each "
                "reaches the whole world at once. Left: the median of 32 agents drawn at random in each world, "
                "after the games of iterations 1,000, 2,000 and 2,900 (the median of the three moments). Right: "
                "the past cone of everyone alive at that moment together. The line is the median of the worlds at "
                "each phase back, and the band holds the middle half of them (from the 25th to the 75th "
                "percentile).",
        recipe=recipe(RUNS, DATA, [
            "After the game of iteration 1,000, 2,000 and 2,900, draw 32 agents (seeded by the run's seed).",
            "Go back a phase at a time. An agent of the later frame that was in the earlier one marks itself "
            "there; a newborn marks its parent. News: every agent within four steps of a marked agent, three "
            "without `message_prepass` (three and two of a newborn's parent), in the earlier frame's network, "
            "joins the cone. Matter: within one step (none from a newborn's parent). Flow: the senders of the "
            "phase's stakes, nodes won, gifts, children and mates' shares to a member join it.",
            "Each cone's size over the earlier frame's agents, for 40 phases (20 iterations); the median over "
            "the 32 agents, then over the three moments.",
            "The whole world's cone starts from every agent alive at the moment."]))
    pts = [(cross[w["id"]]["still"], cross[w["id"]]["measured"]) for w in worlds
           if cross[w["id"]]["still"] is not None and cross[w["id"]]["measured"] is not None]
    top = max(max(p) for p in pts) + 1 if pts else 10
    ch.figure("crossing", title="How many phases wide a world is",
              x={"label": "width over the speed of news (phases)", "min": 0, "max": top},
              y={"label": "phases back to half the world", "min": 0, "max": top},
              series=[line([p[0] for p in pts], [p[1] for p in pts], None, BLUE, points=True, size=6, alpha=0.8),
                      line([0, top], [0, top], "as in a network that stands still", GREY, dash=[4, 4])],
              height=260,
              caption=f"Each dot is one of the {len(worlds)} worlds of round 3 that lived. Across: the steps within "
                      "which a typical agent's ball holds half the world (from 40 agents drawn at random), over the "
                      "steps news can go in a phase by the rules (four, or three without the look that only talks): "
                      "how many phases a world would be wide if its network stood still. Up: the phases back until "
                      "the median agent's past news cone held half the world, as measured through the world's own "
                      "changing network. Both are the medians of the three moments of the figure above.",
              recipe=recipe(RUNS, DATA, [
                  "The width: the typical ball (the geometric mean over 40 agents) out of the frame after the game of "
                  "iteration 1,000, 2,000 and 2,900; the radius at which it first holds half the world, between "
                  "whole steps by the logarithm of the ball (`research/atlas/space.half_reach`).",
                  "Over the speed of news: 4 steps a phase, or 3 without `message_prepass`.",
                  "Measured: the first phase back at which the median agent's past news cone held half the world.",
                  "The median of the three moments, for each."]))

    # ---- Does matter bend space?
    agent = [w["curvature"]["agent"] for w in worlds if w["curvature"].get("agent")]
    ch.number("matter", {k: _q([a.get(k) for a in agent]) for k in
                         ("rho_tokens", "rho_around", "rho_degree", "rho_activity", "partial_tokens_degree",
                          "partial_around_degree", "partial_activity_degree", "partial_tokens_links",
                          "partial_around_links", "partial_activity_links", "rho_beside", "rho_around_beside",
                          "rho_tokens_degree", "rho_activity_degree", "rho_activity_tokens")})
    ch.number("matter_worlds", len(agent))
    # Across worlds: does a world that rewrites more, or holds more links an agent, curve more?
    settled = {w["id"]: v for w, v in zip(worlds, np.nanmedian(ric[:, grid >= 500], axis=1).tolist())}
    busy = {w["id"]: cen[w["id"]]["births"] + cen[w["id"]]["deaths"] + cen[w["id"]]["made"] + cen[w["id"]]["lapsed"]
            for w in worlds}
    ids = [w["id"] for w in worlds]
    ch.number("across_worlds", {
        "rho_ricci_rewrites": _rho([settled[i] for i in ids], [busy[i] for i in ids]),
        "rho_ricci_links": _rho([settled[i] for i in ids], [cen[i]["links_per_agent"] for i in ids]),
        "rho_rewrites_links": _rho([busy[i] for i in ids], [cen[i]["links_per_agent"] for i in ids]),
        "rewrites": _q([busy[i] for i in ids])})
    dec = [w["curvature"]["deciles"] for w in worlds if w["curvature"].get("deciles")]
    rmax = min(len(d["richest"]) for d in dec)
    radius = np.arange(rmax)
    rich = np.asarray([d["richest"][:rmax] for d in dec])
    poor = np.asarray([d["poorest"][:rmax] for d in dec])
    ch.number("deciles", {"richest": np.median(rich, axis=0).tolist(), "poorest": np.median(poor, axis=0).tolist(),
                          "degree_richest": _q([d["degree_richest"] for d in dec]),
                          "degree_poorest": _q([d["degree_poorest"] for d in dec]),
                          "degree_all": _q([d["degree_all"] for d in dec])})
    columns = ["its links", "its neighbours' links", "its tokens", "the tokens around it", "the rewrites at it"]
    ch.grid("matter", [
        {"title": "balls around the rich and the poor", "x": {"label": "radius r, steps"},
         "y": {"label": "ln of the ball over the typical ball", "min": -0.6, "max": 0.6},
         "series": [{**quantile_band(rich, radius), "label": "the richest tenth", "colour": ORANGE},
          {**quantile_band(poor, radius), "label": "the poorest tenth", "colour": BLUE}],
         "guides": [{"axis": "y", "at": 0, "label": "the typical ball"}], "height": 230},
        {"title": "what goes with an agent's own curvature", "x": {"label": "", "categories": columns},
         "y": {"label": "Spearman's rho, across the agents of a world", "min": -0.6, "max": 0.6},
         "series": dots(columns, [[a["rho_degree"] for a in agent], [a.get("rho_beside") for a in agent],
                                  [a.get("partial_tokens_links") for a in agent],
                                  [a.get("partial_around_links") for a in agent],
                                  [a.get("partial_activity_links") for a in agent]],
                        [GREY, VIOLET, ORANGE, YELLOW, GREEN]),
         "guides": [{"axis": "y", "at": 0, "label": ""}], "legend": False, "height": 230}],
        columns=1, title="Does matter bend a world's space?",
        caption=f"The {len(agent)} worlds of round 3 that lived and whose typical ball needs more than three steps to "
                "hold half of them, in their network after the last game (its largest piece). Left: around the tenth of the agents with the most tokens (orange) and the tenth with the "
                "fewest (blue), how many agents lie within r steps, against the typical ball of the world (the "
                "geometric mean over all its agents), as the natural logarithm of the ratio: 0.1 is about 10% more. "
                + BAND_WORDS.replace("at each iteration", "at each radius") + ". Right: each dot is one world; "
                "across its agents, the rank correlation of an agent's own curvature (the r² term of the growth of "
                "its ball out to five steps, its dimension held at the world's) with its links and with its "
                "neighbours' links (their mean); and, with both held equal (a partial correlation), with its tokens, "
                "with the tokens around it (the mean over it and its neighbours) and with how much was rewritten at "
                "it in the last 20 iterations (links made or cut, children, its node taken). Positive: more of it, "
                "more positive curvature, a ball that grows more slowly.",
        recipe=recipe(RUNS, DATA, [
            "Read the last frame, after the game of iteration 2,999, and keep the largest piece of its network.",
            "Every agent's ball V(r), r = 0 to 5, by spreading from all agents at once.",
            "The world's dimension d from ln V(r) = a + d ln r − R r² / (6 (d + 2)) fitted to the typical ball "
            "(Wolfram 2020); each agent's own R from the same with d held at the world's. Its ranks do not depend "
            "on d.",
            "Rewrites at an agent: over the last 40 phases, the links made or cut at it, the children it had and "
            "the times its node took another brain, per phase.",
            "Spearman's rho across the agents; a partial rho correlates the residuals of both ranks after a plane "
            "in the ranks of the agent's links and of its neighbours' mean links."]))
