"""
Round 3 ranked by evolution, every reading against its null (2026-11-12).

The shortlist of research/screen/funnel.py is drawn from health's orders; the
family trees it rules worlds out by follow births only, although conquests copy
brains several times as often. This ranks the 82 worlds that lived by what
research/screen/braintree.py, selection.py and brainplay.py read on the
brain-copy tree, each reading set against a null model:

  shadow     lines spread beyond luck: the shadow world keeps the same copies
             and the same living but draws every copy's parent at random among
             the living; the mean over the eras of three z-scores, how many
             fewer of the first living's clades still have descendants at the
             era's end, how much larger the largest clade is then, and how much
             more of the evolutionary activity lies in clades more active than
             99% of the shadow's (Bedau and Packard). How long the average
             clade lives is read but not scored: copies are local, and kin
             taking kin's nodes makes clades last longer whatever selection does
  novelty    the line of descent's late novelty, above its own noise, less that
             of lines that start from its brain at iteration 250 and make the
             same changes at random (the mutation-only null)
  selection  the strength of selection on behaviour: the median size of the
             standardized selection gradient, wealth held fixed, over four
             behaviours and twelve eras (its null: fitness shuffled)
  heredity   the heredity index on the brain tree's own parent-child pairs (its
             null: children given another pair's parent)

Each reading is scored as the share of the worlds it beats (the best 1, the worst
1/n, as health.py scores its measures: continuous, positive, bigger is better,
no pass marks), and the four shares are combined three ways (the author's
style, 2026-11-09): multiplied (read as their geometric average), added (read
as their average), and the worst of them. `score` is the product; the rank
orders by it, then by the sum, the worst and the world's id, so every rank is
distinct. Each world carries its four readings (`values`), their z-scores
against their nulls (`z`), and beside them what is not scored (`beside`): how
its brains were copied, and the births-only numbers these readings replace. The best of each order are listed, and their union. Health is beside
it only as a filter: its product and rank (research/screen/health.json),
whether the world is on the shortlist (research/screen/candidates.json), and
`filter`, the funnel's reason where one of its rules that still filter rules
the world out (a star, one holding, standing still; dead worlds are not
ranked). The rank itself scores every world that lived.

    python3 research/screen/evolution.py      writes research/screen/evolution.json

Worlds whose readings are not all made yet are listed as missing, and the file
says `"provisional": true` until all 82 are.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import braintree as BT  # noqa: E402

OUT = os.path.join(HERE, "evolution.json")
#: How far down each order the bests reach.
TOP = 10

READINGS = [
    {"key": "shadow", "name": "lines spread beyond luck", "better": "more",
     "null": "the shadow world of pure luck: the same copies at the same moments and the same living, every "
             "copy's parent drawn uniformly among the living (100 replicates); the mean over the twelve eras of the "
             "z-scores of how many fewer of the clades of the era's first living still have descendants at its "
             "end, how much larger the largest clade is then, and how much more of the evolutionary activity is "
             "held by clades more active than 99% of the shadow's"},
    {"key": "novelty", "name": "novelty above the mutation-only null", "better": "more",
     "null": "six lines that start from the line of descent's brain at iteration 250 and make, era by era, as many "
             "changes of the world's own kind as the real line made, at random (mutations by the world's "
             "mutation_noise_std and mutation_sparsity; mixes with a contemporary drawn at random); the real line's "
             "late novelty above its own noise, less the null lines' mean (z: in the null lines' spreads)"},
    {"key": "selection", "name": "selection-gradient strength", "better": "more",
     "null": "fitness (descendants at the next iteration on the brain tree) shuffled among the agents of each "
             "iteration, whose mean (0) and variance the permutation fixes exactly; the value is the median size of "
             "the standardized gradient, wealth held fixed, over four behaviours and twelve eras (a median, since "
             "the links a newborn gets are heavy-tailed); z is the root mean square of the eras' gradient z-scores"},
    {"key": "heredity", "name": "heredity on the brain tree", "better": "more",
     "null": "children given another pair's parent at random (200 shuffles): an index of 0 means a child acts "
             "like a stranger, 1 like its parent; z is how many spreads of the shuffled medians the real median "
             "child-parent distance lies below them"},
]
KEYS = [r["key"] for r in READINGS]
ORDER = ("score is the product of the four readings' shares (each the share of the 82 worlds it beats: the best 1, "
         "the worst 1/n), read as their geometric average; rank orders by score, then by the sum of the shares (as "
         "their average), then by the worst share, then by the world's id. The orders by the sum and by the worst "
         "are kept beside it, and the union of the first ten of each (bests).")


#: The funnel's rules (research/screen/funnel.py) that still filter: health of 9 November, and a line that stands
#: still. Its rule of no heredity read the births-only tree, which played children against parents whose brains
#: had mostly been replaced by conquests since; on the brain tree it does not hold, so it filters nothing.
FILTERS = ("died", "a star", "one holding", "standing still")


def filtered(out: Optional[str]) -> Optional[str]:
    """The funnel's reason for ruling a world out, where it is one that still filters."""
    return out if out and out.startswith(FILTERS) else None


def shares(values: List[float]) -> List[float]:
    """Each value's place among them all, as the share of them it does better than: the best 1, the worst 1/n."""
    n = len(values)
    return [(sum(v < x for v in values) + (sum(v == x for v in values) + 1) / 2) / n for x in values]


def _mean(values: List[Optional[float]]) -> Optional[float]:
    have = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(have)) if have else None


def readings_of(have: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    """A world's four readings: each its value (the one scored) and its z against its null."""
    sh = have["shadow"]["z"]
    # How long the average clade lives is left out: copies are made next door, kin taking kin's nodes, and that
    # alone makes clades last longer than in a shadow whose parents come from anywhere, whatever selection does.
    # The three kept are the ones locality pushes the other way (as in a voter model, lines meet and sweep more
    # slowly where copies are local), so a world that beats its shadow on them does so in spite of its locality.
    shadow = _mean([sh.get("persist"), sh.get("top"), sh.get("active_share")])
    line = have.get("line") or {}
    sel = have.get("selection") or {}
    her = [h for h in (have.get("heredity") or {}).values() if isinstance(h, dict)]
    return {"shadow": {"value": shadow, "z": shadow},
            "novelty": {"value": line.get("late_above_null"), "z": line.get("z")},
            "selection": {"value": sel.get("strength"), "z": sel.get("visible_gradient")},
            "heredity": {"value": _mean([h.get("index") for h in her]), "z": _mean([h.get("z") for h in her])}}


def values_of(have: Dict[str, Any], births_only: Dict[str, Any]) -> Dict[str, Any]:
    """What is not scored but read beside the score: how brains were copied, and the births-only numbers it replaces."""
    tree, sh, sel = have["tree"], have["shadow"], have.get("selection") or {}
    copies = tree["copies"]
    end = tree["generations"].get(str(max(int(k) for k in tree["generations"]))) or {}
    her = {k: v for k, v in (have.get("heredity") or {}).items() if isinstance(v, dict)}
    line = have.get("line") or {}
    return {
        "birth_share": copies.get("birth_share"), "conquests_per_agent": copies.get("conquests_per_agent"),
        "conquests_per_agent_2000": copies.get("conquests_per_agent_2000"),
        "births_per_agent": copies.get("births_per_agent"), "conquests_per_birth": copies.get("conquests_per_birth"),
        "generations": end.get("mean"), "births_on_path": end.get("births"), "conquests_on_path": end.get("conquests"),
        "generations_births_only": births_only.get("generations"),
        "common_ancestor_age": tree.get("common_ancestor_age"),
        "common_ancestor_age_shadow": sh["coalescence"].get("shadow_median"),
        "coalescence_p": sh["coalescence"].get("p"),
        "effective_population": tree.get("effective_population"),
        "effective_population_shadow": sh["effective_population"].get("shadow"),
        "effective_population_births_only": births_only.get("effective_population"),
        "heredity_births_only": births_only.get("heredity"),
        "heredity_birth_pairs": _mean([(h.get("birth") or {}).get("index") for h in her.values()]),
        "heredity_conquest_pairs": _mean([(h.get("conquest") or {}).get("index") for h in her.values()]),
        "births_only_pairs_still_brain_pairs": _mean([p.get("births_only_still_brain_pairs")
                                                      for p in tree.get("pairs", {}).values()]),
        "novelty_above_noise": line.get("late_above_noise"),
        "novelty_null_above_noise": _mean((line.get("null") or {}).get("late_above_noise") or []),
        "novelty_births_only": births_only.get("late_novelty"),
        "shadow_z": sh["z"],
        "selection_visible": sel.get("visible"), "selection_clear": sel.get("clear"),
        "selection_cells": sel.get("cells"), "selection_follows": sel.get("follows"),
        "selection_same_sign": sel.get("same_sign"),
    }


def rank(worlds: List[Dict[str, Any]], top: int = TOP) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Each world's shares of its readings (`values`), the three orders over them, its score and distinct rank;
    the worlds sorted by rank, and the first `top` of each order. A reading not made scores as the worst.
    """
    if not worlds:
        return [], {}
    each = {k: shares([w["values"][k] if w["values"].get(k) is not None and math.isfinite(w["values"][k])
                       else -math.inf for w in worlds]) for k in KEYS}
    for i, w in enumerate(worlds):
        s = {k: each[k][i] for k in KEYS}
        w["shares"] = s
        w["orders"] = {"product": float(math.prod(s.values()) ** (1 / len(s))),
                       "sum": float(np.mean(list(s.values()))), "worst": float(min(s.values()))}
        w["score"] = w["orders"]["product"]
    worlds = sorted(worlds, key=lambda w: (-w["orders"]["product"], -w["orders"]["sum"], -w["orders"]["worst"],
                                           w["id"]))
    for i, w in enumerate(worlds, 1):
        w["rank"] = i
    bests = {}
    for order in ("product", "sum", "worst"):
        ranked = sorted(worlds, key=lambda w: (-w["orders"][order], -w["orders"]["product"], w["id"]))
        bests[order] = [w["id"] for w in ranked[:top]]
        for i, w in enumerate(ranked, 1):
            w.setdefault("ranks", {})[order] = i
    return worlds, bests


def build() -> Dict[str, Any]:
    living = BT.living_worlds()
    with open(os.path.join(HERE, "health.json")) as fh:
        health = {w["id"]: w for w in json.load(fh)["worlds"]}
    cand = {}
    shortlist: List[str] = []
    if os.path.exists(os.path.join(HERE, "candidates.json")):
        with open(os.path.join(HERE, "candidates.json")) as fh:
            c = json.load(fh)
        cand = {w["id"]: w for w in c["worlds"]}
        shortlist = list(c.get("shortlist") or [])
    done, missing = [], []
    for run_id, _, _ in living:
        have = BT.kept(run_id)
        (done if all(k in have for k in BT.ITEMS) else missing).append((run_id, have))
    worlds = []
    for run_id, have in done:
        h = health[run_id]
        r = readings_of(have)
        worlds.append({"id": run_id, "round": h["round"], "seed": h["seed"],
                       "conquest_share": have["tree"]["copies"].get("conquest_share"),
                       "z": {k: r[k]["z"] for k in KEYS}, "values": {k: r[k]["value"] for k in KEYS},
                       "beside": values_of(have, (cand.get(run_id) or {}).get("numbers") or {}),
                       "health": {**(h.get("health") or {}), "ranks": h.get("ranks")},
                       "shortlisted": run_id in shortlist, "out": (cand.get(run_id) or {}).get("out"),
                       "filter": filtered((cand.get(run_id) or {}).get("out"))})
    worlds, bests = rank(worlds)
    return {"made": datetime.date.today().isoformat(), "provisional": bool(missing),
            "complete": len(done), "of": len(living), "missing": [r for r, _ in missing],
            "order": ORDER, "readings": READINGS, "top": TOP, "bests": bests,
            "union": sorted({i for ids in bests.values() for i in ids}), "worlds": worlds}


def write() -> Dict[str, Any]:
    out = build()
    with open(OUT + ".tmp", "w") as fh:
        json.dump(BT._plain(out), fh, indent=1)
    os.replace(OUT + ".tmp", OUT)
    return out


def label(w: Dict[str, Any]) -> str:
    return f"s{w['seed']}" + {"E41": " w", "E42": " s"}.get(w["round"], "")


def main() -> None:
    out = write()
    print(f"{out['complete']} of {out['of']} worlds read" + (" (provisional)" if out["provisional"] else ""))
    print("rank  world      score  product sum worst  shadow  novelty  selection  heredity  conquests  health  short")
    for w in out["worlds"]:
        r = w["values"]
        f = lambda v, form="{:.2f}": "-" if v is None else form.format(v)
        print(f"{w['rank']:4d}  {label(w):9s} {w['score']:.3f}  {w['ranks']['product']:4d} {w['ranks']['sum']:4d} "
              f"{w['ranks']['worst']:4d}  {f(r['shadow'], '{:6.1f}')}  {f(r['novelty'], '{:+.3f}')}  "
              f"{f(r['selection'], '{:.3f}')}      {f(r['heredity'])}      {f(w['conquest_share'])}     "
              f"{(w['health'].get('ranks') or {}).get('product', '-')}  {'yes' if w['shortlisted'] else ''}")


if __name__ == "__main__":
    main()
