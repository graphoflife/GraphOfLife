"""
How the drawn settings shape a world: research/screen/effects.py's tests of
every setting of round 3 against every outcome (Experiment 40's 84 worlds,
each a draw of its own), the pairs of values whose worlds die more than
either value's alone, the worlds' mechanisms against their health, and the
eight draws played a token at a time, weakest first and strongest first
(Experiments 41 and 42). Nothing is read here but effects.json and health.json.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict

import book_figures as F
from book_chapters.common import BLUE
from book_figures import chapter, dots, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")
EFFECTS = os.path.join(SCREEN, "effects.json")
HEALTH = os.path.join(SCREEN, "health.json")
#: The strongest evidence a test can give: one shuffle in all of them as extreme.
SHUFFLES = 20000
#: A finding: a false discovery rate below this.
FINDING = 0.1

RUNS = ("Experiment 40's 84 worlds, each with its settings drawn at random by the screen's second sampler "
        "(30,000 tokens, 3,000 iterations); 14 died and 70 lived to the end")
DATA = ("each world's drawn settings (its plan, `book/experiments/E40.json`, drawn by `gol_screen.py`), and its "
        "outcomes: whether it died, and, for the living, its health and numbers "
        "(`research/screen/health.json`) and its dossier (`GraphOfLifeRuns/.pilots/screen/candidates/<run>.json`), "
        "tested by `research/screen/effects.py` into `research/screen/effects.json`")


#: Each outcome's name under the map, short enough for its column; the caption says them in full.
SHORT = {"died": "died", "died_rest": "died*", "health": "health", "c_decentralised": "decent.",
         "c_resilient": "resil.", "c_space": "3-D", "c_diverse": "diverse", "c_uneven": "uneven",
         "c_culls": "culls", "culls": "culled",
         "agents": "agents", "apart": "apart", "gini": "Gini", "heredity": "hered.", "generations": "gener.",
         "population": "popul.", "novelty": "novelty", "fertility": "fertil."}


def _setting(key: str) -> Dict[str, str]:
    return {"text": f"`{key}`", "href": f"pages/settings.md#{key}"}


def _value(v: str) -> str:
    return json.loads(v) if v.startswith('"') else v


@chapter
def settingseffects(ch: F.Chapter) -> None:
    if not (os.path.exists(EFFECTS) and os.path.exists(HEALTH)):
        return
    e = json.load(open(EFFECTS))
    h = json.load(open(HEALTH))
    tests = e["tests"]
    settings = sorted({r["setting"] for t in tests for r in t["rows"]})

    # Which settings matter for what: every test's evidence, as a map.
    strength = lambda chance: min(math.log10(SHUFFLES + 1), -math.log10(chance))
    cells = []
    for j, t in enumerate(tests):
        for r in t["rows"]:
            i = len(settings) - 1 - settings.index(r["setting"])
            cells.append((j - 0.5, j + 0.5, i - 0.5, i + 0.5, strength(r["chance"])))
    ch.figure("matrix", title="Which settings matter for what",
              x={"label": "", "categories": [SHORT.get(t["outcome"], t["outcome"]) for t in tests]},
              y={"label": "", "categories": list(reversed(settings))},
              series=[F.cells(cells)], legend=False, height=16 * len(settings),
              colourbar={"map": "viridis", "min": 0, "max": math.log10(SHUFFLES + 1),
                         "label": "evidence: -log10 of the chance by shuffling (2 = 1 in 100)"},
              caption="Each row is a drawn setting, each column an outcome of the worlds; a cell is bright where "
                      "the worlds drawn with the setting's different values differ in that outcome more than "
                      "shuffling the outcome among the worlds would make them differ. Its brightness is how "
                      "unlikely that is by chance: 1 for a chance of 1 in 10, 2 for 1 in 100, 4.3 for none in "
                      f"{SHUFFLES:,} shuffles. Many cells are tested, so a few are bright by luck alone: the table "
                      "below gives each its false discovery rate. Whether a world died is tested over all of them; "
                      "every other outcome over the living. A blank cell was not tested (a setting that cannot "
                      "matter there, or too few worlds). The columns: "
                      + "; ".join(f"{SHORT.get(t['outcome'], t['outcome'])}, {t['name']}" for t in tests) + ".",
              recipe=recipe(RUNS, DATA, [
                  "For each setting, the worlds where it can matter (as the sampler draws it), grouped by its value.",
                  "Died: the chi-square statistic of the deaths across the groups; numbers: the Kruskal-Wallis "
                  "statistic of their ranks.",
                  f"The share of {SHUFFLES:,} shuffles of the outcome among the worlds whose statistic is at least "
                  "as large (seed 11): the chance.",
                  "The false discovery rate of each chance among all settings tested on that outcome "
                  "(Benjamini-Hochberg)."]))

    found = []
    for t in tests:
        key = "rate" if t["binary"] else "median"
        for r in t["rows"]:
            if r["fdr"] < FINDING:
                by = "; ".join(f"{_value(g)}: {d[key]:.2f} ({d['worlds']})" for g, d in r["by"].items())
                found.append([t["name"], _setting(r["setting"]), f"{r['chance']:.4f}", f"{r['fdr']:.3f}", by])
    ch.number("table:found", {"head": ["outcome", "setting", "chance by shuffling", "false discovery rate",
                                       "by value: the share that died, or the median (worlds)"],
                              "rows": found})
    ch.number("found", len(found))

    # Every setting against death and health, in numbers.
    died = next(t for t in tests if t["outcome"] == "died")
    health = next(t for t in tests if t["outcome"] == "health")
    rows = []
    for s in settings:
        d = next((r for r in died["rows"] if r["setting"] == s), None)
        hh = next((r for r in health["rows"] if r["setting"] == s), None)
        rows.append([_setting(s),
                     "; ".join(f"{_value(g)}: {v['rate'] * v['worlds']:.0f}/{v['worlds']}" for g, v in d["by"].items())
                     if d else "—", f"{d['chance']:.3f}" if d else "—",
                     "; ".join(f"{_value(g)}: {v['median']:.2f}" for g, v in hh["by"].items()) if hh else "—",
                     f"{hh['chance']:.3f}" if hh else "—"])
    ch.number("table:all", {"head": ["setting", "died, by value", "chance", "median health of the living, by value",
                                     "chance"], "rows": rows})

    # Deaths by value, for the settings that go with dying: the cap on links over all worlds, the others with
    # the six capped worlds set aside.
    died_rest = next(t for t in tests if t["outcome"] == "died_rest")
    charts = []
    for test, setting in ((died, "valence"), (died_rest, "allow_gifting"), (died_rest, "contest"),
                          (died_rest, "link_search")):
        row = next((r for r in test["rows"] if r["setting"] == setting), None)
        if row is None:
            continue
        values = sorted(row["by"], key=lambda g: (not g[0].isdigit(), float(g) if g[0].isdigit() else 0, g))
        names = [str(_value(g)) for g in values]
        chance = f"{row['chance']:.3f}" if row["chance"] >= 0.001 else "< 0.001"
        charts.append({"title": f"{setting}{'' if setting == 'valence' else '*'}: chance {chance}",
                       "x": {"label": "", "categories": [f"{n} ({row['by'][g]['worlds']})" for n, g in
                                                         zip(names, values)]},
                       "y": {"label": "share of the worlds that died", "min": 0, "max": 1},
                       "series": [F.bars(names, [row["by"][g]["rate"] for g in values], BLUE)],
                       "legend": False, "height": 150})
    ch.grid("deaths", charts, columns=2, title="Which worlds died",
            caption="For four settings, the share of Experiment 40's worlds drawn with each value that died; the "
                    "number in brackets is how many worlds had that value. A cap of two links killed every world it "
                    "was drawn in, so the three charts marked * leave those six worlds out. Each title gives the "
                    "chance of differences this large by shuffling the deaths among the worlds.",
            recipe=recipe(RUNS, DATA, ["For each value, the deaths over the worlds drawn with it.",
                                       "The chance: the share of 20,000 shuffles of the deaths with a chi-square "
                                       "statistic at least as large."]))

    # The settings no outcome showed any sign of: candidates to fix at their simplest value.
    names = {t["outcome"]: t["name"] for t in tests}
    ch.number("table:quiet", {
        "head": ["setting", "its smallest chance on any outcome", "on which"],
        "rows": [[_setting(q["setting"]), f"{q['smallest_chance']:.2f}", names[q["on"]]] for q in e["quiet"]]})
    ch.number("quiet", [q["setting"] for q in e["quiet"]])

    # The pairs of values whose worlds died more than either alone.
    ch.number("table:pairs", {
        "head": ["one value", "and another", "worlds with both", "of them died", "died with each alone",
                 "chance", "false discovery rate"],
        "rows": [[f"`{p['pair'][0][0]}` = {p['pair'][0][1]}", f"`{p['pair'][1][0]}` = {p['pair'][1][1]}",
                  p["worlds"], p["died"], f"{p['alone'][0]:.0%} / {p['alone'][1]:.0%}", f"{p['chance']:.2g}",
                  f"{p['fdr']:.2f}"] for p in e["pairs"][:12]]})

    # Mechanisms switched on, against health.
    simp = e["simplicity"]
    living = [r for r in simp["worlds"] if r["health"] is not None]
    dead = [r for r in simp["worlds"] if r["health"] is None]
    counts = sorted({r["count"] for r in simp["worlds"]})
    groups = [[r["health"] for r in living if r["count"] == k] for k in counts]
    ch.figure("simplicity", title="Do worlds with more mechanisms switched on do better?",
              x={"label": "mechanisms switched on beyond the simplest rules", "categories": [str(k) for k in counts]},
              y={"label": "health (the six criteria multiplied, as their geometric average)", "min": 0, "max": 1},
              series=dots([str(k) for k in counts], groups, [BLUE] * len(counts)), legend=False,
              caption=f"Each dot is a living world of Experiment 40, placed by how many of {len(simp['simplest'])} "
                      "mechanisms it switches on beyond the simplest rules (no messages, no noise inputs, the "
                      "largest stake wins, no coalitions, no handing over, no gifts, no mending, no mixing, estates "
                      "shared uniformly, no search, no cap on links, no token in links, no upkeep) and by its "
                      "health; the bar is the median. Rank correlation "
                      f"{simp['spearman']:.2f}.",
              recipe=recipe(RUNS, DATA, ["Count each world's drawn values that differ from `SIMPLEST` in effects.py.",
                                         "Plot against its health; the Spearman correlation of the two."]))
    ch.number("simplicity", {"spearman": simp["spearman"], "dead_counts": F.describe([r["count"] for r in dead]),
                             "living_counts": F.describe([r["count"] for r in living])})

    # The eight draws played a token at a time, weakest first and strongest first.
    by_seed: Dict[int, Dict[str, Any]] = {}
    for w in h["worlds"]:
        if w["round"] in ("E41", "E42"):
            by_seed.setdefault(w["seed"], {})[w["round"]] = w
    full = [w for w in h["worlds"] if w["round"] == "E40"]
    fate = lambda w: (f"{w['health']['product']:.3f}" if w.get("health") else
                      ("lived" if w["alive"] else f"died at {w['ended_at']:,}"))
    ch.number("table:allocation", {
        "head": ["draw", "weakest first", "strongest first"],
        "rows": [[f"s{seed}", fate(pair["E41"]), fate(pair["E42"])] for seed, pair in sorted(by_seed.items())
                 if "E41" in pair and "E42" in pair]})
    alive = lambda ws: [w for w in ws if w["alive"] and w.get("health")]
    ch.number("allocation", {
        "full": {"worlds": len(full), "died": sum(not w["alive"] for w in full),
                 "health": F.describe([w["health"]["product"] for w in alive(full)])},
        **{r: {"worlds": sum(1 for p in by_seed.values() if r in p),
               "died": sum(1 for p in by_seed.values() if r in p and not p[r]["alive"]),
               "health": F.describe([p[r]["health"]["product"] for p in by_seed.values()
                                     if r in p and p[r]["alive"] and p[r].get("health")])}
           for r in ("E41", "E42")}})
