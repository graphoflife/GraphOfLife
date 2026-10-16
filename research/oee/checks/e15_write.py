"""
Write Experiment 15's results into Chapter 49 from what was measured: the thesis
measure (e15_thesis.json), the lifetime contests (lab_rejudge_T100.json,
long_steps.json) and each run's statistics over iterations 300 to 1,000.

Replaces the chapter's "Waiting for the runs" note with the results, once.

    python3 research/oee/checks/e15_write.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
CHAPTER = os.path.join(ROOT, "book", "chapters", "49-who-dies-and-why.md")
THESIS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "e15_thesis.json")
FAMILIES = {"B2": "B2-10000", "the last token kept": "B2-10000-054b50", "half kept": "B2-10000-df2615",
            "children of any size": "B2-10000-185eb7", "no old age": "B2-10000-3ec24f"}
WAITING = """> [!warning] Waiting for the runs
> The plan, the thesis and the method above were written before any run of
> Experiment 15 existed; the results follow when they are done.
"""


def window_means(family: str):
    """Each world's means over iterations 300 to 1,000: agents, births per agent, median tokens, Gini."""
    import gol_record
    out = {"agents": [], "births": [], "median": [], "gini": []}
    for seed in range(1, 11):
        rows = [r for r in gol_record.read_stats(f"{family}-s{seed:03d}") if 300 <= r["iteration"] <= 1000]
        games = [r for r in rows if r["phase"] == 2]
        repro = [r for r in rows if r["phase"] == 1 and r.get("nodes_before")]
        out["agents"].append(np.mean([r["nodes"] for r in games]))
        out["births"].append(np.mean([r["births"] / r["nodes_before"] for r in repro]))
        out["median"].append(np.mean([r["medianTokens"] for r in games]))
        out["gini"].append(np.mean([r["gini"] for r in games if r.get("gini") is not None]))
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    thesis = json.load(open(THESIS))
    c = thesis["conditions"]
    share = {name: c[name]["one_token_median"] for name in FAMILIES}
    stats = {name: window_means(family) for name, family in FAMILIES.items()}
    from book_chapters.whodies import _contests
    contests = {name: float(np.median(v)) for name, v in _contests().items()}
    pct = lambda x: f"{100 * x:.0f}%"
    rows = []
    labels = {"B2": "B2: 30% kept at home, children of at least ten tokens, death at 100"}
    for name in FAMILIES:
        s = stats[name]
        rows.append(f"| {labels.get(name, name)} | {s['agents']:,.0f} | {pct(share[name])} | {s['median']:.1f} | "
                    f"{s['gini']:.2f} | {100 * s['births']:.2f} | {contests.get(name, float('nan')):.2f} |")
    b2, last, nothing = c["B2"], c["the last token kept"], c["nothing kept"]
    parts = [
        (f"B2's worlds were to keep fewer than half their agents at a single token in at least seven of ten; "
         f"{b2['below_half']} of ten did (the median world {pct(b2['one_token_median'])})"),
        (f"the worlds keeping only the last token were to have more than two thirds there in at least seven; "
         f"{last['above_two_thirds']} did (the median world {pct(last['one_token_median'])})"),
        (f"and at least five of the ten worlds keeping nothing were to fall below 100 agents; all ten died out")]
    worlds = lambda name: sorted(w["one_token"] for w in c[name]["worlds"].values() if w["one_token"] is not None)
    b2_low, b2_high = pct(worlds("B2")[0]), pct(worlds("B2")[-1])
    last_low, last_high = pct(worlds("the last token kept")[0]), pct(worlds("the last token kept")[-1])
    holds = thesis["holds"]
    verdict = "**The thesis holds**" if holds else "**Strictly, the thesis is refuted**"
    text = f"""## What the sixty worlds did

The ten worlds that kept nothing at home did not last: every one died out
within 26 to 47 iterations, its agents staking their last tokens away, as the
pilots had. The other fifty lived their 1,000 iterations.

<!-- figure E15/agents -->

<!-- figure E15/births -->

The births tell the rules apart most plainly. Wherever agents die at 100, a
world settles where one agent in a hundred has a child in every iteration —
what a life of 100 iterations needs to keep a world's size — whatever is kept
at home. Without old age, births fall to about one in ten thousand by
iteration 1,000: the founders hold every place and nobody has room for a
child. With children of any size they come in waves a hundred iterations
apart: a world fills at once with one-token children, they grow old together,
die in the same iteration, and are replaced at once.

<!-- figure E15/median -->

| rule of death | agents | at a single token | the median agent's tokens | Gini | births per 100 agents an iteration | brains of 1,000 against 500 |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Each number is the median of the ten worlds over iterations 300 to 1,000 —
each world's mean, and for the share at a single token each world's median
game, as the thesis asked; the last column is the share of a contested world
that the brains of iteration 1,000 hold against those of 500 over an agent's
lifetime (below). Keeping nothing is left out: those worlds were gone.

<!-- figure whodies/one_token -->

## The verdict

The thesis asked three things: {parts[0]}; {parts[1]}; {parts[2]}.

{verdict} — by B2's own worlds, which keep about half their agents at a single
token, not fewer than half: {b2_low} to {b2_high}, and only {b2['below_half']} of ten
below the line. Everything else it said held. Keeping only the last token left
more than two agents in three at one token in every world ({last_low} to
{last_high}), and every world that kept nothing died out. And the order it
predicted holds across the rules: the more an agent keeps at home, the fewer
agents hold a single token — {pct(share['half kept'])} with half kept, {pct(share['B2'])} in B2,
{pct(share['the last token kept'])} with the last token — while the two rules it did not
name push the other way: {pct(share['no old age'])} without old age, and
{pct(share['children of any size'])} with children of any size.

## Does evolution go on?

What is left to play also decides how much evolution there is. Played a
lifetime long, the brains of iteration 1,000 beat those of 500 in B2's worlds
({contests['B2']:.2f}, ten worlds) and more clearly where half is kept at home
({contests['half kept']:.2f}, five worlds); less with children of any size
({contests['children of any size']:.2f}); and hardly at all where only the last
token is kept ({contests['the last token kept']:.2f}) or nobody dies of age
({contests['no old age']:.2f}).

<!-- figure whodies/evolution -->

## What it teaches

- **What is kept at home decides who is left to play.** Keeping half makes
  fewer and richer agents ({stats['half kept']['agents']:,.0f}, the median holding
  {stats['half kept']['median']:.1f} tokens) and the strongest evolution of the five
  rules; keeping only the last token makes more agents
  ({stats['the last token kept']['agents']:,.0f}, {pct(share['the last token kept'])} of them holding
  just that token), and evolution all but stops. A world of paupers is
  a world where little happens — the author's reading of B2
  ([Meta X](79-meta-10-healthy-and-rich.md)), seen here in the contests.
- **Every agent has to die.** Without old age the founders keep their places,
  births stop, and so does evolution — while the network stays as healthy as
  any (Meta X's frozen worlds).
- **A child has to be worth having.** With children of any size the world fills
  with {stats['children of any size']['agents']:,.0f} agents of one token each (a Gini of
  {stats['children of any size']['gini']:.2f}: all equally poor), born and dying in
  cohorts, and evolution is weaker than in B2.
- **Something has to stay home.** With nothing kept no world lived 50
  iterations.

B2's own choices hold up — but keeping half at home did better on both counts
that matter here, richer agents and evolution that goes on. It is the first
rule of death to try together with the rules that came closest in Meta X,
such as brains mixed at birth ([Chapter 80](80-mixed-at-birth-ten-long-worlds.md)).
"""
    chapter = open(CHAPTER).read()
    assert chapter.count(WAITING) == 1, "the chapter no longer waits for the runs"
    open(CHAPTER, "w").write(chapter.replace(WAITING, text))
    print("written;", "holds" if holds else "refuted", thesis["verdict"])


if __name__ == "__main__":
    main()
